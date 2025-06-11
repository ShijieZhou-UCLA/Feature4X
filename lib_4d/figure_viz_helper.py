# This helpers is from solver_viz_helper for making figures
from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import logging
import imageio
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
import open3d as o3d
from render_helper import render
from loss_helper import compute_rgb_loss, compute_dep_loss, compute_normal_loss



from lib_render.gs3d.sh_utils import RGB2SH, SH2RGB

from autoencoder.model import Feature_heads
from clip_utils import CLIPEditor


from gs_viz_helpers import viz_scene
from matplotlib import cm
import cv2 as cv
import glob
import torch.nn as nn

import torch.nn.functional as F

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)

from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
import yaml
import time


def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            scores = (scores >= score_threshold).float()
        else:
            # scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = (scores >= score_threshold).float()
            else:
                scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
        return scores

def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            mask = (scores >= score_threshold).float()
        else:
            # scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            
            scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
            mask = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda())
            
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                mask = torch.bitwise_or((scores >= score_threshold), mask).float()
        
        return mask

@torch.no_grad()
def make_viz_np(
    gt,
    pred,
    error,
    error_cm=cv.COLORMAP_WINTER,
    img_cm=cv.COLORMAP_VIRIDIS,
    text0="target",
    text1="pred",
    text2="error",
    gt_margin=5,
):
    assert error.ndim == 2
    error = (error / error.max()).detach().cpu().numpy()
    error = (error * 255).astype(np.uint8)
    error = cv.applyColorMap(error, error_cm)[:, :, ::-1]
    viz_frame = torch.cat([gt, pred], 1)
    if viz_frame.ndim == 2:
        viz_frame = viz_frame / viz_frame.max()
    viz_frame = viz_frame.detach().cpu().numpy()
    viz_frame = np.clip(viz_frame * 255, 0, 255).astype(np.uint8)
    if viz_frame.ndim == 2:
        viz_frame = cv.applyColorMap(viz_frame, img_cm)[:, :, ::-1]
    viz_frame = np.concatenate([viz_frame, error], 1)
    # split the image to 3 draw the text onto the image
    viz_frame_list = np.split(viz_frame, 3, 1)
    # draw green border of GT target, don't pad, draw inside

    viz_frame_list[0] = cv.copyMakeBorder(
        viz_frame_list[0][gt_margin:-gt_margin, gt_margin:-gt_margin],
        gt_margin,
        gt_margin,
        gt_margin,
        gt_margin,
        cv.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )
    for i, text in enumerate([text0, text1, text2]):
        if len(text) > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = TEXTCOLOR
            lineType = 2
            cv.putText(
                viz_frame_list[i],
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
    viz_frame = np.concatenate(viz_frame_list, 1)
    return viz_frame


def save_frame_list(frame_list, name):
    os.makedirs(name, exist_ok=True)
    imageio.mimsave(name + ".mp4", frame_list)
    for i, frame in enumerate(frame_list):
        imageio.imwrite(osp.join(name, f"{i:04d}.jpg"), frame.astype(np.uint8))
    return


@torch.no_grad()
def get_global_3D_cam_T_cw(
    s_model,
    d_model,
    cams,
    H,
    W,
    ref_tid,
    back_ratio=1.0,
    up_ratio=0.2,
):
    render_dict = render(
        [s_model(), d_model(ref_tid)],
        H,
        W,
        cams.rel_focal,
        cams.cxcy_ratio,
        cams.T_cw(ref_tid),
    )
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
    focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)  # in cam frame

    T_c_new = torch.eye(4).to(cams.T_wc(0))
    T_c_new[2, -1] = -center_dep * back_ratio  # z
    T_c_new[1, -1] = -center_dep * up_ratio  # y
    _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
    _x_dir = F.normalize(
        torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
    )
    _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
    T_c_new[:3, 0] = _x_dir
    T_c_new[:3, 1] = _y_dir
    T_c_new[:3, 2] = _z_dir
    T_base = cams.T_wc(ref_tid)
    T_w_new = T_base @ T_c_new
    T_new_w = T_w_new.inverse()
    return T_new_w


@torch.no_grad()
def get_move_around_cam_T_cw(
    s_model,
    d_model,
    cams,
    H,
    W,
    move_around_angle_deg,
    total_steps,
    center_id=None,
):

    # in the xy plane, the new camera is forming a circle
    move_around_view_list = []
    for i in tqdm(range(total_steps)):
        if center_id is None:
            move_around_id = i
            assert total_steps - 1 < cams.T
            render_dict = render(
                [s_model(), d_model(move_around_id)],
                H,
                W,
                cams.rel_focal,
                cams.cxcy_ratio,
                cams.T_cw(move_around_id),
            )
            # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
            depth = render_dict["dep"][0]
            center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
            if center_dep < 1e-2:
                center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
            focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
            move_around_radius = np.tan(move_around_angle_deg) * focus_point[2].item()
        else:
            move_around_id = center_id
            if i == 0:
                render_dict = render(
                    [s_model(), d_model(move_around_id)],
                    H,
                    W,
                    cams.rel_focal,
                    cams.cxcy_ratio,
                    cams.T_cw(move_around_id),
                )
                # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
                depth = render_dict["dep"][0]
                center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
                if center_dep < 1e-2:
                    center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
                focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
                move_around_radius = (
                    np.tan(move_around_angle_deg) * focus_point[2].item()
                )

        x = (
            move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
            - move_around_radius
        )
        y = move_around_radius * np.sin(2 * np.pi * i / (total_steps - 1))
        T_c_new = torch.eye(4).to(cams.T_wc(0))
        T_c_new[0, -1] = x
        T_c_new[1, -1] = y
        _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
        _x_dir = F.normalize(
            torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
        )
        _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
        T_c_new[:3, 0] = _x_dir
        T_c_new[:3, 1] = _y_dir
        T_c_new[:3, 2] = _z_dir

        T_base = cams.T_wc(move_around_id)

        T_w_new = T_base @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list


@torch.no_grad()
def draw_gs_point_line(start, end, n=32):
    # start, end is N,3 tensor
    line_dir = end - start
    xyz = (
        start[:, None]
        + torch.linspace(0, 1, n)[None, :, None].to(start) * line_dir[:, None]
    )
    return xyz


def map_colors(points, mod=1):
    # normalized_points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))

    # do pca for the points
    pca = PCA(n_components=3)
    pca_points = pca.fit_transform(points)
    # normalzie
    pca_points = (pca_points - np.min(pca_points, axis=0)) / (
        np.max(pca_points, axis=0) - np.min(pca_points, axis=0)
    )

    # Map coordinates to HSV colors
    # # H: X-coordinate, S: 1 (high saturation), V: Z-coordinate
    hsv_colors = np.zeros_like(pca_points)
    hue = pca_points[:, 0]
    if mod > 1:
        # set periodical mod times
        hue = hue * mod
        hue = hue - np.floor(hue)
    hsv_colors[:, 0] = hue
    hsv_colors[:, 1] = 0.9
    hsv_colors[:, 2] = 0.9
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors


@torch.no_grad()
def viz_single_2d_flow_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    N_max=512,
    color_mod=5,
    max_T=1000,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.001,  # 0.05,
    # line
    line_N=32,
    line_opa=0.5,
    line_r_factor=0.001,
    rel_focal=None,
):
    rgb_viz_list = []

    # ! color the node
    pts_first = d_model(0)[0]
    if len(pts_first) > N_max:
        # ! do a filtering for viz purpose, only viz dense area
        # use open3d
        inlier_mask = outlier_removal_o3d(pts_first, std_ratio=1.0)
        print(f"Filtered {len(pts_first) - inlier_mask.sum()} points")
        candidates = torch.arange(len(pts_first))[inlier_mask.cpu()]
        step = max(1, len(candidates) // N_max)
        # viz_choice = candidates[torch.randperm(len(candidates))[:N_max]]
        viz_choice = candidates[::step][:N_max]
        pts_first = pts_first[viz_choice]
    node_colors = map_colors(pts_first.detach().cpu().numpy(), mod=color_mod)
    NUM_SEMANTIC_CHANNELS = d_model.semantic_feature_dim
    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    flow_semantic_feature = torch.ones(len(flow_sph), NUM_SEMANTIC_CHANNELS).to(pts_first.device) # changed 37 to 128
    pad_sph_dim = s_model()[4].shape[1]
    if pad_sph_dim > flow_sph.shape[1]:
        flow_sph = F.pad(flow_sph, (0, pad_sph_dim - flow_sph.shape[1], 0, 0))

    flow_mu = pts_first
    flow_fr = (
        torch.eye(3).to(flow_mu.device).unsqueeze(0).expand(flow_mu.shape[0], -1, -1)
    )
    flow_s = (
        torch.ones(len(flow_mu), 3).to(flow_mu)
        * d_model.scf.spatial_unit
        * node_r_factor
    )
    flow_o = torch.ones_like(flow_s[:, :1]) * 0.99
    last_flow_mu = flow_mu
    last_flow_sph = flow_sph

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    if gray_scale_bg_flag:
        bg_rgb = SH2RGB(gs5_bg[4][:, :3])
        bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
        # convert to gray scale
        bg_sph = RGB2SH(bg_gray)
        if pad_sph_dim > bg_sph.shape[1]:
            bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
        gs5_bg[4] = bg_sph

    max_buffer_size = len(flow_mu) * (line_N + 1) * max_T

    for cam_tid in tqdm(range(len(pose_list))):
        # working_t = cam_tid if model_t is None else model_t
        working_t = cam_tid

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        d_gs5[-3] = 0.2 * d_gs5[-3]
        gs5.append(d_gs5)

        if cam_tid > 0:
            new_xyz = d_gs5[0][viz_choice]
            new_flow_sph = last_flow_sph
            # first draw lines of the flow
            if line_N > 0:
                line_xyz = draw_gs_point_line(new_xyz, last_flow_mu, n=line_N).reshape(
                    -1, 3
                )
                line_fr = (
                    torch.eye(3)
                    .to(flow_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = (
                    torch.ones_like(line_xyz) * d_model.scf.spatial_unit * line_r_factor
                )
                line_o = torch.ones_like(line_s[:, :1]) * line_opa
                line_sph = draw_gs_point_line(
                    new_flow_sph,
                    last_flow_sph,
                    n=line_N,
                ).reshape(-1, flow_sph.shape[-1])
                line_semantic_feature = torch.ones(len(line_s), NUM_SEMANTIC_CHANNELS).to(flow_mu.device) # changed 37 to 128
                flow_mu = torch.cat([flow_mu, line_xyz], dim=0)
                flow_fr = torch.cat([flow_fr, line_fr], dim=0)
                flow_s = torch.cat([flow_s, line_s], dim=0)
                flow_o = torch.cat([flow_o, line_o], dim=0)
                flow_sph = torch.cat([flow_sph, line_sph], dim=0)
                flow_semantic_feature = torch.cat( [flow_semantic_feature, line_semantic_feature], dim=0)
                last_flow_mu = new_xyz
                last_flow_sph = new_flow_sph
            flow_mu = torch.cat([flow_mu, new_xyz], dim=0)
            new_fr = (
                torch.eye(3)
                .to(new_xyz.device)
                .unsqueeze(0)
                .expand(new_xyz.shape[0], -1, -1)
            )
            flow_fr = torch.cat([flow_fr, new_fr], dim=0)
            flow_s = torch.cat(
                [
                    flow_s,
                    torch.ones_like(new_xyz) * d_model.scf.spatial_unit * node_r_factor,
                ],
                dim=0,
            )
            flow_o = torch.cat([flow_o, torch.ones_like(flow_s[:, :1]) * 0.99], dim=0)
            flow_sph = torch.cat([flow_sph, new_flow_sph], dim=0)
            flow_semantic_feature = torch.cat([flow_semantic_feature, torch.ones(len(new_xyz), NUM_SEMANTIC_CHANNELS).to(new_xyz.device)], dim=0) # changed 37 to 128
        if len(flow_mu) > max_buffer_size:
            flow_mu = flow_mu[-max_buffer_size:]
            flow_fr = flow_fr[-max_buffer_size:]
            flow_s = flow_s[-max_buffer_size:]
            flow_o = flow_o[-max_buffer_size:]
            flow_sph = flow_sph[-max_buffer_size:]
            flow_semantic_feature = flow_semantic_feature[-max_buffer_size:]

        gs5.append([flow_mu, flow_fr, flow_s, flow_o, flow_sph, flow_semantic_feature])
        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render(
            gs5,
            H,
            W,
            rel_focal,
            cams.cxcy_ratio,
            T_cw=pose_list[cam_tid],
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return


@torch.no_grad()
def viz_single_2d_node_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    model_t=None,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.08,
    # line
    line_N=32,
    line_color=[0.7] * 3,
    line_opa=0.1,
    line_r_factor=0.001,
    line_colorful_flag=True,
    rel_focal=None,
):
    rgb_viz_list = []

    # ! color the node
    node_first = d_model.scf._node_xyz[0]
    node_colors = map_colors(node_first.detach().cpu().numpy())
    node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())
    pad_sph_dim = s_model()[4].shape[1]
    if pad_sph_dim > node_sph.shape[1]:
        node_sph = F.pad(node_sph, (0, pad_sph_dim - node_sph.shape[1], 0, 0))

    NUM_SEMANTIC_CHANNELS = d_model.semantic_feature_dim
    node_s1 = d_model.scf.node_sigma.expand(-1, 3) * 0.333  # * 0.05
    node_s1 = torch.clamp(node_s1, 1e-6, d_model.scf.spatial_unit * 3)
    node_o1 = torch.ones_like(node_s1[:, :1]) * 0.1
    node_semantic_feature1 = torch.ones(len(node_s1), NUM_SEMANTIC_CHANNELS).to(node_first.device) # changed 37 to 128

    node_s2 = torch.ones_like(node_s1) * d_model.scf.spatial_unit * node_r_factor
    node_o2 = torch.ones_like(node_s2[:, :1]) * 0.99
    node_semantic_feature2 = torch.ones(len(node_s2), NUM_SEMANTIC_CHANNELS).to(node_first.device) # changed 37 to 128

    line_sph = torch.tensor(line_color).to(node_sph.device).float()[None]
    line_sph = RGB2SH(line_sph)
    if pad_sph_dim > line_sph.shape[1]:
        line_sph = F.pad(line_sph, (0, pad_sph_dim - line_sph.shape[1], 0, 0))

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    if gray_scale_bg_flag:
        bg_rgb = SH2RGB(gs5_bg[4][:, :3])
        bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
        # convert to gray scale
        bg_sph = RGB2SH(bg_gray)
        if pad_sph_dim > bg_sph.shape[1]:
            bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
        gs5_bg[4] = bg_sph

    for cam_tid in tqdm(range(len(pose_list))):
        working_t = cam_tid if model_t is None else model_t

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        d_gs5[-3] = 0.2 * d_gs5[-3]
        gs5.append(d_gs5)
        node_mu = d_model.scf._node_xyz[working_t]
        node_fr = (
            torch.eye(3)
            .to(node_mu.device)
            .unsqueeze(0)
            .expand(node_mu.shape[0], -1, -1)
        )

        #make node_semantic_features

        gs5.append([node_mu, node_fr, node_s1, node_o1, node_sph * 0.5, node_semantic_feature1])
        gs5.append([node_mu, node_fr, node_s2, node_o2, node_sph, node_semantic_feature2])
        ##################################################
        if line_N > 0:
            scf = d_model.scf
            dst_xyz = node_mu[scf.topo_knn_ind]
            src_xyz = node_mu[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
            line_xyz = draw_gs_point_line(
                src_xyz[scf.topo_knn_mask], dst_xyz[scf.topo_knn_mask], n=line_N
            ).reshape(-1, 3)
            line_fr = (
                torch.eye(3)
                .to(node_mu.device)
                .unsqueeze(0)
                .expand(line_xyz.shape[0], -1, -1)
            )
            line_s = torch.ones_like(line_xyz) * scf.spatial_unit * line_r_factor
            line_o = torch.ones_like(line_s[:, :1]) * line_opa
            line_semantic_feature = torch.ones(len(line_s), NUM_SEMANTIC_CHANNELS).to(node_first.device) # changed 37 to 128
            if line_colorful_flag:
                src_sph = node_sph[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
                dst_sph = node_sph[scf.topo_knn_ind]
                l_sph = draw_gs_point_line(
                    src_sph[scf.topo_knn_mask], dst_sph[scf.topo_knn_mask], n=line_N
                ).reshape(-1, node_sph.shape[-1])
            else:
                l_sph = line_sph.expand(len(line_xyz), -1)

            # make line semantic features

            gs5.append([line_xyz, line_fr, line_s, line_o, l_sph, line_semantic_feature])

        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render(
            gs5,
            H,
            W,
            rel_focal,
            cams.cxcy_ratio,
            T_cw=pose_list[cam_tid],
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return

def parse_edit_config_and_text_encoding_agents(edit_config):

    print('edit config', edit_config)
    edit_dict = {}
    if edit_config is not None:
        # with open(edit_config, 'r') as f:
        #     edit_config = yaml.safe_load(f)
        # print('edit_config', edit_config)
        item_list = edit_config.split('objects:')[1].split('operation')[0]
        # objects = [f'"{x}"' for x in objects]
        objects = []

        # Remove the brackets
        item_list = item_list[1:-3].replace(']', '')
        # Split by comma
        for item in item_list.split(","):
            print('item', item)
            objects.append(item.strip())

        # objects = ['car', 'tree', 'building', 'sidewalk', 'road']
        print('item_list', objects)
        # objects = [str(obj) for obj in objects]

        targets = [edit_config.split("targets:")[-1].split('threshold')[0].strip()]
        print('targets', targets)
        edit_dict["positive_ids"] = [objects.index(t) for t in targets if t in objects]
        print('edit dict positive ids', edit_dict['positive_ids'])
        edit_dict["score_threshold"] = float(edit_config.split('threshold:')[-1].splitlines()[0].strip())
        print('edit dict', float(edit_dict['score_threshold']))
        
        '''
        edit_config {'edit': {'objects': ['car', 'tree', 'building', 'sidewalk', 'road'], 'operations': 'deletion', 'targets': 'car', 'threshold': 0.198}} [20/10 18:02:05]
        objects ['car', 'tree', 'building', 'sidewalk', 'road'] [20/10 18:02:05]
        targets ['car'] [20/10 18:02:05]
        edit dict positive ids [0] [20/10 18:02:05]
        edit dict 0.198 [20/10 18:02:05]
        '''
        
        # text encoding
        clip_editor = CLIPEditor()
        text_feature = clip_editor.encode_text([obj.replace("_", " ") for obj in objects])
        # text_feature = clip_editor.encode_text([obj for obj in objects])

        # setup editing
        op_dict = {}
        for operation in [str(edit_config.split("operations:")[-1].split('targets')[0]).strip()]:
            # for i in operation:
            #     print(i)
            # print("deletion")
            print(operation=="deletion")
            if operation == "extraction":
                op_dict["extraction"] = True
            elif operation == "deletion":
                op_dict["deletion"] = True
            elif "color_func" in operation:
                func = edit_config.split("function:")[1].splitlines()[0]
                # Create evaluation context with necessary variables
                eval_globals = {
                    'torch': torch,
                    'mode': 'assign',  # default mode
                    'assign_values': [-2, -2, -2],  # default black color values
                    'adjustment': [0, 0, 0]  # default no adjustment
                }
                op_dict["color_func"] = eval(func, eval_globals)
            else:
                raise NotImplementedError
        edit_dict["operations"] = op_dict

        # idx = edit_dict["positive_ids"][0]

    return edit_dict, text_feature

def parse_edit_config_and_text_encoding(edit_config):
    edit_dict = {}
    if edit_config is not None:
        with open(edit_config, 'r') as f:
            edit_config = yaml.safe_load(f)
            print('edit_config', edit_config)
        objects = edit_config["edit"]["objects"]
        print('objects', objects)
        targets = edit_config["edit"]["targets"].split(",")
        print('targets', targets)
        edit_dict["positive_ids"] = [objects.index(t) for t in targets if t in objects]
        print('edit dict positive ids', edit_dict['positive_ids'])
        edit_dict["score_threshold"] = edit_config["edit"]["threshold"]
        print('edit dict', edit_dict['score_threshold'])
        
        '''
        edit_config {'edit': {'objects': ['car', 'tree', 'building', 'sidewalk', 'road'], 'operations': 'deletion', 'targets': 'car', 'threshold': 0.198}} [20/10 18:02:05]
        objects ['car', 'tree', 'building', 'sidewalk', 'road'] [20/10 18:02:05]
        targets ['car'] [20/10 18:02:05]
        edit dict positive ids [0] [20/10 18:02:05]
        edit dict 0.198 [20/10 18:02:05]
        '''
        
        # text encoding
        clip_editor = CLIPEditor()
        text_feature = clip_editor.encode_text([obj.replace("_", " ") for obj in objects])

        # setup editing
        op_dict = {}
        for operation in edit_config["edit"]["operations"].split(","):
            if operation == "extraction":
                op_dict["extraction"] = True
            elif operation == "deletion":
                op_dict["deletion"] = True
            elif operation == "color_func":
                # Create evaluation context with necessary variables
                eval_globals = {
                    'torch': torch,
                    'mode': 'assign',  # default mode
                    'assign_values': [-2, -2, -2],  # default black color values
                    'adjustment': [0, 0, 0]  # default no adjustment
                }
                op_dict["color_func"] = eval(edit_config["edit"]["colorFunc"], eval_globals)
            else:
                raise NotImplementedError
        edit_dict["operations"] = op_dict

        idx = edit_dict["positive_ids"][0]

    return edit_dict, text_feature, targets[idx]

def find_gs_from_nodes(attach_ind, node_ind):
    mask = torch.isin(attach_ind, node_ind)
    return mask

@torch.no_grad()
def viz_single_2d_video_agent(
    H, W, cams, s_model, d_model, save_fn, pose_list, model_t=None, rel_focal=None, bg_flag=True, fg_flag=True, gptv_prompts=None, args={}, round_best=None,
    feature_head={}
):
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    semantic_feature_viz_list=[]
    all_results = {}
    if rel_focal is None:
        rel_focal = cams.rel_focal

    nodes_semantic_feature = feature_head.decode("langseg", d_model.scf._node_semantic_feature).squeeze() # feature_head.decode(d_model.scf._init_node_semantic_feature) #num_points, 512
    static_semantic_feature = feature_head.decode("langseg" , s_model.get_semantic_feature) # [0]

    attach_ind = d_model.attach_ind

    # print('semantic feature live', nodes_semantic_feature.shape) # [num_points_live, 3]
    # print('semantic feature static', static_semantic_feature.shape) # [num_points_live, 3]


    edit_dict, text_feature = parse_edit_config_and_text_encoding_agents(gptv_prompts)

    positive_ids = edit_dict["positive_ids"]
    # print('postive ids======', positive_ids)
    score_threshold = edit_dict["score_threshold"]
    op_dict = edit_dict["operations"]
    
    # print('op_dict', op_dict)
    # print('score threshold=====================', score_threshold)

    # editing 
    if "deletion" in op_dict:
        scores_static = calculate_selection_score_delete(static_semantic_feature, text_feature, 
                    score_threshold=score_threshold, positive_ids=positive_ids) # # torch.Size([331617])
        delete_static_gs_indices = (scores_static >= 0.5).nonzero()
        s_model._opacity[delete_static_gs_indices] = -torch.inf

    if "extraction" in op_dict:
        scores_static = calculate_selection_score(static_semantic_feature, text_feature,
                                           score_threshold=score_threshold, positive_ids=positive_ids)
        delete_static_gs_indices = (scores_static <= 0.5).nonzero()
        s_model._opacity[delete_static_gs_indices] = -torch.inf

    if "color_func" in op_dict:
        scores_static = calculate_selection_score(static_semantic_feature, text_feature,
                                                  score_threshold=score_threshold, positive_ids=positive_ids)
        static_shs = s_model.get_c
        s_model._features_dc = torch.nn.Parameter(static_shs[:, :3] * (1 - scores_static[:, None]) + op_dict["color_func"](static_shs[:, :3]) * scores_static[:, None])
    
    for cam_tid in tqdm(range(len(pose_list))):
        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        if fg_flag:
            d_model_gs = d_model(cam_tid if model_t is None else model_t, delete_node_indices=None)
            gs_semantic_feature = d_model_gs[-1]
            gs_semantic_feature_clip = feature_head.decode("langseg", gs_semantic_feature).squeeze()

            if "deletion" in op_dict:
                score = calculate_selection_score_delete(gs_semantic_feature_clip, text_feature, score_threshold=score_threshold, positive_ids=positive_ids)
                d_model_gs[3][score>=0.5] = 0
                gs5.append(d_model_gs)
            elif "extraction" in op_dict:
                score = calculate_selection_score(gs_semantic_feature_clip, text_feature, score_threshold=score_threshold, positive_ids=positive_ids)
                d_model_gs[3][score<=0.5] = 0
                gs5.append(d_model_gs)
            elif "color_func" in op_dict:
                score = calculate_selection_score(gs_semantic_feature_clip, text_feature, score_threshold=score_threshold, positive_ids=positive_ids)
                sph = d_model_gs[4]
                sph[:, :3] = sph[:, :3] * (1 - score[:, None]) + op_dict["color_func"](sph[:, :3]) * score[:, None]
                gs5.append(d_model_gs)

            gs5.append(d_model_gs)

        render_dict = render(
            gs5,
            H,
            W,
            rel_focal,
            cams.cxcy_ratio,
            T_cw=pose_list[cam_tid],
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)

        imageio.imsave(args.save_name, rgb_viz)
        rgb_viz_list.append(rgb_viz)


    # # extract rendered features for inference
    # os.makedirs(save_fn, exist_ok=True)
    # torch.save(all_results, osp.join(save_fn, f"rendered_results.pth"))

    # # # use disp map to viz the depth!
    # # viz_dep = np.stack(dep_viz_list, axis=0)
    # # valid_mask = viz_dep > 0
    # # max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    # # viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # # # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    # # viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]

    # # save_frame_list(viz_dep, save_fn + "_dep")

    # if round_best is not None:
    save_frame_list(rgb_viz_list, save_fn + f"edited_rgb_{list(op_dict.keys())[0]}_{score_threshold}")

    # save_frame_list(semantic_feature_viz_list, save_fn + "_semantic_feature")
    # if len(normal_viz_list) > 0:
    #     print(normal_viz_list[0].shape)
    #     save_frame_list(normal_viz_list, save_fn + "_normal")
    return




@torch.no_grad()
def feature_map(feature, pca_mean:dict, top_vector:dict,type="latent"):
    # global pca_mean
    # global top_vector
    fmap = feature[None, :, :, :]  # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)

    # Reshape and normalize
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3]

    # Perform PCA using torch
    if pca_mean[type] is None:
        pca_mean[type] = f_samples.mean(dim=0, keepdim=True)
    mean = pca_mean[type]
    f_samples_centered = f_samples - mean
    covariance_matrix = f_samples_centered.T @ f_samples_centered / (f_samples_centered.shape[0] - 1)

    eig_values, eig_vectors = torch.linalg.eigh(covariance_matrix)
    if top_vector[type] is None:
        top_vector[type] = eig_vectors[:, -3:]
    top_eig_vectors = top_vector[type]

    transformed = f_samples_centered @ top_eig_vectors

    q1, q99 = transformed.quantile(0.01, dim=0), transformed.quantile(0.99, dim=0)
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = q99 - q1

    # if type == "internvideo":
    #     cache_device = "cpu"
    # else:
    #     cache_device = fmap.device
    cache_device = fmap.device
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).to(cache_device) - mean.to(cache_device)) @ top_eig_vectors.to(cache_device)
    vis_feature = vis_feature.to(feature_pca_postprocess_sub)
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3))

    return vis_feature#.permute(2, 0, 1)  # torch.Size([3, h, w])

@torch.no_grad()
def viz_single_2d_video(
    H, W, cams, s_model, d_model, save_fn, pose_list, model_t=None, rel_focal=None, bg_flag=True, fg_flag=True,
        feature_head=None, save_lseg=False,
):
    ###########################################################################################################
    # Record the start time
    start_time = time.time()

    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    semantic_feature_viz_list=[]

    pca_mean={"latent":None}
    top_vector={"latent":None}
    if feature_head is not None:
        semantic_feature_decoded_viz={}
        for key in feature_head.keys():
            semantic_feature_decoded_viz[key]=[]
            pca_mean[key]=None
            top_vector[key]=None
    else:
        semantic_feature_decoded_viz = None
    all_results = {}
    if rel_focal is None:
        rel_focal = cams.rel_focal
    os.makedirs(save_fn, exist_ok=True)
    for cam_tid in tqdm(range(len(pose_list))):
        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        if fg_flag:
            gs5.append(d_model(cam_tid if model_t is None else model_t))
        render_dict = render(
            gs5,
            H,
            W,
            rel_focal,
            cams.cxcy_ratio,
            T_cw=pose_list[cam_tid],
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)

        feature_vis = feature_map(render_dict["feature_map"], pca_mean=pca_mean, top_vector=top_vector,
                                    type="latent")
        feature_vis = (feature_vis.cpu().numpy() *255).astype(np.uint8)
        semantic_feature_viz_list.append(feature_vis)

        if feature_head is not None:
            latent_feature = render_dict["feature_map"].permute(1,2,0)
            for key in feature_head.keys():
                semantic_feature = feature_head.decode(key, latent_feature)
                semantic_feature = semantic_feature.permute(2,0,1)
                if save_lseg and key == "langseg":
                    save_dir = save_fn + f"_decoded_{key}_semantic_feature/feature_map"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(semantic_feature, osp.join(save_dir, f"{cam_tid:05d}_fmap_CxHxW.pt"))

                decoded_feature_vis = feature_map(semantic_feature, pca_mean=pca_mean, top_vector=top_vector,
                                    type=key)
                decoded_feature_vis = (decoded_feature_vis.cpu().numpy() *255).astype(np.uint8)
                semantic_feature_decoded_viz[key].append(decoded_feature_vis)
                torch.cuda.empty_cache()

        dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        dep_viz_list.append(dep)
        if "normal" in render_dict:
            normal = render_dict["normal"].detach().cpu().numpy()
            normal_viz = (1 - normal) / 2
            normal_viz_list.append(normal_viz.transpose(1, 2, 0))

        # extract rendered features for inference
        pred_feature_map = render_dict["feature_map"] # render_dim(256) x 480 x 480
        # pred_feature_map = F.interpolate(pred_feature_map[None, ...], size=(64, 64), mode='bilinear', align_corners=False)[0] # render_dim(256) x 64 x 64
        render_dict["feature_map"] =  pred_feature_map #s_model.cnn_decoder(pred_feature_map)
        all_results[cam_tid] = render_dict

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")
    ############################################################################################

    # extract rendered features for inference
    torch.save(all_results, osp.join(save_fn, f"rendered_results.pth"))

    # use disp map to viz the depth!
    viz_dep = np.stack(dep_viz_list, axis=0)
    valid_mask = viz_dep > 0
    max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]

    save_frame_list(viz_dep, save_fn + "_dep")

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    save_frame_list(semantic_feature_viz_list, save_fn + "_semantic_feature")
    if semantic_feature_decoded_viz is not None:
        for key in semantic_feature_decoded_viz.keys():
            save_frame_list(semantic_feature_decoded_viz[key], save_fn + f"_decoded_{key}_semantic_feature")


    if len(normal_viz_list) > 0:
        print(normal_viz_list[0].shape)
        save_frame_list(normal_viz_list, save_fn + "_normal")
    return


def outlier_removal_o3d(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    _, inlier_ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_mask = torch.zeros_like(xyz[:, 0]).bool()
    inlier_mask[inlier_ind] = True
    return inlier_mask


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R
