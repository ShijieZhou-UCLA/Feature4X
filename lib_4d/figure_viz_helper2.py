# newer verions

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


from gs_viz_helpers import viz_scene
from matplotlib import cm
import cv2 as cv
import glob

import torch.nn.functional as F

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)

from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA


def save_frame_list(frame_list, name):
    os.makedirs(osp.dirname(name), exist_ok=True)
    imageio.mimsave(name + ".mp4", frame_list)
    # ! no need for save this
    # os.makedirs(name, exist_ok=True)
    # for i, frame in enumerate(frame_list):
    #     imageio.imwrite(osp.join(name, f"{i:04d}.jpg"), frame)
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

    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    NUM_SEMANTIC_CHANNELS = s_model.semantic_feature_dim
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
    line_opa=0.5,
    line_r_factor=0.001,
    line_colorful_flag=True,
    rel_focal=None,
    #
    append_fg=False,
    subsample_node=0.5,
):
    scf = d_model.scf
    rgb_viz_list = []

    # ! color the node
    node_first = d_model.scf._node_xyz[0]
    node_colors = map_colors(node_first.detach().cpu().numpy())
    node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())
    pad_sph_dim = s_model()[4].shape[1]
    if pad_sph_dim > node_sph.shape[1]:
        node_sph = F.pad(node_sph, (0, pad_sph_dim - node_sph.shape[1], 0, 0))

    if subsample_node != 1.0:
        selected_node = torch.randperm(len(node_first))[
            : int(len(node_first) * subsample_node)
        ]
    else:
        selected_node = torch.arange(len(node_first))
    node_mask = torch.zeros(len(node_first)).bool()
    node_mask[selected_node] = True
    edge_mask = node_mask[:, None].expand(-1, d_model.scf.topo_knn_ind.shape[1])
    edge_mask = edge_mask & node_mask[scf.topo_knn_ind.cpu()]
    edge_mask = edge_mask.to(scf.topo_knn_ind.device)

    node_s1 = d_model.scf.node_sigma.expand(-1, 3) * 0.333  # * 0.05
    node_s1 = torch.clamp(node_s1, 1e-6, d_model.scf.spatial_unit * 3)
    node_o1 = torch.ones_like(node_s1[:, :1]) * 0.1
    NUM_SEMANTIC_CHANNELS = s_model.semantic_feature_dim
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

        if append_fg:
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
        gs5.append(
            [
                node_mu[node_mask],
                node_fr[node_mask],
                node_s1[node_mask],
                node_o1[node_mask],
                node_sph[node_mask] * 0.5,
                node_semantic_feature1[node_mask],
            ]
        )
        gs5.append(
            [
                node_mu[node_mask],
                node_fr[node_mask],
                node_s2[node_mask],
                node_o2[node_mask],
                node_sph[node_mask],
                node_semantic_feature2[node_mask],
            ]
        )
        ##################################################
        if line_N > 0:
            scf = d_model.scf
            dst_xyz = node_mu[scf.topo_knn_ind]
            src_xyz = node_mu[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
            line_xyz = draw_gs_point_line(
                src_xyz[scf.topo_knn_mask * edge_mask],
                dst_xyz[scf.topo_knn_mask * edge_mask],
                n=line_N,
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
                    src_sph[scf.topo_knn_mask * edge_mask],
                    dst_sph[scf.topo_knn_mask * edge_mask],
                    n=line_N,
                ).reshape(-1, node_sph.shape[-1])
            else:
                l_sph = line_sph.expand(len(line_xyz), -1)
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


@torch.no_grad()
def viz_single_2d_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    save_fn,
    pose_list,
    model_t=None,
    rel_focal=None,
    bg_flag=True,
    fg_flag=True,
):
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    semantic_feature_viz_list=[]
    if rel_focal is None:
        rel_focal = cams.rel_focal
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

        semantic_feature_flat = render_dict["feature_map"].detach().cpu().numpy().reshape(
            render_dict["feature_map"].shape[0], -1
        ).T
        # PCA visualize
        # feature_pca_mean = np.mean(semantic_feature_flat, axis=0)
        # semantic_feature_flat = np.random.randn(10000, 37)
        pca = PCA(n_components=3)
        pca.fit(semantic_feature_flat)
        feature_pca = pca.transform(semantic_feature_flat)
        feature_pca = (feature_pca - np.min(feature_pca, axis=0)) / (
            np.max(feature_pca, axis=0) - np.min(feature_pca, axis=0)
        )
        feature_pca = (feature_pca * 255).astype(np.uint8)
        feature_vis = feature_pca.reshape(
            render_dict["feature_map"].shape[1], render_dict["feature_map"].shape[2], 3
        )
        semantic_feature_viz_list.append(feature_vis)

        dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        dep_viz_list.append(dep)
        if "normal" in render_dict:
            normal = render_dict["normal"].detach().cpu().numpy()
            normal_viz = (1 - normal) / 2
            normal_viz_list.append(normal_viz.transpose(1, 2, 0))

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
