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

from render_helper import render
from loss_helper import compute_rgb_loss, compute_dep_loss, compute_normal_loss


from lib_4d.render_helper import render_cam_pcl
from lib_render.gs3d.sh_utils import RGB2SH, SH2RGB


from gs_viz_helpers import viz_scene
from matplotlib import cm
import cv2 as cv
import glob

import torch.nn.functional as F

import yaml


TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)
BG_COLOR1 = [1.0,1.0,1.0]

def make_video_from_pattern(pattern, dst):
    fns = glob.glob(pattern)
    fns.sort()
    frames = []
    for fn in fns:
        frames.append(imageio.imread(fn))
    __video_save__(dst, frames)
    return


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


def __get_move_around_cam_T_cw__(
    s_model,
    d_model,
    cams,
    solver,
    move_around_id,
    move_around_angle_deg,
    start_from,
    end_at,
):
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(move_around_id))
    render_dict = render(
        gs5,
        solver.prior2d.H,
        solver.prior2d.W,
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
    # in the xy plane, the new camera is forming a circle
    total_steps = end_at - start_from + 1
    move_around_view_list = []
    for i in range(total_steps):
        x = move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
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
        T_w_new = cams.T_wc(move_around_id) @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list


def viz2d_total_video(
    solver,
    start_from,
    end_at,
    skip_t,
    cams,
    s_model,
    d_model,
    prefix="",
    save_dir=None,
    subsample=1,
    mask_type="all",
    # move_around_angle_deg=15.0,
    move_around_angle_deg=60.0,
    move_around_id=None,
    remove_redundant_flag=True,
    max_num_frames=500,  # 150,
):
    logging.info(f"Viz 2D video from {start_from} to {end_at} ...")
    if save_dir is None:
        save_dir = osp.join(solver.log_dir, "viz_step")
    viz_vid_fn = osp.join(
        save_dir,
        f"{prefix}2D_video_{start_from}-{end_at}.mp4",
    )

    frame_list = []
    # mid_cam_id = (start_from + end_at) // 2
    fix_cam_id = start_from

    # prepare the novel camera poses
    move_around_angle = np.deg2rad(move_around_angle_deg)
    if move_around_id is None:
        move_around_id = (start_from + end_at) // 2
    move_around_view_list = __get_move_around_cam_T_cw__(
        s_model,
        d_model,
        cams,
        solver,
        move_around_id,
        move_around_angle,
        start_from,
        end_at,
    )
    for _ind, view_ind in tqdm(enumerate(range(start_from, min(end_at + 1, cams.T)))):
        if view_ind % skip_t != 0:
            continue
        frame = viz2d_one_frame(
            view_ind,
            solver.prior2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
        )
        # fix cam, vary time
        _frame = viz2d_one_frame(
            view_ind,
            solver.prior2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Fix={fix_cam_id} ",
            T_cw=cams.T_cw(fix_cam_id),
        )
        frame = np.concatenate([frame, _frame], 0)
        # fix time, vary cam
        _frame = viz2d_one_frame(
            move_around_id,
            solver.prior2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"T={move_around_id} ",
            T_cw=move_around_view_list[_ind],
        )
        frame = np.concatenate([frame, _frame], 0)
        # vary cam and time
        _frame = viz2d_one_frame(
            view_ind,
            solver.prior2d,
            cams,
            s_model,
            d_model,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Novel",
            T_cw=move_around_view_list[_ind],
        )
        frame = np.concatenate([frame, _frame], 0)

        frame_list.append(frame)

    # viz the flow
    if d_model is not None:
        flow_frame_list1, choice = viz2d_flow_video(
            solver,
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=True,
            text="WithBG",
            skip_t=skip_t,
        )
        flow_frame_list2, choice = viz2d_flow_video(
            solver,
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=False,
            text="NoBG",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list3, choice = viz2d_flow_video(
            solver,
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            viz_bg=False,
            view_cam_id=fix_cam_id,
            text=f"FixCam={fix_cam_id} ",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list = [
            np.concatenate(
                [flow_frame_list1[i], flow_frame_list2[i], flow_frame_list3[i]], 1
            )
            for i in range(len(flow_frame_list1))
        ]
        final_frame_list = [
            np.concatenate([flow_frame_list[i], frame_list[i]], 0)
            for i in range(len(flow_frame_list))
        ]
    else:
        final_frame_list = frame_list

    # ! always split to two columns
    if (final_frame_list[0].shape[0] // solver.H) % 2 != 0:
        dummy_frame = np.zeros((solver.H, solver.W * 3, 3))
        final_frame_list = [
            np.concatenate([f, dummy_frame], 0) for f in final_frame_list
        ]
        assert (final_frame_list[0].shape[0] // solver.H) % 2 == 0
    # * save
    re_aranged_list = []
    for frame in final_frame_list:
        H = frame.shape[0]
        frame_top = frame[: H // 2]
        frame_bottom = frame[H // 2 :]
        if remove_redundant_flag:
            # check whether the right image has all redundant gt and error
            first_col = frame_bottom[:, 0]
            if (first_col == first_col[:1]).all() and first_col[0].tolist() == list(
                BORDER_COLOR
            ):
                frame_bottom_w = frame_bottom.shape[1]
                frame_bottom = frame_bottom[
                    :, frame_bottom_w // 3 : -frame_bottom_w // 3
                ]

        frame = np.concatenate([frame_top, frame_bottom], 1)
        re_aranged_list.append(frame)

    cnt = 0
    cur = 0
    T = len(re_aranged_list)
    while cur < T:
        __video_save__(
            viz_vid_fn[:-4] + f"_{cnt}.mp4",
            [
                f[::subsample, ::subsample, :]
                for f in re_aranged_list[cur : cur + max_num_frames]
            ],
        )
        cnt += 1
        cur += max_num_frames

    return


@torch.no_grad()
def viz2d_one_frame(
    model_tid,
    prior2d,
    cams,
    s_model,
    d_model=None,
    subsample=1,
    loss_mask_type="sta",
    view_cam_id=None,
    prefix_text="",
    save_path=None,
    append_depth=True,
    append_dyn=True,
    append_graph=True,
    rgb_mask=None,
    dep_mask=None,
    T_cw=None,
):
    if T_cw is None:
        if view_cam_id is None:
            T_cw = cams.T_cw(model_tid)
        else:
            T_cw = cams.T_cw(view_cam_id)

    # * normal viz
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(model_tid))
    render_dict = render(
        gs5, prior2d.H, prior2d.W, cams.rel_focal, cams.cxcy_ratio, T_cw
    )
    if rgb_mask is None:
        rgb_mask = prior2d.get_mask_by_key(loss_mask_type, model_tid)
    _, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        prior2d, model_tid, render_dict, rgb_mask
    )
    viz_frame = make_viz_np(
        gt_rgb * rgb_mask[:, :, None],
        pred_rgb,
        rgb_loss_i.max(dim=-1).values,
        text0=f"{prefix_text}Fr={model_tid:03d} GT",
        text1=f"{prefix_text}Fr={model_tid:03d} Pred",
        text2=f"{prefix_text}Fr={model_tid:03d} Err",
    )

    # * ed viz
    if append_graph and d_model is not None:
        viz_frame_graph = make_viz_graph(d_model, model_tid, cams, prior2d, view_cam_id)
        viz_frame = np.concatenate([viz_frame, viz_frame_graph], 0)

    # * depth viz
    if append_depth:
        if dep_mask is None:
            dep_mask = prior2d.get_mask_by_key(loss_mask_type + "_dep", model_tid)
        _, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
            prior2d, model_tid, render_dict, dep_mask
        )
        viz_frame_dep = make_viz_np(
            prior_dep * dep_mask,
            pred_dep,
            dep_loss_i,
            text0="DEP Target",
            text1="DEP Pred",
            text2="DEP Error",
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dep], 0)
        # * when append depth, also append normal
        try:
            _, normal_loss_i, pred_normal, gt_normal = compute_normal_loss(
                prior2d, model_tid, render_dict, dep_mask
            )
            viz_frame_normal = make_viz_np(
                (1.0 - gt_normal) / 2.0,
                (1.0 - pred_normal) / 2.0,
                normal_loss_i,
                text0="Nrm GT",
                text1="Nrm Pred",
                text2="Nrm Error",
            )
            viz_frame = np.concatenate([viz_frame, viz_frame_normal], 0)
        except:
            # logging.warning("Failed to viz normal, skip...")
            pass

    # * fg only viz
    if d_model is not None and append_dyn:
        dyn_render_dict = render(
            [d_model(model_tid)],
            prior2d.H,
            prior2d.W,
            cams.rel_focal,
            cams.cxcy_ratio,
            T_cw,
            bg_color=[0.5, 0.5, 0.5],
        )
        _, dyn_rgb_loss_i, dyn_pred_rgb, dyn_gt_rgb = compute_rgb_loss(
            prior2d, model_tid, dyn_render_dict, rgb_mask
        )
        viz_frame_dyn = make_viz_np(
            dyn_gt_rgb,
            dyn_pred_rgb,
            dyn_rgb_loss_i.max(dim=-1).values,
            text0="FG Only",
            text1="FG Pred",
            text2="FG Error",
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dyn], 0)

    viz_frame = viz_frame[::subsample, ::subsample, :]
    if save_path is not None:
        imageio.imwrite(save_path, viz_frame)
    return viz_frame


@torch.no_grad()
def make_viz_graph(
    d_model, view_ind, cams, prior2d, view_cam_id=None, max_radius=0.001
):
    node_mu_w = d_model.scf._node_xyz[d_model.get_tlist_ind(view_ind)]
    if view_cam_id is None:
        render_cam_id = view_ind
    else:
        render_cam_id = view_cam_id
    R_cw, t_cw = cams.Rt_cw(render_cam_id)
    node_mu = node_mu_w @ R_cw.T + t_cw[None]
    order = torch.arange(len(node_mu))
    c_id = torch.from_numpy(cm.hsv(order / len(node_mu))[:, :3]).to(node_mu)
    create_time = (d_model.scf._node_create_time.float() / cams.T).cpu()
    c_time = torch.from_numpy(cm.hsv(create_time)[:, :3]).to(node_mu)
    acc_w = d_model.get_node_sinning_w_acc().detach().cpu().numpy()
    acc_w_binary = (acc_w > float(d_model.scf.skinning_k) / 2.0).astype(np.float32)
    acc_w = acc_w / acc_w.max()
    c_w = torch.from_numpy(cm.viridis(acc_w)[:, :3]).to(node_mu)
    c_wb = torch.from_numpy(cm.viridis(acc_w_binary)[:, :3]).to(node_mu)

    H, W = prior2d.H, prior2d.W
    pf = cams.rel_focal / 2 * min(H, W)
    NUM_SEMANTIC_CHANNELS = prior2d.latent_feature_channel
    viz_frames = []
    # for color in [c_id, c_time, c_w]:
    viz_r = min(max_radius, d_model.scf.spatial_unit / 10.0)
    for color, text in zip(
        [c_id, c_time, c_w], ["Nodes-id", "Nodes-create-t", "Nodes-acc-w"]
    ):
        sph = RGB2SH(color)
        semantic_feature = torch.zeros(len(node_mu), NUM_SEMANTIC_CHANNELS).to(node_mu) # TODO: what about semantic feature? (changed 37 to 128)
        fr = torch.eye(3).to(node_mu)[None].expand(len(node_mu), -1, -1)
        s = torch.ones(len(node_mu), 3).to(node_mu) * viz_r
        o = torch.ones(len(node_mu), 1).to(node_mu) * 1.0

        render_dict = render_cam_pcl(
            node_mu, fr, s, o, sph, semantic_feature, H, W, fx=pf, bg_color=BG_COLOR1
        )
        rgb = render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8).copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = TEXTCOLOR
        lineType = 2
        cv.putText(
            rgb,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        viz_frames.append(rgb)
    ret = np.concatenate(viz_frames, 1).copy()

    return ret


@torch.no_grad()
def viz2d_flow_video(
    solver,
    start_from,
    end_at,
    cams,
    s_model,
    d_model,
    subsample=1,
    view_cam_id=None,
    N=128,
    line_n=32,
    viz_bg=False,
    choice=None,
    text="",
    skip_t=1,
):
    H, W = solver.prior2d.H, solver.prior2d.W
    frame_list = []
    if viz_bg:
        s_mu_w, s_fr_w, s_s, s_o, s_sph, s_semantic_feature = s_model(0)
    prev_mu, _, s_traj, o_traj, sph_traj, semantic_feature_traj = d_model(
        start_from, 0, nn_fusion=-1
    )  # get init coloring
    s_traj = torch.ones_like(s_traj) * 0.0015
    o_traj = torch.ones_like(o_traj) * 0.999

    # sph_int = torch.rand_like(sph_traj)
    # sph_traj = RGB2SH(sph_int)
    # make new color, first sort the prev mu and then use a HSV in the order
    color_id_coord = d_model._xyz
    color_id = (
        color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    )
    color_id = color_id.argsort().float() / len(color_id)
    # track_color = cm.rainbow(color_id.cpu())[:, :3]
    track_color = cm.hsv(color_id.cpu())[:, :3]
    track_color = torch.from_numpy(track_color).to(prev_mu)
    # _track_color = torch.zeros_like(sph_traj)
    # _track_color[:, :3] = track_color
    # track_color = _track_color
    sph_traj = RGB2SH(track_color)
    NUM_SEMANTIC_CHANNELS=s_model.semantic_feature_dim
    semantic_feature_traj = torch.zeros([sph_traj.shape[0], NUM_SEMANTIC_CHANNELS]).to(sph_traj) # changed 37 to 128

    # only viz the first frame visible points!!
    render_dict = render(
        [s_model(0), d_model(start_from, 0, nn_fusion=-1)],
        H,
        W,
        cams.rel_focal,
        cams.cxcy_ratio,
        cams.T_cw(start_from),
    )
    visibility_mask = render_dict["visibility_filter"][-d_model.N :]
    valid_ind = torch.arange(len(visibility_mask)).to(visibility_mask.device)[
        visibility_mask
    ]
    if choice is None:
        choice = valid_ind[torch.randperm(len(valid_ind))[:N]]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    semantic_feature_traj = semantic_feature_traj[choice]
    prev_mu = prev_mu[choice]

    mu_w = torch.zeros(0, 3).to(prev_mu)
    fr_w = torch.zeros(0, 3, 3).to(prev_mu)
    s = torch.zeros(0, 3).to(prev_mu)
    o = torch.zeros(0, 1).to(prev_mu)
    sph = torch.zeros(0, sph_traj.shape[-1]).to(prev_mu)
    semantic_feature = torch.zeros(0, semantic_feature_traj.shape[-1]).to(prev_mu)
    for view_ind in tqdm(range(start_from, min(end_at + 1, cams.T))):
        if view_ind % skip_t != 0:
            continue
        d_mu_w, d_fr_w, d_s, d_o, d_sph, d_semantic_feature = d_model(view_ind, 0, nn_fusion=-1)
        # draw the line
        src_mu = prev_mu
        dst_mu = d_mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * 0.0015 * 0.3
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * 0.999
        intermediate_sph = (
            sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, sph_traj.shape[-1])
        )
        intermediate_semantic_feature = (
            semantic_feature_traj[:, None].expand(-1, line_n, -1).reshape(-1, semantic_feature_traj.shape[-1])
        )
        prev_mu = dst_mu

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), d_mu_w[choice].clone()], 0)
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), d_fr_w[choice].clone()], 0)
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)
        semantic_feature = torch.cat([semantic_feature, intermediate_semantic_feature.clone(), semantic_feature_traj.clone()], 0)
        # transform
        if view_cam_id is None:
            _render_cam_id = view_ind
        else:
            _render_cam_id = view_cam_id
        if viz_bg:
            working_mu_w = torch.cat([s_mu_w, mu_w, d_mu_w], 0)
            working_fr_w = torch.cat([s_fr_w, fr_w, d_fr_w], 0)
            working_s = torch.cat([s_s, s, d_s], 0)
            working_o = torch.cat([s_o, o, d_o], 0)
            working_sph = torch.cat([s_sph, sph, d_sph], 0)
            working_semantic_feature = torch.cat([s_semantic_feature, semantic_feature, d_semantic_feature], 0)
        else:
            working_mu_w = mu_w
            working_fr_w = fr_w
            working_s = s
            working_o = o
            working_sph = sph
            working_semantic_feature = semantic_feature
        R_cw, t_cw = cams.Rt_cw(_render_cam_id)
        working_mu_cur = (
            torch.einsum("ij, nj->ni", R_cw, working_mu_w.clone()) + t_cw[None]
        )
        working_fr_cur = torch.einsum("ij, njk->nik", R_cw, working_fr_w.clone())
        # render
        pf = cams.rel_focal / 2 * min(solver.prior2d.H, solver.prior2d.W)
        assert (
            len(working_mu_cur)
            == len(working_fr_cur)
            == len(working_s)
            == len(working_o)
            == len(working_sph)
        )
        render_dict = render_cam_pcl(
            working_mu_cur.contiguous(),
            working_fr_cur.contiguous(),
            working_s.contiguous(),
            working_o.contiguous(),
            working_sph.contiguous(),
            working_semantic_feature.contiguous(),
            H,
            W,
            fx=pf,
            bg_color=BG_COLOR1,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0)
        viz_frame = pred_rgb.detach().cpu().numpy()
        viz_frame = (np.clip(viz_frame, 0.0, 1.0) * 255).astype(np.uint8).copy()
        if len(text) > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = TEXTCOLOR
            lineType = 2
            cv.putText(
                viz_frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        viz_frame = viz_frame[::subsample, ::subsample, :]
        frame_list.append(viz_frame)
    # # ! debug
    # np.savetxt(
    #     "./debug/flow.xyz",
    #     torch.cat([mu_w[model.N :], torch.clamp((SH2RGB(s[model.N :]) * 255),0,255).long()], -1)
    #     .detach()
    #     .cpu()
    #     .numpy(),
    # )
    return frame_list, choice


@torch.no_grad()
def viz3d_total_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    save_path,
    res=240,
    s_model=None,
    time_interval=5,
    bg_color=BG_COLOR1,
    max_num_frames=500,  # 150,
):
    logging.info(f"Viz 3D scene from {start_tid} to {end_tid} ...")
    node_xyz = d_model.scf._node_xyz
    NUM_SEMANTIC_CHANNELS = d_model.semantic_feature_dim
    frames_node = viz_curve(
        node_xyz,
        torch.zeros_like(node_xyz),
        torch.zeros([node_xyz.shape[0], node_xyz.shape[1], NUM_SEMANTIC_CHANNELS]), # changed 37 to 128
        torch.ones_like(node_xyz[..., 0]).bool(),
        cams,
        res=res,
        text="All ED Nodes",
        viz_n=-1,
        # viz_n=128,
        time_window=30,
    )

    # first row is all
    frames1 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            bg_color=bg_color,
        )
    )
    frames2 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            bg_color=BG_COLOR1,
        )
    )
    frames3 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            bg_color=bg_color,
        )
    )
    frames4 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            bg_color=BG_COLOR1,
        )
    )

    frames_up = [
        np.concatenate([frames1[i], frames2[i]], 1) for i in range(len(frames1))
    ]
    frames_down = [
        np.concatenate([frames3[i], frames4[i]], 1) for i in range(len(frames3))
    ]
    frames = [
        np.concatenate([frames_up[i], frames_down[i]], 0) for i in range(len(frames1))
    ]
    frames = [
        np.concatenate([frames[i], frames_node[i] * 255], 1)
        for i in range(len(frames1))
    ]

    # __video_save__(save_path, frames)
    cnt = 0
    cur = 0
    T = len(frames)
    while cur < T:
        __video_save__(
            save_path[:-4] + f"_{cnt}.mp4", frames[cur : cur + max_num_frames]
        )
        cnt += 1
        cur += max_num_frames
    return


def _combine_frames(frames: dict):
    T = len(frames[list(frames.keys())[0]])
    for v in frames.values():
        assert len(v) == T
    ret = []
    for i in range(T):
        ret.append(np.concatenate([frames[k][i] for k in frames.keys()], 1))
    return ret


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


@torch.no_grad()
def viz3d_scene_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    bg_color=BG_COLOR1,
):
    viz_cam_R = q2R(cams.q_wc[start_tid : end_tid + 1])
    viz_cam_t = cams.t_wc[start_tid : end_tid + 1]
    viz_cam_R, viz_cam_t = cams.Rt_wc_list()
    viz_cam_R = viz_cam_R[start_tid : end_tid + 1].clone()
    viz_cam_t = viz_cam_t[start_tid : end_tid + 1].clone()

    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}
    for viz_time in tqdm(range(start_tid, end_tid + 1)):
        gs5_param = d_model(viz_time)
        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model())
        viz_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=gs5_param,
            draw_camera_frames=False,
            bg_color=bg_color,
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


@torch.no_grad()
def viz3d_scene_flow_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    N=128,
    line_n=16,
    time_window=10,
    bg_color=BG_COLOR1,
):
    viz_R = quaternion_to_matrix(cams.q_wc[start_tid : end_tid + 1])
    viz_t = cams.t_wc[start_tid : end_tid + 1]
    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}

    # prepare flow gaussians
    prev_mu, _, s_traj, o_traj, sph_traj, semantic_feature_traj = d_model(
        start_tid, 0, nn_fusion=-1
    )  # get init coloring
    s_traj = torch.ones_like(s_traj) * 0.0015
    o_traj = torch.ones_like(o_traj) * 0.999
    sph_int = torch.rand_like(sph_traj)
    sph_traj = RGB2SH(sph_int)
    NUM_SEMANTIC_CHANNELS=d_model.semantic_feature_dim
    semantic_feature_traj = torch.zeros([sph_traj.shape[0], NUM_SEMANTIC_CHANNELS]).to(sph_traj) # changed 37 to 128

    choice = torch.randperm(len(prev_mu))[:N]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    semantic_feature_traj = semantic_feature_traj[choice]
    prev_mu = prev_mu[choice]

    # dummy
    mu_w, fr_w, s, o, sph, semantic_feature = d_model(start_tid, 0, nn_fusion=-1)
    s = s * 0.0
    o = o * 0.0

    for viz_time in range(start_tid, end_tid + 1):

        _mu_w, _fr_w, _, _, _,_ = d_model(viz_time, 0, nn_fusion=-1)
        # draw the line
        src_mu = prev_mu
        dst_mu = _mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * 0.0015 * 0.3
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * 0.999
        intermediate_sph = sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, 3)
        intermediate_semantic_feature = semantic_feature_traj[:, None].expand(-1, line_n, -1).reshape(-1, NUM_SEMANTIC_CHANNELS) # changed 37 to 128
        prev_mu = dst_mu
        # pad
        if sph.shape[1] > 3:
            intermediate_sph = torch.cat(
                [
                    intermediate_sph,
                    torch.zeros(len(intermediate_sph), sph.shape[1] - 3).to(sph),
                ],
                1,
            )
            intermediate_semantic_feature = torch.cat(
                [
                    intermediate_semantic_feature,
                    torch.zeros(len(intermediate_semantic_feature), semantic_feature.shape[1] - NUM_SEMANTIC_CHANNELS).to(semantic_feature), # changed 37 to 128
                ],
                1,
            )

        one_time_N = len(src_mu) * (line_n + 1)
        max_N = time_window * N

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), _mu_w[choice].clone()], 0)[
            -max_N:
        ]
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), _fr_w[choice].clone()], 0)[
            -max_N:
        ]
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)[-max_N:]
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)[-max_N:]
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)[-max_N:]
        semantic_feature = torch.cat([semantic_feature, intermediate_semantic_feature.clone(), semantic_feature_traj.clone()], 0)[-max_N:]
        gs5_param = (mu_w, fr_w, s, o, sph, semantic_feature)

        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model(0))
        viz_dict = viz_scene(
            res, res, viz_R, viz_t, viz_f=viz_f, gs5_param=gs5_param, bg_color=bg_color
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


def cat_gs(m1, f1, s1, o1, c1, feat1, m2, f2, s2, o2, c2, feat2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    feat = torch.cat([feat1, feat2], dim=0).contiguous() # TODO: what about semantic feature?
    # feat = torch.zeros(m.shape[0], NUM_SEMANTIC_CHANNELS).to(m) # TODO: what about semantic feature? (changed 37 to 128)
    return m, f, s, o, c, feat


def viz_o_hist(model, save_path, title_text=""):
    o = model.get_o.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(o, bins=100)
    plt.title(f"{title_text} o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_s_hist(model, save_path, title_text=""):
    s = model.get_s.detach()
    s = s.sort(dim=-1).values
    s = s.cpu().numpy()
    fig = plt.figure(figsize=(20, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(s[..., i], bins=100)
        plt.title(f"{title_text} s hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_sigma_hist(scf, save_path, title_text=""):
    sig = scf.node_sigma.abs().detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    # sig = sig.reshape(-1)
    if sig.shape[1] == 1:
        plt.hist(sig, bins=100)
        plt.title(f"{title_text} Node Sigma hist (Total {scf.M} nodes)")
    else:
        C = sig.shape[1]
        for i in range(C):
            plt.subplot(1, C, i + 1)
            plt.hist(sig[:, i], bins=100)
            plt.title(f"{title_text} Node Sigma [{scf.M}] dim={i}")
    plt.savefig(save_path)
    plt.close()
    return


def viz_dyn_o_hist(model, save_path, title_text=""):
    dyn_o = model.get_d.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(dyn_o, bins=100)
    plt.title(f"{title_text} dyn_o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_hist(d_model, viz_dir, postfix):
    viz_s_hist(d_model, osp.join(viz_dir, f"s_hist_{postfix}.jpg"))
    viz_o_hist(d_model, osp.join(viz_dir, f"o_hist_{postfix}.jpg"))


def viz_dyn_hist(scf, viz_dir, postfix):
    viz_sigma_hist(scf, osp.join(viz_dir, f"sigma_hist_{postfix}.jpg"))
    # viz_dyn_o_hist(d_model, osp.join(viz_dir, f"dyn_o_hist_{postfix}.jpg"))
    # viz the skinning K count
    valid_sk_count = scf.topo_knn_mask.sum(-1).detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(valid_sk_count), plt.title(f"Valid node neighbors count {scf.M}")
    plt.savefig(osp.join(viz_dir, f"valid_sk_count_{postfix}.jpg"))
    return


def viz_N_count(N_count_list, path):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(N_count_list), plt.title("Noodle Count")
    plt.savefig(path)
    plt.close()


def viz_sup_cnt(cnt_list, path):
    # plot bar plot of all the counts
    fig = plt.figure(figsize=(8, 6))
    plt.bar(range(len(cnt_list)), cnt_list), plt.title("Sup Count")
    plt.savefig(path)
    plt.close()
    return


def viz_depth_list(depth_list_pt, save_path):
    assert isinstance(depth_list_pt, torch.Tensor)

    # depth_min = depth_list_pt.min()
    # depth_max = depth_list_pt.max()
    # depth_list_pt = (depth_list_pt - depth_min) / (depth_max - depth_min)
    viz_list = []
    for dep in tqdm(depth_list_pt):
        depth_min = dep.min()
        depth_max = dep.max()
        dep = (dep - depth_min) / (depth_max - depth_min)
        viz = cm.viridis(dep.detach().cpu().numpy())[:, :, :3]
        viz_list.append(viz)
    __video_save__(save_path, viz_list)
    return


def viz_global_ba(
    point_world,
    rgb,
    semantic_feature,
    mask,
    cams,
    pts_size=0.001,
    res=480,
    error=None,
    text="",
):
    T, M = point_world.shape[:2]
    device = point_world.device
    mu = point_world.clone()[mask]
    sph = RGB2SH(rgb.clone()[mask])
    # semantic_feature=semantic_feature.clone()[mask]
    feat=semantic_feature.clone()[mask]
    s = torch.ones_like(mu) * pts_size
    fr = torch.eye(3, device=device).expand(len(mu), -1, -1)
    o = torch.ones(len(mu), 1, device=device)
    viz_cam_R = quaternion_to_matrix(cams.q_wc)
    viz_cam_t = cams.t_wc
    viz_cam_R, viz_cam_t = cams.Rt_wc_list()
    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frame_dict = viz_scene(
        res,
        res,
        viz_cam_R,
        viz_cam_t,
        viz_f=viz_f,
        gs5_param=(mu, fr, s, o, sph, feat),
        bg_color=BG_COLOR1,
        draw_camera_frames=True,
    )
    frame = np.concatenate([v for v in frame_dict.values()], 1)
    # * also do color and error if provided
    id_color = cm.hsv(np.arange(M, dtype=np.float32) / M)[:, :3]
    id_color = torch.from_numpy(id_color).to(device)
    id_color = id_color[None].expand(T, -1, -1)
    sph = RGB2SH(id_color[mask])
    id_frame_dict = viz_scene(
        res,
        res,
        viz_cam_R,
        viz_cam_t,
        viz_f=viz_f,
        gs5_param=(mu, fr, s, o, sph, feat),
        bg_color=BG_COLOR1,
        draw_camera_frames=True,
    )
    if error is None:
        id_frame = np.concatenate([v for v in id_frame_dict.values()], 1)
        frame = np.concatenate([frame, id_frame], 0)
    else:
        # render error as well
        error = error[mask]
        error_th = error.max()
        error_color = (error / (error_th + 1e-9)).detach().cpu().numpy()
        text = text + f" ErrorVizTh={error_th:.6f}"
        error_color = cm.viridis(error_color)[:, :3]
        error_color = torch.from_numpy(error_color).to(device)
        sph = RGB2SH(error_color)
        error_frame_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=(mu, fr, s, o, sph, feat),
            bg_color=BG_COLOR1,
            draw_camera_frames=True,
        )
        add_frame = np.concatenate(
            [
                id_frame_dict["scene_camera_20deg"],
                error_frame_dict["scene_camera_20deg"],
            ],
            1,
        )
        frame = np.concatenate([frame, add_frame], 0)
    # imageio.imsave("./debug/viz.jpg", frame)
    frame = frame.copy()
    if len(text) > 0:
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (1.0, 1.0, 1.0)
        lineType = 2
        cv.putText(
            frame,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
    return frame


def viz_plt_missing_slot(track_mask, path, max_viz=2048):
    # T,N
    T, N = track_mask.shape
    choice = torch.randperm(N)[:max_viz]

    viz_mask = track_mask[:, choice].clone().float()
    resort = torch.argsort(viz_mask.sum(0), descending=True)
    viz_mask = viz_mask[:, resort]
    plt.figure(figsize=(2.0 * max_viz / T, 3.0))
    plt.imshow((viz_mask * 255.0).cpu().numpy(), cmap="viridis")
    plt.title("MissingSlot=0"), plt.xlabel("Sorted Noodles"), plt.ylabel("T")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # also viz how many valid slot in each frame
    valid_count = viz_mask.sum(1)
    plt.figure(figsize=(5, 3))
    # use bar plot
    plt.bar(range(T), valid_count.cpu().numpy())
    plt.title("ValidSlotCount"), plt.xlabel("T"), plt.ylabel("ValidCount")
    plt.tight_layout()
    plt.savefig(path.replace(".jpg", "_count_perframe.jpg"))
    plt.close()

    return


@torch.no_grad()
def viz_curve(
    curve_world,
    rgb,
    semantic_feature,
    mask,
    cams,
    res=480,
    pts_size=0.0003,
    viz_n=128,
    n_line=8,
    line_radius_factor=0.1,
    only_viz_last_frame=False,
    no_rgb_viz=True,
    text="",
    time_window=10,
):
    # TODO: add error viz
    # draw the curve
    T, M = curve_world.shape[:2]
    device = curve_world.device
    if viz_n == -1:
        viz_n = M
        choice = torch.arange(M).to(device)
    else:
        step = max(1, M // viz_n)
        choice = torch.arange(M)[::step].to(device)
        viz_n = len(choice)
    mu = curve_world[:, choice].clone()
    rgb = rgb[:, choice].clone()
    semantic_feature = semantic_feature.clone() # TODO: what about the mask for semantic feature?
    NUM_SEMANTIC_CHANNELS = semantic_feature.shape[-1]
    if no_rgb_viz:
        rgb = torch.zeros_like(rgb)
        rgb[..., 1] = 1.0
    mask = mask[:, choice].clone()

    id_sph = torch.from_numpy(
        cm.hsv(np.arange(viz_n, dtype=np.float32) / viz_n)[:, :3]
    ).to(rgb)
    id_sph = RGB2SH(id_sph)

    frame_list = []

    line_mu = torch.zeros([0, 3], device=device)
    line_fr = torch.zeros([0, 3, 3], device=device)
    line_s = torch.zeros([0, 3], device=device)
    line_o = torch.zeros([0, 1], device=device)
    line_sph = torch.zeros([0, 3], device=device)
    line_semantic_feature = torch.zeros([0, NUM_SEMANTIC_CHANNELS], device=device) # (keep this at 1408 or illegal memory error will occur after dyn gaussain training)

    # only draw stuff within a time window!
    for t in tqdm(range(T)):
        # draw the continuous geo
        _mu, _rgb, _semantic_feature, _mask = (
            mu[max(0, t - time_window + 1) : t + 1].reshape(-1, 3),
            rgb[max(0, t - time_window + 1) : t + 1].reshape(-1, 3),
            semantic_feature[max(0, t - time_window + 1) : t + 1].reshape(-1),
            mask[max(0, t - time_window + 1) : t + 1].reshape(-1),
        )
        _rgb[~_mask] = torch.Tensor([1.0, 0.0, 0.0]).to(device)
        _sph = RGB2SH(_rgb.clone())
        _semantic_feature = torch.zeros(len(_mu), NUM_SEMANTIC_CHANNELS).to(device) # TODO: what about semantic feature? (keep this at 1408 or illegal memory error will occur after dyn gaussain training)
        _s = torch.ones_like(_mu) * pts_size
        _fr = torch.eye(3, device=device).expand(len(_mu), -1, -1)
        _o = torch.ones(len(_mu), 1, device=device)

        if t == 0:
            line_mu = _mu
            line_fr = _fr
            line_s = _s
            line_o = _o
            line_sph = id_sph
            line_semantic_feature = _semantic_feature
        else:
            # ! this can't grow as time grows, for super long sequence !!
            prev_mu = mu[t - 1]
            cur_mu = mu[t]
            _line_mu = (
                prev_mu[None]
                + torch.linspace(0.0, 1.0, n_line, device=device)[:, None, None]
                * (cur_mu - prev_mu)[None]
            )
            _line_fr = torch.eye(3, device=device)[None, None].expand(
                n_line, viz_n, -1, -1
            )
            _line_s = (
                torch.ones(n_line, viz_n, 3, device=device)
                * pts_size
                * line_radius_factor
            )
            _line_s[0] = pts_size
            _line_s[-1] = pts_size
            _line_o = torch.ones(n_line, viz_n, 1, device=device)
            _line_sph = id_sph[None].expand(n_line, -1, -1)
            _line_semantic_feature = torch.zeros(n_line, viz_n, NUM_SEMANTIC_CHANNELS).to(device)

            _N = viz_n * n_line * time_window
            line_mu = torch.cat([line_mu, _line_mu.reshape(-1, 3)], 0)[-_N:]
            line_fr = torch.cat([line_fr, _line_fr.reshape(-1, 3, 3)], 0)[-_N:]
            line_s = torch.cat([line_s, _line_s.reshape(-1, 3)], 0)[-_N:]
            line_o = torch.cat([line_o, _line_o.reshape(-1, 1)], 0)[-_N:]
            line_sph = torch.cat([line_sph, _line_sph.reshape(-1, 3)], 0)[-_N:]
            line_semantic_feature = torch.cat([line_semantic_feature, _line_semantic_feature.reshape(-1, NUM_SEMANTIC_CHANNELS)], 0)[-_N:]
        # render
        if only_viz_last_frame and t < T - 1:
            continue

        # viz_cam_R = quaternion_to_matrix(cams.q_wc)[: t + 1]
        # viz_cam_t = cams.t_wc[: t + 1]
        viz_cam_R, viz_cam_t = cams.Rt_wc_list()
        viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
        #print("Huis question 1:", _semantic_feature.shape)
        assert _sph.shape[0] == _semantic_feature.shape[0]

        frame_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=(_mu, _fr, _s, _o, _sph, _semantic_feature),
            bg_color=BG_COLOR1,
            draw_camera_frames=False,
        )
        frame = np.concatenate([v for v in frame_dict.values()], 1)
        frame_dict = viz_scene(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            gs5_param=(line_mu, line_fr, line_s, line_o, line_sph, line_semantic_feature),
            bg_color=BG_COLOR1,
            draw_camera_frames=False,
        )
        #torch.cuda.empty_cache()
        frame_line = np.concatenate([v for v in frame_dict.values()], 1)
        frame = np.concatenate([frame, frame_line], 0).copy()

        if len(text) > 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (1.0, 1.0, 1.0)
            lineType = 2
            cv.putText(
                frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        frame_list.append(frame)
    return frame_list


@torch.no_grad()
def viz_mv_model_frame(
    prior2d, cams, s_model, scf, mv_model, t, support_t, T_cw=None, support_m=None
):
    if s_model is None:
        gs5 = []
        order = 0
    else:
        gs5 = [s_model()]
        order = s_model.max_sph_order
    if support_m is not None:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            support_mask=support_m,
            pad_sph_order=order,
        )
    else:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            torch.Tensor([support_t]).long(),
            pad_sph_order=order,
        )
    gs5.append(d_gs5)
    if T_cw is None:
        T_cw = cams.T_cw(t)
    render_dict = render(
        gs5,
        prior2d.H,
        prior2d.W,
        cams.rel_focal,
        cams.cxcy_ratio,
        T_cw,
    )
    # do the rendering loss
    rgb_sup_mask = prior2d.get_mask_by_key("all", t)
    loss_rgb, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        prior2d, t, render_dict, rgb_sup_mask
    )
    dep_sup_mask = prior2d.get_mask_by_key("all_dep", t)
    loss_dep, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
        prior2d, t, render_dict, dep_sup_mask
    )
    viz_figA = make_viz_np(
        gt_rgb,
        pred_rgb,
        rgb_loss_i.max(-1).values,
        text0=f"q={t},s={support_t}",
    )
    viz_figB = make_viz_np(
        prior_dep,
        pred_dep,
        dep_loss_i,
        text0=f"q={t},s={support_t}",
    )
    ret = np.concatenate([viz_figA, viz_figB], 0)
    return ret


@torch.no_grad()
def viz_scf_frame(prior2d, cams, s_model, scf, t, T_cw=None, opa=0.3, r=None):

    s_gs5 = s_model()

    node_xyz = scf._node_xyz[t]
    node_R = q2R(scf._node_rotation[t])
    node_scale = scf.node_sigma[:, None].expand(-1, 3)
    if r is not None:
        node_scale = torch.ones_like(node_scale) * r
    node_opacity = torch.ones_like(node_xyz[..., :1]) * opa
    node_color = cm.hsv(np.arange(len(node_xyz), dtype=np.float32) / len(node_xyz))[
        :, :3
    ]
    node_sph = RGB2SH(torch.from_numpy(node_color).to(node_xyz))
    # pad node_sph
    if node_sph.shape[1] != s_gs5[-1].shape[1]:
        node_sph = torch.cat(
            [
                node_sph,
                torch.zeros(len(node_sph), s_gs5[-1].shape[1] - node_sph.shape[1]).to(
                    node_sph
                ),
            ],
            1,
        )
    node_gs5 = (node_xyz, node_R, node_scale, node_opacity, node_sph)

    if T_cw is None:
        T_cw = cams.T_cw(t)
    render_dict = render(
        [s_gs5, node_gs5],
        prior2d.H,
        prior2d.W,
        cams.rel_focal,
        cams.cxcy_ratio,
        T_cw,
    )
    # do the rendering loss
    rgb_sup_mask = prior2d.get_mask_by_key("all", t)
    loss_rgb, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        prior2d, t, render_dict, rgb_sup_mask
    )
    dep_sup_mask = prior2d.get_mask_by_key("all_dep", t)
    loss_dep, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
        prior2d, t, render_dict, dep_sup_mask
    )
    viz_figA = make_viz_np(
        gt_rgb,
        pred_rgb,
        rgb_loss_i.max(-1).values,
        text0=f"q={t}",
    )
    viz_figB = make_viz_np(
        prior_dep,
        pred_dep,
        dep_loss_i,
        text0=f"q={t}",
    )
    ret = np.concatenate([viz_figA, viz_figB], 0)
    return ret


@torch.no_grad()
def viz_d_model_grad(prior2d, d_model, cams, max_grad_value=None):
    d_acc_grad = d_model.xyz_gradient_accum / torch.clamp(
        d_model.xyz_gradient_denom, 1e-6
    )
    if max_grad_value is not None:
        d_acc_grad = torch.clamp(d_acc_grad, 0, max_grad_value)
    d_acc_grad_max = d_acc_grad.max() + 1e-6
    d_acc_grad_viz_color = d_acc_grad / d_acc_grad_max
    d_acc_grad_viz_color = cm.viridis(d_acc_grad_viz_color.cpu().numpy())[:, :3]
    d_acc_grad_viz_color = torch.from_numpy(d_acc_grad_viz_color).to(d_acc_grad)
    viz_grad_sph = RGB2SH(d_acc_grad_viz_color)
    viz_grad_list = []
    for t in tqdm(range(d_model.T)):
        gs5 = list(d_model(t))
        gs5[-2] = viz_grad_sph
        viz_render_dict = render(
            gs5,
            prior2d.H,
            prior2d.W,
            cams.rel_focal,
            cams.cxcy_ratio,
            cams.T_cw(t),
            bg_color=[0.5, 0.5, 0.5],
        )
        viz_grad = viz_render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
        viz_grad = (np.clip(viz_grad, 0, 1) * 255).astype(np.uint8).copy()
        # use opencv to put the normalization factor as text
        viz_grad = cv.putText(
            viz_grad,
            f"Max={d_acc_grad_max:.6f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        viz_grad_list.append(viz_grad)
    return viz_grad_list


def __video_save__(fn, imgs, fps=10):
    logging.info(f"Saving video to {fn} ...")
    # H, W = imgs[0].shape[:2]
    # image_size = (W, H)
    # out = cv.VideoWriter(fn, cv.VideoWriter_fourcc(*"MP4V"), fps, image_size)
    # for img in tqdm(imgs):
    #     out.write(img[..., ::-1])
    # out.release()
    imageio.mimsave(fn, imgs)
    logging.info(f"Saved!")
    return
