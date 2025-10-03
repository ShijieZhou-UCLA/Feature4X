# This file handle all the gaussian based rendering that requires plotting 3D GS additionally
import torch, numpy as np, cv2 as cv
import torch
from torch import nn
import torch.nn.functional as F
from glob import glob
import imageio
import os, sys, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from lib_4d.render_helper import render_cam_pcl
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from transforms3d.euler import mat2euler, euler2mat

import imageio.core.util


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

import yaml



def make_video(src_pattern, dst):
    image_fns = glob(src_pattern)
    image_fns.sort()
    if len(image_fns) == 0:
        print(f"no image found in {src_pattern}")
        return
    frames = []
    for i, fn in enumerate(image_fns):
        img = cv.imread(fn)[..., ::-1]
        frames.append(img)
    imageio.mimwrite(dst, frames)


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def cat_gs(m1, f1, s1, o1, c1, feat1, m2, f2, s2, o2, c2, feat2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    feat = torch.cat([feat1, feat2], dim=0).contiguous() # TODO: what about semantic feature?
    # feat = torch.zeros(m.shape[0], NUM_SEMANTIC_CHANNELS).to(m) # TODO: what about semantic feature? (changed 37 to 128)
    return m, f, s, o, c, feat


def draw_line(start, end, radius, rgb, opa=1.0, semantic_feature_dim=32):
    if not isinstance(start, torch.Tensor):
        start = torch.as_tensor(start)
    if not isinstance(end, torch.Tensor):
        end = torch.as_tensor(end)
    line_len = torch.norm(end - start)
    assert line_len > 0
    N = line_len / radius * 3
    line_dir = (end - start) / line_len
    # draw even points on the line
    mu = torch.linspace(0, float(line_len), int(N)).to(start)
    mu = start + mu[:, None] * line_dir[None]
    fr = torch.eye(3)[None].to(mu).expand(len(mu), -1, -1)
    s = radius * torch.ones(len(mu), 3).to(mu)
    o = opa * torch.ones(len(mu), 1).to(mu)
    assert len(rgb) == 3
    c = torch.as_tensor(rgb)[None].to(mu) * torch.ones(len(mu), 3).to(mu)
    c = RGB2SH(c)
    semantic_feature = torch.zeros(len(mu), semantic_feature_dim).to(mu) # TODO: what about semantic feature? (changed 37 to 128)
    return mu, fr, s, o, c, semantic_feature


def draw_frame(R_wc, t_wc, size=0.1, weight=0.01, color=None, opa=1.0, semantic_feature_dim=32):
    if not isinstance(R_wc, torch.Tensor):
        R_wc = torch.as_tensor(R_wc)
    if not isinstance(t_wc, torch.Tensor):
        t_wc = torch.as_tensor(t_wc)
    origin = t_wc
    for i in range(3):
        end = t_wc + size * R_wc[:, i]
        if color is None:
            _color = torch.eye(3)[i].to(R_wc)
        else:
            _color = torch.as_tensor(color).to(R_wc)
        _mu, _fr, _s, _o, _c, _semantic_feature = draw_line(origin, end, weight, _color, opa, semantic_feature_dim=semantic_feature_dim)
        if i == 0:
            mu, fr, s, o, rgb, semantic_feature = _mu, _fr, _s, _o, _c, _semantic_feature
        else:
            mu, fr, s, o, rgb, semantic_feature = cat_gs(mu, fr, s, o, rgb, semantic_feature, _mu, _fr, _s, _o, _c, _semantic_feature)
    return mu, fr, s, o, rgb, semantic_feature


def look_at_R(look_at, cam_center, right_dir=None):
    if right_dir is None:
        right_dir = torch.tensor([1.0, 0.0, 0.0]).to(look_at)
    z_dir = F.normalize(look_at - cam_center, dim=0)
    y_dir = F.normalize(torch.cross(z_dir, right_dir), dim=0)
    x_dir = F.normalize(torch.cross(y_dir, z_dir), dim=0)
    R = torch.stack([x_dir, y_dir, z_dir], 1)
    return R


def add_camera_frame(
    gs5_param, cam_R_wc, cam_t_wc, viz_first_n_cam=-1, add_global=False
):
    mu_w, fr_w, s, o, sph, semantic_feature = gs5_param
    semantic_feature_dim = semantic_feature.shape[-1]
    N_scene = len(mu_w)
    if viz_first_n_cam <= 0:
        viz_first_n_cam = len(cam_R_wc)
    for i in range(viz_first_n_cam):
        if cam_R_wc.ndim == 2:
            R_wc = quaternion_to_matrix(F.normalize(cam_R_wc[i : i + 1], dim=-1))[0]
        else:
            assert cam_R_wc.ndim == 3
            R_wc = cam_R_wc[i]
        t_wc = cam_t_wc[i]
        _mu, _fr, _s, _o, _sph, _semantic_feature = draw_frame(
            R_wc.clone(), t_wc.clone(), size=0.1, weight=0.0003, semantic_feature_dim=semantic_feature_dim
        )
        # pad the _sph to have same order with the input
        if sph.shape[1] > 3:
            _sph = torch.cat(
                [_sph, torch.zeros(len(_sph), sph.shape[1] - 3).to(_sph)], dim=1
            )
        mu_w, fr_w, s, o, sph, semantic_feature = cat_gs(mu_w, fr_w, s, o, sph, semantic_feature, _mu, _fr, _s, _o, _sph, _semantic_feature)
    if add_global:
        _mu, _fr, _s, _o, _sph, _semantic_feature = draw_frame(
            torch.eye(3).to(s),
            torch.zeros(3).to(s),
            size=0.3,
            weight=0.001,
            color=[0.5, 0.5, 0.5],
            opa=0.3,
            semantic_feature_dim=semantic_feature_dim,
        )
        mu_w, fr_w, s, o, sph, semantic_feature = cat_gs(mu_w, fr_w, s, o, sph, semantic_feature, _mu, _fr, _s, _o, _sph, _semantic_feature)
    cam_pts_mask = torch.zeros_like(o.squeeze(-1)).bool()
    cam_pts_mask[N_scene:] = True
    return mu_w, fr_w, s, o, sph, semantic_feature, cam_pts_mask


def get_global_viz_cam_Rt(
    mu_w,
    param_cam_R_wc,
    param_cam_t_wc,
    viz_f,
    z_downward_deg=0.0,
    factor=1.0,
    auto_zoom_mask=None,
    scene_center_mode="mean",
    shift_margin_ratio=1.5,
):
    # always looking towards the scene center
    if scene_center_mode == "mean":
        scene_center = mu_w.mean(0)
    else:
        scene_bound_max, scene_bound_min = mu_w.max(0)[0], mu_w.min(0)[0]
        scene_center = (scene_bound_max + scene_bound_min) / 2.0
    cam_center = param_cam_t_wc.mean(0)

    cam_z_direction = F.normalize(scene_center - cam_center, dim=0)
    cam_y_direction = F.normalize(
        torch.cross(cam_z_direction, param_cam_R_wc[0, :, 0]),
        dim=0,
    )
    cam_x_direction = F.normalize(
        torch.cross(cam_y_direction, cam_z_direction),
        dim=0,
    )
    R_wc = torch.stack([cam_x_direction, cam_y_direction, cam_z_direction], 1)
    additional_R = euler2mat(-np.deg2rad(z_downward_deg), 0, 0, "rxyz")
    additional_R = torch.as_tensor(additional_R).to(R_wc)
    R_wc = R_wc @ additional_R
    # transform the mu to cam_R and then identify the distance
    mu_viz_cam = (mu_w - scene_center[None, :]) @ R_wc.T
    desired_shift = (
        viz_f / factor * mu_viz_cam[:, :2].abs().max(-1)[0] - mu_viz_cam[:, 2]
    )
    # # the nearest point should be in front of camera!
    # nearest_shift =   mu_viz_cam[:, :2].mean()
    # desired_shift = max(desired_shift)
    if auto_zoom_mask is not None:
        desired_shift = desired_shift[auto_zoom_mask]
    shift = desired_shift.max() * shift_margin_ratio
    t_wc = -R_wc[:, -1] * shift + scene_center
    return R_wc, t_wc


@torch.no_grad()
def viz_scene(
    H,
    W,
    param_cam_R_wc,
    param_cam_t_wc,
    model=None,
    viz_f=40.0,
    save_name=None,
    viz_first_n_cam=-1,
    gs5_param=None,
    bg_color=[1.0, 1.0, 1.0],
    draw_camera_frames=False,
    return_full=False,
):
    # auto select viewpoint
    # manually add the camera viz to to
    if model is None:
        assert gs5_param is not None
        mu_w, fr_w, s, o, sph, semantic_feature = gs5_param
    else:
        mu_w, fr_w, s, o, sph, semantic_feature = model()

    #print("Huis question 2:", semantic_feature.shape)
    # add the cameras to the GS
    if draw_camera_frames:
        mu_w, fr_w, s, o, sph, semantic_feature, cam_viz_mask = add_camera_frame(
            (mu_w, fr_w, s, o, sph, semantic_feature), param_cam_R_wc, param_cam_t_wc, viz_first_n_cam
        )
        #print("add camera_frames called!", semantic_feature.shape)

    # * prepare the viz camera
    # * (1) global scene viz
    # viz camera set manually
    global_R_wc, global_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
    )
    global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20
    )
    if draw_camera_frames:
        camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
        )
        camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            20,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
        )

    ret = {}
    ret_full = {}
    todo = {  # "scene_global": (global_R_wc, global_t_wc),
        "scene_global_20deg": (global_down20_R_wc, global_down20_t_wc)
    }
    if draw_camera_frames:
        # todo["scene_camera"] = (camera_R_wc, camera_t_wc)
        todo["scene_camera_20deg"] = (camera_down20_R_wc, camera_down20_t_wc)
    for name, Rt in todo.items():
        viz_cam_R_wc, viz_cam_t_wc = Rt
        viz_cam_R_cw = viz_cam_R_wc.transpose(1, 0)
        viz_cam_t_cw = -viz_cam_R_cw @ viz_cam_t_wc
        viz_mu = torch.einsum("ij,nj->ni", viz_cam_R_cw, mu_w) + viz_cam_t_cw[None]
        viz_fr = torch.einsum("ij,njk->nik", viz_cam_R_cw, fr_w)

        pf = viz_f / 2 * min(H, W)
        render_dict = render_cam_pcl( # 544147
            viz_mu, viz_fr, s, o, sph, semantic_feature, H=H, W=W, fx=pf, bg_color=bg_color
        )
        #print("Huis question 3:", semantic_feature.shape)
        # feature_map = render_dict["feature_map"]
        #print("rgb",render_dict["rgb"].shape)
        rgb = render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
        ret[name] = rgb
        if return_full:
            ret_full[name] = render_dict
        if save_name is not None:
            base_name = osp.basename(save_name)
            dir_name = osp.dirname(save_name)
            os.makedirs(dir_name, exist_ok=True)
            save_img = np.clip(ret[name] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(osp.join(dir_name, f"{name}_{base_name}.jpg"), save_img)
    if return_full:
        return ret, ret_full
    return ret
