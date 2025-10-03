import torch, numpy as np
import os, sys, os.path as osp

from lib_4d.autoencoder.model import Feature_heads

sys.path.append(osp.dirname(osp.abspath(__file__)))

from prior2d import Prior2D

# from gs_ed_model import EmbedDynGaussian
from camera import SimpleFovCamerasDelta
from ssim_helper import ssim
from index_helper import (
    get_valid_flow_pixel_int_index,
    query_image_buffer_by_pix_int_coord,
    uv_to_pix_int_coordinates,
    round_int_coordinates,
)
from projection_helper import backproject, project
import time
import logging
import torch.nn.functional as F

from autoencoder.model import Autoencoder

def compute_rgb_loss(
    prior2d: Prior2D,
    ind: int,
    render_dict: dict,
    sup_mask: torch.Tensor,
    ssim_lambda=0.1,
):
    pred_rgb = render_dict["rgb"].permute(1, 2, 0)
    gt_rgb = prior2d.get_rgb(ind)
    sup_mask = sup_mask.float()
    rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * sup_mask[..., None]
    rgb_loss = rgb_loss_i.sum() / sup_mask.sum()
    if ssim_lambda > 0:
        ssim_loss = 1.0 - ssim(
            (render_dict["rgb"] * sup_mask[None, ...])[None], # 1, 3, 480, 480
            (prior2d.get_rgb(ind).permute(2, 0, 1) * sup_mask[None, ...])[None], # 1, 3, 480, 480
        )
        rgb_loss = rgb_loss + ssim_loss * ssim_lambda
    return rgb_loss, rgb_loss_i, pred_rgb, gt_rgb

def compute_semantic_feature_loss(
    prior2d: Prior2D,
    ind: int,
    render_dict: dict,
    sup_mask: torch.Tensor,
    ssim_lambda=0.1,
    semantic_heads:Feature_heads=None,
):
    pred_semantic_feature = render_dict["feature_map"] # render_dim(256) x 480 x 480
    print("rendered feature map", pred_semantic_feature.shape)
    losses = {}
    gt_semantic_feature = prior2d.get_semantic_feature(ind)
    for key in gt_semantic_feature:
        if key == "internvideo":
            resize_mode = "area"
        else:
            resize_mode = "bilinear"
        gt_semantic_feature_sub = gt_semantic_feature[key].permute(2, 0, 1) # channel x H x W
        pred_semantic_feature_sub = F.interpolate(pred_semantic_feature[None, ...], size=(gt_semantic_feature_sub.shape[1], gt_semantic_feature_sub.shape[2]), mode=resize_mode)[0] # render_dim(256) x 64 x 64

        sup_mask_sub = F.interpolate(sup_mask[None, None, ...].float(), size=(gt_semantic_feature_sub.shape[1], gt_semantic_feature_sub.shape[2]), mode=resize_mode)[0][0]

        if semantic_heads is not None:
            pred_semantic_feature_sub = pred_semantic_feature_sub.permute(1, 2, 0)
            pred_semantic_feature_sub = semantic_heads.decode(key, pred_semantic_feature_sub)
            pred_semantic_feature_sub = pred_semantic_feature_sub.permute(2, 0, 1)
        print("decoded pred_semantic_feature", pred_semantic_feature_sub.shape)
        semantic_feature_loss_i = torch.abs(pred_semantic_feature_sub - gt_semantic_feature_sub) * sup_mask_sub[None, ...]
        semantic_feature_loss = semantic_feature_loss_i.sum() / sup_mask_sub.sum() / pred_semantic_feature_sub.shape[0]
        if ssim_lambda > 0:
            ssim_loss = 1.0 - ssim(
                (pred_semantic_feature_sub * sup_mask_sub[None, ...])[None], # 1, 1408, 16, 16
                (gt_semantic_feature_sub * sup_mask_sub[None, ...])[None], # 1, 1408, 16, 16
                window_size=max((gt_semantic_feature_sub.shape[1] // 40),2),
            )
            semantic_feature_loss = semantic_feature_loss + ssim_loss * ssim_lambda
        losses[key] = semantic_feature_loss

    all_losses = torch.stack([losses[key] for key in losses]).mean()
    return all_losses, losses

def compute_dep_loss(
    prior2d: Prior2D,
    ind: int,
    render_dict: dict,
    sup_mask: torch.Tensor,
    st_invariant=True,
    gt_depth=None,
):
    if gt_depth is None:
        prior_dep = prior2d.get_depth(ind)
    else:
        prior_dep = gt_depth
    # pred_dep = render_dict["dep"][0] / torch.clamp(render_dict["alpha"][0], min=1e-6)
    # ! warning, gof does not need divide alpha
    pred_dep = render_dict["dep"][0]
    if st_invariant:
        prior_t = torch.median(prior_dep[sup_mask > 0.5])
        pred_t = torch.median(pred_dep[sup_mask > 0.5])
        prior_s = (prior_dep[sup_mask > 0.5] - prior_t).abs().mean()
        pred_s = (pred_dep[sup_mask > 0.5] - pred_t).abs().mean()
        prior_dep_norm = (prior_dep - prior_t) / prior_s
        pred_dep_norm = (pred_dep - pred_t) / pred_s
    else:
        prior_dep_norm = prior_dep
        pred_dep_norm = pred_dep
    sup_mask = sup_mask.float()
    loss_dep_i = torch.abs(pred_dep_norm - prior_dep_norm) * sup_mask
    loss_dep = loss_dep_i.sum() / sup_mask.sum()
    return loss_dep, loss_dep_i, pred_dep, prior_dep


def compute_normal_loss(
    prior2d: Prior2D, ind: int, render_dict: dict, sup_mask: torch.Tensor
):
    # ! below two normals are all in camera frame, pointing towards camera
    gt_normal = prior2d.get_normal(ind)  # H,W,3
    pred_normal = render_dict["normal"].permute(1, 2, 0)
    loss, error = __normal_loss__(gt_normal, pred_normal, sup_mask)
    return loss, error, pred_normal, gt_normal


def __normal_loss__(gt_normal, pred_normal, sup_mask):
    valid_gt_mask = gt_normal.norm(dim=-1) > 1e-6
    sup_mask = sup_mask * valid_gt_mask
    normal_error1 = 1 - (pred_normal * gt_normal).sum(-1)
    normal_error2 = 1 + (pred_normal * gt_normal).sum(-1)
    error = torch.min(normal_error1, normal_error2)
    loss = (error * sup_mask).sum() / sup_mask.sum()
    return loss, error


def compute_dep_reg_loss(prior2d: Prior2D, ind: int, render_dict):
    # ! for now, the reg loss has nothing to do with mask .etc, everything is intrinsic
    gt_image = prior2d.get_rgb(ind).permute(2, 0, 1)
    distortion_map = render_dict["distortion_map"][0]
    distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
    distortion_loss = distortion_map.mean()
    return distortion_loss, distortion_map


def compute_normal_reg_loss(prior2d: Prior2D, cams: SimpleFovCamerasDelta, render_dict):
    # ! for now, the reg loss has nothing to do with mask .etc, everything is intrinsic
    dep = render_dict["dep"][0]
    dep_mask = dep > 0
    pts = cams.backproject(prior2d.homo_map[dep_mask], dep[dep_mask])
    v_map = torch.zeros_like(render_dict["rgb"]).permute(1, 2, 0)
    v_map[dep_mask] = v_map[dep_mask] + pts

    dep_normal = torch.zeros_like(v_map)
    dx = torch.cat([v_map[2:, 1:-1] - v_map[:-2, 1:-1]], dim=0)
    dy = torch.cat([v_map[1:-1, 2:] - v_map[1:-1, :-2]], dim=1)
    dep_normal[1:-1, 1:-1, :] = torch.nn.functional.normalize(
        torch.cross(dx, dy, dim=-1), dim=-1
    )
    pred_normal = render_dict["normal"].permute(1, 2, 0)
    loss, error = __normal_loss__(dep_normal, pred_normal, dep_mask)
    return loss, error, pred_normal, dep_normal


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0
    )
    grad_img_right = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0
    )
    grad_img_top = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0
    )
    grad_img_bottom = torch.mean(
        torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0
    )
    max_grad = torch.max(
        torch.stack(
            [grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1
        ),
        dim=-1,
    )[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
