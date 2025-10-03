# Single File
from matplotlib import pyplot as plt
from copy import deepcopy
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import os, sys, os.path as osp
from typing import Union
import open3d as o3d
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
import time
from matplotlib import cm
from pytorch3d.ops import knn_points
from torch import nn
from scipy.interpolate import pchip_interpolate

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
# from graph_utils import *
import torch_geometric.nn.pool as pyg_pool
import kornia

from prior2d import Prior2D

##################################################################
from fov_helper import fov_graph_init
from save_helpers import save_gauspl_ply
from index_helper import (
    query_image_buffer_by_pix_int_coord,
    uv_to_pix_int_coordinates,
    round_int_coordinates,
)

# from procruste_helper import compute_so3_from_knn_flow
from projection_helper import (
    project,
    backproject,
    fovdeg2focal,
)
from lib_4d_misc import *
from camera import SimpleFovCamerasDelta, SimpleFovCamerasIndependent
from view_sampler import BiasViewSampler
from cfg_helpers import OptimCFG, GSControlCFG
from lib_prior.tracking.cotracker_wrapper import Visualizer

# load NUM_SEMANTIC_CHANNELS
import yaml


def get_world_points(homo_list, dep_list, cams, cam_t_list=None):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = backproject(
        homo_list.reshape(-1, 2),
        dep_list.reshape(-1),
        cams,
    )
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]
    return point_world


def prepare_track_buffers(prior2d, track, track_mask, t_list):
    # track: T,N,2, track_mask: T,N
    normal_list, homo_list, ori_dep_list, rgb_list = [], [], [], []
    # semantic_feature_list=[]
    normal_flag = hasattr(prior2d, "_normals")
    for ind, tid in enumerate(t_list):
        _uv = track[ind]
        _int_uv, _inside_mask = round_int_coordinates(_uv, prior2d.H, prior2d.W)
        _dep = query_image_buffer_by_pix_int_coord(prior2d.depths[tid], _int_uv)
        _homo = query_image_buffer_by_pix_int_coord(prior2d.homo_map, _int_uv)
        if normal_flag:
            _normal = query_image_buffer_by_pix_int_coord(prior2d.normals[tid], _int_uv)
            normal_list.append(F.normalize(_normal, dim=-1))
        ori_dep_list.append(_dep)
        homo_list.append(_homo)
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(prior2d.rgbs[tid], _int_uv)
        rgb_list.append(_rgb)
        # _semantic_feature = query_image_buffer_by_pix_int_coord(prior2d.semantic_features[tid], _int_uv) # TODO: how to map semantic feature to uv?
        # _semantic_feature = prior2d.semantic_features[tid].cpu()
        # _semantic_feature = torch.zeros([len(_rgb), NUM_SEMANTIC_CHANNELS])
        # semantic_feature_list.append(_semantic_feature)
    rgb_list = torch.stack(rgb_list, 0)
    # semantic_feature_list = torch.stack(semantic_feature_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    if normal_flag:
        normal_list = torch.stack(normal_list, 0)
        normal_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, normal_list, rgb_list#, semantic_feature_list


@staticmethod
def query_2d_masks(prior2d: Prior2D, uv, start_tid=None, end_tid=None, mask_type="dep"):
    assert uv.ndim == 3 and uv.shape[-1] == 2
    start_tid = 0 if start_tid is None else start_tid
    end_tid = prior2d.T - 1 if end_tid is None else end_tid
    ret = []
    for tid in range(start_tid, end_tid + 1):
        _uv = uv[tid]
        # valid in side image
        _int_uv, _uv_inside_mask = round_int_coordinates(_uv, prior2d.H, prior2d.W)
        # valid in 2D mask
        _valid_2d_mask = query_image_buffer_by_pix_int_coord(
            prior2d.get_mask_by_key(mask_type, tid), _int_uv
        )
        ret.append(_valid_2d_mask)
    ret = torch.stack(ret, 0)
    return ret


def detect_sharp_changes_in_curve(track_mask, curve, max_vel_th, valid_type="and"):
    assert len(track_mask) >= 2, "too short!"
    diff = (curve[:-1] - curve[1:]).norm(dim=-1)  # T-1,N
    if valid_type == "and":
        valid = track_mask[:-1] * track_mask[1:]  # both side are valid
    elif valid_type == "or":
        valid = (track_mask[:-1] + track_mask[1:]) > 0  # one is valid
    to_next_diff = torch.cat([diff, torch.zeros_like(diff[:1])], 0)
    to_prev_diff = torch.cat([torch.zeros_like(diff[:1]), diff], 0)
    to_next_valid = torch.cat([valid, torch.ones_like(valid[:1])], 0)
    to_prev_valid = torch.cat([torch.ones_like(valid[:1]), valid], 0)
    # only when two points are all valid, the vel compute there is meaningful, other wise mask the diff to zero so it won't exceed the th
    to_next_diff = to_next_diff * to_next_valid.float()
    to_prev_diff = to_prev_diff * to_prev_valid.float()
    max_diff = torch.max(to_next_diff, to_prev_diff)
    invalid_mask = max_diff > max_vel_th
    # assert track_mask[invalid_mask].all()
    logging.info(
        f"curve velocity check th={max_vel_th} has {invalid_mask.sum()} ({invalid_mask.sum()/(track_mask.sum()+1e-6)*100.0:.2f}%) invalid slots"
    )
    new_track_mask = track_mask.clone()
    new_track_mask[invalid_mask] = False
    return new_track_mask, invalid_mask


@torch.no_grad()
def pchip_init(track_mask, point_ref):
    logging.info(
        "Hallucinating the missing slots with Piecewise Cubic Hermite Interpolating Polynomial"
    )
    T, N = track_mask.shape
    interpolated_xyz = []
    x = np.arange(T)
    inc_ind = np.arange(T)
    dec_ind = np.arange(T)[::-1]
    for i in tqdm(range(N)):
        m = track_mask[:, i].cpu().numpy()
        y = point_ref[:, i].cpu().numpy()

        # make sure the two end are static instead of linearly grow
        if not m[0]:
            # start with invalid, fill the front with first valid slot
            first_valid_ind = np.argmax(m * dec_ind)
            m[:first_valid_ind] = True
            y[:first_valid_ind] = y[first_valid_ind]
        if not m[-1]:
            # end with invalid, fill the end with last valid slot
            last_valid_ind = np.argmax(m * inc_ind)
            m[last_valid_ind + 1 :] = True
            y[last_valid_ind + 1 :] = y[last_valid_ind]
        interpolated_xyz.append(pchip_interpolate(x[m], y[m], x))
    interpolated_xyz = np.stack(interpolated_xyz, 1)
    return torch.from_numpy(interpolated_xyz).to(point_ref).detach().clone()


@torch.no_grad()
def __cpu_warp_position__(
    t, query_t, i, D, D_ind, track_mask, curve, K=4, sigma_factor=1.0
):
    co_mask = track_mask[query_t] * track_mask[t]
    assert co_mask.sum() > K, "no co-valid mask!"
    # find nearest neighbor in left valid step
    _D = D[query_t][i].to(curve)
    _D_ind = D_ind[query_t][i].to(curve.device)
    _D_ind_mask = co_mask[_D_ind]
    _valid_ind = _D_ind[_D_ind_mask]
    nn_ind = _valid_ind[1 : K + 1]
    # _D[~co_mask] = np.inf
    nn_dist = _D[nn_ind]
    # nn_dist, nn_ind = torch.topk(_D, K + 1, largest=False)
    # nn_dist, nn_ind = nn_dist[1:], nn_ind[1:]
    sigma = max(nn_dist[0] * sigma_factor, 1e-3)
    w = torch.exp(-(nn_dist**2) / (2.0 * sigma**2))
    w = w / torch.clamp(w.sum(), min=1e-8)
    flow = curve[t][nn_ind] - curve[query_t][nn_ind]
    xyz = curve[query_t][i] + (flow * w[:, None]).sum(0)
    dt = abs(t - query_t)
    return xyz, dt


@torch.no_grad()
def se3_segment_init_parallel(track_mask, point_ref, K=16, sigma_factor=1.0):
    # ! sigma_factor: always use the nearest nn dist as sigma
    # T,N; T,N,3
    # the filling is purely based on what is observed, and the order of filling does not affect the results

    logging.info(
        f"Neighbor rigid Warping (K={K}, sigma={sigma_factor:.3f}), naive for loop implementation for now!"
    )
    T, N = track_mask.shape
    # compute mutual distance
    D, D_ind = [], []
    for t in tqdm(range(T)):
        _D = (point_ref[t, :, None] - point_ref[t, None]).norm(dim=-1)
        D.append(_D.cpu())
        D_ind.append(_D.argsort(-1, descending=False))
    D = torch.stack(D, 0)
    D_ind = torch.stack(D_ind, 0)

    # fill
    ret = point_ref.clone()
    for t in range(T):
        # identify left and right valid time for all invalid slots at this time

        # do left and right warping for all points in parallel

        #

        _mask = track_mask[:, i].cpu()

        # todo: at least this can be boosted by handle a segment at
        if track_mask[t, i]:
            continue

        l_flag, r_flag = False, False
        if _mask[:t].any():
            l_flag = True
            # find the left and right valid time
            lt = np.max(np.where(_mask[:t] == True))
            l_xyz, l_dt = __cpu_warp_position__(
                t, lt, i, D, D_ind, d_track_mask, point_ref, K, sigma_factor
            )
        if _mask[t:].any():
            r_flag = True
            # find the left and right valid time
            rt = np.min(np.where(_mask[t:] == True))
            r_xyz, r_dt = __cpu_warp_position__(
                t, rt, i, D, D_ind, d_track_mask, point_ref, K, sigma_factor
            )
        assert l_flag or r_flag
        if l_flag and r_flag:
            # mix the two
            w = 1.0 * r_dt / (l_dt + r_dt)
            ret[t, i] = l_xyz * w + r_xyz * (1.0 - w)
        elif l_flag:
            ret[t, i] = l_xyz
        else:
            ret[t, i] = r_xyz

    # ! debug
    np.savez("../debug/ret.npz", ret=ret.cpu())
    for t in tqdm(range(T)):
        np.savetxt(
            f"../debug/{t}.xyz",
            torch.cat([ret[t], d_track_mask[t, :, None]], -1).cpu().numpy(),
            fmt="%.6f",
        )

    # every segment has left-right mixing!
    return


@torch.no_grad()
def se3_segment_init_slow(d_track_mask, point_ref, K=16, sigma_factor=1.0):
    # ! sigma_factor: always use the nearest nn dist as sigma
    # T,N; T,N,3
    # the filling is purely based on what is observed, and the order of filling does not affect the results

    logging.info(
        f"Neighbor rigid Warping (K={K}, sigma={sigma_factor:.3f}), naive for loop implementation for now!"
    )
    T, N = d_track_mask.shape
    # compute mutual distance
    D = []
    for t in tqdm(range(T)):
        _D = (point_ref[t, :, None] - point_ref[t, None]).norm(dim=-1)
        D.append(_D.cpu())
    D = torch.stack(D, 0)

    # pre-sort
    # todo: chunk cuda
    D_ind = D.argsort(-1, descending=False)

    # fill
    ret = point_ref.clone()
    for i in tqdm(range(N)):
        _mask = d_track_mask[:, i].cpu()
        for t in range(T):
            # todo: at least this can be boosted by handle a segment at
            if d_track_mask[t, i]:
                continue

            l_flag, r_flag = False, False
            if _mask[:t].any():
                l_flag = True
                # find the left and right valid time
                lt = np.max(np.where(_mask[:t] == True))
                l_xyz, l_dt = __cpu_warp_position__(
                    t, lt, i, D, D_ind, d_track_mask, point_ref, K, sigma_factor
                )
            if _mask[t:].any():
                r_flag = True
                # find the left and right valid time
                rt = np.min(np.where(_mask[t:] == True))
                r_xyz, r_dt = __cpu_warp_position__(
                    t, rt, i, D, D_ind, d_track_mask, point_ref, K, sigma_factor
                )
            assert l_flag or r_flag
            if l_flag and r_flag:
                # mix the two
                w = 1.0 * r_dt / (l_dt + r_dt)
                ret[t, i] = l_xyz * w + r_xyz * (1.0 - w)
            elif l_flag:
                ret[t, i] = l_xyz
            else:
                ret[t, i] = r_xyz

    # # ! debug
    # np.savez("../debug/ret.npz", ret=ret.cpu())
    # for t in tqdm(range(T)):
    #     np.savetxt(
    #         f"../debug/{t}.xyz",
    #         torch.cat([ret[t], d_track_mask[t, :, None]], -1).cpu().numpy(),
    #         fmt="%.6f",
    #     )

    # every segment has left-right mixing!
    return ret.to(point_ref)


@torch.no_grad()
def line_segment_init(track_mask, point_ref):
    # ! this function is a bad init, but this doesn't matter, later will directly optimize the curve
    logging.info("Naive Line Segment Init")
    point_ref = point_ref.detach().clone()
    T, N = track_mask.shape
    # point_ref # T,N,3
    # scan the T, for each empty slot, identify the right ends, and compute linear interpolation, if there is only one side, stay at the same position, if two end are empty, assert error, there shouldn't be an empty noodle!
    inverse_muti = torch.Tensor([i + 1 for i in range(T)][::-1]).to(point_ref)
    for t in tqdm(range(T)):
        to_fill_mask = ~track_mask[t]
        if not to_fill_mask.any():
            continue  # skip this time if everything is filled
        # identify the left and right nearest valid side

        if t == T - 1:  # if right end, use the previous one
            value = point_ref[t - 1, to_fill_mask].clone()
        else:
            # identify the right end, the left end must be filled in already
            to_fill_valid_curve = track_mask[t + 1 :, to_fill_mask]  # T,M
            # find the left most True slot
            to_fill_valid_curve = (
                to_fill_valid_curve.float() * inverse_muti[t + 1 :, None]
            )
            max_value, max_ind = to_fill_valid_curve.max(dim=0)
            # for no right mask case, use the left
            select_from = point_ref[t + 1 :, to_fill_mask]
            valid_right_end = torch.gather(
                select_from, 0, max_ind[None, :, None].expand(-1, -1, 3)
            )[
                0, max_value > 0
            ]  # valid when max_value > 0
            if t == 0:
                assert (
                    len(valid_right_end) == to_fill_valid_curve.shape[1]
                ), "empty noodle!"
                value = valid_right_end
            else:
                # must have a left end
                value = point_ref[t - 1, to_fill_mask].clone()
                valid_left_end = value[max_value > 0]
                delta_t = (
                    max_ind[max_value > 0] + 2
                )  # left valid, current, [0] in the max_ind
                delta_x = valid_right_end - valid_left_end
                inc = 1.0 * delta_x / delta_t[:, None]
                value[max_value > 0] = valid_left_end + inc
        point_ref[t, to_fill_mask] = value.clone()
    # np.savetxt("./debug/line_segment_init.xyz", point_ref.reshape(-1, 3).cpu().numpy())
    return point_ref.detach().clone()


def q2R(q):
    return quaternion_to_matrix(F.normalize(q, dim=-1))


def compute_knn_scale(mu, K, max_radius=0.03, min_radius=1e-6):
    # mu: N, 3
    assert mu.ndim == 2 and mu.shape[1] == 3
    dist_sq, _, _ = knn_points(mu[None], mu[None], K=K + 1)
    dist = torch.sqrt(torch.clamp(dist_sq[0, :, 1:], min=min_radius**2))
    radius = torch.mean(dist, dim=-1)
    radius = torch.clamp(radius, min=min_radius, max=max_radius)
    return radius  # N


def apply_gs_control(
    render_list,
    model,
    gs_control_cfg,
    step,
    optimizer_gs,
    first_N=None,
    last_N=None,
):
    for render_dict in render_list:
        if first_N is not None:
            assert last_N is None
            grad = render_dict["viewspace_points"].grad[:first_N]
            radii = render_dict["radii"][:first_N]
            visib = render_dict["visibility_filter"][:first_N]
        elif last_N is not None:
            assert first_N is None
            grad = render_dict["viewspace_points"].grad[-last_N:]
            radii = render_dict["radii"][-last_N:]
            visib = render_dict["visibility_filter"][-last_N:]
        else:
            grad = render_dict["viewspace_points"].grad
            radii = render_dict["radii"]
            visib = render_dict["visibility_filter"]
        model.record_xyz_grad_radii(grad, radii, visib)
    if (
        step in gs_control_cfg.densify_steps
        or step in gs_control_cfg.prune_steps
        or step in gs_control_cfg.reset_steps
    ):
        logging.info(f"GS Control at {step}")
    if step in gs_control_cfg.densify_steps:
        N_old = model.N
        model.densify(
            optimizer=optimizer_gs,
            max_grad=gs_control_cfg.densify_max_grad,
            percent_dense=gs_control_cfg.densify_percent_dense,
            extent=0.5,
            verbose=True,
        )
        logging.info(f"Densify: {N_old}->{model.N}")
    if step in gs_control_cfg.prune_steps:
        N_old = model.N
        model.prune_points(
            optimizer_gs,
            min_opacity=gs_control_cfg.prune_opacity_th,
            max_screen_size=1e10,  # disabled
        )
        logging.info(f"Prune: {N_old}->{model.N}")
    if step in gs_control_cfg.reset_steps:
        model.reset_opacity(optimizer_gs, gs_control_cfg.reset_opacity)
    return


def spatial_interpolation(src_xyz, src_buffer, query_xyz, K=16, rbf_sigma_factor=0.333):
    # src_xyz: M,3 src_buffer: M,C query_xyz: N,3
    # build RBG on each src and smoothly interpolate the buffer to query
    # first construct src_xyz nn graph
    _dist_sq_to_nn, _, _ = knn_points(src_xyz[None], src_xyz[None], K=2)
    dist_to_nn = torch.sqrt(torch.clamp(_dist_sq_to_nn[0, :, 1:], min=1e-8)).squeeze(-1)
    rbf_sigma = dist_to_nn * rbf_sigma_factor  # M
    # find the nearest K neighbors for each query point to the src
    dist_sq, ind, _ = knn_points(query_xyz[None], src_xyz[None], K=K)
    dist_sq, ind = dist_sq[0], ind[0]

    w = torch.exp(-dist_sq / (2.0 * rbf_sigma[ind]))  # N,K
    w = w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)

    value = src_buffer[ind]  # N,K,C
    ret = torch.einsum("nk, nkc->nc", w, value)
    return ret


@torch.no_grad()
def fetch_leaves_in_world_frame(
    mask_type: str,
    prior2d: Prior2D,
    cams: SimpleFovCamerasIndependent,
    n_attach: int,  # if negative use all
    save_fn=None,
    start_t=0,
    end_t=-1,
    t_list=None,
    subsample=1,
    normal_dir_ratio=10.0,
):
    logging.info(f"Fetching world leaves from {mask_type} ...")
    device = prior2d.depths.device

    if end_t == -1:
        end_t = cams.T
    if t_list is None:
        t_list = list(range(start_t, end_t))
    if subsample > 1:
        logging.info(f"2D Subsample {subsample} for fetching ...")

    mu_list, quat_list, scale_list, rgb_list, time_index_list = [], [], [], [], []
    # semantic_feature_list=[]
    for t in tqdm(t_list):
        mask2d = prior2d.get_mask_by_key(mask_type, t)
        if subsample > 1:
            mask2d[::subsample, ::subsample] = False

        dep_map = prior2d.get_depth(t)
        rgb_map = prior2d.get_rgb(t)
        # semantic_feature_map = prior2d.get_semantic_feature(t)
        nrm_map = prior2d.get_normal(t)

        cam_pcl = cams.backproject(prior2d.homo_map[mask2d], dep_map[mask2d])
        cam_nrm = nrm_map[mask2d]

        cam_R_wc, cam_t_wc = cams.Rt_wc(t)
        mu = cams.trans_pts_to_world(t, cam_pcl)
        nrm = F.normalize(torch.einsum("ij,nj->ni", cam_R_wc, cam_nrm), dim=-1)
        rgb = rgb_map[mask2d]
        # semantic_feature = semantic_feature_map[mask2d] # TODO: how to mask 16*16*1408 feature map???
        # semantic_feature = semantic_feature_map
        radius = cam_pcl[:, -1] / cams.rel_focal * prior2d.pixel_size * float(subsample)
        scale = torch.stack([radius / normal_dir_ratio, radius, radius], dim=-1)
        time_index = torch.ones_like(mu[:, 0]).long() * t

        rx = nrm.clone()
        ry = F.normalize(torch.cross(rx, mu, dim=-1), dim=-1)
        rz = F.normalize(torch.cross(rx, ry, dim=-1), dim=-1)
        rot = torch.stack([rx, ry, rz], dim=-1)
        quat = matrix_to_quaternion(rot)

        mu_list.append(mu.cpu())
        quat_list.append(quat.cpu())
        scale_list.append(scale.cpu())
        rgb_list.append(rgb.cpu())
        # semantic_feature_list.append(semantic_feature)

        time_index_list.append(time_index.cpu())

    mu_all = torch.cat(mu_list, 0)
    quat_all = torch.cat(quat_list, 0)
    scale_all = torch.cat(scale_list, 0)
    rgb_all = torch.cat(rgb_list, 0)
    # semantic_feature_all = torch.cat(semantic_feature_list, 0)

    logging.info(f"Fetching {n_attach/1000.0:.3f}K out of {len(mu_all)/1e6:.3}M pts")
    if n_attach > len(mu_all) or n_attach <= 0:
        choice = torch.arange(len(mu_all))
    else:
        choice = torch.randperm(len(mu_all))[:n_attach]

    # make gs5 param (mu, fr, s, o, sph) no rescaling
    mu_init = mu_all[choice].clone()
    q_init = quat_all[choice].clone()
    s_init = scale_all[choice].clone()
    o_init = torch.ones(len(choice), 1).to(mu_init)
    rgb_init = rgb_all[choice].clone()
    # semantic_feature_init = semantic_feature_all[choice].clone()
    NUM_SEMANTIC_CHANNELS = prior2d.latent_feature_channel
    semantic_feature_init = torch.zeros(len(choice), NUM_SEMANTIC_CHANNELS).to(mu_init) # TODO: HOW to init semantic feature? (changed 37 to 128)
    time_init = torch.cat(time_index_list, 0)[choice]
    if save_fn is not None:
        np.savetxt(
            save_fn,
            torch.cat([mu_init, rgb_init * 255], 1).detach().cpu().numpy(),
            fmt="%.6f",
        )
        # TODO: what about semantic_feature?
    torch.cuda.empty_cache()
    return (
        mu_init.to(device),
        q_init.to(device),
        s_init.to(device),
        o_init.to(device),
        rgb_init.to(device),
        semantic_feature_init.to(device),
        time_init.to(device),
    )


@torch.no_grad()
def compute_spatial_unit_from_pcl(xyz, k):
    # xyz: N,3
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    knn_dist_sq, _, _ = knn_points(xyz[None], xyz[None], K=k + 1)
    k_dist = torch.sqrt(torch.clamp(knn_dist_sq[0, :, -1], min=1e-8))
    unit = k_dist.median()
    return unit.item()


@torch.no_grad()
def compute_spatial_unit_from_nndist(dist, th=0.75, left_th=None):
    spatial_unit = dist.sort().values
    mask = spatial_unit < torch.quantile(spatial_unit, th)
    if left_th is not None:
        mask = mask & (spatial_unit > torch.quantile(spatial_unit, left_th))
    spatial_unit = spatial_unit[mask].mean()
    return spatial_unit


if __name__ == "__main__":
    dbg = np.load("../debug/to_complete.npz")
    d_track_mask = torch.from_numpy(dbg["d_track_mask"]).bool().cuda()
    xyz = torch.from_numpy(dbg["xyz"]).float().cuda()

    se3_segment_init_slow(d_track_mask, xyz)

    print()
