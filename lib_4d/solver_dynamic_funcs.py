# Solver functions for dynamic scaffold
from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
import kornia
from prior2d import Prior2D
from render_helper import render
from matplotlib import cm
from save_helpers import save_gauspl_ply
from index_helper import (
    query_image_buffer_by_pix_int_coord,
    round_int_coordinates,
)
import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import torch.nn.functional as F
from torch import nn
from pytorch3d.ops import knn_points

##################################################################
from solver_viz_helper import (
    viz3d_total_video,
    # viz2d_total_video,
    make_video_from_pattern,
    viz2d_one_frame,
    viz_hist,
    viz_dyn_hist,
    viz_curve,
    viz_plt_missing_slot,
    make_viz_np,
    viz_mv_model_frame,
    viz_scf_frame,
    viz_sigma_hist,
)
from lib_4d_misc import *
from scf4d_model import Scaffold4D

from camera import SimpleFovCamerasDelta, SimpleFovCamerasIndependent
from projection_helper import backproject, project
import wandb

def __compute_physical_losses__(
    scf: Scaffold4D,
    temporal_diff_shift: list,
    temporal_diff_weight: list,
    max_time_window: int,
    reduce="sum",
):
    if scf.T > max_time_window:
        start = torch.randint(0, scf.T - max_time_window + 1, (1,)).item()
        sup_tids = torch.arange(start, start + max_time_window)
    else:
        sup_tids = torch.arange(scf.T)
    sup_tids = sup_tids.to(scf.device)

    # * compute losses from the scaffold
    loss_coord, loss_len = scf.compute_arap_loss(
        tids=sup_tids,
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        reduce_type=reduce,
    )
    _, _, loss_p_acc, loss_q_acc = scf.compute_vel_acc_loss(
        tids=sup_tids, reduce_type=reduce
    )
    return loss_coord, loss_len, loss_p_acc, loss_q_acc


def __outlier_removal_o3d__(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    _, inlier_ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_mask = torch.zeros_like(xyz[:, 0]).bool()
    inlier_mask[inlier_ind] = True
    return inlier_mask


@torch.no_grad()
def __get_bake_data__(bake_mask, bake_xyz_list, bake_nrm_list, n_flow_pair):
    device = bake_mask.device
    T = len(bake_mask)
    src_tid = torch.randint(0, T, (n_flow_pair,)).to(device)
    dst_tid = torch.randint(0, T, (n_flow_pair,)).to(device)
    co_visible_mask = bake_mask[src_tid] * bake_mask[dst_tid]  # T,N
    src_t = src_tid[:, None].expand(-1, co_visible_mask.shape[1])  # T,N
    dst_t = dst_tid[:, None].expand(-1, co_visible_mask.shape[1])
    src_t = src_t[co_visible_mask].long()
    dst_t = dst_t[co_visible_mask].long()
    src_xyz = bake_xyz_list[src_tid][co_visible_mask]  # T,N,3
    dst_xyz = bake_xyz_list[dst_tid][co_visible_mask]  # T,N,3
    src_nrm = bake_nrm_list[src_tid][co_visible_mask]  # T,N,3
    dst_nrm = bake_nrm_list[dst_tid][co_visible_mask]
    return src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm


def __compute_bake_losses__(
    scf: Scaffold4D,
    src_t,
    src_xyz,
    src_nrm,
    dst_t,
    dst_xyz,
    dst_nrm,
    reduce="sum",
):
    attach_node_ind = scf.identify_nearest_node_id(src_xyz, src_t)
    if scf.gs_sk_approx_flag:
        # compute the skinning consistency as well
        dst_xyz_pred, dst_nrm_pred, sk_w_src = scf.warp(
            attach_node_ind=attach_node_ind,
            query_xyz=src_xyz.detach(),
            query_dir=src_nrm[..., None].detach(),
            query_tid=src_t,
            target_tid=dst_t,
            return_sk_w=True,
        )
        _, sk_w_dst, _, _, _ = scf.get_skinning_weights(
            query_xyz=dst_xyz_pred,
            query_t=dst_t,
            attach_ind=attach_node_ind,  # ! must use the same attach id!
        )
        loss_sk_w_consist = (sk_w_src - sk_w_dst).norm(dim=-1)
        metric_sk_w_consist = loss_sk_w_consist.mean().item()
        if reduce == "sum":
            loss_sk_w_consist = loss_sk_w_consist.sum()
        elif reduce == "mean":
            loss_sk_w_consist = loss_sk_w_consist.mean()
        else:
            raise ValueError(f"Unknown reduce type {reduce}")
    else:
        dst_xyz_pred, dst_nrm_pred = scf.warp(
            attach_node_ind=attach_node_ind,
            query_xyz=src_xyz.detach(),
            query_dir=src_nrm[..., None].detach(),
            query_tid=src_t,
            target_tid=dst_t,
        )
        loss_sk_w_consist = torch.tensor(0.0).to(dst_xyz.device)
        metric_sk_w_consist = 0.0
    dst_nrm_pred = dst_nrm_pred.squeeze(-1)
    loss_flow_xyz_i = (dst_xyz_pred - dst_xyz.detach()).norm(dim=-1)
    nrm_inner = (dst_nrm_pred * dst_nrm.detach()).sum(-1)
    loss_flow_nrm_i = 1.0 - nrm_inner

    if reduce == "sum":
        loss_flow_xyz = loss_flow_xyz_i.sum()
        loss_flow_nrm = loss_flow_nrm_i.sum()
    elif reduce == "mean":
        loss_flow_xyz = loss_flow_xyz_i.mean()
        loss_flow_nrm = loss_flow_nrm_i.mean()
    else:
        raise ValueError(f"Unknown reduce type {reduce}")

    metric_flow_error = loss_flow_xyz_i.mean().item()
    metric_normal_angle = (
        torch.arccos(torch.clamp(nrm_inner, -1.0, 1.0)).mean().item() / np.pi * 180.0
    )

    return (
        loss_flow_xyz,
        loss_flow_nrm,
        loss_sk_w_consist,
        metric_flow_error,
        metric_normal_angle,
        metric_sk_w_consist,
    )


@torch.no_grad()
def __compute_R_from_xyz__(scf: Scaffold4D) -> Scaffold4D:
    init_node_rx = F.normalize(scf._curve_normal_init[0], dim=-1)
    init_node_ry = F.normalize(
        torch.cross(init_node_rx, scf._node_xyz[0], dim=-1), dim=-1
    )
    init_node_rz = F.normalize(torch.cross(init_node_rx, init_node_ry, dim=-1), dim=-1)
    init_node_R = torch.stack([init_node_rx, init_node_ry, init_node_rz], dim=-1)
    new_R_list = [init_node_R]
    nn_ind, nn_mask = scf.topo_knn_ind, scf.topo_knn_mask
    for new_tid in tqdm(range(1, scf.T)):

        src_xyz, dst_xyz = scf._node_xyz[new_tid - 1], scf._node_xyz[new_tid]
        src_p = src_xyz[nn_ind] - src_xyz[:, None]
        dst_q = dst_xyz[nn_ind] - dst_xyz[:, None]
        # dst_q = R_optimal @ src_p
        src_p_len = src_p.norm(dim=-1)
        dst_q_len = dst_q.norm(dim=-1)
        # ! use the distance to weight the importance!
        w = torch.exp(-0.5 * (torch.max(dst_q_len, src_p_len) / scf.spatial_unit) ** 2)
        w = w * nn_mask
        w = w / w.sum(dim=-1, keepdim=True)  # M,K
        W = torch.einsum("nki,nkj->nkij", src_p, dst_q)
        W = W.sum(1)

        U, s, V = torch.svd(
            W.double()
        )  # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
        U, s, V = U.float(), s.float(), V.float()
        # R_star = V @ (U.T) # ! handling flipping
        R_tmp = torch.einsum("nij,nkj->nik", V, U)
        det = torch.det(R_tmp)
        dia = torch.ones(len(det), 3).to(det)
        dia[:, -1] = det
        Sigma = torch.diag_embed(dia)
        V = torch.einsum("nij,njk->nik", V, Sigma)
        R_star = torch.einsum("nij,nkj->nik", V, U)
        # dst = R_star @ src
        next_R = torch.einsum("nij,njk->nik", R_star, new_R_list[-1])
        new_R_list.append(next_R)
    new_R_list = torch.stack(new_R_list, dim=0)
    new_q_list = matrix_to_quaternion(new_R_list)
    scf._node_rotation = nn.Parameter(new_q_list)
    scf.to(scf.device)
    return scf


@torch.no_grad()
def __uniform_subsample_vtx__(xyz, margin):
    D = torch.cdist(xyz, xyz)  # N,N
    # iteratively include xyz to a return set, the order of the xyz is the rank, and include if the distance to selected set is larger than the margin
    selected_id = []
    for ind in range(len(xyz)):
        if len(selected_id) == 0:
            selected_id.append(ind)
            continue
        if (D[ind, selected_id] < margin).any():
            continue
        selected_id.append(ind)
    selected_id = torch.tensor(selected_id).long().to(xyz.device)
    return selected_id


@torch.no_grad()
def __semantic_match__(
    dep_feat,  # include rgb
    dep_xyz,
    node_feat,
    node_std,
    node_xyz,
    node_ori_id,
    scf: Scaffold4D,
    K=4,
    matching_method="sem",
    spatial_radius_ratio=3.0,
    rgb_std_ratio=3.0,
    feat_std_ratio=3.0,
    verbose=False,
    block_size=2048 * 2048,
):
    N, M = len(dep_feat), len(node_feat)
    # * support chunk operations to reduce memory, N sometimes can be very large

    # ! safely compute below distance
    n_chunk = max(1, block_size // M)
    cur = 0
    xyz_D, feat_D, rgb_D = [], [], []
    while cur < N:
        xyz_D.append((dep_xyz[cur : cur + n_chunk, None] - node_xyz[None]).norm(dim=-1))
        if matching_method == "sem":
            feat_D.append(
                (dep_feat[cur : cur + n_chunk, None, 3:] - node_feat[None, :, 3:]).norm(
                    dim=-1
                )
            )
            rgb_D.append(
                (dep_feat[cur : cur + n_chunk, None, :3] - node_feat[None, :, :3]).norm(
                    dim=-1
                )
            )
        cur += n_chunk
    xyz_D = torch.cat(xyz_D, 0)
    if matching_method == "sem":
        feat_D = torch.cat(feat_D, 0)
        rgb_D = torch.cat(rgb_D, 0)

    xyz_valid_mask = xyz_D < scf.spatial_unit * spatial_radius_ratio
    match_valid_mask = xyz_valid_mask
    if matching_method == "sem":
        rgb_valid_mask = rgb_D < rgb_std_ratio * node_std[None, :, :3].sum(-1)
        feat_valid_mask = feat_D < feat_std_ratio * node_std[None, :, 3:].sum(-1)
        match_valid_mask = match_valid_mask & rgb_valid_mask & feat_valid_mask
        # use the joint mask to constrain !!
        feat_D[~match_valid_mask] = 1e10
        rgb_D[~match_valid_mask] = 1e10

    xyz_D[~match_valid_mask] = 1e10
    matched_mask = match_valid_mask.any(1)

    if matching_method == "sem":
        D = feat_D[matched_mask]
    elif matching_method == "spa":
        D = xyz_D[matched_mask]
    else:
        raise ValueError(f"Unknown matching method {matching_method}")

    match_value, _match_node_id = D.topk(K, dim=1, largest=False)
    final_mask = match_value < 1e10

    matched_dep_xyz = dep_xyz[matched_mask][:, None, :].expand(-1, K, -1)
    matched_dep_xyz = matched_dep_xyz[final_mask].detach().cpu()
    matched_node_id = node_ori_id[_match_node_id[final_mask]].detach().cpu()
    match_value = match_value[final_mask]

    matching_weight = match_value / (
        node_std.sum(-1)[_match_node_id[final_mask]] + 1e-6
    )
    matching_weight = torch.exp(-(matching_weight**2)).detach().cpu()
    match_value = match_value.detach().cpu()
    if verbose:
        logging.info(
            f"Sem Match: xyz-radius-th={(scf.spatial_unit * spatial_radius_ratio).item():.3f} valid={xyz_valid_mask.float().mean()*100.0:.2f}%"
        )
        logging.info(
            f"Sem Match: rgb-std-th={rgb_std_ratio} valid={rgb_valid_mask.float().mean()*100.0:.2f}%"
        )
        logging.info(
            f"Sem Match: feat-std-th={feat_std_ratio} valid={feat_valid_mask.float().mean()*100.0:.2f}%"
        )
        logging.info(
            f"Sem Match: [{matched_mask.float().mean()*100.0:.2f}%] uncovered leaves matched to free nodes"
        )
    return matched_dep_xyz, matched_node_id, matching_weight, match_value


@torch.no_grad()
def __build_semantic_drag__(
    prior2d: Prior2D,
    scf: Scaffold4D,
    cams: SimpleFovCamerasIndependent,
    K=4,
    node_coverage_K=-1,
    matching_method="sem",
    spatial_radius_ratio=3.0,
    rgb_std_ratio=3.0,
    feat_std_ratio=3.0,
    viz_dir=None,
    verbose=True,
    max_drag_dep_num=4096,
):
    # * a node is free if does not have node_coverage_K near leaves
    # * a leaf is un-covered if it's not in the support range of K nodes
    target_xyz_list, node_id_list = [], []
    scf_range_association_time_list, match_weight_list = [], []
    matched_ratio_list = []
    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    for st, pt in tqdm(enumerate(scf._t_list)):
        # * First see how the depth and scf aligned, identify un-covered depth pts
        fg_mask = prior2d.get_dynamic_mask(pt) * prior2d.get_depth_mask(pt)
        xyz_cam = backproject(
            prior2d.homo_map[fg_mask], prior2d.get_depth(pt)[fg_mask], cams
        )
        xyz_world = cams.trans_pts_to_world(pt, xyz_cam)
        xyz_scf_node = scf._node_xyz[st]

        # ! this th is decided by the spatial unit, when use small unit, a lot more points are identified as not covered!!
        coverage_mask = scf.check_points_coverage(xyz_world, st, scf.spatial_unit, K=K)
        if verbose:
            logging.info(
                f"{coverage_mask.float().mean()*100.0:.2f}% covered by current scf"
            )
        # * the fg point that are un-covered, are the regions that co-tracker failed
        leaf_uncovered_mask = ~coverage_mask  # further do outlier removal with open3d
        inlier_mask = __outlier_removal_o3d__(xyz_world[leaf_uncovered_mask])
        leaf_uncovered_mask[leaf_uncovered_mask.clone()] = inlier_mask
        if leaf_uncovered_mask.sum() > max_drag_dep_num:
            old_N = leaf_uncovered_mask.sum()
            choice = torch.randperm(old_N)[:max_drag_dep_num]
            sub_mask = torch.zeros(old_N).bool().to(leaf_uncovered_mask)
            sub_mask[choice] = True
            leaf_uncovered_mask[leaf_uncovered_mask.clone()] = sub_mask

        # check the un-used nodes, the dof we can re-use
        node_free_mask = torch.ones_like(xyz_scf_node[:, 0]).bool()

        if node_coverage_K != 0:  # if use all node, set node_coverage_K to zero
            # only match to un-used node!
            # ! remove the hard unset of the solid mask below
            # node_free_mask[:M_init] = ~init_scf_mask[t]  # the free node should not be solid
            # * the "un-used" also means not supporting leaves, otherwise not good idea to move it
            if node_coverage_K < 0:
                node_coverage_K = K
            sq_dst_to_leaf, _, _ = knn_points(
                xyz_scf_node[None], xyz_world[None], K=node_coverage_K
            )
            sq_dst_to_leaf = sq_dst_to_leaf[0]
            node_used_mask = (sq_dst_to_leaf < (scf.spatial_unit**2)).all(-1)
            node_free_mask = node_free_mask & (~node_used_mask)

        if viz_dir is not None:
            np.savetxt(
                osp.join(viz_dir, f"bake_node_{pt}.xyz"),
                torch.cat([xyz_scf_node, node_free_mask[:, None]], -1)
                .detach()
                .cpu()
                .numpy(),
                fmt="%.6f",
            )
            np.savetxt(
                osp.join(viz_dir, f"bake_dep_{pt}.xyz"),
                torch.cat([xyz_world, coverage_mask[:, None]], 1)
                .detach()
                .cpu()
                .numpy(),
                fmt="%.6f",
            )

        # * match depth pixels onto the scf with curve feature
        # for each un-covered leaf, find the semantically matched un-used node
        uncovered_int_uv = prior2d.pixel_int_map[fg_mask][leaf_uncovered_mask]
        if len(uncovered_int_uv) == 0:
            continue
        uncovered_feat = prior2d.query_low_res_semantic_feat(uncovered_int_uv, pt)
        uncovered_rgb = query_image_buffer_by_pix_int_coord(
            prior2d.get_rgb(pt), uncovered_int_uv
        )
        uncovered_feat = torch.cat([uncovered_rgb, uncovered_feat], -1)  # N,C
        uncovered_xyz = xyz_world[leaf_uncovered_mask]  # N,3

        free_node_id = torch.arange(scf.M).to(scf.device)[node_free_mask]
        free_node_feat = scf.semantic_feature_mean[node_free_mask]  # M,C
        free_node_std = torch.sqrt(
            torch.clamp(scf.semantic_feature_var[node_free_mask], min=0.0)
        )  # M,C
        free_node_xyz = xyz_scf_node[node_free_mask]  # M,3

        # * semantic match
        target_xyz, matched_node_id, weight, _ = __semantic_match__(
            uncovered_feat,
            uncovered_xyz,
            free_node_feat,
            free_node_std,
            free_node_xyz,
            free_node_id,
            scf,
            K=K,
            matching_method=matching_method,
            spatial_radius_ratio=spatial_radius_ratio,
            rgb_std_ratio=rgb_std_ratio,
            feat_std_ratio=feat_std_ratio,
            verbose=False,
        )

        target_xyz_list.append(target_xyz)
        node_id_list.append(matched_node_id)
        scf_range_association_time_list.append(torch.ones_like(matched_node_id) * st)
        match_weight_list.append(weight)
        # torch.cuda.empty_cache()

    target_xyz_list = torch.cat(target_xyz_list, 0)
    node_id_list = torch.cat(node_id_list, 0)
    scf_range_association_time_list = torch.cat(scf_range_association_time_list, 0)
    match_weight_list = torch.cat(match_weight_list, 0)
    logging.info(
        f"Build [{len(scf_range_association_time_list)/1000.0:.2f}K] semantic matched drag constraints"
    )
    return (
        scf_range_association_time_list.cuda(),  # ! this is in the scf ind, not the p2d index, which is shorter
        node_id_list.cuda(),
        target_xyz_list.cuda(),
        match_weight_list.cuda(),
        matched_ratio_list,
    )


@torch.no_grad()
def __prepare_all_flow_paris__(
    scf_t_list,
    p2d: Prior2D,
    cams: SimpleFovCamerasIndependent,
    device=torch.device("cuda"),
):
    logging.info("Prepare flow baking before loop starts...")
    ret = []
    if not isinstance(scf_t_list, list):
        scf_t_list = scf_t_list.detach().cpu().numpy().tolist()
    # filter the flow by scf._t_list and convert the time ind to the scf ind range
    for flow_id in tqdm(range(len(p2d.flow_ij_list))):
        p_src_t, p_dst_t = p2d.flow_ij_list[flow_id]
        if p_src_t not in scf_t_list or p_dst_t not in scf_t_list:
            continue

        flow = p2d.flows[flow_id]
        flow_mask = p2d.flow_masks[flow_id]
        # flow valid, dynamic fg, and depth valid
        mask = flow_mask * p2d.get_dynamic_mask(p_src_t) * p2d.get_depth_mask(p_src_t)
        src_uv_int = p2d.pixel_int_map[mask]
        dst_uv_int = src_uv_int + flow[mask]
        dst_dep_valid = query_image_buffer_by_pix_int_coord(
            p2d.get_depth_mask(p_dst_t), dst_uv_int
        )
        src_uv_int = src_uv_int[dst_dep_valid]
        dst_uv_int = dst_uv_int[dst_dep_valid]

        src_uv = query_image_buffer_by_pix_int_coord(p2d.homo_map, src_uv_int)
        dst_uv = query_image_buffer_by_pix_int_coord(p2d.homo_map, dst_uv_int)
        src_dep = query_image_buffer_by_pix_int_coord(p2d.get_depth(p_src_t), src_uv_int)
        dst_dep = query_image_buffer_by_pix_int_coord(p2d.get_depth(p_dst_t), dst_uv_int)

        src_xyz_cam = backproject(src_uv, src_dep, cams)
        dst_xyz_cam = backproject(dst_uv, dst_dep, cams)

        src_xyz = cams.trans_pts_to_world(p_src_t, src_xyz_cam)
        dst_xyz = cams.trans_pts_to_world(p_dst_t, dst_xyz_cam)

        src_nrm_cam = query_image_buffer_by_pix_int_coord(
            p2d.get_normal(p_src_t), src_uv_int
        )
        dst_nrm_cam = query_image_buffer_by_pix_int_coord(
            p2d.get_normal(p_dst_t), dst_uv_int
        )

        src_R_wc, _ = cams.Rt_wc(p_src_t)
        src_nrm = torch.einsum("ij,nj->ni", src_R_wc, src_nrm_cam)
        dst_R_wc, _ = cams.Rt_wc(p_dst_t)
        dst_nrm = torch.einsum("ij,nj->ni", dst_R_wc, dst_nrm_cam)

        ret.append(
            {
                "src_t": scf_t_list.index(p_src_t),
                "src_xyz": src_xyz.to(device),
                "src_nrm": src_nrm.to(device),
                "dst_t": scf_t_list.index(p_dst_t),
                "dst_xyz": dst_xyz.to(device),
                "dst_nrm": dst_nrm.to(device),
            }
        )
    assert len(ret) > 0
    return ret


@torch.no_grad()
def __sample_flow_pairs__(flow_pairs_data: list, n_t: int, n_pair_per_t: int):
    tid = torch.randperm(len(flow_pairs_data))[:n_t]
    src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm = [], [], [], [], [], []
    for t in tid:
        data = flow_pairs_data[t]
        device = data["src_xyz"].device
        N = len(data["src_xyz"])
        if n_pair_per_t < N:
            choice = torch.randperm(N)[:n_pair_per_t].to(device)
        else:
            choice = torch.arange(N).to(device)
        src_xyz.append(data["src_xyz"][choice])
        src_nrm.append(data["src_nrm"][choice])
        dst_xyz.append(data["dst_xyz"][choice])
        dst_nrm.append(data["dst_nrm"][choice])
        src_t.append(torch.ones(len(choice)).long().to(device) * data["src_t"])
        dst_t.append(torch.ones(len(choice)).long().to(device) * data["dst_t"])
    src_t = torch.cat(src_t, 0)
    src_xyz = torch.cat(src_xyz, 0)
    src_nrm = torch.cat(src_nrm, 0)
    dst_t = torch.cat(dst_t, 0)
    dst_xyz = torch.cat(dst_xyz, 0)
    dst_nrm = torch.cat(dst_nrm, 0)
    return src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm


def __geo_loop__(
    viz_dir,
    log_dir,
    #
    prior2d: Prior2D,
    scf: Scaffold4D,
    cams: SimpleFovCamerasDelta,
    mlevel_resample_steps=32,
    #
    lr_q=0.1,
    lr_p=0.1,
    lr_sig=0.03,
    lr_sk_q=0.03,
    #
    total_steps=1000,
    max_time_window=200,
    # * Basic Phy losses
    temporal_diff_shift=[1],
    temporal_diff_weight=[1.0],
    lambda_local_coord=1.0,
    lambda_metric_len=0.0,
    lambda_xyz_acc=0.0,
    lambda_q_acc=0.1,
    # * stablize
    lambda_small_corr=0.0,
    hard_fix_valid=True,
    # * Bake
    n_flow_pair=0,
    bake_mask: torch.Tensor = None,
    bake_xyz_list: torch.Tensor = None,
    bake_nrm_list: torch.Tensor = None,
    lambda_flow_xyz=1.0,
    lambda_flow_nrm=1.0,
    # * sk weight consistency
    lambda_sk_w_consistency=1.0,
    ################################
    # * include depth baking
    n_depth_pair_per_t=0,
    t_depth_pair=0,
    ################################
    # * drag
    # drag_association_interval=32,
    # lambda_drag_xyz=1.0,
    drag_association_interval=0,
    lambda_drag_xyz=0.0,
    drag_spatial_radius_ratio=3.0,
    drag_rgb_std_ratio=3.0,
    drag_feat_std_ratio=3.0,
    drag_K=4,
    node_cover_K=4,
    drag_matching_method="sem",
    lambda_dense_flow_xyz=1.0,
    lambda_dense_flow_nrm=1.0,
    ################
    # topo
    update_full_topo=False,
    use_mask_topo=True,
    #
    decay_start=400,
    decay_factor=10,
    # viz
    viz_debug_interval=99,
    viz_interval=100,
    # save
    prefix="",
    save_flag=False,
):
    torch.cuda.empty_cache()

    # * The small change is resp. to the init stage
    solid_mask = scf._curve_slot_init_valid_mask
    solid_xyz = scf._node_xyz.clone().detach()
    rgb_list = scf._curve_color_init.clone().detach()
    # semantic_feature_list = scf._curve_semantic_feature_init.clone().detach()
    semantic_feature_list = torch.zeros([rgb_list.shape[0], rgb_list.shape[1], scf._node_semantic_feature.shape[-1]]).to(rgb_list.device)

    optimizer = torch.optim.Adam(
        scf.get_optimizable_list(
            lr_np=lr_p, lr_nq=lr_q, lr_nsig=lr_sig, lr_sk_q=lr_sk_q
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        (total_steps - decay_start),
        eta_min=min(lr_p, lr_q) / decay_factor,
    )

    loss_list, loss_coord_list, loss_len_list = [], [], []
    loss_small_corr_list = []
    loss_p_acc_list, loss_q_acc_list = [], []
    loss_flow_xyz_list, loss_flow_nrm_list = [], []
    metric_flow_error_list, metric_normal_angle_list = [], []

    loss_drag_xyz_list, metric_drag_xyz_list = [], []
    loss_dense_flow_xyz_list, loss_dense_flow_nrm_list = [], []
    metric_dense_flow_error_list, metric_dense_normal_angle_list = [], []

    loss_sk_w_consist_list, metric_sk_w_consist_list = [], []
    loss_dense_sk_w_consist_list, metric_dense_sk_w_consist_list = [], []

    # before start, update topo
    scf.update_topology(curve_mask=solid_mask if use_mask_topo else None)

    # prepare drag
    drag_flag = drag_association_interval > 0 and lambda_drag_xyz > 0
    if drag_flag:
        (
            association_time_list,
            node_id_list,
            target_xyz_list,
            match_weight_list,
            succ_list,
        ) = __build_semantic_drag__(
            prior2d,
            scf,
            cams,
            spatial_radius_ratio=drag_spatial_radius_ratio,
            rgb_std_ratio=drag_rgb_std_ratio,
            feat_std_ratio=drag_feat_std_ratio,
            # viz_dir=viz_dir,
            verbose=False,
            K=drag_K,
            node_coverage_K=node_cover_K,
            matching_method=drag_matching_method,
        )

    # prepare flow baking
    dense_flow_baking_flag = (
        lambda_dense_flow_xyz > 0 or lambda_dense_flow_nrm > 0
    ) and( n_depth_pair_per_t * t_depth_pair) > 0
    logging.info(f"dense flow baking flag={dense_flow_baking_flag}")
    logging.info(f"long term tracking backing flag={n_flow_pair > 0}")
    if dense_flow_baking_flag:
        flow_baking_data = __prepare_all_flow_paris__(
            scf._t_list, prior2d, cams, device=torch.device("cuda")
        )

    logging.info(f"4DSCF-Solver-loop prefix=[{{prefix}}] summary: ")
    logging.info(
        f"total_steps={total_steps}, decay_start={decay_start}, drag_flag={drag_flag}, dense_flow_baking_flag={dense_flow_baking_flag}, hard_fix_valid_flag={hard_fix_valid}"
    )
    logging.info(f"lr_p={lr_p}, lr_q={lr_q}, lr_sig={lr_sig}")

    # before start viz
    if viz_interval > 0:
        viz_dyn_hist(scf, viz_dir, f"{prefix}dyn_scf_init_000000_graph_stat")
        viz_frame = viz_curve(
            scf._node_xyz.detach(),
            rgb_list,
            semantic_feature_list,
            solid_mask,
            cams,
            viz_n=256,
            res=480,
            pts_size=0.001,
            only_viz_last_frame=True,
            no_rgb_viz=False,
            text=f"Step=0",
            time_window=cams.T,
        )
        imageio.imsave(
            osp.join(viz_dir, f"{prefix}dyn_scf_init_000000.jpg"), viz_frame[0]
        )

    for step in tqdm(range(total_steps)):
        if step % mlevel_resample_steps == 0 and step > 0:
            if update_full_topo:
                scf.update_topology(curve_mask=solid_mask if use_mask_topo else None)
            else:
                scf.update_multilevel_arap_topo()
        if drag_flag and step % drag_association_interval == 0 and step > 0:
            (
                association_time_list,
                node_id_list,
                target_xyz_list,
                match_weight_list,
                succ_list,
            ) = __build_semantic_drag__(
                prior2d,
                scf,
                cams,
                spatial_radius_ratio=drag_spatial_radius_ratio,
                rgb_std_ratio=drag_rgb_std_ratio,
                feat_std_ratio=drag_feat_std_ratio,
                # viz_dir=viz_dir,
                verbose=False,
                K=drag_K,
                node_coverage_K=node_cover_K,
                matching_method=drag_matching_method,
            )

        optimizer.zero_grad()

        loss_coord, loss_len, loss_p_acc, loss_q_acc = __compute_physical_losses__(
            scf, temporal_diff_shift, temporal_diff_weight, max_time_window
        )

        # loss of near original curve
        diff_to_solid_xyz = (scf._node_xyz - solid_xyz.detach()).norm(dim=-1)
        loss_small_corr = diff_to_solid_xyz[solid_mask].sum()  # ! use sum

        loss = (
            lambda_local_coord * loss_coord
            + lambda_metric_len * loss_len
            + lambda_xyz_acc * loss_p_acc
            + lambda_q_acc * loss_q_acc
            + lambda_small_corr * loss_small_corr
        )
        with torch.no_grad():
            loss_list.append(loss.item())
            loss_coord_list.append(loss_coord.item())
            loss_len_list.append(loss_len.item())
            loss_p_acc_list.append(loss_p_acc.item())
            loss_q_acc_list.append(loss_q_acc.item())
            loss_small_corr_list.append(loss_small_corr.item())

            wandb.log(
                {
                    f"dyn_scf_{prefix}loss": loss.item(),
                    f"dyn_scf_{prefix}loss_coord": loss_coord.item(),
                    f"dyn_scf_{prefix}loss_len": loss_len.item(),
                    f"dyn_scf_{prefix}loss_p_acc": loss_p_acc.item(),
                    f"dyn_scf_{prefix}loss_q_acc": loss_q_acc.item(),
                    f"dyn_scf_{prefix}loss_small_corr": loss_small_corr.item(),
                }

            )

        # baking loss
        if n_flow_pair > 0:
            src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm = __get_bake_data__(
                bake_mask, bake_xyz_list, bake_nrm_list, n_flow_pair
            )

            (
                loss_flow_xyz,
                loss_flow_nrm,
                loss_sk_w_consist,
                metric_flow_error,
                metric_normal_angle,
                metric_sk_w_consist,
            ) = __compute_bake_losses__(
                scf, src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm
            )

            loss = (
                loss
                + lambda_flow_xyz * loss_flow_xyz
                + lambda_flow_nrm * loss_flow_nrm
                + lambda_sk_w_consistency * loss_sk_w_consist
            )

            with torch.no_grad():
                loss_flow_xyz_list.append(loss_flow_xyz.item())
                loss_flow_nrm_list.append(loss_flow_nrm.item())
                metric_flow_error_list.append(metric_flow_error)
                metric_normal_angle_list.append(metric_normal_angle)

                loss_sk_w_consist_list.append(loss_sk_w_consist.item())
                metric_sk_w_consist_list.append(metric_sk_w_consist)

                wandb.log(
                    {
                        f"dyn_scf_{prefix}loss_flow_xyz": loss_flow_xyz.item(),
                        f"dyn_scf_{prefix}loss_flow_nrm": loss_flow_nrm.item(),
                        f"dyn_scf_{prefix}loss_sk_w_consist": loss_sk_w_consist.item(),
                        f"dyn_scf_{prefix}metric_flow_error": metric_flow_error,
                        f"dyn_scf_{prefix}metric_normal_angle": metric_normal_angle,
                        f"dyn_scf_{prefix}metric_sk_w_consist": metric_sk_w_consist,
                    }
                )


        if dense_flow_baking_flag:
            # todo: smart sample the depth flow supervision
            src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm = __sample_flow_pairs__(
                flow_baking_data, n_t=t_depth_pair, n_pair_per_t=n_depth_pair_per_t
            )
            (
                loss_dense_flow_xyz,
                loss_dense_flow_nrm,
                loss_dense_sk_w_consist,
                metric_dense_flow_error,
                metric_dense_normal_angle,
                metric_dense_sk_w_consist,
            ) = __compute_bake_losses__(
                scf, src_t, src_xyz, src_nrm, dst_t, dst_xyz, dst_nrm
            )
            loss = (
                loss
                + lambda_dense_flow_xyz * loss_dense_flow_xyz
                + lambda_dense_flow_nrm * loss_dense_flow_nrm
                + lambda_sk_w_consistency * loss_dense_sk_w_consist
            )

            with torch.no_grad():
                loss_dense_flow_xyz_list.append(loss_dense_flow_xyz.item())
                loss_dense_flow_nrm_list.append(loss_dense_flow_nrm.item())
                metric_dense_flow_error_list.append(metric_dense_flow_error)
                metric_dense_normal_angle_list.append(metric_dense_normal_angle)

                loss_dense_sk_w_consist_list.append(loss_dense_sk_w_consist.item())
                metric_dense_sk_w_consist_list.append(metric_dense_sk_w_consist)

                wandb.log(
                    {
                        f"dyn_scf_{prefix}loss_dense_flow_xyz": loss_dense_flow_xyz.item(),
                        f"dyn_scf_{prefix}loss_dense_flow_nrm": loss_dense_flow_nrm.item(),
                        f"dyn_scf_{prefix}loss_dense_sk_w_consist": loss_dense_sk_w_consist.item(),
                        f"dyn_scf_{prefix}metric_dense_flow_error": metric_dense_flow_error,
                        f"dyn_scf_{prefix}metric_dense_normal_angle": metric_dense_normal_angle,
                        f"dyn_scf_{prefix}metric_dense_sk_w_consist": metric_dense_sk_w_consist,
                    }
                )

        if drag_flag:
            drag_node_xyz, _, drag_node_semantic_feature = scf.get_async_knns(
                association_time_list, node_id_list[:, None]
            )
            drag_node_xyz = drag_node_xyz.squeeze(1)  # N,3
            # weight the drag be matching value, value lower, more stronger loss
            drag_error_i = (drag_node_xyz - target_xyz_list).norm(dim=-1)
            loss_drag = (drag_error_i * match_weight_list).sum()
            loss = loss + lambda_drag_xyz * loss_drag
            metric_drag_error = drag_error_i.mean()
            with torch.no_grad():
                loss_drag_xyz_list.append(loss_drag.item())
                metric_drag_xyz_list.append(metric_drag_error.item())

                wandb.log(
                    {
                        f"dyn_scf_{prefix}loss_drag_xyz": loss_drag.item(),
                        f"dyn_scf_{prefix}metric_drag_xyz": metric_drag_error.item(),
                    }
                )

        loss.backward()
        if hard_fix_valid:
            scf.mask_xyz_grad(~solid_mask)
        optimizer.step()

        # * control
        if step > decay_start:
            scheduler.step()

        if step % 50 == 0:
            logging.info(f"step={step}, loss={loss:.6f}")
            msg = f"[{prefix}] loss_coord={loss_coord:.6f}, loss_len={loss_len:.6f}, loss_p_acc={loss_p_acc:.6f}, loss_R_acc={loss_q_acc:.6f}, loss_small_corr={loss_small_corr:.6f}"
            if n_flow_pair > 0:
                msg += f", loss_flow_xyz={loss_flow_xyz:.6f}, loss_flow_nrm={loss_flow_nrm:.6f}, loss_sk_w_consist={loss_sk_w_consist:.6f} , metric_flow_error={metric_flow_error:.6f}, metric_normal_angle={metric_normal_angle:.6f}, metric_w_consist={metric_sk_w_consist:.6f}"
            if drag_flag:
                msg += f", loss_drag_xyz={loss_drag:.6f}, metric_drag_xyz={metric_drag_error:.6f}"
            if dense_flow_baking_flag:
                msg += f", loss_dense_flow_xyz={loss_dense_flow_xyz:.6f}, loss_dense_flow_nrm={loss_dense_flow_nrm:.6f}, loss_dense_sk_w_consist={loss_dense_sk_w_consist:.6f} metric_dense_flow_error={metric_dense_flow_error:.6f}, metric_dense_normal_angle={metric_dense_normal_angle:.6f}, metric_dense_w_consist={metric_dense_sk_w_consist:.6f}"
            logging.info(msg)

        if (step % viz_interval == 0 or step == total_steps - 1) and viz_interval > 0:
            viz_dyn_hist(scf, viz_dir, f"{prefix}dyn_scf_init_{step:06d}_graph_stat")
            viz_frame = viz_curve(
                scf._node_xyz.detach(),
                rgb_list,
                semantic_feature_list,
                solid_mask,
                cams,
                viz_n=256,
                res=480,
                pts_size=0.001,
                only_viz_last_frame=True,
                no_rgb_viz=False,
                text=f"Step={step}",
                time_window=cams.T,
            )
            imageio.imsave(
                osp.join(viz_dir, f"{prefix}dyn_scf_init_{step+1:06d}.jpg"),
                viz_frame[0],
            )
        if viz_debug_interval > 0 and step % viz_debug_interval == 0:
            dbg_save_dict = scf.export_node_edge_dict()
            dbg_save_dict["rgb"] = rgb_list.detach().cpu().numpy()
            dbg_save_dict["mask"] = solid_mask.detach().cpu().numpy()
            if drag_flag:
                dbg_save_dict["sem_drag_t"] = (
                    association_time_list.detach().cpu().numpy()
                )
                dbg_save_dict["sem_drag_node_id"] = node_id_list.detach().cpu().numpy()
                dbg_save_dict["sem_drag_dep_xyz"] = (
                    target_xyz_list.detach().cpu().numpy()
                )
                dbg_save_dict["sem_drag_w"] = match_weight_list.detach().cpu().numpy()
            np.savez(
                osp.join(viz_dir, f"DEBUG_dyn_scf_init_{step:06d}.npz"),
                **dbg_save_dict,
            )
            viz_sigma_hist(
                scf, osp.join(viz_dir, f"DEBUG_dyn_scf_sig_init_{step:06d}.jpg")
            )

            fig = plt.figure(figsize=(25, 7))
            for plt_i, plt_pack in enumerate(
                [
                    ("loss", loss_list),
                    ("loss_coord", loss_coord_list),
                    ("loss_len", loss_len_list),
                    ("loss_p_acc", loss_p_acc_list),
                    ("loss_q_acc", loss_q_acc_list),
                    ("loss_small_corr", loss_small_corr_list),
                    #
                    ("loss_flow_xyz", loss_flow_xyz_list),
                    ("loss_flow_nrm", loss_flow_nrm_list),
                    ("metric_flow_error", metric_flow_error_list),
                    ("metric_normal_angle", metric_normal_angle_list),
                    #
                    ("loss_drag_xyz", loss_drag_xyz_list),
                    ("metric_drag_xyz", metric_drag_xyz_list),
                    #
                    ("loss_dense_flow_xyz", loss_dense_flow_xyz_list),
                    ("loss_dense_flow_nrm", loss_dense_flow_nrm_list),
                    ("metric_dense_flow_error", metric_dense_flow_error_list),
                    ("metric_dense_normal_angle", metric_dense_normal_angle_list),
                    #
                    ("loss_sk_w_consist", loss_sk_w_consist_list),
                    ("metric_sk_w_consist", metric_sk_w_consist_list),
                    #
                    ("loss_dense_sk_w_consist", loss_dense_sk_w_consist_list),
                    ("metric_dense_sk_w_consist", metric_dense_sk_w_consist_list),
                ]
            ):
                plt.subplot(2, 10, plt_i + 1)
                plt.plot(plt_pack[1]), plt.title(plt_pack[0])
            plt.tight_layout()
            plt.savefig(
                osp.join(viz_dir, f"{prefix}DEBUG_dynamic_scaffold_init_{step}.jpg")
            )
            plt.close()

    fig = plt.figure(figsize=(25, 7))
    for plt_i, plt_pack in enumerate(
        [
            ("loss", loss_list),
            ("loss_coord", loss_coord_list),
            ("loss_len", loss_len_list),
            ("loss_p_acc", loss_p_acc_list),
            ("loss_q_acc", loss_q_acc_list),
            ("loss_small_corr", loss_small_corr_list),
            #
            ("loss_flow_xyz", loss_flow_xyz_list),
            ("loss_flow_nrm", loss_flow_nrm_list),
            ("metric_flow_error", metric_flow_error_list),
            ("metric_normal_angle", metric_normal_angle_list),
            #
            ("loss_drag_xyz", loss_drag_xyz_list),
            ("metric_drag_xyz", metric_drag_xyz_list),
            #
            ("loss_dense_flow_xyz", loss_dense_flow_xyz_list),
            ("loss_dense_flow_nrm", loss_dense_flow_nrm_list),
            ("metric_dense_flow_error", metric_dense_flow_error_list),
            ("metric_dense_normal_angle", metric_dense_normal_angle_list),
            #
            ("loss_sk_w_consist", loss_sk_w_consist_list),
            ("metric_sk_w_consist", metric_sk_w_consist_list),
            #
            ("loss_dense_sk_w_consist", loss_dense_sk_w_consist_list),
            ("metric_dense_sk_w_consist", metric_dense_sk_w_consist_list),
        ]
    ):
        plt.subplot(2, 10, plt_i + 1)
        plt.plot(plt_pack[1]), plt.title(plt_pack[0])
    plt.tight_layout()
    plt.savefig(osp.join(log_dir, f"{prefix}dynamic_scaffold_init.jpg"))
    plt.close()
    viz_sigma_hist(scf, osp.join(viz_dir, f"{prefix}dyn_scf_sig_final_hist.jpg"))
    make_video_from_pattern(
        osp.join(viz_dir, f"{prefix}dyn_scf_init*.jpg"),
        osp.join(viz_dir, f"{prefix}dyn_scf_init.mp4"),
    )

    if save_flag:
        viz_frame = viz_curve(
            scf._node_xyz.detach(),
            rgb_list,
            semantic_feature_list,
            solid_mask,
            cams,
            viz_n=-1,
            time_window=1,
            res=480,
            pts_size=0.001,
            only_viz_last_frame=False,
            no_rgb_viz=True,
            n_line=4,
            text=f"Step={step}",
        )
        imageio.mimsave(
            osp.join(log_dir, f"{prefix}dynamic_scaffold_init.mp4"),
            viz_frame,
        )
        torch.save(
            scf.state_dict(), osp.join(log_dir, f"{prefix}dynamic_scaffold_init.pth")
        )
        print(f"Save scaffold to {osp.join(log_dir, f'{prefix}dynamic_scaffold_init.pth')}")

    return scf


def solve_4dscf(
    prior2d: Prior2D,
    scf: Scaffold4D,
    cams: SimpleFovCamerasDelta,
    viz_dir,
    log_dir,
    #
    resample_flag=True,
    #
    max_num_nodes=8192,
    mlevel_resample_steps=32,
    lr_p=0.1,
    lr_q=0.1,
    lr_sig=0.03,
    lr_sk_q=0.03,
    lr_p_finetune=0.01,
    lr_q_finetune=0.01,
    lr_sig_finetune=0.003,
    lr_sk_q_finetune=0.003,
    stage1_steps=300,
    stage1_decay_start_ratio=0.5,
    stage1_decay_factor=100.0,
    temporal_diff_shift=[1],
    temporal_diff_weight=[1.0],
    #
    n_flow_pair=50,
    stage2_steps=300,
    stage2_decay_start_ratio=0.5,
    stage2_decay_factor=100.0,
    #
    stage3_steps=300,
    stage3_decay_start_ratio=0.5,
    stage3_decay_factor=100.0,
    #
    lr_p_sem=0.01,
    lr_q_sem=0.01,
    lr_sig_sem=0.003,
    lr_sk_q_sem=0.003,
    sem_spa_radius_ratio=5.0,
    sem_rgb_std_ratio=3.0,
    sem_feat_std_ratio=3.0,
    sem_drag_K=4,
    sem_node_cover_K=0,  # 4
    sem_drag_matching_method="sem",
    drag_association_interval=50,
    stage4_steps=300,
    stage4_decay_start_ratio=0.5,
    stage4_decay_factor=100.0,
    stage4_dense_flow_bake_n=2048,
    #
    viz_interval=1000000,  # 100
    viz_debug=False,
    # ABL flags
    no_baking_flag=False,
    no_semantic_drag_flag=False,
    no_geo_flag=False,
):
    # ! ABL CFG
    if no_geo_flag:
        logging.warning(f"ABL: disable all geo init stage, use svd to solve SO(3) and return!")
        scf = __compute_R_from_xyz__(scf)
        return  
    if no_baking_flag:
        logging.warning(f"ABL: disable all baking")
        n_flow_pair = 0
        stage4_dense_flow_bake_n=0
    if no_semantic_drag_flag:
        logging.warning(f"ABL: disable all semantic drag")
        stage4_steps = 0
    # ! END ABL CFG
    
    
    if scf.M > max_num_nodes:
        logging.warning(
            f"To work on a reasonable number of nodes, random resample the nodes to {max_num_nodes}!"
        )
        random_ind = torch.randperm(scf.M)[:max_num_nodes]
        scf.resample_node(resample_ind=random_ind)

    # * Stage 1 Fill in the xyz with len arap
    logging.info(f"Start SCF 4D Stage-1 with M={scf.M}, T={scf.T}")
    stage_viz_dir = osp.join(viz_dir, "4dscf_geo_stage1")
    os.makedirs(stage_viz_dir, exist_ok=True)
    scf = __geo_loop__(
        viz_dir=stage_viz_dir,
        log_dir=log_dir,
        #
        prior2d=prior2d,
        scf=scf,
        cams=cams,
        mlevel_resample_steps=mlevel_resample_steps,
        #
        lr_q=0.0,
        lr_p=lr_p,
        lr_sig=0.0,
        #
        total_steps=stage1_steps,
        max_time_window=200,
        # * Basic Phy losses
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        lambda_local_coord=0.0,
        lambda_metric_len=1.0,
        lambda_xyz_acc=0.1,
        lambda_q_acc=0.0,
        # * Bake
        n_flow_pair=0,
        # topo
        update_full_topo=False,
        use_mask_topo=True,
        #
        decay_start=int(stage1_decay_start_ratio * stage1_steps),
        decay_factor=stage1_decay_factor,
        # viz
        viz_interval=viz_interval,
        viz_debug_interval=-1,
        #
        prefix="stage1_",
    )

    # * Stage 2 Fill in rotation and coord arap
    logging.info(f"Prepare for stage2, resampling and procrustes ...")
    bake_mask = scf._curve_slot_init_valid_mask.clone()
    bake_nrm_list = scf._curve_normal_init.clone().detach()
    bake_xyz_list = scf._node_xyz.clone().detach()
    scf = __compute_R_from_xyz__(scf)

    logging.info(
        f"Stage 2 Baking tracks and normals, filling in frames with M={scf.M} T={scf.T} ..."
    )
    stage_viz_dir = osp.join(viz_dir, "4dscf_geo_stage2")
    os.makedirs(stage_viz_dir, exist_ok=True)
    scf = __geo_loop__(
        viz_dir=stage_viz_dir,
        log_dir=log_dir,
        #
        prior2d=prior2d,
        scf=scf,
        cams=cams,
        mlevel_resample_steps=mlevel_resample_steps,
        #
        lr_q=lr_q,
        lr_sk_q=lr_sk_q,
        lr_p=0.0,
        lr_sig=lr_sig,
        #
        total_steps=stage2_steps,
        max_time_window=200,
        # * Basic Phy losses
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        lambda_local_coord=1.0,
        lambda_metric_len=0.0,
        lambda_xyz_acc=0.0,
        lambda_q_acc=0.1,
        # * Bake
        n_flow_pair=n_flow_pair,
        bake_mask=bake_mask,
        bake_xyz_list=bake_xyz_list,
        bake_nrm_list=bake_nrm_list,
        lambda_flow_xyz=1.0,
        lambda_flow_nrm=1.0,
        # topo
        update_full_topo=False,  # because the xyz is not changed, no need to recompute
        use_mask_topo=False,
        #
        decay_start=int(stage2_decay_start_ratio * stage2_steps),
        decay_factor=stage2_decay_factor,
        # viz
        viz_interval=viz_interval,
        viz_debug_interval=-1,
        #
        prefix="stage2_",
    )

    # ! resample again, without mask
    if resample_flag:
        scf.resample_node(1.0, use_mask=False)

    # * Stage3 Joint Tuning
    logging.info(f"Stage 3 Joint Tuning with M={scf.M} T={scf.T} ...")
    stage_viz_dir = osp.join(viz_dir, "4dscf_geo_stage3")
    os.makedirs(stage_viz_dir, exist_ok=True)
    scf = __geo_loop__(
        viz_dir=stage_viz_dir,
        log_dir=log_dir,
        #
        prior2d=prior2d,
        scf=scf,
        cams=cams,
        mlevel_resample_steps=mlevel_resample_steps,
        #
        lr_q=lr_q_finetune,
        lr_p=lr_p_finetune,
        lr_sig=lr_sig_finetune,
        lr_sk_q=lr_sk_q_finetune,
        #
        total_steps=stage3_steps,
        max_time_window=200,
        # * Basic Phy losses
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        lambda_local_coord=0.5,  # ! debug
        lambda_metric_len=1.0,
        lambda_xyz_acc=0.1,
        lambda_q_acc=0.1,
        # * Bake
        n_flow_pair=n_flow_pair,
        bake_mask=bake_mask,
        bake_xyz_list=bake_xyz_list,
        bake_nrm_list=bake_nrm_list,
        lambda_flow_xyz=1.0,
        lambda_flow_nrm=1.0,
        # topo
        update_full_topo=True,  # because the xyz is not changed, no need to recompute
        use_mask_topo=False,
        #
        decay_start=int(stage3_decay_start_ratio * stage3_steps),
        decay_factor=stage3_decay_factor,
        # viz
        viz_interval=viz_interval,
        viz_debug_interval=-1,
        #
        prefix="stage3_",  # no prefix
        save_flag=stage4_steps == 0,
    )

    # * Stage4 Semantic Drag
    if stage4_steps == 0: # ! no early return, save the correct name!
        return
    logging.info(f"Stage 4 Semantic Drag ...")
    stage_viz_dir = osp.join(viz_dir, "4dscf_geo_stage4")
    os.makedirs(stage_viz_dir, exist_ok=True)
    scf = __geo_loop__(
        viz_dir=stage_viz_dir,
        log_dir=log_dir,
        #
        prior2d=prior2d,
        scf=scf,
        cams=cams,
        mlevel_resample_steps=mlevel_resample_steps,
        #
        lr_q=lr_q_sem,
        lr_p=lr_p_sem,
        lr_sig=lr_sig_sem,
        lr_sk_q=lr_sk_q_sem,
        #
        total_steps=stage4_steps,
        max_time_window=200,
        # * Basic Phy losses
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        lambda_local_coord=0.5,
        lambda_metric_len=1.0,
        lambda_xyz_acc=0.1,
        lambda_q_acc=0.1,
        # * Bake
        n_flow_pair=n_flow_pair,
        bake_mask=bake_mask,
        bake_xyz_list=bake_xyz_list,
        bake_nrm_list=bake_nrm_list,
        lambda_flow_xyz=1.0,
        lambda_flow_nrm=1.0,
        # * drag
        drag_association_interval=drag_association_interval,
        lambda_drag_xyz=1.0,
        drag_spatial_radius_ratio=sem_spa_radius_ratio,
        drag_rgb_std_ratio=sem_rgb_std_ratio,
        drag_feat_std_ratio=sem_feat_std_ratio,
        drag_K=sem_drag_K,
        node_cover_K=sem_node_cover_K,
        drag_matching_method=sem_drag_matching_method,
        # * flow drag
        n_depth_pair_per_t=stage4_dense_flow_bake_n,
        t_depth_pair=10,
        lambda_dense_flow_xyz=1.0,
        lambda_dense_flow_nrm=1.0,
        # topo
        update_full_topo=False,
        use_mask_topo=False,
        #
        decay_start=int(stage4_decay_start_ratio * stage4_steps),
        decay_factor=stage4_decay_factor,
        # viz
        viz_interval=viz_interval,
        viz_debug_interval=drag_association_interval - 1 if viz_debug else -1,
        #
        prefix="stage4_",  # no prefix
        save_flag=True,
    )

    return


@torch.no_grad()
def grow_node_by_coverage(
    grow_interval: int,
    prior2d: Prior2D,
    scf: Scaffold4D,
    cams: SimpleFovCamerasIndependent,
    matching_method="sem",
    leaf_coverage_K=4,
    spatial_radius_ratio=3.0,
    rgb_std_ratio=3.0,
    feat_std_ratio=3.0,
    max_uncover_num=4096,
    viz_dir=None,
):
    # * first append every frame
    t_list, attach_list, xyz_list = [], [], []
    for t in tqdm(range(0, cams.T, grow_interval)):
        # * First see how the depth and scf aligned, identify un-covered depth pts
        fg_mask = prior2d.get_dynamic_mask(t) * prior2d.get_depth_mask(t)
        xyz_cam = backproject(
            prior2d.homo_map[fg_mask], prior2d.get_depth(t)[fg_mask], cams
        )
        xyz_world = cams.trans_pts_to_world(t, xyz_cam)
        xyz_scf_node = scf._node_xyz[t]
        coverage_mask = scf.check_points_coverage(
            xyz_world, t, scf.spatial_unit, K=leaf_coverage_K
        )
        leaf_uncovered_mask = ~coverage_mask  # further do outlier removal with open3d
        # bound the maximum uncover num
        if leaf_uncovered_mask.sum() > max_uncover_num:
            old_N = leaf_uncovered_mask.sum()
            choice = torch.randperm(old_N)[:max_uncover_num]
            sub_mask = torch.zeros(old_N).bool().to(leaf_uncovered_mask)
            sub_mask[choice] = True
            leaf_uncovered_mask[leaf_uncovered_mask.clone()] = sub_mask

        inlier_mask = __outlier_removal_o3d__(xyz_world[leaf_uncovered_mask])
        leaf_uncovered_mask[leaf_uncovered_mask.clone()] = inlier_mask

        uncovered_int_uv = prior2d.pixel_int_map[fg_mask][leaf_uncovered_mask]
        uncovered_feat = prior2d.query_low_res_semantic_feat(uncovered_int_uv, t)
        uncovered_rgb = query_image_buffer_by_pix_int_coord(
            prior2d.get_rgb(t), uncovered_int_uv
        )
        uncovered_feat = torch.cat([uncovered_rgb, uncovered_feat], -1)  # N,C
        uncovered_xyz = xyz_world[leaf_uncovered_mask]  # N,3

        node_id = torch.arange(scf.M).to(scf.device)
        node_feat = scf.semantic_feature_mean  # M,C
        node_std = torch.sqrt(torch.clamp(scf.semantic_feature_var, min=0.0))  # M,C
        node_xyz = xyz_scf_node  # M,3

        append_xyz, matched_node_id, _, append_value = __semantic_match__(
            dep_feat=uncovered_feat,
            dep_xyz=uncovered_xyz,
            node_feat=node_feat,
            node_xyz=node_xyz,
            node_std=node_std,
            node_ori_id=node_id,
            scf=scf,
            K=1,
            matching_method=matching_method,
            spatial_radius_ratio=spatial_radius_ratio,
            rgb_std_ratio=rgb_std_ratio,
            feat_std_ratio=feat_std_ratio,
        )

        # resort the matching for later sample
        rank_ind = append_value.argsort().to(scf.device)
        matched_node_id = matched_node_id.to(scf.device)[rank_ind]
        append_xyz = append_xyz.to(scf.device)[rank_ind]

        # subsample the xyz
        sampled_ind = __uniform_subsample_vtx__(append_xyz, scf.spatial_unit)
        xyz = append_xyz[sampled_ind]
        attach_ind = matched_node_id[sampled_ind]
        t = torch.ones_like(attach_ind) * t

        t_list.append(t)
        attach_list.append(attach_ind)
        xyz_list.append(xyz)

    t_list = torch.cat(t_list, 0)
    attach_list = torch.cat(attach_list, 0)
    xyz_list = torch.cat(xyz_list, 0)
    quat_list = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(t_list), 1).to(xyz_list)

    logging.info(f"SCF INIT first over filling {len(xyz_list)} nodes ...")
    # warp

    node_xyz_list, node_quat_list = [], []
    for t in tqdm(range(cams.T)):

        _xyz, _R = scf.warp(
            attach_node_ind=attach_list,
            query_xyz=xyz_list,
            query_dir=quaternion_to_matrix(quat_list),
            query_tid=t_list,
            target_tid=t,
        )
        _quat = matrix_to_quaternion(_R)
        node_xyz_list.append(_xyz)
        node_quat_list.append(_quat)
    node_xyz_list = torch.stack(node_xyz_list, 0)
    node_quat_list = torch.stack(node_quat_list, 0)

    scf.append_nodes_traj(
        torch.optim.Adam(scf.get_optimizable_list()),  # dummy
        node_xyz_list,
        node_quat_list,
        0,
    )

    # * then use a re-sample to select the curves if they are too dense!
    scf.resample_node(1.0, use_mask=True)

    viz_frame = viz_curve(
        scf._node_xyz.detach(),
        scf._curve_color_init,
        scf._curve_slot_init_valid_mask,
        cams,
        viz_n=-1,
        time_window=1,
        res=480,
        pts_size=0.001,
        only_viz_last_frame=False,
        no_rgb_viz=True,
        n_line=4,
        text=f"After Filling",
    )
    imageio.mimsave(
        osp.join(viz_dir, f"growed_by_coverage_after_nodes.mp4"),
        viz_frame,
    )

    # dbg_save_dict = scf.export_node_edge_dict()
    # dbg_save_dict["rgb"] = scf._curve_color_init.detach().cpu().numpy()
    # dbg_save_dict["mask"] = scf._curve_slot_init_valid_mask.detach().cpu().numpy()
    # np.savez(
    #     osp.join(viz_dir, f"DEBUG_dyn_scf_init_filling.npz"),
    #     **dbg_save_dict,
    # )

    return


def load_saved_bake(fn, device=torch.device("cuda")):
    data = np.load(fn)
    mask, xyz, nrm = data["mask"], data["xyz"], data["nrm"]
    mask = torch.from_numpy(mask).to(device)
    xyz = torch.from_numpy(xyz).to(device)
    nrm = torch.from_numpy(nrm).to(device)
    return mask, xyz, nrm
