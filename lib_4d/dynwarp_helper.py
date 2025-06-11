from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops import knn_points
import logging
import torch
import os, sys, os.path as osp
import torch.nn.functional as F
import numpy as np

sys.path.append(osp.dirname(osp.abspath(__file__)))
from render_helper import render
from index_helper import query_image_buffer_by_pix_int_coord, uv_to_pix_int_coordinates
from projection_helper import project, backproject

# from procruste_helper import grow_apple_to_tree, compute_so3_from_knn_flow
from index_helper import round_int_coordinates, get_subsample_mask_like
from dualquat_helper import Rt2dq, dq2unitdq, dq2Rt


@torch.no_grad()
def __align_pixel_depth_scale_backproject__(
    homo_map,
    src_align_mask,
    src,
    src_mask,
    dst,
    dst_mask,
    cams,
    knn=8,
    infrontflag=True,
):
    # src_align_mask: which pixel is going to be aligned
    # src_mask: the valid pixel in the src
    # dst_mask: the valid pixel in the dst
    # use 3D nearest knn nn to find the best scaling, local rigid warp

    # the base pts build correspondence between current frame and
    base_mask = src_mask * dst_mask * (~src_align_mask)
    query_mask = src_align_mask

    ratio = dst / (src + 1e-6)
    base_pts_ratio = ratio[base_mask]

    query_pts = backproject(homo_map[query_mask], src[query_mask], cams)

    # backproject src depth to 3D
    if knn > 0:
        base_pts = backproject(homo_map[base_mask], src[base_mask], cams)
        _, ind, _ = knn_points(query_pts[None], base_pts[None], K=knn)
        ind = ind[0]
        ratio = base_pts_ratio[ind]
    else:
        ratio = base_pts_ratio.mean()[None, None].expand(len(query_pts), -1)
    ret = ratio.mean(-1, keepdim=True) * query_pts

    if infrontflag:
        # ! make sure the ret-z is always smaller than the dst z
        ret_z = ret[:, -1]
        dst_z = dst[query_mask]
        assert (dst_z > -1e-6).all()
        new_z = torch.min(ret_z, dst_z - 1e-4)
        logging.info(f"Make sure the aligned points is in front of the dst depth")
        ratio = new_z / torch.clamp(ret_z, min=1e-6)
        # assert (ratio <= 1 + 1e-6).all()
    ret = ret * ratio[:, None]
    return ret


def align_to_model_depth(
    prior2d,
    working_mask,
    cams,
    tid,
    s_model,
    d_model=None,
    dep_align_knn=9,
    sub_sample=1,
    src_valid_mask_type="sky_dep_sta",
):
    gs5 = [s_model()]
    if d_model:
        gs5.append(d_model(tid))
    render_dict = render(
        gs5, prior2d.H, prior2d.W, cams.rel_focal, cams.cxcy_ratio, cams.T_cw(tid)
    )
    model_alpha = render_dict["alpha"].squeeze(0)
    # model_dep = render_dict["dep"].squeeze(0) / (model_alpha + 1e-6)
    model_dep = render_dict["dep"].squeeze(0)

    # align prior depth to current depth
    sub_mask = get_subsample_mask_like(working_mask, sub_sample)
    ret_mask = working_mask * sub_mask
    new_mu_cam = __align_pixel_depth_scale_backproject__(
        homo_map=prior2d.homo_map,
        src_align_mask=ret_mask,
        src=prior2d.depths[tid],
        src_mask=prior2d.get_mask_by_key(src_valid_mask_type, tid) * sub_mask,
        dst=model_dep,
        dst_mask=(model_alpha > 0.5)
        & (
            ~working_mask
        ),  # ! warning, here manually use the original non-subsampled mask, because the dilated place is not reliable!
        cams=cams,
        knn=dep_align_knn,
    )
    return new_mu_cam, ret_mask


def dq_arap_interpolation(
    new_mu,
    new_fr,
    curve_x,
    curve_fr,
    associate_at: int,
    # config
    K=16,
    topo_nncheck_ratio=6.0,
    topo_cutoff_ratio=10.0,
    sk_sigma_ratio=1.0,
    topo_scale_auto_k=64,
    steps=32,
    unit_scale=None,
):
    # todo: this func can be boosted definitely, by avoiding recomputation, e.g. the dq conversion.
    assert new_mu.ndim == 2 and new_mu.shape[1] == 3
    assert new_fr.ndim == 3 and new_fr.shape[1] == 3 and new_fr.shape[2] == 3
    assert curve_x.ndim == 3 and curve_x.shape[2] == 3
    assert curve_fr.ndim == 4 and curve_fr.shape[2] == 3 and curve_fr.shape[3] == 3

    verify_mu = new_mu.clone()
    T, M = len(curve_x), len(new_mu)
    # ret_mu = torch.zeros((T, M, 3), device=new_mu.device)
    # ret_fr = torch.zeros((T, M, 3, 3), device=new_mu.device)

    # identify the scene scale by knn distance
    curve_x_asso = curve_x[associate_at]
    if unit_scale is None:
        unit_scale = knn_points(
            curve_x_asso[None], curve_x_asso[None], K=topo_scale_auto_k + 1
        )[0][0]
        unit_scale = unit_scale[:, 1:].reshape(-1).mean()
    topo_nncheck_distsq_th = unit_scale * topo_nncheck_ratio
    topo_cutoff_distsq_th = unit_scale * topo_cutoff_ratio
    sk_sigma = unit_scale * sk_sigma_ratio

    # multi-step incremental warp
    n = np.ceil(M / steps).astype(np.int)
    trace_ind = torch.arange(M, device=new_mu.device)
    recover_ind = []

    for _ in range(steps):
        curve_x_asso = curve_x[associate_at]
        curve_fr_asso = curve_fr[associate_at]

        # * identiy first nearest n pts
        d_sq, _, _ = knn_points(new_mu[None], curve_x_asso[None], K=1)
        d_sq = d_sq.squeeze(0).squeeze(-1)
        num_valid_knn = (d_sq < topo_cutoff_distsq_th).sum()
        # logging.info(
        #     f"Valid KNN: {num_valid_knn}/{len(new_mu)} with sq_th={topo_cutoff_distsq_th}"
        # )
        n = min(n, num_valid_knn)
        if n == 0:  # early break and use identity
            break
        _nearest_ind = d_sq.argsort()[:n]
        recover_ind.append(trace_ind[_nearest_ind])
        working_new_mu_src = new_mu[_nearest_ind]
        working_new_fr_src = new_fr[_nearest_ind]

        # * find knn in _mu and build skinning in src frame
        d_sq, topo_ind, _ = knn_points(
            working_new_mu_src[None], curve_x_asso[None], K=K
        )
        d_sq, topo_ind = d_sq.squeeze(0), topo_ind.squeeze(0)
        skinning_w = torch.exp(-d_sq / (2 * (sk_sigma**2)))
        skinning_w = torch.clamp(skinning_w, min=1e-6)
        skinning_w = skinning_w / (skinning_w.sum(dim=-1, keepdim=True))

        # * Use the known curve to filter out the topo outlier
        dst_curves = curve_x[:, topo_ind, :]  # T,n,K,3
        nearest_curve = dst_curves[:, :, :1, :]  # T,n,1,3
        # compute the distance to the nearest curve
        curve_dist_to_nearest = (
            (dst_curves - nearest_curve).norm(dim=-1, p=2).max(dim=0).values
        )  # n,K
        # make sure that the nearest 3 points are not masked out
        curve_dist_to_nearest[:, 0] = 1e10
        for _ in range(2):
            curve_dist_to_nearest = torch.scatter(
                curve_dist_to_nearest,
                1,
                curve_dist_to_nearest.argmin(dim=-1, keepdim=True),
                1e10,
            )
        valid_curve_mask = curve_dist_to_nearest < topo_nncheck_distsq_th
        valid_curve_mask[curve_dist_to_nearest == 1e10] = True
        skinning_w[~valid_curve_mask] = 0.0
        skinning_w = skinning_w / (skinning_w.sum(dim=-1, keepdim=True) + 1e-6)

        # prepare associate to all rest time steps transformations
        R_dst_asso, t_dst_asso = delta_Rt_ij_batch(
            R_wl_i=curve_fr,
            R_wl_j=curve_fr_asso[None].expand(T, -1, -1, -1),
            t_wl_i=curve_x,
            t_wl_j=curve_x_asso[None].expand(T, -1, -1),
        )
        # DQ interpolation
        curve_dq_dst_asso = Rt2dq(R_dst_asso, t_dst_asso)  # T,N_curve,8
        nn_dq_dst_asso = curve_dq_dst_asso[:, topo_ind]  # T,n,K,8
        dq_dst_asso = (nn_dq_dst_asso * skinning_w[None, :, :, None]).sum(2)
        dq_dst_asso = dq2unitdq(dq_dst_asso)
        working_R_dst_asso, working_t_dst_asso = dq2Rt(dq_dst_asso)
        # Apply DQ
        working_new_mu_dst = (
            torch.einsum("tnij,nj->tni", working_R_dst_asso, working_new_mu_src)
            + working_t_dst_asso
        )
        working_new_fr_dst = torch.einsum(
            "tnij,njk->tnik", working_R_dst_asso, working_new_fr_src
        )

        # cat to src and dst; remove from new_mu and track_ind
        curve_x = torch.cat([curve_x, working_new_mu_dst], 1)
        curve_fr = torch.cat([curve_fr, working_new_fr_dst], 1)

        remaining_mask = torch.ones_like(trace_ind).bool().to(trace_ind.device)
        remaining_mask[_nearest_ind] = False
        trace_ind = trace_ind[remaining_mask]
        new_mu = new_mu[remaining_mask]
        new_fr = new_fr[remaining_mask]

    # handle the far away points id
    assert len(new_mu) == len(new_fr)
    if len(new_mu) > 0:
        logging.info(
            f"{len(new_mu)} {float(len(new_mu))/float(len(verify_mu))*100.0:.2f}% far away points not warped!"
        )
        curve_x = torch.cat([curve_x, new_mu[None].expand(T, -1, -1)], 1)
        curve_fr = torch.cat([curve_fr, new_fr[None].expand(T, -1, -1, -1)], 1)
        recover_ind.append(trace_ind)
    # recover the correct order
    recover_ind = torch.cat(recover_ind, 0)
    ret_mu = curve_x[:, -M:]
    ret_fr = curve_fr[:, -M:]
    back_ind = torch.argsort(recover_ind)
    ret_mu = ret_mu[:, back_ind]
    ret_fr = ret_fr[:, back_ind]
    assert not torch.isnan(ret_mu).any()

    error = abs(verify_mu - ret_mu[associate_at]).max()
    assert error < 1e-5, f"DQ Skinning Error: {error.max()}"
    return ret_mu, ret_fr


def delta_Rt_ij(R_wl_i, R_wl_j, t_wl_i, t_wl_j):
    # the input has meaning: p_world = R_wl @ p_local + t_wl, the i,j means at time i and j
    # p_local = R_j.T @ (p_w_t=j - t_j)
    # p_w_t=i = R_i @ p_local + t_i = (R_i @ R_j.T) @ (p_w_t=j - t_j) + t_i
    # p_w_t=i = (R_i @ R_j.T) @ p_w_t=j + (t_i - R_i @ R_j.T @ t_j)
    node_R_ij = torch.einsum(
        "mnk,mlk->mnl", R_wl_i, R_wl_j
    )  # M,3,3 R_ij = R_i @ (R_j.T)
    node_t_ij = t_wl_i - torch.einsum("mij,mj->mi", node_R_ij, t_wl_j)  # M,3
    # verified by
    # R_ij, t_ij = self.get_node_Rt_ij(i_ind=self.get_tlist_ind(tid), j_ind=self.ref_ind)
    # R_ji, t_ji = self.get_node_Rt_ij(i_ind=self.ref_ind, j_ind=self.get_tlist_ind(tid))
    # R_error2 = abs(R_ij - torch.linalg.inv(R_ji)).max()
    # t_error2 = abs(t_ij - (-torch.einsum("nij,nj->ni", R_ij, t_ji))).max()
    return node_R_ij, node_t_ij


def delta_Rt_ij_batch(R_wl_i, R_wl_j, t_wl_i, t_wl_j):
    # shape: B,N,X
    node_R_ij = torch.einsum(
        "bmnk,bmlk->bmnl", R_wl_i, R_wl_j
    )  # B,M,3,3 R_ij = R_i @ (R_j.T)
    node_t_ij = t_wl_i - torch.einsum("bmij,bmj->bmi", node_R_ij, t_wl_j)  # B,M,3
    return node_R_ij, node_t_ij
