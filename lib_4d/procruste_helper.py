# include more stuff than procruste
import torch, numpy as np
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


def project_to_so3(W):
    # ...,3,3
    ori_shape = W.shape
    assert ori_shape[-1] == ori_shape[-2] == 3
    W = W.reshape(-1, 3, 3)
    U, s, V = torch.svd(W.double())
    # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
    U, s, V = U.float(), s.float(), V.float()
    R_tmp = torch.einsum("nij,nkj->nik", U, V)
    det = torch.det(R_tmp)
    dia = torch.ones(len(det), 3).to(det)
    dia[:, -1] = det
    Sigma = torch.diag_embed(dia)
    U = torch.einsum("nij,njk->nik", U, Sigma)
    R_star = torch.einsum("nij,nkj->nik", U, V)
    R_star = R_star.reshape(*ori_shape)
    return R_star


def procruste(src, dst, weight=None, return_t=False):
    # src, dst: B,N,3; weight: None or B,N
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[2] == 3
    assert src.shape == dst.shape
    with torch.no_grad():
        if weight is None:
            weight = torch.ones_like(src[:, :, 0]) / src.shape[1]
        else:
            assert weight.ndim == 2
            assert weight.shape == src.shape[:2]
            assert torch.allclose(
                weight.sum(-1), torch.ones_like(weight[..., -1]), 1e-4
            )
    # ! warning, when doing weight, have to weight the centroid
    p_bar = (src * weight[..., None]).sum(1, keepdim=True)
    q_bar = (dst * weight[..., None]).sum(1, keepdim=True)
    p_nn = src - p_bar  # N,K,3
    q_nn = dst - q_bar  # N,K,3
    # # debug
    # np.savetxt("./debug/p_nn.xyz", p_nn[0].cpu().numpy())
    # np.savetxt("./debug/q_nn.xyz", q_nn[0].cpu().numpy())

    W = torch.einsum("nki,nkj->nkij", p_nn, q_nn)
    W = W * weight[..., None, None]
    W = W.sum(1)
    U, s, V = torch.svd(
        W.double()
    )  # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
    U, s, V = U.float(), s.float(), V.float()
    # R_star = V @ (U.T)
    # ! handling flipping
    R_tmp = torch.einsum("nij,nkj->nik", V, U)
    det = torch.det(R_tmp)
    dia = torch.ones(len(det), 3).to(det)
    dia[:, -1] = det
    Sigma = torch.diag_embed(dia)
    V = torch.einsum("nij,njk->nik", V, Sigma)
    R_star = torch.einsum("nij,nkj->nik", V, U)
    # q = Rp -> dst = R @ src + t
    if return_t:
        t_star = q_bar.squeeze(1) - torch.einsum("bij,bj->bi", R_star, p_bar.squeeze(1))
        return R_star, t_star
    return R_star


def compute_so3_from_knn_flow(src, dst, K):
    assert src.ndim == 2 and dst.ndim == 2
    # build topology
    # use 6 dim vec to compute knn
    joint_xyz = torch.cat([src, dst], -1)
    _, knn_idx, _ = knn_points(joint_xyz[None], joint_xyz[None], K=K)
    knn_idx = knn_idx[0]
    dst_nn = dst[knn_idx]  # N,K,3
    src_nn = src[knn_idx]  # N,K,3
    return procruste(src_nn, dst_nn)


def grow_apple_to_tree(
    new_mu,
    new_fr,
    src_mu,
    dst_mu,
    topo_curve_dist_th,
    knn_th,
    steps=32,
    K=6,
    lbs_alpha=64.0,
):
    # raise NotImplementedError("OLD after v3.0")
    # grow apples on the tree, the assumption is to make the distance to nn in the base not changing to much
    # will use identity for those in new_mu far away from the src_mu
    # all the src and dst are base (tree), and new points (apples on the tree) are growing on them
    # new_mu and src_mu, src_fr are in the same step, while dst can be multiple frames
    # all the src, dst should correspond in the order
    assert new_mu.ndim == 2 and src_mu.ndim == 2
    assert dst_mu.ndim == 3 and new_fr.ndim == 3
    assert new_mu.shape[1] == 3 and src_mu.shape[1] == 3 and dst_mu.shape[2] == 3

    T, N = len(dst_mu), len(new_mu)
    ret_mu = torch.zeros((T, N, 3), device=new_mu.device)
    ret_fr = torch.zeros((T, N, 3, 3), device=new_mu.device)

    n = np.ceil(N / steps).astype(np.int)
    track_ind = torch.arange(N, device=new_mu.device)
    recover_ind = []

    dst_fr_list = []
    for _ in range(steps):
        # * identiy first nearest n pts
        d_sq, _, _ = knn_points(new_mu[None], src_mu[None], K=1)
        d_sq = d_sq.squeeze(0).squeeze(-1)
        num_valid_knn = (d_sq < knn_th**2).sum()
        n = min(n, num_valid_knn)
        if n == 0:  # early break and use identity
            break
        _nearerst_ind = d_sq.argsort()[:n]
        recover_ind.append(track_ind[_nearerst_ind])
        working_new_mu_src = new_mu[_nearerst_ind]
        working_new_fr_src = new_fr[_nearerst_ind]

        # * find knn in _mu and build skinning in src frame
        d_sq, topo_ind, _ = knn_points(working_new_mu_src[None], src_mu[None], K=K)
        d_sq, topo_ind = d_sq.squeeze(0), topo_ind.squeeze(0)
        lbs_weight = torch.exp(-d_sq * lbs_alpha) + 1e-6  # ! important + eps
        lbs_weight = lbs_weight / (lbs_weight.sum(dim=-1, keepdim=True))

        # * Use the known flow to filter out the topo outlier
        dst_curves = dst_mu[:, topo_ind, :]  # T,n,K,3
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
        # if the distance is too large, maskout the lbs_weight, but
        valid_curve_mask = curve_dist_to_nearest < topo_curve_dist_th
        valid_curve_mask[curve_dist_to_nearest == 1e10] = True
        lbs_weight[~valid_curve_mask] = 0.0
        lbs_weight = lbs_weight / (lbs_weight.sum(dim=-1, keepdim=True) + 1e-6)

        # * compute the flow by weighted average, this obvious the solution when assumption the new point has relatively un-changed distance to its neighbors
        flow_src2dst = dst_mu[:, topo_ind, :] - src_mu[topo_ind][None]  # T,n,K,3
        averaged_flow = (flow_src2dst * lbs_weight[None, ..., None]).sum(dim=2)  # T,n,3
        working_new_mu_dst = working_new_mu_src[None] + averaged_flow

        # * compute the fr changes to the dst by solveing a weigthed procruste
        proc_src = (
            src_mu[topo_ind][None]
            .expand(T, -1, -1, -1)
            .reshape(T * len(topo_ind), K, 3)
        )
        proc_dst = dst_mu[:, topo_ind, :].reshape(T * len(topo_ind), K, 3)
        proc_w = lbs_weight[None, ...].expand(T, -1, -1).reshape(T * len(topo_ind), K)
        delta_R = procruste(src=proc_src, dst=proc_dst, weight=proc_w)
        delta_R = delta_R.reshape(T, len(topo_ind), 3, 3)
        working_new_fr_dst = torch.einsum("tnij,njk->tnik", delta_R, working_new_fr_src)

        # * convert working_new_fr_dst to nearest SO(3)
        working_new_fr_dst = project_to_so3(working_new_fr_dst)

        # * cat to src and dst; remove from new_mu and track_ind
        dst_mu = torch.cat([dst_mu, working_new_mu_dst], 1)
        dst_fr_list.append(working_new_fr_dst)
        src_mu = torch.cat([src_mu, working_new_mu_src], 0)
        del_mask = torch.ones_like(track_ind).bool().to(track_ind.device)
        del_mask[_nearerst_ind] = False
        track_ind = track_ind[del_mask]
        new_mu = new_mu[del_mask]
        new_fr = new_fr[del_mask]
    # handle the far away points id
    if len(new_mu) > 0:
        dst_mu = torch.cat([dst_mu, new_mu[None].expand(T, -1, -1)], 1)
        dst_fr_list.append(new_fr[None].expand(T, -1, -1, -1))
        recover_ind.append(track_ind)
    # recover the correct order
    recover_ind = torch.cat(recover_ind, 0)
    ret_mu = dst_mu[:, -N:]
    ret_fr = torch.cat(dst_fr_list, 1)
    back_ind = torch.argsort(recover_ind)
    ret_mu = ret_mu[:, back_ind]
    ret_fr = ret_fr[:, back_ind]
    assert not torch.isnan(ret_mu).any()
    ret_q = matrix_to_quaternion(ret_fr)
    return ret_mu, ret_q
