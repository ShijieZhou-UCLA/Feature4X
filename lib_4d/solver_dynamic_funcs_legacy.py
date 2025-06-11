from matplotlib import pyplot as plt
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import os, sys, os.path as osp
from tqdm import tqdm
import time

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from camera import SimpleFovCamerasIndependent
from prior2d import Prior2D
from solver_viz_helper import (
    make_video_from_pattern,
    viz_curve,
    viz_plt_missing_slot,
)
from lib_4d_misc import *
from solver_utils import (
    get_world_points,
    prepare_track_buffers,
    detect_sharp_changes_in_curve,
    line_segment_init,
    q2R,
    compute_spatial_unit_from_nndist,
)

from scf4d_model import Scaffold4D


def solve_4dscf_legacy(
    prior2d, cams, topo_k, viz_dir, log_dir, spatial_unit, total_steps=3000
):
    device = prior2d.working_device
    d_track, d_scf_mask, d_scf_xyz, d_scf_quat, _ = compute_dynamic_scaffold(
        prior2d=prior2d,
        viz_dir=viz_dir,
        log_dir=log_dir,
        cams=cams,
        topo_k=topo_k * 3,
        total_steps=total_steps,
        open_visible_after_step=10000000,  # ! fix the visible still make good apple resutls before
        topo_dist_th_ratio=16.0,
    )
    d_scf_xyz, d_scf_quat, d_scf_mask = resample_scf(
        d_scf_xyz,
        d_scf_quat,
        d_scf_mask,
        spatial_unit,
    )

    scf = Scaffold4D(
        node_xyz=d_scf_xyz,
        node_quat=d_scf_quat,
        skinning_k=topo_k,
        device=device,
        curve_slot_init_valid_mask=d_scf_mask,
        dyn_o_flag=True,
        # !
        topo_curve_dist_top_k=1, # ! remove
        topo_curve_dist_sample_T=100,
        topo_th_ratio=16.0 # ! 
    )
    scf = scf.to(device)
    return scf


def compute_dynamic_scaffold(
    viz_dir,
    log_dir,
    prior2d: Prior2D,
    cams: SimpleFovCamerasIndependent,
    max_time_window=30,
    lr_q=0.003,
    lr_p=0.003,
    lr_d=0.001,
    total_steps=3000,
    open_visible_after_step=1000000,  # never open
    temporal_diff_shift=[1, 3],
    temporal_diff_weight=[0.7, 0.3],
    lambda_global_std=0.0,
    lambda_local_coord=1.0,
    lambda_metric_len=1.0,
    dep_corr_ref_id=-1,
    # dep reg
    lambda_dep_corr=0.01,
    # small energy
    # ! waring, the acc and vel lambda should depends on the FPS!!!!
    # ! acc losses are very important
    lambda_xyz_std=0.0,
    lambda_xyz_vel=0.0,
    lambda_q_vel=0.0,
    lambda_xyz_acc=0.3,
    lambda_q_acc=0.3,
    decay_start=500,
    decay_factor=100,
    topo_k=16,
    # topo_dist_th_ratio=6.0,
    topo_dist_th_ratio=4.0,
    max_vel_th=0.1,
    min_valid_cnt_ratio=0.2,  # a noodle must have at least N valid points
    # min_valid_cnt_ratio=0.0001,
    viz_debug=False,
    # viz_interval=300,
    viz_interval=300,
    viz_video_interval=600,
):
    ##########################################
    # * Correction of dynamic depth as well
    ##########################################
    device = prior2d.track.device

    # prepare noodles
    d_track = prior2d.track[:, prior2d.track_dynamic_mask]
    d_track_mask = prior2d.track_mask[:, prior2d.track_dynamic_mask]

    homo_list, dep_list, _, rgb_list = prepare_track_buffers(
        prior2d, d_track, d_track_mask, t_list=torch.arange(cams.T)
    )

    viz_plt_missing_slot(
        d_track_mask, osp.join(viz_dir, "dyn_scaffold_init_completeness.jpg")
    )

    ########################################################################
    d_track_mask, _ = detect_sharp_changes_in_curve(
        d_track_mask,
        get_world_points(homo_list, dep_list, cams),
        max_vel_th=max_vel_th,
    )
    viz_plt_missing_slot(
        d_track_mask,
        osp.join(viz_dir, "dyn_scaffold_after_vel_filtering_completeness.jpg"),
    )
    # assert (d_track_mask.sum(dim=0) > int(cams.T * min_valid_cnt_ratio)).all()
    # ensure that at least one slot is visible
    min_valid_cnt = int(cams.T * min_valid_cnt_ratio)
    recheck_noodle_mask = d_track_mask.sum(dim=0) > min_valid_cnt
    logging.info(
        f"Each noodle must have at least {min_valid_cnt} valid nodes before completion, {recheck_noodle_mask.float().mean()*100:.2f}% pass the check"
    )
    # assert recheck_noodle_mask.any(), "no valid noodle anymore"
    d_track = d_track[:, recheck_noodle_mask]
    d_track_mask = d_track_mask[:, recheck_noodle_mask]
    homo_list = homo_list[:, recheck_noodle_mask]
    dep_list = dep_list[:, recheck_noodle_mask]
    rgb_list = rgb_list[:, recheck_noodle_mask]
    logging.info(
        f"Dyn Scaffold {d_track_mask.float().mean()*100.0:.2f}% empty slots need to be filled in!"
    )
    ########################################################################

    # * init the hallucination with naive line segments
    curve_ref = line_segment_init(
        d_track_mask, get_world_points(homo_list, dep_list, cams).clone()
    )
    # curve_ref = pchip_init(
    #     d_track_mask, get_world_points(homo_list, dep_list, cams).clone()
    # )

    # * optimize the physical loss for the noodles, where the empty slot are three dim and the valid slots are only drive by depth
    # * let each scaffold to also have SO(3) local frame!
    # the topology don;t have to be always have, e.g. if a point is always far away from all other points, then it shouldn't be draged by other
    # * Here also optimize the node rotation
    T, M = curve_ref.shape[:2]
    param_free_xyz = curve_ref[~d_track_mask].clone()
    param_vis_dep_corr = torch.zeros_like(dep_list[d_track_mask])
    param_q = torch.zeros(T, M, 4).to(param_free_xyz)
    param_q[..., 0] = 1.0
    param_free_xyz.requires_grad_(True)
    param_vis_dep_corr.requires_grad_(True)
    param_q.requires_grad_(True)
    if dep_corr_ref_id >= 0:
        dep_corr_unchange_mask = torch.zeros_like(dep_list).bool()
        dep_corr_unchange_mask[dep_corr_ref_id] = True
        dep_corr_unchange_mask = dep_corr_unchange_mask[d_track_mask]

    optimizer = torch.optim.Adam(
        [
            {
                "params": [param_free_xyz],
                "lr": lr_p,
            },
            {
                "params": [param_vis_dep_corr],
                "lr": lr_d,
            },
            {
                "params": [param_q],
                "lr": lr_q,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        (total_steps - decay_start),
        eta_min=min(lr_p, lr_q) / decay_factor,
    )

    # identify the topo average weight
    # * analysis the noodle topology, the topology for below optimization should only depends on the existing slots, but later when return, should re-analyze the topology
    topo_ind, topo_dist = compute_curve_topo_dist(
        curve_xyz=get_world_points(homo_list, dep_list, cams),
        K=topo_k,
        curve_mask=d_track_mask,
    )

    torch.cuda.empty_cache()
    topo_max_nn_dist = (
        compute_spatial_unit_from_nndist(topo_dist[:, 0]) * topo_dist_th_ratio
    )
    topo_w = (topo_dist < topo_max_nn_dist).float()
    valid_topo_cnt = topo_w.sum(1)
    logging.info(
        f"Topo dist th={topo_max_nn_dist:.5f} with {topo_w.mean()*100.0:.2f}% valid edge in K={topo_k}; unique stat [{valid_topo_cnt.unique(return_counts=True)}]"
    )
    topo_w = topo_w / torch.clamp(topo_w.sum(dim=-1, keepdim=True), min=1e-6)
    topo_w = topo_w.to(param_free_xyz)

    loss_list, loss_coord_list, loss_len_list = [], [], []
    loss_coord_global_list, loss_len_global_list = [], []
    loss_p_vel_list, loss_q_vel_list = [], []
    loss_p_acc_list, loss_q_acc_list = [], []
    loss_dep_corr_list, metric_dep_corr_list = [], []
    loss_xyz_std_list, metric_xyz_std_list = [], []

    vel_acc_flag = (
        lambda_xyz_vel > 0.0
        or lambda_q_vel > 0.0
        or lambda_xyz_acc > 0.0
        or lambda_q_acc > 0.0
    )

    # prepare the visible ray direction and origin
    cam_R_wi, cam_t_wi = cams.Rt_wc_list()
    vis_dep_base = dep_list[d_track_mask].detach()
    vis_ray_o = cam_t_wi[:, None].expand(-1, dep_list.shape[1], -1)[d_track_mask]
    ray_d_all = torch.cat(
        [homo_list / cams.rel_focal, torch.ones_like(homo_list[..., :1])], -1
    )
    vis_ray_d = torch.einsum("tij,tnj->tni", cam_R_wi, ray_d_all)[d_track_mask]
    vis_ray_o = vis_ray_o.detach()
    vis_ray_d = vis_ray_d.detach()

    logging.info(f"Start to init {curve_ref.shape[1]} scf...")

    for step in tqdm(range(total_steps)):
        optimizer.zero_grad()

        # compose curve in ref frame
        dep_corr = param_vis_dep_corr.clone()
        if dep_corr_ref_id >= 0:
            dep_corr[dep_corr_unchange_mask] = dep_corr[dep_corr_unchange_mask] * 0.0
        if step > open_visible_after_step:
            vis_dep = vis_dep_base.detach() + dep_corr
        else:
            vis_dep = vis_dep_base.detach() + dep_corr.detach()
        vis_xyz = vis_ray_d * vis_dep[..., None] + vis_ray_o
        xyz_all = torch.zeros_like(curve_ref)
        xyz_all[d_track_mask] = xyz_all[d_track_mask] + vis_xyz
        xyz_all[~d_track_mask] = xyz_all[~d_track_mask] + param_free_xyz

        # determine the window
        if len(xyz_all) > max_time_window:
            start = torch.randint(0, len(xyz_all) - max_time_window + 1, (1,)).item()
            window_sel = torch.arange(start, start + max_time_window).to(device)
        else:
            window_sel = torch.arange(len(xyz_all)).to(device)
        xyz = xyz_all[window_sel]
        R_wi = q2R(param_q[window_sel])  # always supervise the node R

        # ! note in scf init stage, all the arap and vel loss are SUM reduction
        local_coord = get_local_coord(xyz, topo_ind, R_wi)
        loss_coord, loss_len, loss_coord_global, loss_len_global = compute_arap(
            local_coord,
            topo_w,
            temporal_diff_shift,
            temporal_diff_weight,
            reduce="sum",
        )

        if vel_acc_flag:
            xyz_vel, ang_vel, xyz_acc, ang_acc = compute_vel_acc(xyz, R_wi)
            loss_p_vel, loss_q_vel = xyz_vel.sum(), ang_vel.sum()
            loss_p_acc, loss_q_acc = xyz_acc.sum(), ang_acc.sum()
        else:
            loss_p_vel = loss_q_vel = loss_p_acc = loss_q_acc = torch.zeros_like(
                loss_coord
            )

        loss_dep_corr = abs(param_vis_dep_corr).sum()

        xyz_mean = xyz.mean(0).detach()
        xyz_std = (xyz - xyz_mean[None]).norm(dim=-1).mean(dim=0)
        loss_xyz_std = xyz_std.sum()

        loss = (
            lambda_local_coord * loss_coord
            + lambda_metric_len * loss_len
            + lambda_global_std
            * (
                lambda_local_coord * loss_coord_global
                + lambda_metric_len * loss_len_global
            )
            + lambda_xyz_vel * loss_p_vel
            + lambda_q_vel * loss_q_vel
            + lambda_xyz_acc * loss_p_acc
            + lambda_q_acc * loss_q_acc
            + lambda_dep_corr * loss_dep_corr
            + lambda_xyz_std * loss_xyz_std
        )

        loss.backward()
        optimizer.step()

        if step > decay_start:
            scheduler.step()
        with torch.no_grad():
            loss_list.append(loss.item())
            loss_coord_list.append(loss_coord.item())
            loss_len_list.append(loss_len.item())
            loss_coord_global_list.append(loss_coord_global.item())
            loss_len_global_list.append(loss_len_global.item())
            loss_p_vel_list.append(loss_p_vel.item())
            loss_q_vel_list.append(loss_q_vel.item())
            loss_p_acc_list.append(loss_p_acc.item())
            loss_q_acc_list.append(loss_q_acc.item())
            loss_dep_corr_list.append(loss_dep_corr.item())
            metric_dep_corr_list.append(abs(param_vis_dep_corr).mean().item())
            loss_xyz_std_list.append(loss_xyz_std.item())
            metric_xyz_std_list.append(xyz_std.mean().item())

        if step % 100 == 0:
            logging.info(f"step={step}, loss={loss:.6f}")
            logging.info(
                f"loss_coord={loss_coord:.6f}, loss_len={loss_len:.6f}, loss_coord_global={loss_coord_global:.6f}, loss_len_global={loss_len_global:.6f}"
            )
        # if step % 50 == 0 or step < 30:
        if step % viz_interval == 0 and viz_interval > 0:
            viz_frame = viz_curve(
                xyz_all,
                rgb_list,
                d_track_mask,
                cams,
                # viz_n=128,
                viz_n=-1,
                res=480,
                pts_size=0.001,
                only_viz_last_frame=True,
                no_rgb_viz=False,
                text=f"Step={step}",
                time_window=cams.T,
            )
            imageio.imsave(
                osp.join(viz_dir, f"dynamic_scaffold_init_{step:06d}.jpg"),
                viz_frame[0],
            )
        if step % viz_video_interval == 0 and viz_video_interval > 0:
            viz_frame = viz_curve(
                xyz_all,
                rgb_list,
                d_track_mask,
                cams,
                viz_n=256,
                time_window=6,
                res=480,
                pts_size=0.001,
                only_viz_last_frame=False,
                no_rgb_viz=True,
                n_line=25,
                text=f"Step={step}",
            )
            imageio.mimsave(
                osp.join(viz_dir, f"dynamic_scaffold_init_{step:06d}.mp4"),
                viz_frame,
            )
            viz_frame = viz_curve(
                xyz_all,
                rgb_list,
                d_track_mask,
                cams,
                viz_n=-1,
                time_window=1,
                res=480,
                pts_size=0.001,
                only_viz_last_frame=False,
                no_rgb_viz=True,
                n_line=25,
                text=f"Step={step}",
            )
            imageio.mimsave(
                osp.join(viz_dir, f"dynamic_scaffold_init_{step:06d}_2.mp4"),
                viz_frame,
            )

    # compose final curve
    with torch.no_grad():
        final_xyz = torch.zeros_like(curve_ref)
        final_xyz[d_track_mask] = (
            vis_ray_d * (vis_dep_base + param_vis_dep_corr)[..., None] + vis_ray_o
        )
        final_xyz[~d_track_mask] = param_free_xyz
    # viz
    make_video_from_pattern(
        osp.join(viz_dir, "dynamic_scaffold_init_*.jpg"),
        osp.join(log_dir, "dynamic_scaffold_init.mp4"),
    )
    fig = plt.figure(figsize=(21, 7))
    for plt_i, plt_pack in enumerate(
        [
            ("loss", loss_list),
            ("loss_coord", loss_coord_list),
            ("loss_len", loss_len_list),
            ("loss_coord_global", loss_coord_global_list),
            ("loss_len_global", loss_len_global_list),
            ("loss_p_vel", loss_p_vel_list),
            ("loss_q_vel", loss_q_vel_list),
            ("loss_p_acc", loss_p_acc_list),
            ("loss_q_acc", loss_q_acc_list),
            ("loss_dep_corr", loss_dep_corr_list),
            ("metric_dep_corr", metric_dep_corr_list),
            ("loss_xyz_std", loss_xyz_std_list),
            ("metric_xyz_std", metric_xyz_std_list),
        ]
    ):
        plt.subplot(2, 8, plt_i + 1)
        plt.plot(plt_pack[1]), plt.title(plt_pack[0])
    plt.tight_layout()
    plt.savefig(osp.join(log_dir, f"dynamic_scaffold_init.jpg"))
    plt.close()

    ret_dep_corr = torch.zeros_like(dep_list)
    ret_dep_corr[d_track_mask] = param_vis_dep_corr.detach().clone()

    torch.save(
        {
            "param_xyz": final_xyz.detach().clone(),
            "param_q": param_q.detach().clone(),
            "d_track": d_track,
            "d_track_mask": d_track_mask,
            "d_dep_corr": ret_dep_corr,
        },
        osp.join(log_dir, "dynamic_scaffold.pth"),
    )

    # * viz before start
    if viz_debug:
        viz_frame = viz_curve(
            final_xyz,
            rgb_list,
            d_track_mask,
            cams,
            res=480,
            pts_size=0.001,
            only_viz_last_frame=False,
            no_rgb_viz=True,
            viz_n=-1,
            text=f"Before init",
        )
        imageio.mimsave(
            osp.join(viz_dir, f"DEBUG_after_dynamic_scaffold_init.mp4"),
            viz_frame,
        )

    logging.info("Dynamic Scaffold Done!")

    return d_track, d_track_mask, final_xyz, param_q, ret_dep_corr


def compute_vel_acc(xyz, R_wi):
    xyz_vel = (xyz[1:] - xyz[:-1]).norm(dim=-1)
    xyz_acc = (xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]).norm(dim=-1)

    delta_R = torch.einsum("tnij,tnkj->tnik", R_wi[1:], R_wi[:-1])
    ang_vel = matrix_to_axis_angle(delta_R).norm(dim=-1)
    ang_acc_mag = abs(ang_vel[1:] - ang_vel[:-1])
    return xyz_vel, ang_vel, xyz_acc, ang_acc_mag


def compute_vel_acc_loss(self, tids=None, detach_mask=None):
    if tids is None:
        tids = torch.arange(self.T).to(self.device)
    assert tids.max() <= self.T - 1
    xyz = self._node_xyz[tids]
    R_wi = q2R(self._node_rotation[tids])
    if detach_mask is not None:
        detach_mask = detach_mask.float()[:, None, None]
        xyz = xyz.detach() * detach_mask + xyz * (1 - detach_mask)
        R_wi = (
            R_wi.detach() * detach_mask[..., None] + R_wi * (1 - detach_mask)[..., None]
        )

    xyz_vel, ang_vel, xyz_acc, ang_acc = compute_vel_acc(xyz, R_wi)

    loss_p_vel, loss_q_vel = xyz_vel.mean(), ang_vel.mean()
    loss_p_acc, loss_q_acc = xyz_acc.mean(), ang_acc.mean()
    return loss_p_vel, loss_q_vel, loss_p_acc, loss_q_acc


def compute_arap(
    local_coord,
    topo_w,
    temporal_diff_shift,
    temporal_diff_weight,
    reduce="mean",
):
    local_coord_len = local_coord.norm(dim=-1, p=2)  # T,N,K
    # the coordinate should be similar
    # the metric should be similar
    loss_coord, loss_len = 0.0, 0.0
    for shift, _w in zip(temporal_diff_shift, temporal_diff_weight):
        diff_coord = (local_coord[:-shift] - local_coord[shift:]).norm(dim=-1)
        diff_len = (local_coord_len[:-shift] - local_coord_len[shift:]).abs()
        diff_coord = (diff_coord * topo_w[None]).sum(-1)
        diff_len = (diff_len * topo_w[None]).sum(-1)
        # ! fixed this bug on 2024.3.17
        if reduce == "sum":
            loss_coord = loss_coord + _w * diff_coord.sum()
            loss_len = loss_len + _w * diff_len.sum()
        elif reduce == "mean":
            loss_coord = loss_coord + _w * diff_coord.mean()
            loss_len = loss_len + _w * diff_len.mean()
        else:
            raise NotImplementedError()
    loss_coord_global = (local_coord.std(0) * topo_w[..., None]).sum()
    loss_len_global = (local_coord_len.std(0) * topo_w).sum()
    return loss_coord, loss_len, loss_coord_global, loss_len_global


@torch.no_grad()
def compute_curve_topo_dist(
    curve_xyz, K, curve_mask=None, chunk=6 * 4096, max_N=4096, return_dist=False
):
    # TODO: can boost by compute a half, because the dist is symmetric
    # curve_xyz: T,N,3
    T, N = curve_xyz.shape[:2]
    t_chunk = max(1, chunk // N)
    cur = 0
    dist = torch.zeros(N, N).to(curve_xyz)
    while cur < T:
        chunk_p = curve_xyz[cur : cur + t_chunk].half()
        # T,N
        # may do block-wise to avoid oom
        # ! tmp solution here
        _A = chunk_p[:, None].expand(-1, N, -1, -1)
        _B = chunk_p[:, :, None].expand(-1, -1, N, -1)
        block = 0
        _D = []
        while block < N:
            _D.append(
                (_A[:, block : block + max_N] - _B[:, block : block + max_N]).norm(
                    dim=-1
                )
            )
            block += max_N
        _D = torch.cat(_D, dim=1)
        # _D = (chunk_p[:, None] - chunk_p[:, :, None]).norm(dim=-1)
        if curve_mask is not None:
            chunk_m = curve_mask[cur : cur + t_chunk]
            _M = chunk_m[:, None] * chunk_m[:, :, None]
            _D[~_M] = 0
        _cur_dist = _D.max(dim=0).values
        dist = torch.max(_cur_dist, dist)
        cur = cur + t_chunk
    invalid_mask = dist == 0
    dist[invalid_mask] = 1e10
    knn_dist, knn_ind = torch.topk(dist, K, dim=-1, largest=False)
    if return_dist:
        return knn_ind, knn_dist, dist
    else:
        return knn_ind, knn_dist


def get_local_coord(xyz, topo_ind, R_wi):
    nn_xyz = xyz[:, topo_ind, :]
    nn_R_wi = R_wi[:, topo_ind, :]
    self_xyz = xyz[:, :, None]
    local_coord = torch.einsum(
        "tnkji,tnkj->tnki", nn_R_wi, self_xyz - nn_xyz
    )  # T,N,K,3
    return local_coord


@torch.no_grad()
def resample_scf(scf_xyz, scf_buffer, scf_mask, spatial_unit):
    device = scf_xyz.device
    T, N = scf_xyz.shape[:2]

    # scan the scf in sorted order, if the curve is not covered by the unit, add it, otherwise skip it
    scf_solid_cnt = scf_mask.sum(0)
    scan_inds = torch.argsort(scf_solid_cnt, descending=True)

    ret_xyz = scf_xyz[:, scan_inds[:1]].clone()
    ret_buffer = scf_buffer[:, scan_inds[:1]].clone()
    ret_mask = scf_mask[:, scan_inds[:1]].clone()
    original_inds = [scan_inds[:1]]

    _, _, D = compute_curve_topo_dist(scf_xyz, K=1, return_dist=True)

    # compute
    for ind in tqdm(scan_inds[1:]):
        _d = D[ind]
        _d_to_set = _d[torch.cat(original_inds)]
        covered = _d_to_set.min() < spatial_unit
        if not covered:
            # append
            original_inds.append(ind[None])
            ret_xyz = torch.cat([ret_xyz, scf_xyz[:, ind[None]]], 1)
            ret_buffer = torch.cat([ret_buffer, scf_buffer[:, ind[None]]], 1)
            ret_mask = torch.cat([ret_mask, scf_mask[:, ind[None]]], 1)
    logging.info(
        f"SCF Resample with th={spatial_unit:.4f} N={ret_xyz.shape[1]} out of {N} ({ret_xyz.shape[1]/N * 100.0:.2f}%)"
    )
    return ret_xyz, ret_buffer, ret_mask
