# Single File
from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
from tqdm import tqdm
import logging, imageio

sys.path.append(osp.dirname(osp.abspath(__file__)))
from prior2d import Prior2D
from fov_helper import fov_graph_init
from projection_helper import project, fovdeg2focal, backproject
from solver_viz_helper import make_video_from_pattern, viz_global_ba
from camera import SimpleFovCamerasDelta, SimpleFovCamerasIndependent
from solver_utils import prepare_track_buffers, get_world_points
import wandb
import yaml


def compute_static_ba(
    log_dir,
    prior2d: Prior2D,
    cams: SimpleFovCamerasDelta,
    max_t_per_step=32,
    total_steps=2000,  # 6000
    switch_to_ind_step=1000,  # this is also the scheduler start!
    depth_correction_after_n_ratio=0.5,
    lr_cam_q=0.0003,
    lr_cam_t=0.0003,
    lr_cam_f=0.0003,
    lr_dep_s=0.001,
    lr_dep_c=0.001,
    lambda_flow=1.0,
    lambda_depth=0.1,
    lambda_small_correction=0.03,
    fov_search_min_interval=2,
    fov_interval_single=True,
    viz_verbose_n=300,
    viz_fig_n=300,
    # viz_denser_range=[[0, 40], [1000, 1040]],
    viz_denser_range=[],
    viz_denser_interval=10,
    gt_cam_flag=False,
    fov_search_flag=True,
    max_num_of_tracks=10000,
    viz_additional_flag=False,
    # 
    robust_alignment_jitter_th_ratio=2.0
):

    viz_dir = osp.join(log_dir, "viz_step/static_ba_scf")
    os.makedirs(viz_dir, exist_ok=True)

    # * prepare noodles
    s_track = prior2d.track[:, prior2d.track_static_mask]
    s_track_mask = prior2d.track_mask[:, prior2d.track_static_mask]
    device = s_track.device

    # * filter the dense tracks
    if max_num_of_tracks < s_track.shape[1]:
        logging.info(
            f"Track is too dense {s_track.shape[1]}, radom sample to {max_num_of_tracks}"
        )
        choice = torch.randperm(s_track.shape[1])[:max_num_of_tracks]
        s_track = s_track[:, choice]
        s_track_mask = s_track_mask[:, choice]

    homo_list, dep_list, _, rgb_list = prepare_track_buffers(
        prior2d, s_track, s_track_mask, torch.arange(prior2d.T).to(device)
    )

    if gt_cam_flag:
        # logging.info("Static BA use GT camera, only update the video depth")
        # ! can also update camera pose

        lr_cam_f = 0.0
        # lr_cam_q, lr_cam_t = 0.0, 0.0
        logging.warning(
            f"Use GT cam, fix the focal lenght, but still optimize the camera pose!"
        )
        with torch.no_grad():

            # ! this optimal scale should be robust!!

            # mask out large depth jitter
            neighbor_frame_mask = s_track_mask[1:] * s_track_mask[:-1]
            depth_diff = abs(dep_list[1:] - dep_list[:-1])
            large_depth_jitter_th = depth_diff[neighbor_frame_mask].median()
            jitter_mask = (depth_diff > large_depth_jitter_th * robust_alignment_jitter_th_ratio) * neighbor_frame_mask
            logging.info(
                f"When solving optimal cam scale, ignore {jitter_mask.sum() / neighbor_frame_mask.sum()*100.0:.2f}% potential jitters"
            )
            neighbor_frame_mask = neighbor_frame_mask * (~jitter_mask)

            T, M = homo_list.shape[:2]
            point_cam = backproject(
                homo_list.reshape(-1, 2),
                dep_list.reshape(-1),
                cams,
            )
            point_cam = point_cam.reshape(T, M, 3)
            R_wc, t_wc = cams.Rt_wc_list()
            point_ref_rot = torch.einsum("tij,tmj->tmi", R_wc, point_cam)
            point_ref = point_ref_rot + t_wc[:, None]
            # the optimal scale has a closed form solution from a quadratic form

            # we only consider the neighboring two frames for now!
            a = point_ref_rot[1:] - point_ref_rot[:-1]
            b = (t_wc[1:] - t_wc[:-1])[:, None].expand(-1, M, -1)
            a, b = a[neighbor_frame_mask], b[neighbor_frame_mask]
            # should be masked
            s_optimal = float(-(a * b).sum() / (b * b).sum())
            # avoid singular case
            # if s_optimal < 0.01:
            if s_optimal < 0.00001:
                logging.warning(
                    f"optimal rescaling of gt camera translation degenerate to {s_optimal}, use 1.0 instead!"
                )
                s_optimal = 1.0
            logging.info(
                f"Rescale the GT camera pose ot our depth scale (median=1) with a global scale factor {s_optimal} by closed form solution"
            )
            # * manually rescale the translation
            cams.t_wc.data = cams.t_wc.data * s_optimal

    elif fov_search_flag:
        optimal_fov = fov_graph_init(
            s_track_mask,
            homo_list,
            dep_list,
            viz_fn=osp.join(log_dir, "fov_search.jpg"),
            fallback_fov=cams.fov,
            search_N=100,
            search_start=20.0,
            search_end=70.0,
            min_interval=fov_search_min_interval,
            early_break=fov_interval_single,
        )
        with torch.no_grad():
            cams.rel_focal.data = torch.ones_like(cams.rel_focal) * fovdeg2focal(
                optimal_fov
            )

    # * start solve global init of the camera
    depth_correction_after_n = int(total_steps * depth_correction_after_n_ratio)

    logging.info(
        f"Static Scaffold BA: Depth correction after {depth_correction_after_n}; Lr Scheduling and Ind after {switch_to_ind_step} steps (total {total_steps})"
    )
    param_scale = torch.ones(cams.T).to(device)
    param_scale.requires_grad_(True)
    param_dep_corr = torch.zeros_like(dep_list).clone()
    param_dep_corr.requires_grad_(True)
    optimizer = torch.optim.Adam(
        cams.get_optimizable_list(lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t)
        + [{"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}]
        + [{"params": [param_dep_corr], "lr": lr_dep_c, "name": "dep_correction"}]
    )
    scheduler = None
    s_track_mask_w = s_track_mask.float()
    s_track_mask_w = s_track_mask_w / s_track_mask_w.sum(0)
    # huber_loss = nn.HuberLoss(reduction="none", delta=0.5)

    loss_list, std_list, fov_list = [], [], []
    flow_loss_list, dep_loss_list, dep_corr_loss_list = [], [], []

    logging.info(f"Start Static BA with {cams.T} frames and {dep_list.shape[1]} points")

    for step in tqdm(range(total_steps)):
        if step == switch_to_ind_step and not isinstance(
            cams, SimpleFovCamerasIndependent
        ):
            logging.info(
                "Switch to Independent Camera Optimization and Start Scheduling"
            )
            cams = SimpleFovCamerasIndependent(cam_delta=cams).to(device)
            optimizer = torch.optim.Adam(
                cams.get_optimizable_list(lr_f=lr_cam_f, lr_q=lr_cam_q, lr_t=lr_cam_t)
                + [{"params": [param_scale], "lr": lr_dep_s, "name": "cam_scale"}]
                + [
                    {
                        "params": [param_dep_corr],
                        "lr": lr_dep_c,
                        "name": "dep_correction",
                    }
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                total_steps - switch_to_ind_step,
                eta_min=min(lr_cam_q, lr_cam_t) / 10.0,
            )

        optimizer.zero_grad()

        ########################
        # * after a while, fuse all point to one point and continue the optimize globally
        # * Or after a while, still based on depth, and modify the depth a little bit
        # * in both way, we have to use this key point information to deform the original depth to form a clean model
        ########################

        ########################
        # dep_scale = param_scale.abs() / (param_scale.abs().mean() + 1e-6)
        dep_scale = param_scale.abs()  # ! try direct no manual handling!
        scaled_depth_list = dep_list * dep_scale[:, None]
        if step > depth_correction_after_n:
            scaled_depth_list = scaled_depth_list + param_dep_corr
            dep_corr_loss = abs(param_dep_corr).mean()
        else:
            dep_corr_loss = torch.zeros_like(dep_scale[0])
        point_ref = get_world_points(homo_list, scaled_depth_list, cams)  # T,N,3

        # transform to each frame!
        if cams.T > max_t_per_step:
            tgt_inds = torch.randperm(cams.T)[:max_t_per_step].to(device)
        else:
            tgt_inds = torch.arange(cams.T).to(device)
        R_cw, t_cw = cams.Rt_cw_list()
        R_cw, t_cw = R_cw[tgt_inds], t_cw[tgt_inds]

        point_ref_to_every_frame = (
            torch.einsum("tij,snj->stni", R_cw, point_ref) + t_cw[None, :, None]
        )  # Src,Tgt,N,3
        uv_src_to_every_frame = project(point_ref_to_every_frame, cams)  # Src,Tgt,N,3
        # because projection of a too close point will create numerical instability, handle this
        projection_singular_mask = abs(point_ref_to_every_frame[..., -1]) < 1e-5
        if projection_singular_mask.any():
            logging.warning(
                f"Projection of very close point happened, mask out this to avoid singularity {projection_singular_mask.sum()}"
            )
        # no matter where the src is, it should be mapped to every frame with the gt tracking
        cross_time_mask = (s_track_mask[:, None] * s_track_mask[None, tgt_inds]).float()
        cross_time_mask = cross_time_mask * (~projection_singular_mask).float()
        uv_target = homo_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1, -1
        )
        uv_loss_i = (uv_src_to_every_frame - uv_target).norm(dim=-1)

        uv_loss = (uv_loss_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)
        # compute depth loss
        dep_target = scaled_depth_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1
        )
        warped_depth = point_ref_to_every_frame[..., -1]
        # dep_consistency_i = abs(dep_target - warped_depth)
        dep_consistency_i = 0.5 * abs(
            dep_target / torch.clamp(warped_depth, min=1e-6) - 1
        ) + 0.5 * abs(warped_depth / torch.clamp(dep_target, min=1e-6) - 1)
        dep_loss = (dep_consistency_i * cross_time_mask).sum() / (
            cross_time_mask.sum() + 1e-6
        )
        # TODO: do we need feature loss in static one? - Hui
        loss = (
            lambda_depth * dep_loss
            + lambda_flow * uv_loss
            + lambda_small_correction * dep_corr_loss
        )
        # logging.info(
        #     f"loss={loss:.6f}, uv_loss={uv_loss:.6f}, dep_loss={dep_loss:.6f}, dep_corr_loss={dep_corr_loss:.6f}, rot_reg_loss={rot_reg_loss}"
        # )
        assert torch.isnan(loss).sum() == 0 and torch.isinf(loss).sum() == 0
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # viz
        with torch.no_grad():
            point_ref_mean = (point_ref * s_track_mask_w[:, :, None]).sum(0)
            std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            metric_std = (std * s_track_mask_w).sum(0).mean()
            loss_list.append(loss.item())
            dep_corr_loss_list.append(dep_corr_loss.item())
            flow_loss_list.append(uv_loss.item())
            dep_loss_list.append(dep_loss.item())
            std_list.append(metric_std.item())
            fov_list.append(cams.fov.item())
            if step % viz_verbose_n == 0 or step == total_steps - 1:
                logging.info(f"loss={loss:.6f}, fov={cams.fov:.4f}")
                logging.info(f"scale max={param_scale.max()} min={param_scale.min()}")

            wandb.log(
                {
                    "static_scf/loss": loss.item(),
                    "static_scf/flow_loss": uv_loss.item(),
                    "static_scf/dep_loss": dep_loss.item(),
                    "static_scf/dep_corr_loss": dep_corr_loss.item(),
                    "static_scf/std": metric_std.item(),
                    "static_scf/fov": cams.fov.item(),
                },
            )

            viz_flag = (
                np.array([step >= r[0] and step <= r[1] for r in viz_denser_range])
                .any()
                .item()
            )
            viz_flag = viz_flag and step % viz_denser_interval == 0
            viz_flag = viz_flag or step % viz_fig_n == 0 or step == total_steps - 1
            if viz_flag:
                # viz the 3D aggregation as well as the pcl in 3D!
                NUM_SEMANTIC_CHANNELS = prior2d.latent_feature_channel
                semantic_feature_list = torch.zeros([rgb_list.shape[0], rgb_list.shape[1], NUM_SEMANTIC_CHANNELS]).to(rgb_list)
                viz_frame = viz_global_ba(
                    point_ref,
                    rgb_list,
                    semantic_feature_list,
                    s_track_mask,
                    cams,
                    error=std,
                    text=f"Step={step}",
                )
                imageio.imsave(
                    osp.join(viz_dir, f"static_scaffold_init_{step:06d}.jpg"),
                    viz_frame,
                )

    # viz
    make_video_from_pattern(
        osp.join(viz_dir, "static_scaffold_init_*.jpg"),
        osp.join(log_dir, "static_scaffold_init.mp4"),
    )

    if total_steps > 0:
        fig = plt.figure(figsize=(18, 3))
        for plt_i, plt_pack in enumerate(
            [
                ("loss", loss_list),
                ("loss_flow", flow_loss_list),
                ("loss_dep", dep_loss_list),
                ("loss_dep_corr", dep_corr_loss_list),
                ("std", std_list),
                ("fov", fov_list),
            ]
        ):
            plt.subplot(1, 6, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
            if plt_pack[0].startswith("loss"):
                plt.yscale("log")
        plt.tight_layout()
        plt.savefig(osp.join(log_dir, f"static_scaffold_init.jpg"))
        plt.close()

        # save the fused and optimized xyz
        if viz_additional_flag:
            np.savetxt(
                osp.join(log_dir, "static_scaffold_fused.xyz"),
                point_ref_mean.detach().cpu().numpy(),
                fmt="%.6f",
            )
            np.savetxt(
                osp.join(log_dir, "static_scaffold_unfused.xyz"),
                point_ref.reshape(-1, 3).detach().cpu().numpy(),
                fmt="%.6f",
            )

    # update the depth scale
    dep_scale = param_scale.abs()
    prior2d.rescale_depth(dep_scale)
    logging.info(
        "Static Scaffold Done, the depth stored in [Prior2D] is re-scaled and the camera parameters in [cams] are updated"
    )
    # save
    torch.save(cams.state_dict(), osp.join(log_dir, "static_scaffold_cam.pth"))
    torch.save(
        {
            # sol
            "dep_scale": dep_scale,
            "dep_correction": param_dep_corr,
            "s_track": s_track,
            "s_track_mask": s_track_mask,
        },
        osp.join(log_dir, "static_scaffold.pth"),
    )
    return cams, s_track, s_track_mask, param_dep_corr.detach().clone()
