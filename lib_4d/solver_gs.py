from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
from tqdm import tqdm

from lib_4d.autoencoder.model import Feature_heads

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
import kornia
from prior2d import Prior2D
from prior2d_utils import viz_mask_video
from render_helper import render, GS_BACKEND
from matplotlib import cm
import cv2 as cv
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops import knn_points
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio  # viz purpose
#from lib_4d.motion_utils.init_utils import init_motion_params_with_procrustes

##################################################################
from save_helpers import save_gauspl_ply
from index_helper import (
    query_image_buffer_by_pix_int_coord,
    round_int_coordinates,
)
from loss_helper import (
    compute_rgb_loss,
    compute_semantic_feature_loss,
    compute_dep_loss,
    compute_normal_loss,
    compute_dep_reg_loss,
    compute_normal_reg_loss,
)
from solver_viz_helper import (
    viz3d_total_video,
    viz2d_total_video,
    make_video_from_pattern,
    viz2d_one_frame,
    viz_hist,
    viz_dyn_hist,
    viz_curve,
    viz_plt_missing_slot,
    make_viz_np,
    viz_mv_model_frame,
    viz_scf_frame,
    viz_d_model_grad,
)
from lib_4d_misc import *
from gs_optim_helpers import update_learning_rate

from gs_static_model import StaticGaussian
from gs_ed_model import DynSCFGaussian
from scf4d_model import Scaffold4D

from dynwarp_helper import align_to_model_depth
from camera import SimpleFovCamerasDelta, SimpleFovCamerasIndependent
from view_sampler import RandomSampler
from cfg_helpers import OptimCFG, GSControlCFG
from lib_prior.tracking.cotracker_wrapper import Visualizer
from solver_utils import (
    get_world_points,
    prepare_track_buffers,
    detect_sharp_changes_in_curve,
    line_segment_init,
    apply_gs_control,
    fetch_leaves_in_world_frame,
)
from solver_static_funcs import compute_static_ba
from campose_alignment import align_ate_c2b_use_a2b
from projection_helper import backproject, project
from autoencoder.model import Autoencoder
import yaml
import wandb

class Solver:
    def __init__(
        self,
        working_dir,
        output_dir,
        device,
        seed=12345,
        radius_init_factor=0.5,
        opacity_init_factor=0.9,
        log_prefix="",
        # * for arap reg
        temporal_diff_shift=[2, 8, 16, 32],
        temporal_diff_weight=[0.4, 0.3, 0.2, 0.1],
    ) -> None:
        self.seed = seed
        seed_everything(self.seed)
        self.working_dir = osp.abspath(working_dir)
        self.output_dir = osp.abspath(output_dir)
        self.device = device

        # config log
        # Use simple output directory without long prefix and timestamp
        self.log_dir = self.output_dir
        self.viz_dir = osp.join(self.log_dir, "viz_step")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = create_log(self.log_dir, debug=False)

        self.radius_init_factor = radius_init_factor
        self.opacity_init_factor = opacity_init_factor

        self.track_visualizer = Visualizer(
            save_dir=self.viz_dir, linewidth=2, mode="rainbow", draw_invisible=False
        )

        # arap cfg
        self.temporal_diff_shift = temporal_diff_shift
        self.temporal_diff_weight = temporal_diff_weight

        # dummy
        self.prior2d: Prior2D = None
        return

    @property
    def H(self):
        return self.prior2d.H

    @property
    def W(self):
        return self.prior2d.W

    @property
    def T(self):
        return self.prior2d.T

    @torch.no_grad()
    def get_cams(
        self, fovdeg, model_fn=None, gt_pose=None, gt_fovdeg=None, cxcy_ratio=None
    ):
        if gt_fovdeg is not None:
            assert gt_pose is not None
            cams: SimpleFovCamerasIndependent = SimpleFovCamerasIndependent(
                self.prior2d.T, gt_fovdeg, gt_pose, cxcy_ratio=cxcy_ratio
            )
        else:
            cams: SimpleFovCamerasDelta = SimpleFovCamerasDelta(
                self.prior2d.T, fovdeg, cxcy_ratio=cxcy_ratio
            )
        if model_fn is not None:
            cams.load_state_dict(torch.load(model_fn), strict=True)
        cams.to(self.device)
        return cams

    @torch.no_grad()
    def load_static_scaffold(self, s_scf_saved_dir):
        logging.info(f"Load static scaffold from {s_scf_saved_dir}")
        cams = SimpleFovCamerasIndependent(self.T, fovdeg_init=40.0)
        cams.to(self.device)
        saved_cam = torch.load(osp.join(s_scf_saved_dir, "static_scaffold_cam.pth"))
        cams.load_state_dict(saved_cam, strict=True)
        saved_static_scaffold_data = torch.load(
            osp.join(s_scf_saved_dir, "static_scaffold.pth")
        )
        self.prior2d.rescale_depth(
            saved_static_scaffold_data["dep_scale"].to(self.device)
        )
        s_dep_corr = saved_static_scaffold_data["dep_correction"].to(self.device)
        s_track = saved_static_scaffold_data["s_track"].to(self.device)
        s_track_mask = saved_static_scaffold_data["s_track_mask"].to(self.device)
        return cams, s_track, s_track_mask, s_dep_corr

    def compute_static_scaffold(
        self,
        cams: SimpleFovCamerasDelta,
        max_t_per_step=32,
        total_steps=2000,  # ! 6000
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
        viz_verbose_n=200,
        viz_fig_n=300,
        viz_denser_range=[[0, 40], [1000, 1040]],
        viz_denser_interval=10,
        gt_cam_flag=False,
        fov_search_flag=True,
    ):
        return compute_static_ba(
            log_dir=self.log_dir,
            prior2d=self.prior2d,
            cams=cams,
            max_t_per_step=max_t_per_step,
            total_steps=total_steps,
            switch_to_ind_step=switch_to_ind_step,
            depth_correction_after_n_ratio=depth_correction_after_n_ratio,
            lr_cam_q=lr_cam_q,
            lr_cam_t=lr_cam_t,
            lr_cam_f=lr_cam_f,
            lr_dep_s=lr_dep_s,
            lr_dep_c=lr_dep_c,
            lambda_flow=lambda_flow,
            lambda_depth=lambda_depth,
            lambda_small_correction=lambda_small_correction,
            fov_search_min_interval=fov_search_min_interval,
            fov_interval_single=fov_interval_single,
            viz_verbose_n=viz_verbose_n,
            viz_fig_n=viz_fig_n,
            viz_denser_range=viz_denser_range,
            viz_denser_interval=viz_denser_interval,
            gt_cam_flag=gt_cam_flag,
            fov_search_flag=fov_search_flag,
        )

    def interpolate_by_depth_map(
        self,
        cams,
        track_uv_list,
        track_mask_list,
        src_buffer,
        viz_flag=True,
        mask2d_type="sta_dep",
        K=16,
    ):
        logging.info(
            "Interpolating depth prior with Static Scaffold (Both FG and BG are deformed) ..."
        )
        # ! this works for both static and dynamic components!
        # depth_list: T,H,W; track_uv_list: T,N,2; track_mask_list: T,N; src_buffer: T,N,C
        assert (
            len(track_mask_list)
            == len(track_uv_list)
            == len(src_buffer)
            == self.prior2d.T
        )
        prior2d: Prior2D = self.prior2d
        dep_corr_map_list, dep_new_map_list = [], []
        for tid in tqdm(range(self.prior2d.T)):
            mask2d = prior2d.get_mask_by_key(mask2d_type, tid)
            scf_mask = track_mask_list[tid]
            dep_map = prior2d.get_depth(tid)
            scf_uv = track_uv_list[tid][scf_mask]
            scf_int_uv, scf_inside_mask = round_int_coordinates(
                scf_uv, prior2d.H, prior2d.W
            )
            if not scf_inside_mask.all():
                logging.warning(
                    f"Warning, {(~scf_inside_mask).sum()} invalid uv in t={tid}! may due to round accuracy"
                )

            scf_dep = query_image_buffer_by_pix_int_coord(
                prior2d.get_depth(tid), scf_int_uv
            )
            scf_homo = query_image_buffer_by_pix_int_coord(prior2d.homo_map, scf_int_uv)
            # this pts is used to distribute the carrying interp_src in 3D cam frame
            scf_cam_pts = backproject(scf_homo, scf_dep, cams)
            dst_cam_pts = backproject(prior2d.homo_map[mask2d], dep_map[mask2d], cams)
            scf_buffer = src_buffer[tid][scf_mask]

            interp_dep_corr = spatial_interpolation(
                src_xyz=scf_cam_pts, src_buffer=scf_buffer, query_xyz=dst_cam_pts, K=K
            )

            # viz
            dep_corr_map = torch.zeros_like(dep_map)
            dep_corr_map[mask2d] = interp_dep_corr.squeeze(-1)
            scf_corr_interp = query_image_buffer_by_pix_int_coord(
                dep_corr_map, scf_int_uv
            )
            check_interp_error = (
                abs(scf_corr_interp - scf_buffer.squeeze(-1)).median()
                / abs(scf_buffer).median()
            )
            # logging.info(
            #     f"Interp {tid} with scf relative interp error {check_interp_error*100.0:.3f}%"
            # )
            dep_corr_map_list.append(dep_corr_map.detach())
            dep_new_map_list.append((dep_corr_map + dep_map).detach())
            # check the interp error of src points

        if viz_flag:
            # viz the correction and corrected depth map
            viz_corr_list, viz_dep_list = [], []
            for tid in tqdm(range(0, self.prior2d.T, 5)):
                viz_corr = dep_corr_map_list[tid]
                viz_dep = dep_new_map_list[tid]
                viz_corr_radius = abs(viz_corr).max()
                viz_corr = (viz_corr / viz_corr_radius) + 0.5
                viz_dep = (viz_dep - viz_dep.min()) / (viz_dep.max() - viz_dep.min())
                viz_corr = cm.viridis(viz_corr.cpu().numpy())
                viz_dep = cm.viridis(viz_dep.cpu().numpy())
                viz_corr = (viz_corr * 255).astype(np.uint8)
                viz_dep = (viz_dep * 255).astype(np.uint8)
                viz_corr_list.append(viz_corr)
                viz_dep_list.append(viz_dep)
            imageio.mimsave(
                osp.join(self.viz_dir, f"interp_corr_mask={mask2d_type}.mp4"),
                viz_corr_list,
                fps=10,
            )
            imageio.mimsave(
                osp.join(self.viz_dir, f"interp_dep_mask={mask2d_type}.mp4"),
                viz_dep_list,
                fps=10,
            )

        dep_new_map_list = torch.stack(dep_new_map_list, 0)
        self.prior2d.__reset_depth_all__(dep_new_map_list)
        return

    @torch.no_grad()
    def recompute_dynamic_masks_and_tracks(
        self,
        s_model: StaticGaussian,
        cams: SimpleFovCamerasIndependent,
        viz_flag=True,  # ! debug
        dyn_color_error_th=0.1,
        dyn_mask_open_ksize=5,
        dyn_track_min_valid=3,
        consider_inside_dyn=True,  # ! debug
    ):
        # ! this function has an issue, can't handle the move + static case! will mark all invalid when land in static zone, this is not true for a dynamic object that is stopped!!
        prior2d: Prior2D = self.prior2d
        logging.info(
            f"Recompute dyn mask with long track filtering cnt-th={dyn_track_min_valid}"
        )
        # todo: check the th
        logging.warning(f"above may be too agressive!!")

        # * Conservatively make sure that the fg is fg, similar to the initial static mask
        # ! This mask can be smaller because we can aggregate across time to gather the leaves
        bg_pred_list, bg_gt_list, bg_error_list = [], [], []
        for tid in range(cams.T):
            render_dict = render(
                [s_model()],
                prior2d.H,
                prior2d.W,
                cams.rel_focal,
                cams.cxcy_ratio,
                cams.T_cw(tid),
            )
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            gt_rgb = prior2d.get_rgb(tid)
            error = abs(pred_rgb - gt_rgb).max(-1).values
            bg_error_list.append(error.detach().cpu())
            bg_pred_list.append(pred_rgb.detach().cpu())
            bg_gt_list.append(gt_rgb.detach().cpu())
        bg_pred_list = torch.stack(bg_pred_list, 0)
        bg_gt_list = torch.stack(bg_gt_list, 0)
        bg_error_list = torch.stack(bg_error_list, 0)
        fg_mask = bg_error_list > dyn_color_error_th
        open_kernel = torch.ones(dyn_mask_open_ksize, dyn_mask_open_ksize)
        filtered_fg_mask = kornia.morphology.opening(
            fg_mask[:, None].float(), kernel=open_kernel.to(fg_mask.device)
        ).squeeze(1)
        filtered_fg_mask = filtered_fg_mask.bool()
        if consider_inside_dyn:
            filtered_fg_mask = (
                filtered_fg_mask * prior2d.dynamic_masks.to(filtered_fg_mask).bool()
            )

        # update prior2d
        prior2d.update_mask(filtered_fg_mask, "dynamic")
        if viz_flag:
            # also save the pred images and errors
            imageio.mimsave(
                osp.join(self.viz_dir, "recomputed_pred_bg.mp4"),
                (bg_pred_list.cpu()).float().numpy(),
            )
            imageio.mimsave(
                osp.join(self.viz_dir, "recomputed_noisy_fg_mask.mp4"),
                (fg_mask[..., None] * prior2d.rgbs.cpu()).float().numpy(),
            )
            imageio.mimsave(
                osp.join(self.viz_dir, "recomputed_fg_mask.mp4"),
                (filtered_fg_mask[..., None] * prior2d.rgbs.cpu()).float().numpy(),
            )

        ##################################################
        d_track = prior2d.track[:, prior2d.track_dynamic_mask]
        d_track_mask = prior2d.track_mask[:, prior2d.track_dynamic_mask]
        d_track_mask_fg_stat = d_track_mask.clone()
        if prior2d.use_short_track:
            short_d_track = prior2d.short_track[:, prior2d.short_track_dynamic_mask]
            short_d_track_mask = prior2d.short_track_mask[
                :, prior2d.short_track_dynamic_mask
            ]
            short_d_track_mask_fg_stat = short_d_track_mask.clone()

        for tid in range(cams.T):
            _uv = d_track[tid]
            _int_uv, _inside_mask = round_int_coordinates(_uv, prior2d.H, prior2d.W)
            _fg = query_image_buffer_by_pix_int_coord(
                filtered_fg_mask[tid].to(self.device), _int_uv
            )
            d_track_mask_fg_stat[tid, _inside_mask] = (
                d_track_mask_fg_stat[tid, _inside_mask] * _fg[_inside_mask]
            )
            if prior2d.use_short_track:
                _uv = short_d_track[tid]
                _int_uv, _inside_mask = round_int_coordinates(_uv, prior2d.H, prior2d.W)
                _fg = query_image_buffer_by_pix_int_coord(
                    filtered_fg_mask[tid].to(self.device), _int_uv
                )
                short_d_track_mask_fg_stat[tid, _inside_mask] = (
                    short_d_track_mask_fg_stat[tid, _inside_mask] * _fg[_inside_mask]
                )

        # update the valid masks and dynamic track mask in prior2d
        # ! don't update the track solid mask!!
        # prior2d.track_mask[:, prior2d.track_dynamic_mask] = d_track_mask_updated
        d_track_valid_cnt = d_track_mask_fg_stat.sum(0)
        d_track_valid_mask = d_track_valid_cnt > dyn_track_min_valid
        logging.info(
            f"Update dynamic track mask, prev has {d_track_mask.shape[1]} valid tracks, now has {d_track_valid_mask.sum()} valid tracks (dyn_track_min_valid={dyn_track_min_valid})"
        )
        track_dynamic_mask = prior2d.track_dynamic_mask.clone()
        track_dynamic_mask[track_dynamic_mask.clone()] = d_track_valid_mask
        prior2d.track_dynamic_mask = (
            track_dynamic_mask.clone() * prior2d.hard_dyn_marker
        )

        if prior2d.use_short_track:
            prior2d.short_track_mask[:, prior2d.short_track_dynamic_mask] = (
                short_d_track_mask_fg_stat
            )
            short_d_track_valid_cnt = short_d_track_mask_fg_stat.sum(0)
            # short_d_track_valid_mask = short_d_track_valid_cnt > dyn_track_min_valid
            # logging.info(
            #     f"Update short dynamic track mask, prev has {short_d_track_mask.shape[1]} valid tracks, now has {short_d_track_valid_mask.sum()} valid tracks (dyn_track_min_valid={dyn_track_min_valid})"
            # )
            # ! always accpet short dyn track??
            # ! here the problem is how you identify the dynamic tracks!!
            short_d_track_valid_mask = short_d_track_valid_cnt > 0
            logging.warning(
                f"Accept all the short tracks with at least on valid ({short_d_track_valid_mask.float().mean() *100.0:.2f}%), with unique = {torch.unique(short_d_track_valid_cnt[short_d_track_valid_mask], return_counts=True)}"
            )

            short_track_dynamic_mask = prior2d.short_track_dynamic_mask.clone()
            short_track_dynamic_mask[short_track_dynamic_mask.clone()] = (
                short_d_track_valid_mask
            )
            prior2d.short_track_dynamic_mask = short_track_dynamic_mask.clone()

        # if viz_flag:
        #     # if prior2d.T > 150:
        #     #     viz_skip = prior2d.T // 50
        #     # else:
        #     #     viz_skip = 1
        #     # viz_n = 1024
        #     viz_skip, viz_n = 1, d_track.shape[1]

        #     d_track = prior2d.track[:, prior2d.track_dynamic_mask]
        #     d_track_mask = prior2d.track_mask[:, prior2d.track_dynamic_mask]
        #     viz_choice = torch.randperm(d_track.shape[1])[:viz_n].cpu().numpy()
        #     self.track_visualizer.visualize(
        #         video=prior2d.rgbs.permute(0, 3, 1, 2)[::viz_skip][None].cpu()
        #         * 255,  # B,T,C,H,W
        #         tracks=d_track[::viz_skip][:, viz_choice][None],
        #         visibility=d_track_mask[::viz_skip][:, viz_choice][None],
        #         filename=f"dynamic_filtered",
        #     )
        return

    def specify_spatial_unit(self, unit, world_flag=True):
        # ! consider the rescale of prior2d
        if world_flag:
            factor = self.prior2d.depth_rescale_factor_model_world
            self.spatial_unit = unit * factor
            logging.info(
                f"Model spatial unit = {self.spatial_unit:.6f} = {unit:.3f} x {factor:.3f}"
            )
        else:
            self.spatial_unit = unit
            logging.info(
                f"Directly set in abs unit! spatial_unit={self.spatial_unit:.6f}"
            )
        return

    @torch.no_grad()
    def __preprocess_4dscf__(
        self,
        cams: SimpleFovCamerasIndependent,
        vel_jitter_th_value,
        min_valid_cnt_ratio,
    ):
        prior2d: Prior2D = self.prior2d

        t_list = prior2d.dyn_track_subsample_t_list
        logging.info(f"Dyn SCF t-subsampled with {len(t_list)} frames")
        t_list = torch.tensor(t_list).to(self.device).long()

        # * load track
        d_track = prior2d.track[t_list][:, prior2d.track_dynamic_mask]
        d_track_mask = prior2d.track_mask[t_list][:, prior2d.track_dynamic_mask]
        d_feat_mean = prior2d.track_feat_mean[prior2d.track_dynamic_mask]  # N,C+3
        d_feat_var = prior2d.track_feat_var[prior2d.track_dynamic_mask]  # N,C+3

        # * filter tracks
        viz_plt_missing_slot(
            d_track_mask,
            osp.join(
                self.viz_dir, "dyn_scaffold_before_vel_filtering_completeness.jpg"
            ),
        )
        homo_list, dep_list, normal_cam_list, rgb_list = prepare_track_buffers(
            prior2d, d_track, d_track_mask, t_list
        )

        logging.warning(f"Apply vel filtering wiht th={vel_jitter_th_value}")
        d_track_mask, _ = detect_sharp_changes_in_curve(
            d_track_mask,
            get_world_points(homo_list, dep_list, cams, t_list),
            max_vel_th=vel_jitter_th_value,
        )
        viz_plt_missing_slot(
            d_track_mask,
            osp.join(self.viz_dir, "dyn_scaffold_after_vel_filtering_completeness.jpg"),
        )

        # ensure that at least one slot is visible
        min_valid_cnt = int(len(t_list) * min_valid_cnt_ratio)
        recheck_curve_cnt = d_track_mask.sum(dim=0)
        recheck_curve_mask = recheck_curve_cnt > min_valid_cnt

        logging.info(
            f"Each curve must have at least {min_valid_cnt} valid nodes before completion, {recheck_curve_mask.float().mean()*100:.2f}% pass the check"
        )
        assert recheck_curve_mask.any(), "no valid noodle anymore"
        d_track = d_track[:, recheck_curve_mask]
        d_track_mask = d_track_mask[:, recheck_curve_mask]
        homo_list = homo_list[:, recheck_curve_mask]
        dep_list = dep_list[:, recheck_curve_mask]
        rgb_list = rgb_list[:, recheck_curve_mask]
        # semantic_feature_list = semantic_feature_list # TODO: how to deal with mask?
        normal_cam_list = normal_cam_list[:, recheck_curve_mask]  # T,N,3
        d_feat_mean = d_feat_mean[recheck_curve_mask]
        d_feat_var = d_feat_var[recheck_curve_mask]

        # ! rotate the normal to world frame
        R_wc_list, _ = cams.Rt_wc_list()
        R_wc_list = R_wc_list[t_list]  # T,3,3
        normal_world_list = torch.einsum("tij,tnj->tni", R_wc_list, normal_cam_list)

        logging.info(
            f"Dyn Scaffold {d_track_mask.float().mean()*100.0:.2f}% empty slots need to be filled in!"
        )

        # * complete init the curve
        node_xyz = line_segment_init(
            d_track_mask, get_world_points(homo_list, dep_list, cams, t_list).clone()
        ) # 80, 7046, 3

        # node_xyz = node_xyz.transpose(0, 1)
        # # to basis motion (shape of motion)
        # cano_t = d_track_mask.sum(-1).argmax().item()
        #
        # motion_bases, motion_coefs, node_xyz = init_motion_params_with_procrustes(node_xyz, num_bases=10, rot_type='6d', cano_t=cano_t)
        # motion_coefs = motion_coefs[None]
        # node_xyz = node_xyz.transpose(0, 1)
        #
        # rgb_list = rgb_list[cano_t][None]
        # semantic_feature_list = semantic_feature_list[cano_t][None]
        # d_track_mask = d_track_mask
        # normal_world_list = normal_world_list[cano_t][None]

        return (
            node_xyz,
            normal_world_list,
            d_track_mask,
            rgb_list,
            # semantic_feature_list,
            t_list,
            d_feat_mean,
            d_feat_var,
            # motion_bases,
            # motion_coefs,
            # cano_t,
        )

    def get_dynamic_scaffold(
        self,
        cams: SimpleFovCamerasIndependent,
        skinning_method="dqb",
        topo_curve_dist_top_k=8,
        topo_curve_dist_sample_T=80,
        topo_k=16,
        topo_th_ratio=10.0,
        sigma_max_ratio=1.0,
        sigma_init_ratio=0.2,
        vel_jitter_th_value=0.1,
        min_valid_cnt_ratio=0.1,
        mlevel_arap_flag=True,
        mlevel_list=[1, 8],
        mlevel_k_list=[16, 8],
        mlevel_w_list=[0.4, 0.3],
        max_node_num=10000,
        gs_sk_approx_flag=False,
        dyn_o_flag=False,
        resample_flag=True,
        # ! abl
        skinning_topology="graph",
    ):
        if not mlevel_arap_flag:
            logging.warning(f"ABL: disable multi-level")
        if skinning_method != "dqb":
            logging.warning(f"ABL: use {skinning_method} instead of dqb")
        if skinning_topology != "graph":
            logging.warning(
                f"ABL: use {skinning_topology} skinning topo instead of graph"
            )

        node_xyz, node_normal, d_track_mask, rgb_list, t_list, feat_mean, feat_var = (
            self.__preprocess_4dscf__(
                cams,
                vel_jitter_th_value=vel_jitter_th_value,
                min_valid_cnt_ratio=min_valid_cnt_ratio,
            )
        )
        T, M = node_xyz.shape[:2]
        node_quat = torch.Tensor([1.0, 0.0, 0.0, 0.0])[None, None, :].expand(T, M, -1)
        # semantic_feature_list
        NUM_SEMANTIC_CHANNELS = self.prior2d.latent_feature_channel
        node_semantic_feature = torch.zeros(M,NUM_SEMANTIC_CHANNELS)
        # init scf instance
        scf: Scaffold4D = Scaffold4D(
            node_xyz=node_xyz,
            node_quat=node_quat,
            node_semantic_feature=node_semantic_feature,
            skinning_k=topo_k,
            skinning_method=skinning_method,
            skinning_topology=skinning_topology,
            topo_curve_dist_top_k=topo_curve_dist_top_k,
            topo_curve_dist_sample_T=topo_curve_dist_sample_T,
            topo_th_ratio=topo_th_ratio,
            sigma_max_ratio=sigma_max_ratio,
            sigma_init_ratio=sigma_init_ratio,
            spatial_unit=self.spatial_unit,
            mlevel_arap_flag=mlevel_arap_flag,
            mlevel_list=mlevel_list,
            mlevel_k_list=mlevel_k_list,
            mlevel_w_list=mlevel_w_list,
            device=self.device,
            max_node_num=max_node_num,  # hard coded for now
            # other buffer
            curve_slot_init_valid_mask=d_track_mask,
            curve_color_init=rgb_list,
            curve_semantic_feature_init=node_semantic_feature,
            curve_normal_init=node_normal,
            t_list=t_list,
            # semantic features
            semantic_feature_mean=feat_mean,
            semantic_feature_var=feat_var,
            # GS vs RBF
            gs_sk_approx_flag=gs_sk_approx_flag,
            # dbug
            dyn_o_flag=dyn_o_flag,
        )
        logging.info(f"Finish init dynamic scaffold with {scf.M} nodes")
        if resample_flag:
            logging.warning(
                f"Just after scf init, resample it, which means lossing info for scf init"
            )
            # resample after the init!
            scf.resample_node(1.0, use_mask=True)
        return scf, t_list

    @torch.no_grad()
    def get_static_model(
        self,
        cams: SimpleFovCamerasIndependent,
        n_static_init=30000,
        radius_max=0.1,
        max_sph_order=0,
        normal_dir_ratio=10.0,
    ):
        prior2d = self.prior2d
        device = self.device

        logging.info(
            f"Attaching to Static Scaffold with normal factor={normal_dir_ratio} ..."
        )
        viz_fn = osp.join(self.viz_dir, "static_fetched.xyz")
        mu_init, q_init, s_init, o_init, static_rgb_init, semantic_feature_init, _ = (
            fetch_leaves_in_world_frame(
                "sta_dep",
                prior2d,
                cams,
                n_static_init,
                viz_fn,
                normal_dir_ratio=normal_dir_ratio,
            )
        )
        s_model: StaticGaussian = StaticGaussian(
            init_mean=mu_init.clone().to(device),
            init_q=q_init.clone().to(device),
            init_s=s_init.clone().to(device) * self.radius_init_factor,
            init_o=o_init.clone().to(device) * self.opacity_init_factor,
            init_rgb=static_rgb_init.clone().to(device),
            init_semantic_feature=semantic_feature_init.clone().to(device),
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
            device=device,
        )
        s_model.to(device)
        return s_model

    @torch.no_grad()
    def get_dynamic_model(
        self,
        cams: SimpleFovCamerasIndependent,
        scf: Scaffold4D,
        image_stride=2,
        radius_max=0.1,
        max_sph_order=0,
        n_init=-1,
        start_t=0,
        end_t=0,  # included
        attach_t_interval=1,
        opa_init_value=None,
        leaf_local_flag=True,
        topo_th_ratio=10.0,
        normal_dir_ratio=10.0,
        # cfg for canonincal frame
        canonical_ref_flag=False,
        canonical_tid_mode="largest",
        # *
        abl_nn_fusion=-1,
    ):
        prior2d = self.prior2d
        device = self.device
        logging.info(f"Get dynamic model with normal facotr={normal_dir_ratio} ...")

        # * fetch leaves
        if end_t == -1:
            end_t = cams.T - 1  # the last frame, included!
        append_t_list = append_t_list = [
            i for i in range(start_t, end_t + 1, attach_t_interval)
        ]
        if end_t not in append_t_list:
            append_t_list.append(end_t)
        logging.info(f"Append at t={append_t_list}")

        mu_init, q_init, s_init, o_init, rgb_init, semantic_feature_init, time_init = (
            fetch_leaves_in_world_frame(
                "dyn_dep",
                prior2d,
                cams,
                n_attach=n_init,
                end_t=end_t + 1,  # include the end_t
                # save_fn=osp.join(self.viz_dir, "dynamic_init_fetched.xyz"),
                subsample=image_stride,
                t_list=append_t_list,
                normal_dir_ratio=normal_dir_ratio,
            )
        )

        # * Reset SCF topo th!
        old_th_ratio = scf.topo_th_ratio
        scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
        logging.info(
            f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
        )

        # * Init the scf
        d_model: DynSCFGaussian = DynSCFGaussian(
            scf=scf,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
            device=device,
            leaf_local_flag=leaf_local_flag,
            nn_fusion=abl_nn_fusion,
        )
        d_model.to(device)

        # * Init the leaves
        optimizer = torch.optim.Adam(
            d_model.get_optimizable_list(
                **OptimCFG(
                    lr_p=0.0, lr_dyn=0.0, lr_np=0.0, lr_nq=0.0, lr_nsig=0.0
                ).get_dynamic_lr_dict
            )
        )

        # Append leaves to a time frame
        if canonical_ref_flag:
            logging.info("Move all different time leaves to the same time")
            if canonical_tid_mode == "largest":
                fg_masks = prior2d.dynamic_masks
                fg_count = fg_masks.sum(dim=(1, 2))
                ref_tid = fg_count.argmax()
                logging.info(f"Largest mode, use [{ref_tid}] as the reference frame")
                scf: Scaffold4D = d_model.scf
                # warp all the leaves to this time
                ref_mu_init = mu_init.clone()
                ref_q_init = q_init.clone()
                for tid in tqdm(time_init.unique()):
                    t_mask = time_init == tid
                    attach_node_id = scf.identify_nearest_node_id(mu_init[t_mask], tid)
                    _xyz, _R = scf.warp(
                        attach_node_id,
                        query_xyz=mu_init[t_mask],
                        query_tid=tid,
                        target_tid=ref_tid,
                        query_dir=quaternion_to_matrix(q_init[t_mask]),
                    )
                    ref_mu_init[t_mask] = _xyz
                    ref_q_init[t_mask] = matrix_to_quaternion(_R)
                np.savetxt(
                    osp.join(self.viz_dir, f"dyn_gs_init_ref={ref_tid}.xyz"),
                    ref_mu_init.detach().cpu(),
                    fmt="%.4f",
                )
                d_model.append_new_gs(
                    optimizer,
                    tid=ref_tid,
                    mu_w=ref_mu_init,
                    quat_w=ref_q_init,
                    scales=s_init,
                    opacity=o_init,
                    rgb=rgb_init,
                    semantic_feature=semantic_feature_init,
                )
            else:
                raise NotImplementedError()
        else:
            unique_tid = time_init.unique()
            logging.info("Attach to Dynamic Scaffold ...")
            if opa_init_value is None:
                opa_init_value = self.opacity_init_factor
            for tid in tqdm(unique_tid):
                t_mask = time_init == tid
                d_model.append_new_gs(
                    optimizer,
                    tid=tid,
                    mu_w=mu_init[t_mask],
                    quat_w=q_init[t_mask],
                    scales=s_init[t_mask],
                    opacity=o_init[t_mask] * opa_init_value,
                    rgb=rgb_init[t_mask],
                    semantic_feature=semantic_feature_init[t_mask],
                )
        return d_model

    def finetune_gs_model(
        self,
        semantic_heads:Feature_heads,
        cams: SimpleFovCamerasIndependent,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian = None,
        optim_cam_after_steps=0,
        total_steps=8000,
        topo_update_feq=50,
        skinning_corr_start_steps=1e10,
        node_ctrl_start_steps=5000,
        reset_at_beginning=False,
        s_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00025,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        d_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00015,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        s_gs_ctrl_start_ratio=0.2,
        s_gs_ctrl_end_ratio=0.9,
        d_gs_ctrl_start_ratio=0.2,
        d_gs_ctrl_end_ratio=0.9,
        # optim
        optimizer_cfg: OptimCFG = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0001,
            lr_cam_t=0.0001,
            lr_p=0.0003,
            lr_q=0.002,
            lr_s=0.01,
            lr_o=0.1,
            lr_sph=0.005,
            # dyn
            lr_np=0.001,
            lr_nq=0.01,
            lr_w=0.3,
        ),
        n_frame=1,
        # cfg
        lambda_rgb=1.0,
        lambda_semantic_feature=1.0,
        lambda_dep=1.0,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        physical_reg_until_step=100000000000, #  early stop the phy reg
        reg_radius=None,
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        sup_mask_type="all",
        phase_name="finetune",
        use_decay=True,
        decay_start=2000,
        gt_cam_flag=False,
        # * error grow
        dyn_error_grow_steps=[],
        dyn_error_grow_th=0.2,
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        # * novel views for visualization
        viz_ref_train_camera_T_wc=None,  # use this to solve the alignment of solved camera with the
        viz_test_camera_T_wc_list=None,  # can have several test traj
        # * others
        random_bg=False,
        # * SDS
        sd_prior=None,
        sd_novel_view_list=[],
        lambda_sds=1.0,
        sds_guidance_scale=25.0,
        viz_sds_interval=25,
        # * DEBUG
        freeze_static_after=10000000000,
        unfreeze_static_after=10000000000,
    ):
        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND}")

        assert (
            not gt_cam_flag
        ), "for now, not matter start from gt cam or not, always use the camera!"

        # before start viz the static dynmic masks
        viz_mask_video(
            self.prior2d.rgbs,
            self.prior2d.dynamic_masks,
            osp.join(self.viz_dir, f"{phase_name}_working_dyn_mask.mp4"),
        )
        viz_mask_video(
            self.prior2d.rgbs,
            self.prior2d.static_masks,
            osp.join(self.viz_dir, f"{phase_name}_working_sta_mask.mp4"),
        )

        torch.cuda.empty_cache()
        assert n_frame == 1, "n_frame should be 1 for now, more is a bad idea for now"
        d_flag = d_model is not None
        if not d_flag:
            logging.warning(
                "No dynamic model is provided, only optimize the static model"
            )
            assert "sta" in sup_mask_type, "sup_mask_type should be static"
        else:
            d_model.summary()

        sds_flag = sd_prior is not None
        if sds_flag:
            logging.info("Enable SDS")

        sampler = RandomSampler()
        optimizer_static = torch.optim.Adam(
            s_model.get_optimizable_list(**optimizer_cfg.get_static_lr_dict)
        )
        s_model.train()
        print(f"optimizer static: {optimizer_static}")
        optimizer_semantic_heads = torch.optim.Adam(
            semantic_heads.parameters(), lr=optimizer_cfg.lr_semantic_heads
        )
        print(f"optimizer semantic heads: {optimizer_semantic_heads}")

        if d_flag:
            optimizer_dynamic = torch.optim.Adam(
                d_model.get_optimizable_list(**optimizer_cfg.get_dynamic_lr_dict)
            )
            d_model.train()
            if reg_radius is None:
                reg_radius = int(np.array(self.temporal_diff_shift).max()) * 2
            logging.info(f"Set reg_radius={reg_radius} for dynamic model")
        if not gt_cam_flag:
            optimizer_cam = torch.optim.Adam(
                cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
            )

        if use_decay:
            gs_scheduling_func_dict, cam_scheduling_func_dict = (
                optimizer_cfg.get_scheduler(total_steps=total_steps - decay_start)
            )
        else:
            gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}

        loss_rgb_list, loss_dep_list, loss_nrm_list = [], [], []
        loss_semantic_feature_list = []
        viz_loss_sam2_list = []
        viz_loss_internvideo_list = []
        viz_loss_langseg_list = []
        loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

        loss_arap_coord_list, loss_arap_len_list = [], []
        loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
        loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
        s_n_count_list, d_n_count_list = [], []
        d_m_count_list = []
        loss_sds_list = []

        prior2d: Prior2D = self.prior2d
        s_gs_ctrl_start = int(total_steps * s_gs_ctrl_start_ratio)
        d_gs_ctrl_start = int(total_steps * d_gs_ctrl_start_ratio)
        s_gs_ctrl_end = int(total_steps * s_gs_ctrl_end_ratio)
        d_gs_ctrl_end = int(total_steps * d_gs_ctrl_end_ratio)
        # s_gs_ctrl_start=5 # ! debug,
        assert s_gs_ctrl_start >= 0
        assert d_gs_ctrl_start >= 0

        if reset_at_beginning:
            s_model.reset_opacity(optimizer_static, s_gs_ctrl_cfg.reset_opacity)
            if d_flag:
                logging.warning("DEBUG, not reset d model")
                # d_model.reset_opacity(optimizer_dynamic, d_gs_ctrl_cfg.reset_opacity)

        sup_count = torch.zeros(cams.T, device=s_model.device)
        bg_frozen_flag = False


        for step in tqdm(range(total_steps)):
            # * debug
            if step == freeze_static_after and d_flag:
                logging.warning(f"Freeze BG at {step}")
                # bg_cache_list = self.__cache_static_bg__(cams, s_model)
                bg_frozen_flag = True
                s_model.eval()
            if step == unfreeze_static_after and d_flag:
                # bg_cache_list = None
                logging.warning(f"Open BG at {step}")
                bg_frozen_flag = False
                s_model.train()
            if step == physical_reg_until_step:
                logging.warning(f"Disable ARAP, VEL, ACC reg and set their lambdas to zero!")
                lambda_acc_rot_reg, lambda_acc_xyz_reg = 0.0, 0.0
                lambda_vel_rot_reg, lambda_vel_xyz_reg = 0.0, 0.0
                lambda_arap_len, lambda_arap_coord = 0.0, 0.0
            semantic_heads.train()
            # * control the w correction
            if d_flag and step == skinning_corr_start_steps:
                logging.info(
                    f"at {step} stop all the topology update and add skinning weight correction"
                )
                d_model.set_surface_deform()

            optimizer_static.zero_grad()
            optimizer_semantic_heads.zero_grad()
            if not gt_cam_flag:
                optimizer_cam.zero_grad()
            if d_flag:
                optimizer_dynamic.zero_grad()
                if step % topo_update_feq == 0:
                    d_model.scf.update_topology()

            if step > decay_start:
                for k, v in gs_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, optimizer_static)
                    if d_flag:
                        update_learning_rate(v(step), k, optimizer_dynamic)
                for k, v in cam_scheduling_func_dict.items():
                    if not gt_cam_flag:
                        update_learning_rate(v(step), k, optimizer_cam)

            # view_ind_list = sampler.sample(cams.T - 1, n_frame)
            view_ind_list = sampler.sample(0, cams.T - 1, n_frame)
            # optimize
            render_dict_list = []
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_semantic_feature = 0.0
            loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0

            viz_loss_sam2 = 0.0
            viz_loss_internvideo = 0.0
            viz_loss_langseg = 0.0
            for view_ind in view_ind_list:
                sup_count[view_ind] += 1
                gs5 = [list(s_model())]
                if d_flag:
                    gs5.append(list(d_model(view_ind)))
                bg_cache = None
                # print(gs5[0][5].sum())
                assert not gs5[0][5].isnan().any()
                if random_bg:
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = [1.0, 1.0, 1.0]
                render_dict = render(
                    gs5,
                    prior2d.H,
                    prior2d.W,
                    cams.rel_focal,
                    cams.cxcy_ratio,
                    cams.T_cw(view_ind),
                    bg_cache_dict=bg_cache,
                    bg_color=bg_color,
                )
                assert not render_dict["feature_map"].isnan().any()
                render_dict_list.append(render_dict)
                # compute losses
                rgb_sup_mask = self.prior2d.get_mask_by_key(sup_mask_type, view_ind)
                _l_rgb, _, _, _ = compute_rgb_loss(
                    self.prior2d, view_ind, render_dict, rgb_sup_mask
                )
                _l_semantic_feature, loss_dict, = compute_semantic_feature_loss(
                    self.prior2d, view_ind, render_dict, rgb_sup_mask, semantic_heads=semantic_heads # TODO: all s_model cnn decoder?
                )
                if "sam2" in loss_dict:
                    viz_loss_sam2 += loss_dict["sam2"].item()
                if "internvideo" in loss_dict:
                    viz_loss_internvideo += loss_dict["internvideo"].item()
                if "langseg" in loss_dict:
                    viz_loss_langseg += loss_dict["langseg"].item()

                logging.info(f"feat losses: {loss_dict}")

                dep_sup_mask = self.prior2d.get_mask_by_key(
                    sup_mask_type + "_dep", view_ind
                )
                _l_dep, _, _, _ = compute_dep_loss(
                    self.prior2d,
                    view_ind,
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant,
                )
                loss_rgb = loss_rgb + _l_rgb
                loss_semantic_feature = loss_semantic_feature + _l_semantic_feature
                loss_dep = loss_dep + _l_dep

                # * GOF normal and regularization
                if GS_BACKEND == "gof":
                    _l_nrm, _, _, _ = compute_normal_loss(
                        self.prior2d, view_ind, render_dict, dep_sup_mask
                    )
                    loss_nrm = loss_nrm + _l_nrm
                    if step > geo_reg_start_steps:
                        _l_reg_nrm, _, _, _ = compute_normal_reg_loss(
                            prior2d, cams, render_dict
                        )
                        _l_reg_distortion, _ = compute_dep_reg_loss(
                            prior2d, view_ind, render_dict
                        )
                    else:
                        _l_reg_nrm = torch.zeros_like(_l_rgb)
                        _l_reg_distortion = torch.zeros_like(_l_rgb)
                    loss_dep_nrm_reg = loss_dep_nrm_reg + _l_reg_nrm
                    loss_distortion_reg = loss_distortion_reg + _l_reg_distortion
                else:
                    loss_nrm = torch.zeros_like(loss_rgb)
                    loss_dep_nrm_reg = torch.zeros_like(loss_rgb)
                    loss_distortion_reg = torch.zeros_like(loss_rgb)

            if d_flag:
                _l = max(0, view_ind_list[0] - reg_radius)
                _r = min(cams.T, view_ind_list[0] + 1 + reg_radius)
                reg_tids = torch.arange(_l, _r, device=s_model.device)
            if (lambda_arap_coord > 0.0 or lambda_arap_len > 0.0) and d_flag:
                loss_arap_coord, loss_arap_len = d_model.scf.compute_arap_loss(
                    reg_tids,
                    temporal_diff_shift=self.temporal_diff_shift,
                    temporal_diff_weight=self.temporal_diff_weight,
                )
                assert torch.isnan(loss_arap_coord).sum() == 0
                assert torch.isnan(loss_arap_len).sum() == 0
            else:
                loss_arap_coord = torch.zeros_like(loss_rgb)
                loss_arap_len = torch.zeros_like(loss_rgb)

            if (
                lambda_vel_xyz_reg > 0.0
                or lambda_vel_rot_reg > 0.0
                or lambda_acc_xyz_reg > 0.0
                or lambda_acc_rot_reg > 0.0
            ) and d_flag:
                (
                    loss_vel_xyz_reg,
                    loss_vel_rot_reg,
                    loss_acc_xyz_reg,
                    loss_acc_rot_reg,
                ) = d_model.scf.compute_vel_acc_loss(reg_tids)
            else:
                loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                    loss_acc_rot_reg
                ) = torch.zeros_like(loss_rgb)
            logging.info(f"semantic feature loss: {loss_semantic_feature.item()*lambda_semantic_feature}", )
            loss = (
                loss_rgb * lambda_rgb
                + loss_semantic_feature * lambda_semantic_feature
                + loss_dep * lambda_dep
                + loss_nrm * lambda_normal
                + loss_dep_nrm_reg * lambda_depth_normal
                + loss_distortion_reg * lambda_distortion
                + loss_arap_coord * lambda_arap_coord
                + loss_arap_len * lambda_arap_len
                + loss_vel_xyz_reg * lambda_vel_xyz_reg
                + loss_vel_rot_reg * lambda_vel_rot_reg
                + loss_acc_xyz_reg * lambda_acc_xyz_reg
                + loss_acc_rot_reg * lambda_acc_rot_reg
            )

            if sds_flag:

                # render a novel view
                sds_cam_T_wi = sd_novel_view_list[
                    np.random.choice(len(sd_novel_view_list))
                ]
                gs5 = [list(s_model())]
                if d_flag:
                    # ! use the same time
                    gs5.append(list(d_model(view_ind)))
                bg_cache = None
                if random_bg:
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = [1.0, 1.0, 1.0]
                render_dict = render(
                    gs5,
                    512,
                    512,
                    cams.rel_focal,
                    cams.cxcy_ratio,
                    torch.linalg.inv(sds_cam_T_wi).to(self.device),
                    bg_color=bg_color,
                )
                # compute the sds loss
                # ! todo: check prev code and set the linear ratio
                loss_sds = sd_prior.train_step( # TODO: what about semantic feature?
                    render_dict["rgb"][None],
                    guidance_scale=sds_guidance_scale,
                    reduction="mean",
                )
                if step % viz_sds_interval == 0:
                    viz_rgb = render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
                    viz_dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
                    viz_normal = (
                        ((1.0 - render_dict["normal"].permute(1, 2, 0)) / 2.0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    viz_dep = cm.viridis(viz_dep)[..., :3]
                    viz_sds = np.concatenate([viz_rgb, viz_dep, viz_normal], axis=1)
                    imageio.imsave(
                        osp.join(self.viz_dir, f"step={step}_sds.jpg"), viz_sds
                    )

                loss = loss + loss_sds * lambda_sds
                loss_sds_list.append(loss_sds.item())

            print('loss:', loss) # Shuwang
            loss.backward()

            if ~bg_frozen_flag:
                optimizer_static.step()
            # optimizer_static.step()
            if d_flag:
                optimizer_dynamic.step()
            if not gt_cam_flag and step >= optim_cam_after_steps:
                optimizer_cam.step()
            optimizer_semantic_heads.step()
            # gs control
            if (
                s_gs_ctrl_cfg is not None
                and step >= s_gs_ctrl_start
                and step < s_gs_ctrl_end
                and ~bg_frozen_flag
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=s_model,
                    gs_control_cfg=s_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_static,
                    first_N=s_model.N,
                )
            if (
                d_gs_ctrl_cfg is not None
                and step >= d_gs_ctrl_start
                and step < d_gs_ctrl_end
                and d_flag
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=d_model,
                    gs_control_cfg=d_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_dynamic,
                    last_N=d_model.N,
                )

            # error grow
            if d_flag and step in dyn_error_grow_steps:
                self.error_grow_dyn_model(
                    cams,
                    s_model,
                    d_model,
                    optimizer_dynamic,
                    step,
                    dyn_error_grow_th,
                    dyn_error_grow_num_frames,
                    dyn_error_grow_subsample,
                )
            if d_flag and step in dyn_scf_prune_steps:
                d_model.prune_nodes(
                    optimizer_dynamic,
                    prune_sk_th=dyn_scf_prune_sk_th,
                    viz_fn=osp.join(self.viz_dir, f"scf_node_prune_at_step={step}"),
                )

            loss_rgb_list.append(loss_rgb.item())
            loss_semantic_feature_list.append(loss_semantic_feature.item())
            viz_loss_sam2_list.append(viz_loss_sam2)
            viz_loss_internvideo_list.append(viz_loss_internvideo)
            viz_loss_langseg_list.append(viz_loss_langseg)
            loss_dep_list.append(loss_dep.item())
            loss_nrm_list.append(loss_nrm.item())

            loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
            loss_distortion_reg_list.append(loss_distortion_reg.item())

            loss_arap_coord_list.append(loss_arap_coord.item())
            loss_arap_len_list.append(loss_arap_len.item())
            loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
            loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
            loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
            loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
            s_n_count_list.append(s_model.N)
            d_n_count_list.append(d_model.N if d_flag else 0)
            d_m_count_list.append(d_model.M if d_flag else 0)

            wandb.log(
                {
                    f"{phase_name}/loss_rgb": loss_rgb.item(),
                    f"{phase_name}/loss_semantic_feature": loss_semantic_feature.item(),
                    f"{phase_name}/loss_dep": loss_dep.item(),
                    f"{phase_name}/loss_nrm": loss_nrm.item(),
                    f"{phase_name}/loss_dep_nrm_reg": loss_dep_nrm_reg.item(),
                    f"{phase_name}/loss_distortion_reg": loss_distortion_reg.item(),
                    f"{phase_name}/loss_arap_coord": loss_arap_coord.item(),
                    f"{phase_name}/loss_arap_len": loss_arap_len.item(),
                    f"{phase_name}/loss_vel_xyz_reg": loss_vel_xyz_reg.item(),
                    f"{phase_name}/loss_vel_rot_reg": loss_vel_rot_reg.item(),
                    f"{phase_name}/loss_acc_xyz_reg": loss_acc_xyz_reg.item(),
                    f"{phase_name}/loss_acc_rot_reg": loss_acc_rot_reg.item(),
                    f"{phase_name}/s_model_N": s_model.N,
                    f"{phase_name}/d_model_N": d_model.N if d_flag else 0,
                    f"{phase_name}/d_model_M": d_model.M if d_flag else 0,
                }
            )

            # viz
            viz_flag = viz_interval > 0 and (step % viz_interval == 0)
            if viz_flag:
                viz_hist(
                    s_model, self.viz_dir, f"phase={phase_name}_step={step}_static"
                )
                ### sz: bug for windows
                # viz2d_total_video(
                #     self,
                #     0,
                #     cams.T - 1,
                #     viz_skip_t,
                #     cams,
                #     s_model,
                #     d_model,
                #     save_dir=self.viz_dir,
                #     prefix=f"phase={phase_name}_step={step}_",
                #     subsample=1,
                #     mask_type=sup_mask_type,
                #     move_around_angle_deg=viz_move_angle_deg,
                # )
                if d_flag:
                    viz_hist(
                        d_model, self.viz_dir, f"phase={phase_name}_step={step}_dynamic"
                    )
                    viz_dyn_hist(
                        d_model.scf,
                        self.viz_dir,
                        f"phase={phase_name}_step={step}_dynamic",
                    )
                    viz_path = osp.join(
                        self.viz_dir, f"phase={phase_name}_step={step}_3dviz.mp4"
                    )
                    viz3d_total_video(
                        cams,
                        d_model,
                        0,
                        cams.T - 1,
                        save_path=viz_path,
                        res=480,  # 240
                        s_model=s_model,
                    )
                    # viz novel view
                    if viz_test_camera_T_wc_list is not None:
                        novel_view_tids = torch.arange(d_model.T)
                        for vi in range(len(viz_test_camera_T_wc_list)):
                            frames, test_latent_features = render_test(
                                H=prior2d.H,
                                W=prior2d.W,
                                cams=cams,
                                s_model=s_model,
                                d_model=d_model,
                                train_camera_T_wi=viz_ref_train_camera_T_wc,
                                test_camera_T_wi=viz_test_camera_T_wc_list[vi][
                                    None
                                ].expand(len(novel_view_tids), -1, -1),
                                test_camera_tid=novel_view_tids,
                            )
                            imageio.mimsave(
                                osp.join(
                                    self.viz_dir, f"novel_view_cam{vi}_step={step}.mp4"
                                ),
                                frames,
                            )

            if viz_cheap_interval > 0 and (
                step % viz_cheap_interval == 0 or step == total_steps - 1
            ):
                # ! viz the gradient accumulated map
                if d_flag:

                    viz_grad_list = viz_d_model_grad(
                        prior2d, d_model, cams, d_gs_ctrl_cfg.densify_max_grad
                    )
                    imageio.mimsave(
                        osp.join(self.viz_dir, f"grad_viz_step={step}.mp4"),
                        viz_grad_list,
                    )
                    node_xyz = d_model.scf._node_xyz
                    viz_frame = viz_curve(
                        node_xyz,
                        torch.zeros_like(node_xyz),
                        semantic_feature=torch.zeros_like(d_model.scf._node_semantic_feature),
                        mask=torch.ones_like(node_xyz[..., 0]).bool(),
                        cams = cams,
                        res=480,
                        pts_size=0.001,
                        only_viz_last_frame=True,
                        no_rgb_viz=True,
                        text=f"Step={step}",
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"phase={phase_name}_step={step}_curve.jpg"
                        ),
                        viz_frame[0],
                    )
                # plot sup count
                plt.figure(figsize=(10, 8))
                plt.bar(range(cams.T), sup_count.cpu().numpy())
                plt.title(f"Supervision Count")
                plt.savefig(osp.join(self.viz_dir, f"sup_count_{step}.jpg"))
                plt.close()
                # save txt viz for 3D
                cam_viz_xyz = cams.t_wc.detach().cpu().numpy()
                cam_viz_xyz = np.concatenate(
                    [cam_viz_xyz, np.ones_like(cam_viz_xyz[..., :1])], -1
                )
                static_viz_xyz = s_model()[0].detach().cpu().numpy()
                static_viz_xyz = np.concatenate(
                    [static_viz_xyz, 2 * np.ones_like(static_viz_xyz[..., :1])], -1
                )
                viz_xyz = np.concatenate([cam_viz_xyz, static_viz_xyz], 0)
                if d_flag:
                    dynamic_viz_xyz = d_model(0)[0].detach().cpu().numpy()
                    dynamic_viz_xyz = np.concatenate(
                        [dynamic_viz_xyz, 3 * np.ones_like(dynamic_viz_xyz[..., :1])],
                        -1,
                    )
                    viz_xyz = np.concatenate([viz_xyz, dynamic_viz_xyz], 0)

                viz_one = viz2d_one_frame(
                    0,
                    self.prior2d,
                    cams,
                    s_model,
                    d_model,
                    loss_mask_type="all",
                )
                imageio.imsave(
                    osp.join(self.viz_dir, f"phase={phase_name}_step={step}_2d.jpg"),
                    viz_one,
                )

                # also viz the curves
                # viz
                fig = plt.figure(figsize=(30, 12))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_semantic_feature", loss_semantic_feature_list),
                        ("loss_sam2", viz_loss_sam2_list),
                        ("loss_internvideo", viz_loss_internvideo_list),
                        ("loss_langseg", viz_loss_langseg_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_nrm", loss_nrm_list),
                        ("loss_sds", loss_sds_list),
                        ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                        ("loss_distortion_reg", loss_distortion_reg_list),
                        ("loss_arap_coord", loss_arap_coord_list),
                        ("loss_arap_len", loss_arap_len_list),
                        ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                        ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                        ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                        ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                        ("S-N", s_n_count_list),
                        ("D-N", d_n_count_list),
                        ("D-M", d_m_count_list),
                    ]
                ):
                    plt.subplot(3, 8, plt_i + 1)
                    value_end = 0 if len(plt_pack[1]) == 0 else plt_pack[1][-1]
                    plt.plot(plt_pack[1]), plt.title(
                        plt_pack[0] + f" End={value_end:.4f}"
                    )
                plt.savefig(
                    osp.join(self.viz_dir, f"{phase_name}_optim_loss_step={step}.jpg")
                )
                plt.close()

        # save static, camera and dynamic model
        s_save_fn = osp.join(self.log_dir, f"{phase_name}_s_model.pth")
        torch.save(s_model.state_dict(), s_save_fn)
        torch.save(
            cams.state_dict(), osp.join(self.log_dir, f"{phase_name}_s_model_cam.pth")
        )
        torch.save(
            semantic_heads.state_dict(), osp.join(self.log_dir, f"{phase_name}_semantic_heads.pth")
        )

        if d_model is not None:
            d_save_fn = osp.join(self.log_dir, f"{phase_name}_d_model.pth")
            torch.save(d_model.state_dict(), d_save_fn)

        # viz
        fig = plt.figure(figsize=(30, 12))
        for plt_i, plt_pack in enumerate(
            [
                ("loss_rgb", loss_rgb_list),
                ("loss_semantic_feature", loss_semantic_feature_list),
                ("loss_sam2", viz_loss_sam2_list),
                ("loss_internvideo", viz_loss_internvideo_list),
                ("loss_langseg", viz_loss_langseg_list),
                ("loss_dep", loss_dep_list),
                ("loss_nrm", loss_nrm_list),
                ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                ("loss_distortion_reg", loss_distortion_reg_list),
                ("loss_arap_coord", loss_arap_coord_list),
                ("loss_arap_len", loss_arap_len_list),
                ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                ("S-N", s_n_count_list),
                ("D-N", d_n_count_list),
                ("D-M", d_m_count_list),
            ]
        ):
            plt.subplot(3, 8, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
        plt.savefig(osp.join(self.log_dir, f"{phase_name}_optim_loss.jpg"))
        plt.close()
        save_gauspl_ply(
            osp.join(self.log_dir, f"{phase_name}_s_model.ply"),
            *s_model(),
        )
        make_video_from_pattern(
            osp.join(self.viz_dir, f"phase={phase_name}_step=*curve*.jpg"),
            osp.join(self.log_dir, f"{phase_name}_curve.mp4"),
        )

        ### sz: bug for windows
        # viz2d_total_video(
        #     self,
        #     0,
        #     cams.T - 1,
        #     1,
        #     cams,
        #     s_model,
        #     d_model,
        #     save_dir=self.log_dir,
        #     prefix=f"{phase_name}_",
        #     move_around_angle_deg=viz_move_angle_deg,
        # )
        viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz.mp4")
        if d_flag:
            viz3d_total_video(
                cams,
                d_model,
                0,
                cams.T - 1,
                save_path=viz_path,
                res=480,
                s_model=s_model,
            )
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def error_grow_dyn_model(
        self,
        cams,
        s_model,
        d_model,
        optimizer_dynamic,
        step,
        dyn_error_grow_th,
        dyn_error_grow_num_frames,
        dyn_error_grow_subsample,
        open_k_size=3,
    ):
        prior2d: Prior2D = self.prior2d
        # * identify the error mask
        error_list = identify_rendering_error(
            cams, s_model, d_model, prior2d
        )
        T = len(error_list)
        imageio.mimsave(
            osp.join(self.viz_dir, f"error_raw_{step}.mp4"), error_list.cpu().numpy()
        )
        # imageio.mimsave(
        #     osp.join(self.viz_dir, f"error_rendered_{step}.mp4"),
        #     torch.stack([x["rgb"] for x in error_rendered_list], 0)
        #     .permute(0, 2, 3, 1)
        #     .cpu()
        #     .numpy(),
        # )

        grow_fg_masks = (error_list > dyn_error_grow_th).to(self.device)
        open_kernel = torch.ones(open_k_size, open_k_size)
        # handle large time by chunk the time
        cur = 0
        chunk = 50
        grow_fg_masks_morph = []
        while cur < T:
            _grow_fg_masks = kornia.morphology.opening(
                grow_fg_masks[cur : cur + chunk, None].float(),
                kernel=open_kernel.to(grow_fg_masks.device),
            ).squeeze(1)
            grow_fg_masks_morph.append(_grow_fg_masks.bool())
            cur = cur + chunk
        grow_fg_masks = torch.cat(grow_fg_masks_morph, 0)
        grow_fg_masks = grow_fg_masks * prior2d.depth_masks * prior2d.dynamic_masks
        # viz
        imageio.mimsave(
            osp.join(self.viz_dir, f"error_{step}.mp4"),
            grow_fg_masks.detach().cpu().float().numpy(),
        )

        if dyn_error_grow_num_frames < T:
            # sample some frames to grow
            grow_cnt = grow_fg_masks.reshape(T, -1).sum(-1)
            grow_cnt = grow_cnt.detach().cpu().numpy()
            # grow_tids = np.random.choice(
            #     len(error_list),
            #     dyn_error_grow_num_frames,
            #     p=grow_cnt / grow_cnt.sum(),
            #     replace=False,
            # ).tolist()
            grow_tids = np.argsort(grow_cnt)[-dyn_error_grow_num_frames:][::-1]
            # plot the grow_cnt with bars
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(grow_cnt)), grow_cnt)
            plt.savefig(osp.join(self.viz_dir, f"error_{step}_grow_cnt.jpg"))
            plt.close()
        else:
            grow_tids = [i for i in range(T)]
        logging.info(f"Grow Error at {step} on {grow_tids}")

        for grow_tid in tqdm(grow_tids.tolist()):
            grow_mask = grow_fg_masks[grow_tid]
            if grow_mask.sum() == 0:
                continue
            # ! The append points must be in front of the model depth!!!
            grow_mu_cam, grow_mask = align_to_model_depth(
                prior2d,
                working_mask=grow_mask,
                cams=cams,
                tid=grow_tid,
                s_model=s_model,
                d_model=d_model,
                dep_align_knn=-1,  # 400, #32,
                sub_sample=dyn_error_grow_subsample,
                src_valid_mask_type="dep",
            )
            if len(grow_mu_cam) == 0:
                continue
            # convert to mu_w and get s_inti
            R_wc, t_wc = cams.Rt_wc(grow_tid)
            grow_mu_w = torch.einsum("ij,aj->ai", R_wc, grow_mu_cam) + t_wc[None]
            grow_s = (
                grow_mu_cam[:, -1]
                / cams.rel_focal
                * prior2d.pixel_size
                * dyn_error_grow_subsample
            )
            # # ! debug
            # np.savetxt(
            #     osp.join(self.viz_dir, f"error_{step}_grow_model_t={grow_tid}.xyz"),
            #     d_model(grow_tid)[0].detach().cpu().numpy(),
            # )
            # np.savetxt(
            #     osp.join(self.viz_dir, f"error_{step}_grow_mu_w_t={grow_tid}.xyz"),
            #     grow_mu_w.detach().cpu().numpy(),
            # )

            # ! special function that sample nodes from candidates and select attached nodes
            quat_w = torch.zeros(len(grow_mu_w), 4).to(grow_mu_w)
            quat_w[:, 0] = 1.0
            d_model.append_new_node_and_gs(
                optimizer_dynamic,
                tid=grow_tid,
                mu_w=grow_mu_w,
                quat_w=quat_w,
                scales=grow_s[:, None].expand(-1, 3),
                opacity=torch.ones_like(grow_s)[:, None] * self.opacity_init_factor,
                rgb=prior2d.get_rgb(grow_tid)[grow_mask],
                # semantic_feature=torch.zeros([len(grow_s), NUM_SEMANTIC_CHANNELS]).to(grow_mu_w) , # TODO: what about this - Hui (changed 37 to 128)
            )
        return

    # TODO: render semantic_feautures, append cls_features and store as feature_map.pth under log_dir
    @torch.no_grad()
    def get_semantic_feature_map(
        self,
        cams,
        s_model,
        d_model,
    ):
        #TODO: how to define gs5 in this case?
        view_ind_list = np.arange(0, cams.T).tolist()

        all_results = {}
        for view_ind in view_ind_list:
            gs5 = [list(s_model())]
            gs5.append(list(d_model(view_ind)))
            bg_color = [1.0, 1.0, 1.0]
            bg_cache = None
            render_dict = render(
                gs5,
                self.prior2d.H,
                self.prior2d.W,
                cams.rel_focal,
                cams.cxcy_ratio,
                cams.T_cw(view_ind), #TODO: how to defien T_cw in this case?
                bg_cache_dict=bg_cache,
                bg_color=bg_color,
            )
            pred_feature_map = render_dict["feature_map"] # render_dim(256) x 480 x 480
            pred_feature_map = F.interpolate(pred_feature_map[None, ...], size=(64, 64), mode='bilinear', align_corners=False)[0] # render_dim(256) x 64 x 64

            # encoder_hidden_dims = [256, 128, 64, 32, 16]
            # decoder_hidden_dims = [32, 64, 128, 256, 512, 1024, 1408]
            # checkpoint = torch.load("/public/home/renhui/code/4d/feature-4dgs/lib_4d/autoencoder/ckpt/train2/best_ckpt.pth", weights_only=True)
            # model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
            # model.load_state_dict(checkpoint)
            # model.eval()

            # pred_feature_map = pred_feature_map.permute(1,2,0) # 16 x 16 x render_dim
            # pred_feature_map = pred_feature_map.view(256, -1) # 256 x render_dim
            # pred_feature_map = model.decode(pred_feature_map) # 256 x 1408
            # render_dict["feature_map"] = pred_feature_map.view(16, 16, 1408).permute(2, 0, 1) # 1408 x 16 x 16

            #render_dict["feature_map"] = s_model.cnn_decoder(pred_feature_map)
            
            render_dict["feature_map"] = pred_feature_map
            all_results[view_ind] = render_dict


        torch.save(all_results, osp.join(self.log_dir, f"rendered_results.pth"))

        return all_results



@torch.no_grad()
def identify_rendering_error(
    cams: SimpleFovCamerasIndependent,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    prior2d: Prior2D,
):
    # render all frames and compare photo error
    logging.info("Compute rendering errors ...")
    error_list, rendered_list = [], []
    for t in tqdm(range(d_model.T)):  # ! warning, d_model.T may smaller than cams.T
        gs5 = [s_model(), d_model(t)]
        render_dict = render(
            gs5, prior2d.H, prior2d.W, cams.rel_focal, cams.cxcy_ratio, cams.T_cw(t)
        )
        rgb_pred = render_dict["rgb"].permute(1, 2, 0)
        rgb_gt = prior2d.get_rgb(t)
        error = (rgb_pred - rgb_gt).abs().max(dim=-1).values
        error_list.append(error.detach().cpu())
        # rendered_list.append(render_dict)
    error_list = torch.stack(error_list, 0)
    return error_list #, rendered_list


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


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


#########
# test helper
#########


@torch.no_grad()
def render_test(
    H,
    W,
    cams: SimpleFovCamerasIndependent,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # cover_factor=0.3,
):
    # prior2d: Prior2D = self.prior2d
    # device = self.device
    device = s_model.device

    # first align the camera
    solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
    # solved_cam_T_wi = train_camera_T_wi
    aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
        traj_a=train_camera_T_wi,
        traj_b=solved_cam_T_wi.detach().cpu(),
        traj_c=test_camera_T_wi,
    )
    # aligned_test_camera_T_wi = test_camera_T_wi
    # render
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    test_ret, viz_ret = [], []
    test_latent_features = []
    for i in tqdm(range(len(test_camera_tid))):
        working_t = test_camera_tid[i]
        render_dict = render(
            [s_model(), d_model(working_t)],
            H,
            W,
            focal,
            cxcy_ratio,
            T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        )
        rgb = render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        test_ret.append(rgb)
        semantic_feature = render_dict["feature_map"].detach().cpu()
        test_latent_features.append(semantic_feature)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)

        # #####################################################################
        # # * also render a visiblity mask, to better understand how much of the scene is un-observed from the observation training view
        # d_mu, d_fr, d_s, d_o, d_sph = d_model(working_t)
        # s_mu, s_fr, s_s, s_o, s_sph = s_model()

        # render_observed_dict = render(
        #     [d_mu, d_fr, d_s, d_o, d_sph],
        #     prior2d.H,
        #     prior2d.W,
        #     cams.rel_focal,
        #     cams.cxcy_ratio,
        #     T_cw=cams.T_cw(working_t),
        # )
        # obs_alpha = render_observed_dict["alpha"].squeeze(0).detach()
        # # obs_dep = render_observed_dict["dep"].squeeze(0).detach() / torch.clamp(
        # #     obs_alpha, min=1e-6
        # # )
        # obs_dep = render_observed_dict["dep"].squeeze(0).detach()
        # fg_mask = obs_alpha > 0.95
        # pts_cam = cams.backproject(prior2d.homo_map[fg_mask], obs_dep[fg_mask])
        # pts_world = cams.trans_pts_to_world(working_t, pts_cam)
        # dist_sq, nn_ind, _ = knn_points(d_mu[None], pts_world[None], K=1)
        # certain_gs_mask = (
        #     dist_sq[0, :, 0] < (d_model.scf.spatial_unit * cover_factor) ** 2
        # )

        # certain_color = RGB2SH(torch.tensor([0.0, 1.0, 0.0])).to(device)
        # certain_color = torch.cat(
        #     [certain_color, torch.zeros(d_sph.shape[1] - 3).to(device)]
        # )
        # free_color = RGB2SH(torch.tensor([1.0, 0.0, 0.0])).to(device)
        # free_color = torch.cat(
        #     [free_color, torch.zeros(d_sph.shape[1] - 3).to(device)]
        # )
        # bg_color = RGB2SH(torch.tensor([0.0, 0.0, 1.0])).to(device)
        # bg_color = torch.cat([bg_color, torch.zeros(d_sph.shape[1] - 3).to(device)])
        # d_sph[certain_gs_mask] = certain_color[None]
        # d_sph[~certain_gs_mask] = free_color[None]
        # s_sph[:] = bg_color[None]
        # render_mask_dict = render(
        #     [(s_mu, s_fr, s_s, s_o, s_sph), (d_mu, d_fr, d_s, d_o, d_sph)],
        #     prior2d.H,
        #     prior2d.W,
        #     cams.rel_focal,
        #     cams.cxcy_ratio,
        #     T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        #     bg_color=[1.0, 0.0, 0.0],
        # )
        # viz_mask = render_mask_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        # viz_mask = np.clip(viz_mask, 0, 1)
        # # maks viz the rgb
        # uncertain_region = viz_mask[..., 0] > 0.5
        # blend_alpha = 0.3
        # rgb_viz = rgb.copy()
        # rgb_viz[uncertain_region] = (
        #     blend_alpha * rgb_viz[uncertain_region]
        #     + (1.0 - blend_alpha) * viz_mask[uncertain_region]
        # )
        # viz_ret.append(np.concatenate([viz_mask, rgb_viz], axis=1))
        # #####################################################################

        # #####################################################################
        # # * viz the obs depth backproject
        # observed_dep_mask = prior2d.depth_masks[working_t]
        # observed_dep = prior2d.depths[working_t][observed_dep_mask]
        # pts_cam = cams.backproject(
        #     prior2d.homo_map[observed_dep_mask], observed_dep
        # )
        # radius = observed_dep / cams.rel_focal * prior2d.pixel_size
        # mu = cams.trans_pts_to_world(working_t, pts_cam)
        # fr = torch.eye(3)[None].to(pts_cam).expand(len(pts_cam), -1, -1)
        # s = radius[:, None].expand(-1, 3)
        # o = torch.ones_like(s[:, :1])
        # sph = RGB2SH(prior2d.rgbs[working_t][observed_dep_mask])
        # render_dep_dict = render(
        #     (mu, fr, s, o, sph),
        #     prior2d.H,
        #     prior2d.W,
        #     cams.rel_focal,
        #     cams.cxcy_ratio,
        #     T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        #     bg_color=[1.0, 0.0, 0.0],
        # )
        # viz_dep_rgb = render_dep_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        # viz_dep_rgb = np.clip(viz_dep_rgb, 0, 1)
        # viz_ret[-1] = np.concatenate([viz_ret[-1], viz_dep_rgb], axis=1)

        # #####################################################################

    return test_ret, test_latent_features  # , viz_ret


def render_test_tto(
    H,
    W,
    cams: SimpleFovCamerasIndependent,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir,
    save_pose_fn,
    ##
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    ###
    gt_mask_dir=None,
    save_dir=None,
    save_viz_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # viz
    viz_interval=50,
    # dbg
    use_sgd=False,
    loss_type="psnr",
):
    # * Optimize the test camera pose, nost simply do the global sim(3) alignment
    s_model.eval()
    d_model.eval()

    assert (
        gt_mask_dir is None
    ), "WARNING, FOR NOW DON'T USE EVAL MASK DURING TTO, ONLY TO FIND THE CORRECT CAMERA POSE!"

    device = s_model.device

    # first align the camera
    with torch.no_grad():
        solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
        )

    # render
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if save_viz_dir is not None:
        os.makedirs(save_viz_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    test_ret, viz_ret = [], []
    test_latent_features = []
    psnr_evaluator = PeakSignalNoiseRatio(data_range=1).to(device)
    solved_pose_list = []
    for i in tqdm(range(len(test_camera_tid))):
        viz_plot_flag = viz_interval > 0 and i % viz_interval == 0
        viz_vid_flag = viz_interval > 0 and i % viz_interval == 0

        working_t = test_camera_tid[i]
        # load gt rgb and mask
        img_path = osp.join(gt_rgb_dir, f"{fn_list[i]}.jpg")
        if not osp.exists(img_path):
            img_path = osp.join(gt_rgb_dir, f"{fn_list[i]}.png")
            if not osp.exists(img_path):
                raise ValueError(f"Cannot find image at {img_path}")
        gt_rgb = imageio.imread(img_path) / 255.0
        if gt_mask_dir is None:
            gt_mask = np.ones_like(gt_rgb[..., 0])
        else:
            gt_mask = imageio.imread(osp.join(gt_mask_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_mask = torch.tensor(gt_mask, device=device).float()
        gt_mask_sum = gt_mask.sum()

        T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device)
        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
        q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
        if use_sgd:
            optimizer_type = torch.optim.SGD
        else:
            optimizer_type = torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p},
                {"params": q_init, "lr": lr_q},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tto_steps - decay_start, eta_min=lr_final
        )

        loss_list, psnr_list = [], []
        viz_list = []

        with torch.no_grad():
            gs5 = [s_model(), d_model(working_t)]  # ! this does not change
        for _step in range(tto_steps):
            optimizer.zero_grad()
            _T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            T_cw = torch.cat([_T_cw, T_bottom[None]], 0)
            render_dict = render(gs5, H, W, focal, cxcy_ratio, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)

            if loss_type == "abs":
                rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None]
                rgb_loss = rgb_loss_i.sum() / gt_mask_sum
            elif loss_type == "psnr":
                rgb_loss = -psnr_evaluator(
                    pred_rgb.permute(2, 0, 1)[None], gt_rgb.permute(2, 0, 1)[None]
                )
            else:
                raise ValueError(f"Unknown loss tyoe {loss_type}")

            loss = rgb_loss
            loss.backward()
            optimizer.step()
            if _step >= decay_start:
                scheduler.step()

            loss_list.append(loss.item())
            if viz_plot_flag:
                psnr = psnr_evaluator(
                    pred_rgb.permute(2, 0, 1)[None], gt_rgb.permute(2, 0, 1)[None]
                )
                psnr_list.append(psnr.item())

            if viz_vid_flag:
                viz_error = (
                    (torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None])
                    .detach()
                    .cpu()
                    .numpy()
                )
                viz_error = cv.applyColorMap(
                    (viz_error * 255).astype(np.uint8), cv.COLORMAP_JET
                )
                viz_gt = (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                viz_pred = (pred_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                viz_frame = np.concatenate([viz_gt, viz_pred, viz_error], 1)
                viz_list.append(viz_frame)

        if viz_vid_flag:
            imageio.mimsave(osp.join(save_viz_dir, f"{fn_list[i]}_tto.mp4"), viz_list)

        if viz_plot_flag:
            fig = plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(loss_list), plt.title("TTO Loss")
            plt.subplot(1, 2, 2)
            plt.plot(psnr_list), plt.title("TTO PSNR")
            plt.tight_layout()
            plt.savefig(osp.join(save_viz_dir, f"{fn_list[i]}_tto.jpg"))
            plt.close()

        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        solved_pose_list.append(solved_T_cw.detach().cpu().numpy())
        with torch.no_grad():
            render_dict = render(
                [s_model(), d_model(working_t)],
                H,
                W,
                focal,
                cxcy_ratio,
                T_cw=T_cw,
            )
            rgb = render_dict["rgb"].detach().cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 1)  # ! important
            test_ret.append(rgb)
            semantic_feature = render_dict["feature_map"].detach().cpu()
            test_latent_features.append(semantic_feature)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
        logging.info(f"TTO {fn_list[i]}: {loss_list[0]:.3f}->{loss_list[-1]:.3f}")
    np.savez(save_pose_fn, poses=solved_pose_list)
    test_latent_features = torch.stack(test_latent_features, 0)
    return test_ret, test_latent_features
