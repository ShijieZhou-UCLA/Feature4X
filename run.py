import torch
import os, os.path as osp
import logging

import yaml

from lib_4d.cfg_helpers import OptimCFG, GSControlCFG
from lib_4d.solver_gs import Solver
from lib_prior.diffusion.sd_sds import StableDiffusionSDS
import numpy as np
from lib_4d.camera import SimpleFovCamerasIndependent
from lib_4d.gs_static_model import StaticGaussian
from lib_4d.gs_ed_model import DynSCFGaussian
from lib_4d.scf4d_model import Scaffold4D
import imageio
from omegaconf import OmegaConf
from lib_data.iphone_helpers import load_iphone_gt_poses
from lib_data.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test
from lib_data.nerfies_helpers import load_nerfies_gt_poses

from lib_4d.solver_viz_helper import viz_curve
from lib_4d.prior2d import Prior2D
from lib_4d.render_helper import GS_BACKEND

from lib_4d.solver_dynamic_funcs import solve_4dscf, grow_node_by_coverage

# from test import test_main  # Commented out - test_main is not used in current code
from lib_4d.autoencoder.model import Feature_heads
import shutil
import wandb
import PIL
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

def get_cfg(cfg_fn):
    cfg = OmegaConf.load(cfg_fn)
    for key in ["static_scf", "static_gs", "dyn_scf", "dyn_gs"]:
        if not hasattr(cfg, key):
            setattr(cfg, key, None)
    for key in ["d_ctrl", "s_ctrl"]:
        if not hasattr(cfg.dyn_gs, key):
            setattr(cfg.dyn_gs, key, None)
    if not hasattr(cfg.static_gs, "s_ctrl"):
        setattr(cfg.static_gs, "s_ctrl", None)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def main(
    args,
    cfg_fn,
    src,
    output_dir,
    device,
    sta_scf_dir=None,
    sta_gs_dir=None,
    dyn_scf_dir=None,
    dyn_gs_dir=None,
    depth_mode="uni",
    use_gt_cam=False,
    save_viz_flag=True,
):

    # get cfg
    cfg = get_cfg(cfg_fn)
    dataset_mode = getattr(cfg, "dataset_mode", "iphone")
    max_sph_order = getattr(cfg, "max_sph_order", 1)
    ######################################################################
    ######################################################################

    # * load gt camera if dataset has
    logging.info(f"Dataset mode: {dataset_mode}")
    if dataset_mode == "iphone":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(src, getattr(cfg, "t_subsample", 1))
    elif dataset_mode == "nerfies":
        (
            gt_training_cam_T_wi,
            gt_training_tids,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_nerfies_gt_poses(osp.join(src, "../../"), getattr(cfg, "t_subsample", 1))
    elif dataset_mode == "nvidia":
        (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio) = (
            load_nvidia_gt_pose(osp.join(src, "../poses_bounds.npy"))
        )
        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)

    else:
        gt_training_cam_T_wi = None
        gt_testing_cam_T_wi_list = []
        logging.info("No camera loaded, skip")

    ######################################################################
    ######################################################################

    with open(args.feature_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    NUM_SEMANTIC_CHANNELS = config["NUM_SEMANTIC_CHANNELS"]
    os.environ["NUM_SEMANTIC_CHANNELS"] = str(NUM_SEMANTIC_CHANNELS)

    log_prefix=f"{args.comment}_" + osp.basename(cfg_fn) + f"_compactgs_mixfeat_nomotion_channel{NUM_SEMANTIC_CHANNELS}_dep={depth_mode}_gt_cam={use_gt_cam}_lrfeat={args.lr_semantic_feature}_reversed={args.reverse}_"
    # * Get solver
    solver = Solver(
        src,
        output_dir,
        device,
        temporal_diff_shift=getattr(cfg, "temporal_diff_shift", [2, 8]),
        temporal_diff_weight=getattr(cfg, "temporal_diff_weight", [0.6, 0.4]),
        log_prefix=log_prefix if not args.debug else "tmp",
    )
    all_config = OmegaConf.to_container(cfg)
    all_config["head_config"] = head_config
    all_config["args"] = vars(args)
    if not args.debug:
        wandb_name =f"{args.comment}_{os.path.basename(output_dir)}_{os.path.basename(solver.log_dir)}"
        wandb.init(project="4dgs", name = wandb_name, config=all_config, dir=solver.log_dir, notes=args.comment,
                   settings=wandb.Settings(start_method='fork'))
    else:
        wandb.init(mode="disabled")

    # if using slurm, link the slurm log file(--output and --error) to the save folder
    # os.environ["EXP_LOG_DIR"] = solver.log_dir
    print(f"EXP_LOG_DIR:{solver.log_dir}")
    if os.environ.get("SLURM_JOB_ID") is not None:
        try:
            slurm_log_out_file = os.environ.get("SLURM_LOG_OUTPUT")
            slurm_log_err_file = os.environ.get("SLURM_LOG_ERROR")
        except:
            logging.warning(f"Cannot get slurm log file, all envs: {os.environ}")
        try:
            os.symlink(osp.abspath(slurm_log_out_file), osp.join(solver.log_dir, os.path.basename(slurm_log_out_file)))
            os.symlink(osp.abspath(slurm_log_err_file), osp.join(solver.log_dir, os.path.basename(slurm_log_err_file)))
        except Exception as e:
            logging.warning(f"Cannot link slurm log file, {e}")

    # save args
    with open(osp.join(solver.log_dir, "train_args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    solver.prior2d = Prior2D(
        dino_name=getattr(cfg, "dino_name", "dino"),
        log_dir=solver.log_dir,
        src_dir=src,
        working_device=device,
        depth_mode=depth_mode,
        min_valid_cnt=1,
        epi_error_th_factor=getattr(cfg, "epi_error_th_factor", 400.0),
        mask_prop_steps=getattr(cfg, "mask_prop_steps", 0),
        mask_consider_track=getattr(cfg, "mask_consider_track", False),
        mask_consider_track_dilate_radius=getattr(
            cfg, "mask_consider_track_dilate_radius", 7
        ),
        mask_init_erode=getattr(cfg, "mask_init_erode", 0),
        use_short_track=getattr(cfg, "use_short_flow", False),
        flow_interval=getattr(cfg, "flow_interval", [1]),
        semantic_th_quantile=getattr(cfg, "semantic_th_quantile", 0.95),
        depth_boundary_th=getattr(cfg, "depth_boundary_th", 0.5),
        nerfies_flag=getattr(cfg, "nerfies_flag", False),
        feature_config=args.feature_config,
    )
    with open(osp.join(solver.log_dir, "config_backup.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)  # backup
    shutil.copy(args.feature_config, osp.join(solver.log_dir, "feature_config.yaml"))


    ##########################  SEMANTIC FEATURE HEADS  ########################
    semantic_channel_dict = solver.prior2d.semantic_features.channels()
    logging.info(f"Semantic channels: {semantic_channel_dict}")

    feature_heads = Feature_heads(head_config).to(device)

    ######################################################################

    logging.info("Static Background Stage Start!")
    # * compute static scaffold

    if sta_scf_dir is not None:  # speed up debugging
        cams, s_track, s_track_mask, s_dep_corr = solver.load_static_scaffold(
            sta_scf_dir
        )
    else:
        if use_gt_cam:
            logging.info(f"Use GT camera")
            cams = solver.get_cams(
                fovdeg=float(gt_training_fov),
                gt_pose=gt_training_cam_T_wi,
                gt_fovdeg=float(gt_training_fov),
                cxcy_ratio=gt_training_cxcy_ratio[0],  # gt camera center
            )
        else:
            cams = solver.get_cams(fovdeg=getattr(cfg.static_scf, "fov_fallback", 40.0))
        # decide whether to do the static scaffold

        cams, s_track, s_track_mask, s_dep_corr = solver.compute_static_scaffold(
            cams=cams,
            gt_cam_flag=use_gt_cam,
            total_steps=(
                getattr(cfg.static_scf, "total_steps", 0 if use_gt_cam else 4000)
            ),
            lr_cam_f=0.0 if use_gt_cam else getattr(cfg.static_scf, "lr_cam_f", 0.0 if use_gt_cam else 0.0003),
            lr_cam_q=0.0 if use_gt_cam else getattr(cfg.static_scf, "lr_cam_q", 0.0003),
            lr_cam_t=0.0 if use_gt_cam else getattr(cfg.static_scf, "lr_cam_t", 0.0003),
            fov_search_min_interval=getattr(
                cfg.static_scf, "fov_search_min_interval", 2
            ),
            fov_interval_single=getattr(cfg.static_scf, "fov_interval_single", False),
        )
    if getattr(cfg.static_scf, "deform_depth", False):
        logging.warning(f"Deforming the depth!")
        solver.interpolate_by_depth_map(
            cams,
            s_track,
            s_track_mask,
            s_dep_corr[..., None],
            mask2d_type=getattr(cfg.static_scf, "deform_depth_mask", "dep"),
            K=getattr(cfg.static_scf, "deform_depth_K", 64),
        )

    ######################################################################
    ######################################################################

    # after the camera intrinsic is optimized, compute the normal of the depth map
    solver.prior2d.compute_normal_maps(
        cams,
        viz_flag=False,
        patch_size=getattr(cfg, "normal_patch_size", 7),
        nn_dist_th=getattr(cfg, "normal_nn_dist_th", 0.03),
        nn_min_cnt=getattr(cfg, "normal_nn_min_cnt", 4),
    )

    ######################################################################
    ######################################################################

    if sta_gs_dir is not None:
        assert (
            sta_scf_dir is not None
        ), "Must init the rescaled depth from static scaffold loading!"
        saved_cam = torch.load(osp.join(sta_gs_dir, "static_s_model_cam.pth"))
        cams.load_state_dict(saved_cam, strict=True)
        s_model = StaticGaussian(
            load_fn=osp.join(sta_gs_dir, "static_s_model.pth"),
            max_sph_order=max_sph_order,
        ).to(device)
        feature_heads.load_state_dict(
            torch.load(osp.join(sta_gs_dir, "static_semantic_heads.pth"))
        )

        logging.info(f"Load static model from {sta_gs_dir} done!")
    else:
        s_model = solver.get_static_model(
            cams=cams,
            max_sph_order=max_sph_order,
            n_static_init=getattr(cfg.static_gs, "n_init", 10000),
            normal_dir_ratio=getattr(
                cfg.static_gs, "normal_dir_ratio", 10.0 if GS_BACKEND == "gof" else 1.0
            ),
            radius_max=getattr(cfg.static_gs, "radius_max", 0.1),
        )
        solver.finetune_gs_model(
            semantic_heads=feature_heads,
            cams=cams,
            s_model=s_model,
            total_steps=getattr(cfg.static_gs, "total_steps", 4000),
            optim_cam_after_steps=getattr(cfg.static_gs, "optim_cam_after_steps", 3000),
            optimizer_cfg=OptimCFG(
                lr_cam_f=0.0,
                lr_cam_q=0.0 if use_gt_cam else 0.00003,
                lr_cam_t=0.0 if use_gt_cam else 0.00003,
                lr_p=getattr(cfg.static_gs, "lr_p", 0.0003),
                lr_q=getattr(cfg.static_gs, "lr_q", 0.002),
                lr_s=getattr(cfg.static_gs, "lr_s", 0.01),
                lr_o=getattr(cfg.static_gs, "lr_o", 0.1),
                lr_sph=getattr(cfg.static_gs, "lr_sph", 0.005),
                lr_p_final=getattr(cfg.static_gs, "lr_p_final", None),
                lr_sph_rest_factor=getattr(cfg.static_gs, "lr_sph_rest_factor", 20.0),
                lr_semantic_feature=args.lr_semantic_feature,
                lr_semantic_heads=args.lr_semantic_feature,
            ),
            s_gs_ctrl_cfg=GSControlCFG(
                densify_steps=getattr(cfg.static_gs.s_ctrl, "densify_steps", 300),
                reset_steps=getattr(cfg.static_gs.s_ctrl, "reset_steps", 900),
                prune_steps=getattr(cfg.static_gs.s_ctrl, "prune_steps", 300),
                densify_max_grad=getattr(
                    cfg.static_gs.s_ctrl, "densify_max_grad", 0.0002
                ),
                densify_percent_dense=getattr(
                    cfg.static_gs.s_ctrl, "densify_percent_dense", 0.01
                ),
                prune_opacity_th=getattr(
                    cfg.static_gs.s_ctrl, "prune_opacity_th", 0.012
                ),
                reset_opacity=getattr(cfg.static_gs.s_ctrl, "reset_opacity", 0.01),
            ),
            lambda_rgb=getattr(cfg.static_gs, "lambda_rgb", 1.0),
            lambda_dep=getattr(cfg.static_gs, "lambda_dep", 0.5),
            lambda_normal=getattr(cfg.static_gs, "lambda_normal", 0.5),
            dep_st_invariant=getattr(cfg.static_gs, "dep_st_invariant", True),
            # gt_cam_flag=use_gt_cam,
            gt_cam_flag=False,  # ! always optimize with photometric
            phase_name="static",
            sup_mask_type="sta",
            viz_interval=getattr(cfg.static_gs, "viz_interval", -1),
            viz_cheap_interval=getattr(cfg.static_gs, "viz_cheap_interval", 1000),
            viz_skip_t=1 if cams.T < 120 else max(1, cams.T // 50),
            viz_move_angle_deg=getattr(cfg.static_gs, "viz_move_angle_deg", 30.0),
            random_bg=getattr(cfg.static_gs, "random_bg", True),
        )
    logging.info("Static Done, now start Dynamic Foreground Stage!")

    ######################################################################
    ######################################################################

    # * Update the dynamic mask and tracks
    if getattr(cfg, "recomp_dyn_mask", True):
        solver.recompute_dynamic_masks_and_tracks(
            s_model,
            cams,
            consider_inside_dyn=getattr(
                cfg, "recomp_dyn_mask_consider_inside_dyn", False
            ),
        )
    solver.specify_spatial_unit(
        unit=getattr(cfg, "spatial_unit_meter", 0.04),
        world_flag=getattr(cfg, "spatial_unit_world_flag", True),
    )

    ######################################################################
    ######################################################################

    # * compute dynamic scaffold
    if dyn_scf_dir is not None:
        logging.info(f"Load dynamic scaffold from {dyn_scf_dir}")
        if osp.exists(osp.join(dyn_scf_dir, "stage4_" + "dynamic_scaffold_init.pth")):
            dyn_scf_fn = osp.join(dyn_scf_dir, "stage4_" + "dynamic_scaffold_init.pth")
        else:
            dyn_scf_fn = osp.join(dyn_scf_dir, "stage3_" + "dynamic_scaffold_init.pth")
        saved_dyn_scf_ckpt = torch.load(dyn_scf_fn)
        scf = Scaffold4D.load_from_ckpt(saved_dyn_scf_ckpt, device=device)
    else:
        scf, t_list = solver.get_dynamic_scaffold(
            cams=cams,
            skinning_method=getattr(cfg.dyn_scf, "skinning_method", "dqb"),
            skinning_topology=getattr(cfg.dyn_scf, "skinning_topology", "graph"),
            topo_k=getattr(cfg.dyn_scf, "topo_k", 16),
            topo_curve_dist_top_k=getattr(cfg.dyn_scf, "topo_curve_dist_top_k", 8),
            topo_curve_dist_sample_T=getattr(
                cfg.dyn_scf, "topo_curve_dist_sample_T", 80
            ),
            max_node_num=getattr(cfg.dyn_scf, "max_node_num", 30000),
            topo_th_ratio=getattr(cfg.dyn_scf, "topo_th_ratio", 10.0),
            sigma_max_ratio=getattr(cfg.dyn_scf, "sigma_max_ratio", 1.0),
            sigma_init_ratio=getattr(cfg.dyn_scf, "sigma_init_ratio", 0.2),
            vel_jitter_th_value=getattr(cfg.dyn_scf, "vel_jitter_th_value", 0.1),
            min_valid_cnt_ratio=getattr(cfg.dyn_scf, "min_valid_cnt_ratio", 0.1),
            mlevel_list=getattr(cfg.dyn_scf, "mlevel_list", [1, 8]),
            mlevel_k_list=getattr(cfg.dyn_scf, "mlevel_k_list", [16, 8]),
            mlevel_w_list=getattr(cfg.dyn_scf, "mlevel_w_list", [0.4, 0.3]),
            gs_sk_approx_flag=getattr(cfg.dyn_scf, "gs_sk_approx_flag", False),
            dyn_o_flag=getattr(cfg.dyn_scf, "dyn_o_flag", False),
            resample_flag=getattr(cfg.dyn_scf, "resample_flag", True),
            # ! abl
            mlevel_arap_flag=getattr(cfg.dyn_scf, "mlevel_arap_flag", True),
        )
        solve_4dscf(
            prior2d=solver.prior2d,
            scf=scf,
            cams=cams,
            viz_dir=solver.viz_dir,
            log_dir=solver.log_dir,
            mlevel_resample_steps=getattr(cfg.dyn_scf, "mlevel_resample_steps", 32),
            lr_p=getattr(cfg.dyn_scf, "lr_p", 0.1),
            lr_q=getattr(cfg.dyn_scf, "lr_q", 0.1),
            lr_sig=getattr(cfg.dyn_scf, "lr_sig", 0.03),
            #
            lr_p_finetune=getattr(cfg.dyn_scf, "lr_p_finetune", 0.01),
            lr_q_finetune=getattr(cfg.dyn_scf, "lr_q_finetune", 0.01),
            lr_sig_finetune=getattr(cfg.dyn_scf, "lr_sig_finetune", 0.0),
            #
            stage1_steps=getattr(cfg.dyn_scf, "stage1_steps", 300),
            stage1_decay_start_ratio=getattr(
                cfg.dyn_scf, "stage1_decay_start_ratio", 0.5
            ),
            stage1_decay_factor=getattr(cfg.dyn_scf, "stage1_decay_factor", 100.0),
            temporal_diff_shift=solver.temporal_diff_shift,
            temporal_diff_weight=solver.temporal_diff_weight,
            #
            n_flow_pair=getattr(cfg.dyn_scf, "n_flow_pair", 50),
            stage2_steps=getattr(cfg.dyn_scf, "stage2_steps", 300),
            stage2_decay_start_ratio=getattr(
                cfg.dyn_scf, "stage2_decay_start_ratio", 0.5
            ),
            stage2_decay_factor=getattr(cfg.dyn_scf, "stage2_decay_factor", 100.0),
            #
            stage3_steps=getattr(cfg.dyn_scf, "stage3_steps", 300),
            stage3_decay_start_ratio=getattr(
                cfg.dyn_scf, "stage3_decay_start_ratio", 0.5
            ),
            stage3_decay_factor=getattr(cfg.dyn_scf, "stage3_decay_factor", 100.0),
            #
            stage4_steps=getattr(cfg.dyn_scf, "stage4_steps", 300),
            stage4_decay_start_ratio=getattr(
                cfg.dyn_scf, "stage4_decay_start_ratio", 0.5
            ),
            stage4_decay_factor=getattr(cfg.dyn_scf, "stage4_decay_factor", 100.0),
            viz_interval=getattr(cfg.dyn_scf, "viz_interval", 100),
            resample_flag=getattr(cfg.dyn_scf, "resample_flag", True),
            # ! ABL
            no_baking_flag=getattr(cfg, "abl_no_baking_flag", False),
            no_semantic_drag_flag=getattr(cfg, "abl_no_semantic_drag_flag", False),
            no_geo_flag=getattr(cfg, "abl_no_geo_flag", False),
        )

    if cams.T != scf.T:
        logging.info(
            f"SCF has subsampled time {scf.T} while prior2d and cam has {cams.T} frames, resample the time dim"
        )
        scf.resample_time(torch.arange(0, cams.T, 1))
        viz_frame = viz_curve(
            scf._node_xyz.detach(),
            scf._curve_color_init,
            semantic_feature=scf._node_semantic_feature,
            mask = scf._curve_slot_init_valid_mask,
            cams = cams,
            viz_n=-1,
            time_window=1,
            res=480,
            pts_size=0.001,
            only_viz_last_frame=False,
            no_rgb_viz=True,
            n_line=4,
            text=f"resample",
        )
        imageio.mimsave(
            osp.join(solver.viz_dir, f"t-resampled-scf.mp4"),
            viz_frame,
        )

    ######################################################################
    ######################################################################

    if getattr(cfg.dyn_scf, "grow_node_by_coverage", True):
        grow_node_by_coverage(
            grow_interval=getattr(cfg.dyn_scf, "grow_interval", 1),
            prior2d=solver.prior2d,
            scf=scf,
            cams=cams,
            matching_method="sem",
            spatial_radius_ratio=3.0,
            rgb_std_ratio=3.0,
            feat_std_ratio=3.0,
            viz_dir=solver.viz_dir,
        )
    else:
        # make sure the scf is re-sampled
        if not getattr(cfg.dyn_scf, "resample_flag", True):
            logging.info("make sure the scf is resampled before sending to d_model")
            scf.resample_node(1.0)

    ######################################################################
    ######################################################################
    if dyn_gs_dir is not None:
        # * directly load
        d_model_ckpt = torch.load(osp.join(dyn_gs_dir, "finetune_d_model.pth"))
        d_model = DynSCFGaussian.load_from_ckpt(d_model_ckpt, device=device)
        # * load static model and camera again, because it's also finetuned
        saved_cam = torch.load(osp.join(dyn_gs_dir, "finetune_s_model_cam.pth"))
        cams.load_state_dict(saved_cam, strict=True)
        s_model = StaticGaussian(
            load_fn=osp.join(dyn_gs_dir, "finetune_s_model.pth"),
            max_sph_order=max_sph_order,
        ).to(device)
        feature_heads.load_state_dict(
            torch.load(osp.join(dyn_gs_dir, "finetune_semantic_heads.pth"))
        )
    else:
        d_model = solver.get_dynamic_model(
            topo_th_ratio=getattr(cfg.dyn_gs, "topo_th_ratio", 3.0),
            cams=cams,
            scf=scf,
            max_sph_order=max_sph_order,
            image_stride=getattr(cfg.dyn_gs, "model_pixel_subsample", 1),
            n_init=getattr(cfg.dyn_gs, "n_init", 10000),
            end_t=cams.T - 1,
            attach_t_interval=max(
                1, cams.T // getattr(cfg.dyn_gs, "init_key_frames", 10)
            ),
            # opa_init_value=0.99,
            opa_init_value=getattr(cfg.dyn_gs, "opa_init_value", 0.99),
            leaf_local_flag=getattr(cfg.dyn_gs, "leaf_local_flag", True),
            # normal_dir_ratio=getattr(cfg.dyn_gs, "normal_dir_ratio", 100.0),
            normal_dir_ratio=getattr(
                cfg.static_gs, "normal_dir_ratio", 10.0 if GS_BACKEND == "gof" else 1.0
            ),
            # reference settings
            canonical_ref_flag=getattr(cfg.dyn_gs, "canonical_ref_flag", False),
            canonical_tid_mode=getattr(cfg.dyn_gs, "canonical_tid_mode", "largest"),
            # abl
            abl_nn_fusion=getattr(cfg, "abl_nn_fusion", -1),
        )
        solver.finetune_gs_model(
            semantic_heads=feature_heads,
            total_steps=getattr(cfg.dyn_gs, "total_steps", 8000),
            optim_cam_after_steps=getattr(cfg.dyn_gs, "optim_cam_after_steps", 5000),
            skinning_corr_start_steps=getattr(
                cfg.dyn_gs, "skinning_corr_start_steps", 7000
            ),
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            # losses
            lambda_rgb=getattr(cfg.dyn_gs, "lambda_rgb", 1.0),
            lambda_dep=getattr(cfg.dyn_gs, "lambda_dep", 0.05),
            dep_st_invariant=getattr(cfg.dyn_gs, "dep_st_invariant", True),
            lambda_normal=getattr(cfg.dyn_gs, "lambda_normal", 0.05),
            lambda_depth_normal=getattr(cfg.dyn_gs, "lambda_depth_normal", 0.05),
            lambda_distortion=getattr(cfg.dyn_gs, "lambda_distortion", 100.0),
            lambda_vel_xyz_reg=getattr(cfg.dyn_gs, "lambda_vel_xyz_reg", 0.0),
            lambda_vel_rot_reg=getattr(cfg.dyn_gs, "lambda_vel_rot_reg", 0.0),
            lambda_acc_rot_reg=getattr(cfg.dyn_gs, "lambda_acc_rot_reg", 1.0),
            lambda_acc_xyz_reg=getattr(cfg.dyn_gs, "lambda_acc_xyz_reg", 1.0),
            lambda_arap_coord=getattr(cfg.dyn_gs, "lambda_arap_coord", 3.0),
            lambda_arap_len=getattr(cfg.dyn_gs, "lambda_arap_len", 3.0),
            physical_reg_until_step=getattr(
                cfg.dyn_gs, "physical_reg_until_step", 100000000000
            ),
            geo_reg_start_steps=getattr(cfg.dyn_gs, "geo_reg_start_steps", 0),
            reg_radius=getattr(cfg.dyn_gs, "reg_radius", None),
            gt_cam_flag=False,  # ! always optimize with photometric
            reset_at_beginning=getattr(cfg.dyn_gs, "reset_at_beginning", False),
            optimizer_cfg=OptimCFG(
                lr_cam_f=0.0,
                lr_cam_q=0.0 if use_gt_cam else 0.00003,  # ! always can optimize the camera
                lr_cam_t=0.0 if use_gt_cam else 0.00003,
                # gs
                lr_p=getattr(cfg.dyn_gs, "lr_p", 0.00016),
                lr_q=getattr(cfg.dyn_gs, "lr_q", 0.001),
                lr_s=getattr(cfg.dyn_gs, "lr_s", 0.005),
                lr_o=getattr(cfg.dyn_gs, "lr_o", 0.05),
                lr_sph=getattr(cfg.dyn_gs, "lr_sph", 0.0025),
                lr_sph_rest_factor=getattr(cfg.dyn_gs, "lr_sph_rest_factor", 20.0),
                lr_semantic_feature=args.lr_semantic_feature,
                lr_semantic_heads=args.lr_semantic_feature,
                lr_p_final=0.00016 / 100.0,
                # node
                lr_np=getattr(cfg.dyn_gs, "lr_np", 0.00016),
                lr_nq=getattr(cfg.dyn_gs, "lr_nq", 0.00016),
                lr_nsig=getattr(cfg.dyn_gs, "lr_nsig", 0.003),
                lr_np_final=getattr(cfg.dyn_gs, "lr_np_final", 0.00016 / 100.0),
                lr_nq_final=getattr(cfg.dyn_gs, "lr_nq_final", 0.00016 / 100.0),
                lr_sk_q=0.00016,  # ! debug, to tune
                lr_w=getattr(cfg.dyn_gs, "lr_w", 0.1),
                lr_w_final=getattr(
                    cfg.dyn_gs, "lr_w_final", getattr(cfg.dyn_gs, "lr_w", 0.1) / 10.0
                ),
            ),
            d_gs_ctrl_cfg=GSControlCFG(
                densify_steps=getattr(cfg.dyn_gs.d_ctrl, "densify_steps", 300),
                # densify_steps=10, # debug
                reset_steps=getattr(cfg.dyn_gs.d_ctrl, "reset_steps", 2000),
                prune_steps=getattr(cfg.dyn_gs.d_ctrl, "prune_steps", 300),
                densify_max_grad=getattr(
                    cfg.dyn_gs.d_ctrl, "densify_max_grad", 0.00012
                ),
                densify_percent_dense=getattr(
                    cfg.dyn_gs.d_ctrl, "densify_percent_dense", 0.01
                ),
                prune_opacity_th=getattr(cfg.dyn_gs.d_ctrl, "prune_opacity_th", 0.05),
                reset_opacity=getattr(cfg.dyn_gs.d_ctrl, "reset_opacity", 0.01),
            ),
            s_gs_ctrl_cfg=GSControlCFG(
                densify_steps=getattr(cfg.dyn_gs.s_ctrl, "densify_steps", 1200),
                reset_steps=getattr(cfg.dyn_gs.s_ctrl, "reset_steps", 1501),
                prune_steps=getattr(cfg.dyn_gs.s_ctrl, "prune_steps", 300),
                densify_max_grad=getattr(cfg.dyn_gs.s_ctrl, "densify_max_grad", 0.0008),
                densify_percent_dense=getattr(
                    cfg.dyn_gs.s_ctrl, "densify_percent_dense", 0.01
                ),
                prune_opacity_th=getattr(cfg.dyn_gs.s_ctrl, "prune_opacity_th", 0.05),
                reset_opacity=getattr(cfg.dyn_gs.s_ctrl, "reset_opacity", 0.01),
            ),
            s_gs_ctrl_start_ratio=getattr(cfg.dyn_gs, "s_gs_ctrl_start_ratio", 0.01),
            d_gs_ctrl_start_ratio=getattr(cfg.dyn_gs, "d_gs_ctrl_start_ratio", 0.1),
            # NODE CONTROL
            dyn_error_grow_steps=getattr(cfg.dyn_gs, "dyn_error_grow_steps", []),
            dyn_error_grow_th=getattr(cfg.dyn_gs, "dyn_error_grow_th", 0.2),
            dyn_error_grow_num_frames=getattr(
                cfg.dyn_gs, "dyn_error_grow_num_frames", 4
            ),
            dyn_scf_prune_steps=getattr(cfg.dyn_gs, "dyn_scf_prune_steps", []),
            dyn_scf_prune_sk_th=getattr(cfg.dyn_gs, "dyn_scf_prune_sk_th", 0.02),
            # viz
            viz_skip_t=1 if cams.T < 120 else max(1, cams.T // 50),
            viz_interval=getattr(cfg.dyn_gs, "viz_interval", 999),
            viz_cheap_interval=getattr(cfg.dyn_gs, "viz_cheap_interval", 1000),
            viz_move_angle_deg=getattr(cfg.dyn_gs, "viz_move_angle_deg", 30.0),
            viz_ref_train_camera_T_wc=gt_training_cam_T_wi,
            viz_test_camera_T_wc_list=(
                [T[0] for T in gt_testing_cam_T_wi_list]
                if len(gt_testing_cam_T_wi_list) > 0
                else None
            ),
            random_bg=getattr(cfg.dyn_gs, "random_bg", True),
            # OLD
            freeze_static_after=getattr(cfg.dyn_gs, "freeze_static_after", 2000),
            unfreeze_static_after=getattr(cfg.dyn_gs, "unfreeze_static_after", 7000),
        )
    d_model.summary()
    logging.info(f"finish optim, d_model has {d_model.M} nodes")

    wandb.finish()

    # output trained semantic feature field here
    solver.get_semantic_feature_map(
        cams = cams,
        s_model = s_model, 
        d_model = d_model
                                    
    )

    if save_viz_flag:
        #try:
            viz_dir = osp.join(solver.log_dir, "final_viz")
            logging.info(f"Start viz to {viz_dir}...")
            from viz import viz_main

            viz_main(
                save_dir=viz_dir,
                log_dir=dyn_gs_dir if dyn_gs_dir is not None else solver.log_dir,
                cfg_fn=cfg_fn,
                N=2,
                H=solver.prior2d.H,
                W=solver.prior2d.W,
                move_angle_deg=5 #10.0,
            )
            #torch.cuda.empty_cache()
        #except:
            #logging.warning(f"VIZ fail, skip")

    # logging.info("Start Testing...")
    # if dataset_mode == "iphone":
    #     test_main(
    #         cfg=cfg,
    #         saved_dir=solver.log_dir,
    #         data_root=src,
    #         device=solver.device,
    #         tto_flag=getattr(cfg, "tto_flag", True),
    #     )
    # elif dataset_mode == "nerfies":
    #     test_main(
    #         cfg=cfg,
    #         saved_dir=solver.log_dir,
    #         data_root=src,
    #         device=solver.device,
    #         tto_flag=True,  # ! for now don't use tto for nerfies
    #     )
    # elif dataset_mode == "nvidia":
    #     test_main(
    #         cfg=cfg,
    #         saved_dir=solver.log_dir,
    #         data_root=src,
    #         device=solver.device,
    #         tto_flag=True,
    #     )

    # logging.info(f"Finished, saved to {solver.log_dir}")

    # if using slurm, move the slurm log file(--output and --error) to the save folder
    if os.environ.get("SLURM_JOB_ID") is not None:
        try:
            slurm_log_out_file = os.environ.get("SLURM_LOG_OUTPUT")
            slurm_log_err_file = os.environ.get("SLURM_LOG_ERROR")
        except:
            logging.warning(f"Cannot get slurm log file, all envs: {os.environ}")
        try:
            os.remove(osp.join(solver.log_dir, os.path.basename(slurm_log_out_file)))
            os.remove(osp.join(solver.log_dir, os.path.basename(slurm_log_err_file)))
            shutil.copy(slurm_log_out_file, solver.log_dir)
            shutil.copy(slurm_log_err_file, solver.log_dir)
        except Exception as e:
            logging.warning(f"Cannot copy slurm log file, {e}")

    return


if __name__ == "__main__":
    import argparse
    import time

    # Record the start time
    start_time = time.time()

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", type=str, required=True)
    args.add_argument("--src", "-s", type=str, required=True)
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--lr_semantic_feature", type=float, default=0.01)
    args.add_argument("--debug", action="store_true")

    args.add_argument("--sta_scf_dir", type=str, default=None)
    args.add_argument("--sta_gs_dir", type=str, default=None)
    args.add_argument("--dyn_scf_dir", type=str, default=None)
    args.add_argument("--dyn_gs_dir", type=str, default=None)

    args.add_argument("--depth_mode", type=str, default="uni")
    args.add_argument("--gt_cam", action="store_true")

    args.add_argument("--reverse", type=bool, default=False)
    args.add_argument("--feature_config", type=str, default="configs/default_config.yaml")
    args.add_argument("--comment", type=str, default="default")
    args.add_argument("--save_dir", type=str, default="output")
    args = args.parse_args()
    print(args.__dict__)


    if args.reverse is False:
        src = osp.join(args.src,"preprocess")
        if not os.path.exists(src):
            src = osp.join(args.src,"code_output")
            if not os.path.exists(src):
                raise ValueError(f"No preprocess or code_output folder found in {args.src}")
    else:
        src = osp.join(args.src,"preprocess_reversed")

    data_name = os.path.basename(os.path.normpath(args.src))
    output_dir = osp.join(args.save_dir, data_name)
    os.makedirs(output_dir, exist_ok = True)

    main(
        args,
        cfg_fn=args.config,
        src=src,
        output_dir = output_dir,
        device=torch.device(args.device),
        depth_mode=args.depth_mode,
        use_gt_cam=args.gt_cam,
        sta_scf_dir=args.sta_scf_dir,
        sta_gs_dir=args.sta_gs_dir,
        dyn_scf_dir=args.dyn_scf_dir,
        dyn_gs_dir=args.dyn_gs_dir,
    )

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")

    
    
 