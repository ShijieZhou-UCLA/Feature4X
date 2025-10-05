import torch
import os, os.path as osp
import logging
import numpy as np
from lib_4d.camera import SimpleFovCamerasIndependent
from lib_4d.gs_static_model import StaticGaussian
from lib_4d.gs_ed_model import DynSCFGaussian
import imageio
from omegaconf import OmegaConf
from lib_4d.render_helper import GS_BACKEND
from lib_4d.figure_viz_helper import *
from lib_4d.autoencoder.model import Feature_heads
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

def load_model_cfg(cfg, log_dir, device=torch.device("cuda")):

    # get cfg
    if log_dir.endswith("/"):
        log_dir = log_dir[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "dataset_mode", "iphone")
    max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    d_model_ckpt = torch.load(osp.join(log_dir, "finetune_d_model.pth"))
    d_model = DynSCFGaussian.load_from_ckpt(d_model_ckpt, device=device)
    # * load static model and camera again, because it's also finetuned
    saved_cam = torch.load(osp.join(log_dir, "finetune_s_model_cam.pth"))
    cams: SimpleFovCamerasIndependent = SimpleFovCamerasIndependent(
        T=len(saved_cam["q_wc"]),
        fovdeg_init=40.0,  # dummy init
    )
    cams.load_state_dict(saved_cam, strict=True)

    s_model = StaticGaussian(
        load_fn=osp.join(log_dir, "finetune_s_model.pth"),
        max_sph_order=max_sph_order,
    ).to(device)

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    # feature_head
    feature_config = osp.join(log_dir, "feature_config.yaml")
    if osp.exists(feature_config):
        with open(feature_config, "r") as f:
            feature_config = OmegaConf.load(f)
            head_config = feature_config["Head"]

        feature_head = Feature_heads(head_config).to(device)

        feature_head_ckpt_path = osp.join(log_dir, "finetune_semantic_heads.pth")
        feature_head_state = torch.load(feature_head_ckpt_path, weights_only=True)
        feature_head.load_state_dict(feature_head_state)
        feature_head.eval()
    else:
        print(f"feature head config {feature_config} not found")
        feature_head = None


    return cfg, d_model, s_model, cams, feature_head


@torch.no_grad()
def viz_main(
    save_dir,
    log_dir,
    cfg_fn,
    N=1,
    H=480,
    W=854,
    move_angle_deg=5, #5.0, #10.0,
    H_3d=960,
    W_3d=960,
    fov_3d=70,
    save_lseg=False,
):
    os.makedirs(save_dir, exist_ok = True)

    cfg, d_model, s_model, cams, feature_head = load_model_cfg(cfg_fn, log_dir)

    rel_focal_3d = 1.0 / np.tan(np.deg2rad(fov_3d) / 2.0)

    key_steps = [cams.T // 2, cams.T - 1, 0, cams.T // 4, 3 * cams.T // 4][:N]

    # * Get pose
    global_pose_list = get_global_3D_cam_T_cw(
        s_model,
        d_model,
        cams,
        H,
        W,
        cams.T // 2,
        back_ratio=0.5,
        up_ratio=0.2,
    )
    global_pose_list = global_pose_list[None].expand(cams.T, -1, -1)
    training_pose_list = [cams.T_cw(t) for t in range(cams.T)]

    # * #############################################################################
    # viz 3D
    save_fn_prefix = osp.join(save_dir, f"3D_moving")
    viz_single_2d_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        feature_head=feature_head
    )

    save_fn_prefix = osp.join(save_dir, f"3D_moving_node")
    viz_single_2d_node_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
    )

    save_fn_prefix = osp.join(save_dir, f"3D_moving_flow")
    viz_single_2d_flow_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
    )

    # flow
    save_fn_prefix = osp.join(save_dir, f"training_moving_flow")
    viz_single_2d_flow_video(
        H, W, cams, s_model, d_model, save_fn_prefix, training_pose_list
    )
    # node
    save_fn_prefix = osp.join(save_dir, f"training_moving_node")
    viz_single_2d_node_video(
        H, W, cams, s_model, d_model, save_fn_prefix, training_pose_list
    )
    # rgb
    save_fn_prefix = osp.join(save_dir, f"training_moving")
    viz_single_2d_video(
        H, W, cams, s_model, d_model, save_fn_prefix, training_pose_list, feature_head=feature_head,
        save_lseg=save_lseg
    )

    # * #############################################################################
    # key_time_step = cams.T // 2
    for key_time_step in key_steps:
        fixed_pose_list = [cams.T_cw(key_time_step) for _ in range(cams.T)]
        round_pose_list = get_move_around_cam_T_cw(
            s_model,
            d_model,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )

        # viz flow
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_flow")
        logging.info(f"Vizing fixed_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list
        )
        logging.info(f"Vizing round_moving_flow")
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        )
        # Viz node

        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_node")
        logging.info(f"Vizing round_moving_node")
        viz_single_2d_node_video(
            H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing_node")
        logging.info(f"Vizing round_freezing_node")
        viz_single_2d_node_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_node")
        logging.info(f"Vizing fixed_moving_node")
        viz_single_2d_node_video(
            H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list,
        )

        # Viz rgb
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving")
        logging.info(f"Vizing round_moving")
        viz_single_2d_video(
            H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list,
            feature_head=feature_head, save_lseg=save_lseg
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing")
        logging.info(f"Vizing round_freezing")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
            feature_head=feature_head
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving")
        logging.info(f"Vizing fixed_moving")
        viz_single_2d_video(
            H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list,  feature_head=feature_head
        )

    return


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", type=str, required=True)
    args.add_argument("--root", "-r", type=str, required=True)
    args.add_argument("--save", "-s", type=str, required=True)
    args.add_argument("--N", "-n", type=int, default=1)
    args.add_argument("--H", type=int, default=480)
    args.add_argument("--W", type=int, default=480)
    args.add_argument("--save_lseg", action="store_true")
    args = args.parse_args()

    viz_main(
        args.save,
        args.root,
        args.config,
        N=2,
        H=args.H,
        W=args.W,
        save_lseg=args.save_lseg,
    )
