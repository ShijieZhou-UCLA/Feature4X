# run preprocessing, compute 2D priors

import sys, os, os.path as osp
import torch, numpy as np, cv2 as cv
import time, logging
from omegaconf import OmegaConf
from glob import glob
from PIL import Image
from torchvision import transforms
import cv2

from lib_4d.graph_utils import *
from lib_4d.lib_4d_misc import configure_logging

from lib_prior.optical_flow.raft_wrapper import get_raft_model, raft_process_folder
from lib_prior.motion_mask import epi_motion_mask_from_optical_flow_process_folder
from lib_prior.seg_track.segformer_wrapper import (
    get_segformer_model,
    segformer_sky_process_folder,
    dummy_segformer_sky_process_folder,
)
from lib_prior.tracking.cotracker_wrapper import (
    get_cotracker,
    process_folder_track,
)

from lib_prior.superfeat.dino_wrapper import (
    get_dino_model,
    dino_process_folder,
    get_feat_up_model,
    dino_featup_process_folder,
)
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

def preprocess_views(src, intervals=[1, 8], dense_flag=False, nerfies_flag=False):
    logging.info(f"Preprocess views start...")
    image_fns = get_view_list(src, nerfies_flag=False)
    print("views:", image_fns)

    if dense_flag:
        view_pairs = get_dense_sub_complete_graph_pairs(image_fns, intervals)
    else:
        view_pairs = get_interval_sub_complete_graph_pairs(image_fns, intervals)

    save_pairs_to_txt(view_pairs, osp.join(src, "view_pairs.txt"))
    view_pairs = read_pairs_from_txt(osp.join(src, "view_pairs.txt"))

    viz_view_pairs(
        image_fns, view_pairs, save_path=osp.join(src, "view_pairs.jpg"), show=False
    )
    print(f"Has {len(view_pairs)} view pairs")
    print("pairs:", view_pairs)
    logging.info(f"Preprocess views end!")
    return image_fns, view_pairs


def compute_raft(src, view_pairs, device):
    logging.info(f"Compute RAFT start...")
    raft_model = get_raft_model("./weights/raft_models/raft-things.pth", device)
    raft_process_folder(
        raft_model,
        osp.join(src, "images"),
        osp.join(src, "flows"),
        osp.join(src, "flow_viz"),
        view_pairs,
    )

    raft_model.cpu()
    del raft_model
    torch.cuda.empty_cache()
    logging.info(f"Compute RAFT end!")
    return


def compute_epi_error(src):
    logging.info(f"Compute motion mask start...")
    epi_motion_mask_from_optical_flow_process_folder(src)
    # epi_motion_mask_from_optical_flow_process_folder(src, use_maskrcnn=True)
    logging.info(f"Compute motion mask end!")
    return


def compute_zoe_depths(src, device, mode="K"):
    from lib_prior.mono_depth.zoedepth_wrapper import (
        get_zoedepth_model,
        zoedepth_process_folder,
    )

    logging.info(f"Compute ZoeDepth with {mode} start...")
    # * ZoeDepth
    # zoedepth_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zoedepth_device = device
    # zoedepth_model = get_zoedepth_model(device=zoedepth_device, type="NK")
    zoedepth_model = get_zoedepth_model(device=zoedepth_device, type=mode)
    zoedepth_process_folder(
        zoedepth_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "zoe_depth"),
    )

    zoedepth_model.cpu()
    del zoedepth_model
    torch.cuda.empty_cache()
    logging.info(f"Compute ZoeDepth end!")
    return


def compute_uni_depths(src, device):
    from lib_prior.mono_depth.unidepth_wrapper import (
        get_unidepth_model,
        unidepth_process_folder,
    )

    logging.info(f"Compute Unidepth start...")
    # * ZoeDepth
    unidepth_model = get_unidepth_model(device=device)
    unidepth_process_folder(
        unidepth_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "uni_depth"),
    )

    unidepth_model.cpu()
    del unidepth_model
    torch.cuda.empty_cache()
    logging.info(f"Compute UniDepth end!")
    return


def compute_any_depths(src, device):
    raise RuntimeError("Not use this!")
    from lib_prior.mono_depth.depthanything_wrapper import (
        get_zoedepth_anything_model,
        zoedepth_anything_process_folder,
    )

    logging.info(f"Compute AnyDepth start...")
    # * DepthAnything Zoe FineTuned
    depthanything_mode = get_zoedepth_anything_model(device)
    zoedepth_anything_process_folder(
        depthanything_mode,
        src=osp.join(src, "images"),
        dst=osp.join(src, "anyzoe_depth"),
    )
    depthanything_mode.cpu()
    del depthanything_mode
    torch.cuda.empty_cache()
    logging.info(f"Compute AnyDepth end!")
    return


def compute_mar_depths(src, device):
    raise RuntimeError("Not use this!")
    from lib_prior.mono_depth.marigold_wrapper import (
        get_marigold_model,
        marigold_process_folder,
    )

    logging.info(f"Compute MarigoldDepth start...")
    # * Marigold depth
    model = get_marigold_model(device=device)
    marigold_process_folder(
        model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "marigold_depth"),
        denoise_steps=5,
        ensemble_size=3,
    )

    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    logging.info(f"Compute MarigoldDepth end!")
    return


def maskout_sky(src, device, dummy_flag=True):
    if dummy_flag:
        logging.info(f"Dummy mask out sky start...")
        dummy_segformer_sky_process_folder(
            src=osp.join(src, "images"), dst=osp.join(src, "segformer_sky_mask")
        )
        logging.info(f"Dummy mask out sky end!")
        return
    logging.info(f"Mask out sky start...")
    segformer_feat_extractor, segformer_model = get_segformer_model(device=device)
    segformer_sky_process_folder(
        segformer_feat_extractor,
        segformer_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "segformer_sky_mask"),
    )
    torch.cuda.empty_cache()
    logging.info(f"Mask out sky end!")
    return


def get_long_tracking(
    src,
    device,
    d_num_pts,
    s_num_pts,
    d_num_chunk,
    s_num_chunk,
    epi_th,
    keyframe_interval,
    dyn_subsample,
):
    logging.info(
        f"Get long tracking dyn_num_pts={d_num_pts} sta_num_pts={s_num_pts} start..."
    )
    model = get_cotracker(device)
    process_folder_track(
        src,
        model,
        torch.device("cuda"),
        dyn_total_n_pts=d_num_pts,
        sta_total_n_pts=s_num_pts,
        sta_chunk_n_pts=s_num_chunk,
        dyn_chunk_n_pts=d_num_chunk,
        dyn_epi_th=epi_th,
        keyframe_interval=keyframe_interval,
        dyn_subsample=dyn_subsample,
    )
    model.cpu()
    del model
    logging.info(f"Get long tracking end!")
    return


def compute_dino_superfeat(src, device, dino_mode="giant", pca_dim=64):
    logging.info(f"Compute Dino SuperFeat start ({dino_mode})...")
    dino_model, dino_cfg = get_dino_model(dino_mode, device)
    dst = osp.join(src, "dino.npz")
    src = osp.join(src, "images")
    dino_process_folder(dst, src, dino_model, dino_cfg, device=device, pca_dim=pca_dim)
    del dino_model
    logging.info(f"Complete Dino SuperFeat start ({dino_mode})!")
    return


def compute_dino_featup_superfeat(src, device, dino_mode="giant", pca_dim=64):
    raise RuntimeError("Not use this!")
    logging.info(f"Compute Dino SuperFeat start ({dino_mode})...")
    dino_model, dino_cfg = get_feat_up_model(device)
    dst = osp.join(src, "dino.npz")
    src = osp.join(src, "images")
    dino_featup_process_folder(
        dst, src, dino_model, dino_cfg, device=device, pca_dim=pca_dim
    )
    del dino_model
    logging.info(f"Complete Dino SuperFeat start ({dino_mode})!")
    return


def main(
    src,
    device,
    intervals=[1],
    depth_mode="zoe",
    zoe_mode="N",
    num_long_term_d=10000,
    num_long_term_s=10000,
    chunk_long_term_d=5000,
    chunk_long_term_s=10000,
    epi_th_long_term=400.0,
    dummy_sky_seg=False,
    dense_interval=False,
    long_track_keyframe_interval=8,
    long_track_dyn_subsample=1,
    #
    nerfies_flag=False,
):
    configure_logging(osp.join(src, f"prepare.log"), debug=False)

    intervals.sort()

    image_fns, view_pairs = preprocess_views(
        src, intervals=intervals, dense_flag=dense_interval, nerfies_flag=nerfies_flag
    )

    compute_raft(src, view_pairs, device)
    compute_epi_error(src)

    get_long_tracking(
        src,
        device,
        d_num_pts=num_long_term_d,
        s_num_pts=num_long_term_s,
        d_num_chunk=chunk_long_term_d,
        s_num_chunk=chunk_long_term_s,
        epi_th=epi_th_long_term,
        keyframe_interval=long_track_keyframe_interval,
        dyn_subsample=long_track_dyn_subsample,
    )

    # ! always compute a zoe
    # compute_zoe_depths(src, device, mode=zoe_mode)
    if depth_mode == "zoe":
        compute_zoe_depths(src, device, mode=zoe_mode)
    elif depth_mode == "uni":
        compute_uni_depths(src, device)
    elif depth_mode == "marigold":
        compute_mar_depths(src, device)
    elif depth_mode == "anyzoe":
        compute_any_depths(src, device)
    elif depth_mode == "gt":
        assert os.path.exists(osp.join(src, "gt_depth"))
    else:
        raise ValueError(f"Unknown depth mode: {depth_mode}")

    compute_dino_superfeat(src, device, dino_mode="base", pca_dim=64)
    # compute_dino_featup_superfeat(src, device, dino_mode="base", pca_dim=64)
    # todo: check DINO-Tracker delta dino feature

    maskout_sky(src, device, dummy_sky_seg)

    return

def process_images(src, reverse):
    image_src = osp.join(src, "images")
    if reverse is False:
        output_src = osp.join(src,"preprocess")
    else:
        output_src = osp.join(src,"preprocess_reversed")
    os.makedirs(output_src, exist_ok = True)
    output_image_src = osp.join(output_src,"images")
    os.makedirs(output_image_src, exist_ok = True)

    # add functionality: if reverse = True, store all images in reverse order by renaming them (example name 00000.jpg)
    filenames = sorted([f for f in os.listdir(image_src) if f.endswith(".jpg") or f.endswith(".png")])
    if reverse:
        filenames = filenames[::-1]

    for idx, filename in enumerate(filenames):
        image_path = os.path.join(image_src, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            square = min(width, height)
            # img = np.array(img)
            # resolution = 480
            
            # transform = transforms.Compose(
            #     [
            #         transforms.ToPILImage(),
            #         transforms.CenterCrop(square),
            #         # transforms.Lambda(lambda img: img.crop((
            #         #     (img.width // 2) - square // 2 - 350,  # Shift 350 pixels left from center
            #         #     (img.height - square) // 2,            # Vertically center the crop
            #         #     (img.width // 2) + square // 2 - 350,  # Shift 350 pixels left from center
            #         #     (img.height + square) // 2             # Vertically center the crop
            #         # ))),
            #         transforms.Resize(resolution),
            #         transforms.ToTensor(),
            #     ]
            # )
            # img = transform(img)
            # img = img.numpy().transpose(1, 2, 0)
            # img = Image.fromarray((img * 255).astype(np.uint8))

            # Create a new filename with zero-padded index (e.g., 00000.jpg, 00001.jpg, etc.)
            new_filename = f"{str(idx).zfill(5)}.jpg"
            output_path = os.path.join(output_image_src, new_filename)
            img.save(output_path)

    return output_src

def process_video(src):
    # Check if there is a .mp4 file in src
    video_files = [f for f in os.listdir(src) if f.endswith('.mp4')]
    if not video_files:
        print("No .mp4 file found in the directory.")
        return src

    video_path = os.path.join(src, video_files[0])
    images_folder = os.path.join(src, 'images')
    os.makedirs(images_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {video_path}")
        return src
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        filename = f"{frame_index:05d}.jpg"
        frame_path = os.path.join(images_folder, filename)

        # Save the current frame as an image
        cv2.imwrite(frame_path, frame)
        frame_index += 1
    cap.release()

    print(f"Frames saved in {images_folder}")
    return src


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="./data/DAVIS/soapbox")
    parser.add_argument("--config", "-c", type=str, default="./configs/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reverse", type=bool, default=False)
    parser.add_argument("--video", action='store_true', help="Process video file by extracting frames")
    args = parser.parse_args()

    src = args.src
    if args.video is True:
        src = process_video(src)
    src = process_images(src, args.reverse)

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config).prepare
    # OmegaConf.set_readonly(cfg, True)
    with open(osp.join(src, "config_backup.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    main(
        src=src,
        device=device,
        intervals=getattr(cfg, "intervals", [1, 4]),
        depth_mode=getattr(cfg, "depth_mode", "uni"),
        zoe_mode=getattr(cfg, "zoe_mode", "N"),
        num_long_term_d=getattr(cfg, "num_long_term_d", 10000),
        num_long_term_s=getattr(cfg, "num_long_term_s", 10000),
        chunk_long_term_d=getattr(cfg, "chunk_long_term_d", 5000),
        chunk_long_term_s=getattr(cfg, "chunk_long_term_s", 10000),
        epi_th_long_term=getattr(cfg, "epi_th_long_term", 400.0),  # old 200.0
        dummy_sky_seg=getattr(cfg, "dummy_sky_seg", True),
        dense_interval=getattr(cfg, "dense_interval", False),
        long_track_keyframe_interval=getattr(cfg, "long_track_keyframe_interval", 8),
        long_track_dyn_subsample=getattr(cfg, "long_track_dyn_subsample", 1),
        nerfies_flag=getattr(cfg, "nerfies_flag", False),
    )
