import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import yaml
import matplotlib.animation as animation
import imageio
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib_4d')))
from autoencoder.model import Feature_heads
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of CLIP-LSeg')
    parser.add_argument('--rendered_results_path', type=str, default='/usr/project/Feature4X/output/bear/final_viz/41_round_moving/rendered_results.pth')
    parser.add_argument("--labels", type=str, default="default", help="segment using specified labels",)
    parser.add_argument("--head_config", type=str, default="../configs/default_config.yaml")
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save outputs. If not provided, defaults to a subdirectory in the rendered results path.')
    parser.add_argument('--save_features', action='store_true', default=False,
                        help='If set, save decoded feature .pt files to saved_features directory. If not set, features are processed in memory and temporary files are created only for segmentation then cleaned up. Default: process in memory without persistent .pt files.')
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    if args.rendered_results_path is not None:
        args.rendered_results_path = os.path.abspath(args.rendered_results_path)
    
    # Fix head_config path to be absolute
    if not os.path.isabs(args.head_config):
        # If it's a relative path, make it relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from lseg_encoder/ to project root
        args.head_config = os.path.join(project_root, "configs/default_config.yaml")
    
    args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args

def run_ffmpeg(input_pattern, output_video):
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', '12',  # Set the desired frame rate
        '-i', input_pattern,  # Input pattern for the images
        '-c:v', 'libx264',  # Specify the video codec
        '-pix_fmt', 'yuv420p',  # Set pixel format
        output_video  # Output video file name with path
    ]
    subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    args = parse_args()
    if args.save_dir is None:
        save_dir = os.path.join(os.path.dirname(args.rendered_results_path),"lseg_results")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    rendered_results = args.rendered_results_path
    render_dicts = torch.load(rendered_results,weights_only=True)
    print('frams:', len(render_dicts)) # frames
    # print(render_dicts[0].keys())
    # print(render_dicts[0]["rgb"].shape) # [3, 480, 480]
    # print(render_dicts[0]["feature_map"].shape) # [channels, 64, 64]

    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
    feature_heads.load_state_dict(state_dict)
    feature_heads.eval()

    img_save_dir = os.path.join(save_dir,"renders")
    os.makedirs(img_save_dir, exist_ok=True)
    # Only create feature directory if saving features
    feat_save_dir = None
    if args.save_features:
        feat_save_dir = os.path.join(save_dir,"saved_features")
        os.makedirs(feat_save_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()
    
    # Initialize temp directory for non-persistent features
    temp_feat_dir = None
    if not args.save_features:
        temp_feat_dir = os.path.join(save_dir, "saved_features")
        os.makedirs(temp_feat_dir, exist_ok=True)

    # Process frames with progress bar
    print(f"Processing {len(render_dicts)} frames and saving features...")
    for frame_idx in tqdm(range(len(render_dicts)), desc="Saving .pt features", unit="frame"):
        rgb_feat = render_dicts[frame_idx]["rgb"]  # [3, 480, 480]
        image = to_pil(rgb_feat)  # Convert tensor to PIL image
        img_save_path = os.path.join(img_save_dir, f"{frame_idx:05d}.png")
        image.save(img_save_path)

        rendered_feat = render_dicts[frame_idx]["feature_map"] # [channels, 64, 64]
        rendered_feat = F.interpolate(rendered_feat[None], size=(480, 480), mode='bilinear', align_corners=False)[0].permute(1,2,0) # [480, 480, channels]
        rendered_feat = feature_heads.decode("langseg" , rendered_feat).permute(2,0,1).cpu() # [512, 480, 480] gt shape
        
        # Always save feature files for segmentation (will clean up later if not keeping)
        if args.save_features:
            # Save to permanent location
            feat_save_path = os.path.join(feat_save_dir, f"{frame_idx:05d}_fmap_CxHxW.pt")  # type: ignore[arg-type]
            torch.save(rendered_feat, feat_save_path)
        else:
            # Save to temporary location for segmentation
            temp_feat_path = os.path.join(temp_feat_dir, f"{frame_idx:05d}_fmap_CxHxW.pt")  # type: ignore[arg-type]
            torch.save(rendered_feat, temp_feat_path)

    # Report where RGB frames are saved
    print(f"[Info] Rendered RGB frames saved to: {img_save_dir}")
    if args.save_features:
        print(f"[Info] Feature tensors saved to: {feat_save_dir}")  # type: ignore[str-bytes-safe]
    else:
        print(f"[Info] Features written to temporary location: {temp_feat_dir}")

    # Always run segmentation now that features exist
    # Change to lseg_encoder directory to run segmentation.py
    current_dir = os.getcwd()
    lseg_encoder_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(lseg_encoder_dir)
    
    command = [
        "python", "-u", "segmentation.py",
        "--data", os.path.dirname(save_dir),  # Use parent directory of save_dir so segmentation finds lseg_results/
        "--label_src", args.labels
    ]
    subprocess.run(command)
    
    # Change back to original directory
    os.chdir(current_dir)

    # Always generate videos after segmentation
    video_save_dir = os.path.join(save_dir,"segmentation")
    input_folder = os.path.join(video_save_dir,"lseg_results")
    output_video1 = os.path.join(video_save_dir, 'lseg.mp4')
    output_video2 = os.path.join(video_save_dir, 'lseg_viz.mp4')
    output_video3 = os.path.join(video_save_dir, 'lseg_legend.mp4')

    if os.path.isdir(input_folder):
        # Change to the input directory
        os.chdir(input_folder)
        run_ffmpeg('%05d_fmap_CxHxW.png', output_video1)
        run_ffmpeg('%05d_fmap_CxHxW.png_vis.png', output_video2)
        run_ffmpeg('%05d_fmap_CxHxW.png_legend.png', output_video3)
        print(f"[Info] Segmentation frames directory: {input_folder}")
        for vid in [output_video1, output_video2, output_video3]:
            if os.path.exists(vid):
                print(f"[Info] Generated video: {vid}")
    else:
        print(f"[Warning] Expected segmentation output folder '{input_folder}' not found; skipping video generation.")

    # Clean up temporary feature files if they were created
    if temp_feat_dir and os.path.exists(temp_feat_dir):
        import shutil
        try:
            shutil.rmtree(temp_feat_dir)
            print(f"[Info] Removed temporary feature directory: {temp_feat_dir}")
        except Exception as e:
            print(f"[Warning] Failed to remove temporary feature directory {temp_feat_dir}: {e}")
    elif args.save_features:
        print(f"[Info] Retained feature tensors in: {feat_save_dir}")  # type: ignore[str-bytes-safe]

    # Final summary
    print(f"[Info] Base output directory: {save_dir}")