# Authors: Shuwang Zhang
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import argparse

############################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from video')
    parser.add_argument('--video_path', type=str, default="/home/shijie/Desktop/work/feature-4dgs/data/davis_dev/train/code_output", help='path to the video')
    return parser.parse_args()

if __name__ == "__main__":

    # select the device for computation
    if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    args = parse_args()
    data_dir = args.video_path # path to .../<dataset_name>/code_output
    video_dir = data_dir + "/images" # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`

    sam2_feats = predictor.extract_features(video_path=video_dir)
    # print(len(sam2_feats)) # number of frames
    # print(sam2_feats[0].keys())
    # print(sam2_feats[0]['vision_pos_enc'][0].shape) # [1, 256, 64, 64]
    # print(sam2_feats[0]['backbone_fpn'][0].shape) # [1, 256, 64, 64]

    save_dir = osp.join(data_dir, "semantic_features")
    os.makedirs(save_dir, exist_ok = True)
    save_path = osp.join(save_dir, "sam2_feats.pth")
    torch.save(sam2_feats, save_path)

    

    