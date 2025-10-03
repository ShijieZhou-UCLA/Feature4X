# Born out of Issue 36.
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
import os, sys
import imageio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from any_zoedepth.models.builder import build_model
from any_zoedepth.utils.config import get_config
import os, os.path as osp
from matplotlib import cm
import cv2

DATASET = "kitti"


def make_video(src_dir, dst_fn):
    print(f"export video to {dst_fn}...")
    # print(os.listdir(src_dir))
    img_fn = [
        f for f in os.listdir(src_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]
    img_fn.sort()
    frames = []
    for fn in tqdm(img_fn):
        frames.append(imageio.imread(osp.join(src_dir, fn)))
    imageio.mimsave(dst_fn, frames)
    return


def load_image(fn):
    color_image = Image.open(fn).convert("RGB")
    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0)
    return image_tensor


def save_viz(viz_fn, prediction):
    depth_min = prediction.min()
    depth_max = prediction.max()
    viz = cm.viridis((prediction - depth_min) / (depth_max - depth_min))[:, :, :3]
    cv2.imwrite(viz_fn, viz[..., ::-1] * 255)
    return


@torch.no_grad()
def process(in_fn, out_fn, viz_fn, model, device):
    image_tensor = load_image(in_fn).to(device)
    H, W = image_tensor.shape[-2:]
    _pred = model(image_tensor, dataset=DATASET)
    if isinstance(_pred, dict):
        _pred = _pred.get("metric_depth", _pred.get("out"))
    elif isinstance(_pred, (list, tuple)):
        _pred = _pred[-1]
    _pred = _pred.squeeze().detach().cpu().numpy()
    dep = cv2.resize(_pred, (W, H), interpolation=cv2.INTER_NEAREST)
    # resized_pred = Image.fromarray(pred).resize((W, H), Image.NEAREST)
    np.savez_compressed(out_fn, dep=dep.astype(np.float32))
    save_viz(viz_fn, dep)
    return


@torch.no_grad()
def get_zoedepth_anything_model(device, model_type="outdoor"):
    assert (
        model_type == "outdoor"
    ), "Only outdoor model is supported because the global DATASET here set"
    config = get_config("zoedepth", "eval", DATASET)
    config.pretrained_resource = "local::" + osp.abspath(
        osp.join(
            osp.dirname(osp.abspath(__file__)),
            f"checkpoints/depth_anything_metric_depth_{model_type}.pt",
        )
    )
    model = build_model(config).to(device)
    model.eval()
    return model


def zoedepth_anything_process_folder(model, src, dst):
    print("Generating Depth Anything ZoeFineTuned...")
    os.makedirs(dst, exist_ok=True)
    viz_dir = dst + "_viz"
    os.makedirs(viz_dir, exist_ok=True)
    fns = os.listdir(src)
    fns.sort()
    device = next(model.parameters()).device
    for fn in tqdm(fns):
        in_fn = os.path.join(src, fn)
        save_fn = fn.replace(".jpg", ".npz")
        save_fn = save_fn.replace(".png", ".npz")
        out_fn = os.path.join(dst, save_fn)
        viz_fn = os.path.join(viz_dir, fn)
        process(in_fn, out_fn, viz_fn, model, device)
    make_video(viz_dir, dst + ".mp4")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    # parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")

    # args = parser.parse_args()
    # main(args.model, args.pretrained_resource)

    device = "cuda"
    src = "../../data/DAVIS/horsejump-high"
    anydepth_model = get_zoedepth_anything_model(device=device)
    zoedepth_anything_process_folder(
        anydepth_model,
        src=osp.join(src, "images"),
        dst=osp.join(src, "anyzoe_depth"),
    )


    model = get_zoedepth_anything_model("cuda:0", "outdoor")
    in_fn = "/home/ray/projects/vid24d/data/DAVIS/train/images/00000.jpg"
    out_fn = "./dbg."

    # debug
    image_path = "/home/ray/projects/vid24d/data/DAVIS/train/images/00000.jpg"

    process(in_fn, "./debug.npz", "./debug_viz.jpg", model, "cuda:0")
