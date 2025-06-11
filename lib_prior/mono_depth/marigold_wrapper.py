# Script for inference on (in-the-wild) images

# Author: Bingxin Ke
# Last modified: 2023-12-15
# Modified by Jiahui Lei 2023-12-28

import argparse
import os
from glob import glob
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import imageio
import os.path as osp

from .marigold import MarigoldPipeline
from .marigold.util.seed_all import seed_all


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


def get_marigold_model(device, dtype=torch.float16):
    checkpoint_path = "Bingxin/Marigold"
    pipe = MarigoldPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
    try:
        import xformers

        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    pipe = pipe.to(device)
    return pipe


def marigold_process_folder(
    model,
    src,
    dst,
    denoise_steps=10,
    ensemble_size=10,
    processing_res=768,
    match_input_res=True,
    batch_size=0,
    color_map="Spectral",
):
    # batch_size = 0 means auto, but apple silicon needs to set batch_size=1
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        batch_size = 1

    os.makedirs(dst, exist_ok=True)
    viz_dir = dst + "_viz"
    os.makedirs(viz_dir, exist_ok=True)
    fns = os.listdir(src)
    fns.sort()

    for fn in tqdm(fns):
        in_fn = os.path.join(src, fn)
        out_fn = os.path.join(dst, fn.replace(".jpg", ".npz"))
        out_fn = out_fn.replace(".png", ".npz")
        viz_fn = os.path.join(viz_dir, fn)
        input_image = Image.open(in_fn)
        pipe_out = model(
            input_image,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=match_input_res,
            batch_size=batch_size,
            color_map=color_map,
            show_progress_bar=False,  # True,
        )
        depth_pred: np.ndarray = pipe_out.depth_np
        np.savez_compressed(out_fn, dep=depth_pred.astype(np.float32))
        depth_colored: Image.Image = pipe_out.depth_colored
        depth_colored.save(viz_fn)
    make_video(viz_dir, dst + ".mp4")
