#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf

# from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
import cv2 as cv
import numpy as np
import os, os.path as osp
import imageio
import logging
import pandas as pd


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(3, 1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]

        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(), psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]

    return psnr


def readImages(renders_dir, gt_dir, testing_fns=None):
    renders = []
    gts = []
    image_names = []
    # assert len(os.listdir(renders_dir)) == len(
    #     os.listdir(gt_dir)
    # ), "Number of images in renders and gt directories do not match."
    for fname in os.listdir(renders_dir):
        if fname.split(".")[0] not in testing_fns:
            continue
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    return renders, gts, image_names


def evaluate(save_dir, gt_dir, pred_dir, save_prefix="",gt_testing_fns=None):

    if pred_dir.endswith("/"):
        pred_dir = pred_dir[:-1]
    eval_name = osp.basename(pred_dir)
    save_viz_dir = osp.join(save_dir, f"{eval_name}_viz")
    os.makedirs(save_viz_dir, exist_ok=True)

    renders, gts, image_names = readImages(pred_dir, gt_dir, gt_testing_fns)

    ssims = []
    psnrs = []
    # lpipss = []
    # lpipsa = []
    ms_ssims = []
    Dssims = []
    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
        ms_ssims.append(
            ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True)
        )
        # lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
        Dssims.append((1 - ms_ssims[-1]) / 2)

        # record
        error = abs(renders[idx] - gts[idx]).max(dim=1).values.cpu().squeeze(0).numpy()
        error = cv.applyColorMap((error * 255).astype(np.uint8), cv.COLORMAP_JET)[
            ..., [2, 1, 0]
        ]
        viz_img = np.concatenate(
            [
                gts[idx].cpu().squeeze(0).permute(1, 2, 0).numpy() * 255,
                renders[idx].cpu().squeeze(0).permute(1, 2, 0).numpy() * 255,
                error,
            ],
            axis=1,
        ).astype(np.uint8)
        new_fn = image_names[idx].replace("png", "jpg")
        imageio.imwrite(osp.join(save_viz_dir, f"{new_fn}"), viz_img)

    print("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    # print("LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    # print("LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
    print("MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
    print("D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

    logging.info(
        f"PSNR: {torch.tensor(psnrs).mean().item():.6f}, MS-SSIM: {torch.tensor(ms_ssims).mean().item():.6f} save to {save_dir}"
    )

    df = pd.DataFrame(
        {
            "fn": ["AVE"],
            "psnr": [torch.tensor(psnrs).mean().item()],
            "ms-ssim": [torch.tensor(ms_ssims).mean().item()],
        }
    )
    for i in range(len(image_names)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fn": [image_names[i]],
                        "psnr": [psnrs[i].item()],
                        "ms-ssim": [ms_ssims[i].item()],
                    }
                ),
            ],
            ignore_index=True,
        )
    df.to_excel(
        osp.join(save_dir, f"{save_prefix}nerfies_metrics_from_4dgs.xlsx"), index=False
    )

    viz_fns = sorted(
        [
            f
            for f in os.listdir(save_viz_dir)
            if "tto" not in f and f.endswith("jpg")
        ]
    )
    frames = [imageio.imread(osp.join(save_viz_dir, f)) for f in viz_fns]
    imageio.mimsave(save_viz_dir + ".mp4", frames)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--gt_path",
        type=str,
        default="../../data/hypernerf_dev_2x/chicken/test_images",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="../../data/hypernerf_dev_2x/chicken/log/native_nerfies.2.yaml_dep=zoe_gt_cam=True_20240515_154720/tto_test",
    )
    args = parser.parse_args()
    evaluate(
        save_dir=osp.dirname(args.pred_path),
        gt_dir=args.gt_path,
        pred_dir=args.pred_path,
    )
