import logging
import cv2 as cv
import kornia
import torch
import numpy as np
from tqdm import tqdm
import open3d.core as o3c
import open3d as o3d
import imageio
import os, sys, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

from projection_helper import backproject


def __dyn_mask_filtering(
    prior2d, view_ind, cams, mask, open_kernel=None, motion_flag=False, o3d_flag=False
):
    # do morphological opening to remove small noise via kornia
    if open_kernel is None:
        mask_open = mask
    else:
        mask_open = kornia.morphology.opening(
            mask[None, None].float(), open_kernel.float()
        )[0, 0]
    # consider the sky and the valid depth
    mask_open = mask_open.bool() * prior2d.get_sky_depth_mask(view_ind)
    if motion_flag:
        # also consider the motion mask
        mask_open = mask_open * ~prior2d.static_masks[view_ind]
    if o3d_flag:
        # * also use o3d stat filter to remove outlier
        fg_pt = backproject(
            prior2d.homo_map[mask_open],
            prior2d.depths[view_ind][mask_open],
            cams,
        )
        # convert to o3d pcd
        o3dpcd = o3d.t.geometry.PointCloud()
        o3dpcd.point.positions = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(fg_pt.detach().cpu())
        )
        _, mask_inlier = o3dpcd.remove_statistical_outliers(
            nb_neighbors=20, std_ratio=2.0
        )
        mask_inlier = np.asarray(mask_inlier) > 0.5
        mask_inlier = torch.from_numpy(mask_inlier).to(mask)
        mask_open[mask_open.clone()] = mask_inlier
    return mask_open
