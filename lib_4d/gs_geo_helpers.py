# given gs5_param, do some geometry processing, this stands outside the GS models, intended to be used by different GS models

from matplotlib import pyplot as plt
import torch, numpy as np
import os, sys, os.path as osp
import open3d as o3d

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from graph_utils import *
from solver_utils import *
from lib_4d_misc import *

from gs_viz_helpers import *
from camera import SimpleFovCamerasDelta
from view_sampler import *
import open3d as o3d


@torch.no_grad()
def tsdf_meshing(
    rendered_dict_list: list,
    cams: SimpleFovCamerasDelta,
    invalid_masks: list,
    alpha_cutoff: float = 0.8,
    alpha_eps=1e-6,
    voxcell_size=1.0 / 100.0,
    sdf_trunc=0.1,
):
    # todo: use new o3d pytorch interface dl

    assert len(rendered_dict_list) == len(invalid_masks) == cams.T

    # construct intrinsic K
    H, W = invalid_masks[0].shape
    fpix = float(cams.rel_focal * min(H, W) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H, fx=fpix, fy=fpix, cx=cx, cy=cy
    )

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxcell_size,  # meter / R
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for tid in tqdm(range(cams.T)):
        render_dict = rendered_dict_list[tid]
        rgb = render_dict["rgb"].permute(1, 2, 0).cpu().numpy()
        dep = render_dict["dep"].cpu().numpy()[0]
        alp = render_dict["alpha"].cpu().numpy()[0]

        invalid = invalid_masks[tid].cpu().numpy()
        invalid = np.logical_or(invalid, (alp < alpha_cutoff))

        rgb = (np.clip(rgb, a_min=0, a_max=1) * 255).astype(np.uint8)
        alp = np.clip(alp, a_min=alpha_eps, a_max=10.0)
        # dep = dep / alp
        dep[invalid] = 0.0

        o3d_dep = o3d.geometry.Image(
            o3d.geometry.Image((dep * 1000.0).astype(np.uint16))
        )
        o3d_color = o3d.geometry.Image(np.asarray(rgb, order="C"))
        o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_dep, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        T = cams.T_cw(tid).cpu().numpy()
        tsdf_volume.integrate(o3d_rgbd, intrinsic, T)

    mesh = tsdf_volume.extract_triangle_mesh()
    # o3d.io.write_triangle_mesh(
    #     f"./debug/tsdf_cell={voxcell_size:.4f}_trunc={sdf_trunc:.4f}.obj", mesh
    # )
    # print()
    return mesh
