import torch
import numpy as np
from pytorch3d.ops import knn_points
import logging

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


# ! warning, all proj and back proj must use this functions, because it handle the cxcy correctly!!


def backproject(uv, d, cams):
    # uv: always be [-1,+1] on the short side
    assert uv.ndim == d.ndim + 1
    assert uv.shape[-1] == 2
    dep = d[..., None]
    rel_f = torch.as_tensor(cams.rel_focal).to(uv)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(uv) * 2.0 - 1.0
    xy = (uv - cxcy[None, :]) * dep / rel_f
    z = dep
    xyz = torch.cat([xy, z], dim=-1)
    return xyz


def project(xyz, cams, th=1e-5):
    assert xyz.shape[-1] == 3
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    z_close_mask = abs(z) < th
    if z_close_mask.any():
        logging.warning(
            f"Projection may create singularity with a point too close to the camera, detected [{z_close_mask.sum()}] points, clamp it"
        )
        z_close_mask = z_close_mask.float()
        z = (
            z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        )  # ! always clamp to positive
        assert not (abs(z) < th).any()
    rel_f = torch.as_tensor(cams.rel_focal).to(xyz)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(xyz) * 2.0 - 1.0
    uv = (xy * rel_f / z) + cxcy[None, :]
    return uv  # [-1,1]


def fovdeg2focal(fov_deg):
    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return focal
