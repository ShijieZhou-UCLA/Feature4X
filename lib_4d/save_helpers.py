import numpy as np
from plyfile import PlyData, PlyElement
import torch
from pytorch3d.transforms import matrix_to_quaternion
from copy import deepcopy
import trimesh
from torch.distributions.multivariate_normal import MultivariateNormal


def save_gauspl_ply(path, xyz, frame, scale, opacity, color_feat, semantic_feature):
    # ! store in gaussian splatting activation way: opacity use sigmoid and scale use exp
    xyz = xyz.detach().cpu().numpy().squeeze()
    N = xyz.shape[0]
    normals = np.zeros_like(xyz)
    sph_feat = color_feat.reshape(N, -1, 3)
    f_dc = (
        sph_feat[:, :1]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )  # ! self._features_dc: N,1,3
    f_rest = (
        sph_feat[:, 1:]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )  # ! self._features_rest: N, 15, 3
    opacities = (
        torch.logit(opacity).detach().cpu().numpy()
    )  # ! self._opacity, before comp, N,1
    scale = torch.log(scale).detach().cpu().numpy()  # ! _scaling, N,3, before comp
    # rotation = self._rotation.detach().cpu().numpy() # ! self._rotation, N,4 quat
    rotation = (
        matrix_to_quaternion(frame).detach().cpu().numpy()
    )  # ! self._rotation, N,4 quat

    dtype_full = [
        (attribute, "f4")
        for attribute in construct_list_of_attributes(sph_feat.shape[1] - 1)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
    # save semantic feature
    semantic_feature = semantic_feature.detach().cpu()
    torch.save(semantic_feature, path + ".semantic_feature") # TODO: what to do with saving semantic feature?


def construct_list_of_attributes(_features_rest_D):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(1 * 3):
        l.append("f_dc_{}".format(i))
    for i in range(_features_rest_D * 3):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(3):
        l.append("scale_{}".format(i))
    for i in range(4):
        l.append("rot_{}".format(i))
    return l
