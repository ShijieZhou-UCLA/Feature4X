# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import renderutils as ru
from .render_misc import *

# ==============================================================================================
#  Helper functions
# ==============================================================================================


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(),
        rast,
        attr_idx,
        rast_db=rast_db,
        diff_attrs=None if rast_db is None else "all",
    )


# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
    shader,
    shade_additional_arg_dict,
    rast,
    rast_deriv,
    mesh,
    view_pos,
    resolution,
    spp,
    msaa,
):
    full_res = [resolution[0] * spp, resolution[1] * spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = scale_img_nhwc(rast, resolution, mag="nearest", min="nearest")
        rast_out_deriv_s = (
            scale_img_nhwc(rast_deriv, resolution, mag="nearest", min="nearest") * spp
        )
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (
        torch.arange(0, face_normals.shape[0], dtype=torch.int64, device="cuda")[
            :, None
        ]
    ).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(
        face_normals[None, ...], rast_out_s, face_normal_indices.int()
    )

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(
        mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()
    )  # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(
        mesh.v_tex[None, ...],
        rast_out_s,
        mesh.t_tex_idx.int(),
        rast_db=rast_out_deriv_s,
    )

    ################################################################################
    # Shade
    ################################################################################

    buffers = shader(
        # all gb are in world coordinate frame
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        **shade_additional_arg_dict
    )
    # buffers = shade(
    #     gb_pos,
    #     gb_geometric_normal,
    #     gb_normal,
    #     gb_tangent,
    #     gb_texc,
    #     gb_texc_deriv,
    #     view_pos,
    #     lgt,
    #     mesh.material,
    #     bsdf,
    # )

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = scale_img_nhwc(
                buffers[key], full_res, mag="nearest", min="nearest"
            )

    # Return buffers
    return buffers


# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
    ctx,
    mesh,
    mtx_in,
    view_pos,
    resolution,
    shader,
    shade_additional_arg_dict,
    spp=1,
    num_layers=1,
    msaa=True,
    background=None,
):
    def prepare_input_vector(x):
        x = (
            torch.tensor(x, dtype=torch.float32, device="cuda")
            if not torch.is_tensor(x)
            else x
        )
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(
                accum,
                torch.cat(
                    (buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])),
                    dim=-1,
                ),
                alpha,
            )
            if antialias:
                accum = dr.antialias(
                    accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int()
                )
        return accum

    assert (
        mesh.t_pos_idx.shape[0] > 0
    ), "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (
        background.shape[1] == resolution[0] and background.shape[2] == resolution[1]
    )

    full_res = [resolution[0] * spp, resolution[1] * spp]

    # Convert numpy arrays to torch tensors
    mtx_in = (
        torch.tensor(mtx_in, dtype=torch.float32, device="cuda")
        if not torch.is_tensor(mtx_in)
        else mtx_in
    )
    view_pos = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [
                (
                    render_layer(
                        shader,
                        shade_additional_arg_dict,
                        rast,
                        db,
                        mesh,
                        view_pos,
                        resolution,
                        spp,
                        msaa,
                    ),
                    rast,
                )
            ]

    # Setup background
    if background is not None:
        if spp > 1:
            background = scale_img_nhwc(
                background, full_res, mag="nearest", min="nearest"
            )
        background = torch.cat(
            (background, torch.zeros_like(background[..., 0:1])), dim=-1
        )
    else:
        background = torch.zeros(
            1, full_res[0], full_res[1], 4, dtype=torch.float32, device="cuda"
        )

    # Composite layers front-to-back
    out_buffers = {}
    assert "shaded" in layers[0][0].keys(), "Shader must output 'shaded' buffer, to get gradient!"
    for key in layers[0][0].keys():
        if key == "shaded":
            # ! The most important step, NVDR seems highly depend on the antialiasing to get gradient!
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(
                key, layers, torch.zeros_like(layers[0][0][key]), False
            )

        # Downscale to framebuffer resolution. Use avg pooling
        out_buffers[key] = avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers


# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):
    # clip space transform
    uv_clip = mesh.v_tex[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat(
        (
            uv_clip,
            torch.zeros_like(uv_clip[..., 0:1]),
            torch.ones_like(uv_clip[..., 0:1]),
        ),
        dim=-1,
    )

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert (
        all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10
    ), "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (
        (rast[..., -1:] > 0).float(),
        all_tex[..., :-6],
        all_tex[..., -6:-3],
        safe_normalize(perturbed_nrm),
    )
