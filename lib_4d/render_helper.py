import os, sys, os.path as osp
import torch

# sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

# get from env variable
try:
    GS_BACKEND = os.environ["GS_BACKEND"]
except:
    GS_BACKEND = "native_feat"
# GS_BACKEND = "native_feat"
# GS_BACKEND = GS_BACKEND.lower()
print(f"GS_BACKEND: {GS_BACKEND}")

if GS_BACKEND == "native_feat":
    from lib_render.gs3d.gauspl_renderer_native_feature import render_cam_pcl

elif GS_BACKEND == "native":
    from lib_render.gs3d.gauspl_renderer_native import render_cam_pcl
else:
    raise ValueError(f"Unknown GS_BACKEND: {GS_BACKEND}")

from lib_render.gs3d.sh_utils import RGB2SH, SH2RGB


def render(
    gs_param,
    H,
    W,
    rel_focal,
    cxcy_ratio,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
):
    # * Core render interface
    # prepare gs5 param in world system
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph, semantic_feature = gs_param
    else:
        mu, fr, s, o, sph, semantic_feature = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
            semantic_feature = torch.cat([semantic_feature, gs_param[i][5]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

    # render
    pf = rel_focal / 2 * min(H, W)
    render_dict = render_cam_pcl(
        mu_cam,
        fr_cam,
        s,
        o,
        sph,
        semantic_feature=semantic_feature,
        H=H,
        W=W,
        fx=pf,
        fy=pf,
        cx=cxcy_ratio[0] * W,
        cy=cxcy_ratio[1] * H,
        bg_color=bg_color,
    )  # ! assume same fx, fy
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict


def fast_bg_compose_render(bg_cache_dict, render_dict, bg_color=[1.0, 1.0, 1.0]):
    assert GS_BACKEND == "native", "GOF does not support this now"
    
    # manually compose the fg
    # ! warning, be careful when use the visibility masks .etc, watch the len
    fg_rgb, bg_rgb = render_dict["rgb"], bg_cache_dict["rgb"]
    fg_semantic_feature, bg_semantic_feature = (
        render_dict["semantic_feature"],
        bg_cache_dict["semantic_feature"],
    )
    fg_alpha, bg_alpha = render_dict["alpha"], bg_cache_dict["alpha"]
    fg_dep, bg_dep = render_dict["dep"], bg_cache_dict["dep"]
    _fg_alp = torch.clamp(fg_alpha, 1e-8, 1.0)
    _bg_alp = torch.clamp(bg_alpha, 1e-8, 1.0)
    fg_dep_corr = fg_dep / _fg_alp
    bg_dep_corr = bg_dep / _bg_alp
    fg_in_front = (fg_dep_corr < bg_dep_corr).float()
    # compose alpha
    alpha_fg_front_compose = fg_alpha + (1.0 - fg_alpha) * bg_alpha
    alpha_fg_behind_compose = bg_alpha + (1.0 - bg_alpha) * fg_alpha
    alpha_composed = alpha_fg_front_compose * fg_in_front + alpha_fg_behind_compose * (
        1.0 - fg_in_front
    )
    alpha_composed = torch.clamp(alpha_composed, 0.0, 1.0)
    # compose rgb
    bg_color = torch.as_tensor(bg_color, device=fg_rgb.device, dtype=fg_rgb.dtype)
    rgb_fg_front_compose = (
        fg_rgb * fg_alpha
        + bg_rgb * (1.0 - fg_alpha) * bg_alpha
        + (1.0 - fg_alpha) * (1.0 - bg_alpha) * bg_color[:, None, None]
    )
    rgb_fg_behind_compose = (
        bg_rgb * bg_alpha
        + fg_rgb * (1.0 - bg_alpha) * fg_alpha
        + (1.0 - bg_alpha) * (1.0 - fg_alpha) * bg_color[:, None, None]
    )
    rgb_composed = rgb_fg_front_compose * fg_in_front + rgb_fg_behind_compose * (
        1.0 - fg_in_front
    )
    # compose semantic_feat
    semantic_feature_composed = fg_semantic_feature * fg_alpha + bg_semantic_feature * (
        1.0 - fg_alpha
    )
    # compose dep
    dep_fg_front_compose = (
        fg_dep_corr * fg_alpha + bg_dep_corr * (1.0 - fg_alpha) * bg_alpha
    )
    dep_fg_behind_compose = (
        bg_dep_corr * bg_alpha + fg_dep_corr * (1.0 - bg_alpha) * fg_alpha
    )
    dep_composed = dep_fg_front_compose * fg_in_front + dep_fg_behind_compose * (
        1.0 - fg_in_front
    )
    return {
        "rgb": rgb_composed,
        "semantic_feature": semantic_feature_composed,
        "dep": dep_composed,
        "alpha": alpha_composed,
        "visibility_filter": render_dict["visibility_filter"],
        "viewspace_points": render_dict["viewspace_points"],
        "radii": render_dict["radii"],
        "dyn_rgb": fg_rgb,
        "dyn_semantic_feature": fg_semantic_feature,
        "dyn_dep": fg_dep,
        "dyn_alpha": fg_alpha,
    }
