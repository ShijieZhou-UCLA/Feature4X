# the leaf are supported by multi view images
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix


from prior2d import Prior2D
from camera import SimpleFovCamerasIndependent
from scf4d_model import Scaffold4D


class DynMVGaussian(nn.Module):
    ######################################################################
    # * only collection of multi view gaussians
    ######################################################################
    def __init__(
        self,
        prior2d: Prior2D,
        cams: SimpleFovCamerasIndependent,
        max_scale=0.03,
        N=None,
    ) -> None:
        super().__init__()
        # store the parameters in a flatten manner
        (
            time_index,
            ray_o_buffer,
            ray_d_buffer,
            dep_buffer,
            radius,
            param_rgb,
            int_uv,
        ) = collect_perframe_fg_leaves(prior2d, cams)

        if N is not None:
            assert N == len(dep_buffer)
        else:
            N = len(dep_buffer)

        self.register_buffer("time_index", time_index)
        self.register_buffer("ray_o", ray_o_buffer)
        self.register_buffer("ray_d", ray_d_buffer)
        self.register_buffer("dep_base", dep_buffer)
        self.register_buffer("max_scale", torch.tensor(max_scale))
        self.register_buffer("int_uv", int_uv)

        self.param_dep_c = nn.Parameter(torch.zeros(N).to(dep_buffer))
        param_quat_w = torch.zeros(N, 4).to(dep_buffer)
        param_quat_w[..., 0] = 1.0
        self.param_quat_w = nn.Parameter(param_quat_w)
        param_radius_logit = (
            torch.logit(
                torch.clamp(radius[:, None], 0.0, self.max_scale) / self.max_scale
            )
            .expand(-1, 3)
            .to(dep_buffer)
            .clone()
        )
        self.param_s_logit = nn.Parameter(param_radius_logit)
        self.param_o_logit = nn.Parameter(
            torch.logit(torch.ones_like(dep_buffer[:, None]) * 0.99)
        )
        self.param_sph = nn.Parameter(RGB2SH(param_rgb))  # ! zero order

        return

    def get_optimizable_list(
        self, lr_dep=0.00016, lr_q=0.001, lr_s=0.005, lr_o=0.01, lr_sph=0.0025
    ):
        ret = []
        ret.append({"params": self.param_dep_c, "lr": lr_dep, "name": "dep"})
        ret.append({"params": self.param_quat_w, "lr": lr_q, "name": "quat_w"})
        ret.append({"params": self.param_s_logit, "lr": lr_s, "name": "scaling"})
        ret.append({"params": self.param_o_logit, "lr": lr_o, "name": "opacity"})
        ret.append({"params": self.param_sph, "lr": lr_sph, "name": "f_dc"})
        return ret
    
    def get_support_pts(self, support_mask):
        support_t = self.time_index[support_mask]
        support_ray_o = self.ray_o[support_mask]
        support_ray_d = self.ray_d[support_mask]
        support_dep_base = self.dep_base[support_mask]
        support_dep = support_dep_base + self.param_dep_c[support_mask]
        support_xyz = support_ray_o + support_ray_d * support_dep[:, None]
        support_quat = self.param_quat_w[support_mask]
        return support_t, support_xyz, support_quat

    def forward(
        self,
        scf: Scaffold4D,
        t: int,
        support_ts: torch.Tensor = None,
        support_mask: torch.Tensor = None,
        pad_sph_order=None,
    ):
        if support_mask is None:
            with torch.no_grad():
                support_mask = (
                    self.time_index[:, None] == support_ts[None].to(self.time_index)
                ).any(-1)
        else:
            assert len(support_mask) == len(self.time_index)
        
        support_t, support_xyz, support_quat = self.get_support_pts(support_mask)
        with torch.no_grad():
            attach_node_ind = scf.identify_nearest_node_id(support_xyz, support_t)
        mu, fr = scf.warp(
            attach_node_ind=attach_node_ind,
            query_xyz=support_xyz,
            query_dir=q2R(support_quat),
            query_tid=self.time_index[support_mask],
            target_tid=t,
        )
        s = torch.sigmoid(self.param_s_logit[support_mask]) * self.max_scale
        o = torch.sigmoid(self.param_o_logit[support_mask])
        sph = self.param_sph[support_mask]
        if pad_sph_order is not None:
            target_C = 3 * sph_order2nfeat(pad_sph_order)
            if target_C > sph.shape[-1]:
                sph = torch.cat(
                    [sph, torch.zeros(len(sph), target_C - sph.shape[-1]).to(sph)], -1
                )
            elif target_C < sph.shape[-1]:
                sph = sph[:, :target_C]
        return (mu, fr, s, o, sph), support_mask


@torch.no_grad()
def collect_perframe_fg_leaves(prior2d: Prior2D, cams: SimpleFovCamerasIndependent):
    time_buffer, dep_base_buffer = [], []
    ray_o_buffer, ray_d_buffer = [], []
    radius_buffer, rgb_buffer = [], []
    uv_int_buffer = []
    for t in tqdm(range(prior2d.T)):
        mask = prior2d.get_mask_by_key("dyn_dep", t)
        uv_int_buffer.append(prior2d.pixel_int_map[mask])
        homo = prior2d.homo_map[mask]
        homo = torch.cat([homo / cams.rel_focal, torch.ones_like(homo[:, :1])], -1)
        R_wc, t_wc = cams.Rt_wc(t)
        ray_d = torch.einsum("ij,nj->ni", R_wc, homo)
        ray_o = t_wc[None].expand(len(homo), -1)
        dep_base = prior2d.depths[t][mask]
        rgb = prior2d.rgbs[t][mask]
        radius = dep_base / cams.rel_focal * prior2d.pixel_size
        time = torch.ones_like(radius).int() * t

        time_buffer.append(time)
        ray_o_buffer.append(ray_o)
        ray_d_buffer.append(ray_d)
        dep_base_buffer.append(dep_base)
        radius_buffer.append(radius)
        rgb_buffer.append(rgb)

    time_buffer = torch.cat(time_buffer, 0)
    ray_o_buffer = torch.cat(ray_o_buffer, 0)
    ray_d_buffer = torch.cat(ray_d_buffer, 0)
    dep_base_buffer = torch.cat(dep_base_buffer, 0)
    radius_buffer = torch.cat(radius_buffer, 0)
    rgb_buffer = torch.cat(rgb_buffer, 0)
    uv_int_buffer = torch.cat(uv_int_buffer, 0)

    return (
        time_buffer,
        ray_o_buffer,
        ray_d_buffer,
        dep_base_buffer,
        radius_buffer,
        rgb_buffer,
        uv_int_buffer,
    )


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def sph_order2nfeat(order):
    return (order + 1) ** 2
