# The dynamic model with Deformation Graph

import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn.pool as pyg_pool

from gs_optim_helpers import *
import logging

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points
from dynwarp_helper import dq_arap_interpolation
import yaml

def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class DenseDynGaussian(nn.Module):
    # * First try the dense version, the ARAP is softly enforced by loss
    # * Can optimize the connectivity as well!
    # * each gaussian is a noodle

    def __init__(
        self,
        init_tid=0,
        init_mean=None,  # 1,N,3
        init_rgb=None,  # N,3
        init_radius=None,  # N
        init_opacity=0.9,
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        iso_scale_flag=False,
        max_sph_order=0,
        device=torch.device("cuda:0"),
        load_fn=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.op_update_exclude = ["cnn_decoder"]

        # self.max_scale = max_scale
        # self.min_scale = min_scale
        self.register_buffer("max_scale", torch.tensor(max_scale).to(self.device))
        self.register_buffer("min_scale", torch.tensor(min_scale).to(self.device))
        self.max_sph_order = max_sph_order
        self._init_act(self.max_scale, self.min_scale)
        self.register_buffer("tid", torch.tensor([init_tid]).long())
        self.register_buffer(
            "iso_scale_flag", torch.tensor([iso_scale_flag]).bool().squeeze()
        )

        self.register_buffer(
            "gs_create_T",
            torch.ones(len(init_radius), dtype=torch.int32, device=self.device),
        )  # ! start from 1

        if load_fn is not None:
            self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
            self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
            self.register_buffer("max_radii2D", torch.zeros(self.N).float())
            # * init the parameters from file
            logging.info(f"Loading dynamic model from {load_fn}")
            self.load(torch.load(load_fn))
            self.summary()
            return
        else:
            assert (
                init_mean is not None
                and init_rgb is not None
                and init_radius is not None
            )

        # * init the parameters
        assert init_mean.ndim == 3  # consider time
        assert init_radius.ndim == 2  # invariant across time
        assert init_mean.shape[0] == self.T
        self._xyz = nn.Parameter(torch.as_tensor(init_mean).float())  # T,N,3
        init_q = torch.Tensor([1, 0, 0, 0]).float()[None].expand(self.N, -1)[None]
        self._rotation = nn.Parameter(init_q)  # T,N,3
        if not isinstance(init_radius, torch.Tensor):
            self._scaling = nn.Parameter(
                torch.ones_like(init_mean)
                * self.s_inv_act(init_radius).squeeze(-1)[:, None]
            )
        else:
            assert len(init_radius) == self.N
            assert init_radius.ndim == 2
            assert init_radius.shape[1] == 3
            self._scaling = nn.Parameter(self.s_inv_act(init_radius))
        o = self.o_inv_act(init_opacity) * torch.ones_like(init_mean[0, :, :1])
        self._opacity = nn.Parameter(o)
        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(RGB2SH(init_rgb))
        self._features_rest = nn.Parameter(torch.zeros(self.N, sph_rest_dim))

        # * init states
        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        self.summary()
        return

    def summary(self):
        logging.info(
            f"DenseDynGaussian: {self.N/1000.0:.1f}K points and {self.T} time step"
        )
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info("-" * 30)
        return

    def _init_act(self, max_s_value, min_s_value):
        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            x = torch.clamp(
                x, min=min_s_value + 1e-6, max=max_s_value - 1e-6
            )  # ! clamp
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.logit(y)
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y

        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act

        return

    @property
    def N(self):
        try:  # for loading from file dummy init
            return self._xyz.shape[1]
        except:
            return 0

    @property
    def T(self):
        if hasattr(self, "_xyz"):
            assert len(self._xyz) == len(self.tid)
        return len(self.tid)

    @property
    def get_t_range(self):
        r = torch.max(self.tid)
        l = torch.min(self.tid)
        assert (
            torch.sort(self.tid).values == torch.arange(l, r + 1).to(self.tid)
        ).all()
        return int(l.cpu()), int(r.cpu())

    def get_tlist_ind(self, t):
        return self.tid.tolist().index(t)

    def get_x(self, t: int):
        return self._xyz[self.get_tlist_ind(t)]

    def get_R(self, t: int):
        return quaternion_to_matrix(self._rotation[self.get_tlist_ind(t)])

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_s(self):
        s = self.s_act(self._scaling)
        if self.iso_scale_flag:
            s = s.mean(dim=1, keepdim=True).expand(-1, 3)
        return s

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    @property
    def get_semantic_feature(self):
        return self._semantic_feature


    def forward(self, t: int, active_sph_order=0):
        xyz = self.get_x(t)
        frame = self.get_R(t)
        s = self.get_s
        o = self.get_o
        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c[:, :sph_dim]
        semantic_feature = self.get_semantic_feature
        return xyz, frame, s, o, sph, semantic_feature

    def get_optimizable_list(
        self,
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
        lr_semantic_feature=0.001,
    ):
        print(f"lr_semantic_feature: {lr_semantic_feature}")
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._rotation], "lr": lr_q, "name": "rotation"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"},
            {"params": [self._semantic_feature], "lr": lr_semantic_feature, "name": "semantic_feature"},
            # {"params": self.cnn_decoder.parameters(), "lr": 0.0001 ,"name": "cnn_decoder"}
        ]
        return l

    @torch.no_grad()
    def append_one_timestep(
        self, optimizer, new_tid, pre_tid=None, new_xyz=None, new_r=None
    ):
        # copy the frame xyz and rotation, and do so by replace the optimizer tensor
        if pre_tid is None:
            assert new_xyz is not None and new_r is not None
        else:
            pre_ind = self.get_tlist_ind(pre_tid)
            new_xyz = self._xyz[pre_ind : pre_ind + 1]
            new_r = self._rotation[pre_ind : pre_ind + 1]

        self.tid = torch.cat((self.tid, torch.tensor([new_tid]).long().to(self.tid)))
        assert new_xyz.shape[0] == 1 and new_xyz.ndim == 3
        assert new_r.shape[0] == 1 and new_r.ndim == 3
        d = {"xyz": new_xyz, "rotation": new_r}
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)
        self._xyz = optimizable_tensors["xyz"]
        self._rotation = optimizable_tensors["rotation"]
        logging.info(f"DenseDynGaussian: Increase time step to {self.T}")
        return

    @torch.no_grad()
    def append_points(
        self,
        new_tid,
        optimizer,
        new_mu,
        new_quat,
        new_scaling_logit,
        new_opacities_logit,
        new_features_dc,
        new_features_rest,
        new_semantic_feature,
    ):
        # ! warning, this func assume the new_tid is already warping optimized, the append is incremental for occlusion and error or density append
        # Input single frame x,fr, propagate to all other frames
        # use ARAP interpolation to propagate the new points
        # compute the mu, fr with dual quaternion interpolation to the existing noodles
        new_ind = self.get_tlist_ind(new_tid)
        assert new_mu.ndim == 2 and new_quat.ndim == 2
        assert new_mu.shape[1] == 3 and new_quat.shape[1] == 4
        if len(new_mu) == 0:
            logging.info(f"Append: 0 new points, skip")
            return
        new_fr = q2R(new_quat)
        curve_x = self._xyz
        curve_fr = q2R(self._rotation)
        new_mu_curve, new_fr_curve = dq_arap_interpolation(
            new_mu, new_fr, curve_x, curve_fr, new_ind
        )
        new_q_curve = matrix_to_quaternion(new_fr_curve)
        logging.info(f"Append: [+]{new_mu_curve.shape[1]}")
        self._densification_postprocess(
            optimizer,
            new_xyz=new_mu_curve,
            new_r=new_q_curve,
            new_s=new_scaling_logit,
            new_o=new_opacities_logit,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=new_semantic_feature,
        )
        create_at = self.T
        self.gs_create_T = torch.cat(
            (self.gs_create_T, torch.ones_like(new_mu[:, 0]).int() * create_at)
        )
        return

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def record_xyz_grad_radii(self, viewspace_point_tensor_grad, radii, update_filter):
        # ! not, here the parameter is different!!
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor_grad) == self.N
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.xyz_gradient_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s,
        new_o,
        new_sph_dc,
        new_sph_rest,
        new_semantic_feature,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            "semantic_feature" : new_semantic_feature,
            "opacity": new_o,
            "scaling": new_s,
            "rotation": new_r,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # First cat to optimizer and then return to self
        optimizable_tensors = cat_tensors_to_optimizer(
            optimizer, d, {"xyz": 1, "rotation": 1}
        )
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = torch.zeros(self.N, device=self.device)
        self.xyz_gradient_denom = torch.zeros(self.N, device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[0, :, 0])], dim=0
        )
        return

    def clean_gs_control_record(self):
        self.xyz_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)
        self.xyz_gradient_denom = torch.zeros_like(self.xyz_gradient_denom)
        self.max_radii2D = torch.zeros_like(self.max_radii2D)

    def _densify_and_clone(self, optimizer, grad_norm, grad_threshold, scale_th):
        # Extract points that satisfy the gradient condition
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_s, dim=1).values <= scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        new_xyz = self._xyz[:, selected_pts_mask]
        new_rotation = self._rotation[:, selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        semantic_feature = self._semantic_feature[selected_pts_mask]
        new_gs_create_T = self.gs_create_T[selected_pts_mask]
        self.gs_create_T = torch.cat((self.gs_create_T, new_gs_create_T))

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=semantic_feature,
        )

        return new_xyz.shape[1]

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        # TODO: check whether this has a bug, seems very noisy
        # Extract points that satisfy the gradient condition
        _scaling = self.get_s
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(_scaling, dim=1).values > scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        stds = _scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((self.T, stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[:, selected_pts_mask]).repeat(
            1, N, 1, 1
        )
        new_xyz = torch.einsum("tnij,tnj->tni", rots, samples) + self._xyz[
            :, selected_pts_mask
        ].repeat(1, N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[:, selected_pts_mask].repeat(1, N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)

        new_gs_create_T = self.gs_create_T[selected_pts_mask].repeat(N)
        self.gs_create_T = torch.cat((self.gs_create_T, new_gs_create_T))

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=semantic_feature,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return new_xyz.shape[1]

    def densify(self, optimizer, max_grad, percent_dense, extent, verbose=True):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0

        # n_clone = self._densify_and_clone(optimizer, grads, max_grad)
        n_clone = self._densify_and_clone(
            optimizer, grads, max_grad, percent_dense * extent
        )
        n_split = self._densify_and_split(
            optimizer, grads, max_grad, percent_dense * extent, N=2
        )

        if verbose:
            logging.info(f"Densify: Clone[+] {n_clone}, Split[+] {n_split}")
            # logging.info(f"Densify: Clone[+] {n_clone}")
        # torch.cuda.empty_cache()
        return

    def prune_points(
        self,
        optimizer,
        min_opacity,
        max_screen_size,
        max_3d_radius=1000,
        min_scale=0.0,
        verbose=True,
        additional_remove_mask=None,  # mark removal as True
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        if verbose:
            logging.info(f"{prune_mask.sum()} opa small points")

        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            # ! also consider the scale
            max_scale = self.get_s.max(dim=1).values
            big_points_scale = max_scale > max_3d_radius
            big_points_vs = torch.logical_or(big_points_vs, big_points_scale)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        # also check the scale
        too_small_mask = (self.get_s < min_scale).all(dim=-1)
        if verbose:
            logging.info(f"{too_small_mask.sum()} all 3 scale small points")
        prune_mask = torch.logical_or(prune_mask, too_small_mask)
        if additional_remove_mask is not None:
            prune_mask = torch.logical_or(prune_mask, additional_remove_mask)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        # if valid_points_mask.all():
        #     return
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        if getattr(self, "color_memory", None) is None:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]

        self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.gs_create_T = self.gs_create_T[valid_points_mask]
        # torch.cuda.empty_cache()
        return

    def reset_opacity(self, optimizer, value=0.01, verbose=True):
        opacities_new = self.o_inv_act(
            torch.min(self.o_act(self._opacity), torch.ones_like(self._opacity) * value)
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, opacities_new, "opacity"
        )
        if verbose:
            logging.info(f"Reset opacity to {value}")
        self._opacity = optimizable_tensors["opacity"]

    def load(self, ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(torch.as_tensor(ckpt["_xyz"], dtype=torch.float32))

        self._features_dc = nn.Parameter(
            torch.as_tensor(ckpt["_features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(ckpt["_features_rest"], dtype=torch.float32)
        )

        self._semantic_feature = nn.Parameter(
            torch.as_tensor(ckpt["_semantic_feature"], dtype=torch.float32)
        )

        self._opacity = nn.Parameter(
            torch.as_tensor(ckpt["_opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(ckpt["_scaling"], dtype=torch.float32)
        )
        self._rotation = nn.Parameter(
            torch.as_tensor(ckpt["_rotation"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(ckpt["max_radii2D"], dtype=torch.float32)
        self.gs_create_T = torch.as_tensor(ckpt["gs_create_T"], dtype=torch.int32)
        # load others
        self.max_scale = ckpt["max_scale"]
        self.min_scale = ckpt["min_scale"]
        self.tid = ckpt["tid"]
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)

        return

    ######################################################################
    # * Regularization
    ######################################################################

    def multi_level_arap_reg(
        self,
        tid_list,
        w_list=[0.4, 0.3, 0.2, 0.1],
        detach_mask=None,  # if detach, True
    ):
        # ! manually build the topo outside this func
        # * build edge energy in parallel!
        assert self.topo_check_N == self.N, "Topology is not updated"
        # randomly permute the order!
        tid_list = torch.as_tensor(tid_list, device=self.device).long()
        tid_list = tid_list[torch.randperm(len(tid_list), device=self.device)]
        # fetch the src, dst p and fr
        time_xyz = self._xyz[tid_list]  # T,N,3
        time_rotation = self._rotation[tid_list]  # T,N,4
        if detach_mask is not None:  # for long term arap!
            detach_mask = torch.as_tensor(detach_mask, device=self.device).float()
            assert len(detach_mask) == len(tid_list)
            detached_xyz = time_xyz.detach()
            detached_rotation = time_rotation.detach()
            time_xyz = (
                time_xyz * (1.0 - detach_mask)[:, None, None]
                + detached_xyz * detach_mask[:, None, None]
            )
            time_rotation = (
                time_rotation * (1.0 - detach_mask)[:, None, None]
                + detached_rotation * detach_mask[:, None, None]
            )
        time_rotation = q2R(time_rotation)  # T,N,3,3

        self_p = time_xyz[:, self.topo_edge_src_list]  # T,E,3, t_wi
        nn_p = time_xyz[:, self.topo_edge_dst_list]  # T,E,3
        # self_fr = time_rotation[:, self.topo_edge_src_list]  # T,E,3,3, R_wi
        nn_fr = time_rotation[:, self.topo_edge_dst_list]  # T,E,3,3
        # compute edge energy
        # p_world_ti = R_wi @ p_locali + p_wi
        # p_locali = R_wi.T @ (p_world - p_wi)
        # p_world_tj  = R_wj @ R_wi.T @ (p_world - p_wi) + p_wj
        # p_tj = (R_wj @ R_wi.T) @ p_ti + (p_wj - (R_wj @ R_wi.T) @ p_wi)
        # next is j;  pre is i
        nn_R_next_pre = torch.einsum("teij,tekj->teik", nn_fr[1:], nn_fr[:-1])
        nn_t_next_pre = nn_p[1:] - torch.einsum(
            "teij,tej->tei", nn_R_next_pre, nn_p[:-1]
        )

        target = self_p[1:]  # self next; pre is use pre self but nn warping
        pred = torch.einsum("teij,tej->tei", nn_R_next_pre, self_p[:-1]) + nn_t_next_pre
        diff = (target - pred).norm(dim=-1)

        loss = diff.mean(0)  # ave across time

        loss = loss[None] * self.topo_edge_mask_list
        loss = loss.sum(-1) / self.topo_edge_mask_sum

        # re weight multi scale
        weight = torch.tensor(w_list).to(self.device)
        loss = (loss * weight).sum()

        return loss

    @torch.no_grad()
    def build_multilevel_topology(
        self,
        resample_list=[0, 1, 4, 8],
        K_list=[8, 8, 4, 4],
        max_num_pts=8192,
    ):
        # This function will register some buffer to self to store the topology, so it may be called less frequently than the arap loss compute function

        src_list, dst_list = [], []  # all ind is in original node list
        num_edge_list = []

        # randomly select one time frame to do resample
        resample_at = self.tid[torch.randint(0, self.T, (1,))].item()
        resample_base_xyz = self.get_x(resample_at)

        voxel_size = compute_adaptive_voxel_size(
            resample_base_xyz, adapt_voxel_size_knn=16, adapt_voxel_size_factor=4.0
        )

        # build downsampled node knn graph
        for K, resample_factor in zip(K_list, resample_list):
            # also bound the max points to control the speed
            if resample_factor == 0:
                # not resample, do the GS pcl KNN
                if self.N > max_num_pts:
                    src_ind = torch.randperm(self.N, device=self.device)[:max_num_pts]
                else:
                    src_ind = torch.arange(self.N, device=self.device)
            else:
                if self.N > max_num_pts:
                    bound_ind = torch.randperm(self.N, device=self.device)[:max_num_pts]
                    bounded_base_xyz = resample_base_xyz[bound_ind]
                else:
                    bounded_base_xyz = resample_base_xyz
                src_ind = vox_pcl_resample(
                    bounded_base_xyz, voxel_size * resample_factor
                )
            src_list.append(src_ind[:, None].expand(-1, K).reshape(-1))  # N,K
            sampled_traj = self._xyz[:, src_ind, :]
            _M = len(src_ind)
            # build knn on curve
            node_curve = sampled_traj.permute(1, 0, 2).reshape(_M, self.T * 3)  # M,T,3
            _, knn_ind, _ = knn_points(node_curve[None], node_curve[None], K=K + 1)
            knn_ind = knn_ind.squeeze(0)[:, 1:]
            dst_ind = src_ind[knn_ind]  # _M,K
            dst_list.append(dst_ind.reshape(-1))  # M*K
            num_edge_list.append(_M * K)
            # logging.info(f"Resample level={resample_factor} K={K} with {_M} points")

        # save the checksum, to avoid implicit buggy topo when graph is modified
        self.topo_check_N = self.N  # avoid accidental GS control
        self.topo_edge_src_list = torch.cat(src_list, 0)
        self.topo_edge_dst_list = torch.cat(dst_list, 0)
        cur = 0
        self.topo_edge_mask_list = []
        for i in range(len(num_edge_list)):
            mask = torch.zeros_like(self.topo_edge_src_list, dtype=bool)
            mask[cur : cur + num_edge_list[i]] = True
            cur += num_edge_list[i]
            self.topo_edge_mask_list.append(mask)
        self.topo_edge_mask_list = torch.stack(self.topo_edge_mask_list, 0).float()
        self.topo_edge_mask_sum = torch.tensor(num_edge_list).to(self.device)
        assert (self.topo_edge_mask_list.sum(1) == self.topo_edge_mask_sum).all()
        logging.info(
            f"Topology built: |E|={len(self.topo_edge_src_list)/1000.0:.3f}K, with MaxN={max_num_pts} PTS and levels={resample_list}"
        )
        return

    def small_deform_reg(self, i_ind, j_ind, rot_w=0.1):
        # ! this only works for neighbor frames
        assert (
            abs(i_ind - j_ind) == 1
        ), f"{i_ind}, {j_ind}, Small Deform only works for neighbor frames"
        trans = (self.get_x(i_ind) - self.get_x(j_ind)).norm(dim=-1)
        rot = torch.norm(
            matrix_to_axis_angle(self.get_R(i_ind) @ self.get_R(j_ind).transpose(1, 2)),
            dim=-1,
        )
        trans_loss = (trans**2).mean()
        rot_loss = (rot**2).mean()
        return trans_loss + rot_w * rot_loss

    def vel_reg(self, i_ind, j_ind, detach_flag=True):
        # ! note, currently the acc reg does not count the rotation
        # find double side
        before_i_ind = i_ind - (j_ind - i_ind)
        before_j_ind = j_ind - (i_ind - j_ind)
        i_xyz, j_xyz = self.get_x(i_ind), self.get_x(j_ind)
        vel_loss = torch.zeros_like(i_xyz[0, 0])
        if before_i_ind in self.tid:  # i is mid point
            another_xyz = self.get_x(before_i_ind)
            if detach_flag:
                another_xyz = another_xyz.detach()
            vel_loss = (
                vel_loss + ((j_xyz + another_xyz - 2 * i_xyz) ** 2).sum(-1).mean()
            )
        if before_j_ind in self.tid:  # j is mid point
            another_xyz = self.get_x(before_j_ind)
            if detach_flag:
                another_xyz = another_xyz.detach()
            vel_loss = (
                vel_loss + ((i_xyz + another_xyz - 2 * j_xyz) ** 2).sum(-1).mean()
            )
        return vel_loss

    @torch.no_grad()
    def sort_time(self, optimizer):
        old_tid = self.tid
        remap = torch.argsort(old_tid)
        new_xyz = self._xyz[remap]
        new_rotation = self._rotation[remap]

        optimizable_tensors = replace_tensor_to_optimizer(optimizer, new_xyz, "xyz")
        self._xyz = optimizable_tensors["xyz"]
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, new_rotation, "rotation"
        )
        self._rotation = optimizable_tensors["rotation"]

        self.tid = self.tid[remap]
        assert (self.tid == torch.arange(self.T).to(self.tid)).all()
        logging.info(f"Sort time from {old_tid} to {self.tid}")
        return


############################################################
############################################################
############################################################
############################################################
############################################################


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def compute_adaptive_voxel_size(
    mu, adapt_voxel_size_knn=16, adapt_voxel_size_factor=4.0
):
    # adaptively compute voxel size
    sq_dist, _, _ = knn_points(mu[None], mu[None], K=adapt_voxel_size_knn + 1)
    sq_dist = sq_dist[0, :, 1:]
    sq_dist = sq_dist.mean(-1)
    base_voxel_size = sq_dist.median().sqrt()
    voxel_size = float(adapt_voxel_size_factor * base_voxel_size)
    return voxel_size


@torch.no_grad()
def vox_pcl_resample(base_xyz, vox_size):
    assert base_xyz.ndim == 2 and base_xyz.shape[1] == 3
    M = len(base_xyz)
    device = base_xyz.device
    pooling_vox_ind = pyg_pool.voxel_grid(pos=base_xyz, size=vox_size)
    unique_ind, compact_ind = pooling_vox_ind.unique(return_inverse=True)
    # compact_ind store each node where it should go to in the subsampled graph
    random_perm_ind = torch.randperm(M).to(compact_ind)
    perm_compact_ind = compact_ind[random_perm_ind]
    perm_src_ind = torch.arange(M).to(device)[random_perm_ind]
    ret_ind = torch.ones_like(unique_ind) * -1
    # * use scatter to randomly reduce the duplication
    ret_ind.scatter_(dim=0, index=perm_compact_ind, src=perm_src_ind)
    assert ret_ind.min() >= 0, "scater has error!"
    return ret_ind


@torch.no_grad()
def adaptive_voxel_size(xyz, adapt_voxel_size_knn=16, adapt_voxel_size_factor=4.0):
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    sq_dist, _, _ = knn_points(xyz[None], xyz[None], K=adapt_voxel_size_knn + 1)
    sq_dist = sq_dist[0, :, 1:]
    sq_dist = sq_dist.mean(-1)
    base_voxel_size = sq_dist.median().sqrt()
    voxel_size = float(adapt_voxel_size_factor * base_voxel_size)
    return voxel_size
