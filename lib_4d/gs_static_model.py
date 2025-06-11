# the GS controlling model for static scene

# given a colored pcl, construct GS models.

import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F

from gs_optim_helpers import *
import logging

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points
from lib_4d.models.networks import CNN_decoder

def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class StaticGaussian(nn.Module):
    def __init__(
        self,
        init_mean=None,  # N,3
        init_q=None,  # N,4
        init_s=None,  # N,3
        init_o=None,  # N,1
        init_rgb=None,  # N,3
        init_semantic_feature=None,
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        max_sph_order=0,
        device=torch.device("cuda:0"),
        load_fn=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.op_update_exclude = ["cnn_decoder"]

        self.register_buffer(
            "max_scale", torch.tensor(max_scale).squeeze().to(self.device)
        )
        self.register_buffer(
            "min_scale", torch.tensor(min_scale).squeeze().to(self.device)
        )
        self.max_sph_order = max_sph_order
        self._init_act(self.max_scale, self.min_scale)

        if load_fn is not None:
            self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
            self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
            self.register_buffer("max_radii2D", torch.zeros(self.N).float())
            # * init the parameters from file
            logging.info(f"Loading static model from {load_fn}")
            self.load(torch.load(load_fn, weights_only=True))
            self.summary()
            return
        else:
            assert init_mean is not None and init_rgb is not None and init_s is not None

        # * init the parameters
        self._xyz = nn.Parameter(torch.as_tensor(init_mean).float())
        if init_q is None:
            logging.warning("init_q is None, using default")
            init_q = torch.Tensor([1, 0, 0, 0]).float()[None].repeat(len(init_mean), 1)
        self._rotation = nn.Parameter(init_q)
        assert len(init_s) == len(init_mean)
        assert init_s.ndim == 2
        assert init_s.shape[1] == 3
        self._scaling = nn.Parameter(self.s_inv_act(init_s))
        o = self.o_inv_act(init_o)
        self._opacity = nn.Parameter(o)
        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(RGB2SH(init_rgb))
        self._features_rest = nn.Parameter(torch.zeros(self.N, sph_rest_dim))
        self._semantic_feature = nn.Parameter(init_semantic_feature)
        self.semantic_feature_dim = self._semantic_feature.shape[-1]
        # self.cnn_decoder = CNN_decoder(self._semantic_feature.shape[1], 1408) # TODO: Now I just write 1408 as semantic feature dim, but it maybe be a variable - Hui


        # * init states
        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        self.summary()
        return

    def summary(self):
        logging.info(f"StaticGaussian: {self.N/1000.0:.1f}K points")
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
            return len(self._xyz)
        except:
            return 0

    @property
    def get_x(self):
        return self._xyz

    @property
    def get_R(self):
        return quaternion_to_matrix(self._rotation)

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    @property
    def get_semantic_feature(self):
        return self._semantic_feature

    def forward(self, active_sph_order=None, delete_static_node_indices=None):
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        else:
            assert active_sph_order <= self.max_sph_order
        xyz = self.get_x
        frame = self.get_R
        s = self.get_s
        o = self.get_o

        if delete_static_node_indices is not None:
            o[delete_static_node_indices] = 0

        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c
        sph = sph[:, :sph_dim]
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
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._rotation], "lr": lr_q, "name": "rotation"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"},
            {"params": [self._semantic_feature], "lr": lr_semantic_feature, "name": "semantic_feature"},
            # {"params": self.cnn_decoder.parameters(), "lr": 0.0001 ,"name": "cnn_decoder"}
        ]
        return l

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def record_xyz_grad_radii(self, viewspace_point_tensor_grad, radii, update_filter):
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
            "opacity": new_o,
            "scaling": new_s,
            "rotation": new_r,
            "semantic_feature": new_semantic_feature,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # First cat to optimizer and then return to self
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], device=self.device)
        self.xyz_gradient_denom = torch.zeros(self._xyz.shape[0], device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )
        return

    def clean_gs_control_record(self):
        self.xyz_gradient_accum = torch.zeros_like(self._xyz[:, 0])
        self.xyz_gradient_denom = torch.zeros_like(self._xyz[:, 0])
        self.max_radii2D = torch.zeros_like(self.max_radii2D)

    def append_points(
        self,
        new_tid,  # not used here
        optimizer,
        new_xyz,
        new_quat,
        new_scaling_logit,
        new_opacities_logit,
        new_features_dc,
        new_features_rest,
        verbose=True,
    ):
        if verbose:
            logging.info(f"Append: [+]{len(new_xyz)}")
        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_quat,
            new_s=new_scaling_logit,
            new_o=new_opacities_logit,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
        )
        return

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

        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=new_semantic_feature,
        )

        return len(new_xyz)

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
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
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=new_semantic_feature,
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
        return len(new_xyz)

    def densify(
        self,
        optimizer,
        max_grad,
        percent_dense,
        extent,
        max_grad_node=None,
        node_ctrl_flag=True,
        verbose=True,
    ):
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

    def random_grow(self, optimizer, num_factor=0.05, std=0.1, init_opa_value=0.1):
        # * New operation, randomly add largely disturbed points to the geometry
        ind = torch.randperm(self.N)[: int(self.N * num_factor)]
        selected_pts_mask = torch.zeros(self.N, dtype=bool, device=self.device)
        selected_pts_mask[ind] = True

        new_xyz = self._xyz[selected_pts_mask]
        noise = torch.randn_like(new_xyz) * std
        new_xyz = new_xyz + noise
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask]

        new_opacities = torch.ones_like(self._opacity[selected_pts_mask])
        new_opacities = new_opacities * self.o_inv_act(init_opa_value)

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_semantic_feature=new_semantic_feature,
        )
        logging.info(f"Random grow: {len(new_xyz)}")
        return len(new_xyz)

    def prune_points(
        self,
        optimizer,
        min_opacity,
        max_screen_size,
        verbose=True,
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        logging.info(f"opacity_pruning {prune_mask.sum()}")
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            logging.info(f"radii2D_pruning {big_points_vs.sum()}")
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
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
        # self.cnn_decoder = CNN_decoder(self._semantic_feature.shape[1], 1408) # TODO: Now I just write 1408 as semantic feature dim, but it maybe be a variable - Hui
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
        # load others
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)
        return
