import sys, os, os.path as osp
import torch_geometric.nn.pool as pyg_pool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F

from gs_optim_helpers import *
import logging
import time
from matplotlib import pyplot as plt

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points
import open3d as o3d
from tqdm import tqdm

from scf4d_model import Scaffold4D
from lib_4d.models.networks import CNN_decoder

import yaml


class DynSCFGaussian(nn.Module):
    #################################################################
    # * Trace buffers and parameters
    #################################################################
    # * buffers
    # max_scale
    # min_sacle
    # max_sph_order

    # attach_ind
    # ref_time

    # xyz_gradient_accum
    # xyz_gradient_denom
    # max_radii2D

    # * parameters
    # _xyz
    # _rotation
    # _scaling
    # _opacity
    # _features_dc
    # _features_rest
    # _skinning_weight
    # _dynamic_logit
    #################################################################

    def __init__(
        self,
        scf: Scaffold4D = None,
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        max_sph_order=0,
        device=torch.device("cuda:0"),
        max_node_num=10000,
        leaf_local_flag=True,  # save nodes to local nodes
        init_N=None,
        node_ctrl=True,
        nn_fusion=-1,  # if -1 use all gs, otherwise, find the nearest n time frame to warp
    ) -> None:

        # * hardcoded settings
        use_dyn_o = False

        # * Init only init the pre-computed SCF, later will append leaves!
        super().__init__()
        self.device = device
        self.op_update_exclude = [
            "node_xyz",
            "node_rotation",
            "node_sigma",
            "node_semantic_feature",
            "sk_rotation",
            "cnn_decoder",
        ]

        # self.nn_fusion = nn_fusion
        self.register_buffer("nn_fusion", torch.tensor(nn_fusion))
        logging.info(f"ED Model use default {nn_fusion} nearest frame fusion")

        self.scf = scf
        self.semantic_feature_dim = self.scf.semantic_feature_dim

        self.node_ctrl = node_ctrl
        self.use_dyn_o = use_dyn_o  # ! This use_dyn_o only determin correction
        if self.node_ctrl:
            logging.info(
                f"Node control is enabled! (max number of nodes {max_node_num})"
            )

        self.register_buffer("w_correction_flag", torch.tensor([0]).squeeze().bool())
        self.register_buffer(
            "leaf_local_flag", torch.tensor([leaf_local_flag]).squeeze().bool()
        )

        # * prepare activation
        self.register_buffer("max_scale", torch.tensor(max_scale).to(self.device))
        self.register_buffer("min_scale", torch.tensor(min_scale).to(self.device))
        self.register_buffer(
            "max_sph_order", torch.tensor(max_sph_order).to(self.device)
        )
        self._init_act(self.max_scale, self.min_scale)

        # * Init the empty leaf attr
        if init_N is None:
            init_N = self.N
        self._xyz = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._rotation = nn.Parameter(torch.zeros(init_N, 4))  # N,4
        self._scaling = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._opacity = nn.Parameter(torch.zeros(init_N, 1))  # N,1
        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(torch.zeros(init_N, 3))  # N,3
        self._features_rest = nn.Parameter(torch.zeros(init_N, sph_rest_dim))
        # self._semantic_feature = nn.Parameter(torch.zeros(init_N, NUM_SEMANTIC_CHANNELS))  # N, (changed 37 to 128)
        # self.cnn_decoder = CNN_decoder(self._semantic_feature.shape[1], 1408) # TODO: Now I just write 1408 as semantic feature dim, but it maybe be a variable - Hui
        self._skinning_weight = nn.Parameter(torch.zeros(init_N, self.scf.skinning_k))
        self._dynamic_logit = nn.Parameter(self.o_inv_act(torch.ones(init_N, 1) * 0.99))
        # * leaf important status
        self.register_buffer("attach_ind", torch.zeros(init_N).long())  # N
        self.register_buffer("ref_time", torch.zeros(init_N).long())  # N
        # * init states
        self.register_buffer("xyz_gradient_accum", torch.zeros(init_N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(init_N).long())
        self.register_buffer("max_radii2D", torch.zeros(init_N).float())

        self.to(self.device)
        self.summary(lite=True)
        return

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        # first recover the
        scf_sub_ckpt = {k[4:]: v for k, v in ckpt.items() if k.startswith("scf.")}
        scf = Scaffold4D.load_from_ckpt(scf_sub_ckpt, device=device)
        model = cls(
            scf=scf,
            device=device,
            init_N=ckpt["_xyz"].shape[0],
            max_sph_order=ckpt["max_sph_order"],
        )
        model.load_state_dict(ckpt, strict=True)
        model.summary()
        return model

    def __load_from_file_init__(self, load_fn):
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())
        self.register_buffer("attach_ind", torch.zeros(self.N))
        self.register_buffer("ref_time", torch.zeros(self.N))  # N
        # * init the parameters from file
        logging.info(f"Loading dynamic model from {load_fn}")
        self.load(torch.load(load_fn))
        self.forward(0)
        self.summary()
        return

    def summary(self, lite=False):
        logging.info(
            f"DenseDynGaussian: {self.N/1000.0:.1f}K points; {self.M} Nodes; K={self.scf.skinning_k}; and {self.T} time step"
        )
        if lite:
            return
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info("-" * 30)
        logging.info(
            f"ED MODEL DEBUG: w_corr: max={self._skinning_weight.max()}, min={self._skinning_weight.min()}, mean={self._skinning_weight.mean()}"
        )
        logging.info(f"ED MODEL DEBUG: use nn_fusion={self.nn_fusion}")
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
            return self._xyz.shape[0]
        except:
            return 0

    @property
    def M(self):
        return self.scf.M

    @property
    def T(self):
        return self.scf.T

    @property
    def get_t_range(self):
        # ! this is not useful in current version, but to be compatible with old, maintain this
        return 0, self.T - 1

    def get_tlist_ind(self, t):
        # ! this is not useful in current version, but to be compatible with old, maintain this
        assert t < self.T
        return t

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_d(self):
        return self.o_act(self._dynamic_logit)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    # @property
    # def get_semantic_feature(self):
    #     return self._semantic_feature

    def get_xyz(self):
        if self.leaf_local_flag:
            nn_ref_node_xyz, sk_ref_node_quat, sk_ref_node_semantic_feature = self.scf.get_async_knns(
                self.ref_time, self.attach_ind[:, None]
            )
            nn_ref_node_xyz = nn_ref_node_xyz.squeeze(1)
            nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
            return (
                torch.einsum("nij,nj->ni", nn_ref_node_R_wi, self._xyz)
                + nn_ref_node_xyz
            )
        else:
            return self._xyz

    def get_R_mtx(self):
        if self.leaf_local_flag:
            _, sk_ref_node_quat, sk_ref_node_semantic_feature = self.scf.get_async_knns(
                self.ref_time, self.attach_ind[:, None]
            )
            nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
            return torch.einsum("nij,njk->nik", nn_ref_node_R_wi, q2R(self._rotation))
        else:
            return q2R(self._rotation)

    def set_surface_deform(self):
        # * different to using RBF field approximate the deformation field (more flexible when changing the scf topology), set the deformation to surface model, the skinning is saved on each GS
        logging.info(f"ED Model convert to surface mode")
        self.w_correction_flag = torch.tensor(True).to(self.device)
        self.scf.fixed_topology_flag = torch.tensor(True).to(self.device)
        return

    def forward(self, t: int, active_sph_order=None, nn_fusion=None, delete_node_indices=None):
        assert t < self.T, "t is out of range!"
        if active_sph_order is None:
            active_sph_order = int(self.max_sph_order)
        else:
            assert active_sph_order <= self.max_sph_order

        s = self.get_s
        o = self.get_o
        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c[:, :sph_dim]
        # semantic_feature = self.get_semantic_feature

        mu_live, fr_live, semantic_feature_live = self.scf.warp(
            attach_node_ind=self.attach_ind,
            query_xyz=self.get_xyz(), # the nearest ref point
            query_dir=self.get_R_mtx(),
            query_tid=self.ref_time,
            target_tid=t,
            skinning_w_corr=self._skinning_weight if self.w_correction_flag else None, # a parameter
            # dyn_o_flag=self.use_dyn_o, # ! This use_dyn_o only determin correction
            dyn_o_corr=self.get_d.squeeze(1) if self.use_dyn_o else None,
        )
        # print('sph====', sph.shape)
        if delete_node_indices is not None:
            mask = torch.isin(self.attach_ind, delete_node_indices)
            # o[mask] = 0
            sph[mask, :3] = torch.Tensor([1, 0, 0])

        if nn_fusion is None:
            nn_fusion = self.nn_fusion.item()
        if nn_fusion > 0:
            # filter the gaussian by nn mask
            with torch.no_grad():
                supporting_t = torch.unique(self.ref_time)
                # find nearest supporting t
                t_dist = (supporting_t - t).abs()
                supporting_t = supporting_t[torch.argsort(t_dist)[:nn_fusion]]
                # get the nn mask
                filter_mask = (self.ref_time[:, None] == supporting_t[None]).any(dim=1)
            if not filter_mask.any():
                logging.warning(f"nn_fusion has no gs supporting, dummy mask!")
                # ! set the first two gs to be visible
                filter_mask[:2] = True
            # mu_live = mu_live[filter_mask]
            # fr_live = fr_live[filter_mask]
            # s = s[filter_mask]
            # o = o[filter_mask]
            # sph = sph[filter_mask]
            # ! do this by mask the opacity to zero
            o = o * filter_mask[:, None].float()  # ! mask the opacity to zero
        return mu_live, fr_live, s, o, sph, semantic_feature_live

    def get_optimizable_list(
        self,
        # leaf
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
        lr_semantic_feature=0.0005,
        lr_dyn=0.01,
        lr_w=None,
        # node
        lr_np=0.0001,
        lr_nq=0.0001,
        lr_nsig=0.00001,
        lr_sk_q=0.0001,
    ):
        l = []
        if lr_p is not None:
            l.append({"params": [self._xyz], "lr": lr_p, "name": "xyz"})
        if lr_q is not None:
            l.append({"params": [self._rotation], "lr": lr_q, "name": "rotation"})
        if lr_s is not None:
            l.append({"params": [self._scaling], "lr": lr_s, "name": "scaling"})
        if lr_o is not None:
            l.append({"params": [self._opacity], "lr": lr_o, "name": "opacity"})
        if lr_sph is not None:
            if lr_sph_rest is None:
                lr_sph_rest = lr_sph / 20
            l.append({"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"})
            l.append(
                {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"}
            )
        # if lr_semantic_feature is not None:
        #     l.append(
        #         {
        #             "params": [self._semantic_feature],
        #             "lr": lr_semantic_feature,
        #             "name": "semantic_feature",
        #         }
        #     )
        # l.append({"params": self.cnn_decoder.parameters(), "lr": 0.0001, "name": "cnn_decoder"})
        if lr_dyn is not None:
            l.append(
                {"params": [self._dynamic_logit], "lr": lr_dyn, "name": "dyn_logit"}
            )
        if lr_w is not None:
            # logging.warning("lr_w is not supported yet!")
            l.append(
                {"params": [self._skinning_weight], "lr": lr_w, "name": "skinning_w"}
            )
        l = l + self.scf.get_optimizable_list(
            lr_np=lr_np, lr_nq=lr_nq, lr_nsig=lr_nsig, lr_sk_q=lr_sk_q, lr_semantic_feature=lr_semantic_feature
        )
        return l

    ######################################################################
    # * Model Grow
    ######################################################################

    def __check_uncovered_mu__(self, mu_w, tid):
        node_xyz_live = self.scf._node_xyz[self.get_tlist_ind(tid)]
        dist_sq, _, _ = knn_points(mu_w[None], node_xyz_live[None], K=1)
        covered_by_existing_node = dist_sq[0].squeeze(-1) < self.scf.spatial_unit**2
        uncovered_mu_w = mu_w[~covered_by_existing_node]
        return uncovered_mu_w

    @torch.no_grad()
    def append_new_gs(self, optimizer, tid, mu_w, quat_w, scales, opacity, rgb, semantic_feature):
        # ! note, this function never grow new nodes for now
        # * Append leaves
        if scales.ndim == 1:
            scales = scales[:, None].expand(-1, 3)
        new_s_logit = self.s_inv_act(scales)
        assert opacity.ndim == 2 and opacity.shape[1] == 1
        new_o_logit = self.o_inv_act(opacity)
        new_feat_dc = RGB2SH(rgb)
        new_feat_rest = torch.zeros(len(scales), self._features_rest.shape[1]).to(
            self.device
        )
        new_semantic_feature = semantic_feature
        # first update the knn ind
        _, attach_ind, _ = knn_points(mu_w[None], self.scf._node_xyz[tid][None], K=1)
        self.attach_ind = torch.cat([self.attach_ind, attach_ind[0, :, 0]], dim=0)
        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )

        # if use local leaf storage, have to convert to local!
        if self.leaf_local_flag:
            attach_node_xyz = self.scf._node_xyz[tid][attach_ind[0, :, 0]]
            attach_node_R_wi = q2R(self.scf._node_rotation[tid][attach_ind[0, :, 0]])
            mu_save = torch.einsum(
                "nji,nj->ni", attach_node_R_wi, mu_w - attach_node_xyz
            )
            R_new = torch.einsum("nji,njk->nik", attach_node_R_wi, q2R(quat_w))
            quat_save = matrix_to_quaternion(R_new)
        else:
            mu_save, quat_save = mu_w, quat_w

        # finally update the parameters
        self._densification_postprocess(
            optimizer,
            new_xyz=mu_save,  # ! store the position in live frame
            new_r=quat_save,
            new_s_logit=new_s_logit,
            new_o_logit=new_o_logit,
            new_sph_dc=new_feat_dc,
            # new_semantic_feature=new_semantic_feature,
            new_sph_rest=new_feat_rest,
            new_skinning_w=torch.zeros(len(mu_w), self.scf.skinning_k).to(mu_w),
            new_dyn_logit=self.o_inv_act(torch.ones_like(scales[:, :1]) * 0.99),
        )
        # self.summary(lite=True)
        return

    ######################################################################
    # * Node Control
    ######################################################################

    @torch.no_grad()
    def append_new_node_and_gs(
        self,
        optimizer,
        tid,
        mu_w,
        quat_w,
        scales,
        opacity,
        rgb,
        # semantic_feature,
    ):
        # ! note, this function never grow new nodes for now

        # * Analysis and append the topology
        # check whether the new leaves covered by existing nodes

        # ! all uncovered
        new_node_xyz_l = subsample_vtx(mu_w, self.scf.spatial_unit)
        new_node_quat_l = (
            torch.tensor([1.0, 0.0, 0.0, 0.0])
            .to(self.device)[None]
            .expand(len(new_node_xyz_l), -1)
        )

        logging.info(f"Grow [{len(new_node_xyz_l)}] new nodes from {len(mu_w)} leaves")
        old_M = self.scf.M
        succ = self.scf.append_nodes_pnt(
            optimizer,
            new_node_xyz_l,
            new_node_quat_l,
            tid,
        )
        if not succ:
            logging.warning(f"ED model grow failed, skip the append!")
            return

        if scales.ndim == 1:
            scales = scales[:, None].expand(-1, 3)
        new_s_logit = self.s_inv_act(scales)
        assert opacity.ndim == 2 and opacity.shape[1] == 1
        new_o_logit = self.o_inv_act(opacity)
        new_feat_dc = RGB2SH(rgb)
        new_feat_rest = torch.zeros(len(scales), self._features_rest.shape[1]).to(
            self.device
        )
        # new_semantic_feature = semantic_feature

        # first update the knn ind
        # ! always attach to newly added node!
        _, attach_ind, _ = knn_points(
            mu_w[None], self.scf._node_xyz[tid][old_M:][None], K=1
        )
        attach_ind = attach_ind + old_M
        self.attach_ind = torch.cat([self.attach_ind, attach_ind[0, :, 0]], dim=0)
        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )

        # if use local leaf storage, have to convert to local!
        if self.leaf_local_flag:
            attach_node_xyz = self.scf._node_xyz[tid][attach_ind[0, :, 0]]
            attach_node_R_wi = q2R(self.scf._node_rotation[tid][attach_ind[0, :, 0]])
            mu_save = torch.einsum(
                "nji,nj->ni", attach_node_R_wi, mu_w - attach_node_xyz
            )
            R_new = torch.einsum("nji,njk->nik", attach_node_R_wi, q2R(quat_w))
            quat_save = matrix_to_quaternion(R_new)
        else:
            mu_save, quat_save = mu_w, quat_w

        # finally update the parameters
        old_N = self.N
        self._densification_postprocess(
            optimizer,
            new_xyz=mu_save,  # ! store the position in live frame
            new_r=quat_save,
            new_s_logit=new_s_logit,
            new_o_logit=new_o_logit,
            new_sph_dc=new_feat_dc,
            new_sph_rest=new_feat_rest,
            # new_semantic_feature=new_semantic_feature,
            new_skinning_w=torch.zeros(len(mu_w), self.scf.skinning_k).to(mu_w),
            new_dyn_logit=self.o_inv_act(torch.ones_like(scales[:, :1]) * 0.99),
        )
        # self.summary(lite=True)
        # check_xyz = self.forward(tid)[0]
        # error = (check_xyz[old_N:] - mu_w).norm(dim=-1).max()
        # logging.info(f"error append check error {error:.3f}")
        # verify the append xyz is still there
        return

    # @torch.no_grad()
    # def __node_densify__(self, optimizer, leaf_grads, max_grad):
    #     torch.cuda.empty_cache()
    #     # * Split
    #     # compute gathered node grad and overflow flag
    #     attach_ind = self.attach_ind
    #     leaf_over_mask = (leaf_grads > max_grad).float()
    #     node_mean_overflow = torch.zeros(self.M, device=self.device)
    #     node_mean_overflow = node_mean_overflow.scatter_reduce(
    #         dim=0,
    #         index=attach_ind,
    #         src=leaf_over_mask,
    #         include_self=False,
    #         reduce="mean",
    #     )
    #     node_mean_grad = torch.zeros(self.M, device=self.device)
    #     node_mean_grad = node_mean_grad.scatter_reduce(
    #         dim=0,
    #         index=attach_ind,
    #         src=leaf_grads,
    #         include_self=False,
    #         reduce="mean",
    #     )
    #     node_max_grad = torch.zeros(self.M, device=self.device)
    #     node_max_grad = node_max_grad.scatter_reduce(
    #         dim=0,
    #         index=attach_ind,
    #         src=leaf_grads,
    #         include_self=False,
    #         reduce="max",
    #     )

    #     # ! to decide which to use
    #     leaf_count = torch.zeros(self.M, device=self.device)
    #     leaf_count = leaf_count.scatter_add(
    #         dim=0, index=attach_ind, src=torch.ones_like(attach_ind).float()
    #     )

    #     node_split_mask = (node_mean_grad > max_grad) & (leaf_count >= 2)
    #     add_M = node_split_mask.sum().item()
    #     logging.info(
    #         f"Node split {add_M} (mean>th); ({(node_max_grad>max_grad).sum().item()} has leaves grad>th)"
    #     )

    #     if self.scf.M + add_M > self.scf.max_node_num:
    #         logging.warning(
    #             f"Node number {self.scf.M} + {add_M} exceeds the limit {self.scf.max_node_num}, too many nodes, to maintain efficiency and the topology, skip the split!"
    #         )
    #         return

    #     # actual split
    #     # compute the new node position and update the attach id for some of the leaves
    #     # ! naive for loop, later can speed up!
    #     _xyz_list, _quat_list = [], []
    #     for t in tqdm(range(self.T)):
    #         _xyz, _quat, _, _, _ = self.forward(t)
    #         _xyz_list.append(_xyz.detach().cpu())
    #         _quat_list.append(_quat.detach().cpu())
    #     _xyz_list = torch.stack(_xyz_list, 0)
    #     _quat_list = torch.stack(_quat_list, 0)

    #     # ! select the max grad leaf in the node supporting sub set and use his traj to clone!
    #     node_split_ind = torch.arange(self.M, device=self.device)[node_split_mask]
    #     new_node_xyz_list, new_node_quat_list = [], []
    #     new_xyz, new_rotation = self._xyz.clone(), self._rotation.clone()
    #     new_attach_ind = attach_ind.clone()
    #     if add_M == 0:
    #         logging.info("No node to split, skip!")
    #         self.scf.update_topology()
    #         # early stop
    #         return
    #     for i in tqdm(range(add_M)):
    #         node_ind = node_split_ind[i]
    #         ori_xyz_traj = self.scf._node_xyz[:, node_ind]
    #         leaf_mask = attach_ind == node_ind
    #         assert (
    #             leaf_mask.sum() >= 2
    #         ), "the split node must have at least two leaf because this is called after leaf densification"
    #         max_grad_leaf_ind = torch.argmax(leaf_grads * leaf_mask.float())
    #         new_xyz_traj = _xyz_list[:, max_grad_leaf_ind].to(self.device)
    #         new_quat_traj = _quat_list[:, max_grad_leaf_ind].to(self.device)
    #         new_node_xyz_list.append(new_xyz_traj)
    #         new_node_quat_list.append(matrix_to_quaternion(new_quat_traj))
    #         # update sum the attach ind to the new node
    #         ref_time = self.ref_time[leaf_mask]
    #         ori_xyz_ref = ori_xyz_traj[ref_time]
    #         new_xyz_ref = new_xyz_traj[ref_time]
    #         # make sure each node at least have a leaf
    #         # ! have to consider the change of attach node!
    #         dist_2_ori = (self.get_xyz()[leaf_mask] - ori_xyz_ref).norm(dim=-1)
    #         dist_2_new = (self.get_xyz()[leaf_mask] - new_xyz_ref).norm(dim=-1)
    #         # assert dist_2_new.min() < 1e-6, "the max grad one should be in the set"
    #         ori_safe_ind = dist_2_ori.argmin()
    #         new_safe_ind = dist_2_new.argmin()
    #         if ori_safe_ind == new_safe_ind:
    #             # solve the conflict
    #             dist_2_new[ori_safe_ind] = 1e6
    #             new_safe_ind = dist_2_new.argmin()
    #         _up = dist_2_new < dist_2_ori
    #         _up[ori_safe_ind] = False
    #         _up[new_safe_ind] = True
    #         update_mask = leaf_mask.clone()
    #         update_mask[leaf_mask] = _up
    #         assert update_mask.any(), "must update some leaf to the new node"
    #         # ! fix below bug 2024.4.28
    #         # update_mask[leaf_mask] = dist_2_new < dist_2_ori
    #         new_attach_ind[update_mask] = self.M + i
    #         assert update_mask.any(), "assign shift error"
    #         # ! note, the node split when partitioning the node's supporting leaves into two set, the reference time does not change!
    #         if self.leaf_local_flag:
    #             # ! if save in local frame, when changing the attach node, have to change the local xyz and rot!
    #             ori_mu_w = self.get_xyz()[update_mask]
    #             ori_R_w = self.get_R_mtx()[update_mask]
    #             # convert this world coords into self.M+i's local coords
    #             new_node_xyz = new_node_xyz_list[-1][
    #                 self.ref_time[update_mask]
    #             ]  # different reference time maybe
    #             new_node_quat = new_node_quat_list[-1][self.ref_time[update_mask]]
    #             new_node_R_wi = q2R(new_node_quat)
    #             new_mu_i = torch.einsum(
    #                 "nji,nj->ni", new_node_R_wi, ori_mu_w - new_node_xyz
    #             )
    #             new_R_i = torch.einsum("nji,njk->nik", new_node_R_wi, ori_R_w)
    #             new_quat_i = matrix_to_quaternion(new_R_i)
    #             new_xyz[update_mask] = new_mu_i
    #             new_rotation[update_mask] = new_quat_i

    #     # replace xyz and rotation
    #     optimizable_tensors = replace_tensor_to_optimizer(optimizer, new_xyz, "xyz")
    #     self._xyz = optimizable_tensors["xyz"]
    #     optimizable_tensors = replace_tensor_to_optimizer(
    #         optimizer, new_rotation, "rotation"
    #     )
    #     self._rotation = optimizable_tensors["rotation"]

    #     # ! fix bellow bug  2024.4.28
    #     # attach_ind = new_attach_ind  # ! because when scan, the nodes attach ind is used, but the nodes are not appended, so have to update this later
    #     self.attach_ind = new_attach_ind

    #     new_node_xyz_list = torch.stack(new_node_xyz_list, 1)
    #     new_node_quat_list = torch.stack(new_node_quat_list, 1)
    #     new_node_sigma_logit = self.scf._node_sigma_logit[node_split_mask]
    #     if self.scf.gs_sk_approx_flag:
    #         new_node_sk_quat_list = self.scf._sk_rotation[:, node_split_mask]
    #     else:
    #         new_node_sk_quat_list = None
    #     self.scf.append_nodes_traj(
    #         optimizer,
    #         new_node_xyz_list,
    #         new_node_quat_list,
    #         self.T - 1,  # use last tid
    #         new_node_sigma_logit=new_node_sigma_logit,
    #         new_node_sk_q_traj=new_node_sk_quat_list,
    #     )
    #     logging.info(f"Node split: [+{add_M}]; now has {self.M} nodes")
    #     assert attach_ind.max() < self.M
    #     self.scf.update_topology()
    #     assert (
    #         self.get_node_attached_leaves_count()[-add_M:].min() >= 1
    #     ), "split error, the attach ind is not set properly!"
    #     return

    @torch.no_grad()
    def prune_nodes(self, optimizer, prune_sk_th=0.02, viz_fn=None):
        # if a node is not carrying leaves, and the effect to all neighbors are small. then can prune it; and also update the knn skinning weight, during this update, also have to be careful about the inner scf-knn-ind for scf, only replace some where!!!

        acc_w = self.get_node_sinning_w_acc("max")
        # viz
        if viz_fn is not None:
            fig = plt.figure(figsize=(10, 5))
            plt.hist(acc_w.cpu().numpy(), bins=100)
            plt.plot([prune_sk_th, prune_sk_th], [0, 100], "r--")
            plt.title(f"Node Max supporting sk-w hist")
            plt.savefig(f"{viz_fn}prune_sk_hist.jpg")
            plt.close()

        prune_mask_sk = acc_w < prune_sk_th  # if prune, true

        # also check whether this node carries some leaves
        supporting_node_id = torch.unique(self.attach_ind)
        prune_mask_carry = torch.ones(self.M, device=self.device).bool()
        prune_mask_carry[supporting_node_id] = False

        node_prune_mask = prune_mask_sk & prune_mask_carry
        logging.info(
            f"Prune {node_prune_mask.sum()} nodes (max_sk<th={prune_sk_th}) with carrying check ({node_prune_mask.float().mean()*100.0:.3f}%)"
        )

        prune_M = node_prune_mask.sum().item()
        if prune_M == 0:
            return

        # first remove the leaves
        # ! actually this is not used in our case for now
        leaf_pruning_mask = (
            self.attach_ind[:, None]
            == torch.arange(self.M, device=self.device)[None, node_prune_mask]
        ).any(-1)
        if leaf_pruning_mask.any():
            self._prune_points(optimizer, leaf_pruning_mask)

        if self.w_correction_flag:
            # identify the sk corr that related to the old node
            sk_corr_affect_mask = node_prune_mask[
                self.scf.topo_knn_ind[self.attach_ind]
            ]
            logging.warning(
                f"Prune under surface mode, check {sk_corr_affect_mask.sum()}({sk_corr_affect_mask.float().mean()*100:.3f}%) sk_corr to be updated"
            )
            # ! later make these position sk to be zero

        # then update the attach ind
        new_M = self.M - prune_M
        ind_convert = torch.ones(self.M, device=self.device).long() * -1
        ind_convert[~node_prune_mask] = torch.arange(new_M, device=self.device)
        self.attach_ind = ind_convert[self.attach_ind]

        # finally remove the nodes
        self.scf.remove_nodes(optimizer, node_prune_mask)

        # now update the sk corr again, make sure the updated = 0.0
        if self.w_correction_flag:

            _, sk_w, sk_w_sum, _, _ = self.scf.get_skinning_weights(
                query_xyz=self.get_xyz(),
                query_t=self.ref_time,
                attach_ind=self.attach_ind,
                # skinning_weight_correction=self._skinning_weight,
            )
            sk_w_field = sk_w * sk_w_sum[:, None]
            # field + old_corr = sk_w
            # filed + new_corr = 0
            # sk_w - old_corr + new_corr = 0
            # new_corr = old_corr - sk_w
            new_sk_corr = self._skinning_weight.clone()
            new_sk_corr[sk_corr_affect_mask] = -sk_w_field[sk_corr_affect_mask]
            # replace sk_w again

            optimizable_tensors = replace_tensor_to_optimizer(
                optimizer,
                [new_sk_corr],
                ["skinning_w"],
            )
            self._skinning_weight = optimizable_tensors["skinning_w"]

            # _, sk_w_check, sk_w_sum_check, _, _ = self.scf.get_skinning_weights(
            #     query_xyz=self.get_xyz(),
            #     query_t=self.ref_time,
            #     attach_ind=self.attach_ind,
            #     skinning_weight_correction=self._skinning_weight,
            # )
            # check_modified_sk = sk_w_check[sk_corr_affect_mask]
            # logging.info(f"Prune check: modified slot max sk-weight={check_modified_sk.max()}")
        return

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
        new_s_logit,
        new_o_logit,
        new_sph_dc,
        new_sph_rest,
        # additional parameters
        # new_semantic_feature,
        new_skinning_w,
        new_dyn_logit,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            # "semantic_feature": new_semantic_feature,
            "opacity": new_o_logit,
            "scaling": new_s_logit,
            "rotation": new_r,
            "skinning_w": new_skinning_w,
            "dyn_logit": new_dyn_logit,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # * update parameters
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._skinning_weight = optimizable_tensors["skinning_w"]
        self._dynamic_logit = optimizable_tensors["dyn_logit"]

        # * update the recording buffer
        # ! Note, must update the other buffers outside this function
        self.xyz_gradient_accum = torch.zeros(self.N, device=self.device)
        self.xyz_gradient_denom = torch.zeros(self.N, device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
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

        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        # new_semantic_feature = self._semantic_feature[selected_pts_mask]
        new_skinning_w = self._skinning_weight[selected_pts_mask]
        new_dyn_logit = self._dynamic_logit[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s_logit=new_scaling,
            new_o_logit=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            # new_semantic_feature=new_semantic_feature,
            new_skinning_w=new_skinning_w,
            new_dyn_logit=new_dyn_logit,
        )

        # update leaf buffer
        new_attach_ind = self.attach_ind[selected_pts_mask]
        new_ref_time = self.ref_time[selected_pts_mask]
        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind], dim=0
        )  # ! now copy the topology, but the best is to recompute the topology!
        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)

        return new_xyz.shape[0]

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
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        # ! no matter local or global, such disturbance is correct
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.einsum("nij,nj->ni", rots, samples) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        # new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_dyn_logit = self._dynamic_logit[selected_pts_mask].repeat(N, 1)
        new_skinning_w = self._skinning_weight[selected_pts_mask].repeat(N, 1)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s_logit=new_scaling,
            new_o_logit=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            # new_semantic_feature=new_semantic_feature,
            new_skinning_w=new_skinning_w,
            new_dyn_logit=new_dyn_logit,
        )

        # update leaf buffer
        new_attach_ind = self.attach_ind[selected_pts_mask].repeat(N)
        new_ref_time = self.ref_time[selected_pts_mask].repeat(N)
        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind], dim=0
        )  # ! now copy the topology, but the best is to recompute the topology!
        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return new_xyz.shape[0]

    def densify(
        self,
        optimizer,
        max_grad,
        percent_dense,
        extent,
        verbose=True,
    ):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0
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
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        logging.info(f"opacity_pruning {prune_mask.sum()}")
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            logging.info(f"radii2D_pruning {big_points_vs.sum()}")
            # ! also consider the scale
            max_scale = self.get_s.max(dim=1).values
            big_points_scale = max_scale > max_3d_radius
            logging.info(f"radii3D_pruning {big_points_scale.sum()}")
            big_points_vs = torch.logical_or(big_points_vs, big_points_scale)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        # also check the scale
        too_small_mask = (self.get_s < min_scale).all(dim=-1)
        logging.info(f"small_pruning {big_points_scale.sum()}")
        prune_mask = torch.logical_or(prune_mask, too_small_mask)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")
        return

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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._skinning_weight = optimizable_tensors["skinning_w"]
        self._dynamic_logit = optimizable_tensors["dyn_logit"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # update leaf buffer
        self.attach_ind = self.attach_ind[valid_points_mask]
        self.ref_time = self.ref_time[valid_points_mask]
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
        # * manually assign self attr with the keys
        for key in ckpt:
            if hasattr(self, key):
                # if is parameter, then assign
                if isinstance(getattr(self, key), nn.Parameter):
                    getattr(self, key).data = torch.as_tensor(ckpt[key])
                else:
                    setattr(self, key, ckpt[key])
            else:
                logging.warning(f"Key [{key}] is not in the model!")

        # load others
        self._init_act(self.max_scale, self.min_scale)
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)

        return

    ######################################################################
    # * Regularization
    ######################################################################

    # ! note, all the vel and arap loss here are mean reduction
    def compute_vel_acc_loss(self, tids=None, detach_mask=None):
        return self.scf.compute_vel_acc_loss(tids, detach_mask)

    def compute_arap_loss(
        self,
        tids=None,
        temporal_diff_weight=[0.75, 0.25],
        temporal_diff_shift=[1, 4],
        detach_tids_mask=None,
    ):
        return self.scf.compute_arap_loss(
            tids, temporal_diff_weight, temporal_diff_shift, detach_tids_mask
        )

    #########
    # viz
    ########

    def get_node_sinning_w_acc(self, reduce="sum"):
        sk_ind, sk_w, _, _, _ = self.scf.get_skinning_weights(
            query_xyz=self.get_xyz(),
            query_t=self.ref_time,
            attach_ind=self.attach_ind,
            skinning_weight_correction=(
                self._skinning_weight if self.w_correction_flag else None
            ),
        )
        sk_ind, sk_w = sk_ind.reshape(-1), sk_w.reshape(-1)
        acc_w = torch.zeros_like(self.scf._node_xyz[0, :, 0])
        acc_w = acc_w.scatter_reduce(0, sk_ind, sk_w, reduce=reduce, include_self=False)
        return acc_w

    def get_node_attached_leaves_count(self):
        attach_ind = self.attach_ind
        leaf_count = torch.zeros(self.M, device=self.device)
        leaf_count = leaf_count.scatter_add(
            dim=0, index=attach_ind, src=torch.ones_like(attach_ind).float()
        )
        return leaf_count


##################################################################################


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def subsample_vtx(vtx, voxel_size):
    # vtx: N,3
    # according to surfelwarp global_config.h line28 d_node_radius=0.025 meter; and warpfieldinitializer.cpp line39 the subsample voxel is 0.7 * 0.025 meter
    # reference: VoxelSubsamplerSorting.cu line  119
    # Here use use the mean of each voxel cell
    pooling_ind = pyg_pool.voxel_grid(pos=vtx, size=voxel_size)
    unique_ind, compact_ind = pooling_ind.unique(return_inverse=True)
    candidate = torch.scatter_reduce(
        input=torch.zeros(len(unique_ind), 3).to(vtx),
        src=vtx,
        index=compact_ind[:, None].expand_as(vtx),
        dim=0,
        reduce="mean",
        # dim_size=len(unique_ind),
        include_self=False,
    )
    assert not (candidate == 0).all(dim=-1).any(), "voxel resampling has an error!"
    return candidate
