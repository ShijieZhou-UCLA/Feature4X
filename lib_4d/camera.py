import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os, sys, os.path as osp
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from graph_utils import *

# from solver_utils import *
from lib_4d_misc import *

# ! note, the image center is not opitimizable at this moment


class SimpleFovCamerasDelta(nn.Module):
    def __init__(self, T, fovdeg_init, cxcy_ratio=None) -> None:
        super().__init__()
        self.T = T
        self.is_delta_flag = True
        if cxcy_ratio is None:
            cxcy_ratio = [0.5, 0.5]
        self.register_buffer("cxcy_ratio", torch.Tensor(cxcy_ratio))
        if cxcy_ratio != [0.5, 0.5]:
            logging.info(f"Set cxcy_ratio to {cxcy_ratio}")

        focal_init = 1.0 / np.tan(np.deg2rad(fovdeg_init) / 2.0)
        focal = torch.Tensor([focal_init]).squeeze()

        param_cam_q = torch.zeros(self.T, 4)
        param_cam_q[:, 0] = 1.0
        param_cam_t = torch.zeros(self.T, 3)

        self.rel_focal = nn.Parameter(focal)
        self.q_wc = nn.Parameter(param_cam_q)
        self.t_wc = nn.Parameter(param_cam_t)
        return

    @property
    def focal(self):
        return self.rel_focal

    @property
    def fov(self):
        focal = self.rel_focal.item()
        half_angle = np.arctan(1.0 / focal)
        angle = np.rad2deg(half_angle * 2.0)
        return angle

    def forward(self):
        raise NotImplemented()

    def get_optimizable_list(self, lr_q=1e-4, lr_t=1e-4, lr_f=1e-4):
        ret = [
            {"params": [self.q_wc], "lr": lr_q, "name": "R"},
            {"params": [self.t_wc], "lr": lr_t, "name": "t"},
            {"params": [self.rel_focal], "lr": lr_f, "name": "f"},
        ]
        return ret

    def init_ith_cam_from_prev(self, ind):
        logging.info(f"Copy cam at {ind-1} to init {ind}")
        assert ind >= 1
        with torch.no_grad():
            self.q_wc[ind] = self.q_wc[ind - 1]
            self.t_wc[ind] = self.t_wc[ind - 1]
        return

    def Rt_cw(self, ind):
        R_wc, t_wc = self.Rt_wc(ind)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw

    def Rt_wc(self, ind):
        #     R_wc = quaternion_to_matrix(F.normalize(self.q_wc[ind : ind + 1], dim=-1))[0]
        #     t_wc = self.t_wc[ind]
        T_list = self.forward_T()
        return T_list[ind, :3, :3], T_list[ind, :3, -1]

    def Rt_wc_list(self):
        #     R_wc = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
        #     t_wc = self.t_wc
        T_list = self.forward_T()
        return T_list[:, :3, :3], T_list[:, :3, -1]

    def Rt_cw_list(self):
        R_wc, t_wc = self.Rt_wc_list()
        R_cw = R_wc.transpose(1, 2)
        t_cw = -torch.einsum("bij,bj->bi", R_cw, t_wc)
        return R_cw, t_cw

    def forward_T(self):
        dR = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
        dt = self.t_wc
        dT_list = torch.cat([dR, dt[..., None]], -1)
        bottom = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(dt)[None].expand(self.T, -1)
        dT_list = torch.cat([dT_list, bottom[:, None]], 1)
        ret = [dT_list[0]]  # ! the first frame can also be optimized
        for dT in dT_list[1:]:
            T = dT @ ret[-1]
            ret.append(T)
        ret = torch.stack(ret)
        return ret

    def delta_angle_trans(self):
        d_angle = quaternion_to_axis_angle(F.normalize(self.q_wc, dim=-1)).norm(
            dim=-1
        )  # rad
        d_trans = self.t_wc.norm(dim=-1)
        return d_angle, d_trans

    def T_wc(self, ind):
        R_wc, t_wc = self.Rt_wc(ind)
        T_wc = torch.eye(4).to(R_wc)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc
        return T_wc

    def T_cw(self, ind):
        R_cw, t_cw = self.Rt_cw(ind)
        T_cw = torch.eye(4).to(R_cw)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw
        return T_cw

    def Rt_ij(self, i, j):
        T_wi = self.T_wc(i)
        T_wj = self.T_wc(j)
        T_ij = T_wi.inverse() @ T_wj
        R_ij = T_ij[:3, :3]
        t_ij = T_ij[:3, 3]
        return R_ij, t_ij
    
    def backproject(self, uv, d):
        # uv: always be [-1,+1] on the short side
        assert uv.ndim == d.ndim + 1
        assert uv.shape[-1] == 2
        dep = d[..., None]
        rel_f = torch.as_tensor(self.rel_focal).to(uv)
        cxcy = torch.as_tensor(self.cxcy_ratio).to(uv) * 2.0 - 1.0
        xy = (uv - cxcy[None, :]) * dep / rel_f
        z = dep
        xyz = torch.cat([xy, z], dim=-1)
        return xyz
    
    def project(self, xyz, th=1e-5):
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
        rel_f = torch.as_tensor(self.rel_focal).to(xyz)
        cxcy = torch.as_tensor(self.cxcy_ratio).to(xyz) * 2.0 - 1.0
        uv = (xy * rel_f / z) + cxcy[None, :]
        return uv  # [-1,1]



class SimpleFovCamerasIndependent(nn.Module):
    def __init__(
        self,
        T=None,
        fovdeg_init=None,
        gt_pose=None,
        cam_delta: SimpleFovCamerasDelta = None,
        cxcy_ratio=None,
    ) -> None:
        super().__init__()

        # ! for now, not optimizable
        if cxcy_ratio is None:
            cxcy_ratio = [0.5, 0.5]
        self.register_buffer("cxcy_ratio", torch.Tensor(cxcy_ratio))

        if cxcy_ratio != [0.5, 0.5]:
            logging.info(f"Set cxcy_ratio to {cxcy_ratio}")

        self.is_delta_flag = False
        if cam_delta is None:
            self.__construct_from_args__(T, fovdeg_init, gt_pose)
        else:
            self.__construct_from_delta__(cam_delta)
        return

    def __construct_from_delta__(self, cam_delta):
        self.T = cam_delta.T
        focal = cam_delta.rel_focal.detach().clone()
        T = cam_delta.forward_T().detach().clone()
        param_cam_q_wc = matrix_to_quaternion(T[:, :3, :3])
        param_cam_t_wc = T[:, :3, -1]
        self.rel_focal = nn.Parameter(focal)
        self.q_wc = nn.Parameter(param_cam_q_wc)
        self.t_wc = nn.Parameter(param_cam_t_wc)
        return

    def __construct_from_args__(self, T, fovdeg_init, gt_pose=None):
        self.T = T
        focal_init = 1.0 / np.tan(np.deg2rad(fovdeg_init) / 2.0)
        focal = torch.Tensor([focal_init]).squeeze()
        if gt_pose is not None:
            gt_pose = torch.as_tensor(gt_pose).float()
            R, t = gt_pose[:, :3, :3].float(), gt_pose[:, :3, -1].float()
            param_cam_q_wc = matrix_to_quaternion(R)
            param_cam_t_wc = t
        else:
            param_cam_q_wc = torch.zeros(self.T, 4)
            param_cam_q_wc[:, 0] = 1.0
            param_cam_t_wc = torch.zeros(self.T, 3)
        self.rel_focal = nn.Parameter(focal)
        self.q_wc = nn.Parameter(param_cam_q_wc)
        self.t_wc = nn.Parameter(param_cam_t_wc)
        return

    @property
    def focal(self):
        return self.rel_focal

    @property
    def fov(self):
        focal = self.rel_focal.item()
        half_angle = np.arctan(1.0 / focal)
        angle = np.rad2deg(half_angle * 2.0)
        return angle

    def forward(self):
        raise NotImplemented()

    def get_optimizable_list(self, lr_q=1e-4, lr_t=1e-4, lr_f=1e-4):
        ret = [
            {"params": [self.q_wc], "lr": lr_q, "name": "R"},
            {"params": [self.t_wc], "lr": lr_t, "name": "t"},
            {"params": [self.rel_focal], "lr": lr_f, "name": "f"},
        ]
        return ret

    def init_ith_cam_from_prev(self, ind):
        logging.info(f"Copy cam at {ind-1} to init {ind}")
        assert ind >= 1
        with torch.no_grad():
            self.q_wc[ind] = self.q_wc[ind - 1]
            self.t_wc[ind] = self.t_wc[ind - 1]
        return

    def Rt_cw(self, ind):
        R_wc, t_wc = self.Rt_wc(ind)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw

    def Rt_wc(self, ind):
        R_wc = quaternion_to_matrix(F.normalize(self.q_wc[ind : ind + 1], dim=-1))[0]
        t_wc = self.t_wc[ind]
        return R_wc, t_wc

    def Rt_wc_list(self):
        R_wc = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
        t_wc = self.t_wc
        return R_wc, t_wc

    def Rt_cw_list(self):
        R_wc, t_wc = self.Rt_wc_list()
        R_cw = R_wc.transpose(1, 2)
        t_cw = -torch.einsum("bij,bj->bi", R_cw, t_wc)
        return R_cw, t_cw
    
    def T_wc_list(self):
        R_wc, t_wc = self.Rt_wc_list()
        ret = torch.cat([R_wc, t_wc[...,None]], -1)
        bottom = torch.tensor([0.0,0.0,0.0,1.0]).to(ret)
        ret = torch.cat([ret, bottom[None,None,:].expand(len(R_wc), -1,-1)],-2)
        return ret

    def T_wc(self, ind):
        R_wc, t_wc = self.Rt_wc(ind)
        T_wc = torch.eye(4).to(R_wc)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc
        return T_wc

    def T_cw(self, ind):
        R_cw, t_cw = self.Rt_cw(ind)
        T_cw = torch.eye(4).to(R_cw)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw
        return T_cw

    def Rt_ij(self, i, j):
        T_wi = self.T_wc(i)
        T_wj = self.T_wc(j)
        T_ij = T_wi.inverse() @ T_wj
        R_ij = T_ij[:3, :3]
        t_ij = T_ij[:3, 3]
        return R_ij, t_ij

    def trans_pts_to_world(self, tid, pts_c):
        assert pts_c.ndim == 2 and pts_c.shape[1] == 3
        R, t = self.Rt_wc(tid)
        pts_w = torch.einsum("ij,nj->ni", R, pts_c) + t
        return pts_w

    def trans_pts_to_cam(self, tid, pts_w):
        assert pts_w.ndim == 2 and pts_w.shape[1] == 3
        R, t = self.Rt_cw(tid)
        pts_c = torch.einsum("ij,nj->ni", R, pts_w) + t
        return pts_c
    
    ######
    def backproject(self, uv, d):
        # uv: always be [-1,+1] on the short side
        assert uv.ndim == d.ndim + 1
        assert uv.shape[-1] == 2
        dep = d[..., None]
        rel_f = torch.as_tensor(self.rel_focal).to(uv)
        cxcy = torch.as_tensor(self.cxcy_ratio).to(uv) * 2.0 - 1.0
        xy = (uv - cxcy[None, :]) * dep / rel_f
        z = dep
        xyz = torch.cat([xy, z], dim=-1)
        return xyz
    
    def project(self, xyz, th=1e-5):
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
        rel_f = torch.as_tensor(self.rel_focal).to(xyz)
        cxcy = torch.as_tensor(self.cxcy_ratio).to(xyz) * 2.0 - 1.0
        uv = (xy * rel_f / z) + cxcy[None, :]
        return uv  # [-1,1]
