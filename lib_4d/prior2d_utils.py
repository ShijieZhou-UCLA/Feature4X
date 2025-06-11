# helpers for prior2d
import logging, os, os.path as osp, sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from graph_utils import *
import numpy as np, cv2 as cv
from tqdm import tqdm

import imageio
import cv2
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
import open3d as o3d
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from index_helper import query_image_buffer_by_pix_int_coord

import imageio.core.util
from torchvision import transforms
from glob import glob
def round_int_coordinates(coord, H, W):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask

class Langseg_feature:
    def __init__(self, feature_path:str, keep_in_memory=False):
        self.keep_in_memory = keep_in_memory
        langseg_feature_path_list = glob(osp.join(feature_path, "*_fmap_CxHxW.pt"))
        langseg_feature_path_list.sort()
        self.feature_path_list = langseg_feature_path_list
        if keep_in_memory:
            self.data = [torch.load(p, weights_only=True, map_location="cpu") for p in self.feature_path_list]
    def __len__(self):
        return len(self.feature_path_list)
    def __call__(self, t:int):
        if self.keep_in_memory:
            feat = self.data[t]
        else:
            feat = torch.load(self.feature_path_list[t], weights_only=True)
        feat = feat.permute(1, 2, 0)
        return feat
    def __getitem__(self, t):
        return self(t)
    @property
    def shape(self):
        # with frame number
        return [len(self), *self(0).shape]

class Feature_dict:
    def __init__(self, feature_dict:dict):
        self.frames = list(feature_dict.values())[0].shape[0]
        for v in feature_dict.values():
            assert len(v) == self.frames,  f"frame number not aligned, expect {self.frames} frame, got shape {v.shape}"
        self.data = feature_dict

    def __len__(self):
        return self.frames

    def __call__(self, t:int):
        return {k: v[t] for k, v in self.data.items()}
    def __getitem__(self, t):
        return self(t)
    def keys(self):
          return self.data.keys()
    def channels(self):
        return {k: v.shape[-1] for k, v in self.data.items()}

def load_semantic_features(src:str, feature_config:dict):
    # semantic_features = load_semantic_features(osp.join(src, "semantic_features"), view_list, resize)
    # feature channel in internvideo: 1408

    semantic_features = {}
    feat_names = list(feature_config["Head"].keys())

    if "sam2" in feat_names and feature_config["Head"]["sam2"]["enable"]:
        sam2_feature_path = osp.join(src, feature_config["Head"]["sam2"]["path"])
        if osp.exists(sam2_feature_path):
            sam2_semantic_features_data = torch.load(sam2_feature_path, weights_only=True)

            sam2_semantic_features = []
            for frame_idx in range(len(sam2_semantic_features_data)):
                sam2_feat = sam2_semantic_features_data[frame_idx]['backbone_fpn'][0].to(torch.float32)
                sam2_semantic_features.append(sam2_feat)
            sam2_semantic_features = torch.cat(sam2_semantic_features).permute(0, 2, 3, 1)
            semantic_features["sam2"] = sam2_semantic_features
            print(f"sam2 feature loaded from {sam2_feature_path}")
        else:
            raise ValueError(f"sam2 feature not found in {sam2_feature_path}, please first extract video feature")

    if "internvideo" in feat_names and feature_config["Head"]["internvideo"]["enable"]:
        internvideo_feature_path = osp.join(src, feature_config["Head"]["internvideo"]["path"])
        if osp.exists(internvideo_feature_path):
            internvideo_semantic_features = torch.load(internvideo_feature_path, weights_only=True)["video_feat"]
            semantic_features["internvideo"] = internvideo_semantic_features
            print(f"internvideo feature loaded from {internvideo_feature_path}")
        else:
            raise ValueError(f"internvideo feature not found in {internvideo_feature_path}, please first extract video feature")

    if "langseg" in feat_names and feature_config["Head"]["langseg"]["enable"]:
        langseg_feature_path = osp.join(src, feature_config["Head"]["langseg"]["path"])
        if osp.exists(langseg_feature_path):

            langseg_semantic_features = Langseg_feature(langseg_feature_path, keep_in_memory=True)
            semantic_features["langseg"] = langseg_semantic_features
            # langseg_semantic_features = torch.load(langseg_feature_path, weights_only=True)
            print(f"clip feature loaded from {langseg_feature_path}")
        else:
            raise ValueError(f"clip feature not found in {langseg_feature_path}, please first extract video feature")

    logging.info(f"semantic features loaded: {semantic_features.keys()}")
    semantic_features= Feature_dict(semantic_features)

    return semantic_features


def load_prior_dir(
    src,
    depth_dir="zoe_depth",
    start=0,
    end=-1,
    flow_interval=[1],  # -1,
    load_sky_flag=True,
    # resize=1.0,
    resolution=None,
    skip=1,
    # align_to_zoe=False,
    nerfies_flag=False,
    feature_config=None,
):
    # if specify the edge_interval, e.g. interval = 1 all the 2,4,8 jumps will be ignored
    # load raw data;
    # warning, the end will also be included
    
    view_list = get_view_list(src, nerfies_flag=nerfies_flag)
    
    view_pairs = read_pairs_from_txt(osp.join(src, "view_pairs.txt"))
    if end == -1:
        end = len(view_list)
    view_list = view_list[start : end + 1]
    view_list = view_list[::skip]
    view_pairs, _, _ = find_subgraph(view_list, view_pairs, len(view_list) - 1)

    # filter edge_interval
    if flow_interval != -1:
        try:
            flow_interval = list(flow_interval)
        except:
            pass
        view_pairs = filter_edges(view_list, view_pairs, flow_interval)

    # motion_masks = load_masks(osp.join(src, "epipolar_mask"), view_list, resize)
    if load_sky_flag:
        sky_masks = (
            load_masks(osp.join(src, "segformer_sky_mask"), view_list, resolution=resolution) > 0
        )
    else:
        sky_masks = None
    print(f"Resolution: {resolution}")
    flows = load_flows(osp.join(src, "flows"), view_pairs, resolution=resolution)
    rgbs = load_rgbs(osp.join(src, "images"), view_list, resolution=resolution)

    if feature_config is not None:
        semantic_features = load_semantic_features(osp.join(src, "semantic_features"), feature_config)
    else:
        semantic_features = None

    depths = load_depths(osp.join(src, depth_dir), view_list, resolution=resolution).astype(np.float32)


    epi_err = load_epi_errors(
        osp.join(src, "epipolar_error_npy"), view_list, resolution=resolution
    ).astype(np.float32)

    logging.info(
        f"rgbs: {rgbs.shape}, depths: {depths.shape}, flows: [{len(flows)},{next(iter(flows.values()))['flow'].shape}], epi_err: {epi_err.shape}"
    )
    return (
        view_list,
        view_pairs,
        rgbs,
        semantic_features,
        depths,
        flows,
        sky_masks,
        epi_err,
    )


def load_rgbs(src, fn_list, resolution=None):
    ret = []
    logging.info("Loading rgbs...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
        ])
    for fn in tqdm(fn_list):
        if fn.endswith(".jpg") or fn.endswith(".png"):
            img = cv.imread(osp.join(src, fn))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


            if resolution is not None:
                # resize and crop

                img = transform(img)
                img = img.numpy().transpose(1, 2, 0)
                # scale back
                img = (img * 255).astype(np.uint8)



            ret.append(img)
    ret = np.stack(ret, axis=0)
    return ret  # T,H,W,3

# def load_semantic_features(src, feature_path:str):
#     ret = []
#     logging.info("Loading semantic features...")
#     assert feature_path.endswith(".pth")
#     feature = torch.load(feature_path)
#     return feature  # T,C,H,W


def load_flows(src, view_pairs, resolution=None):
    ret = {}
    logging.info("Loading flows...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
        ])
    for pair in tqdm(view_pairs):
        vi, vj = pair
        ret[(vi, vj)] = dict(np.load(osp.join(src, f"{vi}_to_{vj}.npz")))
        ret[(vj, vi)] = dict(np.load(osp.join(src, f"{vj}_to_{vi}.npz")))
        # if resize != 1.0:
        #     ret[(vi, vj)]["flow"] = (
        #         cv.resize(
        #             ret[(vi, vj)]["flow"],
        #             None,
        #             fx=resize,
        #             fy=resize,
        #             interpolation=cv.INTER_NEAREST,
        #         )
        #         * resize
        #     )
        #     ret[(vj, vi)]["flow"] = (
        #         cv.resize(
        #             ret[(vj, vi)]["flow"],
        #             None,
        #             fx=resize,
        #             fy=resize,
        #             interpolation=cv.INTER_NEAREST,
        #         )
        #         * resize
        #     )
        #     ret[(vi, vj)]["mask"] = (
        #         cv.resize(
        #             ret[(vi, vj)]["mask"].astype(np.uint8),
        #             None,
        #             fx=resize,
        #             fy=resize,
        #             interpolation=cv.INTER_NEAREST,
        #         )
        #         > 0.5
        #     )
        #     ret[(vj, vi)]["mask"] = (
        #         cv.resize(
        #             ret[(vj, vi)]["mask"].astype(np.uint8),
        #             None,
        #             fx=resize,
        #             fy=resize,
        #             interpolation=cv.INTER_NEAREST,
        #         )
        #         > 0.5
        #     )

        if resolution is not None:
            # use transforms to resize and crop

            ret[(vi, vj)]["flow"] = transform(ret[(vi, vj)]["flow"]).numpy().transpose(1, 2, 0)
            ret[(vj, vi)]["flow"] = transform(ret[(vj, vi)]["flow"]).numpy().transpose(1, 2, 0)
            # mask is bool
            ret[(vi, vj)]["mask"] = transform(ret[(vi, vj)]["mask"].astype(np.float32)).numpy().squeeze() > 0.5
            ret[(vj, vi)]["mask"] = transform(ret[(vj, vi)]["mask"].astype(np.float32)).numpy().squeeze() > 0.5


    return ret


def load_depths(src, fn_list, resolution=None):
    logging.info(f"Loading depths from {src} ...")
    ret = []
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
        ])
    for fn in tqdm(fn_list):
        fn = fn.replace("png", "jpg")
        depth = np.load(osp.join(src, fn.replace("jpg", "npz")))["dep"]
        # if resize != 1.0:
        #     depth = cv.resize(
        #         depth, None, fx=resize, fy=resize, interpolation=cv.INTER_NEAREST
        #     )
        if resolution is not None:
            # use transforms to resize and crop
            depth = transform(depth).numpy().squeeze()
        ret.append(depth)
    ret = np.stack(ret, axis=0)
    return ret  # T,H,W


def load_epi_errors(src, fn_list, resolution=None):
    logging.info(f"Loading epi-error from {src} ...")
    ret = []
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
        ])
    for fn in tqdm(fn_list):
        # fn = fn.replace("png", "jpg")
        error = np.load(osp.join(src, fn + ".npy"))
        # if resize != 1.0:
        #     error = cv.resize(
        #         error, None, fx=resize, fy=resize, interpolation=cv.INTER_NEAREST
        #     )
        if resolution is not None:
            # use transforms to resize and crop
            error = transform(error).numpy().squeeze()

        ret.append(error)
    ret = np.stack(ret, axis=0)
    return ret  # T,H,W


def load_masks(src, fn_list, resolution=None):
    ret = []
    logging.info("Loading motion masks...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(resolution) if resolution is not None else transforms.Lambda(lambda x: x),
        ])
    for fn in tqdm(fn_list):
        if fn.endswith(".jpg") or fn.endswith(".png"):
            mask = cv.imread(osp.join(src, fn), cv.IMREAD_GRAYSCALE)
            # if resize != 1.0:
            #     mask = cv.resize(
            #         mask, None, fx=resize, fy=resize, interpolation=cv.INTER_NEAREST
            #     )
            if resolution is not None:
                # use transforms to resize and crop
                mask = transform(mask).numpy().squeeze()
            ret.append(mask)
    ret = np.stack(ret, axis=0)
    return ret  # T,H,W


def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    # by chat gpt
    # grad_x = cv.Sobel(depth_map, cv.CV_64F, 1, 0, ksize=ksize)
    # grad_y = cv.Sobel(depth_map, cv.CV_64F, 0, 1, ksize=ksize)
    # error = cv.magnitude(grad_x, grad_y)
    error = cv.Laplacian(depth_map, cv.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv.threshold(error, threshold, 255, cv.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error


def erode_masks(mask, ksize_raito=0.01):
    # If ksize_raito is 0 Do onthing, if positive do erode, if negative do dilate
    assert mask.ndim == 3
    N, H, W = mask.shape
    if ksize_raito == 0:
        return mask
    if ksize_raito < 0:
        # dilate
        kernel_size = int(min(H, W) * (-ksize_raito))
        method = cv.dilate
    else:
        kernel_size = int(min(H, W) * ksize_raito)
        method = cv.erode
    if kernel_size <= 1:
        return mask
    erode_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ret = []
    for i in range(N):
        new_mask = method(mask[i].astype(np.uint8), erode_kernel, iterations=1)
        ret.append(new_mask.copy())
    ret = np.stack(ret, axis=0) > 0.5
    return ret


def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5):
    logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    for dep in tqdm(depths):
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        dep_valid_masks.append(mask)
        dep_boundary_errors.append(error)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors


def align_depth_to_zoe(depth, zoe_depth, masks):
    N = depth.shape[0]
    assert depth.shape == zoe_depth.shape
    assert masks.shape == depth.shape
    ret = []
    for i in range(N):
        X = depth[i][masks[i]].reshape(-1, 1)
        y = zoe_depth[i][masks[i]].reshape(-1)
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(X, y)
        k = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        new_depth = k * depth[i].copy() + b
        ret.append(new_depth)
    ret = np.stack(ret, axis=0)
    assert ret.shape == depth.shape
    return ret


def get_homo_coordinate_map(H, W):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(np.linspace(*u_range, W), np.linspace(*v_range, H))
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv


def density_filter_depth(depths):
    # use open3d statistical outlier removal to filter the sparse empty space point
    raise NotImplementedError()
    return


def viz_depth_filtering(save_fn, depth, depth_mask, depth_mask_value):
    logging.info("viz_depth_filtering...")
    N = depth.shape[0]
    frames = []
    for i in tqdm(range(N)):
        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.title("depth")
        cmap = plt.imshow(depth[i])
        plt.colorbar(cmap, shrink=0.5)
        plt.subplot(1, 3, 2)
        plt.title("depth mask")
        cmap = plt.imshow(depth_mask[i])
        plt.colorbar(cmap, shrink=0.5)
        plt.subplot(1, 3, 3)
        plt.title("depth mask value")
        cmap = plt.imshow(depth_mask_value[i])
        plt.colorbar(cmap, shrink=0.5)
        plt.tight_layout()
        npfig = fig2nparray(fig)
        frames.append(npfig)
        plt.close(fig)
    imageio.mimsave(save_fn, frames, fps=10)
    return


@torch.no_grad()
def prop_mask_by_flow(prior2d, mask_list, steps=3, operator=torch.logical_or):
    # TODO: this func can be boosted by removing the inner loop to parallel
    logging.info(f"Propagating mask by flow for {steps} steps with {operator}...")
    new_mask_list = mask_list.clone()
    old_mask_list = mask_list.clone()
    for _ in tqdm(range(steps)):
        for i in range(prior2d.T):
            # * each edge absorb the neighbor mask into self
            if i - 1 >= 0:
                new_mask_list = __prop_one_edge_mask__(
                    i, i - 1, prior2d, old_mask_list, new_mask_list, operator
                )
            if i + 1 < prior2d.T:
                new_mask_list = __prop_one_edge_mask__(
                    i, i + 1, prior2d, old_mask_list, new_mask_list, operator
                )
        old_mask_list = new_mask_list.clone()
    torch.cuda.empty_cache()
    return new_mask_list


def __prop_one_edge_mask__(
    src_i, dst_i, prior2d, old_mask_list, new_mask_list, operator=torch.logical_or
):
    flow_ind = prior2d.get_flow_ind(src_i, dst_i)
    flow_mask = prior2d.flow_masks[flow_ind]
    flow = prior2d.flows[flow_ind]
    src_mask = old_mask_list[src_i][flow_mask]
    dst_mask = query_image_buffer_by_pix_int_coord(
        old_mask_list[dst_i],
        prior2d.pixel_int_map[flow_mask] + flow[flow_mask],
    )
    updated_mask = operator(src_mask, dst_mask)
    new_mask_list[src_i, flow_mask] = updated_mask
    return new_mask_list


def viz_mask_video(rgbs, masks, fn):
    viz_dyn_masks = []
    if len(rgbs) > 100:
        step = max(1, len(rgbs) // 50)
    else:
        step = 1
    for t in range(0, len(rgbs), step):
        _viz = rgbs[t]
        _viz = (
            _viz * masks[t][..., None].float()
            + 0.2 * (1 - masks[t][..., None].float()) * _viz
        )
        _viz = (_viz.detach().cpu().numpy() * 255).astype(np.uint8)
        viz_dyn_masks.append(_viz)
    imageio.mimsave(fn, viz_dyn_masks)
    return

def estimate_normal_map(
    vtx_map,
    mask,
    normal_map_patchsize=7,
    normal_map_nn_dist_th=0.03,
    normal_map_min_nn=6,
):
    # * this func is borrowed from my pytorch4D repo in 2022 May.
    # the normal neighborhood is estimated
    # the normal computation refer to pytorch3d 0.6.1, but updated with linalg operations in newer pytroch version

    # note, here the path has zero padding on the border, but due to the computation, sum the zero zero covariance matrix will make no difference, safe!
    H, W = mask.shape
    v_map_patch = F.unfold(
        vtx_map.permute(2, 0, 1).unsqueeze(0),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(3, normal_map_patchsize**2, H, W)

    mask_patch = F.unfold(
        mask.unsqueeze(0).unsqueeze(0).float(),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(1, normal_map_patchsize**2, H, W)

    # Also need to consider the neighbors distance for occlusion boundary
    nn_dist = (vtx_map.permute(2, 0, 1).unsqueeze(1) - v_map_patch).norm(dim=0)
    valid_nn_mask = (nn_dist < normal_map_nn_dist_th)[None, ...]
    v_map_patch[~valid_nn_mask.expand(3, -1, -1, -1)] = 0.0
    mask_patch = mask_patch * valid_nn_mask

    # only meaningful when there are at least 3 valid pixels in the neighborhood, the pixels with less nn need to be exclude when computing the final output normal map, but the mask_patch shouldn't be updated because they still can be used to compute normals for other pixels
    neighbor_counts = mask_patch.sum(dim=1).squeeze(0)  # H,W
    valid_mask = torch.logical_and(mask, neighbor_counts >= normal_map_min_nn)

    v_map_patch = v_map_patch.permute(2, 3, 1, 0)  # H,W,Patch,3
    mask_patch = mask_patch.permute(2, 3, 1, 0)  # H,W,Patch,1
    vtx_patch = v_map_patch[valid_mask]  # M,Patch,3
    neighbor_counts = neighbor_counts[valid_mask]
    mask_patch = mask_patch[valid_mask]  # M,Patch,1

    # compute the curvature normal for the neighbor hood
    # fix several bug here: 1.) the centroid mean bug 2.) the local coordinate should be mask to zero for cov mat
    assert neighbor_counts.min() > 0
    centroid = vtx_patch.sum(dim=1, keepdim=True) / (neighbor_counts[:, None, None])
    vtx_patch = vtx_patch - centroid
    vtx_patch = vtx_patch * mask_patch
    # vtx_patch = vtx_patch.double()
    W = torch.matmul(vtx_patch.unsqueeze(-1), vtx_patch.unsqueeze(-2))
    # todo: here can use distance/confidence to weight the contribution
    W = W.sum(dim=1)  # M,3,3

    # # # one way to solve normal
    # curvature = torch.linalg.eigvalsh(W)
    # c_normal = curvature[:, :1]
    # I = torch.eye(3).to(W.device)[None, ...].expand_as(W)
    # A = W - I * c_normal[..., None]
    # _, _, _Vh = torch.linalg.svd(A)
    # normal = _Vh[:, 2, :]  # M,3

    curvature, local_frame = torch.linalg.eigh(W)
    normal = local_frame[..., 0]

    # # rotate normal towards the camera and filter out some surfels (75deg according to the surfelwarp paper)
    # ray_dir = Homo_cali_unilen[valid_mask]  # M,3
    # inner = (ray_dir * normal).sum(-1)  # the towards cam dir should have cos < 0.0
    # sign_multiplier = -torch.sign(inner)
    # oriented_normal = normal * sign_multiplier[:, None]
    # # debug
    # debug_inner = (ray_dir * oriented_normal).sum(-1)

    # ! warning, when selecting the grazing surfels, here we consider the angel to the principle axis, not the rays
    # inner = oriented_normal[..., 2]  # the z component
    # filter out the surfels whose normal are too tangent to the **ray dir**? or the principle axis? Now use the ray dir
    # valid_normal_mask = inner <= self.normal_valid_cos_th
    # valid_mask[valid_mask.clone()] = valid_normal_mask

    normal_map = torch.zeros_like(vtx_map)
    normal_map[valid_mask] = normal

    return normal_map, valid_mask


def get_dynamic_track_mask(
    d_track,
    d_track_mask,
    H,
    W,
    filter_radius=5,
    filter_cnt=3,
    dilate_radius=3,
):

    # kernel = torch.ones(3, 3).float().to(Working_device)
    filter_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (filter_radius, filter_radius)
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_radius, dilate_radius)
    )
    filter_kernel = torch.from_numpy(filter_kernel).float().to(d_track.device)
    dilate_kernel = torch.from_numpy(dilate_kernel).float().to(d_track.device)

    # mark the d_track on the map
    track_dyn_mask_list = []
    for t in range(len(d_track)):
        buffer = torch.zeros(H, W).bool().to(d_track.device)
        uv = d_track[t][d_track_mask[t]].long()
        uv_mask = (uv[:, 0] >= 0) * (uv[:, 0] < W) * (uv[:, 1] >= 0) * (uv[:, 1] < H)
        uv = uv[uv_mask]
        buffer[uv[:, 1], uv[:, 0]] = True
        # filter out liers
        # count the number of fg points with in a radius, if larger than a th, keep
        buffer_cnt = torch.nn.functional.conv2d(
            buffer[None, None].float(),
            filter_kernel[None, None].float(),
            padding=(filter_radius - 1) // 2,
        ).squeeze()
        filtered_buffer = buffer_cnt > filter_cnt
        # dilate the buffer
        dilated_buffer = (
            torch.nn.functional.conv2d(
                filtered_buffer[None, None].float(),
                dilate_kernel[None, None].float(),
                padding=(dilate_radius - 1) // 2,
            ).squeeze()
            > 0
        )
        track_dyn_mask_list.append(dilated_buffer)
    track_dyn_mask = torch.stack(track_dyn_mask_list, 0)
    return track_dyn_mask
