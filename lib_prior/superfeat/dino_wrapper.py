import os, sys, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

from dino import ViTExtractor
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import imageio


def pad_resize(img, target_res, resize=True, to_pil=False, edge_pad=True):
    # convert numpy to PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge_pad:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize(
                    (
                        target_res,
                        int(np.around(target_res * original_height / original_width)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2 : (width + height) // 2] = img
        else:
            if resize:
                img = img.resize(
                    (
                        int(np.around(target_res * original_width / original_height)),
                        target_res,
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2 : (height + width) // 2] = img
        pad_tblr = (0, 0, 0, 0)
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize(
                    (
                        target_res,
                        int(np.around(target_res * original_height / original_width)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(
                img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode="edge"
            )
            left_pad, right_pad = 0, 0
        else:
            if resize:
                img = img.resize(
                    (
                        int(np.around(target_res * original_width / original_height)),
                        target_res,
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(
                img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode="edge"
            )
            top_pad, bottom_pad = 0, 0
        pad_tblr = (top_pad, bottom_pad, left_pad, right_pad)
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas, np.array(list(pad_tblr))


def get_dino_model(version, device=torch.device("cuda")):
    # default setting for DINO v2
    model_dict = {
        "small": "dinov2_vits14",
        "base": "dinov2_vitb14",
        "large": "dinov2_vitl14",
        "giant": "dinov2_vitg14",
    }
    model_type = model_dict[version]
    dino_input_img_res = 840
    layer = 11
    if "l" in model_type:
        layer = 23
    elif "g" in model_type:
        layer = 39
    facet = "token"
    stride = 14
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0]
    num_patches = int(patch_size / stride * (dino_input_img_res // patch_size - 1) + 1)
    cfg = {
        "num_patches": num_patches,
        "layer": layer,
        "facet": facet,
        "dino_input_img_res": dino_input_img_res,
    }
    return extractor, cfg


@torch.no_grad()
def get_feature(
    img_np,
    dino: ViTExtractor,
    dino_cfg: dict,
    device=torch.device("cuda"),
):
    dino_size = dino_cfg["dino_input_img_res"]
    small_size = dino_cfg["num_patches"]
    img, pad_tblr = pad_resize(
        img_np,
        dino_size,
        resize=True,
        to_pil=True,
        edge_pad=True,
    )
    assert (pad_tblr == 0).sum() >= 2
    img_batch = dino.preprocess_pil(img)
    
    img_desc_dino = dino.extract_descriptors(
        img_batch.to(device), dino_cfg["layer"], dino_cfg["facet"]
    )
    featmap = img_desc_dino.permute(0, 1, 3, 2).reshape(
        -1, img_desc_dino.shape[-1], small_size, small_size
    )  # B,C,Patch,Patch
    # resize back to dino input res
    dino_size_featmap = F.interpolate(
        featmap, size=(dino_size, dino_size), mode="bilinear", align_corners=False
    )  # B,C,DR,DR
    # remove the padding
    dino_size_featmap = dino_size_featmap[
        :,
        :,
        pad_tblr[0] : dino_size - pad_tblr[1],
        pad_tblr[2] : dino_size - pad_tblr[3],
    ]
    # resize to smaller patch size
    if img_np.shape[0] > img_np.shape[1]:
        target_size = (small_size, int(small_size * img_np.shape[1] / img_np.shape[0]))
    else:
        target_size = (int(small_size * img_np.shape[0] / img_np.shape[1]), small_size)

    ret_featmap = F.interpolate(dino_size_featmap, size=target_size, mode="bilinear")
    ret_featmap = ret_featmap[0].permute(1, 2, 0)  # H',W',C

    return ret_featmap.cpu()


def dino_process_folder(
    dst_fn,
    src,
    dino,
    dino_cfg,
    device=torch.device("cuda"),
    pca_dim=256,
):
    img_fns = os.listdir(src)
    img_fns.sort()

    raw_featmap_list = []
    for img_fn in tqdm(img_fns):
        img_np = imageio.imread(osp.join(src, img_fn))
        feat_map = get_feature(img_np, dino, dino_cfg, device=device)
        # ! normalize the length
        feat_map = feat_map / torch.clamp(feat_map.norm(dim=-1, keepdim=True), min=1e-6)
        raw_featmap_list.append(feat_map.detach())
        torch.cuda.empty_cache()
    raw_featmap = torch.stack(raw_featmap_list, dim=0)  # .to(device)
    # get actual size
    actual_H, actual_W = imageio.imread(osp.join(src, img_fns[0])).shape[:2]

    # ! too large, can only do on cpu
    C = raw_featmap.shape[-1]
    raw_featmap_flat = raw_featmap.reshape(-1, C)
    mean = torch.mean(raw_featmap_flat, dim=0, keepdim=True)
    centered_features = raw_featmap_flat - mean
    U, S, V = torch.pca_lowrank(centered_features, q=pca_dim)
    reduced_features = torch.matmul(centered_features, V[:, :pca_dim])  # (t_x+t_y)x(d)
    reduced_features = reduced_features.reshape(
        *raw_featmap.shape[:-1], pca_dim
    )  # T,H',W',C
    reduced_features = reduced_features.permute(0, 3, 1, 2)  # T,C,H',W'

    # ! save the low res feat!!
    np.savez(
        dst_fn,
        H=actual_H,
        W=actual_W,
        low_res_reduced_featmaps=reduced_features.cpu().numpy().astype(np.float32),
        # low_res_raw_featmaps=raw_featmap.cpu().numpy().astype(np.float32),
        basis=V.detach().cpu().numpy().astype(np.float32),
    )

    # resize back
    viz_vid = F.interpolate(
        reduced_features[:, :3].cpu(),
        size=(actual_H, actual_W),
        mode="bilinear",
        align_corners=False,
    )  # T,C,H,W
    # viz a video
    viz_vid = viz_vid.permute(0, 2, 3, 1).cpu().numpy()
    viz_vid = (viz_vid - viz_vid.min()) / (viz_vid.max() - viz_vid.min()) * 255
    viz_vid = viz_vid.astype(np.uint8)
    imageio.mimsave(dst_fn.replace(".npz", ".mp4"), viz_vid)

    return


########################################################################################
# * FeatUP DINO
########################################################################################


def get_feat_up_model(device=torch.device("cuda")):
    dino_input_img_res = 840
    extractor = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False).to(
        device
    )
    cfg = {
        "dino_input_img_res": dino_input_img_res,
    }
    return extractor, cfg


@torch.no_grad()
def get_feature_up(
    img_np,
    dino,
    dino_cfg: dict,
    device=torch.device("cuda"),
):
    dino_size = dino_cfg["dino_input_img_res"]
    img, pad_tblr = pad_resize(
        img_np,
        dino_size,
        resize=True,
        to_pil=True,
        edge_pad=True,
    )
    assert (pad_tblr == 0).sum() >= 2
    # img_batch = dino.preprocess_pil(img)
    # convert from pil to numpy
    img_batch = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    img_batch = img_batch.float() / 255.0

    img_desc_dino = dino(img_batch.to(device))
    img_desc_dino = F.interpolate(
        img_desc_dino, size=(dino_size, dino_size), mode="nearest"
    )  # B,C,DR,DR
    ret_featmap = img_desc_dino[
        :,
        :,
        pad_tblr[0] : dino_size - pad_tblr[1],
        pad_tblr[2] : dino_size - pad_tblr[3],
    ]
    ret_featmap = ret_featmap[0].permute(1, 2, 0)  # H',W',C
    return ret_featmap.cpu()


def dino_featup_process_folder(
    dst_fn,
    src,
    dino,
    dino_cfg,
    device=torch.device("cuda"),
    pca_dim=256,
):
    img_fns = os.listdir(src)
    img_fns.sort()
    
    # # ! debug
    # img_fns = img_fns[:10]

    raw_featmap_list = []
    for img_fn in tqdm(img_fns):
        img_np = imageio.imread(osp.join(src, img_fn))
        feat_map = get_feature_up(img_np, dino, dino_cfg, device=device)
        # ! normalize the lenght
        feat_map = feat_map / torch.clamp(feat_map.norm(dim=-1, keepdim=True), min=1e-6)
        raw_featmap_list.append(feat_map.detach())
        torch.cuda.empty_cache()
    raw_featmap = torch.stack(raw_featmap_list, dim=0)  # .to(device)
    # get actual size
    actual_H, actual_W = imageio.imread(osp.join(src, img_fns[0])).shape[:2]

    # ! too large, can only do on cpu
    C = raw_featmap.shape[-1]
    raw_featmap_flat = raw_featmap.reshape(-1, C)
    mean = torch.mean(raw_featmap_flat, dim=0, keepdim=True)
    centered_features = raw_featmap_flat - mean
    U, S, V = torch.pca_lowrank(centered_features, q=pca_dim)
    reduced_features = torch.matmul(centered_features, V[:, :pca_dim])  # (t_x+t_y)x(d)
    reduced_features = reduced_features.reshape(
        *raw_featmap.shape[:-1], pca_dim
    )  # T,H',W',C
    reduced_features = reduced_features.permute(0, 3, 1, 2)  # T,C,H',W'

    # ! save the low res feat!!
    np.savez(
        dst_fn,
        H=actual_H,
        W=actual_W,
        low_res_reduced_featmaps=reduced_features.cpu().numpy().astype(np.float32),
        # low_res_raw_featmaps=raw_featmap.cpu().numpy().astype(np.float32),
        basis=V.detach().cpu().numpy().astype(np.float32),
    )

    # resize back
    viz_vid = F.interpolate(
        reduced_features[:, :3].cpu(),
        size=(actual_H, actual_W),
        mode="bilinear",
        align_corners=False,
    )  # T,C,H,W
    # viz a video
    viz_vid = viz_vid.permute(0, 2, 3, 1).cpu().numpy()
    viz_vid = (viz_vid - viz_vid.min()) / (viz_vid.max() - viz_vid.min()) * 255
    viz_vid = viz_vid.astype(np.uint8)
    imageio.mimsave(dst_fn.replace(".npz", ".mp4"), viz_vid)

    return


if __name__ == "__main__":
    # fn = "../../data/davis_v4.1/breakdance-flare/images/00000.jpg"
    src = "../../data/davis_v4.1/breakdance-flare/images"
    dst = "../../data/davis_v4.1/breakdance-flare/dino.npz"
    device = torch.device("cuda")

    # resize back to select the padded area
    # img_np = imageio.imread(fn)

    dino, dino_cfg = get_dino_model("giant", device=device)
    dino_process_folder(dst, src, dino, dino_cfg, device=device)
