import numpy as np
import os, os.path as osp
import imageio
from tqdm import tqdm
import os
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
import random


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GeoSDDinoWrapper:
    def __init__(self):
        self.num_patches = 60
        seed_everything(12345)

        print("loading agg net")
        self.aggre_net = AggregationNetwork(
            feature_dims=[640, 1280, 1280, 768], projection_dim=768, device="cuda"
        )
        self.aggre_net.load_pretrained_weights(torch.load("results_spair/best_856.PTH"))

        print("loading SD")
        self.sd_model, self.sd_aug = load_model(
            diffusion_ver="v1-5",
            image_size=self.num_patches * 16,
            num_timesteps=50,
            block_indices=[2, 5, 8, 11],
        )

        print("loading dino")
        self.extractor_vit = ViTExtractor("dinov2_vitb14", stride=14, device="cuda")

    def __call__(self, img):
        sd_model, sd_aug, aggre_net = self.sd_model, self.sd_aug, self.aggre_net
        extractor_vit, num_patches = self.extractor_vit, self.num_patches

        img_sd_input = resize(
            img, target_res=num_patches * 16, resize=True, to_pil=True
        )
        features_sd = process_features_and_mask(
            sd_model, sd_aug, img_sd_input, mask=False, raw=True
        )
        del features_sd["s2"]

        # extract dinov2 features
        img_dino_input = resize(
            img, target_res=num_patches * 14, resize=True, to_pil=True
        )
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = (
            extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet="token")
            .permute(0, 1, 3, 2)
            .reshape(1, -1, num_patches, num_patches)
        )

        desc_gathered = torch.cat(
            [
                features_sd["s3"],
                F.interpolate(
                    features_sd["s4"],
                    size=(num_patches, num_patches),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    features_sd["s5"],
                    size=(num_patches, num_patches),
                    mode="bilinear",
                    align_corners=False,
                ),
                features_dino,
            ],
            dim=1,
        )

        desc = aggre_net(desc_gathered)  # 1, 768, 60, 60
        # normalize the descriptors
        norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
        desc = desc / (norms_desc + 1e-8)
        return desc


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


@torch.no_grad()
def get_feature(img_np, model):
    dino_size = 480
    small_size = 60
    img, pad_tblr = pad_resize(
        img_np,
        dino_size,
        resize=True,
        to_pil=True,
        edge_pad=True,
    )
    assert (pad_tblr == 0).sum() >= 2

    with torch.no_grad():
        featmap = model(img)

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


def geo_sd_dino_process_folder(
    model: GeoSDDinoWrapper,
    dst_fn,
    src,
    pca_dim=256,
):
    img_fns = os.listdir(src)
    img_fns.sort()

    raw_featmap_list = []
    for img_fn in tqdm(img_fns):
        img_np = imageio.imread(osp.join(src, img_fn))
        feat_map = get_feature(img_np, model)
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


if __name__ == "__main__":
    import argparse

    model = GeoSDDinoWrapper()

    geo_sd_dino_process_folder(
        model=model,
        dst_fn="/home/leijh/projects/vid24d/data/iphone_1x_dev/apple/dino_lr.npz",
        src="/home/leijh/projects/vid24d/data/iphone_1x_dev/apple/images",
        pca_dim=64,
    )
