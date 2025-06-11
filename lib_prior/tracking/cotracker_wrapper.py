import torch
import os, sys
import os.path as osp
import imageio
from tqdm import tqdm
import numpy as np

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer, read_video_from_path
import logging, time
from torchvision.transforms import GaussianBlur
import random


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cotracker(device, online_flag=True):
    if online_flag:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker:cotracker2v1_release", "cotracker2_online"
        ).to(device) ###
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker:cotracker2v1_release", "cotracker2").to(
            device
        ) ###
    cotracker.eval()
    return cotracker


def load_data(src):
    image_src = osp.join(src, "images")
    img_fns = os.listdir(image_src)
    img_fns.sort()
    img_frames = []
    for fn in tqdm(img_fns):
        img_frames.append(imageio.imread(osp.join(image_src, fn)))
    img_frames = np.stack(img_frames, 0)
    video_pt = torch.tensor(img_frames).permute(0, 3, 1, 2)[None].float()
    return video_pt  # B T C H W


@torch.no_grad()
def __online_inference_one_pass__(
    video_pt_cpu, queries_cpu, model, device, add_support_grid=True
):
    T = video_pt_cpu.shape[1]
    first_flag = True
    queries = queries_cpu.to(device)
    for i in tqdm(range(T)):
        if i % model.step == 0 and i > 0:
            video_chunk = video_pt_cpu[:, max(0, i - model.step * 2) : i].to(device)
            pred_tracks, pred_visibility = model(
                video_chunk,
                is_first_step=first_flag,
                queries=queries,
                add_support_grid=add_support_grid,
            )
            first_flag = False
    pred_tracks, pred_visibility = model(
        video_pt_cpu[:, -(i % model.step) - model.step - 1 :].to(device),
        False,
        queries=queries,
        add_support_grid=add_support_grid,
    )
    torch.cuda.empty_cache()
    return pred_tracks.cpu(), pred_visibility.cpu()


@torch.no_grad()
def online_track_point(video_pt, queries, model, device, add_support_grid=True):
    T = video_pt.shape[1]
    N = queries.shape[1]
    # * forward
    pred_tracks_fwd, pred_visibility_fwd = __online_inference_one_pass__(
        video_pt, queries, model, device, add_support_grid
    )
    pred_tracks_fwd = pred_tracks_fwd[0, :, :N]  # T,N,2
    pred_visibility_fwd = pred_visibility_fwd[0, :, :N]  # T,N
    # * inverse manually
    video_pt_inv = video_pt.flip(1)
    queries_inv = queries.clone()
    queries_inv[..., 0] = T - 1 - queries_inv[..., 0]
    pred_tracks_bwd, pred_visibility_bwd = __online_inference_one_pass__(
        video_pt_inv, queries_inv, model, device, add_support_grid
    )
    pred_tracks_bwd = pred_tracks_bwd.flip(1)[0, :, :N]  # T,N,2
    pred_visibility_bwd = pred_visibility_bwd.flip(1)[0, :, :N]  # T,N
    # * fuse the forward and backward
    bwd_mask = torch.arange(T)[:, None] < queries[0, :, 0][None, :]  # T,N
    pred_tracks = torch.where(bwd_mask[..., None], pred_tracks_bwd, pred_tracks_fwd)
    pred_visibility = torch.where(bwd_mask, pred_visibility_bwd, pred_visibility_fwd)
    return pred_tracks, pred_visibility


def get_completeness_queries(
    tracks_cpu,
    track_visibility_cpu,
    video_pt_cpu,
    n_pts,
    sigma_ratio=0.03,
    viz_fn=None,
):
    # TODO: this function is now somehow slow, need speed up!
    w_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    _, T, _, H, W = video_pt_cpu.shape
    mask_buffer = torch.zeros(H, W).to(w_device)

    sigma = int(min(H, W) * sigma_ratio)
    ksize = int(4 * sigma + 1)
    blur_layer = GaussianBlur(kernel_size=ksize, sigma=sigma)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u, v], -1)  # H,W,2
    uv = torch.from_numpy(uv).to(w_device).float().reshape(-1, 2)

    queries, vis_cnt_list, viz_list = [], [], []

    for t in tqdm(range(T)):
        visible_uv = tracks_cpu[t][track_visibility_cpu[t] > 0].to(w_device)
        visible_uv_int = visible_uv.int()
        visible_uv_valid_mask = (
            (visible_uv_int[:, 0] >= 0)
            & (visible_uv_int[:, 0] < W)
            & (visible_uv_int[:, 1] >= 0)
            & (visible_uv_int[:, 1] < H)
        )
        visible_uv_int = visible_uv_int[visible_uv_valid_mask]
        mask_buffer = (mask_buffer * 0.0).reshape(-1)  # H*W
        flat_uv_index = visible_uv_int[:, 1] * W + visible_uv_int[:, 0]
        mask_buffer[flat_uv_index] = 1.0
        tracked_pixel_mask = mask_buffer.reshape(H, W)
        # soft the mask with a kernel
        blurred_tracked_pixel_mask = blur_layer(tracked_pixel_mask[None, None])[0, 0]
        vis_cnt = (blurred_tracked_pixel_mask > 0.0).float().mean()
        vis_cnt_list.append(vis_cnt.item())
        # normalize this
        blurred_tracked_pixel_mask = blurred_tracked_pixel_mask / (
            blurred_tracked_pixel_mask.max() + 1e-6
        )
        sampling_weights = (1.0 - blurred_tracked_pixel_mask) * (
            1.0 - tracked_pixel_mask
        )
        choice = torch.multinomial(
            sampling_weights.reshape(-1), 3 * (n_pts // T), replacement=False
        )
        chosen_uv = uv[choice].cpu()
        chosen_uv = torch.cat([torch.ones(chosen_uv.shape[0], 1) * t, chosen_uv], -1)
        queries.append(chosen_uv)
        # imageio.imsave(
        #     "../../debug/debug.png",
        #     (sampling_weights.cpu().numpy() * 255).astype(np.uint8),
        # )
        if viz_fn is not None:
            viz_list.append(
                (
                    (torch.cat([tracked_pixel_mask, sampling_weights], 1) * 255)
                    .cpu()
                    .numpy()
                ).astype(np.uint8)
            )
    if viz_fn is not None:
        imageio.mimsave(viz_fn, viz_list)
    queries = torch.stack(queries, 0)
    weight = (
        (1.0 - torch.tensor(vis_cnt_list)).float()[:, None].expand(-1, queries.shape[1])
    )
    choice = torch.multinomial(weight.reshape(-1), n_pts, replacement=False)
    queries = queries.reshape(-1, 3)[choice]
    return queries.cpu()[None]  # 1,N,3


def __get_uv__(mask, fid, num=1024):
    mask = mask.float()
    H, W = mask.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u, v], -1)  # H,W,2
    dyn_uv = uv[mask > 0]
    weight = mask[mask > 0]
    if num < dyn_uv.shape[0] and num > 0:  # set num to -1 if use all
        choice = torch.multinomial(weight, num, replacement=False)
        # choice = np.random.choice(dyn_uv.shape[0], num, replace=False)
        dyn_uv = dyn_uv[choice]
    queries = torch.tensor(dyn_uv).float()  # N,2
    queries = torch.cat([torch.ones(queries.shape[0], 1) * fid, queries], -1)  # N,3
    return queries


def get_sampling_mask(fg_mask, coverage_mask, sigma_ratio=0.02, scale=10.0):
    # sigma_ratio=0.02
    # scale = 10.0
    T, H, W = fg_mask.shape
    coverage_mask = coverage_mask.to(fg_mask)
    sigma = int(min(H, W) * sigma_ratio)
    ksize = int(4 * sigma + 1)
    blur_layer = GaussianBlur(kernel_size=ksize, sigma=sigma)
    # blur the coverage mask
    weight = blur_layer(coverage_mask.float().reshape(T, 1, H, W)).squeeze(1)
    weight = torch.clamp(weight * scale, 0, 1)
    weight = (1 - weight) * fg_mask
    # viz = [(f*255).cpu().numpy().astype(np.uint8) for f in weight]
    # imageio.mimsave("./debug/w.mp4", viz)
    return weight


def get_uniform_random_queries(
    video_pt, n_pts, t_max=-1, mask_list=None, interval=1, shift=0
):
    queries = []
    T = video_pt.shape[1]
    if t_max > 0:
        T = min(t_max, T)
    key_inds = [i for i in range(shift, T) if i % interval == 0]
    if shift == 0 and T - 1 not in key_inds:
        key_inds.append(T - 1)
    if shift == 0 and 0 not in key_inds:
        key_inds = [0] + key_inds

    T = len(key_inds)

    if mask_list is not None:
        mask_list = mask_list[key_inds]
        _count = (mask_list.reshape(T, -1) > 0).sum(-1)
        mask_weight = _count / _count.sum()
    for i, t in enumerate(key_inds):
        if mask_list is None:
            mask = torch.ones_like(video_pt[0, t, 0])
            target_num = n_pts / T
        else:
            mask = mask_list[i]
            target_num = n_pts * mask_weight[i]
        q = __get_uv__(mask, fid=t, num=int(target_num) * 3)
        queries.append(q)
    queries = torch.cat(queries, 0)
    choice = torch.randperm(queries.shape[0])[:n_pts]
    queries = queries[None, choice]
    return queries  # 1,N,3


def load_epi_error(save_dir):
    fns = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    fns.sort()
    epi_error = []
    for fn in fns:
        epi_error.append(np.load(osp.join(save_dir, fn)))
    epi_error = np.stack(epi_error, 0)
    epi_error = torch.tensor(epi_error).float()
    return epi_error


def get_epi_driven_queries(epi_error, n_pts, t_max=-1):
    w_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    T, H, W = epi_error.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u, v], -1)  # H,W,2
    uv = torch.from_numpy(uv).to(w_device).float().reshape(-1, 2)

    queries, epi = [], []
    if t_max > 0:
        T = min(t_max, T)

    queries, epi = [], []
    for i in range(T):
        w = epi_error[i].to(w_device).reshape(-1)
        w = w + 1e-4  # make others have a chance
        choice = torch.multinomial(w, 3 * n_pts // T, replacement=False)
        q = uv[choice].cpu()
        q = torch.cat([torch.ones(q.shape[0], 1) * i, q], -1)
        queries.append(q)
        epi.append(w[choice].cpu())
    queries = torch.cat(queries, 0)
    epi = torch.cat(epi, 0)
    choice = torch.multinomial(epi.to(w_device), n_pts, replacement=False)
    queries = queries[choice.cpu()]
    return queries[None]


@torch.no_grad()
def process_folder_track(
    src,
    model,
    device,
    dyn_total_n_pts,
    sta_total_n_pts,
    dyn_chunk_n_pts=5000,
    sta_chunk_n_pts=5000,
    dyn_epi_th=200.0,
    # debug
    online_flag=True,
    first_T=-1,
    keyframe_interval=8,
    dyn_subsample=1,  # only at these steps will infer co-tracker, to enlarge the tine fov
    max_round=3,
):
    viz_dir = osp.join(src, "cotracker_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = src
    os.makedirs(save_dir, exist_ok=True)
    full_video_pt = load_data(src).cpu()  # [B,T,3,H,W] save on cpu
    if first_T > 0 and first_T < full_video_pt.shape[1]:
        full_video_pt = full_video_pt[:, :first_T]
    _, full_T, _, H, W = full_video_pt.shape
    logging.info(f"T=[{full_T}], video shape: {full_video_pt.shape}")
    vis = Visualizer(
        save_dir=viz_dir,
        linewidth=2,
        # draw_invisible=False,  # ! debug
        draw_invisible=True,  # ! debug
        tracks_leave_trace=10,
    )

    epi_error_cpu = load_epi_error(osp.join(src, "epipolar_error_npy"))
    if first_T > 0 and first_T < epi_error_cpu.shape[0]:
        epi_error_cpu = epi_error_cpu[:first_T]

    # make the epi mask
    epi_th = (H * W) / (dyn_epi_th**2)
    logging.info(f"epi_th={epi_th:.6f}")
    full_fg_mask = epi_error_cpu > epi_th
    # viz the fg mask
    imageio.mimsave(
        osp.join(viz_dir, f"fg_mask_epi_th={dyn_epi_th}.mp4"),
        (full_fg_mask[..., None] * full_video_pt[0].permute(0, 2, 3, 1))
        .cpu()
        .numpy()
        .astype(np.uint8),
    )

    for i in range(2):
        if i == 0:
            # dyn cfg
            phase = "dyn"
            sub_t_list = [t for t in range(full_T) if t % dyn_subsample == 0]
            if full_T - 1 not in sub_t_list:
                sub_t_list.append(full_T - 1)

            sub_t_list = torch.tensor(sub_t_list).long()
            video_pt = full_video_pt[:, sub_t_list].clone()
            masks = full_fg_mask[sub_t_list].clone()
            T = len(sub_t_list)

            total_n_pts = dyn_total_n_pts
            chunk_n_pts = dyn_chunk_n_pts
            mode_keyframe_interval = keyframe_interval // dyn_subsample
        else:
            # sta cfg
            phase = "sta"

            sub_t_list = torch.arange(full_T).long()
            video_pt = full_video_pt.clone()
            masks = ~full_fg_mask.clone()
            T = full_T

            total_n_pts = sta_total_n_pts
            chunk_n_pts = sta_chunk_n_pts
            mode_keyframe_interval = keyframe_interval
        logging.info(f"Start {phase} tracking...")

        tracks = torch.zeros(T, 0, 2)
        visibility = torch.zeros(T, 0).bool()

        start_t = time.time()
        round_cnt = 0
        while tracks.shape[1] < total_n_pts:
            if round_cnt >= max_round:
                break
            sampling_masks = masks.clone()
            if round_cnt > 0:  # use coverage mask to update the sampling mask
                cover_list = torch.stack([torch.tensor(it) for it in coverage_list], 0)
                sampling_masks = get_sampling_mask(
                    masks, cover_list, sigma_ratio=0.02, scale=10.0
                )
                viz_sampling_masks = [
                    (f * 255).cpu().numpy().astype(np.uint8) for f in sampling_masks
                ]
                imageio.mimsave(
                    osp.join(viz_dir, f"{phase}_sampling_masks_{round_cnt}.mp4"),
                    viz_sampling_masks,
                )
                print()
            queries = get_uniform_random_queries(
                video_pt,
                min(chunk_n_pts, total_n_pts - tracks.shape[1]),
                mask_list=sampling_masks,
                interval=mode_keyframe_interval,
                # shift=round_cnt, # no shift
            )
            viz_list = viz_queries(queries.squeeze(0), H, W, T)
            imageio.mimsave(
                osp.join(viz_dir, f"{phase}_quries_{round_cnt}.mp4"), viz_list
            )

            if online_flag:
                _tracks, _visibility = online_track_point(
                    video_pt, queries, model, device
                )  # T,N,2; T,N
            else:
                _tracks, _visibility = model(
                    video_pt.to(device), queries.to(device), backward_tracking=True
                )
                _tracks, _visibility = _tracks[0].cpu(), _visibility[0].cpu()
            if tracks is None:
                tracks = _tracks
                visibility = _visibility
            else:
                tracks = torch.cat([tracks, _tracks], 1)
                visibility = torch.cat([visibility, _visibility], 1)
            # viz current coverage
            coverage_list = viz_coverage(tracks, visibility, H, W)
            imageio.mimsave(
                osp.join(viz_dir, f"{phase}_coverage_{round_cnt}.mp4"), coverage_list
            )
            round_cnt += 1
        end_t = time.time()
        logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")
        mode = "online" if online_flag else "offline"
        vis.visualize(
            video=video_pt,
            tracks=tracks[None],
            visibility=visibility[None],
            filename=f"cotracker_{phase}_global_{mode}",
        )
        logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

        if phase == "dyn":
            # pad back to full
            pad_track = torch.zeros(full_T, tracks.shape[1], 2)
            pad_track[sub_t_list] = tracks
            pad_visibility = torch.zeros(full_T, tracks.shape[1]).bool()
            pad_visibility[sub_t_list] = visibility
            tracks = pad_track
            visibility = pad_visibility

        np.savez_compressed(
            osp.join(save_dir, f"cotracker_{phase}_global_{mode}.npz"),
            queries=queries.numpy(),  # useless
            tracks=tracks.numpy(),
            visibility=visibility.numpy(),
            sub_t_list=sub_t_list.numpy(),
        )
    return


def viz_queries(queries, H, W, T):
    ret = []
    uv, t = queries[:, 1:], queries[:, 0]
    for _t in range(T):
        mask = _t == t
        _uv = uv[mask].int()
        buffer = torch.zeros(H * W)
        buffer[_uv[:, 1] * W + _uv[:, 0]] = 1
        ret.append((buffer.reshape(H, W).float().numpy() * 255).astype(np.uint8))
    return ret


def viz_coverage(track, track_mask, H, W):
    ret = []
    for t in range(track.shape[0]):
        mask = torch.zeros(H * W)
        uv_int = track[t].int()
        valid_mask = (
            (uv_int[:, 0] < W)
            * (uv_int[:, 0] >= 0)
            * (uv_int[:, 1] < H)
            * (uv_int[:, 1] >= 0)
        )
        vis_mask = track_mask[t][valid_mask]
        uv_int = uv_int[valid_mask]
        flat_ind = uv_int[:, 1] * W + uv_int[:, 0]
        mask[flat_ind[vis_mask]] = 1
        ret.append((mask.reshape(H, W).float().numpy() * 255).astype(np.uint8))
    return ret


# if __name__ == "__main__":
#     # conclusions:
#     # * online and offline are the same!!!
#     # * use full FPS and later can subsample from prior2D!!

#     # ! maybe first try dot
#     # ! the raft may guide the cotracker when the noddle is connected!!!! RAFT guided Co-Tracker

#     seed_everything(12345)

#     online_flag = True

#     # src = "../../data/iphone_5x_2/spin"
#     src = "../../data/iphone/spin"
#     model = get_cotracker("cuda", online_flag=online_flag)
#     process_folder_track(
#         src,
#         model,
#         "cuda",
#         dyn_total_n_pts=8192,
#         chunk_n_pts=8192,
#         # total_n_pts=12000,
#         # chunk_n_pts=12000,
#         # total_n_pts=1024,
#         # chunk_n_pts=1024,
#         # epi_ratio=1.0,
#         # first_T=150,
#         online_flag=online_flag,
#     )
#     print()
