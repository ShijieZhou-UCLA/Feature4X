# connect local raft
# todo: use larger interval to jump the occlusion, but this may also damage the performance because in this case the tracking may not be reliable
# todo: can use dino+delta from dino-tracker to filter such connections
import torch, numpy as np
from tqdm import tqdm
import logging, sys, os, os.path as osp
import torch
import os, logging, sys, os.path as osp
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import imageio


# messy import ...
from index_helper import (
    query_image_buffer_by_pix_int_coord,
    round_int_coordinates,
    scatter_image_buffer_by_pix_int_coord,
)


@torch.no_grad()
def connect_optical_flow(prior2d, verbose=False):
    logging.info("Connect flow")
    ################################################
    # One thing is to use flows to connect and simplify the representation
    # Which track originally belongs to is only decided by the occ mask, later the validity check will just mask out unknown.
    # TODO: here can use longer interval flow to rescue or to verify!
    ################################################
    # * make raw local raft connections with only flow mask

    trace_pixel_coord = (
        prior2d.pixel_int_map[prior2d.flow_masks[prior2d.get_flow_ind(0, 1)]]
        .reshape(1, -1, 2)
        .cpu()
    )
    trace_valid_mask = torch.ones_like(trace_pixel_coord[..., 0]).bool()

    for tid in tqdm(range(prior2d.T - 1)):
        # ? forward, the fwd, bwd check again??
        next_tid = tid + 1
        flow_ind = prior2d.get_flow_ind(tid, next_tid)
        fwd_flow = prior2d.flows[flow_ind]
        fwd_flow_mask = prior2d.flow_masks[flow_ind]

        ##########################################
        # cases:
        # for all previous active track:
        # 1. land on a valid flow place, continue the track
        # 2. land on a invalid flow place, terminate the track
        # for pixels:
        # 3. if a valid flow place is not landed by any track, start a new track
        ##########################################

        # * First handle the new started track and pad it to track, then use one code to append all
        landed_mask = scatter_image_buffer_by_pix_int_coord(
            torch.zeros_like((fwd_flow_mask)).bool(),
            trace_pixel_coord[-1][trace_valid_mask[-1]],
            1,
        )
        # * case 3, if a valid flow place is not landed by any track, start a new track
        new_track_start_mask = (~landed_mask) & fwd_flow_mask
        if new_track_start_mask.any():
            if verbose:
                logging.info(f"{new_track_start_mask.sum()} new track append at {tid}")
            new_start_pixel_coord = prior2d.pixel_int_map[new_track_start_mask].cpu()
            new_start_pixel_coord = new_start_pixel_coord[None].expand(
                len(trace_pixel_coord), -1, -1
            )
            new_start_valid_mask = torch.zeros_like(
                new_start_pixel_coord[..., 0]
            ).bool()
            new_start_valid_mask[-1] = True
            trace_pixel_coord = torch.cat(
                [trace_pixel_coord, new_start_pixel_coord], dim=1
            )
            trace_valid_mask = torch.cat(
                [trace_valid_mask, new_start_valid_mask], dim=1
            )
            # append all previous track

        # * Second for all active track that as still remaining valid
        active_track_src_pixel_coord = trace_pixel_coord[-1][trace_valid_mask[-1]]
        active_track_new_valid_mask = query_image_buffer_by_pix_int_coord(
            fwd_flow_mask, active_track_src_pixel_coord
        ).cpu()  # * case 2, if land on the invalid flow place, set invalid

        if active_track_new_valid_mask.any():
            if verbose:
                logging.info(
                    f"{active_track_new_valid_mask.sum()} continuing track at {tid}"
                )
                logging.info(
                    f"{(~active_track_new_valid_mask).sum()} terminated forever"
                )
            # * case 1 continue the track and verify whether the new next frame pixel is inside the image
            active_next_pixel_coord = (
                active_track_src_pixel_coord[active_track_new_valid_mask]
                + query_image_buffer_by_pix_int_coord(
                    fwd_flow,
                    active_track_src_pixel_coord[active_track_new_valid_mask],
                ).cpu()
            )
            _, active_next_inside_image_mask = round_int_coordinates(
                active_next_pixel_coord, prior2d.H, prior2d.W
            )
            active_track_new_valid_mask[active_track_new_valid_mask.clone()] = (
                active_next_inside_image_mask
            )

            # append
            padded_active_next_mask = trace_valid_mask[-1].clone()
            padded_active_next_mask[padded_active_next_mask.clone()] = (
                active_track_new_valid_mask.clone()
            )

            padded_active_next_pixel_coord = trace_pixel_coord[-1].clone()
            padded_active_next_pixel_coord[padded_active_next_mask] = (
                active_next_pixel_coord
            )
            trace_pixel_coord = torch.cat(
                [trace_pixel_coord, padded_active_next_pixel_coord[None]], dim=0
            )
            trace_valid_mask = torch.cat(
                [trace_valid_mask, padded_active_next_mask[None]], dim=0
            )
        if verbose:
            logging.info(
                f"[{tid}] total track buffer: {trace_valid_mask.shape[1]} ({int(np.sqrt(trace_valid_mask.shape[1]))}Res) and valid ratio={trace_valid_mask.float().mean()*100:.3f}% active"
            )
        # * once a curve is invalid, it lost track so it lost for ever

        # ! what if two point are mapped together?? detect such case, only keep one track of them (the longer one)!!

    # viz the track

    torch.cuda.empty_cache()
    return trace_pixel_coord, trace_valid_mask


@torch.no_grad()
def viz_track_coverage(H, W, trace_pixel_coord, trace_valid_mask, fn):
    viz_list = []
    for t in range(len(trace_pixel_coord)):
        buffer = torch.zeros(H, W)
        flat_ind = trace_pixel_coord[t][trace_valid_mask[t]]
        flat_ind = flat_ind[:, 1] * W + flat_ind[:, 0]
        buffer = buffer.reshape(-1)
        buffer[flat_ind.cpu()] = 1
        viz_list.append(buffer.reshape(H, W))
    imageio.mimsave(fn, viz_list)
    return


@torch.no_grad()
def gather_curve_attr_large_amount(
    prior2d, track, track_base_mask, return_features=True
):
    # * Designed to work with large amount of tracks
    # to collect:
    # max-epi error;
    # depth buffer; depth mask
    # feature mean and variance (append 3dim color as well)
    T, N = track_base_mask.shape
    w_device = prior2d.working_device

    track, _track_int_mask = round_int_coordinates(track, prior2d.H, prior2d.W)
    track_base_mask = track_base_mask * _track_int_mask

    max_epi = torch.zeros_like(track_base_mask[0]).float().to(w_device)
    depth_buffer = -torch.ones_like(track[..., 0], dtype=torch.float32).to(w_device)
    depth_mask_buffer = torch.zeros_like(track_base_mask).to(w_device)

    if return_features:
        # dino_featmaps = prior2d.low_res_dino_featmaps
        visited_tracks = torch.zeros_like(track_base_mask[0]).to(w_device)
        feat_mean = torch.zeros(N, prior2d.SemCh + 3).to(w_device)
        feat_var = torch.zeros(N, prior2d.SemCh + 3).to(w_device)
        feat_running_cnt = torch.zeros(N).int().to(w_device).to(w_device)

    for tid in tqdm(range(prior2d.T)):
        _mask = track_base_mask[tid].to(w_device)
        _uv_int = track[tid].to(w_device)[_mask]
        # * collect max-epi error
        _epi = query_image_buffer_by_pix_int_coord(prior2d.epi_errs[tid], _uv_int)
        max_epi[_mask] = torch.max(max_epi[_mask].to(w_device), _epi.to(w_device))
        # * collect depth buffer
        _dep = query_image_buffer_by_pix_int_coord(prior2d.depths[tid], _uv_int)
        depth_buffer[tid][_mask] = _dep
        # * collect depth mask
        _dep_mask = query_image_buffer_by_pix_int_coord(
            prior2d.depth_masks[tid], _uv_int
        )
        depth_mask_buffer[tid][_mask] = _dep_mask
        # ! the below mean and variance only count for original mask, don't count above depth mask

        if not return_features:
            continue
        # * collect feature and rgb
        _dino_feat = prior2d.query_low_res_semantic_feat(_uv_int, tid)
        # _grid = _uv_int.float()[None, None].to(w_device)
        # _grid[..., 0] = (_grid[..., 0] / prior2d.W) * 2 - 1
        # _grid[..., 1] = (_grid[..., 1] / prior2d.H) * 2 - 1
        # _dino_feat = F.grid_sample(
        #     dino_featmaps[tid : tid + 1].to(w_device),
        #     grid=_grid,
        #     mode="bilinear",
        #     align_corners=False,
        # )[0, :, 0].permute(1, 0)
        _rgb = query_image_buffer_by_pix_int_coord(prior2d.rgbs[tid], _uv_int)
        _feat = torch.cat([_rgb.to(w_device), _dino_feat], dim=1)
        # * compute running mean and variance for the masked tracks

        start_mask = ~visited_tracks * _mask  # .to(device)
        append_mask = visited_tracks * _mask  # .to(device)
        # _mask = _mask.to(device)
        if start_mask.any():
            feat_mean[start_mask] = _feat[start_mask[_mask]]
            feat_running_cnt[start_mask] = 1
            visited_tracks[start_mask] = True
            # var don't need to be init, because already zero
        if append_mask.any():
            prev_feat_mean = feat_mean[append_mask].clone().to(w_device)
            feat_mean[append_mask] = feat_mean[append_mask].to(w_device) + (
                _feat[append_mask[_mask]] - prev_feat_mean
            ) / (feat_running_cnt[append_mask][..., None].to(w_device) + 1)

            feat_var[append_mask] = feat_var[append_mask].to(w_device) + (
                _feat[append_mask[_mask]] - prev_feat_mean
            ) * (_feat[append_mask[_mask]] - feat_mean[append_mask].to(w_device)) / (
                feat_running_cnt[append_mask][..., None].to(w_device) + 1
            )
            feat_running_cnt[append_mask] = feat_running_cnt[append_mask] + 1
        torch.cuda.empty_cache()

    max_epi = max_epi.cpu()
    depth_buffer = depth_buffer.cpu()
    depth_mask_buffer = depth_mask_buffer.cpu()
    if return_features:
        assert (track_base_mask.to(w_device).sum(0) == feat_running_cnt).all()
        feat_mean = feat_mean.cpu()
        feat_var = feat_var.cpu()
    else:
        feat_mean = None
        feat_var = None
    torch.cuda.empty_cache()
    return max_epi, depth_buffer, depth_mask_buffer, feat_mean, feat_var
