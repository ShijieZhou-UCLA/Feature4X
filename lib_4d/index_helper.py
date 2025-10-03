import torch
import numpy as np


@torch.no_grad()
def get_valid_flow_pixel_int_index(prior2d, i_ind, j_ind, mask_key="sky_dep_sta"):
    flow_ind = prior2d.get_flow_ind(i_ind, j_ind)
    flow = prior2d.flows[flow_ind]
    flow_mask = prior2d.flow_masks[flow_ind]
    i_ind_map = prior2d.pixel_int_map[flow_mask]
    j_ind_map = i_ind_map + flow[flow_mask]
    i_valid = query_image_buffer_by_pix_int_coord(
        prior2d.get_mask_by_key(mask_key, i_ind), i_ind_map
    )
    j_valid = query_image_buffer_by_pix_int_coord(
        prior2d.get_mask_by_key(mask_key, j_ind), j_ind_map
    )
    valid_mask = i_valid & j_valid
    valid_i_ind = i_ind_map[valid_mask]
    valid_j_ind = j_ind_map[valid_mask]
    return valid_i_ind, valid_j_ind

def scatter_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate, value):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    buffer = buffer.reshape(H * W, *buffer.shape[2:])
    buffer[index] = value
    return buffer.reshape(H, W, *buffer.shape[1:])


def query_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret


def round_int_coordinates(coord, H, W):
    ret = coord.float().round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask


def uv_to_pix_int_coordinates(uv, H, W, clamp=True, round=True):
    # short side is [-1,1]
    L = min(H, W)
    u = uv[..., 0] * L / 2 + W / 2
    v = uv[..., 1] * L / 2 + H / 2
    ret = torch.stack([u, v], dim=-1)
    if round:
        ret = ret.round().long()
    if clamp:
        valid_mask = (
            (ret[..., 0] >= 0)
            & (ret[..., 0] < W)
            & (ret[..., 1] >= 0)
            & (ret[..., 1] < H)
        )
        ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
        ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
        return ret, valid_mask
    else:
        return ret

def pix_to_rel_uv_coordinates(pixel_uv, H, W):
    # the short side has (-1,+1)
    assert pixel_uv.shape[-1] == 2
    u, v = pixel_uv[...,0], pixel_uv[...,1]
    L = min(H, W)
    new_u = (new_u - W/2.0) / (L/2.0)
    new_v = (new_v - H/2.0) / (L/2.0)
    ret = torch.stack([new_u, new_v], -1)
    return ret


def get_subsample_mask_like(buffer, sub_sample):
    # buffer is H,W,...
    assert buffer.ndim >= 2
    ret = torch.zeros_like(buffer).bool()
    ret[::sub_sample, ::sub_sample, ...] = True
    return ret
