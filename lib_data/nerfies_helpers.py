# deal with nerfies/hypernerf dataset
import json
import os, os.path as osp
import numpy as np
import torch
import logging
from tqdm import tqdm


def load_nerfies_cameras(dir_path):
    ######################################
    # ! dycheck use Nerfies convention, which is opencv format
    # https://github.com/google/nerfies#datasets
    # https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md
    ######################################

    all_fns = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    all_fns.sort()
    all_cam_ids = np.array([int(f.split(".")[0].split("_")[1]) for f in all_fns])
    all_cam_ids = np.unique(all_cam_ids).tolist()
    logging.info(f"Loading nerfies cameras from {dir_path} with camid={all_cam_ids}")
    # ret = {}
    # for cam_id in all_cam_ids:
    # cam_fns = [f for f in all_fns]
    # cam_fns.sort()
    T_wi_list, fovdeg_list, t_list = [], [], []
    cxcy_ratio_list = []
    ret={}
    # ! iphone dataset only has one fov
    for fn in tqdm(all_fns):
        time_ind = int(fn.split("_")[1].split(".")[0])  # ! time start from 0, not 1
        t_list.append(time_ind)

        with open(osp.join(dir_path, fn), "r") as f:
            cam_data = json.load(f)
        focal = cam_data["focal_length"]
        T_wi = np.eye(4)
        t_wi = np.asarray(cam_data["position"])
        R_wi = np.asarray(cam_data["orientation"]).T
        T_wi[:3, :3] = R_wi
        T_wi[:3, 3] = t_wi
        # world = R.T @ local
        # https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/camera.py#L263
        # the optical axis is the last row
        T_wi_list.append(torch.from_numpy(T_wi))

        # ! get the fov in current code format, the focal with the shortest side
        short_size = min(cam_data["image_size"])  # -1, +1
        fovdeg = np.rad2deg(2 * np.arctan(short_size / (2 * focal)))

        # also load camera center
        principal_point = np.asarray(cam_data["principal_point"])
        image_size = np.asarray(cam_data["image_size"])
        cx_ratio = principal_point[0] / image_size[0]
        cy_ratio = principal_point[1] / image_size[1]
        cxcy_ratio_list.append([cx_ratio, cy_ratio])

        fovdeg_list.append(fovdeg)
    ret[0] = {
        "T_wi": torch.stack(T_wi_list, 0).float(),
        "fovdeg": fovdeg_list,
        "cxcy_ratio": cxcy_ratio_list,
        "t": t_list,
        "filenames": [f[:-5] for f in all_fns],
    }
    return ret


def load_nerfies_gt_poses(src, t_subsample):
    gt_training_cams = load_nerfies_cameras(osp.join(src, "camera_left1"))
    assert len(list(gt_training_cams.keys())) == 1
    gt_training_all = {k: [vv for cam in gt_training_cams.values() for vv in cam[k]] for k in gt_training_cams[list(gt_training_cams.keys())[0]]} # ! assume there is only one camera after renaming

    sorted_idx = np.argsort(gt_training_all['t'])
    gt_training_all = {k: [v[idx] for idx in sorted_idx] for k, v in gt_training_all.items()}
    gt_training_all['T_wi'] = torch.stack(gt_training_all['T_wi'], 0).float()
    gt_training_cams = gt_training_all

    gt_training_cams = {k: v[::t_subsample] for k, v in gt_training_cams.items()}

    gt_testing_cams = load_nerfies_cameras(osp.join(src, "camera_right1"))
    gt_training_fov = gt_training_cams["fovdeg"][0]
    logging.info(
        f"Nerfies gt training camera fov is {gt_training_cams['fovdeg'][0]:.3f} deg."
    )
    gt_training_cam_T_wi = gt_training_cams["T_wi"]
    # ! this bound which test frames can be retrieved
    gt_training_tids = gt_training_cams["t"]
    gt_training_cxcy_ratio = gt_training_cams["cxcy_ratio"]

    gt_testing_cam_T_wi_list, gt_testing_tids_list = [], []
    gt_testing_fov_list, gt_testing_fns_list = [], []
    gt_testing_cxcy_ratio_list = []
    for it in gt_testing_cams.values():
        sample_index = [i for i, t in enumerate(it["t"]) if t in gt_training_tids]
        print(f'original length {len(it["t"])}, after sampling {len(sample_index)}')
        gt_testing_tids = [gt_training_tids.index(it["t"][i]) for i in sample_index]
        gt_testing_tids_list.append(gt_testing_tids)
        gt_testing_cam_T_wi = it["T_wi"][sample_index]
        gt_testing_cam_T_wi_list.append(gt_testing_cam_T_wi)
        gt_testing_fns_list.append([it["filenames"][i] for i in sample_index])
        # ! assume all cam stays the same across time, use the first one
        gt_testing_fov_list.append(it["fovdeg"][0])
        gt_testing_cxcy_ratio_list.append(it["cxcy_ratio"][0])

    return (
        gt_training_cam_T_wi,
        gt_training_tids,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    )


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--src_path', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    load_nerfies_gt_poses(args.src_path, 1)