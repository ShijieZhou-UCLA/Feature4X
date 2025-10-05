import os
import sys
os.chdir(os.path.dirname(__file__))
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from modeling_videochat2 import InternVideo2_VideoChat2
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import cv2 as cv
import os.path as osp
import torch
import imageio
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments, start=None):
    seg_size = float(num_frames - 1) / num_segments
    if start is None:
        start = int(seg_size / 2)

    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def get_indices_split(num_frames, num_segments):
    # results = set()
    seg_size = num_frames // num_segments
    offsets_first = np.array([int(seg_size * i) for i in range(num_segments)])
    offsets = [i + offsets_first for i in range(seg_size)]
    if offsets[-1][-1] < num_frames-1:
        new_offset_reverse = [num_frames-1-seg_size*i for i in range(num_segments)]
        offsets.append(np.array(new_offset_reverse[::-1]))

    return offsets

def get_view_list(src, nerfies_flag=False):
    ret = [
        f
        for f in os.listdir(osp.join(src, "images"))
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    if nerfies_flag:
        tid = np.array([int(f.split(".")[0].split("_")[1]) for f in ret])
        tid = np.argsort(tid)
        ret = [ret[i] for i in tid]
    else:
        ret.sort()
    return ret

def load_rgbs(src, fn_list, resize):
    ret = []
    print("Loading rgbs...")
    for fn in tqdm(fn_list):
        if fn.endswith(".jpg") or fn.endswith(".png"):
            img = cv.imread(osp.join(src, fn))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if resize != 1.0:
                img = cv.resize(
                    img, None, fx=resize, fy=resize, interpolation=cv.INTER_LINEAR
                )
            img = torch.tensor(img).float()
            ret.append(img)
    ret = torch.stack(ret, axis=0)
    return ret  # T,H,W,3

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    if os.path.isdir(video_path):
        view_list = get_view_list(video_path)
        vr = load_rgbs(osp.join(video_path, "images"), view_list, 1.0)
        # save as mp4 video file, using imageio
        imageio.mimwrite(osp.join(video_path,'video.mp4') , vr.numpy().astype(np.uint8),macro_block_size=1)
        # video_path = 'video.mp4'
        # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    # frame_indices = get_index(num_frames, num_segments, start=0)
    frame_indices_list = get_indices_split(num_frames, num_segments)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(resolution),
        transforms.Normalize(mean, std)
    ])
    print(f"Warning: now not do crop, resize only.")
    frames_list=[]
    for frame_indices in frame_indices_list:
        if isinstance(vr, torch.Tensor):
            frames = vr[frame_indices]
        else:
            frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)
        frames_list.append((frames, frame_indices))

    T_, C, H, W = frames_list[0][0].shape

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames_list, msg, num_frames
    else:
        return frames_list, num_frames

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from video')
    parser.add_argument('--video_path', type=str, default="yoga.mp4", help='path to the video')
    return parser.parse_args()

if __name__ == "__main__":
    model = InternVideo2_VideoChat2.from_pretrained("OpenGVLab/InternVideo2-Chat-8B",).cuda()
    tokenizer =  AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2-Chat-8B", trust_remote_code=True, use_fast=False,)

    #get encoder
    encoder = model.vision_encoder.cuda()
    args = parse_args()
    video_path = args.video_path
    # sample uniformly 8 frames from the video
    video_tensor_list, num_frames = load_video(video_path, num_segments=8, return_msg=False)

    video_feat_all = torch.zeros(num_frames, 16, 16, 1408, dtype=torch.float32).cuda()
    video_feat_list=[]
    cls_feat_list=[]
    for data in video_tensor_list:
        video_tensor, frame_id = data
        frame_id = torch.tensor(frame_id)
        video_tensor = video_tensor.to("cuda")
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        # specify ret_map=True to get the feat
        vid_feat, cls_feat=encoder(video_tensor, ret_map=True)
        video_feat_all[frame_id] = vid_feat
        cls_feat_list.append(cls_feat)

    cls_feat_avg = torch.stack(cls_feat_list).mean(dim=0).unsqueeze(0)

    result = {
        "video_feat": video_feat_all.cpu(),
        "cls_feat": cls_feat_avg,
    }
    
    save_dir = osp.join(args.video_path, "semantic_features")
    os.makedirs(save_dir, exist_ok = True)
    save_path = osp.join(save_dir, "internvideo_feats.pth")
    torch.save(result, save_path)



