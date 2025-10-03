import os, sys, os.path as osp
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 16],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[32, 64, 128, 256, 512, 1024, 1408],
                    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"
    data_dir = f"{dataset_path}/code_output"

    # output_dir = f"{dataset_path}/language_features_dim3"
    # os.makedirs(output_dir, exist_ok=True)
    # # copy the segmentation map
    # for filename in os.listdir(data_dir):
    #     if filename.endswith("_s.npy"):
    #         source_path = os.path.join(data_dir, filename)
    #         target_path = os.path.join(output_dir, filename)
    #         shutil.copy(source_path, target_path)

    # load ground truth semantic features (80 x 16 x 16 x1408)
    feature_path = osp.join(data_dir, "video_feat.pth")
    if osp.exists(feature_path):
        semantic_features = torch.load(osp.join(feature_path), weights_only=True)
        print(f"semantic feature loaded from {feature_path}")
    else:
        raise ValueError(f"semantic feature not found in {feature_path}, please first extract video feature")
    gt_semantic_features = semantic_features["video_feat"].to(torch.float32).to("cuda:0")
    print("gt_semantic_features", gt_semantic_features.shape)

    train_loader = [gt_semantic_features[i:i+1] for i in range(gt_semantic_features.size(0))]
    test_loader = train_loader

    # train_dataset = Autoencoder_dataset(data_dir)
    # test_loader = DataLoader(
    #     dataset=train_dataset, 
    #     batch_size=256,
    #     shuffle=False, 
    #     num_workers=16, 
    #     drop_last=False   
    # )

    checkpoint = torch.load(ckpt_path)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()

    all_results = {}
    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.view(-1, 1408).to("cuda:0") # 256 x 1408
        with torch.no_grad():
            latent = model.encode(data)
            output = model.decode(latent) # 256 x 1408

        output = output.view(16, 16, 1408).permute(2, 0, 1) # 1408 x 16 x 16
        render_dict={}
        render_dict["feature_map"] = output
        all_results[idx] = render_dict

    torch.save(all_results, osp.join(f"ckpt/{dataset_name}", f"rendered_results.pth"))
    # os.makedirs(output_dir, exist_ok=True)
    # start = 0
    
    # for k,v in train_dataset.data_dic.items():
    #     path = os.path.join(output_dir, k)
    #     np.save(path, features[start:start+v])
    #     start += v
