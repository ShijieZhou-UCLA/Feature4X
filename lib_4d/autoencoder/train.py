import os, sys, os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.007)
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
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    data_dir = f"{dataset_path}/code_output"
    os.makedirs(f'ckpt/{args.dataset_name}', exist_ok=True)


    # load ground truth semantic features (80 x 16 x 16 x1408)
    feature_path = osp.join(data_dir, "video_feat.pth")
    if osp.exists(feature_path):
        semantic_features = torch.load(osp.join(feature_path), weights_only=True)
        print(f"semantic feature loaded from {feature_path}")
    else:
        raise ValueError(f"semantic feature not found in {feature_path}, please first extract video feature")
    gt_semantic_features = semantic_features["video_feat"].to(torch.float32).to("cuda:0")
    print("gt_semantic_features", gt_semantic_features.shape)

    #train_dataset = gt_semantic_features.detach().cpu().numpy() 
    #train_dataset = gt_semantic_features
    #print("train_dataset:", train_dataset.shape)

    train_loader = [gt_semantic_features[i:i+1] for i in range(gt_semantic_features.size(0))]
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=16,
    #     drop_last=False
    # )
    # for batch in train_loader:
    #     print(f"Batch length: {len(batch)}")
    # for labels, data in enumerate(train_loader):
    #     print("Labels shape:", labels)
    #     print("Data shape:", data.shape)

    test_loader = train_loader
    # test_loader = DataLoader( # creates 80 tensors of shape 1x16x16x1408
    #     dataset=train_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=16,
    #     drop_last=False  
    # )

    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = f'ckpt/{args.dataset_name}'
    tb_writer = SummaryWriter(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            #data = feature.squeeze().to("cuda:0") # 16 x 16 x 1408
            data = feature.view(-1, 1408).to("cuda:0") # 256 x 1408
            outputs_dim3 = model.encode(data)
            #print("encoded data", outputs_dim3.shape)
            outputs = model.decode(outputs_dim3)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            #print("loss", loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            #print("GT data shape:", data.shape)
            #print("Outputs shape:", outputs.shape)
            tb_writer.add_histogram("feat", outputs, global_iter)

        if epoch > 95:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.view(-1, 1408).to("cuda:0") # 256 x 1408
                with torch.no_grad():
                    outputs = model(data) 
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_loader)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/best_ckpt.pth')
                
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/{epoch}_ckpt.pth')
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))