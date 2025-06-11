import torch
import torch.nn as nn
import os.path as osp
import yaml

def load_head(feature_config, feature_head_ckpt_path, device="cuda"):
    if osp.exists(feature_config):
        with open(feature_config, "r") as f:
            feature_config = yaml.load(f, Loader=yaml.FullLoader)
            head_config = feature_config["Head"]

        feature_head = Feature_heads(head_config).to(device)

        feature_head_state = torch.load(feature_head_ckpt_path, weights_only=True)
        feature_head.load_state_dict(feature_head_state)
        feature_head.eval()
    else:
        raise "feature head config {feature_config} not found"

    return feature_head



class Feature_heads(nn.Module):
    def __init__(self, head_config:dict,):
        super(Feature_heads, self).__init__()
        self.head_config = head_config
        semantic_heads = {}
        for feature_name in head_config.keys():
            feature_config = head_config[feature_name]
            if not feature_config["enable"]:
                continue
            for type in ["encoder", "decoder"]:
                if len(feature_config[type]) > 0:
                    channel = feature_config[type]
                    if feature_config["head_type"] == "mlp":
                        semantic_heads[f"{feature_name}_{type}"] = MLP(channel)
                    elif feature_config["head_type"] == "cnn":
                        assert len(channel) == 2
                        semantic_heads[f"{feature_name}_{type}"] = CNN_decoder(channel[0], channel[1])
                    else:
                        raise NotImplementedError

        self.semantic_heads = nn.ModuleDict(semantic_heads)
    def decode(self, feature_name, x):
        # assert feature_name in self.head_config.keys()
        if f"{feature_name}_decoder" in self.semantic_heads.keys():
            return self.semantic_heads[f"{feature_name}_decoder"](x)
        else:
            return x
    def encode(self, feature_name, x):
        # assert feature_name in self.head_config.keys()
        if f"{feature_name}_encoder" in self.semantic_heads.keys():
            return self.semantic_heads[f"{feature_name}_encoder"](x)
        else:
            return x
    def keys(self):
        ret= []
        for key in self.head_config.keys():
            if self.head_config[key]["enable"]:
                ret.append(key)
        return ret


class MLP(nn.Module):
    def __init__(self, hidden_dims):
        super(MLP, self).__init__()
        layers = []
        for i in range(1,len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if i != len(hidden_dims)-1:
                layers.append(nn.ReLU())
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x

class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.input_dim = input_dim
        #self.output_dim = output_dim

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):
        x = x.permute(2,0,1)
        x = self.conv(x)
        x = x.permute(1,2,0)
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(1408, encoder_hidden_dims[i])) 
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
             
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        #print(self.encoder, self.decoder)
    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

if __name__ == "__main__":
    feature_config = "/public/home/renhui/code/4d/feature-4dgs/output/Balloon1/native_feat_langseg_mlp_nvidia.yaml_compactgs_mixfeat_nomotion_channel32_dep=uni_gt_cam=False_lrfeat=0.01_reversed=False_20241114_145348/feature_config.yaml"
    feature_head_ckpt_path = "/public/home/renhui/code/4d/feature-4dgs/output/Balloon1/native_feat_langseg_mlp_nvidia.yaml_compactgs_mixfeat_nomotion_channel32_dep=uni_gt_cam=False_lrfeat=0.01_reversed=False_20241114_145348/finetune_semantic_heads.pth"
    feature_head = load_head(feature_config, feature_head_ckpt_path)
    print(feature_head.keys())