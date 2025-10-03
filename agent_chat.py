# Authors: Hui Ren (rhfeiyang.github.io)
import sys
sys.path.append("internvideo_chat_feature")
import argparse
import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer
from internvideo_chat_feature.modeling_videochat2 import InternVideo2_VideoChat2
import yaml
import os
from lib_4d.autoencoder.model import Feature_heads
import time ### timer
import logging

# Suppress the specific transformers warning about pad_token_id
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

def get_indices_split(num_frames, num_segments):
    # results = set()
    seg_size = num_frames // num_segments
    offsets_first = np.array([int(seg_size * i) for i in range(num_segments)])
    offsets = [i + offsets_first for i in range(seg_size)]
    if offsets[-1][-1] < num_frames-1:
        new_offset_reverse = [num_frames-1-seg_size*i for i in range(num_segments)]
        offsets.append(np.array(new_offset_reverse[::-1]))
    return offsets

def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of internvideo')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset (e.g., bear)')
    parser.add_argument('--rendered_view', type=str, default='3D_moving', 
                        help='View of rendered results (e.g., 3D_moving, 41_round_moving, etc.)')
    parser.add_argument("--head_config", type=str, default="configs/default_config.yaml")
    parser.add_argument('--huggingface_token', type=str, help='Hugging Face access token for model access')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive chat mode')
    parser.add_argument('--question', type=str, help='Single question to ask (non-interactive mode)')
    
    # Legacy arguments for backward compatibility
    parser.add_argument('--rendered_results_path', type=str, help='(Legacy) Direct path to rendered results')
    parser.add_argument("--ori_feat_path", type=str, help='(Legacy) Direct path to original features')
    
    args = parser.parse_args()
    
    # Auto-construct paths if using new data_name approach
    if args.data_name and not args.rendered_results_path:
        args.rendered_results_path = f"output/{args.data_name}/final_viz/{args.rendered_view}/rendered_results.pth"
        args.ori_feat_path = f"data/{args.data_name}/preprocess/semantic_features/internvideo_feats.pth"
        
        # Verify paths exist
        if not os.path.exists(args.rendered_results_path):
            print(f"❌ Rendered results not found: {args.rendered_results_path}")
            # Try to find available views
            base_dir = f"output/{args.data_name}/final_viz"
            if os.path.exists(base_dir):
                views = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                if views:
                    print(f"Available views: {', '.join(views)}")
            sys.exit(1)
            
        if not os.path.exists(args.ori_feat_path):
            print(f"❌ Feature file not found: {args.ori_feat_path}")
            sys.exit(1)
            
        print(f"✅ Using rendered results: {args.rendered_results_path}")
        print(f"✅ Using feature file: {args.ori_feat_path}")
    
    args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args


@torch.no_grad()
def main(args):
    # Get Hugging Face access token
    if args.huggingface_token:
        access_token = args.huggingface_token
        print("✅ Using provided Hugging Face token")
    elif os.getenv('HF_TOKEN'):
        access_token = os.getenv('HF_TOKEN')
        print("✅ Using HF_TOKEN from environment variables")
    else:
        print("\n🔐 Hugging Face Access Token Required")
        print("You need a Hugging Face access token to use InternVideo2-Chat-8B.")
        print("")
        print("💡 Setup Options:")
        print("   1. Set environment variable: export HF_TOKEN=your_token_here")
        print("   2. Add to your ~/.bashrc: echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc")
        print("   3. Or provide via command line: --huggingface_token your_token_here")
        print("")
        print("Get your token at: https://huggingface.co/settings/tokens")
        access_token = input("Enter your Hugging Face access token: ").strip()
        
        if not access_token:
            print("❌ Access token is required to proceed.")
            print("💡 Tip: Set HF_TOKEN environment variable to avoid entering it each time.")
            sys.exit(1)
    
    rendered_results = torch.load(args.rendered_results_path,weights_only=True)
    ori_feats = torch.load(args.ori_feat_path,weights_only=True)
    cls_feat = ori_feats["cls_feat"].view(1,1,1408)
    print(rendered_results.keys())
    ###
    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
    feature_heads.load_state_dict(state_dict)
    feature_heads.eval()
    ###
    all_video_feat = []
    for vid, result in rendered_results.items():
        # first resize to 16*16
        feat = F.interpolate(result["feature_map"][None, ...], size=(16, 16), mode='area')[0]
        feat = feat.permute(1,2,0)
        feat = feature_heads.decode("internvideo" , feat)
        all_video_feat.append(feat)
    all_video_feat = torch.stack(all_video_feat)
    split = get_indices_split(all_video_feat.size(0), 8)[0]
    sampled_video_feat = all_video_feat[split]
    # sampled_video_feat = ori_feats['video_feat'][split]

    sampled_video_feat_flatten = sampled_video_feat.view(1,-1,1408)

    gathered_result = torch.cat([cls_feat.cuda(), sampled_video_feat_flatten.cuda()], dim=1)

    model = InternVideo2_VideoChat2.from_pretrained("OpenGVLab/InternVideo2-Chat-8B", token=access_token).cuda()

    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
    
    # Suppress the pad_token_id warning by setting it explicitly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Set generation defaults to prevent warnings
    tokenizer.padding_side = "left"
    
    # Define a clean generation config to avoid sampling warnings
    generation_config = {
        'do_sample': False,
        'top_p': None,
        'top_k': None,
        'temperature': None,
        'feat': gathered_result
    }
    ### timer
    # start_time = time.time()
    ###
    chat_history = []
    
    if args.interactive:
        # Interactive chat mode
        print("\n" + "="*60)
        print("💬 4D Visual Question Answering - Interactive Chat")
        print("="*60)
        print(f"📂 Dataset: {args.data_name}")
        print(f"🎭 View: {args.rendered_view}")
        print("\n💡 Tips:")
        print("   - Ask questions about the 4D scene")
        print("   - Type 'quit', 'exit', or 'bye' to end the conversation")
        print("   - Type 'clear' to clear chat history")
        print("\n🤖 Assistant: Hello! I'm ready to answer questions about your 4D scene. What would you like to know?")
        
        while True:
            try:
                user_message = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_message.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Assistant: Goodbye! Thanks for using 4D VQA!")
                    break
                
                # Check for clear command
                if user_message.lower() == 'clear':
                    chat_history = []
                    print("\n🧹 Chat history cleared!")
                    print("🤖 Assistant: Chat history has been cleared. What would you like to know about the scene?")
                    continue
                
                # Skip empty messages
                if not user_message:
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                response, chat_history = model.chat(tokenizer, '', user_message,
                                                    media_type='video', media_tensor=None, chat_history=chat_history,
                                                    return_history=True, generation_config=generation_config)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Goodbye! Thanks for using 4D VQA!")
                break
            except EOFError:
                print("\n\n🤖 Assistant: Goodbye! Thanks for using 4D VQA!")
                break
                
    else:
        # Single question mode
        if args.question:
            user_message = args.question
        else:
            user_message = "Give a detailed description of the scene."
            
        print(f"\n💬 Question: {user_message}")
        print("🤖 Response: ", end="", flush=True)
        response, chat_history = model.chat(tokenizer, '', user_message,
                                            media_type='video', media_tensor=None, chat_history=chat_history,
                                            return_history=True, generation_config=generation_config)
        print(response)

    ### timer
    # end_time = time.time()
    # print(f"Execution Time: {end_time - start_time:.6f} seconds")
    ###


if __name__ == "__main__":
    args = parse_args()
    main(args)
