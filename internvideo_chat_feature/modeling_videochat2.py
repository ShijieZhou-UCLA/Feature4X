import io
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast as autocast
from modeling_base import (BaseMLLM)
# from .modeling_internvideo2_vit import pretrain_internvideo2_giant_patch14_224_clean, interpolate_pos_embed_internvideo2_new
# from .modeling_qformer import build_qformer
# from .flash_attention_class import FlashAttention

logger = logging.getLogger(__name__)

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"

class InternVideo2_VideoChat2(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        video_idx = None,
        image_idx = None,
        feat=None
    ):  
        if self.use_vision_regression_loss:
            text_embeds, visual, visual_idx = self.pad_text_embeds(input_ids=input_ids, image=image,video=video, return_visual=True, video_idx=video_idx, image_idx=image_idx, instruction = instruction, feat=feat)
        else:
            text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False, video_idx=video_idx, image_idx=image_idx,  instruction = instruction, feat=feat)
        
        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs

    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        return_visual: bool = False,
        instruction = None,
        feat=None
    ):
        # if feat is not None:
        #     return self.pad_text_embeds_feat(input_ids, feat, video=True, video_idx=video_idx, return_visual=return_visual, instruction=instruction)
        # text_embeds
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()

        visual = None
        visual_idx = None

        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            prompt_image_embeds = self.encode_vision(image, instruction=instruction, feat=feat)
            visual = prompt_image_embeds
            prompt_image_embeds = self.project_up(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])
            visual_idx = image_idx
            text_embeds[image_idx == 1] = text_embeds[image_idx == 1] * 0 + prompt_image_embeds.to(text_embeds.device)
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape
            video = video.reshape(B*N, T, C, H, W).permute(0, 2, 1, 3, 4)
            prompt_video_embeds = self.encode_vision(video, instruction=instruction, feat=feat)
            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            visual_idx = video_idx
            text_embeds[video_idx == 1] = text_embeds[video_idx == 1] * 0 + prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)
        elif feat is not None:
            prompt_video_embeds = self.encode_vision(video, instruction=instruction, feat=feat)
            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            visual_idx = video_idx
            text_embeds[video_idx == 1] = text_embeds[video_idx == 1] * 0 + prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)

        else:
            logger.warn(f"don't get visual input, input_ids: {input_ids}")
            
        if return_visual:
            return text_embeds, visual, visual_idx
        
        return text_embeds

    def pad_text_embeds_feat(
            self,
            input_ids: torch.LongTensor = None,
            feat: Optional[torch.Tensor] = None,
            video=True, # False: image, not implemented
            image_idx = None,
            video_idx = None,
            return_visual: bool = False,
            instruction = None,
    ):
        # text_embeds
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()

        # if len(video.shape) == 5:
        #     B, T, C, H, W = video.shape
        #     N = 1
        # else:
        #     B, N, T, C, H, W = video.shape
        # video = video.reshape(B*N, T, C, H, W).permute(0, 2, 1, 3, 4)
        # prompt_video_embeds = self.encode_vision(video, instruction=instruction)
        if video:
            prompt_video_embeds = feat
            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            visual_idx = video_idx
            text_embeds[video_idx == 1] = text_embeds[video_idx == 1] * 0 + prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)


        if return_visual:
            return text_embeds, visual, visual_idx

        return text_embeds

    def encode_vision(
        self,
        image=None,
        instruction=None,
        feat=None
    ):

        if feat is None:
            device = image.device
            B = image.shape[0]
            T = image.shape[2]
            # tmp = self.vision_encoder(image, use_image=use_image, ret_map=True)
            use_image = True if T == 1 else False
            image_embeds = self.vision_encoder(image, use_image=use_image)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_embeds = feat.to(device)
            B = 1 
            C = 1408
            

        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C).to(device)
        self.vision_layernorm = self.vision_layernorm.to(device)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        if instruction is not None:
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :]


    def generate_caption(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
        feat=None
    ):
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx, feat=feat)
        outputs = self.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs
    
    def build_input_ids(
            self, 
            tokenizer, 
            conversation,
            max_length,
            add_special_tokens,
            truncation,
            image = None, 
            video = None, 
            padding = "longest", 
            return_tensors = "pt",
            image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
            video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        input_ids = []
        indexs = []
        attention_mask = []
        start, total_len = 0, 0
        while True:
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            if index == -1:
                inputs = tokenizer(conversation[start:], max_length=max_length-total_len, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = tokenizer(conversation[start:index], max_length=max_length,  truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)
            
            if index != -1:
                input_ids += [torch.zeros(96).long()]
                attention_mask += [torch.ones(96).long()]
                indexs += [torch.ones(96)]
            
            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),
                    'attention_mask': torch.cat(attention_mask),
                    'index': torch.cat(indexs).to(torch.bool),
                }
            start = index + len(DEFAULT_IMG_PLACEHOLDER)
            
    def chat(
      self,
      tokenizer,
      msg,
      user_prompt,
      media_type,
      media_tensor, 
      instruction=None,
      chat_history =[],
      return_history =False,
      generation_config={}
    ):
        input_ids, attention_masks, labels = [], [], []

        conversation = ""
        if instruction:
            conversation += instruction
        conversation += (
                    "[INST]" + " "
                )

        if media_type == 'image':
            conversation +=( "<Image>" + IMG_TOKEN + "</Image>")#*ilen
        else:
            conversation += ("<Video>" + VID_TOKEN + "</Video>")#*ilen


        conversation += (
                    msg.rstrip() + "[/INST]"
                )

        for q,a in chat_history:
            conversation += (" [INST] " + q + " [/INST]")
            conversation += (a + "</s>")

        conversation += (" [INST] " + user_prompt + " [/INST]")
        conversation += ("")


        total_len = 0
        indexs = []
        tokenized = self.build_input_ids(
            tokenizer,
            conversation,
            max_length=248,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        if media_type == 'image':
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device), 
                tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                image_idx = tokenized['index'].unsqueeze(0),
                image = media_tensor.unsqueeze(0) if media_tensor is not None else None,
                **generation_config)
        else:
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device), 
                tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                video_idx = tokenized['index'].unsqueeze(0),
                video = media_tensor.unsqueeze(0) if media_tensor is not None else None,
                **generation_config)
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        if return_history:
            chat_history.append((user_prompt,response))
            return response, chat_history
        return response