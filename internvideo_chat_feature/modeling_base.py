import io
import os
import warnings
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss

from torch.cuda.amp import autocast as autocast

from modeling_internvideo2_vit import pretrain_internvideo2_giant_patch14_224_clean
from modeling_qformer import build_qformer
from model_config import VideoChat2Config

logger = logging.getLogger(__name__)

from transformers import LlamaTokenizer,AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoProcessor
from transformers import AutoConfig, PreTrainedModel

try:
    token = os.environ['HF_TOKEN']
except:
    warnings.warn("The HF_TOKEN was not found in the system variables. Please ensure that it is filled out correctly and that you have requested access to the model. If you haven't applied, please visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 to request access.")
    token=None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def freeze_module(module):
    for _, param in module.named_parameters():
        param.requires_grad = False
    module = module.eval()
    module.train = disabled_train
    return module


class BaseMLLM(PreTrainedModel):
    config_class = VideoChat2Config
    def __init__(self, config):
        self.model_config = config.model_config
        config.model_config = None
        super().__init__(config)
        self.build_vision_encoder()
        self.build_llm()
        self.build_bridge()
        self.build_loss()
        # NOTE place it after freeze llm
        for n, p in self.named_parameters():
            if p.requires_grad:
                logger.info(f'{n} requires_grad')
        
    
    def build_vision_encoder(self):
        # load pretrained internvideo2-1b here, simplified as it receives no args
        # note that we haven't load the internvideo pretrained version
        if 'internvideo2' in self.model_config.vision_encoder.name.lower():
            encoder_name = self.model_config.vision_encoder.name
            logger.info(f"Build vision_encoder: {encoder_name}")
            if encoder_name == 'internvideo2-1B':
                self.vision_encoder = pretrain_internvideo2_giant_patch14_224_clean(self.model_config)
            else:
                raise ValueError(f"Not implemented: {encoder_name}")
        else:
            raise NotImplementedError(self.model_config.vision_encoder.name)

        if self.model_config.vision_encoder.vit_add_ln:
            self.vision_layernorm = nn.LayerNorm(self.model_config.vision_encoder.encoder_embed_dim, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()

        self.freeze_vision_encoder = self.model_config.get("freeze_vision_encoder", False)

        if self.freeze_vision_encoder:
            logger.info("freeze vision encoder")
            freeze_module(self.vision_encoder)
            freeze_module(self.vision_layernorm)


    def build_bridge(self):
        # ViT to LM: 1792 -> 6656 NOTE 768 is qformer dim
        self.project_up = nn.Linear(768, self.lm.config.hidden_size) # whether bias is needed?
        # LM to ViT: 6656 -> 1792
        self.project_down = nn.Linear(self.lm.config.hidden_size, 768)
        
        if 'qformer' in self.model_config.bridge.name.lower():
            from transformers import BertTokenizer
            self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
            self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.qformer_tokenizer.padding_side = "left"
            if self.model_config.bridge.name == 'qformer':
                self.qformer, self.query_tokens = build_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob,
                        qformer_drop_path_rate=self.model_config.bridge.qformer_drop_path_rate,
                )
            self.qformer.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.cls = None
            self.extra_num_query_token = self.model_config.bridge.extra_num_query_token
            if self.model_config.bridge.extra_num_query_token > 0:
                logger.info(f"Add extra {self.model_config.bridge.extra_num_query_token} tokens in QFormer")
                self.extra_query_tokens = nn.Parameter(
                    torch.zeros(1, self.model_config.bridge.extra_num_query_token, self.query_tokens.shape[-1])
                )
            
            self.freeze_bridge = self.model_config.get("freeze_bridge", False)
            if self.freeze_bridge:
                logger.info("freeze bridge")
                freeze_module(self.qformer)
                self.query_tokens.requires_grad = False

    def build_llm(self):
        self.lm_name = self.model_config.llm.name
        if self.model_config.llm.name == 'mistral_7b':
            from transformers import AutoModelForCausalLM
            config = AutoConfig.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                token=token,
                # attn_implementation="flash_attention_2",
            )
            self.lm = AutoModelForCausalLM.from_config(config)
        elif self.model_config.llm.name == 'internlm_20b':
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.lm.gradient_checkpointing = True
            self.lm._set_gradient_checkpointing()
        elif self.model_config.llm.name == 'internlm2_5_7b':
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            raise NotImplementedError(self.model_config.llm.name)

        self.freeze_llm = self.model_config.get("freeze_llm", True)
        logger.info(f'freeze_llm: {self.freeze_llm}')
        if self.freeze_llm:
            logger.info("freeze llm")
            freeze_module(self.lm)
        
        if self.model_config.llm.use_lora:
            self.use_lora = True
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("Use lora")
            if self.model_config.llm.name == 'internlm_20b':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj", "lm_head"]
                )
                
            self.lm = get_peft_model(self.lm, peft_config)
            self.lm.enable_input_require_grads()
            self.lm.print_trainable_parameters()
        else:
            self.use_lora = False


    def build_loss(self):
        self.use_vision_regression_loss = self.model_config.loss.get("use_vision_regression_loss", False)
        if self.use_vision_regression_loss:
            self.image_loss_fct = MSELoss()
        
    @property
    def dtype(self):
        return self.lm.dtype


    @property
    def device(self):
        return self.lm.device
