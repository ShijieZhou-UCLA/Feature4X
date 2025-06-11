import copy
import re, ast
from transformers import AutoConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from easydict import EasyDict as MyEasyDict
from importlib import import_module
import os.path as osp
import argparse
import json
from copy import deepcopy
import sys


class VideoChat2Config(PretrainedConfig):
    model_type = 'InternVideo2_VideoChat2'

    def __init__(
            self,
            model_config=None,
            **kwargs):
        super().__init__(**kwargs)
        self.model_config  = MyEasyDict(model_config)