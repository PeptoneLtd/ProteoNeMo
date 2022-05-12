import numpy as np
import nemo 
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tt

import os
import sys
from omegaconf import OmegaConf

import copy
import h5py

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataloader,
)
from nemo.collections.nlp.modules.common import BertPretrainingTokenClassifier, SequenceClassifier
#from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

from itertools import groupby

import proteonemo
import bert_prot_attn_model

def _get_bert_prot_attn_model(filepath_nemo):
    ''' Returns an instance of a model that is extended with attention_maps
        input: nemo file - trained with weights 
        output: instance of a model extended with attention maps'''

    # extract and save the model weights from the .nemo file
    parent_folder = os.path.dirname(filepath_nemo)
    if not os.path.isdir(f'{parent_folder}/pt_ckpt/'):
        os.mkdir(f'{parent_folder}/pt_ckpt/')
    
    # create a temporary model based on the .nemo file 
    tmp_bert = bert_prot_attn_model.BERTPROTModel_attn.restore_from(f'{filepath_nemo}')
    tmp_bert.extract_state_dict_from(f'{filepath_nemo}', save_dir=f'{os.path.dirname(filepath_nemo)}/pt_ckpt/')

    # modify the config
    cfg = copy.deepcopy(tmp_bert.cfg)

    # OmegaConf won't allow you to add new config items, so we temporarily disable this safeguard.
    OmegaConf.set_struct(cfg, False)
    # add new value
    cfg.language_model.config['output_attentions'] = True
    # change the target value
    cfg.target = 'BERTPROTModel_attn'
    # Here, we restore the safeguards so no more additions can be made to the config
    OmegaConf.set_struct(cfg, True)

    # now create an updated model instance based on the modified config
    tmp_bert_attn = bert_prot_attn_model.BERTPROTModel_attn(cfg)
    # load the weights
    tmp_bert_attn.load_state_dict(torch.load(f'{os.path.dirname(filepath_nemo)}/pt_ckpt/model_weights.ckpt'))

    return tmp_bert_attn


