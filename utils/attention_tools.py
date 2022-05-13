import array
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
import typing
from typing import List, Tuple, Dict, Optional

import proteonemo
import bert_prot_attn_model
from proteonemo.data import prot_bert_dataset

# Some default parameter for inference
DEFAULT = {}
DEFAULT['num_workers'] = 8
DEFAULT['batch_size'] = 8
DEFAULT['tensor_model_parallel_size'] = 1
DEFAULT['max_seq_length'] = 1024
DEFAULT['do_upper_case'] = True


def _get_bert_prot_attn_model(filepath_nemo: str):
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


def _get_attention_maps(model_instance, input_file: str) -> List[Tuple[str, Tuple[torch.Tensor]]]:
    '''Returns a tuple of sequence_id and attention maps. Attention maps are torch tensors and in a tuple.
       input: 
            model_instance - a porteonemo model (BERTPROTModel_attn) extended with the attention maps features;
            input_file - input file of the sequences in hdf5. File is in standard proteonemo format also used to 
            extract representations, for training, etc.
       output:
            List of tuples, where a tuple contains sequence_ids and attention maps. The latter are a tuple of torch.Tensor objects, 
            which represent the attention maps for each layer and head. One element of the tuple represents one layer and the tensor 
            has a shape [num_heads, sequence_length, sequence_length] 
    '''
    inference_dataset = prot_bert_dataset.BertInferencePreprocessedDataset(f'{input_file}')
    request_dl = DataLoader(inference_dataset, 
        batch_size=DEFAULT['batch_size'],
        shuffle=False,
        num_workers=DEFAULT['num_workers'])

    torch.set_grad_enabled(False)
    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=True)], 
                        devices=DEFAULT['tensor_model_parallel_size'], 
                        precision=16, 
                        accelerator='gpu')
    preds = trainer.predict(model_instance, request_dl)
    
    # get the attention maps
    attn_maps = []
    for pred_ in preds:
        for i, seq_name in enumerate(pred_[0][0]):
            seq_len = np.where(pred_[3][i]==1)[0][-1]
            attn_maps.append((seq_name, tuple([layer[i, ..., 1:seq_len, 1:seq_len].clone() for layer in pred_[2]])))
    
    return attn_maps
