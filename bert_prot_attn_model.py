import numpy as np
import nemo 
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tt

import os
import sys

import copy
import h5py

import torch
from attr import asdict
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
from nemo.utils import AppState, logging

from itertools import groupby

import re
from typing import Dict, Optional, Tuple, List, Union

from nemo.core.classes import NeuralModule
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

from nemo.core.neural_types.elements import ElementType, BoolType, StringType, StringLabel
from transformers import BertModel
from nemo.core.classes import typecheck

from transformers import (
    ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    AlbertConfig,
    AutoModel,
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
)

from nemo.collections.nlp.modules.common.huggingface.albert import AlbertEncoder
#from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.collections.nlp.modules.common.huggingface.distilbert import DistilBertEncoder
from nemo.collections.nlp.modules.common.huggingface.roberta import RobertaEncoder

#from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
#from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
#    get_huggingface_lm_model,
#    get_huggingface_pretrained_lm_models_list,
#)
from nemo.collections.nlp.modules.common.transformer.transformer import NeMoTransformerConfig
from nemo.collections.nlp.modules.common.transformer.transformer_utils import (
    get_huggingface_transformer,
    get_nemo_transformer,
)


# Extend the BertModule with an optional output_type that holds the attentions, similar to huggingface
# ----------------------------------------------------------------------------------------------------


class BertModuleExt(NeuralModule, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "output_attentions": NeuralType(None, BoolType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()), 
                #"attentions": NeuralType(('B', 'H', 'T', 'T'), ChannelType(), optional=True), 
        }
        # the only difference is the added optional 'attentions' output
        
        
    def restore_weights(self, restore_path: str):
        """Restores module/model's weights"""
        logging.info(f"Restoring weights from {restore_path}")

        if not os.path.exists(restore_path):
            logging.warning(f'Path {restore_path} not found')
            return

        pretrained_dict = torch.load(restore_path)

        # backward compatibility with NeMo0.11
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["state_dict"]

        # remove prefix from pretrained dict
        m = re.match("^bert.*?\.", list(pretrained_dict.keys())[0])
        if m:
            prefix = m.group(0)
            pretrained_dict = {k[len(prefix) :]: v for k, v in pretrained_dict.items()}
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # starting with transformers 3.1.0, embeddings.position_ids is added to the model's state dict and could be
        # missing in checkpoints trained with older transformers version
        if 'embeddings.position_ids' in model_dict and 'embeddings.position_ids' not in pretrained_dict:
            pretrained_dict['embeddings.position_ids'] = model_dict['embeddings.position_ids']

        assert len(pretrained_dict) == len(model_dict)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logging.info(f"Weights for {type(self).__name__} restored from {restore_path}")

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=max_dim - 1, size=sz, device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=sz, device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return tuple([input_dict])

# The most important bit - in the superclass allow the result to be full Tuple and do not restrict to the first element of it 
# as originally is written in the nemo package


class BertEncoder(BertModel, BertModuleExt):
    """
    Wraps around the Huggingface transformers implementation repository for easy use within NeMo.
    """

    @typecheck()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=False):
        res = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                 output_attentions=output_attentions)
        return res #BaseModelOutput(last_hidden_state=res[0], attentions=res[1]) 

# Now 'redefine' all the necessary functions with the new BertEncoder 

HUGGINGFACE_MODELS = {
    "BertModel": {
        "default": "bert-base-uncased",
        "class": BertEncoder,
        "config": BertConfig,
        "pretrained_model_list": BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "DistilBertModel": {
        "default": "distilbert-base-uncased",
        "class": DistilBertEncoder,
        "config": DistilBertConfig,
        "pretrained_model_list": DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "RobertaModel": {
        "default": "roberta-base",
        "class": RobertaEncoder,
        "config": RobertaConfig,
        "pretrained_model_list": ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "AlbertModel": {
        "default": "albert-base-v2",
        "class": AlbertEncoder,
        "config": AlbertConfig,
        "pretrained_model_list": ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
}

VOCAB_FILE_NAME = {
    'AlbertTokenizer': "spiece.model",
    'RobertaTokenizer': "vocab.json",
    'BertTokenizer': "vocab.txt",
    'DistilBertTokenizer': "vocab.txt",
}


def get_huggingface_lm_model(
    pretrained_model_name: str, config_dict: Optional[dict] = None, config_file: Optional[str] = None,
):
    """
    Returns lm model instantiated with Huggingface
    Args:
        pretrained_mode_name: specify this to instantiate pretrained model from Huggingface,
            e.g. bert-base-cased. For entire list, see get_huggingface_pretrained_lm_models_list().
        config_dict: model configuration dictionary used to instantiate Huggingface model from scratch
        config_file: path to model configuration file used to instantiate Huggingface model from scratch
    Returns:
        BertModule
    """

    try:
        automodel = AutoModel.from_pretrained(pretrained_model_name)
    except Exception as e:
        raise ValueError(f"{pretrained_model_name} is not supported by HuggingFace. {e}")

    model_type = type(automodel).__name__
    if model_type in HUGGINGFACE_MODELS:
        model_class = HUGGINGFACE_MODELS[model_type]["class"]
        if config_file:
            if not os.path.exists(config_file):
                logging.warning(
                    f"Config file was not found at {config_file}. Will attempt to use config_dict or pretrained_model_name."
                )
            else:
                config_class = HUGGINGFACE_MODELS[model_type]["config"]
                return model_class(config_class.from_json_file(config_file))
        if config_dict:
            config_class = HUGGINGFACE_MODELS[model_type]["config"]
            return model_class(config=config_class(**config_dict))
        else:
            return model_class.from_pretrained(pretrained_model_name)
    else:
        raise ValueError(f"Use HuffingFace API directly in NeMo for {pretrained_model_name}")


def get_huggingface_pretrained_lm_models_list(include_external: bool = False,) -> List[str]:
    """
    Returns the list of pretrained HuggingFace language models
    
    Args:
        include_external if true includes all HuggingFace model names, not only those supported language models in NeMo.
    
    Returns the list of HuggingFace models
    """

    huggingface_models = []
    if include_external:
        huggingface_models = list(ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
    else:
        for model in HUGGINGFACE_MODELS:
            model_names = HUGGINGFACE_MODELS[model]["pretrained_model_list"]
            huggingface_models.extend(model_names)
    return huggingface_models


# The next important step is the get_lm_model function, which now returns a BertModuleExt object

def get_pretrained_lm_models_list(include_external: bool = False) -> List[str]:
    """
    Returns the list of supported pretrained model names
    Args:
        include_external if true includes all HuggingFace model names, not only those supported language models in NeMo.
    """
    return get_huggingface_pretrained_lm_models_list(include_external=include_external)


def get_lm_model(
    pretrained_model_name: str,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    vocab_file: Optional[str] = None,
) -> BertModuleExt:
    """
    Helper function to instantiate a language model encoder, either from scratch or a pretrained model.
    If only pretrained_model_name are passed, a pretrained model is returned.
    If a configuration is passed, whether as a file or dictionary, the model is initialized with random weights.
    Args:
        pretrained_model_name: pretrained model name, for example, bert-base-uncased or megatron-bert-cased.
            See get_pretrained_lm_models_list() for full list.
        config_dict: path to the model configuration dictionary
        config_file: path to the model configuration file
        checkpoint_file: path to the pretrained model checkpoint
        vocab_file: path to vocab_file to be used with Megatron-LM
    Returns:
        Pretrained BertModuleExt
    """

    # check valid model type
    if not pretrained_model_name or pretrained_model_name not in get_pretrained_lm_models_list(include_external=False):
        logging.warning(
            f'{pretrained_model_name} is not in get_pretrained_lm_models_list(include_external=False), '
            f'will be using AutoModel from HuggingFace.'
        )

    # warning when user passes both configuration dict and file
    if config_dict and config_file:
        logging.warning(
            f"Both config_dict and config_file were found, defaulting to use config_file: {config_file} will be used."
        )

    if "megatron" in pretrained_model_name:
        raise ValueError('megatron-lm BERT models have been deprecated in NeMo 1.5+. Please use NeMo 1.4 for support.')
        # TODO: enable megatron bert in nemo
        # model, checkpoint_file = get_megatron_lm_model(
        #     config_dict=config_dict,
        #     config_file=config_file,
        #     pretrained_model_name=pretrained_model_name,
        #     checkpoint_file=checkpoint_file,
        #     vocab_file=vocab_file,
        # )
    else:
        model = get_huggingface_lm_model(
            config_dict=config_dict, config_file=config_file, pretrained_model_name=pretrained_model_name,
        )

    if checkpoint_file:
        app_state = AppState()
        if not app_state.is_model_being_restored and not os.path.exists(checkpoint_file):
            raise ValueError(f'{checkpoint_file} not found')
        model.restore_weights(restore_path=checkpoint_file)

    return model


# @dataclass
# class TransformerConfig:
#     library: str = 'nemo'
#     model_name: Optional[str] = None
#     pretrained: bool = False
#     config_dict: Optional[dict] = None
#     checkpoint_file: Optional[str] = None
#     encoder: bool = True


def get_transformer(
    library: str = 'nemo',
    model_name: Optional[str] = None,
    pretrained: bool = False,
    config_dict: Optional[dict] = None,
    checkpoint_file: Optional[str] = None,
    encoder: bool = True,
    pre_ln_final_layer_norm=True,
) -> Union[EncoderModule, DecoderModule]:
    """Gets Transformer based model to be used as an Encoder or Decoder in NeMo NLP.
       First choose the library to get the transformer from. This can be huggingface,
       megatron, or nemo. Use the model_name arg to get a named model architecture
       and use the pretrained arg to get the named model architecture with pretrained weights.
       If model_name is None, then we can pass in a custom configuration via the config_dict.
       For example, to instantiate a HuggingFace BERT model with custom configuration we would do:
       encoder = get_transformer(library='huggingface',
                                 config_dict={
                                     '_target_': 'transformers.BertConfig',
                                     'hidden_size': 1536
                                 }) 
    Args:
        library (str, optional): Can be 'nemo', 'huggingface', or 'megatron'. Defaults to 'nemo'.
        model_name (Optional[str], optional): Named model architecture from the chosen library. Defaults to None.
        pretrained (bool, optional): Use True to get pretrained weights. 
                                     False will use the same architecture but with randomly initialized weights.
                                     Defaults to False.
        config_dict (Optional[dict], optional): Use for custom configuration of transformer. Defaults to None.
        checkpoint_file (Optional[str], optional): Provide weights for the transformer from a local checkpoint. Defaults to None.
        encoder (bool, optional): True returns an EncoderModule, False returns a DecoderModule. Defaults to True.
    Returns:
        Union[EncoderModule, DecoderModule]: Ensures that Encoder/Decoder will work in EncDecNLPModel
    """

    model = None

    if library == 'nemo':
        if isinstance(config_dict, NeMoTransformerConfig):
            config_dict = asdict(config_dict)
        model = get_nemo_transformer(
            model_name=model_name,
            pretrained=pretrained,
            config_dict=config_dict,
            encoder=encoder,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        if checkpoint_file is not None:
            if os.path.isfile(checkpoint_file):
                raise ValueError(f'Loading transformer weights from checkpoint file has not been implemented yet.')

    elif library == 'huggingface':
        model = get_huggingface_transformer(
            model_name=model_name, pretrained=pretrained, config_dict=config_dict, encoder=encoder
        )

    elif library == 'megatron':
        raise ValueError(
            f'megatron-lm bert support has been deprecated in NeMo 1.5+. Please use NeMo 1.4 for support.'
        )
        # TODO: enable megatron bert in nemo
        # model = get_megatron_transformer(
        #     model_name=model_name,
        #     pretrained=pretrained,
        #     config_dict=config_dict,
        #     encoder=encoder,
        #     checkpoint_file=checkpoint_file,
        # )

    else:
        raise ValueError("Libary must be 'nemo', 'huggingface' or 'megatron'")

    return model


# Finally the prot_bert model with the functionality to return the attentions 

class BERTPROTModel_attn(ModelPT):
    """
    BERT language model pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        output_types_dict = {"mlm_log_probs": self.mlm_classifier.output_types["log_probs"]}
        if not self.only_mlm_loss:
            output_types_dict["nsp_logits"] = self.nsp_classifier.output_types["logits"]
        return output_types_dict

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if cfg.tokenizer is not None:
            self._setup_tokenizer(cfg.tokenizer)
        else:
            self.tokenizer = None

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=self.register_artifact('language_model.config_file', cfg.language_model.config_file),
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
            vocab_file=self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file)
            if cfg.tokenizer
            else None,
        )

        self.hidden_size = self.bert_model.config.hidden_size
        self.vocab_size = self.bert_model.config.vocab_size
        self.only_mlm_loss = cfg.only_mlm_loss
        # add output_attentions flag
        if self.bert_model.config.output_attentions:
            self.output_attentions = self.bert_model.config.output_attentions
        else:
            self.output_attentions = False
            
        self.mlm_classifier = BertPretrainingTokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.vocab_size,
            num_layers=cfg.num_tok_classification_layers,
            activation="gelu",
            log_softmax=True,
            use_transformer_init=True,
        )

        self.mlm_loss = SmoothedCrossEntropyLoss()

        if not self.only_mlm_loss:
            self.nsp_classifier = SequenceClassifier(
                hidden_size=self.hidden_size,
                num_classes=2,
                num_layers=cfg.num_seq_classification_layers,
                log_softmax=False,
                activation="tanh",
                use_transformer_init=True,
            )

            self.nsp_loss = CrossEntropyLoss()
            self.agg_loss = AggregatorLoss(num_inputs=2)

        # # tie weights of MLM softmax layer and embedding layer of the encoder
        if (
            self.mlm_classifier.mlp.last_linear_layer.weight.shape
            != self.bert_model.embeddings.word_embeddings.weight.shape
        ):
            raise ValueError("Final classification layer does not match embedding layer.")
        self.mlm_classifier.mlp.last_linear_layer.weight = self.bert_model.embeddings.word_embeddings.weight
        # create extra bias

        # setup to track metrics
        self.validation_perplexity = Perplexity(compute_on_step=False)

        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        out = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_attentions=self.output_attentions
        )
        mlm_log_probs = self.mlm_classifier(hidden_states=hidden_states)
        if self.only_mlm_loss:
            return (mlm_log_probs,)
        nsp_logits = self.nsp_classifier(hidden_states=hidden_states)
        
        attentions = out['attentions'] if self.output_attentions else None
        return mlm_log_probs, nsp_logits, out['last_hidden_state'], attentions

    def _compute_losses(self, mlm_log_probs, nsp_logits, output_ids, output_mask, labels):
        mlm_loss = self.mlm_loss(log_probs=mlm_log_probs, labels=output_ids, output_mask=output_mask)
        if self.only_mlm_loss:
            loss, nsp_loss = mlm_loss, None
        else:
            nsp_loss = self.nsp_loss(logits=nsp_logits, labels=labels)
            loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)
        return mlm_loss, nsp_loss, loss

    def _parse_forward_outputs(self, forward_outputs):
        if self.only_mlm_loss:
            return forward_outputs[0], None
        else:
            return forward_outputs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        forward_outputs = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        mlm_log_probs, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(mlm_log_probs, nsp_logits, output_ids, output_mask, labels)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {"loss": loss, "lr": lr}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        forward_outputs = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        mlm_log_probs, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(mlm_log_probs, nsp_logits, output_ids, output_mask, labels)
        self.validation_perplexity(logits=mlm_log_probs)
        val_perplexity = self.validation_perplexity(logits=mlm_log_probs)
        self.log('val_loss', loss)
        self.log('val_perplexity', val_perplexity)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """Called at the end of validation to aggregate outputs.

        Args:
            outputs (list): The individual outputs of each validation step.

        Returns:
            dict: Validation loss and tensorboard logs.
        """
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            perplexity = self.validation_perplexity.compute()
            logging.info(f"evaluation perplexity {perplexity.cpu().item()}")
            self.log(f'val_loss', avg_loss)
        
    def predict_step(self, batch, batch_idx):
        """
        Called in the predict command with the data from the inference dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, seq_names = batch
        out = self.bert_model(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask,
            output_attentions=self.output_attentions
        )
        
        attentions = out['attentions'] if self.output_attentions else None
        return (seq_names, out['last_hidden_state'], attentions, input_mask)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = (
            self._setup_preprocessed_dataloader(train_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(train_data_config)
        )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = (
            self._setup_preprocessed_dataloader(val_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(val_data_config)
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        pass

    def _setup_preprocessed_dataloader(self, cfg: Optional[DictConfig]):
        dataset = cfg.data_file
        max_predictions_per_seq = cfg.max_predictions_per_seq
        batch_size = cfg.batch_size

        if os.path.isdir(dataset):
            files = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
        else:
            files = [dataset]
        files.sort()
        dl = BertPretrainingPreprocessedDataloader(
            data_files=files, max_predictions_per_seq=max_predictions_per_seq, batch_size=batch_size,
        )
        return dl

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            vocab_file=cfg.vocab_file,
        )
        self.tokenizer = tokenizer

    def _setup_dataloader(self, cfg: DictConfig):
        dataset = BertPretrainingDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.data_file,
            max_seq_length=cfg.max_seq_length,
            mask_prob=cfg.mask_prob,
            short_seq_prob=cfg.short_seq_prob,
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get("drop_last", False),
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertbaseuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/bertbaseuncased/versions/1.0.0rc1/files/bertbaseuncased.nemo",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertlargeuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/bertlargeuncased/versions/1.0.0rc1/files/bertlargeuncased.nemo",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )
        return result



# How to use this:
# ----------------
# Given a .nemo file from a pretrained BERTPROTModel (found in proteonemo.models) take the config file 'cfg' from it.
#
# 1. add an output_attentions=True flag in the cf.language_model part
# 2. change the 'target' part of the cfg to the new 'BERTPROTModel_attn' model 
# 3. with this updated config, create an instance of 'BERTPROTModel_attn' 
# 4. load the weights in it and the model is ready for use and it will have the attentions in the 'predict_step' function
#
# E.g. 

#import copy

# Get the config file from the .nemo file

#tmp_protbert = BERTPROTModel_attn.restore_from('pretrained_model.nemo')
#cfg = copy.deepcopy(tmp_protbert.cfg)

# Step 1. and 2. - Modify the cfg

#OmegaConf.set_struct(cfg, False)
#cfg.language_model.config['output_attentions'] = True
#cfg.target = 'BERTPROTModel_attn'
#OmegaConf.set_struct(cfg, True)

# Step 3. - create a new instance of the BERTPROTModel_attn with the modified config

#tmp_protbert = BERTPROTModel_attn(cfg)

# Step 4. - load the weights from the model_weight.ckpt of the nemo file

#tmp_protbert.load_state_dict(torch.load('model_weights.ckpt'))