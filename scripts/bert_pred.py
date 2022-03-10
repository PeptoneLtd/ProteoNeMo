# Copyright (c) 2021 Peptone.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from pytorch_lightning.plugins import DDPPlugin
from nemo.utils.app_state import AppState
from proteonemo.models.bert_prot_model import BERTPROTModel
from proteonemo.data.prot_bert_dataset import BertInferencePreprocessedDataset
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from torch.utils.data import DataLoader
from nemo.utils import logging

assert torch.cuda.is_available()


@hydra_runner(config_path="../conf", config_name="bert_inference_from_preprocessed_config")
def main(cfg: DictConfig) -> None:
    torch.set_grad_enabled(False)
    logging.info(f'Config:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=True)],  **cfg.trainer)

    app_state = AppState()
    if cfg.trainer.gpus > 1:
        app_state.model_parallel_size = cfg.trainer.gpus
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = BERTPROTModel.restore_from(restore_path=cfg.model.nemo_path, trainer=trainer)
    model.freeze()
    dataset = BertInferencePreprocessedDataset(input_file=cfg.model.infer_ds.data_file)

    request_dl = DataLoader(dataset, 
        batch_size=cfg.model.infer_ds.batch_size,
        shuffle=cfg.model.infer_ds.shuffle,
        num_workers=cfg.model.infer_ds.num_workers)

    preds = trainer.predict(model, request_dl)

    if cfg.model.representations_path:
        for b, pred in enumerate(preds):
            for pred in preds:
                seq_names_batch, reprs_batch = pred
                seq_names_batch = seq_names_batch[0]
                i=0
                while i<len(seq_names_batch):
                    seq_name = seq_names_batch[i]
                    reprs = reprs_batch[i]
                    torch.save(reprs, f'{cfg.model.representations_path}/bert_preds/bert_reprs_{seq_name}.pt')
                    i+=1
                    

if __name__ == '__main__':
    main()
