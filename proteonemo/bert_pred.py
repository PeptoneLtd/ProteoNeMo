import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from pytorch_lightning.plugins import DDPPlugin
from nemo.utils.app_state import AppState
from bert_prot_model import BERTPROTModel
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import BertPretrainingPreprocessedDataset
from torch.utils.data import DataLoader
from nemo.utils import logging

assert torch.cuda.is_available()


@hydra_runner(config_path="conf", config_name="bert_inference_from_preprocessed_config")
def main(cfg: DictConfig) -> None:
    torch.set_grad_enabled(False)
    logging.info(f'Config:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=True)],  **cfg.trainer)
    app_state = AppState()
    model = BERTPROTModel.restore_from(restore_path=cfg.model.nemo_path, trainer=trainer)
    model.freeze()
    dataset = BertPretrainingPreprocessedDataset(input_file=cfg.model.infer_ds.data_file, 
        max_predictions_per_seq=cfg.model.infer_ds.max_predictions_per_seq)

    request_dl = DataLoader(dataset, 
        batch_size=cfg.model.infer_ds.batch_size,
        shuffle=cfg.model.infer_ds.shuffle,
        num_workers=cfg.model.infer_ds.num_workers)

    preds = trainer.predict(model, request_dl)

    if cfg.model.representations_path:
        i=0
        for pred in preds:
            for sequence in pred:
                torch.save(sequence, f'{cfg.model.representations_path}/bert_results_{i}.pt')
                i+=1


if __name__ == '__main__':
    main()
