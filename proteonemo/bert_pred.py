import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DDPPlugin
#from pytorch_lightning.trainer.trainer import Trainer
from nemo.utils.app_state import AppState
#from nemo.collections.nlp.models.language_modeling import BERTLMModel
from bert_prot_model import BERTPROTModel
import pickle as pkl
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import BertPretrainingPreprocessedDataset, BertPretrainingPreprocessedDataloader
from torch.utils.data import DataLoader, TensorDataset

assert torch.cuda.is_available()
torch.set_grad_enabled(False)

trainer = pl.Trainer(plugins=DDPPlugin(find_unused_parameters=True), gpus=1, fast_dev_run=False)
app_state = AppState()
model = BERTPROTModel.restore_from(restore_path='bert_base_wikipedia.nemo', trainer=trainer)
model.freeze()
base_path = "/workspace/nemo/WIKIPEDIA/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en"
dataset = BertPretrainingPreprocessedDataset(input_file=f"{base_path}/wikicorpus_en_training_0.hdf5", max_predictions_per_seq=80)
input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = dataset.inputs

params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 8}
request_dl = DataLoader(dataset, **params)

n_toks = 512
hidden_dim = 768

preds = trainer.predict(model, request_dl)
embeddings = torch.cat(preds, 1) 
embeddings = embeddings.view(-1, n_toks, hidden_dim) # todo: use the nemo functionality
torch.save(embeddings.clone(), 'bert_results.pt')