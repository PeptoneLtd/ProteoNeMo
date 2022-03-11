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


import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from proteonemo.data.prot_bert_dataset import BertInferencePreprocessedDataset
from proteonemo.data.extract_embeddings import ExtractEmbeddings
from proteonemo.preprocessing.tokenization import ProteoNeMoTokenizer
from proteonemo.models.bert_prot_model import BERTPROTModel
from nemo.utils.app_state import AppState
import numpy as np
import os
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank

assert torch.cuda.is_available()


def get_representations(preds, output_file):
    for pred in preds:
        seq_names_batch, reprs_batch, masks_batch = pred
        seq_names_batch = seq_names_batch[0]
        i=0
        while i<len(seq_names_batch):
            seq_name = seq_names_batch[i]
            reprs = reprs_batch[i]
            mask = masks_batch[i]
            # take only residue level representations
            last_element = np.where(mask==1)[0][-1]
            clean_reprs = reprs[1:last_element]
            torch.save(clean_reprs, f'{output_file}/bert_reprs_{seq_name}.pt')
            i+=1


def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary the BERT model will evaluated on.")
    parser.add_argument("--small_vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary used for the masking procedure.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input evaluation dataset. can be directory with .fasta files or a path to a single file")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the residue level representations will be written.")
    parser.add_argument("--data_file",
                        default='../input_sequences.hdf5',
                        type=str,
                        required=True,
                        help="The .hdf5 file containing the tokenized .fasta inputs.")                    
    parser.add_argument("--model_file", 
                        type=str, 
                        default="", 
                        required=True, 
                        help="Pass path to model's .nemo file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="The maximum total input sequence length. \n"
                             "Sequences longer than this will be truncated.")                             
    parser.add_argument("--do_upper_case",
                        action='store_true',
                        default=True,
                        help="Whether to upper case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--tensor_model_parallel_size", 
                        type=int, 
                        default=1, 
                        required=True,
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=8, 
                        required=True,
    )
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=8, 
                        required=True,
    )

    args = parser.parse_args()

    tokenizer = ProteoNeMoTokenizer(args.vocab_file, args.small_vocab_file, do_upper_case=args.do_upper_case, 
      max_len=args.max_seq_length)

    input_files = []
    if os.path.isfile(args.input_file):
      input_files.append(args.input_file)
    elif os.path.isdir(args.input_file):
      input_files = [os.path.join(args.input_file, f) for f in os.listdir(args.input_file) if (os.path.isfile(os.path.join(args.input_file, f)) and f.endswith('.fasta') )]
    else:
      raise ValueError("{} is not a valid path".format(args.input_file))

    instances = ExtractEmbeddings.load_fasta_files(input_files, tokenizer, args.max_seq_length)
    ExtractEmbeddings.write_instance_to_example_file(instances, tokenizer, args.max_seq_length,
                                    args.data_file)

    torch.set_grad_enabled(False)
    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=True)], devices=args.tensor_model_parallel_size, precision=16, accelerator='gpu')

    app_state = AppState()
    if args.tensor_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = BERTPROTModel.restore_from(restore_path=args.model_file, trainer=trainer)
    model.freeze()
    dataset = BertInferencePreprocessedDataset(input_file=args.data_file)

    request_dl = DataLoader(dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    preds = trainer.predict(model, request_dl)
    get_representations(preds, args.output_file)


if __name__ == '__main__':
    main()