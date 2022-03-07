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

"""Create masked LM/next sentence masked_lm TF examples for BERT."""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import h5py
import numpy as np
from tqdm import tqdm, trange

from proteonemo.preprocessing.tokenization import ProteoNeMoTokenizer
from proteonemo.preprocessing import tokenization as tokenization

import random
import collections




class TrainingInstance(object):
  """A single training instance."""

  def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
    self.tokens = tokens
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_file(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file):
  """Create TF example files from `TrainingInstance`s."""
 

  total_written = 0
  features = collections.OrderedDict()
 
  num_instances = len(instances)
  features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
  features["masked_lm_positions"] =  np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
  features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
  features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")


  for inst_index, instance in enumerate(tqdm(instances)):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)
    
    features["input_ids"][inst_index] = input_ids
    features["input_mask"][inst_index] = input_mask
    features["masked_lm_positions"][inst_index] = masked_lm_positions
    features["masked_lm_ids"][inst_index] = masked_lm_ids

    total_written += 1
 
  print("saving data")
  f= h5py.File(output_file, 'w')
  f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
  f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
  f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
  f.create_dataset("masked_lm_positions", data=features["masked_lm_positions"], dtype='i4', compression='gzip')
  f.create_dataset("masked_lm_ids", data=features["masked_lm_ids"], dtype='i4', compression='gzip')
  f.create_dataset("next_sentence_labels", data=features["next_sentence_labels"], dtype='i1', compression='gzip')
  f.flush()
  f.close()

def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw sequences."""
  all_sequences = []

  # Input file format: 
  # One protein per line.
  for input_file in input_files:
    print("creating instance from {}".format(input_file))
    with open(input_file, "r") as reader:
      for dirty_line in reader:
        line = tokenization.convert_to_unicode(dirty_line)
        line = line.strip()
        tokens = tokenizer.tokenize(line)
        all_sequences.append(tokens)

  rng.shuffle(all_sequences)

  vocab_residues = list(tokenizer.small_vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for sequence_index in range(len(all_sequences)):
      instances.append(
          create_instance_from_sequence(
              all_sequences, sequence_index, max_seq_length,
              masked_lm_prob, max_predictions_per_seq, vocab_residues, rng))

  rng.shuffle(instances)
  return instances


def create_instance_from_sequence(
    all_sequences, sequence_index, max_seq_length,
    masked_lm_prob, max_predictions_per_seq, vocab_residues, rng):
  """Creates `TrainingInstance`s for a single document."""
  sequence = all_sequences[sequence_index]
 
  tokens = sequence[:max_seq_length]
  tokens[0] = '[CLS]'
  tokens[-1] = '[SEP]'
  (tokens, masked_lm_positions,
      masked_lm_labels) = create_masked_lm_predictions(
  tokens, masked_lm_prob, max_predictions_per_seq, vocab_residues, rng)
  instance = TrainingInstance(
    tokens=tokens,
    masked_lm_positions=masked_lm_positions,
    masked_lm_labels=masked_lm_labels)

  return instance


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_residues, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_residues[rng.randint(0, len(vocab_residues) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary the BERT model will train on.")
    parser.add_argument("--small_vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary used for the masking procedure.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")

    ## Other parameters

    # str
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    #int 
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--dupe_factor",
                        default=5,
                        type=int,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--max_predictions_per_seq",
                        default=160,
                        type=int,
                        help="Maximum sequence length.")
                             

    # floats

    parser.add_argument("--masked_lm_prob",
                        default=0.15,
                        type=float,
                        help="Masked LM probability.")

    parser.add_argument("--do_upper_case",
                        action='store_true',
                        default=True,
                        help="Whether to upper case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    args = parser.parse_args()

    tokenizer = ProteoNeMoTokenizer(args.vocab_file, args.small_vocab_file, do_upper_case=args.do_upper_case, 
      max_len=args.max_seq_length)
    
    input_files = []
    if os.path.isfile(args.input_file):
      input_files.append(args.input_file)
    elif os.path.isdir(args.input_file):
      input_files = [os.path.join(args.input_file, f) for f in os.listdir(args.input_file) if (os.path.isfile(os.path.join(args.input_file, f)) and f.endswith('.txt') )]
    else:
      raise ValueError("{} is not a valid path".format(args.input_file))

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, args.max_seq_length, args.dupe_factor,
         args.masked_lm_prob, args.max_predictions_per_seq, rng)

    output_file = args.output_file


    write_instance_to_example_file(instances, tokenizer, args.max_seq_length,
                                    args.max_predictions_per_seq, output_file)


if __name__ == "__main__":
    main()
