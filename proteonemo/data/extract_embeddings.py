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


import collections
from proteonemo.preprocessing import tokenization as tokenization
from Bio import SeqIO
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path


class ExtractEmbeddings:
    """
    Extract residue level representation.
    """

    def __init__(self, input_files, output_file, tokenizer, max_seq_length):
        """
        Args:
            input_files: data files in .fasta format, can be a directory or a single file
            output_file: .hdf5 file output file
            tokenizer: tokenizer to be applied on the input
            max_seq_length: The maximum total input sequence length
        """
        self.input_files = input_files
        self.output_file = output_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


    def load_fasta_files(self):
        instances = collections.OrderedDict()
        for input_file in self.input_files:
            print('input file:', input_file)
            with open(input_file) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequence = tokenization.convert_to_unicode(str(record.seq))
                    sequence = sequence.strip()
                    tokens = self.tokenizer.tokenize(sequence)
                    tokens = tokens[:self.max_seq_length]
                    tokens.insert(0, '[CLS]')
                    tokens.insert(-1, '[SEP]')
                    p_name = str(record.id)
                    p_name = p_name.replace(" ", "_")
                    p_name = p_name.replace("|", "-")
                    try:
                        instances[p_name] = tokens
                    except:
                        print(f'{p_name} is a duplicate, taking into account only the first record')
                        pass
        return instances


    def write_instance_to_example_file(self, instances):
        """
        Args:
            instances: dict with key as protein sequence name and value as token list
        """
        

        total_written = 0
        features = collections.OrderedDict()
        
        num_instances = len(instances)
        features["input_ids"] = np.zeros([num_instances, self.max_seq_length], dtype="int32")
        features["input_mask"] = np.zeros([num_instances, self.max_seq_length], dtype="int32")
        features["segment_ids"] = np.zeros([num_instances, self.max_seq_length], dtype="int32")
        features["sequence_names"] =  list(instances.keys())


        for inst_index, (p_name, tokens) in enumerate(tqdm(instances.items())):
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            assert len(input_ids) <= self.max_seq_length

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            
            features["input_ids"][inst_index] = input_ids
            features["input_mask"][inst_index] = input_mask
            features["sequence_names"][inst_index] = p_name

            total_written += 1
        
        print("saving data")
        output_file = Path(self.output_file)
        output_file.touch(exist_ok=True)

        f= h5py.File(output_file, 'w')
        f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
        f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
        f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
        f.create_dataset("sequence_names", data=features["sequence_names"], compression='gzip')
        f.flush()
        f.close()