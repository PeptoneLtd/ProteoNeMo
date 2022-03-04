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

from collections import defaultdict
from itertools import islice

import multiprocessing
import statistics
from Bio import SeqIO
import numpy as np

class Sharding:
    def __init__(self, input_files, output_name_prefix, n_training_shards, n_test_shards, fraction_test_set, rng):
        assert len(input_files) > 0, 'The input file list must contain at least one file.'
        assert n_training_shards > 0, 'There must be at least one output shard.'
        assert n_test_shards > 0, 'There must be at least one output shard.'

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set

        self.input_files = input_files
        self.rng = rng

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = '_training'
        self.output_test_identifier = '_test'
        self.output_file_extension = '.txt'

        self.output_training_files = defaultdict(list)    # key: filename, value: list of protein sequences to go into file
        self.output_test_files = defaultdict(list)  # key: filename, value: list of protein sequences to go into file


    def load_fastas(self):
        training_shards = np.arange(self.n_training_shards)
        test_shards = np.arange(self.n_test_shards)

        for input_file in self.input_files:
            print('input file:', input_file)
            with open(input_file) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    if self.rng.random() < self.fraction_test_set:
                        test_key = self.rng.choice(test_shards)
                        self.output_test_files[test_key].append(record.seq) 
                    else:
                        training_key = self.rng.choice(training_shards)
                        self.output_training_files[training_key].append(record.seq) 

        for shard in self.output_training_files.values():
            if not shard:
                print('Warning: Too many training shard, reduce them.')

        for shard in self.output_test_files.values():
            if not shard:
                print('Warning: Too many test shard, reduce them.')

    def write_shards_to_disk(self):
        print('Start: Write Shards to Disk')
        for shard in self.output_training_files:
            shardname = f'{self.output_name_prefix}_{self.output_training_identifier}_{shard}{self.output_file_extension}'
            self.write_single_shard(shard_name, self.output_training_files[shard])

        for shard in self.output_test_files:
            shard_name = f'{self.output_name_prefix}_{self.output_test_identifier}_{shard}{self.output_file_extension}'
            self.write_single_shard(shard_name, self.output_test_files[shard])

        print('End: Write Shards to Disk')


    def write_single_shard(self, shard_name, shard):
        with open(shard_name, mode='w', newline='\n') as f:
            for protein in shard:
                f.write(str(protein) + '\n') # Line break between proteins


