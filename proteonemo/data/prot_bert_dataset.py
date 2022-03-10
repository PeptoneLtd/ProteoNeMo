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


import h5py
import numpy as np

from nemo.core.classes import Dataset

__all__ = ['BertInferencePreprocessedDataset']


def load_h5(input_file: str):
    return h5py.File(input_file, "r")


class BertInferencePreprocessedDataset(Dataset):
    """
    Dataset for already preprocessed data.
    """

    def __init__(self, input_file: str):
        """
        Args:
            input_file: data file in hdf5 format with preprocessed data in array format
        """
        self.input_file = input_file
        f = load_h5(input_file)
        num_keys = [
            'input_ids',
            'input_mask',
            'segment_ids',
        ]
        lit_keys = [
            'sequence_names',
        ]
        self.num_inputs = [np.asarray(f[key][:]) for key in num_keys]
        self.lit_inputs = [np.asarray(f[key][:]) for key in num_keys]
        for key in lit_keys:
            for seq_name in f[key][:]:
                seq_name_dec = seq_name.decode(encoding="utf-8")
                self.lit_inputs.append(seq_name_dec)
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.num_inputs[0])

    def __getitem__(self, index: int):
        [input_ids, input_mask, segment_ids] = [
            input[index].astype(np.int64) for input in self.num_inputs
        ]
        seq_names = [seq_name.decode(encoding="utf-8") for seq_name in self.lit_inputs]
      
        return (input_ids, segment_ids, input_mask, seq_names)

