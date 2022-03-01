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

to_download=${1:-"uniref_50_only"}

#Download
if [ "$to_download" = "uniref_all" ] ; then
    python3 /proteonemo/preprocessing/bertPrep.py --action download --dataset uniref_90
    python3 /proteonemo/preprocessing/bertPrep.py --action download --dataset uniref_100
fi

python3 /proteonemo/preprocessing/bertPrep.py --action download --dataset uniref_50

# Properly format the text files
if [ "$to_download" = "wiki_books" ] ; then
    python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset bookscorpus
fi
python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

if [ "$to_download" = "uniref_all" ] ; then
    DATASET="uniref_50_90_100"
else
    DATASET="uniref_50"
    # Shard the text files
fi

# Shard the text files
python3 /proteonemo/preprocessing/bertPrep.py --action sharding --dataset $DATASET

# Create HDF5 files Phase 1
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset $DATASET --max_seq_length 128 \
--max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

# Create HDF5 files Phase 2
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset $DATASET --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
