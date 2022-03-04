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
    python3 bertPrep.py --action download --dataset uniref_90
    python3 bertPrep.py --action download --dataset uniref_100
elif [ "$to_download" = "uniparc" ] ; then
    python3 /proteonemo/preprocessing/bertPrep.py --action download --dataset uniparc
elif [ "$to_download" = "uniprotkb_all" ] ; then
    python3 bertPrep.py --action download --dataset uniprotkb_swissprot
    python3 bertPrep.py --action download --dataset uniprotkb_trembl
    python3 bertPrep.py --action download --dataset uniprotkb_isoformseqs
fi

python3 /proteonemo/preprocessing/bertPrep.py --action download --dataset uniref_50

if [ "$to_download" = "uniref_all" ] ; then
    DATASET="uniref_all"
elif [ "$to_download" = "uniparc" ] ; then
    DATASET="uniparc"
elif [ "$to_download" = "uniprotkb_all" ] ; then
    DATASET="uniprotkb_all"
else
    DATASET="uniref_50"
    # Shard the text files
fi

# Shard the text files
python3 bertPrep.py --action sharding --dataset $DATASET

# Create HDF5 files
python3 bertPrep.py --action create_hdf5_files --dataset $DATASET --max_seq_length 1024 \
--max_predictions_per_seq 160 --vocab_file vocab.txt --small_vocab_file vocab_small.txt --do_upper_case

