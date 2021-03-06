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

from proteonemo.preprocessing.downloader import Downloader
from proteonemo.preprocessing.protein_sharding import Sharding

import argparse
import os
import pprint
import subprocess
import random


def main(args):
    working_dir = os.environ['BERT_PREP_WORKING_DIR']

    print('Working Directory:', working_dir)
    print('Action:', args.action)
    print('Dataset Name:', args.dataset)

    if args.input_files:
        args.input_files = args.input_files.split(',')

    hdf5_tfrecord_folder_prefix = "_seq_len_" + str(args.max_seq_length) \
                                  + "_max_pred_" + str(args.max_predictions_per_seq) + "_masked_lm_prob_" + str(args.masked_lm_prob) \
                                  + "_random_seed_" + str(args.random_seed) + "_dupe_factor_" + str(args.dupe_factor)

    directory_structure = {
        'download' : working_dir + '/download',    # Downloaded and decompressed
        'sharded' : working_dir + '/sharded_' + "training_shards_" + str(args.n_training_shards) + "_test_shards_" + str(args.n_test_shards) + "_fraction_" + str(args.fraction_test_set),
        'hdf5': working_dir + '/hdf5' + hdf5_tfrecord_folder_prefix
    }

    print('\nDirectory Structure:')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(directory_structure)
    print('')

    if args.action == 'download':
        if not os.path.exists(directory_structure['download']):
            os.makedirs(directory_structure['download'])

        downloader = Downloader(args.dataset, directory_structure['download'])
        downloader.download()

    elif args.action == 'sharding':
        # Note: uniref 50+90+100 requires user to provide list of input_files (comma-separated with no spaces)
        if args.dataset == 'uniparc' or 'uniref' in args.dataset or 'uniprotkb' in args.dataset:
            if args.input_files is None:
                if args.dataset == 'uniparc':
                    args.input_files = [directory_structure['download'] + '/uniparc/uniparc_active.fasta']
                elif args.dataset == 'uniref_50':
                    args.input_files = [directory_structure['download'] + '/uniref_50/uniref50.fasta']
                elif args.dataset == 'uniref_90':
                    args.input_files = [directory_structure['download'] + '/uniref_90/uniref90.fasta']
                elif args.dataset == 'uniref_100':
                    args.input_files = [directory_structure['download'] + '/uniref_100/uniref100.fasta']
                elif args.dataset == 'uniref_all':
                    args.input_files = [directory_structure['download'] + '/uniref_50/uniref50.fasta',
                        directory_structure['download'] + '/uniref_90/uniref90.fasta',
                        directory_structure['download'] + '/uniref_100/uniref100.fasta']
                elif args.dataset == 'uniprotkb_swissprot':
                    args.input_files = [directory_structure['download'] + '/uniprotkb_swissprot/uniprot_sprot.fasta']
                elif args.dataset == 'uniprotkb_trembl':
                    args.input_files = [directory_structure['download'] + '/uniprotkb_trembl/uniprot_trembl.fasta']
                elif args.dataset == 'uniprotkb_isoformseqs':
                    args.input_files = [directory_structure['download'] + '/uniprotkb_isoformseqs/uniprot_sprot_varsplic.fasta']
                elif args.dataset == 'uniprotkb_all':
                    args.input_files = [directory_structure['download'] + '/uniprotkb_swissprot/uniprot_sprot.fasta',
                        directory_structure['download'] + '/uniprotkb_trembl/uniprot_trembl.fasta',
                        directory_structure['download'] + '/uniprotkb_isoformseqs/uniprot_sprot_varsplic.fasta']

            output_file_prefix = directory_structure['sharded'] + '/' + args.dataset + '/' + args.dataset

            if not os.path.exists(directory_structure['sharded']):
                os.makedirs(directory_structure['sharded'])

            if not os.path.exists(directory_structure['sharded'] + '/' + args.dataset):
                os.makedirs(directory_structure['sharded'] + '/' + args.dataset)

            rng = random.Random(args.random_seed)
            sharding = Sharding(args.input_files, output_file_prefix, args.n_training_shards, args.n_test_shards, args.fraction_test_set, rng)
            sharding.load_fastas()
            sharding.write_shards_to_disk()

        else:
            assert False, 'Unsupported dataset for sharding'

    elif args.action == 'create_hdf5_files':
        last_process = None

        if not os.path.exists(directory_structure['hdf5'] + "/" + args.dataset):
            os.makedirs(directory_structure['hdf5'] + "/" + args.dataset)

        def create_record_worker(filename_prefix, shard_id, output_format='hdf5'):
            bert_preprocessing_command = 'python ../proteonemo/preprocessing/create_pretraining_data.py'
            bert_preprocessing_command += ' --input_file=' + directory_structure['sharded'] + '/' + args.dataset + '/' + filename_prefix + '_' + str(shard_id) + '.txt'
            bert_preprocessing_command += ' --output_file=' + directory_structure['hdf5'] + '/' + args.dataset + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
            bert_preprocessing_command += ' --vocab_file=' + args.vocab_file
            bert_preprocessing_command += ' --small_vocab_file=' + args.small_vocab_file
            bert_preprocessing_command += ' --do_upper_case' if args.do_upper_case else ''
            bert_preprocessing_command += ' --max_seq_length=' + str(args.max_seq_length)
            bert_preprocessing_command += ' --max_predictions_per_seq=' + str(args.max_predictions_per_seq)
            bert_preprocessing_command += ' --masked_lm_prob=' + str(args.masked_lm_prob)
            bert_preprocessing_command += ' --random_seed=' + str(args.random_seed)
            bert_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)
            bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

            last_process = bert_preprocessing_process

            # This could be better optimized (fine if all take equal time)
            if shard_id % args.n_processes == 0 and shard_id > 0:
                bert_preprocessing_process.wait()
            return last_process

        output_file_prefix = args.dataset

        for i in range(args.n_training_shards):
            last_process = create_record_worker(output_file_prefix + '_training', i)

        last_process.wait()

        for i in range(args.n_test_shards):
            last_process = create_record_worker(output_file_prefix + '_test', i)

        last_process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything BERT-related'
    )

    parser.add_argument(
        '--action',
        type=str,
        help='Specify the action you want the app to take. e.g., download data, or create hdf5 files',
        choices={
            'download',               # Download and verify mdf5/sha sums
            'sharding',               # Convert previous formatted text into shards containing one sentence per line
            'create_hdf5_files'       # Turn each shard into a HDF5 file with masking and next sentence prediction info
        }
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
        choices={
            'uniref_50',
            'uniref_90',
            'uniref_100',
            'uniref_all',
            'uniprotkb_swissprot',
            'uniprotkb_trembl',
            'uniprotkb_isoformseqs',
            'uniprotkb_all',
            'uniparc',
            'all'
        }
    )

    parser.add_argument(
        '--input_files',
        type=str,
        help='Specify the input files in a comma-separated list (no spaces)'
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=256
    )

    parser.add_argument(
        '--n_test_shards',
        type=int,
        help='Specify the number of test shards to generate',
        default=256
    )

    parser.add_argument(
        '--fraction_test_set',
        type=float,
        help='Specify the fraction (0..1) of the data to withhold for the test data split (based on number of sequences)',
        default=0.1
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=4
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Specify the base seed to use for any random number generation',
        default=12345
    )

    parser.add_argument(
        '--dupe_factor',
        type=int,
        help='Specify the duplication factor',
        default=5
    )

    parser.add_argument(
        '--masked_lm_prob',
        type=float,
        help='Specify the probability for masked lm',
        default=0.15
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=1024
    )

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        help='Specify the maximum number of masked words per sequence',
        default=160
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    parser.add_argument(
        '--small_vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    parser.add_argument("--do_upper_case",
        action='store_true',
        default=True,
        help="Whether to upper case the input text. True for uncased models, False for cased models.")

    args = parser.parse_args()
    main(args)
