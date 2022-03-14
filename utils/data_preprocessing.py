import re
import numpy as np
import torch
import nemo
import pandas as pd
import os
import sys
import itertools
import collections
import h5py
from sklearn.model_selection import KFold
import argparse

args_parser = argparse.ArgumentParser(description='Get the CB513 json data path and the output folder path.')

# Add the arguments
args_parser.add_argument('-d',
                        '--data_path',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-o',
                        '--output_folder',
                        action='store',
                        require=True,
                        type=str
                        )

args_parser.add_argument('-s',
                        '--shuffle',
                        action='store_true')

args_parser.add_argument('-rs',
                        '--random_seed',
                        action='store',
                        type=int
                        )

# Execute the parse_args() method
args = args_parser.parse_args()
cb513_json_path = args.data_path
output_folder_path = args.output_folder
shuffle_bool=args.shuffle
random_seed=args.random_seed
# add the shuffle and random seed arguments later

def preprocess_ssp(path_CB513_json):
    '''
    Function that takes the base dataset in .json format containing the CB513 SSP data and
    outputs a list - whose elements are tuples of the form ('pdbid', residue_pos, residue, Q8 class)
    This list can be used to create the train test splits in a flexible way
    '''

    df = pd.read_json(path_CB513_json)
    # drop duplicates
    df = df.drop_duplicates(subset=['pdbid'])

    def res_split(row):
        return list(zip(row['seq'], row['Q8']))

    df['comb_split'] = df.apply(res_split, axis=1)

    # collect the necessary inputs and outputs in tuples of
    # (pdbid, residue_pos, residue, Q8 class)
    # -----------------------------------------------------
    items = []

    for i in range(len(df)):
        pdbid = df.iloc[i]['pdbid']
        for pos, res in enumerate(df.iloc[i]['comb_split']):
            items.append((pdbid, pos, res[0], res[1]))

    return items

def _serve_features(fold):
    '''
    Prepares the data for a set in the right format ready for writing
    '''
    features = collections.OrderedDict()

    num_items = len(fold)

    features["pdbid"] = np.empty([num_items, 1], dtype=h5py.special_dtype(vlen=str))
    features["res_pos"] = np.zeros([num_items, 1], dtype="int32")
    features["res"] = np.empty([num_items, 1], dtype=h5py.special_dtype(vlen=str))
    features["q8"] = np.zeros([num_items, 1], dtype="int32")

    for item_index, item in enumerate(fold):
        features["pdbid"][item_index] = item[0]
        features["res_pos"][item_index] = item[1]
        features["res"][item_index] = item[2]
        features["q8"][item_index] = item[3]

    return features

def _save_features(file_path, features_to_save):
    '''
    Saves the data given in the 'features_to_save' argument as an .hdf5 file
    '''
    # saving data
    f = h5py.File(file_path, 'w')
    f.create_dataset("pdbid", data=features_to_save["pdbid"], dtype=h5py.special_dtype(vlen=str), compression='gzip')
    f.create_dataset("res_pos", data=features_to_save["res_pos"], dtype='i4', compression='gzip')
    f.create_dataset("res", data=features_to_save["res"], dtype=h5py.special_dtype(vlen=str), compression='gzip')
    f.create_dataset("q8", data=features_to_save["q8"], dtype='i4', compression='gzip')
    f.flush()
    f.close()

def write_folds(path_CB513_json, output_folder, shuffle=True, random_seed=None):
    '''
    Writes the 5 folds into training and test hdf5 files and saves them in the
    given output_folder
    '''
    # get the items first
    items = preprocess_ssp(path_CB513_json)

    if shuffle:
        np.random.shuffle(items)

    # split to 5-folds
    kf = KFold(n_splits=5, random_state=random_seed)

    for i, indeces in enumerate(kf.split(items)):
        train_index , test_index = indeces

        train_set = [items[k] for k in train_index]
        test_set = [items[k] for k in test_index]

        features_train = _serve_features(train_set)
        features_test = _serve_features(test_set)

        print(f'Saving fold {i} - test/train sets...')
        _save_features(f'{output_folder}/CB513_training_fold_{i}.hdf5', features_train)
        _save_features(f'{output_folder}/CB513_test_fold_{i}.hdf5', features_test)

def main():
    write_folds(cb513_json_path, output_folder_path, shuffle=shuffle_bool, random_seed=random_seed)

if __name__ == "__main__":
    main()