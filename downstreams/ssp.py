from pyexpat import model
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
import csv
from sklearn.model_selection import KFold
from sklearn import linear_model
import argparse
import time


args_parser = argparse.ArgumentParser(description='Get the path to the hdf5 files and the path for logging.')

# Add the arguments
args_parser.add_argument('-d',
                        '--data_path',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-r',
                        '--repr_path',
                        action='store',
                        required=True,
                        type=str
                        )
args_parser.add_argument('-m',
                        '--model',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-log',
                        '--logging_folder',
                        help='See if this works...',
                        type=str
                        #action='store_true'
                        )


# Execute the parse_args() method
args = args_parser.parse_args()
cb513_data_path = args.data_path
repr_path = args.repr_path
model_name=args.model
logging_folder_path = args.logging_folder

def input_factory_from_hdf5(path_hdf5_file, path_repr, model='nemo'):
    exes = []
    embeddings = {}

    dset = h5py.File(path_hdf5_file, 'r')
    keys = list(dset.keys())
    assert set(['pdbid', 'q8', 'res', 'res_pos']) == set(keys)

    for element in list(zip(dset['pdbid'][...].astype("str").squeeze(), dset['res_pos'][...].squeeze())):
        pdbid, pos = element
        #label = dset['q8'][_].item()

        if pdbid in embeddings.keys():
            embedding = embeddings[pdbid][pos].numpy()
        else:
            if model=='nemo':
                embeddings[pdbid] = torch.load(f'{path_repr}bert_reprs_{pdbid}.pt').clone().cpu().detach()
            else:
                embeddings[pdbid] = torch.load(f'{path_repr}{pdbid}.pt')['representations'][33].clone().cpu().detach()
            embedding = embeddings[pdbid][pos].numpy()

        exes.append(embedding)

    return np.array(exes), dset['q8'][...].squeeze()

def _file_checks(f_names):

    things = [f_name.split('.')[0].split('_') for f_name in f_names]
    checker = {}

    for key, group in itertools.groupby(things, lambda x: x[-1]):
        for thing in group:
            if key in checker.keys():
                if 'training' in [x for x in thing] and 'training' not in checker[key]:
                    checker[key].append('training')
                elif 'test' in [x for x in thing] and 'test' not in checker[key]:
                    checker[key].append('test')
            else:
                if 'training' in [x for x in thing]:
                    checker[key] = ['training']
                elif 'test' in [x for x in thing]:
                    checker[key] = ['test']

    for key, values in checker.items():
        if 'training' not in values or 'test' not in values:
            print(f'Fold {key} is incomplete, i.e. either test or training file is missing.')
            return False

    return True


def main(path_folds, path_repr_folder, *logging_path, model='nemo'):
    f_names = next(os.walk(f'{path_folds}'))[2]
    f_names = [file for file in f_names if file.endswith(".hdf5")]

    assert _file_checks(f_names), 'One of the folds is incomplete, i.e. either test or training file is missing.'

    f_names_split = [f_name.split('.')[0].split('_') for f_name in f_names]
    folds = list(np.unique([x[-1] for x in f_names_split]))

    res = {}

    for fold in folds:
        print(f'Fold {fold} in progress... ')
        # find the right training and test file names
        for x in f_names:
            if fold in x.split('.')[0].split('_'):
                if 'training' in x.split('.')[0].split('_'):
                    training_file = x
                elif 'test' in x.split('.')[0].split('_'):
                    test_file = x

        # read and prepare the representation data for regression
        ex_train, zed_train = input_factory_from_hdf5(f'{path_folds}/{training_file}', path_repr_folder, model=model)
        ex_test, zed_test = input_factory_from_hdf5(f'{path_folds}/{test_file}', path_repr_folder, model=model)

        lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
        lm.fit(ex_train, zed_train)
        res[fold] = lm.score(ex_test, zed_test)
        print(lm.score(ex_test, zed_test))
        print("")


    _mean = np.mean(list(res.values()))
    _std = np.std(list(res.values()))
    res['mean'], res['std'] = _mean, _std

    if logging_path:
        with open(f'{logging_path[0]}', 'w') as f:
            w = csv.DictWriter(f, res.keys())
            w.writeheader()
            w.writerow(res)

        f.close()
        print(f'Results are saved in {logging_path[0]}')

    else:
        if not os.path.isdir(f'{os.getcwd()}/tmp-downstream/'):
            os.mkdir(f'{os.getcwd()}/tmp-downstream')

        f_name_tag = f'{model}-SSP-'+time.strftime('%Y%m%d-%H%M%S')
        with open(f'{os.getcwd()}/tmp-downstream/{f_name_tag}.csv', 'w') as f:
            w = csv.DictWriter(f, res.keys())
            w.writeheader()
            w.writerow(res)

        f.close()
        print(f'Results are saved in {os.getcwd()}/tmp-downstream/{f_name_tag}.csv')

if __name__ == "__main__":
    main(cb513_data_path, repr_path, logging_folder_path, model=model_name)
