import argparse
import numpy as np
import sys 
import os
import tqdm

import cmap_tools

# The main function produces distance and (optionally) contact maps based on pdbids. 
# Inputs to the save_contacts.py file are 
# - path to a .txt file that contains pdbids in a comma separated list;
# - path to an output folder, where results will be saved;
# - path to an mmcif_folder, where the corresponding mmcif files are stored;
# - contact map required - an optional Boolean argument
# Outputs of the main function are .npz files (one per pdbid) that contain 
# a dictionary with the following keys:
# - pdbid;
# - seq - protein sequence;
# - distance_matrix - a len(seq) x len(seq) numpy array with the distances;
# - optionally: contact_map - a len(seq) x len(seq) numpy array with 
# 1 at pos i, j if 0 < distance(residue(i), residue(j)) < 8. 
# 0 otherwise.   
    
args_parser = argparse.ArgumentParser(description='Save distance matrixes/contact maps for each pdbids separately in a txt file.')

# Add the arguments
args_parser.add_argument('-t',
                        '--txt_path',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-o',
                        '--output_folder',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-d',
                        '--mmcif_folder',
                        action='store',
                        required=True,
                        type=str
                        )

args_parser.add_argument('-c',
                        '--contact_required',
                        action='store_true')

# Execute the parse_args() method
args = args_parser.parse_args()
txt_path = args.txt_path
output_folder_path = args.output_folder
mmcif_folder = args.mmcif_folder
contact_required=args.contact_required

def main():

    pdbids = []
    with open(f'{txt_path}') as f:
        pdbids = f.readline().strip()

    pdbids = pdbids.split(',')

    pbar = tqdm.tqdm(total=len(pdbids), desc='PDBID: ', file=sys.stdout)
    for pdbid in pdbids:
        cmap_tools._save_distance_map(pdbid, output_folder_path, mmcif_folder, contact_required)
        pbar.update(1)

if __name__ == "__main__":
    main()