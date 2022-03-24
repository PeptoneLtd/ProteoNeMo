import argparse
import numpy as np
import sys 
import os

import cmap_tools

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

    for pdbid in pdbids:
        cmap_tools._save_distance_map(pdbid, output_folder_path, mmcif_folder, contact_required)


if __name__ == "__main__":
    main()