import argparse
import numpy as np
import sys 
import os
import csv

import scope_tools

# The main function saves, in .csv format, the output of the function 'get_scope_items' defined in scope_tools.
# Inputs to the save_scope_queries.py file are 
# - path to a .txt file that contains pdbids in a comma separated list;
# - path to an output folder, where results will be saved;
# - path to the scope data file 
# Outputs of the main function is a single .csv file containing all the pdbids in the input txt 
    
args_parser = argparse.ArgumentParser(description='Save scope queries based on an set of pdbids')

# Add the arguments
args_parser.add_argument('-p',
                        '--pdbids_txt_path',
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

args_parser.add_argument('-s',
                        '--scope_file',
                        action='store',
                        required=True,
                        type=str
                        )

# Execute the parse_args() method
args = args_parser.parse_args()
pdbids_path = args.pdbids_txt_path
output_folder_path = args.output_folder
scope_file = args.scope_file


def main():

    pdbids = []
    with open(f'{pdbids_path}') as f:
        pdbids = f.readline().strip()

    pdbids = pdbids.split(',')
    output_to_save = scope_tools._get_scope_items(pdbids, scope_file)

    header = ['pdbid', 'sccs', 'description', 'sunid', 'sid']

    with open(f'{output_folder_path}', 'wt', newline ='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(i for i in header)
        for j in output_to_save:
            writer.writerow(j)

    print(f'Results are saved in {output_folder_path}.')

if __name__ == "__main__":
    main()