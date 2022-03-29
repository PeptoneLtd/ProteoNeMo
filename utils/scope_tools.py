# Some helper to interpret scope data 
# -----------------------------------
# sccs - (SCOP(e) concise classification string. This is a dot notation used to concisely 
# describe a SCOP(e) class, fold, superfamily, and family. For example, a.39.1.1 references 
# the "Calbindin D9K" family, where "a" represents the class, "39" represents the fold, 
# "1" represents the superfamily, and the last "1" represents the family.
# 
# sid - Stable domain identifier. A 7-character sid consists of "d" followed by the 4-character PDB ID of the file of origin, 
# the PDB chain ID ('_' if none, '.' if multiple as is the case in genetic domains), and a single character (usually an integer) 
# if needed to specify the domain uniquely ('_' if not). Sids are currently all lower case, even when the chain letter is upper case. 
# Example sids include d4akea1, d9hvpa_, and d1cph.1.
#
# sunid - SCOP(e) unique identifier. This is simply a number that may be used to reference any entry in the SCOP(e) hierarchy,
# from root to leaves (Fold, Superfamily, Family, etc.).

"""Functions for getting scope items using pdbids only"""
import abc
import dataclasses
import datetime
import functools
import collections
import glob
from lib2to3.pgen2.token import OP
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, NoReturn
from os.path import basename
from xmlrpc.client import Boolean

from absl import logging
import numpy as np
from cmap_tools import Error


class ScopeFileError(Error):
  """An error indicating that a given scope file does not exists in the given folder."""


def _filter_scope(
    x: List) -> List:
    """ returns a formatted scope item - it is of the form (pdbid, sccs, description, sunid, sid)"""
    return [x[0][1:5]+x[0][5].upper(), x[3], x[2], float(x[4]), x[0]]


def _item_finder(
    x: List, 
    targets: List) -> Boolean:
    """ Checks if any of the pdbids matches a scope entry. """
    if x[0][1:5]+x[0][5].upper() in targets:
        return True
    elif x[0][5]=='.':
        chains = [c[0] for c in x[2].split(',')]
        _pdb_w_chains = [x[0][1:5]+c for c in chains]
        if not set(_pdb_w_chains).isdisjoint(targets): 
            return True
        else:
            return False 
    else:
        return False


def _get_scope_items(
    pdbids: List[str],
    scope_data_file: str) -> List[Tuple[str, str, str, float, str]]:
    """ Returns a scope item - if exists in the scope dataset - based on a pdbid. 
        Outputs will be blank apart from the pdbid, if the item was not found in scope.
        Inputs: - pdbids,
                - path to the scope database named: 'dir.cla.scope.2.08-stable.txt'
        Outputs: - pdbid, 
                 - sccs: (SCOP(e) concise classification string. This is a dot notation used to concisely 
                 describe a SCOP(e) class, fold, superfamily, and family. For example, a.39.1.1 references 
                 the "Calbindin D9K" family, where "a" represents the class, "39" represents the fold, 
                 "1" represents the superfamily, and the last "1" represents the family.)
                 - description: usually contains chain and residue specific information
                 - sunid: a particular scope id that can be used for further search, 
                 - sid: another unique scope id that can be used for. """
    if not os.path.exists(scope_data_file):
        raise ScopeFileError(f'The file {scope_data_file} does not exists.')

    # read the scope .txt
    lines = []
    with open(f'{scope_data_file}', 'r') as f:
        for line in f.readlines():
            if line[0]!='#':
                lines.append(line.replace('\n','').split('\t'))
    
    filtered_lines = list(map(lambda x: _item_finder(x, pdbids), lines))
    found = list(map(_filter_scope, np.take(lines, np.where(np.array(filtered_lines).astype(float)==1.), axis=0)[0]))
    not_found = [[pdbid, '-', '-', '-', '-'] for pdbid in pdbids if pdbid not in set([x[0] for x in found])]

    return found+not_found
