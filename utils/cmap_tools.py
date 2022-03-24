# Disclaimer - these functions were lifted from alphafold - we only use this functionality, not the whole package


"""Functions for getting contact map features."""
import abc
import dataclasses
import datetime
import functools
import collections
import glob
from lib2to3.pgen2.token import OP
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, NoReturn
from os.path import basename
from xmlrpc.client import Boolean

from MDAnalysis.analysis.distances import self_distance_array, distance_array

from absl import logging
import residue_constants
import mmcif_parsing
import parsers
import numpy as np
from jax import vmap


class Error(Exception):
  """Base class for exceptions."""


class CaDistanceError(Error):
  """An error indicating that a CA atom distance exceeds a threshold."""


class MultipleChainsError(Error):
  """An error indicating that multiple chains were found for a given ID."""



def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
    """Checks if the distance between unmasked neighbor residues is ok."""
    ca_position = residue_constants.atom_order['CA']
    prev_is_unmasked = False
    prev_calpha = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_is_unmasked = bool(mask[ca_position])
        if this_is_unmasked:
            this_calpha = coords[ca_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_calpha - prev_calpha)
                if distance > max_ca_ca_distance:
                    raise CaDistanceError(
                        'The distance between residues %d and %d is %f > limit %f.' % (
                          i, i + 1, distance, max_ca_ca_distance))
            prev_calpha = this_calpha
        prev_is_unmasked = this_is_unmasked

def _get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gets atom positions and mask from a list of Biopython Residues."""
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

    relevant_chains = [c for c in mmcif_object.structure.get_chains()
                     if c.id == auth_chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
                f'Expected exactly one chain in structure with id {auth_chain_id}.')
    chain = relevant_chains[0]

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)
    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[(res_at_position.hetflag,
                         res_at_position.position.residue_number,
                         res_at_position.position.insertion_code)]
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                if atom_name in residue_constants.atom_order.keys():
                    pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
                    # Put the coordinates of the selenium atom in the sulphur column.
                    pos[residue_constants.atom_order['SD']] = [x, y, z]
                    mask[residue_constants.atom_order['SD']] = 1.0

            # Fix naming errors in arginine residues where NH2 is incorrectly
            # assigned to be closer to CD than NH1.
            cd = residue_constants.atom_order['CD']
            nh1 = residue_constants.atom_order['NH1']
            nh2 = residue_constants.atom_order['NH2']
            if (res.get_resname() == 'ARG' and
                all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
                (np.linalg.norm(pos[nh1] - pos[cd]) >
                 np.linalg.norm(pos[nh2] - pos[cd]))):
                pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
                mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    _check_residue_distances(all_positions, all_positions_mask, max_ca_ca_distance)
    return all_positions, all_positions_mask


def _get_distance_map(
    cif_folder: str, 
    pdbid: str) -> Tuple[str, np.ndarray]:
    """Gets sequence and distance matrix"""
    pdb_id, chain_id = pdbid[:-1], pdbid[-1]
    
    cif_path = f'{cif_folder}/{pdb_id}.cif'
    cif_string = open(cif_path, 'r').read()

    parsed = mmcif_parsing.parse(file_id=basename(cif_path), mmcif_string=cif_string)
    chains = list(parsed.mmcif_object.chain_to_seqres.keys())

    if chain_id not in chains:
        raise MultipleChainsError(
            f'Chain {chain_id} could not be found in {pdb_id}.'
        )

    all_atom_positions, all_atom_mask = _get_atom_positions(
            parsed.mmcif_object, chain_id, max_ca_ca_distance=150.0)

    seq = parsed.mmcif_object.chain_to_seqres[chain_id]

    # Get CB (or in glycine case CA) atom position
    is_glycine = (np.array(list(seq))=='G').astype(int) # 0 for not glycine -> 'CB', 1 for glycine -> 'CA'
    mapping = np.array([residue_constants.atom_order[k] for k in ['CB', 'CA']])
    atom_type_index = mapping[is_glycine]
    contact_positions = vmap(lambda x,y: x[y])(all_atom_positions, atom_type_index)

    # distance map
    xyz = np.array(contact_positions)
    distance_map = distance_array(xyz,xyz)
    
    return seq, distance_map


def _save_distance_map(
    pdbid: str,
    save_path: str,
    cif_path: str,
    contact_map_required: Optional[Boolean]=False
) -> NoReturn:
    """Saves a dictionary in an .npz format. Keys are pdbid, seq, distance_map."""
    # initialize a container
    features = collections.OrderedDict()
            
    _id_save, _id_query = pdbid, pdbid 
           
    if _id_query in residue_constants.cif_superseeds.keys():
        _id_query = residue_constants.cif_superseeds[_id_query]
                
            
    seq, dist_mat = _get_distance_map(f'{cif_path}', _id_query)
            
    features['pdbid'] = _id_query
    features['seq'] = seq
    features['distance_matrix'] = dist_mat

    if contact_map_required:
        # contact map thresholding is at 8A
        features['contact_map'] = np.array((dist_mat > 0.) & (dist_mat < 8.)).astype(int)        
    
    # save the container
    np.savez(f'{save_path}/{_id_save}.npz', **features)    

