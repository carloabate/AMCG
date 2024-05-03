"""
This module contains utility functions for evaluating the generated molecules.

Functions:
- get_novelty(orig_uniform_smiles, gen_smiles_dataset): Calculates the novelty of generated molecules compared to the original molecules.
- get_uniqueness(gen_smiles_dataset): Calculates the uniqueness of generated molecules.
- get_validity(gen_dataset): Calculates the validity of generated molecules.
- get_vun(sampled_mols, dataset_smiles): Calculates the VUN (Validity, Uniqueness, Novelty) score of generated molecules.
- get_novelty_non_unique(orig_smiles, gen_smiles): Calculates the novelty of generated molecules (non-unique) compared to the original molecules.
- get_un(sampled_smiles, original_smiles): Calculates the novelty, uniqueness, and UN (Uniqueness * Novelty) score of generated molecules.
- get_cumulative_un(original_smiles, sampled_smiles, step_size): Calculates the cumulative novelty, uniqueness, and UN scores of generated molecules.
- get_un_evaluation_df(original_smiles_path, sampled_smiles_path, step_size, num_samples=200000): Generates a DataFrame with evaluation metrics for generated molecules.
- get_un_smiles(smiles_list, orig_smiles, num_smiles=10000): Returns a list of unique novel generated smiles.
- evaluate_optimization(test_smiles, merged_smiles, dataset_smiles, prop_index): Evaluates the the results of the optimization routine.
"""

from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import rdkit.Chem as Chem

from amcg_utils.mol_prop_utils import get_props
from amcg_utils.build_mol_utils import get_clean_smiles_from_mol, mol_to_nx
from amcg_utils.gen_utils import read_lines_list


def get_novelty(orig_uniform_smiles, gen_smiles_dataset):
    """
    Calculate the novelty of generated SMILES strings compared to the original dataset.

    Parameters:
    orig_uniform_smiles (list): List of original SMILES strings.
    gen_smiles_dataset (list): List of generated SMILES strings.

    Returns:
    float: The novelty score, calculated as the ratio of unique generated SMILES strings to the total number of generated SMILES strings.
    list: A list of unique generated SMILES strings that are not present in the original dataset.
    """
    new = gen_smiles_dataset
    orig = orig_uniform_smiles
    orig_smiles_set = set(orig)
    gen_smiles_set = set(new)
    diff = gen_smiles_set.difference(orig_smiles_set)
    return len(diff)/len(gen_smiles_set), list(diff)


def get_uniqueness(gen_smiles_dataset): 
    """
    Calculate the uniqueness of a given set of SMILES strings.

    Parameters:
    gen_smiles_dataset (list): A list of SMILES strings.

    Returns:
    float: The uniqueness score, calculated as the ratio of unique SMILES strings to total SMILES strings.
    list: A list of unique SMILES strings.
    """
    smiles = gen_smiles_dataset
    smiles_set = set(smiles)
    return len(list(smiles_set))/len(smiles), list(smiles_set)


def get_validity(gen_dataset):
    """
    Calculates the validity of molecules in a given dataset.

    Args:
        gen_dataset (list): A list of molecules.

    Returns:
        tuple: A tuple containing the validity ratio, valid molecules, valid molecule IDs, and valid smiles.

    """
    not_valid = 0
    valid_mols = []
    valid_ids = []
    valid_smiles = []
    for count, mol in enumerate(gen_dataset):
        try:
            graph = mol_to_nx(mol)
        except:
            graph = None
        if graph is None:
            not_valid = not_valid + 1
        elif nx.is_connected(graph):
            try:
                Chem.SanitizeMol(mol)
                Chem.SanitizeMol(mol) #https://sourceforge.net/p/rdkit/mailman/message/36455204/
                smiles = get_clean_smiles_from_mol(mol)
                
                valid_smiles.append(smiles)
                valid_mols.append(mol)
                valid_ids.append(count)
            except:
                not_valid = not_valid + 1
        else:
            not_valid = not_valid + 1
    valid = len(gen_dataset) - not_valid
    return valid / len(gen_dataset), valid_mols, valid_ids, valid_smiles


def get_vun(sampled_mols, dataset_smiles):
    """
    Calculates the Validity, Uniqueness, Novelty, and VUN score for a list of sampled molecules.

    Args:
        sampled_mols (list): List of sampled molecules.
        dataset_smiles (list): List of SMILES strings from the original dataset.

    Returns:
        dict: A dictionary containing the Validity, Uniqueness, Novelty, and VUN score.

    """
    validity, _, _, valid_smiles = get_validity(sampled_mols)
    if len(valid_smiles) == 0:
        return {'Validity': 0, 'Uniqueness': 0, 'Novelty': 0, 'VUN': 0}
    novelty, _ = get_novelty(dataset_smiles + [""], valid_smiles)
    uniqueness, _ = get_uniqueness(valid_smiles)
    vun = validity * novelty * uniqueness
    return {'Validity': validity, 'Uniqueness': uniqueness, 'Novelty': novelty, 'VUN': vun}


def get_novelty_non_unique(orig_smiles, gen_smiles):
    """
    Calculate the novelty of generated smiles compared to original smiles, without considering uniqueness.

    Args:
        orig_smiles (list): List of original smiles.
        gen_smiles (list): List of generated smiles.

    Returns:
        float: A score, ranging from 0 to 1. 
               A score of 0 indicates all generated smiles are present in the original dataset,
               while a score of 1 indicates no generated smiles are present in the original dataset.
    """
    new = gen_smiles
    orig = orig_smiles
    orig_set = set(orig)
    return 1 - sum([v for k, v in Counter(new).items() if k in orig_set])/len(new)


def get_un(sampled_smiles, original_smiles):
    """
    Calculates the novelty, uniqueness, novelty*uniqueness, and novelty non-unique values.

    Args:
        sampled_smiles (list): List of sampled SMILES strings.
        original_smiles (list): List of original SMILES strings.

    Returns:
        tuple: A tuple containing the novelty, uniqueness, novelty*uniqueness, and novelty non-unique values.
    """
    novelty, _ = get_novelty(original_smiles + [""], sampled_smiles)    
    uniqueness, _ = get_uniqueness(sampled_smiles)
    novelty_non_unique = get_novelty_non_unique(original_smiles, sampled_smiles)
    un = novelty*uniqueness
    return novelty, uniqueness, un, novelty_non_unique


def get_cumulative_un(original_smiles, sampled_smiles, step_size):
    """
    Calculate the cumulative uniqueness and novelty of sampled SMILES strings, used for the persistence plot.

    Args:
        original_smiles (list): List of original SMILES strings.
        sampled_smiles (list): List of sampled SMILES strings.
        step_size (int): Step size for calculating uniqueness and novelty.

    Returns:
        tuple: A tuple containing four lists:
            - nn: List of cumulative uniqueness values.
            - uu: List of cumulative novelty values.
            - uunn: List of cumulative uniqueness*novelty values.
            - nnunnu: List of cumulative non unique novelty values.
    """
    partial_size = step_size
    nn = []
    uu = []
    nnunnu = []
    uunn = []
    pbar = tqdm(total=len(sampled_smiles)) # to use in notebook
    while len(sampled_smiles) >= partial_size:
        n, u, un, nnu = get_un(sampled_smiles[:partial_size], original_smiles)
        nn.append(n)
        uu.append(u)
        uunn.append(un)
        nnunnu.append(nnu)
        partial_size = partial_size + step_size
        pbar.update(step_size)
    return nn, uu, uunn, nnunnu


def get_un_evaluation_df(original_smiles_path, sampled_smiles_path, step_size, num_samples=200000):
    """
    Calculate the evaluation metrics for a set of sampled SMILES strings.

    Parameters:
    original_smiles_path (str): The file path to the original SMILES strings dataset (to calculate uniqueness).
    sampled_smiles_path (str): The file path to the sampled SMILES strings to evaluate.
    step_size (int): The step size for calculating cumulative metrics.
    num_samples (int, optional): The number of sampled SMILES strings to consider. Defaults to 200000.

    Returns:
    pandas.DataFrame: A DataFrame containing the evaluation metrics:
        - Novelty: The cumulative novelty score.
        - Uniqueness: The cumulative uniqueness score.
        - Uniqueness * Novelty: The product of uniqueness and novelty.
        - Non Unique Novelty: The cumulative non-unique novelty score.
    """
    original_smiles = read_lines_list(original_smiles_path)
    sampled_smiles = read_lines_list(sampled_smiles_path)[:num_samples]

    novelty, uniqueness, un, novelty_non_unique = get_cumulative_un(original_smiles, sampled_smiles, step_size)

    df = pd.DataFrame({
    'Novelty': novelty,
    'Uniqueness': uniqueness,
    'Uniqueness * Novelty': un,
    'Non Unique Novelty': novelty_non_unique})

    return df


def get_un_smiles(smiles_list, orig_smiles, num_smiles=10000):
    """
    Returns a list of unique and novel (wrt `orig_smiles`) SMILES strings from `smiles_list`.

    Args:
        smiles_list (list): A list of SMILES strings.
        orig_smiles (list): A list of original SMILES strings.
        num_smiles (int, optional): The maximum number of unique SMILES strings to return. Defaults to 10000.

    Returns:
        list: A list of unique SMILES strings not present in `orig_smiles`.
    """
    return list(set(smiles_list).difference(set(orig_smiles)))[:num_smiles]


def evaluate_optimization(test_smiles, merged_smiles, dataset_smiles, prop_index):
    """
    Evaluate the optimization process based on the given test smiles, merged smiles,
    dataset smiles, and property index.

    Args:
        test_smiles (list): List of test smiles.
        merged_smiles (dict): Dictionary of merged smiles (generated via `amcg_utils.opt_utils.merge_dizzs()` function).
        dataset_smiles (list): List of dataset smiles.
        prop_index (int): Index (wrt `amcg_utils.mol_prop_utils.get_props()` function output) of the property to evaluate.

    Returns:
        tuple: A tuple containing the success rate, optimization rate, initial property values,
        optimized property values, initial smiles, and optimized smiles.
    """
    indices = merged_smiles.keys()
    initial_smiles = []
    initial_props = []
    final_smiles = []
    final_props = []
    
    for index in indices:
        initial_prop = get_props(test_smiles[index])[prop_index]
        path_prop = [get_props(x)[prop_index] for x in merged_smiles[index]]
        max_prop = np.max(path_prop)
        if max_prop > initial_prop:
            pos = np.argmax(path_prop)
            initial_smiles.append(test_smiles[index])
            initial_props.append(initial_prop)
            
            final_smiles.append(merged_smiles[index][pos])
            final_props.append(max_prop)
    
    is_new = [x not in dataset_smiles for x in final_smiles]
    opt_rate = len(final_props) / len(indices)
    success_rate = sum(is_new)/len(final_props)
    
    return success_rate, opt_rate, initial_props, final_props, initial_smiles, final_smiles