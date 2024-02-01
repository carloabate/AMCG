"""
This module provides functions for calculating molecular properties and generating pandas DataFrames
containing molecular properties from SMILES strings.

Functions:
- get_tanimoto(smiles_1, smiles_2): Calculate the Tanimoto similarity between two SMILES strings.
- get_diversity(smiles_list): Calculate the diversity of a list of SMILES strings.
- get_num_macrocycles(mol): Get the number of macrocycles in a molecule.
- get_props(smiles_or_mol): Calculate various molecular properties for a given SMILES string or RDKit molecule object.
- get_props_df(path, n_mols=10000): Generate a pandas DataFrame containing molecular properties for a given file of SMILES strings.
- get_props_df_from_list(smiles_list, n_mols=10000): Generate a pandas DataFrame containing molecular properties for a given list of SMILES strings.
- process_dataframe(df): Process a pandas DataFrame to calculate molecular properties table.
"""

import os
import sys
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import DataStructs
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import sascorer
import npscorer
np_f_score = npscorer.readNPModel()

from amcg_utils.gen_utils import read_lines_list
from amcg_utils.build_mol_utils import mol_to_nx

def get_tanimoto(smiles_1, smiles_2):
    """
    Calculate the Tanimoto similarity between two SMILES strings.

    Args:
        smiles_1 (str): The first SMILES string.
        smiles_2 (str): The second SMILES string.

    Returns:
        float: The Tanimoto similarity between the two SMILES strings.
    """
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return tanimoto_similarity


def get_diversity(smiles_list):
    """
    Calculate the diversity of a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        float: The diversity value.
    """
    mols = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles_list]
    fps = [Chem.RDKFingerprint(x) for x in mols]
    total = 0
    denom = 0
    for i in tqdm(range(len(fps))):
        for j in range(i):
            total = total + DataStructs.TanimotoSimilarity(fps[i], fps[j])
            denom = denom + 1
    return total/denom


def get_num_macrocycles(mol):
    """
    Get the number of macrocycles in a molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
        int: The number of macrocycles.
    """
    graph = mol_to_nx(mol)
    num_macrocycles = 0
    for cycle in nx.chordless_cycles(graph):
        cycle_length = len(cycle)
        if cycle_length > 6:
            num_macrocycles = num_macrocycles + 1
    return num_macrocycles


def get_props(smiles_or_mol):
    """
    Calculate various molecular properties for a given SMILES string or RDKit molecule object.

    Args:
        smiles_or_mol (str or rdkit.Chem.rdchem.Mol): The SMILES string or RDKit molecule object.

    Returns:
        list: A list of molecular properties.
    """
    if isinstance(smiles_or_mol, str):
        molly = Chem.MolFromSmiles(smiles_or_mol, sanitize=True)
    else:
        molly = smiles_or_mol
    molly.UpdatePropertyCache(strict=False)
    mol=Chem.AddHs(molly, addCoords=True)
    logp_value = Crippen.MolLogP(mol)
    qed_value = QED.qed(mol)
    sas_value = sascorer.calculateScore(mol)
    mol_wt = Descriptors.MolWt(mol)
    heavy_mol_wt = Descriptors.HeavyAtomMolWt(mol)
    nps_value = npscorer.scoreMol(mol, fscore=np_f_score)
    n_heavy_atoms = mol.GetNumHeavyAtoms()
    plogp_value = logp_value - sas_value - get_num_macrocycles(mol)
    y = [logp_value, qed_value, sas_value, mol_wt, heavy_mol_wt, nps_value, n_heavy_atoms, plogp_value]
    return y


def get_props_df(path, n_mols=10000):
    """
    Generate a pandas DataFrame containing molecular properties for a given file of SMILES strings.

    Args:
        path (str): The path to the file containing SMILES strings.
        n_mols (int, optional): The number of molecules to process. Defaults to 10000.

    Returns:
        pandas.DataFrame: A DataFrame containing molecular properties.
    """
    smiles = read_lines_list(path)[:n_mols]
    props = [get_props(x) for x in tqdm(smiles)]
    np_props = np.array(props)
    return pd.DataFrame(np_props, columns=['logp','qed','sas','molwt','heavymolwt','nps','num_heavy_atoms','plogp'])


def get_props_df_from_list(smiles_list, n_mols=10000):
    """
    Generate a pandas DataFrame containing molecular properties for a given list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.
        n_mols (int, optional): The number of molecules to process. Defaults to 10000.

    Returns:
        pandas.DataFrame: A DataFrame containing molecular properties.
    """
    smiles = random.sample(smiles_list, n_mols)
    props = [get_props(x) for x in tqdm(smiles)]
    np_props = np.array(props)
    return pd.DataFrame(np_props, columns=['logp','qed','sas','molwt','heavymolwt','nps','num_heavy_atoms','plogp'])


def process_dataframe(df):
    """
    Process a pandas DataFrame (to calculate molecular properties table).

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing the processed data.
    """
    new_data = {}
    for col in df.columns:
        val_second_row = df[col][1]
        val_third_row = df[col][2]
        col_string = f"{val_second_row:.2f} ({val_third_row:.2f})"
        new_data[col] = col_string
    return new_data