
"""
This module contains utility functions for optimization and evaluation tasks.

Functions:
- one_step_of_gradient_ascent: Perform one step of gradient ascent on the given samples using the provided predictor model.
- get_equal_indices: Get the indices where the elements of two arrays are equal.
- get_different_indices: Get the indices where the elements of two arrays are different.
- validity_step: Perform validity step on the given latents using the provided model.
- evaluate_val_dizzs: Evaluate the validity dictionaries and reassemble the samples.
- merge_dizzs: Merge a list of dictionaries into a single dictionary.
- gradient_ascent_routine: Perform the gradient ascent routine on the given samples using the provided predictor and model (see Algorithm 1 in the paper).
"""

import numpy as np
import torch
from rdkit import Chem

from amcg_utils.eval_utils import get_validity
from amcg_utils.sampling_utils import get_samples_from_latents, get_molecules_gen_fn


def one_step_of_gradient_ascent(samples, predictor, batch_size, learning_rate=1, descend=False):
    """
    Perform one step of gradient ascent on the given samples using the provided predictor model.

    Args:
        samples (torch.Tensor or numpy.ndarray): The input samples.
        predictor (torch.nn.Module): The predictor model.
        batch_size (int): The batch size for DataLoader.
        learning_rate (float, optional): The learning rate for gradient ascent. Defaults to 1.
        descend (bool, optional): Whether to perform gradient descent instead of ascent. Defaults to False.

    Returns:
        torch.Tensor: The ascended samples.
    """
    if not torch.is_tensor(samples):
        t_samples = torch.tensor(samples, dtype=torch.float32)
    else:
        t_samples = samples
    assert len(t_samples.shape) == 2
    if batch_size > t_samples.shape[0]:
        batch_size = t_samples.shape[0]
    dl = torch.utils.data.DataLoader(t_samples, batch_size=batch_size, shuffle=False)
    ascended = []
    for item in dl:
        predictor.zero_grad()
        x = item.clone().detach().requires_grad_(True)
        y_pred = predictor(x)
# if more than one property here it becomes
        # y_pred = predictor(x)[:TARGET_PROP_X]
        y_pred.backward(torch.ones_like(y_pred))
        with torch.no_grad():
            if descend:
                z = x - learning_rate * x.grad
            else:
                z = x + learning_rate * x.grad
        
        ascended.append(z)
    ascended = torch.cat(ascended, dim=0)
    return ascended


def get_equal_indices(a,b):
    """
    Get the indices where the elements of two arrays are equal.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.

    Returns:
        list: The indices where the elements are equal.
    """
    a = np.array(a)
    b = np.array(b)
    equal_indices = np.where(a==b)[0]
    return list(equal_indices)


def get_different_indices(a,b):
    """
    Get the indices where the elements of two arrays are different.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.

    Returns:
        list: The indices where the elements are different.
    """
    a = np.array(a)
    b = np.array(b)
    different_indices = np.where(a!=b)[0]
    return list(different_indices)


def validity_step(to_evaluate, model, batch_size, get_molecules_fn, n_workers=1):
    """
    Perform validity step on the given latents using the provided model.

    Args:
        to_evaluate (torch.Tensor): The samples to evaluate.
        model (torch.nn.Module): The model for inference.
        batch_size (int): The batch size for inference.
        get_molecules_fn (function): The function to generate molecules from network output.
        n_workers (int, optional): The number of workers. Defaults to 1.

    Returns:
        tuple: A tuple containing the evaluated samples and a dictionary with validity information.
    """
    val_dizz = {}
    samples = get_samples_from_latents(to_evaluate, 
                                       model, 
                                       batch_size,
                                       get_molecules_fn=get_molecules_fn, 
                                       n_workers=n_workers)
    
    _, _, valid_ids, valid_smiles = get_validity(samples)
    invalid_ids = sorted(list(set(range(len(samples))).difference(set(valid_ids))))
    invalid_ids = torch.tensor(invalid_ids).to(torch.long).to(to_evaluate.device)
    to_evaluate = torch.index_select(to_evaluate, 0, invalid_ids).clone()
    val_dizz['indices'] = valid_ids
    val_dizz['smiles'] = valid_smiles
    return to_evaluate, val_dizz


def evaluate_val_dizzs(val_dizzs, n_orig_elements):
    """
    Evaluate the validity dictionaries and reassemble the samples.

    Args:
        val_dizzs (list): A list of validity dictionaries.
        n_orig_elements (int): The number of original elements.

    Returns:
        tuple: A tuple containing the validity scores, valid indices, and valid smiles.
    """
    reassembled = []
    all_indices = list(range(n_orig_elements))
    for dizz in val_dizzs:
        for index,smiles in zip(dizz['indices'], dizz['smiles']):
            reassembled.append((all_indices[index],smiles))
        to_remove = [all_indices[index] for index in dizz['indices']]
        all_indices = list(set(all_indices).difference(set(to_remove)))
        all_indices = sorted(all_indices)
    
    reass_dict = dict(reassembled)
    reass_mols_dict = {k: Chem.MolFromSmiles(v) for k, v in reass_dict.items()}
     
    rebuilt_samples = [reass_mols_dict[index] if index in reass_mols_dict.keys() else None for index in range(n_orig_elements)]
    
    validity, _, valid_ids, valid_smiles = get_validity(rebuilt_samples)
    
    return validity, valid_ids, valid_smiles
    

def merge_dizzs(dizz_list):
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dizz_list (list): A list of dictionaries.

    Returns:
        dict: The merged dictionary.
    """
    merged_dict = {}
    for d in dizz_list:
        for k, v in d.items():
            if k in merged_dict:
                merged_dict[k].append(v)
            else:
                merged_dict[k] = [v]
    return merged_dict
    

def gradient_ascent_routine(samples, predictor, model, ga_bs, dec_bs, learning_rate, 
                            evaluation_step_size, n_steps, n_rejections, descend=False, 
                            get_molecules_fn=get_molecules_gen_fn()):
    """
    Perform the gradient ascent routine on the given samples using the provided predictor and model.
    (See Algorithm 1 in the paper)
    
    Args:
        samples (torch.Tensor or numpy.ndarray): The input samples.
        predictor (torch.nn.Module): The property predictor model.
        model (torch.nn.Module): The inference model.
        ga_bs (int): The batch size for gradient ascent.
        dec_bs (int): The batch size for model inference.
        learning_rate (float): The learning rate for gradient ascent.
        evaluation_step_size (int): The number of steps of gradient ascent between each decoding attempt.
        n_steps (int): The maximum number of steps.
        n_rejections (int): The number of resamplings.
        descend (bool, optional): Whether to perform gradient descent instead of ascent. Defaults to False.
        get_molecules_fn (function, optional): The function to generate molecules from model output. Defaults to get_molecules_gen_fn().

    Returns:
        tuple: A tuple containing the evaluation step summary and the smiles dictionaries.
    """
    if not torch.is_tensor(samples):
        t_samples = torch.tensor(samples, dtype=torch.float32)
    else:
        t_samples = samples
    step = 0
    eval_step_summary = []
    smiles_dizzs = []
    while t_samples.nelement() != 0 and step < n_steps:
        step_diz = {}
        ascended = one_step_of_gradient_ascent(t_samples, predictor, ga_bs, learning_rate, descend=descend)
        
        if step % evaluation_step_size == evaluation_step_size-1:
            to_evaluate = ascended.detach().clone()
            val_dizzs = []
            for _ in range(n_rejections):
                to_evaluate, val_dizz = validity_step(to_evaluate, model, dec_bs, get_molecules_fn=get_molecules_fn)
                val_dizzs.append(val_dizz)
            validity, valid_ids, valid_smiles = evaluate_val_dizzs(val_dizzs, ascended.shape[0])
            step_diz['validity'] = validity
            step_diz['valid_ids'] = valid_ids
            step_diz['valid_smiles'] = valid_smiles
            smiles_dizzs.append(dict(zip(step_diz['valid_ids'], step_diz['valid_smiles'])))
            eval_step_summary.append(step_diz)

        t_samples = ascended
        step = step + 1
    return eval_step_summary, smiles_dizzs