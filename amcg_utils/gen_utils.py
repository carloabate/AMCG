

"""
This module provides utility functions for reading and writing configuration files, manipulating data, and performing various calculations.

Functions:
- read_train_config_file(file_path): Reads train_config.ini file and returns the values as dictionaries.
- read_sample_config_file(file_path): Reads sample_config.ini fine and returns the values as dictionaries.
- read_optim_config_file(file_path): Reads optim_config.ini file and returns the values as dictionaries.
- write_to_log(log_path, line): Writes a line to a log file.
- read_lines_list(samples_path): Reads lines from a file and returns them as a list.
- write_lines_to_file(lines, filepath): Writes lines to a file.
- flatten(mylist): Flattens a nested list.
- at_from_hist(hist, device): Returns atom types from a histogram.
- get_true_mask(n_nodes, device): Returns a mask matrix used for loss.
- single_ls_assignment(sample_np): Performs linear sum assignment on a numpy array.
- distance_matrices(list_x, list_y): Computes pairwise distances between elements of each set.
- get_indices(predictions, targets, n_workers): Computes indices for rearranging predictions based on targets.
- positional_encoding(inputs): Applies positional encoding to inputs.
- rearrange(predictions, targets, n_workers): Rearranges predictions based on targets and computes losses.
- unbatch_output(new_edge_index, atom_numbers, bond_types, batch, hs): Unbatches output tensors.
- get_perturbation_hist(perturb_classes, num_atom_types): Generates a perturbation histogram.
- get_random_hist(num_atoms, num_atom_types): Generates a random histogram.
- get_perturbed_histogram(rhist, perturb_mode, num_atom_types): Generates a perturbed histogram.
"""

import multiprocessing as mp
import random
import configparser
import json
from collections import Counter
import torch
import numpy as np
from torch_geometric.utils import unbatch, unbatch_edge_index
from scipy.optimize import linear_sum_assignment

def read_train_config_file(file_path):
    """
    Reads a configuration file and returns two dictionaries containing the experiment and training settings.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        tuple: A tuple containing two dictionaries:
            - exp_dict (dict): A dictionary containing the experiment settings.
            - train_dict (dict): A dictionary containing the training settings.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    
    # Assigning each item of the config to a variable
    exp_dict = {
        'type': config.get('Experiment', 'type'),
        'dataset_path': config.get('Experiment', 'dataset_path'),
        'load_ckpt': config.getboolean('Experiment', 'load_ckpt', fallback=False),
        'ckpt_path': config.get('Experiment', 'ckpt_path', fallback=None),
        'save_ckpt_path': config.get('Experiment', 'save_ckpt_path'),
        'start_epoch': config.getint('Experiment', 'start_epoch', fallback=0),
        'logging_path': config.get('Experiment', 'logging_path')
    }
    
    train_dict = {
        'epochs': config.getint('Training', 'epochs'),
        'learning_rate': config.getfloat('Training', 'learning_rate'),
        'batch_size': config.getint('Training', 'batch_size'),
        'num_properties': config.getint('Training', 'num_properties', fallback=0),
        'prop_indices': json.loads(config.get('Training', 'prop_indices', fallback='[]'))
    }
    
    return exp_dict, train_dict


def read_sample_config_file(file_path):
    """
    Read a sample configuration file and extract the relevant information.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        tuple: A tuple containing three dictionaries:
            - exp_dict: A dictionary containing experiment-related information.
            - prior_dict: A dictionary containing prior-related information.
            - sample_dict: A dictionary containing sampling-related information.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    exp_dict = {}
    prior_dict = {}
    sample_dict = {}

    # Assigning each item of the config to a variable
    exp_dict['type'] = config.get('Experiment', 'type')    
    exp_dict['test_dataset_path'] = config.get('Experiment', 'test_dataset_path', fallback=None)
    exp_dict['weights_path'] = config.get('Experiment', 'weights_path')
    exp_dict['samples_path'] = config.get('Experiment', 'samples_path')
    exp_dict['n_properties'] = config.getint('Experiment', 'n_properties', fallback=None)
    exp_dict['calculate_vun'] = config.getboolean('Experiment', 'calculate_vun', fallback=False)
    if exp_dict['calculate_vun'] is True:
        exp_dict['smiles_dataset_path'] = config.get('Experiment', 'smiles_dataset_path')
 
    prior_dict['type'] = config.get('Prior', 'type')
    prior_dict['multiplier'] = config.getfloat('Prior', 'multiplier', fallback=1.0)
    prior_dict['n_components'] = config.getint('Prior', 'n_components', fallback=10)

    sample_dict['n_samples'] = config.getint('Sampling', 'n_samples')
    sample_dict['n_workers'] = config.getint('Sampling', 'n_workers', fallback=1)
    sample_dict['sample_size'] = config.getint('Sampling', 'sample_size', fallback=sample_dict['n_samples'])
    sample_dict['batch_size'] = config.getint('Sampling', 'batch_size')
    sample_dict['perturb_mode'] = config.get('Sampling', 'perturb_mode', fallback=None)
    sample_dict['perturb_hist'] = config.getboolean('Sampling', 'perturb_hist', fallback=False)
    sample_dict['use_hs'] = config.getboolean('Sampling', 'use_hs', fallback=True)
    sample_dict['fix_rings'] = config.getboolean('Sampling', 'fix_rings', fallback=False)
    sample_dict['filter_macrocycles'] = config.getboolean('Sampling', 'filter_macrocycles', fallback=False)
    sample_dict['keep_invalid'] = config.getboolean('Sampling', 'keep_invalid', fallback=False)

    return exp_dict, prior_dict, sample_dict


def read_optim_config_file(file_path):
    """
    Read the optimization configuration file and return the experiment and optimization dictionaries.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        tuple: A tuple containing the experiment parameters and optimization parameters.

    """
    config = configparser.ConfigParser()
    config.read(file_path)

    exp_dict = {}
    optim_dict = {}

    # Assigning each item of the config to a variable

    exp_dict['type'] = config.get('Experiment', 'type')
    exp_dict['weights_path'] = config.get('Experiment', 'weights_path')
    exp_dict['prop_index'] = config.getint('Experiment', 'prop_index')
    exp_dict['test_dataset_path'] = config.get('Experiment', 'test_dataset_path', fallback=None)
    exp_dict['smiles_dataset_path'] = config.get('Experiment', 'smiles_dataset_path', fallback=None)
    exp_dict['evaluate_optimization'] = config.getboolean('Experiment', 'evaluate_optimization', fallback=False)
    exp_dict['opt_summary_path'] = config.get('Experiment', 'opt_summary_path')
    exp_dict['opt_smiles_path'] = config.get('Experiment', 'opt_smiles_path')

    optim_dict['use_hs'] = config.getboolean('Optimization', 'use_hs', fallback=True)
    optim_dict['fix_rings'] = config.getboolean('Optimization', 'fix_rings', fallback=False)
    optim_dict['filter_macrocycles'] = config.getboolean('Optimization', 'filter_macrocycles', fallback=False)
    optim_dict['dec_batch_size'] = config.getint('Optimization', 'dec_batch_size')
    optim_dict['learning_rate'] = config.getfloat('Optimization', 'learning_rate')
    optim_dict['eval_step_size'] = config.getint('Optimization', 'eval_step_size')
    optim_dict['n_steps'] = config.getint('Optimization', 'n_steps')
    optim_dict['ga_batch_size'] = config.getint('Optimization', 'ga_batch_size', fallback=optim_dict['dec_batch_size'])
    optim_dict['n_workers'] = config.getint('Optimization', 'n_workers', fallback=1)
    optim_dict['n_rejections'] = config.getint('Optimization', 'n_rejections', fallback=5)

    return exp_dict, optim_dict


def write_to_log(log_path, line):
    """
    Appends a line to a log file.

    Args:
        log_path (str): The path to the log file.
        line (str): The line to be written to the log file.
    """
    with open(log_path, 'a') as f:
        f.write(line + '\n')


def read_lines_list(samples_path):
    """
    Read the lines from a file and return a list of stripped lines.

    Args:
        samples_path (str): The path to the file to be read.

    Returns:
        list: A list of stripped lines from the file.
    """
    infile = open(samples_path, "r")
    lines = infile.readlines()
    stripped_lines = [x.strip() for x in lines]
    infile.close()
    return stripped_lines


def write_lines_to_file(lines, filepath):
    """
    Write a list of lines to a file.

    Args:
        lines (list): The list of lines to write.
        filepath (str): The path to the file.

    Returns:
        None
    """
    with open(filepath, 'w') as file:
        file.write('\n'.join(lines))


def flatten(mylist):
    """
    Flattens a nested list into a single list.

    Args:
        mylist (list): The nested list to be flattened.

    Returns:
        list: The flattened list.
    """
    result = []
    for sublist in mylist:
        for x in sublist:
            if torch.numel(x) > 0:
                result.append(x)
    return result


def at_from_hist(hist, device):
    """
    Generate atom types from a histogram.

    Args:
        hist (list): The histogram containing the counts of each atom type.
        device: The device to store the resulting tensor on.

    Returns:
        torch.Tensor: A tensor containing the atom types.

    """
    lista = [hist[i] * [i] for i in range(len(hist))]
    return torch.tensor(sum(lista, [])).to(torch.long).to(device)


def get_true_mask(n_nodes, device):
    """
    Generate a mask used for loss calculation when batching graphs using PyG.
    Ones represent true edges, zeros represent edges added by aggregating graphs.

    Args:
        n_nodes (list): A list of integers representing the number of nodes in each graph of the batch.
        device (torch.device): The device on which the mask should be created.

    Returns:
        torch.Tensor: The mask.
    """
    matrices = [torch.ones([x,x]) for x in n_nodes]
    mask = torch.block_diag(*matrices)
    return mask.to(device)


def single_ls_assignment(sample_np):
    """
    Performs linear sum assignment on a given numpy array.

    Args:
        sample_np (numpy.ndarray): The input numpy array.

    Returns:
        tuple: A tuple containing two lists - row_sorted and col_sorted.
               row_sorted: A list of row indices after sorting.
               col_sorted: A list of column indices after sorting.
    """
    sample_np[sample_np == np.inf] = 0
    row_idx, col_idx = linear_sum_assignment(sample_np)
    zipped = zip(col_idx, row_idx)
    sorted_pairs = sorted(zipped)
    #indices are sorted so that i can just use the row ones
    col_sorted, row_sorted = [list(t) for t in zip(*sorted_pairs)]
    return row_sorted, col_sorted


def distance_matrices(list_x, list_y):
    """
    Compute pairwise distance matrices between the i-th elements of two lists of tensors.
    Lists must be of the same length (or x can be shorter).
    Each tensor in the list must have the same number of channels.
    Tensors in the list can have different numbers of rows.

    Args:
        list_x (list): List of tensors.
        list_y (list): List of tensors.

    Returns:
        list: List of distance matrices (one per each pair of tensors).
    """
    lengz = [x.shape[0] for x in list_x]
    max_n = max(lengz)
  
    num_sets = len(list_x)
    num_channels = list_x[0].shape[-1]

    x_padded = torch.zeros(num_sets, max_n, num_channels)
    y_padded = torch.zeros(num_sets, max_n, num_channels)

    for i in range(num_sets):
        x_padded[i, :lengz[i], :] = list_x[i]
        y_padded[i, :lengz[i], :] = list_y[i]

    # compute pairwise distances between elements of each set
    distances = torch.cdist(x_padded, y_padded)

    # extract the distance matrix for the first set
    true_distance_matrices = []
    for i in range(num_sets):
        distance_matrix = distances[i, :lengz[i], :lengz[i]]
        true_distance_matrices.append(distance_matrix)

    return true_distance_matrices


def get_indices(predictions, targets, n_workers):
    """
    Calculate the indices for assigning predictions to targets using multiple workers.

    Args:
        predictions (torch.Tensor): Tensor containing the predictions.
        targets (torch.Tensor): Tensor containing the targets.
        n_workers (int): Number of workers to use for parallel processing.

    Returns:
        list: List of indices representing the assignment of predictions to targets.
    """
    with torch.no_grad():
        distances = distance_matrices(predictions, targets)
        distances_np = [x.detach().cpu().numpy() for x in distances]
        pool = mp.Pool(n_workers)
        indices = pool.map(single_ls_assignment, distances_np)
    return indices


def positional_encoding(inputs):
    """
    Apply positional encoding to the input tensor.

    Args:
        inputs (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with positional encoding applied.
    """
    with torch.no_grad():
        position = inputs.shape[0]
        d_model = inputs.shape[-1]
        pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])
        pos_encoding = torch.cat([sines, cosines], dim=-1).to(inputs.device)
    return pos_encoding + inputs


def rearrange(predictions, targets, n_workers):
    """
    Rearranges the predictions and targets based on the indices obtained from get_indices function.
    
    Args:
        predictions (list): List of prediction tensors.
        targets (list): List of target tensors.
        n_workers (int): Number of workers.
    
    Returns:
        tuple: A tuple containing the rearranged predictions tensor and a list of losses.
    """
    loss = torch.nn.MSELoss()
    indices = get_indices(predictions, targets, n_workers)
    losses = [loss(a[row_idx], b) for a,b,(row_idx, _) in zip(predictions,targets,indices)]
    losses = [torch.mean(x) for x in losses]
    out = [a[row_idx] for a,(row_idx, _) in zip(predictions, indices)]
    return torch.cat(out, dim=0), losses


def unbatch_output(new_edge_index, atom_numbers, bond_types, batch, hs):
    """
    Unbatches the output tensors by applying the `unbatch` function to each tensor.

    Args:
        new_edge_index (Tensor): The edge index tensor.
        atom_numbers (Tensor): The atom numbers tensor.
        bond_types (Tensor): The bond types tensor.
        batch (Tensor): The batch tensor.
        hs (Tensor): The hs tensor.

    Returns:
        Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]: A tuple containing the unbatched tensors
        for `new_edge_index`, `atom_numbers`, `bond_types`, and `hs`.
    """
    atom_numbers = unbatch(atom_numbers, batch)
    hs = unbatch(hs, batch)
    new_edge_index = unbatch_edge_index(new_edge_index, batch=batch)

    edge_batch = [[i] * new_edge_index[i].shape[1] for i in range(len(new_edge_index))]
    edge_batch = torch.tensor(sum(edge_batch, []))
    bond_types = unbatch(bond_types, edge_batch)
    return list(new_edge_index), list(atom_numbers), list(bond_types), list(hs)


# Probability densities for QM9 atom types
qm9_at_densities = {0: 0.7196448328323938, 1: 0.11730955161912046,
                    2: 0.16030819332851876, 3: 0.0027374222199669624}
# Weights for QM9 atom types
qm9_at_weights = [0.7196448328323938, 0.11730955161912046, 
                  0.16030819332851876, 0.0027374222199669624]

# Probability densities for Zinc atom types
zinc_at_densities = {0: 0.7367611643029262, 1: 0.12213681727303256,
                     2: 0.09974842911152398, 3: 0.013756689120555445, 
                     4: 2.1833539019133458e-05, 5: 0.017799186198153575, 
                     6: 0.007420110907446935, 7: 0.0022022416459854532, 
                     8: 0.00015352790135676384}
# Weights for Zinc atom types
zinc_at_weights = [0.7367611643029262, 0.12213681727303256,
                   0.09974842911152398, 0.013756689120555445,
                   2.1833539019133458e-05, 0.017799186198153575,
                   0.007420110907446935, 0.0022022416459854532,
                   0.00015352790135676384]


def get_perturbation_hist(perturb_classes, num_atom_types):
    """
    Generate a histogram perturbation.

    Args:
        perturb_classes (list): Type of possible perturbations per atom type.
        num_atom_types (int): Number of atom types.

    Returns:
        torch.Tensor: Tensor of perturbation classes.

    """
    elements = [random.choice(perturb_classes) for _ in range(num_atom_types)]
    return torch.tensor(elements, dtype=torch.long)


def get_random_hist(num_atoms, num_atom_types, weights=None):
    """
    Generate a random histogram of atom types.

    Args:
        num_atoms (int): The total number of atoms.
        num_atom_types (int): The number of different atom types.
        weights (list, optional): A list of weights for each atom type. Defaults to None.

    Returns:
        torch.Tensor: A tensor representing the histogram of atom types.
    """
    classes = list(range(num_atom_types))
    elements = random.choices(classes, weights=weights, k=num_atoms)
    counts = dict(Counter(elements))
    ret = [counts[k] if k in counts.keys() else 0 for k in classes]
    return torch.tensor(ret, dtype=torch.long)


def get_perturbed_histogram(rhist, perturb_mode, num_atom_types):
    """
    Applies perturbations to a histogram based on the specified perturbation mode.

    Args:
        rhist (torch.Tensor): The input histogram.
        perturb_mode (str): The perturbation mode. Can be '1', '1_2', 'random_u', or 'random_p'.
        num_atom_types (int): The number of atom types.

    Returns:
        torch.Tensor: The perturbed histogram.
    """
    n_atoms = torch.sum(rhist, dim=-1).to(torch.long)

    if perturb_mode is None:
        perturb_mode = '1'
    if perturb_mode == '1':
        perturb_classes = [-1, 0, 1]
        perturbation = torch.stack([get_perturbation_hist(perturb_classes, num_atom_types) for _ in n_atoms]).to(rhist.device)
        rhist = rhist + perturbation
        rhist = torch.where(rhist > 0, rhist, torch.zeros_like(rhist))
    elif perturb_mode == '1_2':
        perturb_classes = [-2, -1, 0, 1, 2]
        perturbation = torch.stack([get_perturbation_hist(perturb_classes, num_atom_types) for _ in n_atoms]).to(rhist.device)
        rhist = rhist + perturbation
        rhist = torch.where(rhist > 0, rhist, torch.zeros_like(rhist))
    elif perturb_mode == 'random_u':
        rhist = torch.stack([get_random_hist(x, num_atom_types) for x in n_atoms])
    elif perturb_mode == 'random_p':
        if num_atom_types == 4:
            rhist = torch.stack([get_random_hist(x, num_atom_types, weights=qm9_at_weights) for x in n_atoms])
        elif num_atom_types == 9:
            rhist = torch.stack([get_random_hist(x, num_atom_types, weights=zinc_at_weights) for x in n_atoms])
        else:
            rhist = torch.stack([get_random_hist(x, num_atom_types) for x in n_atoms])

    return rhist