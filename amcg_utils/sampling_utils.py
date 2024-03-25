"""
This module provides classes and functions for sampling molecules from latent spaces using various methods.

Classes:
- LatentSampler: Abstract base class for latent samplers.
- VAE_Sampler: Latent sampler for a unit Gaussian prior.
- VAElike_Sampler: Latent sampler for a generic Gaussian.
- GMM_D_Sampler: Latent sampler for Gaussian Mixture Model (GMM) with diagonal covariance matrix.
- GMM_F_Sampler: Latent sampler for GMM with full covariance matrix.
- GMM_PW_Sampler: Latent sampler for GMM with pre-defined samples.

Functions:
- get_latent_sampler: Function to get the appropriate latent sampler based on the prior type.
- get_mol_zs: Get the latent space representations of molecules from a dataset using a model.
- get_net_output: Get the network output (edge index, atom types, bond types, and hydrogen prediction) from sampled latents.
- get_molecules_gen_fn: Get the function to generate molecules from network output, with options for fixing rings and filtering macrocycles.
- get_samples_from_latents: Generate molecules from sampled latents using a model and the network output function.
- get_samples_from_sampler: Generate N molecules from a latent sampler using a model and the network output function.
"""

from abc import ABC, abstractmethod
import itertools
import random
from joblib import Parallel, delayed

import torch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.data import Batch
from sklearn.mixture import GaussianMixture
from rdkit import Chem

from amcg_utils.gen_utils import unbatch_output
from amcg_utils.build_mol_utils import build_mol, last_try, fix_rings, filter_macrocycles
from amcg_utils.eval_utils import get_validity


class LatentSampler(ABC):
    """
    Abstract base class for latent samplers.
    """

    def __init__(self, mol_zs):
        self.device = mol_zs.device

    @abstractmethod
    def sample(self, N):
        """
        Abstract method for sampling latent vectors.

        Args:
            N (int): Number of samples to generate.

        Returns:
            List: List of sampled latent vectors.
        """
        pass


class VAE_Sampler(LatentSampler):
    """
    A class for sampling latent vectors using a Variational Autoencoder (VAE).

    Args:
        mol_zs (torch.Tensor): Latent vectors of molecules.

    Attributes:
        dim (int): Dimensionality of the latent vectors.

    Methods:
        sample(N): Generates N samples from the VAE latent space.

    """

    def __init__(self, mol_zs):
        super().__init__(mol_zs)
        self.dim = mol_zs.shape[1]
    
    def sample(self, N):
        """
        Generates N samples from the VAE latent space.

        Args:
            N (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples from the VAE latent space.

        """
        return torch.randn((N, self.dim)).to(self.device)
        

class VAElike_Sampler(LatentSampler):
    """
    A class representing a VAE-like sampler.

    Args:
        mol_zs (torch.Tensor): The input tensor of molecular latent vectors.
        multiplier (float, optional): A multiplier to scale the standard deviation. Defaults to 1.

    Attributes:
        dim (int): The dimension of the latent vectors.
        mean (torch.Tensor): The mean of the input latent vectors.
        std (torch.Tensor): The standard deviation of the input latent vectors.

    Methods:
        sample(N): Generates N samples from the VAE-like distribution.

    """

    def __init__(self, mol_zs, multiplier=1):
        super().__init__(mol_zs)
        self.dim = mol_zs.shape[1]
        self.mean = torch.mean(mol_zs, dim=0).to(self.device)
        self.std = torch.std(mol_zs, dim=0).to(self.device) * multiplier
    
    def sample(self, N):
        """
        Generates N samples from the VAE-like distribution.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (N, dim) containing the generated samples.

        """
        return torch.randn((N, self.dim)).to(self.device) * self.std + self.mean


class GMM_D_Sampler(LatentSampler):
    """
    Gaussian Mixture Model (GMM) based sampler for generating samples from a latent space.
    Diagonal covariance matrix is used.
    
    Args:
        mol_zs (torch.Tensor): Latent space tensor representing the molecules.
        n_components (int): Number of components in the GMM. Defaults to 10.
        multiplier (float): Covariance matrix multiplier. Defaults to 1.
    
    Attributes:
        dim (int): The dimensionality of the molecular latent vectors.
        gmm (sklearn.mixture.GaussianMixture): The GMM model.
    """
    def __init__(self, mol_zs, n_components=10, multiplier=1):
        super().__init__(mol_zs)
        self.dim = mol_zs.shape[1]
        mol_zs_np = mol_zs.cpu().numpy()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        self.gmm.fit(mol_zs_np)
        self.gmm.covariances_ *= multiplier  # Scale the covariance matrix
        
    def sample(self, N):
        """
        Generate samples from the GMM.
        
        Args:
            N (int): Number of samples to generate.
        
        Returns:
            torch.Tensor: Generated samples as a tensor.
        """
        samples = self.gmm.sample(N)[0]
        return torch.tensor(samples, dtype=torch.float32).to(self.device)


class GMM_F_Sampler(LatentSampler):
    """
    Gaussian Mixture Model (GMM) based sampler for generating samples from a given set of molecular latent vectors.

    Args:
        mol_zs (torch.Tensor): The molecular latent vectors.
        n_components (int): The number of components in the GMM. Default is 10.

    Attributes:
        dim (int): The dimensionality of the molecular latent vectors.
        gmm (sklearn.mixture.GaussianMixture): The GMM model.

    """

    def __init__(self, mol_zs, n_components=10):
        super().__init__(mol_zs)
        self.dim = mol_zs.shape[1]
        mol_zs_np = mol_zs.cpu().numpy()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.gmm.fit(mol_zs_np)
    
    def sample(self, N):
        """
        Generate samples from the GMM.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated samples.

        """
        samples = self.gmm.sample(N)[0]
        return torch.tensor(samples, dtype=torch.float32).to(self.device)
    
class GMM_PW_Sampler(LatentSampler):
    """
    This class represents the GMM_PW prior.

    Args:
        mol_zs (numpy.ndarray): The array of molecular latent vectors.

    Attributes:
        dim (int): The dimension of the latent vectors.

    Methods:
        sample(N): Sample N latent vectors from the Gaussian Mixture Model.

    """

    def __init__(self, mol_zs):
        super().__init__(mol_zs)
        self.dim = mol_zs.shape[1]
    
    def sample(self, N):
        return random.sample(self.mol_zs, N)


def get_latent_sampler(prior_type, mol_zs, n_components=10, multiplier=1):
    """
    Returns a latent sampler based on the specified prior type.

    Args:
        prior_type (str): The type of prior distribution for the latent space.
        mol_zs (list): List of molecular latent vectors.
        n_components (int, optional): Number of components for GMM-based samplers. Defaults to 10.
        multiplier (int, optional): Multiplier for VAE-like and GMM_D samplers. Defaults to 1.

    Returns:
        object: An instance of the corresponding latent sampler class.

    Raises:
        ValueError: If an invalid type of latent sampler is specified.
    """
    if prior_type == 'VAE':
        return VAE_Sampler(mol_zs)
    elif prior_type == 'VAElike':
        return VAElike_Sampler(mol_zs, multiplier=multiplier)
    elif prior_type == 'GMM_D':
        return GMM_D_Sampler(mol_zs, n_components=n_components, multiplier=multiplier)
    elif prior_type == 'GMM_F':
        return GMM_F_Sampler(mol_zs, n_components=n_components)
    elif prior_type == 'GMM_PW':
        return GMM_PW_Sampler(mol_zs)
    else:
        raise ValueError('Invalid type of latent sampler')


def get_mol_zs(dataset, model, batch_size, device):
    """
    Get the latent representations (mol_zs) for a given dataset using a model.

    Args:
        dataset (Dataset): The dataset containing the input data.
        model (Model): The model used to encode the input data.
        batch_size (int): The batch size used for inference.
        device (str): The device to perform the inference on.

    Returns:
        torch.Tensor: The tensor containing the latent representations.
    """
    dl = DataListLoader(dataset, batch_size=batch_size)
    mol_zs = []
    with torch.inference_mode():
        model.eval()
        for item in dl:
            data_batch = Batch.from_data_list(item).to(device)
            _, mol_z = model.encode_batch(data_batch)
            mol_zs.append(mol_z)
    return torch.cat(mol_zs, dim=0)


def get_net_output(sampled_latent: torch.Tensor, model,
                  batch_size, # batch size for inference
                  perturb_hist, # whether to perturb the histogram
                  perturb_mode): # how to perturb the histogram
    """
    Perform inference using the given model on the sampled latent tensor.

    Args:
        sampled_latent (torch.Tensor): The sampled latent tensor.
        model: The model used for inference.
        batch_size (int): The batch size for inference.
        perturb_hist (bool): Whether to perturb the histogram.
        perturb_mode: How to perturb the histogram.

    Returns:
        Tuple: A tuple containing the following:
            - nnew_edge_index: The edge index.
            - aatom_types: The atom types.
            - bbond_types: The bond types.
            - hhs_pred: The predicted hydrogens.
    """
    nnew_edge_index = []
    aatom_types = []
    bbond_types = []
    hhs_pred = []

    dl = DataLoader(sampled_latent, batch_size=batch_size)
    for item in dl:
        new_edge_index, atom_types, bond_pred, batch, hs_pred = model.infer_from_z(item, perturb_hist=perturb_hist, perturb_mode=perturb_mode)
        hs_pred = torch.round(torch.squeeze(hs_pred)).to(torch.long)
        bond_types = torch.argmax(bond_pred, dim=-1)
        
        new_edge_index = new_edge_index.to('cpu')
        atom_types = atom_types.to('cpu')
        bond_types = bond_types.to('cpu')
        batch = batch.to('cpu')
        hs_pred = hs_pred.to('cpu')
        
        new_edge_index, atom_types, bond_types, hs_pred = unbatch_output(new_edge_index, atom_types, bond_types, batch, hs_pred)
        
        nnew_edge_index = nnew_edge_index + new_edge_index
        aatom_types = aatom_types + atom_types
        bbond_types = bbond_types + bond_types
        hhs_pred = hhs_pred + hs_pred

    return nnew_edge_index, aatom_types, bbond_types, hhs_pred


def get_molecules_gen_fn(fix_rings_flag=False, filter_macrocycles_flag=False, use_hs_flag=True):
    """
    Generate a function that returns molecules based on the given flags.

    Args:
        fix_rings_flag (bool, optional): Flag indicating whether to fix rings in the molecules. Defaults to False.
        filter_macrocycles_flag (bool, optional): Flag indicating whether to filter out macrocycles from the molecules. Defaults to False.
        use_hs_flag (bool, optional): Flag indicating whether to use hydrogen atoms in the molecules. Defaults to True.

    Returns:
        function: A function that takes new_edge_index, atom_types, bond_types, and hs_pred as inputs and returns a molecule.

    """
    if use_hs_flag:
        if fix_rings_flag:
            if filter_macrocycles_flag:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred):
                    m = build_mol(num_atoms=atom_types.shape[0],
                                  edge_index=new_edge_index,
                                  atom_numbers_or_types=atom_types,
                                  bond_types=bond_types,
                                  hs=hs_pred,
                                  is_atom_types=True)
                    m = fix_rings(m)
                    m = filter_macrocycles(m)
                    return m
            else:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred):
                    m = build_mol(num_atoms=atom_types.shape[0],
                                  edge_index=new_edge_index,
                                  atom_numbers_or_types=atom_types,
                                  bond_types=bond_types,
                                  hs=hs_pred,
                                  is_atom_types=True)
                    m = fix_rings(m)
                    return m
        else:
            if filter_macrocycles_flag:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred):
                    m = build_mol(num_atoms=atom_types.shape[0],
                                  edge_index=new_edge_index,
                                  atom_numbers_or_types=atom_types,
                                  bond_types=bond_types,
                                  hs=hs_pred,
                                  is_atom_types=True)
                    m = filter_macrocycles(m)
                    return m
            else:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred):
                    m = build_mol(num_atoms=atom_types.shape[0],
                                  edge_index=new_edge_index,
                                  atom_numbers_or_types=atom_types,
                                  bond_types=bond_types,
                                  hs=hs_pred,
                                  is_atom_types=True)
                    return m
    else:
        if fix_rings_flag:
            if filter_macrocycles_flag:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred=None):
                    m = last_try(num_atoms=atom_types.shape[0],
                                 edge_index=new_edge_index,
                                 atom_numbers_or_types=atom_types,
                                 bond_types=bond_types,
                                 is_atom_types=True)
                    m = fix_rings(m)
                    m = filter_macrocycles(m)
                    return m
            else:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred=None):
                    m = last_try(num_atoms=atom_types.shape[0],
                                 edge_index=new_edge_index,
                                 atom_numbers_or_types=atom_types,
                                 bond_types=bond_types,
                                 is_atom_types=True)
                    m = fix_rings(m)
                    return m
        else:
            if filter_macrocycles_flag:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred=None):
                    m = last_try(num_atoms=atom_types.shape[0],
                                 edge_index=new_edge_index,
                                 atom_numbers_or_types=atom_types,
                                 bond_types=bond_types,
                                 is_atom_types=True)
                    m = filter_macrocycles(m)
                    return m
            else:
                def get_molecules(new_edge_index, atom_types, bond_types, hs_pred=None):
                    m = last_try(num_atoms=atom_types.shape[0],
                                 edge_index=new_edge_index,
                                 atom_numbers_or_types=atom_types,
                                 bond_types=bond_types,
                                 is_atom_types=True)
                    return m
    return get_molecules


def get_samples_from_latents(latents, model, batch_size, get_molecules_fn=get_molecules_gen_fn(), n_workers=1):
    """
    Get samples from latents using a given model.

    Args:
        latents (list): List of latent vectors.
        model: The model used for inference.
        batch_size (int): The batch size for inference.
        get_molecules_fn (function, optional): Function for generating molecules from network output. Defaults to get_molecules_gen_fn().
        n_workers (int, optional): Number of parallel workers for generating samples. Defaults to 1.

    Returns:
        list: List of sampled molecules.
    """
    nnew_edge_index, aatom_types, bbond_types, hhs_pred = get_net_output(latents, model, batch_size, perturb_hist=False, perturb_mode='none')
    sampled_mols = Parallel(n_jobs=n_workers)(delayed(get_molecules_fn)(*args) for args in itertools.zip_longest(nnew_edge_index, aatom_types, bbond_types, hhs_pred))
    return sampled_mols


def get_samples_from_sampler(latent_sampler, model, get_molecules_fn, n_samples, sample_size, batch_size, n_workers, perturb_hist, perturb_mode, keep_invalid):
    """
    Generate a specified number of samples using a latent sampler and a model.

    Args:
        latent_sampler (LatentSampler): The latent sampler object.
        model (Model): The model object.
        get_molecules_fn (function): The function to get molecules from network output.
        n_samples (int): The number of samples to generate.
        sample_size (int): The size of each latent sample size.
        batch_size (int): The batch size for network inference.
        n_workers (int): The number of workers for parallel processing.
        perturb_hist (bool): Whether to perturb the histogram.
        perturb_mode (str): The perturbation mode.
        keep_invalid (bool): Whether to keep invalid molecules.

    Returns:
        list: A list of sampled molecules.
    """
    sampled_mols = []
    print(f'Sampling {n_samples} molecules')
    while len(sampled_mols) < n_samples:
        sampled_latent = latent_sampler.sample(sample_size)
        nnew_edge_index, aatom_types, bbond_types, hhs_pred = get_net_output(sampled_latent, model, batch_size, perturb_hist, perturb_mode)

        # 1 worker is faster for reasonable sample sizes
        sampled_mols_batch = Parallel(n_jobs=n_workers)(delayed(get_molecules_fn)(*args) for args in itertools.zip_longest(nnew_edge_index, aatom_types, bbond_types, hhs_pred))
        if keep_invalid:
            sampled_mols = sampled_mols + sampled_mols_batch
            n_sampled = len(sampled_mols_batch)
        else:
            _, valid_mols, _, _ =  get_validity(sampled_mols_batch)
            sampled_mols = sampled_mols + valid_mols
            n_sampled = len(valid_mols)
        print(f'Sampled {n_sampled} molecules')
        print(f'Total sampled: {min(len(sampled_mols), n_samples)}')

    return sampled_mols[:n_samples]