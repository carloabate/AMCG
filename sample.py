import torch
import sys
from models import get_amcg_qm9, get_amcg_zinc

from amcg_utils.sampling_utils import get_molecules_gen_fn, get_latent_sampler, get_mol_zs, get_samples_from_sampler
from amcg_utils.gen_utils import read_sample_config_file, write_lines_to_file, read_lines_list
from amcg_utils.eval_utils import get_vun, get_validity
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def main():
    """
    Main function that performs the sampling and VUN evaluation of the sampled molecules.
    After that, it saves the samples to a file.
    
    It performs the following steps:
    1. Reads the configuration file.
    2. Loads the data.
    3. Instantiates the model based on the experiment type.
    4. Loads the model weights.
    5. Gets the molecular embeddings.
    6. Fits the latent sampler.
    7. Gets the molecules generation function.
    8. Performs the sampling loop.
    9. Evaluates the samples if required.
    10. Saves the samples.

    Args:
        None

    Returns:
        None
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # READ CONFIG
    config_path = sys.argv[1]
    exp_dict, prior_dict, sample_dict = read_sample_config_file(config_path)

    # LOAD DATA
    if exp_dict['test_dataset_path'] is not None:
        ds = torch.load(exp_dict['test_dataset_path'])

    # INSTANTIATE MODEL
    if exp_dict['type'] == 'qm9':
        if exp_dict['n_properties'] is not None:
            num_properties = exp_dict['n_properties']
            Da_Model = get_amcg_qm9(num_properties=num_properties)
        else:
            Da_Model = get_amcg_qm9()
    elif exp_dict['type'] == 'zinc':
        if exp_dict['n_properties'] is not None:
            num_properties = exp_dict['n_properties']
            Da_Model = get_amcg_zinc(num_properties=num_properties)
        else:
            Da_Model = get_amcg_zinc()

    # LOAD MODEL
    weights_load_path = exp_dict['weights_path']
    state_dict = torch.load(weights_load_path, map_location='cpu')
    Da_Model.load_state_dict(state_dict)

    Da_Model = Da_Model.to(DEVICE)

    
    if prior_dict['type'] == 'GMM_PW':
        _, mol_mus, mol_logstd = get_mol_zs(ds, Da_Model, sample_dict['batch_size'], device=DEVICE)
        latent_sampler = get_latent_sampler(prior_dict['type'], mol_mus, n_components=prior_dict['n_components'],
                                        multiplier=prior_dict['multiplier'], mol_logstd=mol_logstd)

    else:
        mol_zs, _, _ = get_mol_zs(ds, Da_Model, sample_dict['batch_size'], device=DEVICE)
        latent_sampler = get_latent_sampler(prior_dict['type'], mol_zs, n_components=prior_dict['n_components'],
                                        multiplier=prior_dict['multiplier'])

    # GET MOLECULES GENERATION FUNCTION
    mol_gen_fn = get_molecules_gen_fn(fix_rings_flag=sample_dict['fix_rings'],
                                      filter_macrocycles_flag=sample_dict['filter_macrocycles'],
                                      use_hs_flag=sample_dict['use_hs'])

    # SAMPLING LOOP
    samples = get_samples_from_sampler(latent_sampler=latent_sampler,
                          model=Da_Model,
                          get_molecules_fn=mol_gen_fn,
                          n_samples=sample_dict['n_samples'],
                          sample_size=sample_dict['sample_size'],
                          batch_size=sample_dict['batch_size'],
                          n_workers=sample_dict['n_workers'],
                          perturb_hist=sample_dict['perturb_hist'],
                          perturb_mode=sample_dict['perturb_mode'],
                          keep_invalid=sample_dict['keep_invalid'])
    
    # EVALUATE SAMPLES
    if exp_dict['calculate_vun'] is True:
        orig_smiles_ds = read_lines_list(exp_dict['smiles_dataset_path'])
        eval_res = get_vun(samples, orig_smiles_ds)
        print(eval_res)

    # SAVE SAMPLES
    out = get_validity(samples)
    smiles_list = out[3]
    write_lines_to_file(smiles_list, exp_dict['samples_path'])


if __name__ == '__main__':
    main()