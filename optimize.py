import torch
import sys
from amcg_utils.opt_utils import gradient_ascent_routine, merge_dizzs
from amcg_utils.eval_utils import evaluate_optimization
from amcg_utils.gen_utils import read_optim_config_file, read_lines_list
from amcg_utils.sampling_utils import get_molecules_gen_fn, get_mol_zs
from amcg_utils.build_mol_utils import get_clean_smiles
from models import get_amcg_qm9, get_amcg_zinc


def main():
    """
    Main function for running the optimization process.

    This function reads the configuration file, instantiates the model, loads the model weights,
    loads the data, performs optimization loop, evaluates the optimization if required,
    and saves the results.

    Args:
        None

    Returns:
        None
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # READ CONFIG
    config_path = sys.argv[1]
    exp_dict, opt_dict = read_optim_config_file(config_path)

    # INSTANTIATE MODEL
    if exp_dict['type'] == 'qm9':
        Da_Model = get_amcg_qm9()
    elif exp_dict['type'] == 'zinc':
        Da_Model = get_amcg_zinc()

    # LOAD MODEL
    weights_load_path = exp_dict['weights_path']
    state_dict = torch.load(weights_load_path, map_location='cpu')
    Da_Model.load_state_dict(state_dict)

    Da_Model = Da_Model.to(DEVICE)

    # LOAD DATA
    dataset = torch.load(exp_dict['test_dataset_path'])
    if exp_dict['evaluate_optimization']:
        if exp_dict['smiles_dataset_path'] is not None:
            orig_smiles = read_lines_list(exp_dict['smiles_dataset_path'])
        else:
            raise ValueError('Please provide a smiles dataset path to evaluate optimization')
    
    # GET SMILES
    test_smiles = [get_clean_smiles(x.info['canonical_smiles'], remove_hs=True) for x in dataset]

    # GET LATENTS
    mol_zs = get_mol_zs(dataset, Da_Model, opt_dict['dec_batch_size'], device=DEVICE)

    # GET MOLECULES GENERATION FUNCTION
    mol_gen_fn = get_molecules_gen_fn(fix_rings_flag=opt_dict['fix_rings'],
                                      filter_macrocycles_flag=opt_dict['filter_macrocycles'],
                                      use_hs_flag=opt_dict['use_hs'])

    # OPTIMIZATION LOOP
    summary_dizzs, smiles_dizzs = gradient_ascent_routine(samples=mol_zs,
                                                              predictor=Da_Model.generator.prop_preds[0],
                                                              model=Da_Model,
                                                              ga_bs=opt_dict['ga_batch_size'],
                                                              dec_bs=opt_dict['dec_batch_size'],
                                                              learning_rate=opt_dict['learning_rate'],
                                                              evaluation_step_size=opt_dict['eval_step_size'],
                                                              n_steps=opt_dict['n_steps'],
                                                              n_rejections=opt_dict['n_rejections'],
                                                              get_molecules_fn=mol_gen_fn)
    
    # GET RESULTS
    merged_summaries = merge_dizzs(summary_dizzs)
    merged_smiles = merge_dizzs(smiles_dizzs)

    # EVALUATION
    if exp_dict['evaluate_optimization']:
        (success_rate, opt_rate, _, 
         _, _, _) = evaluate_optimization(
            test_smiles=test_smiles,
            merged_smiles=merged_smiles,
            dataset_smiles=orig_smiles,
            prop_index=exp_dict['prop_index']
         )
        print('Validity: ' + str(len(merged_smiles)/len(dataset)))
        print('Optimized rate: ' + str(opt_rate))
        print('Success rate: ' + str(success_rate))


    # SAVE RESULTS
    torch.save(merged_summaries, exp_dict['opt_summary_path'])
    torch.save(merged_smiles, exp_dict['opt_smiles_path'])

if __name__ == "__main__":
    main()