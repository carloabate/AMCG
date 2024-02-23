import timeit
import sys
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataListLoader
from models import get_amcg_qm9, get_amcg_zinc
from losses import loss_fn
from amcg_utils.gen_utils import write_to_log, read_train_config_file

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # READ CONFIG

    config_path = sys.argv[1]
    exp_dict, train_dict = read_train_config_file(config_path)
    
    experiment_type = exp_dict['type']
    dataset_path = exp_dict['dataset_path']
    load_ckpt = exp_dict['load_ckpt']
    ckpt_path = exp_dict['ckpt_path']
    save_ckpt_path = exp_dict['save_ckpt_path']
    start_epoch = exp_dict['start_epoch']
    logging_path = exp_dict['logging_path']

    epochs = train_dict['epochs']
    learning_rate = train_dict['learning_rate']
    batch_size = train_dict['batch_size']
    num_properties = train_dict['num_properties']
    prop_indices = train_dict['prop_indices']

    write_to_log(logging_path, 'Device: ' + str(DEVICE))

    # LOAD DATA
    ds = torch.load(dataset_path)

    # INSTANTIATE MODEL
    if experiment_type == 'qm9':
        Da_Model = get_amcg_qm9(num_properties=num_properties)
    elif experiment_type == 'zinc':
        Da_Model = get_amcg_zinc(num_properties=num_properties)
    
    # LOAD MODEL
    if load_ckpt:
        weights_load_path = ckpt_path+'/weights.pkl'
        opt_load_path = ckpt_path+'/optimizer.pkl'

        state_dict = torch.load(weights_load_path, map_location='cpu')
        Da_Model.load_state_dict(state_dict)
    
    Da_Model = Da_Model.to(DEVICE)    
    
    #TRAINING LOOP
    if DEVICE == 'cpu':
        model = Da_Model
    else:
        model = DataParallel(Da_Model)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.01, patience=15)
    
    if load_ckpt:
        optimizer.load_state_dict(torch.load(opt_load_path))

    dl = DataListLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(start_epoch, start_epoch+epochs):
        write_to_log(logging_path, "Epoch: " + str(epoch))
        start = timeit.default_timer()
        count = 0
        for data_list in dl:
            model.train()
            optimizer.zero_grad()
            output = model(data_list, device=DEVICE, prop_indices=prop_indices)
            loss = loss_fn(output, epoch, count, logging_path, model_type=experiment_type)
            loss.backward()

            optimizer.step()
            count = count+1
        scheduler.step(loss)
        write_to_log(logging_path, "Time elapsed: " + str(timeit.default_timer() - start))
        
        if DEVICE == 'cpu':
            model_state_dict = model.state_dict()
        else:
            model_state_dict = model.module.state_dict()
        opt_state_dict = optimizer.state_dict()
        weights_save_path = save_ckpt_path+'/weights.pkl'
        opt_save_path = save_ckpt_path+'/optimizer.pkl'

        torch.save(model_state_dict, weights_save_path) # last epoch
        torch.save(opt_state_dict, opt_save_path)

        if epoch % 10 == 0: # backup
            torch.save(model_state_dict, weights_save_path + "epoch"+str(epoch)+".pkl")
            torch.save(opt_state_dict, opt_save_path + "epoch"+str(epoch)+".pkl")

if __name__ == "__main__":
    main()
