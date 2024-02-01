
import torch
from torch_geometric.utils import to_dense_adj
from amcg_utils.gen_utils import write_to_log


def kl_loss(mu, logstd):
    """
    Computes the Kullback-Leibler (KL) divergence loss.

    Args:
        mu (torch.Tensor): The mean of the distribution.
        logstd (torch.Tensor): The logarithm of the standard deviation of the distribution.

    Returns:
        torch.Tensor: The KL divergence loss.
    """
    return -0.5 * torch.mean(
        torch.mean(
            1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1
        )
    )


def adj_recon_loss(adj_pred, pos_edge_index, neg_edge_index, pos_dense=None, neg_dense=None):
    """
    Computes the adjacency reconstruction loss.

    Args:
        adj_pred (torch.Tensor): The predicted adjacency matrix.
        pos_edge_index (torch.Tensor): The positive (present) edge indices.
        neg_edge_index (torch.Tensor): The negative (not present) edge indices.
        pos_dense (torch.Tensor, optional): The dense positive adjacency matrix. Defaults to None.
        neg_dense (torch.Tensor, optional): The dense negative adjacency matrix. Defaults to None.

    Returns:
        torch.Tensor: The losses related to present and not present edges.
    """
    with torch.no_grad():
        if pos_dense is None:  
            pos_dense = to_dense_adj(pos_edge_index, max_num_nodes=adj_pred.shape[-1])[0]
        if neg_dense is None:
            neg_dense = to_dense_adj(neg_edge_index, max_num_nodes=adj_pred.shape[-1])[0]

    log_ap = torch.log(adj_pred + 1e-15) / pos_edge_index.shape[1] 
    log_one_minus_ap = torch.log(1 - adj_pred + 1e-15) / neg_edge_index.shape[1]

    pos_loss = -torch.sum(torch.multiply(log_ap, pos_dense))
    neg_loss = -torch.sum(torch.multiply(log_one_minus_ap, neg_dense))

    return pos_loss, neg_loss
   

def target_prop_loss(prop, target_prop):
    """
    Computes the target property loss.

    Args:
        prop (list): The predicted properties.
        target_prop (list): The target properties.

    Returns:
        torch.Tensor: The target property loss.
    """
    assert len(prop) == len(target_prop)
    if len(prop) == 0:
        return torch.tensor(0, dtype=torch.float32)
    mse_loss = torch.nn.MSELoss()
    return sum([mse_loss(x, torch.unsqueeze(y, dim=-1)) for x,y in zip(prop, target_prop)])


def parallel_loss_fn(outputs, weights):
    """
    Aggregates the losses (used with DataParallel)

    Args:
        outputs (list): The list of output tensors.
        weights (list): The list of loss weights.

    Returns:
        torch.Tensor: The total loss.
        list: The individual losses.
        str: The formatted loss string.
    """
    pieces = [torch.mean(item) for item in outputs]
    if weights is None:
        weights = [1] * len(pieces)

    losses = [weight * piece for weight, piece in zip(weights, pieces)]

    loss = sum(losses)
  
    lines = [f"KL: {losses[0].cpu().item():.8f} - ",
             f"Pos: {losses[1].cpu().item():.8f} - ",    
             f"Neg: {losses[2].cpu().item():.8f} - ",    
             f"Bond: {losses[3].cpu().item():.8f} - ",
             f"Hist: {losses[4].cpu().item():.8f} - ",
             f"Recon P: {losses[5].cpu().item():.8f} - ",
             f"Recon N: {losses[6].cpu().item():.8f} - ",
             f"Recon B: {losses[7].cpu().item():.8f} - ",
             f"Recon Mu: {losses[8].cpu().item():.8f} - ",    
             f"Prop: {losses[9].cpu().item():.8f} - ",    
             f"Hs: {losses[10].cpu().item(): .8f} - ",    
             f"Recon Hs: {losses[11].cpu().item(): .8f} - ",    
             f"Full: {loss}"]
    line = "".join(lines)
    return loss, losses, line


def get_loss_weights(epoch, model_type='qm9'):
    """
    Returns the loss weights based on the epoch and model type.

    Args:
        epoch (int): The current epoch.
        model_type (str, optional): The model type. Defaults to 'qm9'.

    Returns:
        list: The loss weights.
    """
    if model_type == 'qm9':
        if epoch < 50:
            return [0, 20, 20, 1, 1, 1, 5, 5, 1, 1, 1, 1]
        elif epoch < 100:
            return [0, 5, 5, 1, 1, 1, 20, 20, 1, 1, 1, 1]
        elif epoch < 150:
            return [.2, 5, 5, 1, 1, 1, 20, 20, 1, 1, 1, 1]
        elif epoch < 200:
            return [1, 5, 5, 1, 1, 1, 20, 20, 1, 1, 1, 1]
        else:
            return [5, 5, 5, 1, 1, 1, 20, 20, 1, 1, 1, 1]
    
    elif model_type == 'zinc':
        if epoch < 50:
            return [0, 20, 20, 1, 1, 1, 5, 5, 1, 1, 1, 1]
        elif epoch < 150:
            return [0, 5, 5, 1, 1, 1, 20, 20, 1, 1, 0, 0]
        elif epoch < 200:
            return [.2, 5, 5, 1, 1, 1, 20, 20, 1, 1, 0, 0]
        elif epoch < 250:
            return [1, 5, 5, 1, 1, 1, 20, 20, 1, 1, 0, 0]
        else:
            return [2, 5, 5, 1, 1, 1, 20, 20, 1, 1, 0, 0]


def loss_fn(outputs, epoch, count, logging_path, model_type='qm9'):
    """
    Computes the loss function.

    Args:
        outputs (list): The list of output tensors.
        epoch (int): The current epoch.
        count (int): The current count.
        logging_path (str): The path to the log file.
        model_type (str, optional): The model type. Defaults to 'qm9'.

    Returns:
        torch.Tensor: The total loss.
    """
    weights = get_loss_weights(epoch, model_type=model_type) # [kl, pos, neg, bond, hist, recon_p, recon_n, recon_b, recon_mu, prop, hs, recon_hs]
                                      # change weights here or comment to use default
    loss, _, line = parallel_loss_fn(outputs, weights)
    if count % 5 == 0:
        write_to_log(logging_path, line)
    return loss