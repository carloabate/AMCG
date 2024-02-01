import torch
import torch.nn as nn
from torchvision.ops import MLP
from torch_geometric.nn.dense.linear import HeteroLinear, Linear
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import dense_to_sparse, remove_self_loops, unbatch
from amcg_utils.gen_utils import positional_encoding, flatten, at_from_hist, rearrange, get_perturbed_histogram
from losses import adj_recon_loss, target_prop_loss

# GENERIC PIECES
class HeteroMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroLinear(in_channels = in_channels,
                                        out_channels = hidden_channels[0],
                                        num_types = num_classes))
        self.out_channels = hidden_channels[-1]
        for i in range(len(hidden_channels)-1):
            self.layers.append(HeteroLinear(in_channels=hidden_channels[i],
                                            out_channels=hidden_channels[i+1],
                                            num_types=num_classes))
    def forward(self, x, atom_types):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x, atom_types))
        return self.layers[-1](x, atom_types)
    
    
class ZA_Embedder(nn.Module): # in_channels = z channels + mol channels
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear_1 = Linear(in_channels = in_channels, out_channels = hidden_channels)
        self.linear_2 = Linear(in_channels = hidden_channels, out_channels = out_channels)

    def forward(self, z, adj):
        z = torch.matmul(adj, z)
        z = torch.relu(self.linear_1(z))
        z = torch.matmul(adj, z)
        z = self.linear_2(z)
        return z

    
class ADJ_Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = InnerProductDecoder()
        self.dec_act = torch.nn.Sigmoid()
        self.dec_scaling = torch.nn.Parameter(torch.randn(1))
        self.dec_bias = torch.nn.Parameter(torch.randn(1))
        self.dec_scaling.requires_grad = True
        self.dec_bias.requires_grad = True
    
    def forward(self, z):
        adj_pred = self.decoder.forward_all(z, sigmoid=False)
        adj_pred = self.dec_scaling * adj_pred + self.dec_bias
        return self.dec_act(adj_pred)


class Bond_Pred(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, za_hc: int,
                 c_hc: int, bond_types: int) -> None:
        super().__init__()
        self.embedder = ZA_Embedder(in_channels=in_channels, hidden_channels=za_hc, out_channels=embedding_dim)
        self.classifier = MLP(in_channels=embedding_dim, hidden_channels=[c_hc,bond_types])

    def forward(self, x, adj, src, dst):
        bond_emb = self.embedder(x, adj)
        bond_pred = bond_emb[src] + bond_emb[dst]
        bond_pred = self.classifier(bond_pred)
        return bond_pred
    

#SUBNETWORKS

#ENCODER

class AtomEncoder(nn.Module):  #OUTPUT = OUT_CHANNELS * 2
    def __init__(self, in_channels, embedding_dim, hidden_channels, out_channels, num_atom_types):
        super(AtomEncoder, self).__init__()
        self.atomic_embedder = MLP(in_channels=in_channels, hidden_channels=[embedding_dim])
        self.gcn_shared_1 = GATv2Conv(embedding_dim, hidden_channels, heads=2, concat=True, edge_dim=13)
        self.shared_act = HeteroLinear(in_channels = hidden_channels*2, out_channels = hidden_channels*2, num_types=num_atom_types)

        self.gcn_mu = GATv2Conv(hidden_channels*2, out_channels, heads=2, concat=True, edge_dim=13)
        self.mu_act = HeteroLinear(in_channels=out_channels*2+in_channels, out_channels=out_channels*2, num_types=num_atom_types)
        self.mu_act_2 = HeteroLinear(in_channels=out_channels*2, out_channels=out_channels*2, num_types=num_atom_types)

        self.gcn_logvar = GATv2Conv(hidden_channels*2, out_channels, heads=2, concat=True, edge_dim=13)
        self.logvar_act = HeteroLinear(in_channels=out_channels*2+in_channels, out_channels=out_channels*2, num_types=num_atom_types)
        self.logvar_act_2 = HeteroLinear(in_channels=out_channels*2, out_channels=out_channels*2, num_types=num_atom_types)

        self.in_channels = in_channels
        self.out_channels = out_channels*2

    def forward(self, x, edge_index, edge_attr, atom_types):
        y = self.atomic_embedder(x)
        y = self.gcn_shared_1(y, edge_index, edge_attr)
        y = self.shared_act(y, atom_types)

        mu = self.gcn_mu(y, edge_index, edge_attr)
        mu = torch.cat([mu,x], dim=-1)
        mu = self.mu_act(mu, atom_types)
        mu = self.mu_act_2(mu, atom_types)

        logvar = self.gcn_logvar(y, edge_index, edge_attr)
        logvar = torch.cat([logvar,x], dim=-1)
        logvar = self.logvar_act(logvar, atom_types)
        logvar = self.logvar_act_2(logvar, atom_types)

        return mu, logvar


class MolEncoder(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        self.aggregator = SumAggregation()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mol_mlp = MLP(in_channels=in_channels, hidden_channels=[hidden_channels, hidden_channels, hidden_channels])

        self.mol_mu = MLP(in_channels=hidden_channels, hidden_channels=[out_channels, out_channels, out_channels])
        self.mol_std = MLP(in_channels=hidden_channels, hidden_channels=[out_channels, out_channels, out_channels])

    def forward(self, x, batch):
        out = self.mol_mlp(x)
        out = self.aggregator(out, batch)

        mu = self.mol_mu(out)
        std = self.mol_std(out)

        return mu, std


class GlobalEncoder(nn.Module):
    def __init__(self, atom_encoder: AtomEncoder, mol_encoder: MolEncoder, max_logstd=None) -> None:
        super().__init__()
        self.atom_encoder = atom_encoder
        self.mol_encoder = mol_encoder
        if max_logstd is None:
            self.max_logstd = 10
        else:
            self.max_logstd = max_logstd
        self.out_channels = self.atom_encoder.out_channels + self.mol_encoder.out_channels

    def forward(self, x, edge_index, edge_attr, atom_types, batch):
        atom_mu, atom_logstd = self.atom_encoder(x, edge_index, edge_attr, atom_types=atom_types)

        atom_logstd = atom_logstd.clamp(max=self.max_logstd)
        atom_z = atom_mu + torch.randn_like(atom_logstd) * torch.exp(atom_logstd)

        mol_mu, mol_logstd = self.mol_encoder(atom_z, batch)
        mol_z = mol_mu + torch.randn_like(mol_logstd) * torch.exp(mol_logstd)
                                      
        return atom_mu, atom_logstd, mol_mu, mol_logstd, atom_z, mol_z


# COMBINER
class Combiner(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_atom_types) -> None:
        super().__init__()
        self.z_MLP = HeteroMLP(in_channels=in_channels,
                               hidden_channels=[hidden_channels,
                                                hidden_channels,
                                                hidden_channels,
                                                out_channels],
                               num_classes=num_atom_types)
        self.out_channels = out_channels
    def forward(self, atom_z, mol_z, atom_types, batch):
        out = torch.cat([atom_z, mol_z[batch]], dim=-1)
        out = self.z_MLP(out, atom_types)
        return out


# SHARED DECODER
class Decoder(nn.Module):
    def __init__(self, adj_decoder: ADJ_Dec, bond_classifier: Bond_Pred, hs_predictor: HeteroMLP) -> None:
        super().__init__()
        self.adj_dec = adj_decoder
        self.bond_class = bond_classifier
        self.hs_pred = hs_predictor

    def forward(self, x, atom_types, true_edge_mask, edge_index=None):  # edge_index is where to evaluate bonds, 
                                                                        # if not provided evaluate everywhere the
                                                                        # model predicts the presence of a bond
        hs_pred = torch.relu(self.hs_pred(x, atom_types))
        adj_pred = self.adj_dec(x)
        true_adj_pred = torch.multiply(true_edge_mask, adj_pred)

        if edge_index is None:
            booladj = (true_adj_pred > 0.5).to(torch.int)
            new_edge_index, _ = dense_to_sparse(booladj)
            assert new_edge_index[0].shape[0] == torch.sum(booladj)

            new_edge_index, _ = remove_self_loops(new_edge_index)

        else:
            new_edge_index = edge_index

        src, dst = new_edge_index
        bond_pred = self.bond_class(x, true_adj_pred, src, dst)
        assert src.shape[0] == bond_pred.shape[0]

        return true_adj_pred, bond_pred, hs_pred, new_edge_index

    def loss(self, x, atom_types, pos_edge_index, neg_edge_index,
             true_edge_mask, target_bond_types, target_hydrogens,
             pos_dense=None, neg_dense=None):

        mse_loss = torch.nn.MSELoss()
        cel_loss = torch.nn.CrossEntropyLoss()

        true_adj_pred, bond_pred, hs_pred, _ = self.forward(x=x,
                                                            atom_types=atom_types,
                                                            true_edge_mask=true_edge_mask,
                                                            edge_index=pos_edge_index)  # bond types are inferred just for real bonds

        pos_loss, neg_loss = adj_recon_loss(true_adj_pred,
                                            pos_edge_index, neg_edge_index,
                                            pos_dense, neg_dense)
        bond_loss = cel_loss(bond_pred, target_bond_types)
        hs_loss = mse_loss(hs_pred, torch.unsqueeze(target_hydrogens, dim=-1))

        return pos_loss, neg_loss, bond_loss, hs_loss


class AtomGenerator(nn.Module):
    def __init__(self, in_channels, num_atom_types, generator_latent_dims,
                 atom_latent_dims, num_properties, max_logstd=None, num_workers=None) -> None:
        super().__init__()
        self.hist_pred = MLP(in_channels=in_channels, hidden_channels=[1024,num_atom_types])
        self.prop_preds = nn.ModuleList()
        self.mu_preds = nn.ModuleList()
        self.sigma_preds = nn.ModuleList()
        self.sumaggregator = SumAggregation()
        if max_logstd is None:
            self.max_logstd = 10
        else:
            self.max_logstd = max_logstd
        self.final_mlp = HeteroMLP(in_channels=generator_latent_dims, hidden_channels=[atom_latent_dims, atom_latent_dims], num_classes=num_atom_types)

        self.num_atom_types = num_atom_types
        self.num_properties = num_properties
        if num_workers is None:
            self.num_workers = 10
        else:
            self.num_workers = num_workers

        for _ in range(num_atom_types):
            self.mu_preds.append(MLP(in_channels=in_channels, hidden_channels=[generator_latent_dims, generator_latent_dims]))
            self.sigma_preds.append(MLP(in_channels=in_channels, hidden_channels=[generator_latent_dims, generator_latent_dims]))

        for _ in range(num_properties):
            self.prop_preds.append(MLP(in_channels=in_channels, hidden_channels=[2048,1024,512,1]))

    def forward(self, mol_z, hist=None,  # hist is used to sample, inferred if none
                perturb_hist=False, perturb_mode=None,
                return_pred_hist=False):  # return predicted hist anyway, useful for loss

        pred_hist = torch.relu(self.hist_pred(mol_z))
        if hist is None:
            hist = pred_hist

        props = [self.prop_preds[i](mol_z) for i in range(self.num_properties)]

        rhist = torch.round(hist)
        
        if perturb_hist:
            rhist = get_perturbed_histogram(rhist, perturb_mode, self.num_atom_types)
        
        lhist = list(torch.unbind(rhist.to(torch.long)))
        n_atoms = torch.sum(rhist, dim=-1).to(torch.long).tolist()

        for i, n_atom in enumerate(n_atoms):  # if empty molecule just give me a carbon
            if n_atom <= 0:
                lhist[i] = torch.tensor([1] + [0]*(self.num_atom_types-1)).to(mol_z.device)
                n_atoms[i] = 1

        batch = [[i] * n_atoms[i] for i in range(len(n_atoms))]
        batch = torch.tensor(sum(batch, []))

        atom_types = [at_from_hist(x, mol_z.device) for x in lhist]

        mus = torch.stack([self.mu_preds[i](mol_z) for i in range(self.num_atom_types)])  # shape (num_atom_types, batch_size, atom_latent_dims)
        sigmas = torch.stack([self.sigma_preds[i](mol_z) for i in range(self.num_atom_types)])  # shape (num_atom_types, batch_size, atom_latent_dims)

        mus = torch.transpose(mus, 0, 1)  # shape (batch_size, num_atom_types, atom_latent_dims)
        sigmas = torch.transpose(sigmas, 0, 1)  # shape (batch_size, num_atom_types, atom_latent_dims)
        sigmas = sigmas.clamp(max=self.max_logstd)

        mus = [x[y] for (x, y) in zip(mus, atom_types)]  # one per mol

        sigmas = [x[y] for (x, y) in zip(sigmas, atom_types)]  # one per mol

        recon_atoms = [x + torch.randn_like(y)*torch.exp(y) for (x, y) in zip(mus, sigmas)]  # one per mol

        recon_atoms = [unbatch(x, y) for (x, y) in zip(recon_atoms, atom_types)]  # one per atom_type set
        recon_atoms = flatten(recon_atoms)

        recon_atoms = [positional_encoding(x) for x in recon_atoms]  # encode local pos
        recon_atoms = torch.cat(recon_atoms, dim=0)

        atom_types = torch.cat(atom_types, dim=-1)
        hist = torch.stack(lhist)
        recon_atoms = self.final_mlp(recon_atoms, atom_types)

        if return_pred_hist:
            return hist, recon_atoms, props, atom_types, batch, pred_hist

        return hist, recon_atoms, props, atom_types, batch

    def loss(self, mol_z, target_hist, target_atoms, target_props, return_atoms=False):
        mse_loss = torch.nn.MSELoss()

        _, recon_atoms, props, atom_types, batch, pred_hist = self.forward(mol_z=mol_z,
                                                                           hist=target_hist,
                                                                           return_pred_hist=True)
        unbatched_target_atoms = unbatch(target_atoms, batch)
        unbatched_recon_mus = unbatch(recon_atoms, batch)
        unbatched_ats = unbatch(atom_types, batch)

        target_mus = [unbatch(x, y) for (x, y) in zip(unbatched_target_atoms, unbatched_ats)]
        predicted_mus = [unbatch(x, y) for (x, y) in zip(unbatched_recon_mus, unbatched_ats)]

        predicted_mus = flatten(predicted_mus)
        target_mus = flatten(target_mus)

        rearranged, recon_mu_losses = rearrange(predicted_mus, target_mus, n_workers=self.num_workers)
        recon_mu_loss = torch.sum(torch.stack(recon_mu_losses))
        hist_loss = mse_loss(pred_hist, target_hist)
        prop_loss = target_prop_loss(props, target_props)

        if return_atoms:
            return hist_loss, recon_mu_loss, prop_loss, rearranged
        return hist_loss, recon_mu_loss, prop_loss