import torch
from torch import nn
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops
from amcg_utils.gen_utils import get_true_mask
from losses import kl_loss
from pieces import GlobalEncoder, Decoder, AtomGenerator, Combiner, AtomEncoder, MolEncoder, ADJ_Dec, Bond_Pred, HeteroMLP

# MODEL
class AMCG(nn.Module):
    """
    AMCG model.

    Args:
        encoder (GlobalEncoder): The global encoder module.
        decoder (Decoder): The shared decoder module.
        generator (AtomGenerator): The molecular decoder - atomic generator module.
        combiner (Combiner): The combiner module.
        num_atom_types (int): The number of atom types.
        num_bond_types (int): The number of bond types.
        max_logstd (float, optional): The maximum value for the log standard deviation. Defaults to None.
    """
    def __init__(self, encoder: GlobalEncoder, decoder: Decoder, 
                 generator: AtomGenerator, combiner: Combiner, num_atom_types,
                 num_bond_types, max_logstd=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder # shared decoder
        self.generator = generator # molecular decoder - atomic generator
        self.combiner = combiner
        self.sumaggregator = SumAggregation()
        if max_logstd is None:
            self.max_logstd = 10
        else:
            self.max_logstd = max_logstd

        self.cel_loss_fn = torch.nn.CrossEntropyLoss()
        self.mse_loss_fn = torch.nn.MSELoss()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.atom_channels = self.encoder.atom_encoder.out_channels
        self.mol_channels = self.encoder.mol_encoder.out_channels

    def encode_batch(self, data_batch, return_all=False):
        """
        Encodes a batch of data.

        Args:
            data_batch (Batch): The input data batch.
            return_all (bool, optional): Whether to return mus and sigmas. Defaults to False.

        Returns:
            tuple: Atomic encodings and molecular encodings.
        """
        x = torch.cat((data_batch.x, data_batch.random_walk_pe), dim=-1)
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr
        batch = data_batch.batch
        atom_types = torch.argmax(data_batch.x[:, :self.num_atom_types], dim=-1)

        atom_mu, atom_logstd, mol_mu, mol_logstd, atom_z, mol_z = self.encoder(x=x, edge_index=edge_index, 
                                                                       edge_attr=edge_attr, batch=batch,
                                                                       atom_types=atom_types)
        if return_all:
            return atom_mu, atom_logstd, mol_mu, mol_logstd, atom_z, mol_z
        elif not return_all:
            return atom_z, mol_z

    def loss(self, x, pos_edge_index, neg_edge_index, edge_attr, batch,
             y, atom_types=None, num_atoms=None):
        """
        Computes the loss for the model.

        Args:
            x: The input features.
            pos_edge_index: The positive edge indices.
            neg_edge_index: The negative edge indices.
            edge_attr: The edge attributes.
            batch: The batch indices.
            y: The target properties.
            atom_types: The atom types. Defaults to None.
            num_atoms: The number of atoms. Defaults to None.

        Returns:
            tuple: The computed losses.
        """
        if atom_types is None:
            atom_types = torch.argmax(x[:,:self.num_atom_types], dim=-1)    

        if num_atoms is None:
            num_atoms = torch.unique(batch, return_counts=True)[1]
        
        target_hist = self.sumaggregator(x[:, :self.num_atom_types], batch)
        target_hydrogens = torch.argmax(x[:, self.num_atom_types+14:self.num_atom_types+19], dim=-1).to(torch.float)

        _, _, mol_mu, mol_logstd, atom_z, mol_z = self.encoder(x=x, edge_index=pos_edge_index, 
                                                                       edge_attr=edge_attr, batch=batch,
                                                                       atom_types=atom_types)

        mol_kl_loss = kl_loss(mol_mu, mol_logstd)
        mol_logstd = mol_logstd.clamp(max=self.max_logstd)

        out = self.combiner(atom_z, mol_z, atom_types, batch)
        true_edge_mask = get_true_mask(num_atoms, out.device)
        target_bond_types = torch.argmax(edge_attr[:,:self.num_bond_types], dim=-1)

        pos_loss, neg_loss, bond_loss, hs_loss = self.decoder.loss(x=out,
                                                                   atom_types=atom_types,
                                                                   pos_edge_index=pos_edge_index,
                                                                   neg_edge_index=neg_edge_index,
                                                                   true_edge_mask=true_edge_mask,
                                                                   target_bond_types=target_bond_types,
                                                                   target_hydrogens=target_hydrogens)

        hist_loss, recon_m_loss, prop_loss, recon_atoms = self.generator.loss(mol_z=mol_z,
                                                                              target_hist=target_hist, 
                                                                              target_atoms=out,
                                                                              target_props=y,
                                                                              return_atoms=True)

        recon_p_loss, recon_n_loss, recon_b_loss, recon_hs_loss = self.decoder.loss(x=recon_atoms,
                                                                                    atom_types=atom_types,
                                                                                    pos_edge_index=pos_edge_index,
                                                                                    neg_edge_index=neg_edge_index,
                                                                                    true_edge_mask=true_edge_mask,
                                                                                    target_bond_types=target_bond_types,
                                                                                    target_hydrogens=target_hydrogens)

        return mol_kl_loss, pos_loss, neg_loss, bond_loss, hist_loss, recon_p_loss, recon_n_loss, recon_b_loss, recon_m_loss, prop_loss, hs_loss, recon_hs_loss
    

    def infer_from_z(self, mol_z, return_props=False, perturb_hist=False, perturb_mode=None):
        """
        Inference step from the given latent embeddings via right branch.

        Args:
            mol_z: The molecular latent embeddings of shape (batch_size, mol_latent_dims).
            return_props (bool, optional): Whether to return the predicted properties. Defaults to False.
            perturb_hist (bool, optional): Whether to perturb the histogram. Defaults to False.
            perturb_mode (str, optional): The perturbation mode. Defaults to None.

        Returns:
            tuple: The inferred adjacency matrix, atom types and bond types, batch indices, predicted hydrogens
            and molecular properties (if required). 
        """
        _, out, props, atom_types, batch = self.generator(mol_z, perturb_hist=perturb_hist, perturb_mode=perturb_mode)

        num_atoms = torch.unique(batch, return_counts=True)[1]    
        true_edge_mask = get_true_mask(num_atoms, mol_z.device)

        _, bond_pred, hs_pred, new_edge_index = self.decoder(x=out,
                                                             atom_types=atom_types,
                                                             true_edge_mask=true_edge_mask)
        if return_props:
            return new_edge_index, atom_types, bond_pred, batch, hs_pred, props
        return new_edge_index, atom_types, bond_pred, batch, hs_pred


    def infer_right(self, data_list, return_latent=False, return_props=False):
        """
        Inference step from data input to output, via right branch.

        Args:
            data_list: The list of Data objects.
            return_latent (bool, optional): Whether to return the latent embeddings. Defaults to False.
            return_props (bool, optional): Whether to return the predicted properties. Defaults to False.

        Returns:
            tuple: The inferred molecular tensors.
        """
        data = Batch.from_data_list(data_list)
        _, mol_z = self.encode_batch(data_batch=data, return_all=False)
        infer_out = self.infer_from_z(mol_z, return_props=return_props)
        if return_latent:
            return *infer_out, mol_z
        return infer_out
    

    def forward(self, data_list, device, prop_indices=None):
        """
        It computes the losses for the model. It is in the forward method to be used
        with the DataParallel module.

        Args:
            data_list: The list of data objects.
            device: The device to run the model on.
            prop_indices: The indices of the properties to predict. Defaults to None.

        Returns:
            tuple: The computed losses.
        """
        if prop_indices is None:
            prop_indices = [0]
        data = Batch.from_data_list(data_list).to(device)
        x = torch.cat((data.x, data.random_walk_pe), dim=-1)
        pos_edge_index=data.edge_index
        neg_edge_index=remove_self_loops(data.neg_edge_index)[0]
        edge_attr=data.edge_attr
        batch=data.batch
        if isinstance(prop_indices, list) and len(prop_indices) > 0:
            y = [data.y[:, idx] for idx in prop_indices]
        else:
            y = []
        
        (mol_kl_loss, pos_loss, neg_loss, bond_loss, 
         hist_loss, recon_p_loss, recon_n_loss, 
         recon_b_loss, recon_m_loss, prop_loss, 
         hs_loss, recon_hs_loss) = self.loss(x=x,
                                             pos_edge_index=pos_edge_index,
                                             neg_edge_index=neg_edge_index,
                                             edge_attr=edge_attr,
                                             batch=batch,
                                             y=y)
        return (mol_kl_loss, pos_loss, neg_loss, bond_loss,
                hist_loss, recon_p_loss, recon_n_loss,
                recon_b_loss, recon_m_loss, prop_loss,
                hs_loss, recon_hs_loss)


def get_amcg_qm9(num_properties=1):
    """
    Returns an instance of the AMCG model for QM9 dataset.

    Args:
        num_properties (int, optional): The number of properties to predict. Defaults to 1.

    Returns:
        AMCG: The AMCG model instance.
    """
    num_atom_types = 4
    num_bond_types = 4
    num_in_channels = 54+num_atom_types

    # build encoder
    atom_encoder = AtomEncoder(in_channels=num_in_channels, embedding_dim=512, 
                               hidden_channels=1024, out_channels=512, 
                               num_atom_types=num_atom_types)
    
    atom_n_channels = atom_encoder.out_channels
    mol_n_channels = 1024+512
    
    mol_encoder = MolEncoder(in_channels=atom_n_channels, 
                             hidden_channels=1024, 
                             out_channels=mol_n_channels)
    encoder = GlobalEncoder(atom_encoder=atom_encoder, 
                            mol_encoder=mol_encoder)

    #build combiner
    combiner = Combiner(in_channels=atom_n_channels+mol_n_channels,
                        hidden_channels=1024, out_channels=1024,
                        num_atom_types=num_atom_types)
    
    #build generator
    generator = AtomGenerator(in_channels=mol_n_channels, 
                              num_atom_types=num_atom_types,
                              generator_latent_dims=combiner.out_channels*2,  # can be changed
                              atom_latent_dims=combiner.out_channels,
                              num_properties=num_properties)

    #build decoder
    adj_dec = ADJ_Dec()
    bond_predictor = Bond_Pred(in_channels=combiner.out_channels, embedding_dim=2048, 
                               za_hc=2048, c_hc=2048, bond_types=num_bond_types)
    hs_predictor = HeteroMLP(in_channels=combiner.out_channels, 
                             hidden_channels=[1024,512,1], num_classes=num_atom_types)
    decoder = Decoder(adj_decoder=adj_dec, 
                      bond_classifier=bond_predictor, 
                      hs_predictor=hs_predictor)

    # BUILD NET
    model = AMCG(encoder=encoder,
                     decoder=decoder,
                     generator=generator,
                     combiner=combiner,
                     num_atom_types=num_atom_types,
                     num_bond_types=num_bond_types)
    return model


def get_amcg_zinc(num_properties=1):
    """
    Returns an instance of the AMCG model for the ZINC dataset.

    Parameters:
    - num_properties (int): The number of properties to predict.

    Returns:
    - model (AMCG): An instance of the AMCG model.

    """
    num_atom_types = 9
    num_bond_types = 4
    num_in_channels = 54+num_atom_types

    # build encoder

    atom_encoder = AtomEncoder(in_channels=num_in_channels, embedding_dim=512,
                               hidden_channels=1024, out_channels=512,
                               num_atom_types=num_atom_types)
    
    atom_n_channels = atom_encoder.out_channels
    mol_n_channels = 1024+512
    
    mol_encoder = MolEncoder(in_channels=atom_n_channels, 
                             hidden_channels=1024, 
                             out_channels=mol_n_channels)
    
    encoder = GlobalEncoder(atom_encoder=atom_encoder, 
                            mol_encoder=mol_encoder)
    
    # build combiner

    combiner = Combiner(in_channels=atom_n_channels+mol_n_channels,
                        hidden_channels=1024, out_channels=1024,
                        num_atom_types=num_atom_types)
    
    # build generator

    generator = AtomGenerator(in_channels=mol_n_channels,
                              num_atom_types=num_atom_types,
                              generator_latent_dims=combiner.out_channels*2,
                              atom_latent_dims=combiner.out_channels,
                              num_properties=num_properties)
    
    # build decoder

    adj_dec = ADJ_Dec()
    bond_predictor = Bond_Pred(in_channels=combiner.out_channels, embedding_dim=2048,
                               za_hc=2048, c_hc=2048, bond_types=num_bond_types)
    hs_predictor = HeteroMLP(in_channels=combiner.out_channels, 
                             hidden_channels=[1024,512,1], num_classes=num_atom_types)
                             
                                
    decoder = Decoder(adj_decoder=adj_dec,
                      bond_classifier=bond_predictor,
                      hs_predictor=hs_predictor)

    # BUILD NET

    model = AMCG(encoder=encoder,
                 decoder=decoder,
                 generator=generator,
                 combiner=combiner,
                 num_atom_types=num_atom_types,
                 num_bond_types=num_bond_types)
    
    return model