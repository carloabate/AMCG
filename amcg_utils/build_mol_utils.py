import networkx as nx
import rdkit.Chem as Chem


def get_bond_type(n):
    """
    Get the RDKit bond type based on the input integer.

    Args:
        n (int): The input integer representing the bond type.

    Returns:
        RDKit bond type: The corresponding RDKit bond type.
    """
    if n == 0:
        return Chem.rdchem.BondType.SINGLE
    elif n == 1:
        return Chem.rdchem.BondType.DOUBLE
    elif n == 2:
        return Chem.rdchem.BondType.TRIPLE
    elif n == 3:
        return Chem.rdchem.BondType.AROMATIC


def get_atom_number(n):
    """
    Get the atomic number based on the input integer.

    Args:
        n (int): The input integer representing the atom type used by the network.

    Returns:
        int: The corresponding atomic number.
    """
    if n == 0:
        return 6
    if n == 1:
        return 7
    if n == 2:
        return 8
    if n == 3:
        return 9
    if n == 4:
        return 15
    if n == 5:
        return 16
    if n == 6:
        return 17
    if n == 7:
        return 35
    if n == 8:
        return 53


def get_atom_type(n):
    """
    Get the atom type based on the input atomic number.

    Args:
        n (int): The input atomic number.

    Returns:
        int: The corresponding atom type.
    """
    if n == 6:
        return 0
    if n == 7:
        return 1
    if n == 8:
        return 2
    if n == 9:
        return 3
    if n == 15:
        return 4
    if n == 16:
        return 5
    if n == 17:
        return 6
    if n == 35:
        return 7
    if n == 53:
        return 8


def get_valence(n):
    """
    Get the RDKit valence of an atom based on the input atomic number.

    Args:
        n (int): The input atomic number.

    Returns:
        int: The corresponding valence.
    """
    if n == 6:
        return 4
    if n == 7:
        return 3
    if n == 8:
        return 2
    if n == 9:
        return 1
    if n == 15:
        return 7
    if n == 16:
        return 6
    if n == 17:
        return 1
    if n == 35:
        return 1
    if n == 53:
        return 5
    

def get_clean_smiles(smiles, remove_hs=False):
    """
    Get the canonical SMILES representation of a molecule.

    Args:
        smiles (str): The input SMILES string.
        remove_hs (bool, optional): Whether to remove hydrogens from the molecule. Defaults to False.

    Returns:
        str: The canonical SMILES representation of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    mol_block = Chem.MolToMolBlock(mol, includeStereo=False, kekulize=True)
    mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return smiles


def get_clean_smiles_from_mol(mol, remove_hs=False):
    """
    Get the canonical SMILES representation of a molecule.

    Args:
        mol (RDKit Mol): The input RDKit molecule.
        remove_hs (bool, optional): Whether to remove hydrogens from the molecule. Defaults to False.

    Returns:
        str: The canonical SMILES representation of the molecule.
    """
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    mol_block = Chem.MolToMolBlock(mol, includeStereo=False, kekulize=True)
    mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return smiles


def mol_to_nx(mol):
    """
    Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (RDKit Mol): The input RDKit molecule.

    Returns:
        NetworkX Graph: The corresponding NetworkX graph.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def maximal_sets(lista):
    """
    Find the maximal sets from a list of sets.

    Args:
        lista (list): The input list of sets.

    Returns:
        list: The list of maximal sets.
    """
    sets = [set(x) for x in lista]
    maximal = []
    for i, s in enumerate(sets):
        if not any(s < other_set for other_set in sets):
            maximal.append(lista[i])
    return maximal


def one_step_fix(m):
    """
    Perform a one-step fix on a molecule by removing a bond in a cycle.

    Args:
        m (RDKit Mol): The input RDKit molecule.

    Returns:
        int: The result code (0 or 1) and the modified molecule.
    """
    G = mol_to_nx(m)
    # Convert to RWMol, which allows for molecule editing
    m = Chem.RWMol(m)
    cycles = list(nx.simple_cycles(G, 7))
    if len(cycles) > 100:
        return 0, None
    cycles = maximal_sets(cycles)
    for cycle in cycles:
        if len(cycle) > 4 and len(cycle) < 7:
            for i in range(len(cycle)):
                for j in range(i + 2, i + len(cycle) - 1):
                    to_scan = cycle * 2
                    if G.has_edge(to_scan[i], to_scan[j]):
                        try:
                            m.RemoveBond(to_scan[i],to_scan[j])
                            return 1, m.GetMol()
                        except:
                            pass
    return 0, m.GetMol()


def fix_rings(m):
    """
    Fix the rings in a molecule by iteratively removing bonds in cycles.

    Args:
        m (RDKit Mol): The input RDKit molecule.

    Returns:
        RDKit Mol: The modified RDKit molecule.
    """
    value = 1
    while value == 1:
        value, m = one_step_fix(m)
    return m


def filter_macrocycles(m, size=6):
    """
    Filter out macrocycles from a molecule based on a maximum cycle size.

    Args:
        m (RDKit Mol): The input RDKit molecule.
        size (int, optional): The maximum cycle size to filter. Defaults to 6.

    Returns:
        RDKit Mol or None: The filtered RDKit molecule or None if it contains a cycle larger than the specified size.
    """
    G = mol_to_nx(m)
    max_cycle_length = 0
    for cycle in nx.chordless_cycles(G):
        cycle_length = len(cycle)
        if cycle_length > max_cycle_length:
            max_cycle_length = cycle_length
    if max_cycle_length > size:
        return None
    else:
        return m
    

def first_try(num_atoms, edge_index, atom_numbers, bond_types, hs):
    """
    Build a molecule using all available information (bond types and hydrogen count).

    Args:
        num_atoms (int): The number of atoms in the molecule.
        edge_index (Tensor): The edge index of the molecule.
        atom_numbers (Tensor): The atomic numbers of the atoms.
        bond_types (Tensor): The bond types of the bonds.
        hs (Tensor): The number of explicit hydrogens for each atom.

    Returns:
        RDKit Mol: The built RDKit molecule.
    """
    mol = Chem.RWMol()
    hs = hs.tolist()
    aromatic = set()
    for k in range(num_atoms):
        atom = Chem.Atom(atom_numbers[k])
        atom.SetNumExplicitHs(hs[k])
        mol.AddAtom(atom)

    if edge_index is not None:
        edges = [tuple(i) for i in edge_index.t().tolist()]
        visited = set()
        
        for i in range(len(edges)):
            src, dst = edges[i]
            if tuple(sorted(edges[i])) in visited:
                continue
            if bond_types[i] == 3: #AROMATIC
                aromatic.add(src)
                aromatic.add(dst)
            bond_type = get_bond_type(bond_types[i])
            mol.AddBond(src, dst, bond_type)
            visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()
    return mol, aromatic


def second_try(num_atoms, edge_index, atom_numbers, bond_types, hs, aromatic):
    """
    Build a molecule using hydrogen information only on aromatic atoms, to be able to kekulize.

    Args:
        num_atoms (int): The number of atoms in the molecule.
        edge_index (Tensor): The edge index of the molecule.
        atom_numbers (Tensor): The atomic numbers of the atoms.
        bond_types (Tensor): The bond types of the bonds.
        hs (Tensor): The number of explicit hydrogens for each atom.
        aromatic (set): The set of aromatic atom indices.

    Returns:
        RDKit Mol: The built RDKit molecule.
    """
    newmol = Chem.RWMol()
    hs = hs.tolist()
    for k in range(num_atoms):
        atom = Chem.Atom(atom_numbers[k])
        if k in aromatic:
            atom.SetNumExplicitHs(hs[k])
        newmol.AddAtom(atom)
    if edge_index is not None:
        edges = [tuple(i) for i in edge_index.t().tolist()]
        visited = set()
        for i in range(len(edges)):
            src, dst = edges[i]
            if tuple(sorted(edges[i])) in visited:
                continue
            bond_type = get_bond_type(bond_types[i])
            newmol.AddBond(src, dst, bond_type)
            visited.add(tuple(sorted(edges[i])))

    newmol = newmol.GetMol()
    return newmol


def last_try(num_atoms, edge_index, atom_numbers_or_types, bond_types, is_atom_types=False):
    """
    Build a molecule using only bond type information.

    Args:
        num_atoms (int): The number of atoms in the molecule.
        edge_index (Tensor): The edge index of the molecule.
        atom_numbers_or_types (Tensor): The atomic numbers or atom types of the atoms.
        bond_types (Tensor): The bond types of the bonds.
        is_atom_types (bool, optional): Whether the input atom_numbers_or_types represent atom types. Defaults to False.

    Returns:
        RDKit Mol: The built RDKit molecule.
    """
    if is_atom_types:
        atom_numbers = [get_atom_number(x.item()) for x in atom_numbers_or_types]
    else:
        atom_numbers = atom_numbers_or_types

    mol = Chem.RWMol()
    for k in range(num_atoms):
        atom = Chem.Atom(atom_numbers[k])
        mol.AddAtom(atom)
    if edge_index is not None:
        edges = [tuple(i) for i in edge_index.t().tolist()]
        visited = set()

        for i in range(len(edges)):
            src, dst = edges[i]
            if tuple(sorted(edges[i])) in visited:
                continue
            
            bond_type = get_bond_type(bond_types[i])
            mol.AddBond(src, dst, bond_type)
            visited.add(tuple(sorted(edges[i])))
        
    mol = mol.GetMol()
    return mol


def build_mol(num_atoms, edge_index, atom_numbers_or_types,
                                bond_types, hs, is_atom_types=True):
    """
    Build a molecule based on the input information.

    Args:
        num_atoms (int): The number of atoms in the molecule.
        edge_index (Tensor): The edge index of the molecule.
        atom_numbers_or_types (Tensor): The atomic numbers or atom types of the atoms.
        bond_types (Tensor): The bond types of the bonds.
        hs (Tensor): The number of explicit hydrogens for each atom.
        is_atom_types (bool, optional): Whether the input atom_numbers_or_types represent atom types. Defaults to True.

    Returns:
        RDKit Mol: The built RDKit molecule.
    """
    if is_atom_types:
        atom_numbers = [get_atom_number(x.item()) for x in atom_numbers_or_types]
    else:
        atom_numbers = atom_numbers_or_types
    
    mol, aromatic = first_try(num_atoms, edge_index, atom_numbers, bond_types, hs) # check for agreement between bonds and hydrogens
    try:
        with io.capture_output() as _:
            Chem.SanitizeMol(mol)
        g = mol_to_nx(mol)
        assert nx.is_connected(g)
    except:
        mol = second_try(num_atoms, edge_index, atom_numbers, bond_types, hs, aromatic) # check for agreement just on aromatic hydrogens
        try:
            with io.capture_output() as _:
                Chem.SanitizeMol(mol)
            g = mol_to_nx(mol)
            assert nx.is_connected(g)
        except:
            mol = last_try(num_atoms, edge_index, atom_numbers, bond_types) # ignore hydrogens
    return mol