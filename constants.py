"""
Shared constants for ai_to_params package.

Centralizes element data, residue filters, and metal definitions
that were previously duplicated across modules.
"""

# Covalent radii in Angstroms for bond inference
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05,
    'F': 0.57, 'CL': 0.99, 'BR': 1.14, 'I': 1.33, 'MG': 1.41, 'CA': 1.76,
    'ZN': 1.22, 'FE': 1.32, 'K': 2.03, 'NA': 1.66, 'MN': 1.39, 'CU': 1.32,
    'NI': 1.24, 'CO': 1.26, 'CD': 1.44, 'PB': 1.46, 'HG': 1.32, 'SE': 1.20,
}

# Residues to ignore during ligand extraction
IGNORE_RESIDUES = {'HOH', 'WAT', 'SO4', 'PO4', 'CL'}

# Metal elements that get special handling (single-atom ligands, no .params needed)
METAL_ELEMENTS = frozenset([
    'MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA'
])

# Available single-character chain IDs for ligands (avoids common protein chain IDs)
LIGAND_CHAIN_IDS = [
    'X', 'Y', 'Z', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N', 'M', 'K', 'J'
]

# Default bond inference tolerance in Angstroms
DEFAULT_BOND_TOLERANCE = 0.45
