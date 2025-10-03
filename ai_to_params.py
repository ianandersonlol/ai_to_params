#!/usr/bin/env python
"""
ai_to_params.py - Convert AI modeling outputs (AlphaFold3, Chai) to Rosetta params

This script parses mmCIF files from AI modeling tools like AlphaFold3 and Chai,
identifies unique ligand molecules, infers their bonds, and generates Rosetta
.params files for them using the core logic from molfile_to_params.py.

Usage:
    python ai_to_params.py -cif input.cif -prefix output_prefix [--clobber]

Based on the original molfile_to_params.py by Ian W. Davis.
Many thanks to Ian W. Davis for the foundational parameterization algorithms.
"""

import os
import sys
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

try:
    from Bio.PDB import MMCIFParser, Structure, Model, Chain, Residue, Atom
    from Bio.PDB.Atom import Atom as BioPythonAtom
    from Bio.PDB.Residue import Residue as BioPythonResidue
except ImportError:
    print("Error: BioPython is required but not installed.")
    print("Please install it with: pip install biopython")
    sys.exit(1)

try:
    from rosetta_params_utils import *
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure rosetta_params_utils.py is available")
    sys.exit(1)

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05,
    'F': 0.57, 'CL': 0.99, 'BR': 1.14, 'I': 1.33, 'MG': 1.41, 'CA': 1.76,
    'ZN': 1.22, 'FE': 1.32, 'K': 2.03, 'NA': 1.66
}

IGNORE_RESIDUES = {'HOH', 'WAT', 'SO4', 'PO4', 'CL'}

@dataclass
class LigandData:
    """Data structure to hold information about a unique ligand"""
    original_name: str
    sanitized_name: str
    atoms: List[MolfileAtom]
    bonds: List[MolfileBond]
    residue: BioPythonResidue

    def __post_init__(self):
        # Ensure atoms and bonds lists are initialized
        if self.atoms is None:
            self.atoms = []
        if self.bonds is None:
            self.bonds = []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert AI modeling outputs (AlphaFold3, Chai) to Rosetta params files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ai_to_params.py -cif AF3_output.cif -prefix my_complex
    python ai_to_params.py -cif Chai_output.cif -prefix ligand_set --clobber
        """
    )

    parser.add_argument("-cif", "--cif",
                        required=True,
                        help="Path to input mmCIF file from AI modeling tools",
                        metavar="FILE")

    parser.add_argument("-prefix", "--prefix",
                        required=True,
                        help="Output prefix for all generated files",
                        metavar="PREFIX")

    parser.add_argument("--clobber",
                        action="store_true",
                        default=False,
                        help="Allow overwriting existing files")

    parser.add_argument("--clean-names",
                        action="store_true",
                        default=False,
                        help="Use cleaned original names for 3-letter codes instead of systematic L01, L02")

    return parser.parse_args()

def parse_cif_file(cif_path: str) -> Structure:
    """
    Parse an mmCIF file and return a BioPython Structure object

    Args:
        cif_path: Path to the mmCIF file

    Returns:
        BioPython Structure object

    Raises:
        Exception: If parsing fails
    """
    try:
        parser = MMCIFParser(QUIET=True)
        # Extract the structure ID from the filename
        structure_id = os.path.splitext(os.path.basename(cif_path))[0]
        structure = parser.get_structure(structure_id, cif_path)

        if structure is None:
            raise ValueError("Failed to parse CIF file - structure is None")

        return structure

    except Exception as e:
        raise Exception(f"Failed to parse CIF file '{cif_path}': {e}")

def clean_ligand_name(original_name: str) -> str:
    """
    Clean a ligand name to make it suitable for Rosetta (≤3 chars, no conflicts)

    Args:
        original_name: Original residue name from CIF

    Returns:
        Cleaned name suitable for Rosetta

    Examples:
        LIG_FRU -> FRU
        UNL123 -> UNL
        GLUCOSE -> GLC
        ATP -> ATP
    """
    name = original_name.strip().upper()

    # Remove common prefixes
    prefixes_to_remove = ['LIG_', 'LIGAND_', 'HET_', 'SMALL_']
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Remove trailing numbers (UNL123 -> UNL)
    import re
    name = re.sub(r'\d+$', '', name)

    # Truncate to 3 characters if longer
    if len(name) > 3:
        name = name[:3]

    # Make sure it's not empty
    if not name:
        name = "LIG"

    return name

def identify_and_extract_ligands(structure: Structure, use_clean_names: bool = False) -> List[LigandData]:
    """
    Identify and extract ligands from the BioPython structure

    Args:
        structure: BioPython Structure object
        use_clean_names: If True, use cleaned names (FRU, ATP). If False (default), use systematic (L01, L02)

    Returns:
        List of LigandData objects for unique ligands

    The function:
    1. Iterates through all residues and identifies HETATM records
    2. Filters out common solvents and simple metal ions
    3. Groups unique ligands by their original 3-letter residue name
    4. Assigns systematic names (L01, L02) by default for internal consistency
    5. Keeps original names for user-friendly filenames
    """
    ligand_residues = {}
    unique_ligands = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()

                if residue.id[0] == ' ':  # Standard residue
                    continue

                if res_name in IGNORE_RESIDUES:
                    continue

                if len(residue) == 1:
                    atom = list(residue)[0]
                    element = atom.element.upper() if atom.element else atom.get_name().strip()[0]
                    metals = ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA']
                    if element in metals:
                        print(f"Found metal ligand: {res_name} (element: {element})")
                    else:
                        print(f"Skipping single-atom ligand: {res_name}")
                        continue

                if res_name not in ligand_residues:
                    ligand_residues[res_name] = []
                ligand_residues[res_name].append(residue)

    ligand_counter = 1
    name_mapping = {}
    used_names = set()

    for original_name, residues in ligand_residues.items():
        template_residue = residues[0]

        is_metal_ligand = False
        if len(template_residue) == 1:
            atom = list(template_residue)[0]
            element = atom.element.upper() if atom.element else atom.get_name().strip()[0]
            metals = ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA']
            if element in metals:
                is_metal_ligand = True
                sanitized_name = element

        if not is_metal_ligand:
            if use_clean_names:
                sanitized_name = clean_ligand_name(original_name)

                if sanitized_name in used_names:
                    counter = 2
                    base_name = sanitized_name
                    while sanitized_name in used_names:
                        if len(base_name) == 3:
                            sanitized_name = base_name[:2] + str(counter)
                        else:
                            sanitized_name = base_name + str(counter)
                        counter += 1

                used_names.add(sanitized_name)
            else:
                sanitized_name = f"L{ligand_counter:02d}"

        name_mapping[original_name] = sanitized_name

        atoms = []
        for atom in template_residue:
            mol_atom = create_molfile_atom_from_biopython(atom, is_metal_ligand)
            atoms.append(mol_atom)

        ligand_data = LigandData(
            original_name=original_name,
            sanitized_name=sanitized_name,
            atoms=atoms,
            bonds=[],  # Will be filled by bond inference
            residue=template_residue
        )

        unique_ligands.append(ligand_data)
        ligand_counter += 1

        print(f"Found ligand: {original_name} -> {sanitized_name} ({len(atoms)} atoms, files: {original_name}.*)")

    return unique_ligands

def create_molfile_atom_from_biopython(biopython_atom: BioPythonAtom, is_metal: bool = False) -> MolfileAtom:
    """
    Convert a BioPython Atom to a MolfileAtom-like object

    Args:
        biopython_atom: BioPython Atom object

    Returns:
        MolfileAtom object
    """
    coord = biopython_atom.get_coord()
    x, y, z = coord[0], coord[1], coord[2]

    atom = MolfileAtom()
    atom.name = sanitize_atom_name(biopython_atom.get_name().strip(), is_metal)
    atom.elem = biopython_atom.element.upper() if biopython_atom.element else atom.name[0]
    atom.x = float(x)
    atom.y = float(y)
    atom.z = float(z)

    atom.partial_charge = None
    atom.bonds = []
    atom.heavy_bonds = []
    atom.is_H = (atom.elem == 'H')

    atom.ros_type = ""
    atom.mm_type = ""
    atom.fragment_id = 1  # Default to fragment 1
    atom.is_virtual = False

    return atom

def infer_bonds(atoms: List[MolfileAtom], tolerance: float = 0.45) -> List[MolfileBond]:
    """
    Infer bonds between atoms based on covalent radii and distances

    Args:
        atoms: List of MolfileAtom objects
        tolerance: Additional tolerance to add to sum of covalent radii (Angstroms)

    Returns:
        List of MolfileBond objects representing inferred bonds

    The function calculates Euclidean distance between every pair of atoms.
    If the distance is less than the sum of their covalent radii plus tolerance,
    a bond is created between them.
    """
    bonds = []
    atom_bonds = {}

    for atom in atoms:
        atom_bonds[atom] = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            atom1, atom2 = atoms[i], atoms[j]

            dx = atom1.x - atom2.x
            dy = atom1.y - atom2.y
            dz = atom1.z - atom2.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            radius1 = COVALENT_RADII.get(atom1.elem, 1.5)
            radius2 = COVALENT_RADII.get(atom2.elem, 1.5)

            bond_threshold = radius1 + radius2 + tolerance
            if distance < bond_threshold:
                bond = MolfileBond()
                bond.a1 = atom1
                bond.a2 = atom2
                bond.order = Bond.SINGLE  # Default to single bond
                bond.is_ring = False  # Will be determined later if needed

                mirror_bond = MolfileBond()
                mirror_bond.a1 = atom2
                mirror_bond.a2 = atom1
                mirror_bond.order = Bond.SINGLE
                mirror_bond.is_ring = False

                bond.mirror = mirror_bond
                mirror_bond.mirror = bond

                bonds.append(bond)

                atom_bonds[atom1].append(bond)
                atom_bonds[atom2].append(mirror_bond)

    for atom in atoms:
        atom.bonds = atom_bonds[atom]
        atom.heavy_bonds = [b for b in atom.bonds if not b.a2.is_H]

    print(f"Inferred {len(bonds)} bonds for {len(atoms)} atoms")

    return bonds

def parameterize_ligand(ligand: LigandData):
    """
    Run a ligand through the Rosetta parameterization pipeline

    Args:
        ligand: LigandData object with atoms and bonds

    This function applies the same sequence of parameterization steps
    as the original molfile_to_params.py main function.
    """
    atoms = ligand.atoms
    bonds = ligand.bonds

    class MockMolfile:
        def __init__(self, atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds
            self.footer = []  # No footer for CIF-derived molecules

    molfile = MockMolfile(atoms, bonds)

    # (same sequence as original molfile_to_params.py main function)
    try:
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        check_bond_count(atoms)
        check_aromaticity(bonds)
        assign_rosetta_types(atoms)
        assign_mm_types(atoms)
        assign_centroid_types(atoms)

        assign_partial_charges(atoms, net_charge=0.0)
        assign_rotatable_bonds(bonds)
        assign_rigid_ids(atoms)

        for atom in atoms:
            atom.fragment_id = 1

        build_fragment_trees(molfile)
        assign_internal_coords(molfile)

        print(f"    Successfully parameterized {len(atoms)} atoms, {len(bonds)} bonds")

    except Exception as e:
        print(f"    Error during parameterization: {e}")
        raise

def write_ligand_files(ligand: LigandData, prefix: str, clobber: bool, ligand_chain_id: str):
    """
    Write output files for a parameterized ligand

    Args:
        ligand: LigandData object that has been parameterized
        prefix: Output file prefix
        clobber: Whether to overwrite existing files

    File naming strategy:
    - Filenames use original names: LIG_FRU.params, UNL123.pdb (user-friendly)
    - Internal 3-letter codes use systematic names: L01, L02 (Rosetta consistency)
    """
    class MockMolfile:
        def __init__(self, atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds
            self.footer = []

    molfile = MockMolfile(ligand.atoms, ligand.bonds)

    filename_base = ligand.original_name
    internal_name = ligand.sanitized_name  # L01, L02, etc.

    params_filename = f"{prefix}_{filename_base}.params"
    if not clobber and os.path.exists(params_filename):
        print(f"    Warning: {params_filename} exists, skipping (use --clobber to overwrite)")
    else:
        try:
            write_param_file(
                params_filename,
                molfile,
                internal_name,  # Use L01, L02 for 3-letter code
                1,  # fragment_id = 1
                1,  # base_confs = 1
                5000,  # max_confs = 5000
                None  # amino_acid = None
            )
            print(f"    Wrote {params_filename} (residue code: {internal_name})")
        except Exception as e:
            print(f"    Error writing params file: {e}")
            raise

    pdb_filename = f"{prefix}_{filename_base}.pdb"
    if not clobber and os.path.exists(pdb_filename):
        print(f"    Warning: {pdb_filename} exists, skipping (use --clobber to overwrite)")
    else:
        try:
            write_ligand_pdb(
                pdb_filename,
                molfile,    # template
                molfile,    # coordinates (same object)
                internal_name,  # Use L01, L02 for 3-letter residue code
                ctr=None,   # No recentering
                chain_id=ligand_chain_id  # Use X, Y, Z, Z1, Y1, X1, ...
            )
            print(f"    Wrote {pdb_filename} (residue code: {internal_name})")
        except Exception as e:
            print(f"    Error writing PDB file: {e}")
            raise

def sanitize_atom_name(atom_name: str, is_metal: bool = False) -> str:
    """
    Sanitize atom names for Rosetta compatibility
    Args:
        atom_name: Original atom name from CIF file (e.g., "C_1", "N_2", "MG1_1")
        is_metal: True if this is a metal atom (uses element name only)
    Returns:
        Sanitized atom name (e.g., "C1", "N2", "MG")
    Examples:
        C_1 -> C1
        C_2 -> C2
        MG_1 -> MG (if is_metal=True)
        MG1_1 -> MG (if is_metal=True)
        CA -> CA (unchanged)
    """
    if is_metal:
        # For metals, strip everything except the base element symbol
        import re
        element_only = re.sub(r'[0-9_].*', '', atom_name)  # Remove numbers, underscores, and everything after
        return element_only

    if '_' in atom_name:
        parts = atom_name.split('_')
        base = parts[0]

        try:
            suffix_num = int(parts[1])
            import re

            # Check if base has numbers (like C1, C2 in C1_1, C2_1)
            base_match = re.match(r'([A-Z]+)(\d+)', base)
            if base_match:
                # Base has numbers (C1_1, C2_1 format)
                element = base_match.group(1)  # C, O, N, etc.
                base_num = int(base_match.group(2))  # 1, 2, 3, etc.

                if base_num == 1:
                    return element  # C1_1 -> C
                else:
                    return f"{element}{base_num-1}"  # C2_1 -> C1, C3_1 -> C2
            else:
                # Base has no numbers (C_1, C_2 format)
                if suffix_num == 1:
                    return base  # C_1 -> C
                else:
                    return f"{base}{suffix_num-1}"  # C_2 -> C1
        except (ValueError, IndexError):
            # If we can't parse the number, just remove the underscore
            return atom_name.replace('_', '')

    # Handle non-underscore format (AF3 style: O1, P1, C1, C2)
    import re
    match = re.match(r'([A-Z]+)(\d+)', atom_name)
    if match:
        element = match.group(1)  # O, P, C, etc.
        num = int(match.group(2))  # 1, 2, 3, etc.

        if num == 1:
            return element  # O1 -> O, C1 -> C
        else:
            return f"{element}{num-1}"  # O2 -> O1, C2 -> C1

    return atom_name

def sanitize_chain_id(chain_id: str) -> str:
    """
    Sanitize chain IDs for Rosetta compatibility

    Args:
        chain_id: Original chain ID from CIF file

    Returns:
        Sanitized chain ID

    Examples:
        C_1 -> C
        C_2 -> C1
        A_1 -> A
        Z_10 -> Z9
    """
    if '_' in chain_id:
        parts = chain_id.split('_')
        base = parts[0]
        try:
            num = int(parts[1])
            if num == 1:
                return base  # C_1 -> C (start from base name)
            else:
                return f"{base}{num-1}"  # C_2 -> C1, C_3 -> C2, etc.
        except (ValueError, IndexError):
            return chain_id.replace('_', '')

    return chain_id

def get_ligand_chain_id(ligand_index: int) -> str:
    """
    Generate single-character chain IDs for ligands (Rosetta compatible)

    Args:
        ligand_index: 0-based index of the ligand

    Returns:
        Single character chain ID for the ligand

    Examples:
        0 -> X
        1 -> Y
        2 -> Z
        3 -> W (continue with remaining letters)
        4 -> V
        5 -> U
    """
    # Use X, Y, Z first, then continue backwards through alphabet
    # Avoid A, B, C, D, E, F, G, H (common protein chains)
    # and avoid I, L (confusing with 1)
    available_chains = ['X', 'Y', 'Z', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N', 'M', 'K', 'J']

    if ligand_index < len(available_chains):
        return available_chains[ligand_index]
    else:
        return available_chains[ligand_index % len(available_chains)]

def write_cleaned_complex_pdb(structure: Structure, name_mapping: Dict[str, str],
                             chain_mapping: Dict[str, str], prefix: str, clobber: bool):
    """
    Write a cleaned PDB of the entire complex with updated ligand names

    Args:
        structure: Original BioPython Structure object
        name_mapping: Dictionary mapping original ligand names to sanitized names
        chain_mapping: Dictionary mapping original ligand names to chain IDs
        prefix: Output file prefix
        clobber: Whether to overwrite existing files

    This function writes a PDB file of the entire complex where:
    - Protein ATOM records are sanitized for chain names (C_1 -> C)
    - HETATM residues use systematic names and proper chain IDs
    - Chain IDs follow X, Y, Z, Z1, Y1, X1, ... for ligands
    - The result is consistent with the generated .params files
    """
    cleaned_filename = f"{prefix}.pdb"
    if not clobber and os.path.exists(cleaned_filename):
        print(f"    Warning: {cleaned_filename} exists, skipping (use --clobber to overwrite)")
        return

    try:
        with open(cleaned_filename, 'w') as f:
            atom_number = 0

            for model in structure:
                for chain in model:
                    original_chain_id = chain.get_id()

                    for residue in chain:
                        res_name = residue.get_resname().strip()
                        res_id = residue.get_id()
                        res_num = res_id[1]

                        is_hetatm = (res_id[0] != ' ')

                        if is_hetatm and res_name in name_mapping:
                            # This is a ligand - use systematic naming and ligand chain ID
                            output_res_name = name_mapping[res_name]
                            output_chain_id = chain_mapping[res_name]
                        else:
                            # This is protein/standard - sanitize chain ID
                            output_res_name = res_name
                            output_chain_id = sanitize_chain_id(original_chain_id)

                        for atom in residue:
                            atom_number += 1
                            coord = atom.get_coord()
                            x, y, z = coord[0], coord[1], coord[2]

                            element = atom.element.upper() if atom.element else atom.get_name().strip()[0]
                            metals = ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA']
                            is_metal = element in metals

                            if is_hetatm:
                                atom_name = sanitize_atom_name(atom.get_name().strip(), is_metal)
                            else:
                                # For protein atoms, don't sanitize
                                atom_name = atom.get_name().strip()

                            record_type = "HETATM" if is_hetatm else "ATOM  "

                            f.write(f"{record_type}{atom_number:5d} {atom_name:4s} {output_res_name:3s} "
                                   f"{output_chain_id}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                                   f"  1.00 20.00          {element:2s}  \n")

                f.write("TER" + " " * 77 + "\n")

            f.write("END" + " " * 77 + "\n")

        print(f"    Wrote {cleaned_filename}")

    except Exception as e:
        print(f"    Error writing cleaned complex PDB: {e}")
        raise

def main():
    """Main function to orchestrate the CIF to params conversion"""
    args = parse_arguments()

    if not os.path.exists(args.cif):
        print(f"Error: Input CIF file '{args.cif}' does not exist")
        return 1

    print(f"AI to Params converter")
    print(f"Input CIF file: {args.cif}")
    print(f"Output prefix: {args.prefix}")
    print(f"Overwrite existing files: {args.clobber}")

    try:
        print("\nStep 1: Parsing CIF file...")
        structure = parse_cif_file(args.cif)
        print(f"Successfully parsed structure with {len(list(structure.get_models()))} model(s)")

        print("\nStep 2: Identifying and extracting ligands...")
        ligands = identify_and_extract_ligands(structure, args.clean_names)
        print(f"Found {len(ligands)} unique ligand type(s)")

        print("\nStep 3: Inferring bonds for each ligand...")
        for ligand in ligands:
            print(f"  Processing {ligand.original_name} -> {ligand.sanitized_name}")
            ligand.bonds = infer_bonds(ligand.atoms)

        print("\nStep 4: Converting to Rosetta parameterization...")
        for ligand in ligands:
            if len(ligand.atoms) == 1:
                element = ligand.atoms[0].elem
                metals = ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA']
                if element in metals:
                    print(f"  Skipping parameterization for metal: {ligand.sanitized_name}")
                    continue

            print(f"  Parameterizing {ligand.sanitized_name}...")
            parameterize_ligand(ligand)

        print("\nStep 5: Writing output files...")
        for ligand_index, ligand in enumerate(ligands):
            if len(ligand.atoms) == 1:
                element = ligand.atoms[0].elem
                metals = ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO', 'CD', 'PB', 'HG', 'K', 'NA']
                if element in metals:
                    print(f"  Skipping file writing for metal: {ligand.sanitized_name}")
                    continue

            print(f"  Writing files for {ligand.sanitized_name}...")
            ligand_chain_id = get_ligand_chain_id(ligand_index)
            write_ligand_files(ligand, args.prefix, args.clobber, ligand_chain_id)

        print("\nStep 6: Writing complex PDB...")
        name_mapping = {ligand.original_name: ligand.sanitized_name for ligand in ligands}
        chain_mapping = {ligand.original_name: get_ligand_chain_id(i) for i, ligand in enumerate(ligands)}
        write_cleaned_complex_pdb(structure, name_mapping, chain_mapping, args.prefix, args.clobber)

        print("Conversion completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())