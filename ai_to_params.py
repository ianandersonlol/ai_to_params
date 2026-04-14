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
import logging
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

from Bio.PDB import MMCIFParser, PDBParser, Structure, Model, Chain, Residue, Atom
from Bio.PDB.Atom import Atom as BioPythonAtom
from Bio.PDB.Residue import Residue as BioPythonResidue

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from rosetta_params_utils import (
    MolfileAtom, MolfileBond, Bond, r3,
    add_fields_to_atoms, add_fields_to_bonds,
    find_virtual_atoms, uniquify_atom_names,
    check_bond_count, check_aromaticity,
    assign_rosetta_types, assign_mm_types, assign_centroid_types,
    assign_partial_charges, assign_rotatable_bonds, assign_rigid_ids,
    build_fragment_trees, assign_internal_coords,
    write_param_file, write_ligand_pdb, choose_neighbor_atom,
)
from constants import (
    COVALENT_RADII, IGNORE_RESIDUES, METAL_ELEMENTS,
    LIGAND_CHAIN_IDS, DEFAULT_BOND_TOLERANCE,
)

logger = logging.getLogger(__name__)


@dataclass
class LigandData:
    """Data structure to hold information about a unique ligand"""
    original_name: str
    sanitized_name: str
    atoms: List[MolfileAtom]
    bonds: List[MolfileBond]
    residue: BioPythonResidue

    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []
        if self.bonds is None:
            self.bonds = []


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert AI modeling outputs (AlphaFold3, Chai, Boltz) to Rosetta params files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ai_to_params.py -cif AF3_output.cif -prefix my_complex
    python ai_to_params.py -cif Chai_output.pdb -prefix ligand_set --clobber
        """
    )

    parser.add_argument("-cif", "--cif", "--input", "-i",
                        required=True,
                        dest="input_file",
                        help="Path to input structure file (.cif, .mmcif, .pdb, or .ent)",
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

    parser.add_argument("--bond-tolerance",
                        type=float,
                        default=DEFAULT_BOND_TOLERANCE,
                        help=f"Bond inference tolerance in Angstroms (default: {DEFAULT_BOND_TOLERANCE})")

    parser.add_argument("--output-dir",
                        default=".",
                        help="Directory to write output files (default: current directory)",
                        metavar="DIR")

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        default=False,
                        help="Enable verbose/debug logging")

    return parser.parse_args()


def parse_structure_file(path: str) -> Structure:
    """
    Parse an mmCIF or PDB file and return a BioPython Structure object.

    Format is auto-detected from the file extension.

    Args:
        path: Path to a .cif/.mmcif or .pdb/.ent file

    Returns:
        BioPython Structure object
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    elif ext in ('.pdb', '.ent'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .cif/.mmcif or .pdb/.ent"
        )

    structure_id = os.path.splitext(os.path.basename(path))[0]
    structure = parser.get_structure(structure_id, path)

    if structure is None:
        raise ValueError(f"Failed to parse structure file '{path}'")

    return structure


# Backward-compatible alias
parse_cif_file = parse_structure_file


def clean_ligand_name(original_name: str) -> str:
    """
    Clean a ligand name to make it suitable for Rosetta (<=3 chars, no conflicts)

    Args:
        original_name: Original residue name from CIF

    Returns:
        Cleaned name suitable for Rosetta
    """
    name = original_name.strip().upper()

    prefixes_to_remove = ['LIG_', 'LIGAND_', 'HET_', 'SMALL_']
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    name = re.sub(r'\d+$', '', name)

    if len(name) > 3:
        name = name[:3]

    if not name:
        name = "LIG"

    return name


def _is_metal_ligand(residue) -> bool:
    """Check if a residue is a pure metal ligand (all atoms are metals).

    Single-atom metals (Fe, Zn, Mg) return True — Rosetta handles these
    natively without .params files. Multi-atom residues that mix metals
    and non-metals (e.g. [2Fe-2S] clusters) return False — these need
    .params files like any other ligand.
    """
    atoms = list(residue)
    if not atoms:
        return False
    return all(
        (a.element.upper() if a.element else a.get_name().strip()[:2].upper()) in METAL_ELEMENTS
        for a in atoms
    )


def identify_and_extract_ligands(structure: Structure, use_clean_names: bool = False) -> List[LigandData]:
    """
    Identify and extract ligands from the BioPython structure

    Args:
        structure: BioPython Structure object
        use_clean_names: If True, use cleaned names (FRU, ATP). If False, use systematic (L01, L02)

    Returns:
        List of LigandData objects for unique ligands
    """
    ligand_residues = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()

                if residue.id[0] == ' ':
                    continue

                if res_name in IGNORE_RESIDUES:
                    continue

                if _is_metal_ligand(residue):
                    elements = [a.element.upper() if a.element else '?' for a in residue]
                    logger.info(f"Found pure metal ligand: {res_name} ({', '.join(elements)})")
                elif len(residue) == 1:
                    logger.info(f"Skipping single-atom non-metal ligand: {res_name}")
                    continue

                if res_name not in ligand_residues:
                    ligand_residues[res_name] = []
                ligand_residues[res_name].append(residue)

    unique_ligands = []
    ligand_counter = 1
    used_names = set()

    for original_name, residues in ligand_residues.items():
        template_residue = residues[0]
        is_metal = _is_metal_ligand(template_residue)

        if is_metal:
            atom = list(template_residue)[0]
            element = atom.element.upper() if atom.element else atom.get_name().strip()[0]
            sanitized_name = element
        elif use_clean_names:
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

        atoms = []
        for atom in template_residue:
            mol_atom = create_molfile_atom_from_biopython(atom, is_metal)
            atoms.append(mol_atom)

        ligand_data = LigandData(
            original_name=original_name,
            sanitized_name=sanitized_name,
            atoms=atoms,
            bonds=[],
            residue=template_residue
        )

        unique_ligands.append(ligand_data)
        ligand_counter += 1

        logger.info(f"Found ligand: {original_name} -> {sanitized_name} ({len(atoms)} atoms)")

    return unique_ligands


def create_molfile_atom_from_biopython(biopython_atom: BioPythonAtom, is_metal: bool = False) -> MolfileAtom:
    """Convert a BioPython Atom to a MolfileAtom object"""
    coord = biopython_atom.get_coord()

    atom = MolfileAtom()
    atom.name = sanitize_atom_name(biopython_atom.get_name().strip(), is_metal)
    atom.elem = biopython_atom.element.upper() if biopython_atom.element else atom.name[0]
    atom.x = float(coord[0])
    atom.y = float(coord[1])
    atom.z = float(coord[2])

    atom.partial_charge = None
    atom.bonds = []
    atom.heavy_bonds = []
    atom.is_H = (atom.elem == 'H')

    atom.ros_type = ""
    atom.mm_type = ""
    atom.fragment_id = 1
    atom.is_virtual = False

    return atom


def _build_rdkit_mol(atoms: List[MolfileAtom]) -> Chem.RWMol:
    """
    Build an RDKit editable molecule from MolfileAtom list for bond perception.

    Args:
        atoms: List of MolfileAtom objects with 3D coordinates

    Returns:
        RDKit RWMol with 3D coordinates set
    """
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))

    for i, atom in enumerate(atoms):
        # RDKit expects properly-cased element symbols (e.g. "Fe" not "FE")
        rd_atom = Chem.Atom(atom.elem.capitalize())
        mol.AddAtom(rd_atom)
        conf.SetAtomPosition(i, (atom.x, atom.y, atom.z))

    mol.AddConformer(conf, assignId=True)
    return mol



def infer_bonds_rdkit(atoms: List[MolfileAtom]) -> List[MolfileBond]:
    """
    Infer bond connectivity using RDKit's DetermineConnectivity.

    Uses RDKit to perceive which atoms are bonded from 3D coordinates.
    Bond orders are set to SINGLE (Rosetta parameterization works with
    single bonds; DetermineBondOrders is too slow for large ligands).

    Args:
        atoms: List of MolfileAtom objects

    Returns:
        List of MolfileBond objects
    """
    mol = _build_rdkit_mol(atoms)

    try:
        rdDetermineBonds.DetermineConnectivity(mol)
    except Exception as e:
        logger.warning(f"RDKit connectivity failed ({e}), falling back to distance-based")
        return infer_bonds_distance(atoms)

    bonds = []
    atom_bonds = {atom: [] for atom in atoms}

    for rd_bond in mol.GetBonds():
        idx1 = rd_bond.GetBeginAtomIdx()
        idx2 = rd_bond.GetEndAtomIdx()
        atom1, atom2 = atoms[idx1], atoms[idx2]
        bond = MolfileBond()
        bond.a1 = atom1
        bond.a2 = atom2
        bond.order = Bond.SINGLE
        bond.is_ring = rd_bond.IsInRing()

        mirror_bond = MolfileBond()
        mirror_bond.a1 = atom2
        mirror_bond.a2 = atom1
        mirror_bond.order = Bond.SINGLE
        mirror_bond.is_ring = rd_bond.IsInRing()

        bond.mirror = mirror_bond
        mirror_bond.mirror = bond

        bonds.append(bond)
        atom_bonds[atom1].append(bond)
        atom_bonds[atom2].append(mirror_bond)

    # Set ring info on atoms
    ring_info = mol.GetRingInfo()
    for i, atom in enumerate(atoms):
        atom.bonds = atom_bonds[atom]
        atom.heavy_bonds = [b for b in atom.bonds if not b.a2.is_H]
        atom.is_ring = ring_info.NumAtomRings(i) > 0
        if atom.is_ring:
            sizes = ring_info.AtomRingSizes(i)
            atom.ring_size = min(sizes) if sizes else 0

    logger.info(f"RDKit inferred {len(bonds)} bonds for {len(atoms)} atoms")

    return bonds


def infer_bonds_distance(atoms: List[MolfileAtom], tolerance: float = DEFAULT_BOND_TOLERANCE) -> List[MolfileBond]:
    """
    Fallback: infer bonds based on covalent radii and distances (all single bonds).

    Args:
        atoms: List of MolfileAtom objects
        tolerance: Additional tolerance to add to sum of covalent radii (Angstroms)

    Returns:
        List of MolfileBond objects (all single bond order)
    """
    bonds = []
    atom_bonds = {atom: [] for atom in atoms}

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            atom1, atom2 = atoms[i], atoms[j]

            dx = atom1.x - atom2.x
            dy = atom1.y - atom2.y
            dz = atom1.z - atom2.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            radius1 = COVALENT_RADII.get(atom1.elem, 1.5)
            radius2 = COVALENT_RADII.get(atom2.elem, 1.5)

            if distance < radius1 + radius2 + tolerance:
                bond = MolfileBond()
                bond.a1 = atom1
                bond.a2 = atom2
                bond.order = Bond.SINGLE
                bond.is_ring = False

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

    logger.info(f"Distance-based: inferred {len(bonds)} bonds for {len(atoms)} atoms")

    return bonds


def infer_bonds(atoms: List[MolfileAtom], tolerance: float = DEFAULT_BOND_TOLERANCE) -> List[MolfileBond]:
    """
    Infer bonds between atoms, using RDKit for bond order perception.

    Tries RDKit first for accurate bond orders and aromaticity detection.
    Falls back to distance-based inference if RDKit fails.

    Args:
        atoms: List of MolfileAtom objects
        tolerance: Tolerance for distance-based fallback (Angstroms)

    Returns:
        List of MolfileBond objects
    """
    return infer_bonds_rdkit(atoms)


def parameterize_ligand(ligand: LigandData):
    """
    Run a ligand through the Rosetta parameterization pipeline

    Args:
        ligand: LigandData object with atoms and bonds
    """
    atoms = ligand.atoms
    bonds = ligand.bonds

    class MockMolfile:
        def __init__(self, atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds
            self.footer = []

    molfile = MockMolfile(atoms, bonds)

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

    logger.info(f"    Parameterized {len(atoms)} atoms, {len(bonds)} bonds")


def write_ligand_files(ligand: LigandData, prefix: str, clobber: bool, ligand_chain_id: str):
    """
    Write output files for a parameterized ligand

    Args:
        ligand: LigandData object that has been parameterized
        prefix: Output file prefix
        clobber: Whether to overwrite existing files
        ligand_chain_id: Chain ID for PDB output
    """
    class MockMolfile:
        def __init__(self, atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds
            self.footer = []

    molfile = MockMolfile(ligand.atoms, ligand.bonds)

    filename_base = ligand.original_name
    internal_name = ligand.sanitized_name

    params_filename = f"{prefix}_{filename_base}.params"
    if not clobber and os.path.exists(params_filename):
        logger.warning(f"    {params_filename} exists, skipping (use --clobber to overwrite)")
    else:
        write_param_file(
            params_filename,
            molfile,
            internal_name,
            1,     # fragment_id
            1,     # base_confs
            5000,  # max_confs
            None   # amino_acid
        )
        logger.info(f"    Wrote {params_filename} (residue code: {internal_name})")

    pdb_filename = f"{prefix}_{filename_base}.pdb"
    if not clobber and os.path.exists(pdb_filename):
        logger.warning(f"    {pdb_filename} exists, skipping (use --clobber to overwrite)")
    else:
        write_ligand_pdb(
            pdb_filename,
            molfile,
            molfile,
            internal_name,
            ctr=None,
            chain_id=ligand_chain_id
        )
        logger.info(f"    Wrote {pdb_filename} (residue code: {internal_name})")


def sanitize_atom_name(atom_name: str, is_metal: bool = False) -> str:
    """
    Sanitize atom names for Rosetta compatibility

    Args:
        atom_name: Original atom name from CIF file (e.g., "C_1", "N_2", "MG1_1")
        is_metal: True if this is a metal atom (uses element name only)

    Returns:
        Sanitized atom name

    Examples:
        C_1 -> C1, C_2 -> C2, MG_1 -> MG (if is_metal=True), CA -> CA (unchanged)
    """
    if is_metal:
        return re.sub(r'[0-9_].*', '', atom_name)

    if '_' in atom_name:
        parts = atom_name.split('_')
        base = parts[0]

        try:
            suffix_num = int(parts[1])
            base_match = re.match(r'([A-Z]+)(\d+)', base)
            if base_match:
                element = base_match.group(1)
                base_num = int(base_match.group(2))
                if base_num == 1:
                    return element
                else:
                    return f"{element}{base_num-1}"
            else:
                if suffix_num == 1:
                    return base
                else:
                    return f"{base}{suffix_num-1}"
        except (ValueError, IndexError):
            return atom_name.replace('_', '')

    match = re.match(r'([A-Z]+)(\d+)', atom_name)
    if match:
        element = match.group(1)
        num = int(match.group(2))
        if num == 1:
            return element
        else:
            return f"{element}{num-1}"

    return atom_name


def sanitize_chain_id(chain_id: str) -> str:
    """
    Sanitize chain IDs for Rosetta compatibility

    Examples:
        C_1 -> C, C_2 -> C1, A -> A (unchanged)
    """
    if '_' in chain_id:
        parts = chain_id.split('_')
        base = parts[0]
        try:
            num = int(parts[1])
            if num == 1:
                return base
            else:
                return f"{base}{num-1}"
        except (ValueError, IndexError):
            return chain_id.replace('_', '')

    return chain_id


def get_ligand_chain_id(ligand_index: int) -> str:
    """
    Generate single-character chain IDs for ligands (Rosetta compatible)

    Args:
        ligand_index: 0-based index of the ligand

    Returns:
        Single character chain ID (X, Y, Z, W, V, ...)
    """
    if ligand_index < len(LIGAND_CHAIN_IDS):
        return LIGAND_CHAIN_IDS[ligand_index]
    else:
        return LIGAND_CHAIN_IDS[ligand_index % len(LIGAND_CHAIN_IDS)]


def write_cleaned_complex_pdb(structure: Structure, name_mapping: Dict[str, str],
                             chain_mapping: Dict[str, str], prefix: str, clobber: bool):
    """
    Write a cleaned PDB of the entire complex with updated ligand names
    """
    cleaned_filename = f"{prefix}.pdb"
    if not clobber and os.path.exists(cleaned_filename):
        logger.warning(f"    {cleaned_filename} exists, skipping (use --clobber to overwrite)")
        return

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
                        output_res_name = name_mapping[res_name]
                        output_chain_id = chain_mapping[res_name]
                    else:
                        output_res_name = res_name
                        output_chain_id = sanitize_chain_id(original_chain_id)

                    for atom in residue:
                        atom_number += 1
                        coord = atom.get_coord()
                        x, y, z = coord[0], coord[1], coord[2]

                        element = atom.element.upper() if atom.element else atom.get_name().strip()[0]
                        is_metal = element in METAL_ELEMENTS

                        if is_hetatm:
                            atom_name = sanitize_atom_name(atom.get_name().strip(), is_metal)
                        else:
                            atom_name = atom.get_name().strip()

                        record_type = "HETATM" if is_hetatm else "ATOM  "

                        f.write(f"{record_type}{atom_number:5d} {atom_name:4s} {output_res_name:3s} "
                               f"{output_chain_id}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                               f"  1.00 20.00          {element:2s}  \n")

            f.write("TER" + " " * 77 + "\n")

        f.write("END" + " " * 77 + "\n")

    logger.info(f"    Wrote {cleaned_filename}")


def main():
    """Main function to orchestrate the CIF to params conversion"""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not os.path.exists(args.input_file):
        logger.error(f"Input structure file '{args.input_file}' does not exist")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    effective_prefix = os.path.join(args.output_dir, args.prefix)

    logger.info(f"AI to Params converter")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Output prefix: {args.prefix}")
    logger.info(f"Overwrite existing files: {args.clobber}")

    try:
        logger.info("\nStep 1: Parsing structure file...")
        structure = parse_structure_file(args.input_file)
        logger.info(f"Successfully parsed structure with {len(list(structure.get_models()))} model(s)")

        logger.info("\nStep 2: Identifying and extracting ligands...")
        ligands = identify_and_extract_ligands(structure, args.clean_names)
        logger.info(f"Found {len(ligands)} unique ligand type(s)")

        logger.info("\nStep 3: Inferring bonds for each ligand...")
        for ligand in ligands:
            if _is_metal_ligand(ligand.residue):
                logger.info(f"  Skipping bond inference for pure metal: {ligand.sanitized_name}")
                continue
            logger.info(f"  Processing {ligand.original_name} -> {ligand.sanitized_name}")
            ligand.bonds = infer_bonds(ligand.atoms, args.bond_tolerance)

        logger.info("\nStep 4: Converting to Rosetta parameterization...")
        for ligand in ligands:
            if _is_metal_ligand(ligand.residue):
                logger.info(f"  Skipping parameterization for metal: {ligand.sanitized_name}")
                continue

            logger.info(f"  Parameterizing {ligand.sanitized_name}...")
            parameterize_ligand(ligand)

        logger.info("\nStep 5: Writing output files...")
        for ligand_index, ligand in enumerate(ligands):
            if _is_metal_ligand(ligand.residue):
                logger.info(f"  Skipping file writing for metal: {ligand.sanitized_name}")
                continue

            logger.info(f"  Writing files for {ligand.sanitized_name}...")
            ligand_chain_id = get_ligand_chain_id(ligand_index)
            write_ligand_files(ligand, effective_prefix, args.clobber, ligand_chain_id)

        logger.info("\nStep 6: Writing complex PDB...")
        name_mapping = {lig.original_name: lig.sanitized_name for lig in ligands}
        chain_mapping = {lig.original_name: get_ligand_chain_id(i) for i, lig in enumerate(ligands)}
        write_cleaned_complex_pdb(structure, name_mapping, chain_mapping, effective_prefix, args.clobber)

        logger.info("Conversion completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
