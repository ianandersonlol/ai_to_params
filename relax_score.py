#!/usr/bin/env python
"""
relax_score.py - Relax and score protein-ligand complexes using PyRosetta

Replaces subprocess calls to Rosetta binaries with direct PyRosetta API usage
for relaxation, scoring, and interface energy calculation.

Usage:
    python relax_score.py --prefix output_prefix [options]
    python relax_score.py --pdb complex.pdb --params L01.params,L02.params [options]
"""

import os
import sys
import argparse
import csv
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pyrosetta
from pyrosetta import pose_from_pdb, Pose
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreFunction
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector, ResidueIndexSelector, NotResidueSelector, AndResidueSelector,
)
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.constraint_movers import ConstraintSetMover

logger = logging.getLogger(__name__)


@dataclass
class LigandInfo:
    """Information about a ligand in the complex"""
    chain_id: str
    residue_name: str
    params_file: str


@dataclass
class RelaxResult:
    """Results from relaxing and scoring a single structure"""
    pdb_file: str
    total_score: float
    rmsd_to_input: Optional[float] = None
    interface_energies: Dict[str, float] = field(default_factory=dict)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Relax and score protein-ligand complexes using PyRosetta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python relax_score.py --prefix my_complex
    python relax_score.py --pdb complex.pdb --params L01.params,L02.params
    python relax_score.py --prefix my_complex --nstruct 5 --constraints catalytic.cst
        """
    )

    input_group = parser.add_argument_group('Input files')
    input_group.add_argument("--prefix",
                            help="Prefix for ai_to_params output (expects {prefix}.pdb and {prefix}_*.params)",
                            metavar="PREFIX")
    input_group.add_argument("--pdb",
                            help="Path to input PDB file (alternative to --prefix)",
                            metavar="FILE")
    input_group.add_argument("--params",
                            help="Comma-separated list of .params files (alternative to --prefix)",
                            metavar="FILE1,FILE2,...")

    constraint_group = parser.add_argument_group('Constraint options')
    constraint_group.add_argument("--constraints",
                                 help="Path to Rosetta constraint file",
                                 metavar="FILE")
    constraint_group.add_argument("--no-coord-constraints",
                                 action="store_true",
                                 help="Disable coordinate constraints (allow structure to move freely)")

    relax_group = parser.add_argument_group('Relaxation options')
    relax_group.add_argument("--nstruct",
                            type=int,
                            default=1,
                            help="Number of relaxed structures to generate (default: 1)")
    relax_group.add_argument("--relax-mode",
                            choices=["cartesian", "torsional"],
                            default="cartesian",
                            help="Relaxation mode (default: cartesian)")
    relax_group.add_argument("--score-function",
                            default="ref2015",
                            help="Rosetta score function (default: ref2015, auto-appends _cart for cartesian)")

    output_group = parser.add_argument_group('Output options')
    output_group.add_argument("--output-dir",
                             default=".",
                             help="Output directory for results (default: current directory)",
                             metavar="DIR")

    runtime_group = parser.add_argument_group('Runtime options')
    runtime_group.add_argument("-v", "--verbose",
                              action="store_true",
                              default=False,
                              help="Enable verbose/debug logging")

    args = parser.parse_args()

    if not args.prefix and not (args.pdb and args.params):
        parser.error("Must specify either --prefix OR both --pdb and --params")
    if args.prefix and (args.pdb or args.params):
        parser.error("Cannot specify both --prefix and --pdb/--params")
    if args.constraints and not os.path.exists(args.constraints):
        parser.error(f"Constraint file not found: {args.constraints}")

    return args


def find_input_files(args) -> Tuple[str, List[str]]:
    """Find input PDB and params files based on arguments."""
    if args.prefix:
        pdb_file = f"{args.prefix}.pdb"
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        params_files = glob.glob(f"{args.prefix}_*.params")
        if not params_files:
            raise FileNotFoundError(f"No params files found matching: {args.prefix}_*.params")

        logger.info(f"Found input files:")
        logger.info(f"  PDB: {pdb_file}")
        logger.info(f"  Params: {', '.join(params_files)}")
        return pdb_file, params_files
    else:
        pdb_file = args.pdb
        params_files = args.params.split(',')

        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        for pf in params_files:
            if not os.path.exists(pf):
                raise FileNotFoundError(f"Params file not found: {pf}")

        return pdb_file, params_files


def extract_residue_name_from_params(params_file: str) -> Optional[str]:
    """Extract the 3-letter residue code from a params file."""
    with open(params_file, 'r') as f:
        for line in f:
            if line.startswith('NAME'):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].strip()
    return None


def detect_ligand_chains(pose: Pose, params_files: List[str]) -> List[LigandInfo]:
    """
    Detect ligand chains in a pose by matching residue names to params files.

    Args:
        pose: PyRosetta Pose object
        params_files: List of params files

    Returns:
        List of LigandInfo objects
    """
    params_residue_names = {}
    for pf in params_files:
        name = extract_residue_name_from_params(pf)
        if name:
            params_residue_names[name] = pf

    ligands = []
    seen_chains = set()
    pdb_info = pose.pdb_info()

    for i in range(1, pose.total_residue() + 1):
        res_name = pose.residue(i).name3().strip()
        if res_name in params_residue_names:
            chain_id = pdb_info.chain(i)
            if chain_id not in seen_chains:
                ligands.append(LigandInfo(
                    chain_id=chain_id,
                    residue_name=res_name,
                    params_file=params_residue_names[res_name]
                ))
                seen_chains.add(chain_id)

    return ligands


def init_pyrosetta(params_files: List[str], score_function: str = "ref2015",
                   cartesian: bool = True):
    """
    Initialize PyRosetta with ligand params and appropriate settings.

    Args:
        params_files: List of .params files to load
        score_function: Base score function name
        cartesian: Whether to use cartesian scoring
    """
    extra_res = " ".join(f"-extra_res_fa {pf}" for pf in params_files)
    init_flags = (
        f"{extra_res} "
        "-load_PDB_components true "
        "-auto_setup_metals true "
        "-ignore_waters true "
        "-mute all "
        "-unmute protocols.relax"
    )

    pyrosetta.init(init_flags)


def create_score_function(base_name: str = "ref2015", cartesian: bool = True) -> ScoreFunction:
    """Create a Rosetta score function, appending _cart for cartesian mode."""
    sfxn_name = f"{base_name}_cart" if cartesian else base_name
    return ScoreFunctionFactory.create_score_function(sfxn_name)


def setup_fast_relax(sfxn: ScoreFunction, cartesian: bool = True,
                     constrain_to_start: bool = True) -> FastRelax:
    """
    Configure FastRelax with appropriate settings.

    Args:
        sfxn: Score function to use
        cartesian: Whether to use cartesian relaxation
        constrain_to_start: Whether to constrain to starting coordinates

    Returns:
        Configured FastRelax mover
    """
    relax = FastRelax()
    relax.set_scorefxn(sfxn)

    if cartesian:
        relax.cartesian(True)
        relax.minimize_bond_angles(True)
        relax.minimize_bond_lengths(True)

    if constrain_to_start:
        relax.constrain_relax_to_start_coords(True)
        relax.coord_constrain_sidechains(True)

    return relax


def apply_constraints(pose: Pose, constraint_file: str):
    """Apply constraints from a file to a pose."""
    cst_mover = ConstraintSetMover()
    cst_mover.constraint_file(constraint_file)
    cst_mover.apply(pose)
    logger.info(f"Applied constraints from {constraint_file}")


def relax_pose(pose: Pose, relax_mover: FastRelax) -> Pose:
    """
    Apply FastRelax to a pose and return the relaxed copy.

    Args:
        pose: Input pose (not modified)
        relax_mover: Configured FastRelax

    Returns:
        Relaxed pose (copy)
    """
    work_pose = pose.clone()
    relax_mover.apply(work_pose)
    return work_pose


def score_pose(pose: Pose, sfxn: ScoreFunction) -> float:
    """Score a pose and return total score."""
    return sfxn(pose)


def calculate_interface_energy(pose: Pose, sfxn: ScoreFunction,
                               ligand_chain: str) -> float:
    """
    Calculate protein-ligand interface energy using per-residue energy decomposition.

    Sums all pairwise interaction energies between the ligand chain and
    all other chains. This avoids the need to physically separate the ligand.

    Args:
        pose: Scored pose
        sfxn: Score function (must have already scored the pose)
        ligand_chain: Chain ID of the ligand

    Returns:
        Interface delta energy (negative = favorable binding)
    """
    sfxn(pose)

    pdb_info = pose.pdb_info()

    # Find ligand and non-ligand residue indices
    ligand_residues = set()
    protein_residues = set()
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) == ligand_chain:
            ligand_residues.add(i)
        else:
            protein_residues.add(i)

    if not ligand_residues:
        logger.warning(f"No residues found for chain {ligand_chain}")
        return 0.0

    # Get pairwise interaction energies from the energy graph
    energy_graph = pose.energies().energy_graph()
    interface_energy = 0.0

    for lig_res in ligand_residues:
        for prot_res in protein_residues:
            edge = energy_graph.find_energy_edge(lig_res, prot_res)
            if edge is None:
                continue
            interface_energy += edge.dot(sfxn.weights())

    return interface_energy


def calculate_interface_energy_separation(pose: Pose, sfxn: ScoreFunction,
                                          ligand_chain: str,
                                          distance: float = 500.0) -> float:
    """
    Calculate interface energy by translating the ligand away (standard method).

    interface_delta = score_complex - score_separated

    Args:
        pose: Scored pose
        sfxn: Score function
        ligand_chain: Chain ID of the ligand
        distance: Distance to translate ligand (Angstroms)

    Returns:
        Interface delta energy
    """
    complex_score = sfxn(pose)

    # Create separated pose
    separated = pose.clone()
    pdb_info = separated.pdb_info()

    for i in range(1, separated.total_residue() + 1):
        if pdb_info.chain(i) == ligand_chain:
            res = separated.residue(i)
            for j in range(1, res.natoms() + 1):
                xyz = res.xyz(j)
                new_xyz = pyrosetta.rosetta.numeric.xyzVector_double_t(
                    xyz.x + distance, xyz.y, xyz.z
                )
                separated.set_xyz(pyrosetta.rosetta.core.id.AtomID(j, i), new_xyz)

    separated_score = sfxn(separated)
    return complex_score - separated_score


def calculate_rmsd(pose1: Pose, pose2: Pose) -> float:
    """
    Calculate CA RMSD between two poses.

    Args:
        pose1: Reference pose
        pose2: Comparison pose

    Returns:
        CA RMSD in Angstroms
    """
    from pyrosetta.rosetta.core.scoring import CA_rmsd
    return CA_rmsd(pose1, pose2)


def write_summary_csv(results: List[Dict], output_file: str):
    """Write summary results to CSV file."""
    if not results:
        logger.warning("No results to write to CSV")
        return

    all_keys = []
    seen = set()
    for result in results:
        for key in result:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Wrote summary CSV: {output_file}")


def main():
    """Main function to orchestrate relaxation and scoring"""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Find input files
    try:
        pdb_file, params_files = find_input_files(args)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    # Determine settings
    cartesian = (args.relax_mode == "cartesian")
    constrain = not args.no_coord_constraints
    output_prefix = args.prefix or os.path.splitext(os.path.basename(pdb_file))[0]

    # Initialize PyRosetta
    logger.info("=" * 80)
    logger.info("PYROSETTA PROTEIN-LIGAND RELAX AND SCORE")
    logger.info("=" * 80)

    init_pyrosetta(params_files, args.score_function, cartesian)

    # Load pose
    input_pose = pose_from_pdb(pdb_file)
    sfxn = create_score_function(args.score_function, cartesian)

    # Apply user constraints if provided
    if args.constraints:
        apply_constraints(input_pose, args.constraints)

    # Detect ligands
    ligands = detect_ligand_chains(input_pose, params_files)
    if ligands:
        logger.info(f"\nDetected {len(ligands)} ligand(s):")
        for lig in ligands:
            logger.info(f"  Chain {lig.chain_id}: {lig.residue_name} ({lig.params_file})")
    else:
        logger.info("\nNo ligands detected. Will perform relaxation and scoring only.")

    # Setup FastRelax
    relax_mover = setup_fast_relax(sfxn, cartesian, constrain)

    logger.info(f"\nRelaxation settings:")
    logger.info(f"  Mode: {args.relax_mode}")
    logger.info(f"  Score function: {args.score_function}{'_cart' if cartesian else ''}")
    logger.info(f"  Structures to generate: {args.nstruct}")
    logger.info(f"  Coordinate constraints: {constrain}")

    # Relax and score
    logger.info("\n" + "=" * 80)
    logger.info("RELAXING AND SCORING")
    logger.info("=" * 80)

    all_results = []

    for struct_num in range(1, args.nstruct + 1):
        logger.info(f"\nStructure {struct_num}/{args.nstruct}")
        logger.info(f"  Relaxing...")

        relaxed_pose = relax_pose(input_pose, relax_mover)

        # Save relaxed PDB
        relaxed_pdb = os.path.join(args.output_dir, f"{output_prefix}_relaxed_{struct_num:04d}.pdb")
        relaxed_pose.dump_pdb(relaxed_pdb)
        logger.info(f"  Saved: {relaxed_pdb}")

        # Score
        total_score = score_pose(relaxed_pose, sfxn)
        logger.info(f"  Total score: {total_score:.2f}")

        # RMSD
        rmsd = calculate_rmsd(input_pose, relaxed_pose)
        logger.info(f"  RMSD to input: {rmsd:.3f} A")

        result = {
            'structure': os.path.basename(relaxed_pdb),
            'total_score': f"{total_score:.2f}",
            'rmsd_to_input': f"{rmsd:.3f}",
        }

        # Interface energies
        if ligands:
            logger.info(f"  Calculating interface energies...")

            for lig in ligands:
                try:
                    interface_delta = calculate_interface_energy(
                        relaxed_pose, sfxn, lig.chain_id
                    )
                    logger.info(f"    Interface dG (chain {lig.chain_id}): {interface_delta:.2f}")
                    result[f'interface_delta_chain_{lig.chain_id}'] = f"{interface_delta:.2f}"
                except Exception as e:
                    logger.error(f"    Error for chain {lig.chain_id}: {e}")
                    result[f'interface_delta_chain_{lig.chain_id}'] = "ERROR"

            if len(ligands) > 1:
                combined = sum(
                    float(result.get(f'interface_delta_chain_{lig.chain_id}', 0))
                    for lig in ligands
                    if result.get(f'interface_delta_chain_{lig.chain_id}', 'ERROR') != 'ERROR'
                )
                logger.info(f"    Interface dG (combined): {combined:.2f}")
                result['interface_delta_combined'] = f"{combined:.2f}"

        all_results.append(result)

    # Write summary
    summary_csv = os.path.join(args.output_dir, f"{output_prefix}_summary.csv")
    write_summary_csv(all_results, summary_csv)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated {args.nstruct} relaxed and scored structure(s)")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Summary: {summary_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
