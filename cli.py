#!/usr/bin/env python
"""
Unified CLI for ai_to_params: convert, relax, and full pipeline.

Usage:
    ai-to-params convert -cif input.cif -prefix output_prefix [options]
    ai-to-params relax --prefix output_prefix [options]
    ai-to-params run -cif input.cif -prefix output_prefix [options]
"""

import sys
import os
import argparse
import logging

from constants import DEFAULT_BOND_TOLERANCE

logger = logging.getLogger(__name__)


def build_parser():
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ai-to-params",
        description="Convert AI-predicted protein-ligand complexes to Rosetta params, relax, and score.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- convert ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert mmCIF to Rosetta .params files",
        description="Parse mmCIF files from AI tools and generate Rosetta .params files.",
    )
    convert_parser.add_argument("-cif", "--cif", required=True, help="Input mmCIF file", metavar="FILE")
    convert_parser.add_argument("-prefix", "--prefix", required=True, help="Output prefix", metavar="PREFIX")
    convert_parser.add_argument("--clobber", action="store_true", help="Overwrite existing files")
    convert_parser.add_argument("--clean-names", action="store_true",
                                help="Use cleaned original names instead of L01, L02")
    convert_parser.add_argument("--bond-tolerance", type=float, default=DEFAULT_BOND_TOLERANCE,
                                help=f"Bond inference tolerance in Angstroms (default: {DEFAULT_BOND_TOLERANCE})")
    convert_parser.add_argument("--output-dir", default=".",
                                help="Directory to write output files (default: current directory)",
                                metavar="DIR")

    # --- relax ---
    relax_parser = subparsers.add_parser(
        "relax",
        help="Relax and score a protein-ligand complex with PyRosetta",
        description="Run FastRelax and interface energy calculation on a complex.",
    )
    relax_input = relax_parser.add_mutually_exclusive_group(required=True)
    relax_input.add_argument("--prefix", help="Prefix for ai_to_params output", metavar="PREFIX")
    relax_input.add_argument("--pdb", help="Path to input PDB file", metavar="FILE")
    relax_parser.add_argument("--params", help="Comma-separated .params files (with --pdb)", metavar="FILES")
    relax_parser.add_argument("--nstruct", type=int, default=1, help="Number of structures (default: 1)")
    relax_parser.add_argument("--relax-mode", choices=["cartesian", "torsional"], default="cartesian")
    relax_parser.add_argument("--score-function", default="ref2015")
    relax_parser.add_argument("--constraints", help="Constraint file", metavar="FILE")
    relax_parser.add_argument("--no-coord-constraints", action="store_true")
    relax_parser.add_argument("--output-dir", default=".", metavar="DIR")

    # --- run (full pipeline) ---
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: convert CIF -> params -> relax -> score",
        description="Run the complete pipeline from mmCIF input to relaxed, scored structures.",
    )
    run_parser.add_argument("-cif", "--cif", required=True, help="Input mmCIF file", metavar="FILE")
    run_parser.add_argument("-prefix", "--prefix", required=True, help="Output prefix", metavar="PREFIX")
    run_parser.add_argument("--clobber", action="store_true", help="Overwrite existing files")
    run_parser.add_argument("--clean-names", action="store_true")
    run_parser.add_argument("--bond-tolerance", type=float, default=DEFAULT_BOND_TOLERANCE)
    run_parser.add_argument("--nstruct", type=int, default=1, help="Number of relaxed structures (default: 1)")
    run_parser.add_argument("--relax-mode", choices=["cartesian", "torsional"], default="cartesian")
    run_parser.add_argument("--score-function", default="ref2015")
    run_parser.add_argument("--constraints", help="Constraint file", metavar="FILE")
    run_parser.add_argument("--no-coord-constraints", action="store_true")
    run_parser.add_argument("--output-dir", default=".", metavar="DIR")

    return parser


def cmd_convert(args):
    """Run the convert subcommand."""
    from ai_to_params import (
        parse_cif_file, identify_and_extract_ligands, infer_bonds,
        parameterize_ligand, write_ligand_files, write_cleaned_complex_pdb,
        get_ligand_chain_id, _is_metal_ligand,
    )

    if not os.path.exists(args.cif):
        logger.error(f"Input CIF file '{args.cif}' does not exist")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    effective_prefix = os.path.join(args.output_dir, args.prefix)

    logger.info(f"AI to Params converter")
    logger.info(f"Input: {args.cif}  Output: {args.output_dir}/{args.prefix}")

    structure = parse_cif_file(args.cif)
    logger.info(f"Parsed structure with {len(list(structure.get_models()))} model(s)")

    ligands = identify_and_extract_ligands(structure, args.clean_names)
    logger.info(f"Found {len(ligands)} unique ligand type(s)")

    for ligand in ligands:
        if _is_metal_ligand(ligand.residue):
            logger.info(f"  Skipping bond inference for pure metal: {ligand.sanitized_name}")
            continue
        logger.info(f"  Inferring bonds for {ligand.original_name}...")
        ligand.bonds = infer_bonds(ligand.atoms, args.bond_tolerance)

    for ligand in ligands:
        if _is_metal_ligand(ligand.residue):
            logger.info(f"  Skipping metal: {ligand.sanitized_name}")
            continue
        logger.info(f"  Parameterizing {ligand.sanitized_name}...")
        parameterize_ligand(ligand)

    for ligand_index, ligand in enumerate(ligands):
        if _is_metal_ligand(ligand.residue):
            continue
        ligand_chain_id = get_ligand_chain_id(ligand_index)
        write_ligand_files(ligand, effective_prefix, args.clobber, ligand_chain_id)

    name_mapping = {lig.original_name: lig.sanitized_name for lig in ligands}
    chain_mapping = {lig.original_name: get_ligand_chain_id(i) for i, lig in enumerate(ligands)}
    write_cleaned_complex_pdb(structure, name_mapping, chain_mapping, effective_prefix, args.clobber)

    logger.info("Conversion complete!")
    return 0


def cmd_relax(args):
    """Run the relax subcommand."""
    from relax_score import (
        find_input_files, init_pyrosetta, detect_ligand_chains,
        create_score_function, setup_fast_relax, apply_constraints,
        relax_pose, score_pose, calculate_interface_energy,
        calculate_rmsd, write_summary_csv,
    )
    from pyrosetta import pose_from_pdb

    if args.pdb and not args.params:
        logger.error("--params is required when using --pdb")
        return 1

    pdb_file, params_files = find_input_files(args)
    cartesian = (args.relax_mode == "cartesian")
    output_prefix = args.prefix or os.path.splitext(os.path.basename(pdb_file))[0]

    os.makedirs(args.output_dir, exist_ok=True)

    init_pyrosetta(params_files, args.score_function, cartesian)
    input_pose = pose_from_pdb(pdb_file)
    sfxn = create_score_function(args.score_function, cartesian)

    if args.constraints:
        apply_constraints(input_pose, args.constraints)

    ligands = detect_ligand_chains(input_pose, params_files)
    relax_mover = setup_fast_relax(sfxn, cartesian, not args.no_coord_constraints)

    all_results = []
    for struct_num in range(1, args.nstruct + 1):
        logger.info(f"\nStructure {struct_num}/{args.nstruct}: relaxing...")
        relaxed = relax_pose(input_pose, relax_mover)

        relaxed_pdb = os.path.join(args.output_dir, f"{output_prefix}_relaxed_{struct_num:04d}.pdb")
        relaxed.dump_pdb(relaxed_pdb)

        total_score = score_pose(relaxed, sfxn)
        rmsd = calculate_rmsd(input_pose, relaxed)
        logger.info(f"  Score: {total_score:.2f}  RMSD: {rmsd:.3f} A")

        result = {
            'structure': os.path.basename(relaxed_pdb),
            'total_score': f"{total_score:.2f}",
            'rmsd_to_input': f"{rmsd:.3f}",
        }

        for lig in ligands:
            try:
                ie = calculate_interface_energy(relaxed, sfxn, lig.chain_id)
                logger.info(f"  Interface dG (chain {lig.chain_id}): {ie:.2f}")
                result[f'interface_delta_chain_{lig.chain_id}'] = f"{ie:.2f}"
            except Exception as e:
                logger.error(f"  Interface energy error (chain {lig.chain_id}): {e}")
                result[f'interface_delta_chain_{lig.chain_id}'] = "ERROR"

        all_results.append(result)

    summary_csv = os.path.join(args.output_dir, f"{output_prefix}_summary.csv")
    write_summary_csv(all_results, summary_csv)
    logger.info(f"\nDone. Summary: {summary_csv}")
    return 0


def cmd_run(args):
    """Run the full pipeline: convert -> relax -> score."""
    # Step 1: Convert
    logger.info("=" * 80)
    logger.info("PHASE 1: CONVERTING CIF TO PARAMS")
    logger.info("=" * 80)

    convert_result = cmd_convert(args)
    if convert_result != 0:
        return convert_result

    # Step 2: Relax and score
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: RELAXING AND SCORING")
    logger.info("=" * 80)

    # Build relax args from run args
    args.pdb = None  # Use prefix mode
    args.params = None
    relax_result = cmd_relax(args)
    return relax_result


def main():
    """Entry point for the unified CLI."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "convert": cmd_convert,
        "relax": cmd_relax,
        "run": cmd_run,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
