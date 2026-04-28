#!/usr/bin/env python
"""
Unified CLI for ai_to_params: convert, relax, score, PLIP analysis, and full pipeline.

Usage:
    ai-to-params convert -i input.cif -prefix output_prefix [options]
    ai-to-params convert -i input.pdb -prefix output_prefix [options]
    ai-to-params relax --prefix output_prefix [options]
    ai-to-params plip --prefix output_prefix [options]
    ai-to-params run -i input.cif -prefix output_prefix [options]
"""

import sys
import os
import argparse
import logging
import glob

from constants import DEFAULT_BOND_TOLERANCE

logger = logging.getLogger(__name__)


def _add_plip_options(parser):
    """Add optional PLIP analysis flags to commands that emit score summaries."""
    parser.add_argument("--plip", action="store_true",
                        help="Run PLIP interaction analysis and append PLIP columns to the summary CSV")
    parser.add_argument("--plip-output", default=None, metavar="FILE",
                        help="Optional PLIP CSV path (default: {prefix}_plip.csv in --output-dir)")
    parser.add_argument("--plip-mutation-resi", type=int, default=None, metavar="N",
                        help="Optional protein residue number for mutation-site PLIP contact counts")


def build_parser():
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ai-to-params",
        description="Convert AI-predicted protein-ligand complexes to Rosetta params, relax, score, and analyze.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- convert ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert mmCIF to Rosetta .params files",
        description="Parse mmCIF files from AI tools and generate Rosetta .params files.",
    )
    convert_parser.add_argument("-cif", "--cif", "--input", "-i",
                                required=True, dest="input_file",
                                help="Input structure file (.cif, .mmcif, .pdb, or .ent)",
                                metavar="FILE")
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
    relax_parser.add_argument("--constraints",
                              help="Constraint file (plain Rosetta csts or enzdes CST::BEGIN blocks)",
                              metavar="FILE")
    relax_parser.add_argument("--cst-weight", type=float, default=1.0,
                              help="Weight on atom_pair/angle/dihedral/coordinate constraint "
                                   "terms when --constraints is used (default: 1.0)")
    relax_parser.add_argument("--no-coord-constraints", action="store_true")
    relax_parser.add_argument("--free-sidechains", action="store_true",
                              help="Skip coord_constrain_sidechains so sidechains repack freely (BindCraft-style)")
    relax_parser.add_argument("--max-iter", type=int, default=None,
                              help="FastRelax max iterations per cycle (default: 2500). BindCraft uses 200.")
    relax_parser.add_argument("--output-dir", default=".", metavar="DIR")
    _add_plip_options(relax_parser)

    # --- score ---
    score_parser = subparsers.add_parser(
        "score",
        help="Score a protein-ligand complex without relaxing",
        description="Score a pose directly and compute interface energies (no FastRelax).",
    )
    score_input = score_parser.add_mutually_exclusive_group(required=True)
    score_input.add_argument("--prefix", help="Prefix for ai_to_params output", metavar="PREFIX")
    score_input.add_argument("--pdb", help="Path to input PDB file", metavar="FILE")
    score_parser.add_argument("--params", help="Comma-separated .params files (with --pdb)", metavar="FILES")
    score_parser.add_argument("--score-function", default="ref2015",
                              help="Rosetta score function (default: ref2015)")
    score_parser.add_argument("--output-dir", default=".", metavar="DIR",
                              help="Directory to write summary CSV (default: current directory)")
    _add_plip_options(score_parser)

    # --- plip ---
    plip_parser = subparsers.add_parser(
        "plip",
        help="Run PLIP interaction analysis on a complex",
        description="Run PLIP interaction analysis on ai_to_params output or an explicit structure file.",
    )
    plip_input = plip_parser.add_mutually_exclusive_group(required=True)
    plip_input.add_argument("--prefix", help="Prefix for ai_to_params output", metavar="PREFIX")
    plip_input.add_argument("--pdb", "--structure", dest="structure",
                            help="Input PDB/mmCIF/CIF structure file", metavar="FILE")
    plip_parser.add_argument("--include-relaxed", action="store_true",
                             help="With --prefix, include {prefix}_relaxed_*.pdb in addition to {prefix}.pdb")
    plip_parser.add_argument("--relaxed-only", action="store_true",
                             help="With --prefix, analyze only {prefix}_relaxed_*.pdb files")
    plip_parser.add_argument("--mutation-resi", type=int, default=None, metavar="N",
                             help="Optional protein residue number for mutation-site contact counts")
    plip_parser.add_argument("--output", default=None, metavar="FILE",
                             help="Output PLIP CSV path (default: {prefix}_plip.csv)")
    plip_parser.add_argument("--output-dir", default=".", metavar="DIR",
                             help="Directory to write the default output CSV")

    # --- run (full pipeline) ---
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: convert CIF -> params -> relax -> score",
        description="Run the complete pipeline from mmCIF input to relaxed, scored structures.",
    )
    run_parser.add_argument("-cif", "--cif", "--input", "-i",
                            required=True, dest="input_file",
                            help="Input structure file (.cif, .mmcif, .pdb, or .ent)",
                            metavar="FILE")
    run_parser.add_argument("-prefix", "--prefix", required=True, help="Output prefix", metavar="PREFIX")
    run_parser.add_argument("--clobber", action="store_true", help="Overwrite existing files")
    run_parser.add_argument("--clean-names", action="store_true")
    run_parser.add_argument("--bond-tolerance", type=float, default=DEFAULT_BOND_TOLERANCE)
    run_parser.add_argument("--nstruct", type=int, default=1, help="Number of relaxed structures (default: 1)")
    run_parser.add_argument("--relax-mode", choices=["cartesian", "torsional"], default="cartesian")
    run_parser.add_argument("--score-function", default="ref2015")
    run_parser.add_argument("--constraints",
                            help="Constraint file (plain Rosetta csts or enzdes CST::BEGIN blocks)",
                            metavar="FILE")
    run_parser.add_argument("--cst-weight", type=float, default=1.0,
                            help="Weight on atom_pair/angle/dihedral/coordinate constraint "
                                 "terms when --constraints is used (default: 1.0)")
    run_parser.add_argument("--no-coord-constraints", action="store_true")
    run_parser.add_argument("--free-sidechains", action="store_true",
                            help="Skip coord_constrain_sidechains so sidechains repack freely (BindCraft-style)")
    run_parser.add_argument("--max-iter", type=int, default=None,
                            help="FastRelax max iterations per cycle (default: 2500). BindCraft uses 200.")
    run_parser.add_argument("--output-dir", default=".", metavar="DIR")
    _add_plip_options(run_parser)

    return parser


def cmd_convert(args):
    """Run the convert subcommand."""
    from ai_to_params import (
        parse_structure_file, identify_and_extract_ligands, infer_bonds,
        parameterize_ligand, write_ligand_files, write_cleaned_complex_pdb,
        get_ligand_chain_id, _is_metal_ligand,
    )

    if not os.path.exists(args.input_file):
        logger.error(f"Input structure file '{args.input_file}' does not exist")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    effective_prefix = os.path.join(args.output_dir, args.prefix)

    logger.info(f"AI to Params converter")
    logger.info(f"Input: {args.input_file}  Output: {args.output_dir}/{args.prefix}")

    structure = parse_structure_file(args.input_file)
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


def _resolve_output_path(path, output_dir, default_name):
    if path:
        return path if os.path.isabs(path) else os.path.join(output_dir, path)
    return os.path.join(output_dir, default_name)


def _run_plip_analysis(structure_paths, output_file, mutation_resi=None):
    from plip_analysis import summarize_and_write

    logger.info(f"\nRunning PLIP analysis on {len(structure_paths)} structure(s)...")
    rows = summarize_and_write(structure_paths, output_file, mutation_resi)
    logger.info(f"PLIP analysis complete: {output_file}")
    return rows


def _merge_plip_rows(summary_rows, plip_rows):
    from plip_analysis import prefix_plip_row

    for summary_row, plip_row in zip(summary_rows, plip_rows):
        summary_row.update(prefix_plip_row(plip_row))


def _all_plip_rows_failed(plip_rows):
    return bool(plip_rows) and all(row.get("plip_error") for row in plip_rows)


def _find_plip_prefix_structures(prefix, output_dir, include_relaxed=False, relaxed_only=False):
    candidates = []
    if output_dir and not os.path.isabs(prefix):
        candidates.append(os.path.join(output_dir, prefix))
    candidates.append(prefix)

    for candidate in candidates:
        structures = []
        if not relaxed_only:
            base_pdb = f"{candidate}.pdb"
            if os.path.exists(base_pdb):
                structures.append(base_pdb)

        if include_relaxed or relaxed_only:
            structures.extend(sorted(glob.glob(f"{candidate}_relaxed_*.pdb")))

        if structures:
            return structures, candidate

    raise FileNotFoundError(
        f"Could not find PLIP input structures for prefix '{prefix}' "
        f"(looked in: {', '.join(candidates)})"
    )


def cmd_relax(args):
    """Run the relax subcommand."""
    from relax_score import (
        find_input_files, init_pyrosetta, detect_ligand_chains,
        create_score_function, setup_fast_relax, apply_constraints,
        enable_constraint_weights, relax_pose, score_pose,
        calculate_interface_energy, calculate_rmsd, write_summary_csv,
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
        enable_constraint_weights(sfxn, args.cst_weight)

    ligands = detect_ligand_chains(input_pose, params_files)
    relax_mover = setup_fast_relax(
        sfxn, cartesian, not args.no_coord_constraints,
        ramp_down_constraints=not bool(args.constraints),
        free_sidechains=getattr(args, "free_sidechains", False),
        max_iter=getattr(args, "max_iter", None),
    )

    all_results = []
    relaxed_pdbs = []
    for struct_num in range(1, args.nstruct + 1):
        logger.info(f"\nStructure {struct_num}/{args.nstruct}: relaxing...")
        relaxed = relax_pose(input_pose, relax_mover)

        relaxed_pdb = os.path.join(args.output_dir, f"{output_prefix}_relaxed_{struct_num:04d}.pdb")
        relaxed.dump_pdb(relaxed_pdb)
        relaxed_pdbs.append(relaxed_pdb)

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

    plip_failed = False
    if getattr(args, "plip", False):
        plip_csv = _resolve_output_path(
            getattr(args, "plip_output", None),
            args.output_dir,
            f"{output_prefix}_plip.csv",
        )
        plip_rows = _run_plip_analysis(
            relaxed_pdbs,
            plip_csv,
            getattr(args, "plip_mutation_resi", None),
        )
        _merge_plip_rows(all_results, plip_rows)
        plip_failed = _all_plip_rows_failed(plip_rows)

    summary_csv = os.path.join(args.output_dir, f"{output_prefix}_summary.csv")
    write_summary_csv(all_results, summary_csv)
    logger.info(f"\nDone. Summary: {summary_csv}")
    if plip_failed:
        logger.error("PLIP analysis failed for all relaxed structures.")
        return 1
    return 0


def cmd_score(args):
    """Score a complex without relaxation."""
    from relax_score import (
        find_input_files, init_pyrosetta, detect_ligand_chains,
        create_score_function, score_pose, calculate_interface_energy,
        write_summary_csv,
    )
    from pyrosetta import pose_from_pdb

    if args.pdb and not args.params:
        logger.error("--params is required when using --pdb")
        return 1

    pdb_file, params_files = find_input_files(args)
    output_prefix = args.prefix or os.path.splitext(os.path.basename(pdb_file))[0]

    os.makedirs(args.output_dir, exist_ok=True)

    # Score without the _cart suffix since there's no cartesian minimization
    init_pyrosetta(params_files, args.score_function, cartesian=False)
    pose = pose_from_pdb(pdb_file)
    sfxn = create_score_function(args.score_function, cartesian=False)

    ligands = detect_ligand_chains(pose, params_files)
    if ligands:
        logger.info(f"\nDetected {len(ligands)} ligand(s):")
        for lig in ligands:
            logger.info(f"  Chain {lig.chain_id}: {lig.residue_name} ({lig.params_file})")

    logger.info(f"\nScoring (no relaxation)...")
    total_score = score_pose(pose, sfxn)
    logger.info(f"  Total score: {total_score:.2f}")

    result = {
        'structure': os.path.basename(pdb_file),
        'total_score': f"{total_score:.2f}",
    }

    for lig in ligands:
        try:
            ie = calculate_interface_energy(pose, sfxn, lig.chain_id)
            logger.info(f"  Interface dG (chain {lig.chain_id}): {ie:.2f}")
            result[f'interface_delta_chain_{lig.chain_id}'] = f"{ie:.2f}"
        except Exception as e:
            logger.error(f"  Interface energy error (chain {lig.chain_id}): {e}")
            result[f'interface_delta_chain_{lig.chain_id}'] = "ERROR"

    plip_failed = False
    if getattr(args, "plip", False):
        plip_csv = _resolve_output_path(
            getattr(args, "plip_output", None),
            args.output_dir,
            f"{output_prefix}_plip.csv",
        )
        plip_rows = _run_plip_analysis(
            [pdb_file],
            plip_csv,
            getattr(args, "plip_mutation_resi", None),
        )
        _merge_plip_rows([result], plip_rows)
        plip_failed = _all_plip_rows_failed(plip_rows)

    summary_csv = os.path.join(args.output_dir, f"{output_prefix}_score.csv")
    write_summary_csv([result], summary_csv)
    logger.info(f"\nDone. Summary: {summary_csv}")
    if plip_failed:
        logger.error("PLIP analysis failed for the scored structure.")
        return 1
    return 0


def cmd_plip(args):
    """Run PLIP analysis as a standalone command."""
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.structure:
            structures = [args.structure]
            output_prefix = os.path.splitext(os.path.basename(args.structure))[0]
        else:
            structures, resolved_prefix = _find_plip_prefix_structures(
                args.prefix,
                args.output_dir,
                include_relaxed=args.include_relaxed,
                relaxed_only=args.relaxed_only,
            )
            output_prefix = os.path.basename(resolved_prefix)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    output_csv = _resolve_output_path(
        args.output,
        args.output_dir,
        f"{output_prefix}_plip.csv",
    )
    rows = _run_plip_analysis(structures, output_csv, args.mutation_resi)

    if _all_plip_rows_failed(rows):
        logger.error("PLIP analysis failed for all input structures.")
        return 1

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
        "score": cmd_score,
        "relax": cmd_relax,
        "plip": cmd_plip,
        "run": cmd_run,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
