#!/usr/bin/env python
"""
PLIP interaction analysis for ai_to_params outputs.

PLIP is an optional runtime dependency. This module intentionally imports PLIP
inside analysis functions so conversion, scoring, and tests do not require a
PLIP environment unless the analysis is requested.
"""

import csv
import json
import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


BASE_COUNT_KEYS = (
    "count_hbond",
    "count_hphobic",
    "count_saltbr",
    "count_pistack",
    "count_pication",
    "count_halogen",
    "count_waterbridge",
    "count_metal",
)

COUNT_COLUMNS = (
    "count_hbond",
    "count_hbond_protein_donor",
    "count_hbond_ligand_donor",
    "count_hphobic",
    "count_saltbr",
    "count_pistack",
    "count_pication",
    "count_halogen",
    "count_waterbridge",
    "count_metal",
    "count_total",
    "n_sites",
)

DISTANCE_TYPES = (
    "hbond",
    "hphobic",
    "saltbr",
    "pistack",
    "pication",
    "halogen",
)

DISTANCE_COLUMNS = tuple(
    f"{name}_{stat}_d"
    for name in DISTANCE_TYPES
    for stat in ("mean", "min", "max")
)

MUTATION_SITE_COLUMNS = (
    "mut_site_total",
    "mut_site_hbond",
    "mut_site_hphobic",
    "mut_site_saltbr",
    "mut_site_pistack",
    "mut_site_pication",
    "mut_site_halogen",
    "mut_site_waterbridge",
    "mut_site_metal",
)

FEATURE_COLUMNS = (
    *COUNT_COLUMNS,
    *DISTANCE_COLUMNS,
    *MUTATION_SITE_COLUMNS,
    "residue_fingerprint_json",
)

OUTPUT_COLUMNS = (
    "structure",
    "structure_path",
    *FEATURE_COLUMNS,
    "runtime_s",
    "plip_error",
)


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    """Convert an mmCIF/CIF structure to PDB for PLIP."""
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("plip_input", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def stats(distances: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return mean/min/max distance stats, or Nones when no distances exist."""
    if not distances:
        return None, None, None
    return sum(distances) / len(distances), min(distances), max(distances)


def _load_pdb_complex_class():
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError as exc:
        raise ImportError(
            "PLIP is required for PLIP analysis. Activate an environment that "
            "already has PLIP installed, then rerun the command."
        ) from exc
    return PDBComplex


def _as_list(value) -> List:
    if value is None:
        return []
    return list(value)


def _first_distance(interaction, attrs: Sequence[str]) -> Optional[float]:
    for attr in attrs:
        value = getattr(interaction, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _residue_key(interaction) -> Optional[Tuple[str, int, str]]:
    resnr = getattr(interaction, "resnr", None)
    restype = getattr(interaction, "restype", None)
    if resnr is None or restype is None:
        return None

    try:
        resnr = int(resnr)
    except (TypeError, ValueError):
        return None

    chain = getattr(interaction, "reschain", "") or ""
    return chain, resnr, str(restype)


def _record_residue_contact(per_residue, interaction_name: str, interaction) -> None:
    key = _residue_key(interaction)
    if key is None:
        return
    per_residue[key][interaction_name] += 1


def _add_interactions(
    features,
    per_type_dist,
    per_residue,
    interactions,
    count_key: str,
    interaction_name: str,
    distance_attrs: Sequence[str],
) -> None:
    for interaction in _as_list(interactions):
        features[count_key] += 1
        distance = _first_distance(interaction, distance_attrs)
        if distance is not None:
            per_type_dist[interaction_name].append(distance)
        _record_residue_contact(per_residue, interaction_name, interaction)


def _fingerprint_to_json(per_residue) -> str:
    fingerprint = {}
    for (chain, resnr, restype), counts in sorted(per_residue.items()):
        key = f"{chain}:{resnr}:{restype}" if chain else f"{resnr}:{restype}"
        fingerprint[key] = dict(sorted(counts.items()))
    return json.dumps(fingerprint, separators=(",", ":"))


def _mutation_site_features(per_residue, mutation_resi: Optional[int]) -> Dict[str, object]:
    if mutation_resi is None:
        return {key: "" for key in MUTATION_SITE_COLUMNS}

    mutation_counts = defaultdict(int)
    for (_chain, resnr, _restype), counts in per_residue.items():
        if resnr != mutation_resi:
            continue
        for interaction_name, count in counts.items():
            mutation_counts[interaction_name] += count

    result = {
        "mut_site_total": sum(mutation_counts.values()),
        "mut_site_hbond": mutation_counts.get("hbond", 0),
        "mut_site_hphobic": mutation_counts.get("hphobic", 0),
        "mut_site_saltbr": mutation_counts.get("saltbr", 0),
        "mut_site_pistack": mutation_counts.get("pistack", 0),
        "mut_site_pication": mutation_counts.get("pication", 0),
        "mut_site_halogen": mutation_counts.get("halogen", 0),
        "mut_site_waterbridge": mutation_counts.get("waterbridge", 0),
        "mut_site_metal": mutation_counts.get("metal", 0),
    }
    return result


def _extract_features_from_pdb(pdb_path: Path, mutation_resi: Optional[int] = None) -> Dict[str, object]:
    PDBComplex = _load_pdb_complex_class()
    mol = PDBComplex()
    mol.load_pdb(str(pdb_path))
    mol.analyze()

    features = defaultdict(int)
    per_type_dist = defaultdict(list)
    per_residue = defaultdict(lambda: defaultdict(int))

    for site in mol.interaction_sets.values():
        hbond_pdon = _as_list(getattr(site, "hbonds_pdon", []))
        hbond_ldon = _as_list(getattr(site, "hbonds_ldon", []))

        features["count_hbond_protein_donor"] += len(hbond_pdon)
        features["count_hbond_ligand_donor"] += len(hbond_ldon)
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            hbond_pdon,
            "count_hbond",
            "hbond",
            ("distance_ad", "distance"),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            hbond_ldon,
            "count_hbond",
            "hbond",
            ("distance_ad", "distance"),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            getattr(site, "hydrophobic_contacts", []),
            "count_hphobic",
            "hphobic",
            ("distance",),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            _as_list(getattr(site, "saltbridge_lneg", []))
            + _as_list(getattr(site, "saltbridge_pneg", [])),
            "count_saltbr",
            "saltbr",
            ("distance",),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            getattr(site, "pistacking", []),
            "count_pistack",
            "pistack",
            ("distance",),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            _as_list(getattr(site, "pication_paro", []))
            + _as_list(getattr(site, "pication_laro", [])),
            "count_pication",
            "pication",
            ("distance",),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            getattr(site, "halogen_bonds", []),
            "count_halogen",
            "halogen",
            ("distance",),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            getattr(site, "water_bridges", []),
            "count_waterbridge",
            "waterbridge",
            ("distance_aw", "distance_dw", "distance"),
        )
        _add_interactions(
            features,
            per_type_dist,
            per_residue,
            getattr(site, "metal_complexes", []),
            "count_metal",
            "metal",
            ("metal_dist", "distance"),
        )

    for interaction_name in DISTANCE_TYPES:
        mean_d, min_d, max_d = stats(per_type_dist[interaction_name])
        features[f"{interaction_name}_mean_d"] = "" if mean_d is None else round(mean_d, 3)
        features[f"{interaction_name}_min_d"] = "" if min_d is None else round(min_d, 3)
        features[f"{interaction_name}_max_d"] = "" if max_d is None else round(max_d, 3)

    features["count_total"] = sum(features[key] for key in BASE_COUNT_KEYS)
    features["n_sites"] = len(mol.interaction_sets)
    features.update(_mutation_site_features(per_residue, mutation_resi))
    features["residue_fingerprint_json"] = _fingerprint_to_json(per_residue)

    for column in FEATURE_COLUMNS:
        features.setdefault(column, 0 if column.startswith("count_") or column == "n_sites" else "")

    return dict(features)


def extract_plip_features(structure_path: str, mutation_resi: Optional[int] = None) -> Dict[str, object]:
    """
    Extract PLIP features for one PDB/mmCIF/CIF structure.

    PLIP itself reads PDB files, so mmCIF/CIF inputs are converted through
    BioPython into a temporary PDB before analysis.
    """
    path = Path(structure_path)
    suffix = path.suffix.lower()

    if suffix in (".cif", ".mmcif"):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "plip_input.pdb"
            cif_to_pdb(path, pdb_path)
            return _extract_features_from_pdb(pdb_path, mutation_resi)

    return _extract_features_from_pdb(path, mutation_resi)


def analyze_structures(
    structure_paths: Iterable[str],
    mutation_resi: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Run PLIP on structures and return one summary row per structure."""
    rows = []
    for structure_path in structure_paths:
        t0 = time.time()
        path = Path(structure_path)
        row = {
            "structure": path.name,
            "structure_path": str(path),
            "plip_error": "",
        }

        try:
            if not path.exists():
                raise FileNotFoundError(path)
            row.update(extract_plip_features(str(path), mutation_resi))
        except Exception as exc:
            row["plip_error"] = repr(exc)[:300]
            logger.error("PLIP analysis failed for %s: %s", path, exc)

        row["runtime_s"] = f"{time.time() - t0:.2f}"
        for column in OUTPUT_COLUMNS:
            row.setdefault(column, "")
        rows.append(row)

    return rows


def write_plip_csv(rows: Sequence[Dict[str, object]], output_file: str) -> None:
    """Write PLIP rows as CSV."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})

    logger.info("Wrote PLIP summary CSV: %s", output_path)


def prefix_plip_row(row: Dict[str, object]) -> Dict[str, object]:
    """Return PLIP fields with names suitable for merging into score summaries."""
    prefixed = {}
    for key, value in row.items():
        if key in ("structure", "structure_path"):
            continue
        if key == "runtime_s":
            prefixed["plip_runtime_s"] = value
        elif key == "plip_error":
            prefixed["plip_error"] = value
        else:
            prefixed[f"plip_{key}"] = value
    return prefixed


def summarize_and_write(
    structure_paths: Iterable[str],
    output_file: str,
    mutation_resi: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Run PLIP on structures, write the CSV, and return the rows."""
    rows = analyze_structures(structure_paths, mutation_resi)
    write_plip_csv(rows, output_file)
    return rows
