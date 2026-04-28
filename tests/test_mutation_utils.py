from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPTS = ROOT / "Validation" / "scripts"
sys.path.insert(0, str(VALIDATION_SCRIPTS))

from mutation_utils import (  # noqa: E402
    affinity_nM_to_ddg_kcal_mol,
    apply_mutations_to_chain_sequences,
    build_reference_sequence_map,
    concatenate_chain_sequences,
    extract_chain_sequences,
    mutation_sites_to_string,
    parse_mutation_sites,
)
from build_mutation_cofold_manifest import build_pair_sequences, state_id  # noqa: E402


def test_parse_mutation_sites_single_and_multi():
    single = parse_mutation_sites("T315I", default_chain="A")
    assert single == [{
        "chain": "A",
        "wt_aa": "T",
        "position": 315,
        "mut_aa": "I",
        "token": "T315I",
    }]

    multi = parse_mutation_sites("A:D30N;A:N88D")
    assert mutation_sites_to_string(multi) == "A:D30N;A:N88D"

    slash_multi = parse_mutation_sites("D30N/N88D", default_chain="B")
    assert [site["token"] for site in slash_multi] == ["D30N", "N88D"]
    assert all(site["chain"] == "B" for site in slash_multi)


def test_affinity_to_ddg_sign():
    assert affinity_nM_to_ddg_kcal_mol(10.0, 100.0) > 0
    assert affinity_nM_to_ddg_kcal_mol(100.0, 10.0) < 0


def test_extract_and_mutate_chain_sequences(tmp_path):
    pdb_text = """\
ATOM      1  N   THR A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  THR A   1      12.000  12.000   2.000  1.00 20.00           C
ATOM      3  C   THR A   1      13.200  12.200   1.100  1.00 20.00           C
ATOM      4  O   THR A   1      14.100  11.400   1.200  1.00 20.00           O
ATOM      5  N   LYS A   2      13.100  13.400   0.300  1.00 20.00           N
ATOM      6  CA  LYS A   2      14.100  13.700  -0.700  1.00 20.00           C
ATOM      7  C   LYS A   2      13.500  14.800  -1.500  1.00 20.00           C
ATOM      8  O   LYS A   2      12.300  14.900  -1.700  1.00 20.00           O
ATOM      9  N   GLY B   5      16.000  15.000  -1.900  1.00 20.00           N
ATOM     10  CA  GLY B   5      16.200  16.300  -2.500  1.00 20.00           C
ATOM     11  C   GLY B   5      17.400  16.200  -3.400  1.00 20.00           C
ATOM     12  O   GLY B   5      18.300  15.400  -3.100  1.00 20.00           O
TER
END
"""
    pdb_path = tmp_path / "mini.pdb"
    pdb_path.write_text(pdb_text)

    chains = extract_chain_sequences(pdb_path)
    assert chains["A"]["sequence"] == "TK"
    assert chains["B"]["sequence"] == "G"

    mutated = apply_mutations_to_chain_sequences(
        chains,
        parse_mutation_sites("A:T1I;B:G5A"),
    )
    assert mutated["A"] == "IK"
    assert mutated["B"] == "A"
    assert concatenate_chain_sequences(mutated) == "IKA"


def test_state_id_deduplicates_wt_but_keeps_mutants_distinct():
    row_a = {
        "dataset": "abl_tki",
        "protein_name": "ABL1",
        "ligand_name": "imatinib",
        "ligand_id": "",
        "ligand_smiles": "CCO",
        "mutation_sites": "A:T315I",
        "mutation": "T315I",
    }
    row_b = dict(row_a)
    row_b["mutation_sites"] = "A:E255K"
    row_b["mutation"] = "E255K"

    wt_a = state_id(row_a, "wt", "ABCDE")
    wt_b = state_id(row_b, "wt", "ABCDE")
    mut_a = state_id(row_a, "mut", "ABIDE")
    mut_b = state_id(row_b, "mut", "ABKDE")

    assert wt_a == wt_b
    assert mut_a != mut_b


def test_build_reference_sequence_map_supports_numbered_mutations():
    chain_sequences = build_reference_sequence_map("TKA", chain_id="A", residue_start=315)
    mutated = apply_mutations_to_chain_sequences(
        chain_sequences,
        parse_mutation_sites("A:T315I;A:A317G"),
    )
    assert mutated["A"] == "IKG"


def test_build_pair_sequences_uses_reference_sequence_without_structure(tmp_path):
    row = {
        "reference_sequence": "TKA",
        "reference_chain": "A",
        "reference_residue_start": "315",
        "mutation_sites": "A:T315I",
    }
    wt_seq, mut_seq = build_pair_sequences(row, tmp_path)
    assert wt_seq == "TKA"
    assert mut_seq == "IKA"
