#!/usr/bin/env python3
"""Tests for optional PLIP analysis helpers."""

import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plip_analysis import analyze_structures, extract_plip_features, prefix_plip_row, write_plip_csv


def _install_fake_plip(monkeypatch):
    plip_module = types.ModuleType("plip")
    structure_module = types.ModuleType("plip.structure")
    preparation_module = types.ModuleType("plip.structure.preparation")

    class FakeSite:
        def __init__(self):
            self.hbonds_pdon = [
                SimpleNamespace(distance_ad=2.7, resnr=42, restype="ASP", reschain="A")
            ]
            self.hbonds_ldon = [
                SimpleNamespace(distance_ad=3.0, resnr=7, restype="SER", reschain="A")
            ]
            self.hydrophobic_contacts = [
                SimpleNamespace(distance=3.8, resnr=42, restype="ASP", reschain="A")
            ]
            self.saltbridge_lneg = []
            self.saltbridge_pneg = [
                SimpleNamespace(distance=4.1, resnr=99, restype="LYS", reschain="B")
            ]
            self.pistacking = []
            self.pication_paro = []
            self.pication_laro = [
                SimpleNamespace(distance=4.5, resnr=42, restype="ASP", reschain="A")
            ]
            self.halogen_bonds = []
            self.water_bridges = [
                SimpleNamespace(distance_aw=2.9, resnr=42, restype="ASP", reschain="A")
            ]
            self.metal_complexes = []

    class FakePDBComplex:
        def load_pdb(self, path):
            self.path = path

        def analyze(self):
            self.interaction_sets = {"LIG:A:1": FakeSite()}

    preparation_module.PDBComplex = FakePDBComplex
    structure_module.preparation = preparation_module
    plip_module.structure = structure_module

    monkeypatch.setitem(sys.modules, "plip", plip_module)
    monkeypatch.setitem(sys.modules, "plip.structure", structure_module)
    monkeypatch.setitem(sys.modules, "plip.structure.preparation", preparation_module)


def test_extract_plip_features_with_fake_plip(monkeypatch, tmp_path):
    _install_fake_plip(monkeypatch)
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("END\n")

    features = extract_plip_features(str(pdb_path), mutation_resi=42)

    assert features["count_hbond"] == 2
    assert features["count_hbond_protein_donor"] == 1
    assert features["count_hbond_ligand_donor"] == 1
    assert features["count_hphobic"] == 1
    assert features["count_saltbr"] == 1
    assert features["count_pication"] == 1
    assert features["count_waterbridge"] == 1
    assert features["count_total"] == 6
    assert features["n_sites"] == 1
    assert features["hbond_mean_d"] == 2.85
    assert features["hbond_min_d"] == 2.7
    assert features["hbond_max_d"] == 3.0
    assert features["mut_site_total"] == 4

    fingerprint = json.loads(features["residue_fingerprint_json"])
    assert fingerprint["A:42:ASP"]["hbond"] == 1
    assert fingerprint["A:42:ASP"]["hphobic"] == 1
    assert fingerprint["A:42:ASP"]["pication"] == 1
    assert fingerprint["A:42:ASP"]["waterbridge"] == 1


def test_analyze_write_and_prefix_plip_rows(monkeypatch, tmp_path):
    _install_fake_plip(monkeypatch)
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("END\n")

    rows = analyze_structures([str(pdb_path)])
    assert len(rows) == 1
    assert rows[0]["plip_error"] == ""

    prefixed = prefix_plip_row(rows[0])
    assert "structure" not in prefixed
    assert prefixed["plip_count_hbond"] == 2
    assert "plip_runtime_s" in prefixed

    out_csv = tmp_path / "plip.csv"
    write_plip_csv(rows, str(out_csv))
    text = out_csv.read_text()
    assert "count_hbond" in text
    assert "complex.pdb" in text
