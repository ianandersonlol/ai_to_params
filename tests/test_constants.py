#!/usr/bin/env python3
"""Tests for shared constants."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from constants import COVALENT_RADII, IGNORE_RESIDUES, METAL_ELEMENTS, LIGAND_CHAIN_IDS


class TestConstants:
    """Sanity checks for shared constants."""

    def test_covalent_radii_positive(self):
        for elem, radius in COVALENT_RADII.items():
            assert radius > 0, f"{elem} has non-positive radius {radius}"

    def test_common_elements_present(self):
        for elem in ['H', 'C', 'N', 'O', 'S', 'P']:
            assert elem in COVALENT_RADII

    def test_metals_present_in_radii(self):
        for metal in METAL_ELEMENTS:
            assert metal in COVALENT_RADII, f"Metal {metal} missing from COVALENT_RADII"

    def test_water_in_ignore(self):
        assert 'HOH' in IGNORE_RESIDUES
        assert 'WAT' in IGNORE_RESIDUES

    def test_ligand_chain_ids_unique(self):
        assert len(LIGAND_CHAIN_IDS) == len(set(LIGAND_CHAIN_IDS))

    def test_ligand_chain_ids_single_char(self):
        for chain_id in LIGAND_CHAIN_IDS:
            assert len(chain_id) == 1
