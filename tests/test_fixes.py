#!/usr/bin/env python3
"""Regression tests for specific bug fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ai_to_params import sanitize_atom_name, get_ligand_chain_id


class TestBugFixes:
    """Regression tests for fixed bugs."""

    def test_atom_sanitization_basic(self):
        assert sanitize_atom_name("C_1") == "C"
        assert sanitize_atom_name("C_2") == "C1"
        assert sanitize_atom_name("N_3") == "N2"
        assert sanitize_atom_name("CA") == "CA"
        assert sanitize_atom_name("CB") == "CB"

    def test_ligand_chain_ids_single_char(self):
        """All chain IDs must be single characters."""
        for i in range(10):
            chain_id = get_ligand_chain_id(i)
            assert len(chain_id) == 1, f"Ligand {i}: '{chain_id}' is not single char"
