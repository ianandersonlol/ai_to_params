#!/usr/bin/env python3
"""Tests for atom name sanitization logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ai_to_params import sanitize_atom_name


class TestAtomNumbering:
    """Test atom name numbering/sanitization."""

    @pytest.mark.parametrize("input_name,is_metal,expected", [
        # Traditional format (C_1, C_2)
        ("C_1", False, "C"),
        ("C_2", False, "C1"),
        ("C_3", False, "C2"),
        ("N_1", False, "N"),
        ("N_2", False, "N1"),
        # Chai format (C1_1, C2_1)
        ("C1_1", False, "C"),
        ("C2_1", False, "C1"),
        ("C3_1", False, "C2"),
        ("O1_1", False, "O"),
        # Unchanged
        ("CA", False, "CA"),
    ])
    def test_non_metal_atoms(self, input_name, is_metal, expected):
        assert sanitize_atom_name(input_name, is_metal) == expected

    @pytest.mark.parametrize("input_name,expected", [
        ("MG1_1", "MG"),
        ("ZN_2", "ZN"),
        ("MG_1", "MG"),
        ("ZN12_3", "ZN"),
        ("FE_1", "FE"),
    ])
    def test_metal_atoms(self, input_name, expected):
        assert sanitize_atom_name(input_name, is_metal=True) == expected

    def test_no_underscore_format(self):
        """AF3 style: O1->O, P1->P, C2->C1"""
        assert sanitize_atom_name("O1") == "O"
        assert sanitize_atom_name("C2") == "C1"
        assert sanitize_atom_name("P1") == "P"

    def test_plain_element(self):
        """Atoms without numbers or underscores pass through unchanged."""
        assert sanitize_atom_name("C") == "C"
        assert sanitize_atom_name("N") == "N"
        assert sanitize_atom_name("CA") == "CA"
        assert sanitize_atom_name("CB") == "CB"
