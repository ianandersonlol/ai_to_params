#!/usr/bin/env python3
"""Tests for metal atom name handling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ai_to_params import sanitize_atom_name


class TestMetalNaming:
    """Test metal atom name sanitization."""

    @pytest.mark.parametrize("input_name,expected", [
        ("MG_1", "MG"),
        ("MG_2", "MG"),
        ("MG1_1", "MG"),
        ("ZN_1", "ZN"),
        ("ZN12_3", "ZN"),
        ("FE_1", "FE"),
        ("CA_1", "CA"),
    ])
    def test_metal_strips_to_element(self, input_name, expected):
        assert sanitize_atom_name(input_name, is_metal=True) == expected

    @pytest.mark.parametrize("input_name,is_metal,expected", [
        ("C_1", False, "C"),
        ("C_2", False, "C1"),
        ("N_3", False, "N2"),
        ("CA", False, "CA"),
    ])
    def test_non_metal_unchanged(self, input_name, is_metal, expected):
        assert sanitize_atom_name(input_name, is_metal) == expected
