#!/usr/bin/env python3
"""Tests for chain ID sanitization and ligand chain assignment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ai_to_params import sanitize_chain_id, get_ligand_chain_id


class TestChainSanitization:
    """Test chain ID sanitization."""

    @pytest.mark.parametrize("input_id,expected", [
        ("A_1", "A"),
        ("A_2", "A1"),
        ("A_3", "A2"),
        ("A_10", "A9"),
        ("C_1", "C"),
        ("C_2", "C1"),
        ("Z_1", "Z"),
        ("Z_2", "Z1"),
    ])
    def test_underscore_format(self, input_id, expected):
        assert sanitize_chain_id(input_id) == expected

    def test_already_clean(self):
        assert sanitize_chain_id("A") == "A"
        assert sanitize_chain_id("B") == "B"
        assert sanitize_chain_id("C") == "C"

    def test_invalid_number(self):
        result = sanitize_chain_id("X_abc")
        assert '_' not in result


class TestLigandChainNaming:
    """Test ligand chain ID assignment."""

    def test_first_three(self):
        assert get_ligand_chain_id(0) == "X"
        assert get_ligand_chain_id(1) == "Y"
        assert get_ligand_chain_id(2) == "Z"

    def test_subsequent(self):
        """After X, Y, Z, continues W, V, U, ..."""
        assert get_ligand_chain_id(3) == "W"
        assert get_ligand_chain_id(4) == "V"
        assert get_ligand_chain_id(5) == "U"

    def test_all_single_character(self):
        """All chain IDs must be single characters for Rosetta compatibility."""
        for i in range(16):
            chain_id = get_ligand_chain_id(i)
            assert len(chain_id) == 1, f"Ligand {i}: chain_id '{chain_id}' is not a single char"

    def test_no_duplicates_in_first_16(self):
        """First 16 ligands should all get unique chain IDs."""
        chain_ids = [get_ligand_chain_id(i) for i in range(16)]
        assert len(chain_ids) == len(set(chain_ids))
