#!/usr/bin/env python3

import sys
sys.path.append('.')

from ai_to_params import sanitize_atom_name, get_ligand_chain_id

def test_atom_name_sanitization():
    """Test atom name sanitization"""
    print("Testing atom name sanitization:")
    test_cases = [
        ("C_1", "C1"),
        ("C_2", "C2"),
        ("N_3", "N3"),
        ("CA", "CA"),
        ("CB", "CB"),
        ("O_1", "O1")
    ]

    for input_name, expected in test_cases:
        result = sanitize_atom_name(input_name)
        status = "✓" if result == expected else "✗"
        print(f"  {input_name:<5} -> {result:<5} {status}")

def test_ligand_chain_naming():
    """Test ligand chain naming (single characters only)"""
    print("\nTesting ligand chain naming (single characters only):")
    for i in range(10):
        chain_id = get_ligand_chain_id(i)
        status = "✓" if len(chain_id) == 1 else "✗"
        print(f"  Ligand {i}: {chain_id} {status}")

if __name__ == "__main__":
    test_atom_name_sanitization()
    test_ligand_chain_naming()