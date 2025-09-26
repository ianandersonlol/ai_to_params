#!/usr/bin/env python3

import sys
sys.path.append('.')

from ai_to_params import sanitize_atom_name

def test_atom_numbering():
    """Test atom name numbering fix"""
    print("Testing corrected atom name numbering:")

    # Test non-metal atoms
    non_metal_tests = [
        # Traditional format (C_1, C_2) -> offset numbering
        ("C_1", False, "C"),    # C_1 -> C (start from base)
        ("C_2", False, "C1"),   # C_2 -> C1 (subtract 1)
        ("C_3", False, "C2"),   # C_3 -> C2 (subtract 1)
        ("N_1", False, "N"),    # N_1 -> N
        ("N_2", False, "N1"),   # N_2 -> N1

        # Chai format (C1_1, C2_1) -> keep base numbers
        ("C1_1", False, "C1"),  # C1_1 -> C1 (keep base)
        ("C2_1", False, "C2"),  # C2_1 -> C2 (keep base)
        ("C3_1", False, "C3"),  # C3_1 -> C3 (keep base)
        ("O1_1", False, "O1"),  # O1_1 -> O1 (keep base)

        ("CA", False, "CA"),    # CA -> CA (unchanged)
    ]

    # Test metal atoms (should still work)
    metal_tests = [
        ("MG1_1", True, "MG"),  # Metal: MG1_1 -> MG
        ("ZN_2", True, "ZN"),   # Metal: ZN_2 -> ZN
    ]

    all_tests = non_metal_tests + metal_tests

    for input_name, is_metal, expected in all_tests:
        result = sanitize_atom_name(input_name, is_metal)
        status = "✓" if result == expected else "✗"
        metal_str = "(metal)" if is_metal else "(non-metal)"
        print(f"  {input_name:<8} {metal_str:<12} -> {result:<5} {status}")

if __name__ == "__main__":
    test_atom_numbering()