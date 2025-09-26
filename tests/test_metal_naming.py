#!/usr/bin/env python3

import sys
sys.path.append('.')

from ai_to_params import sanitize_atom_name

def test_metal_atom_naming():
    """Test metal atom name sanitization"""
    print("Testing metal atom name sanitization:")

    # Test metal atoms
    metal_tests = [
        ("MG_1", True, "MG"),     # Metal: MG_1 -> MG
        ("MG_2", True, "MG"),     # Metal: MG_2 -> MG
        ("MG1_1", True, "MG"),    # Metal: MG1_1 -> MG (Chai format)
        ("ZN_1", True, "ZN"),     # Metal: ZN_1 -> ZN
        ("ZN12_3", True, "ZN"),   # Metal: ZN12_3 -> ZN
    ]

    # Test non-metal atoms
    non_metal_tests = [
        ("C_1", False, "C"),    # Non-metal: C_1 -> C
        ("C_2", False, "C2"),   # Non-metal: C_2 -> C2
        ("N_3", False, "N3"),   # Non-metal: N_3 -> N3
        ("CA", False, "CA"),    # Non-metal: CA -> CA (unchanged)
    ]

    all_tests = metal_tests + non_metal_tests

    for input_name, is_metal, expected in all_tests:
        result = sanitize_atom_name(input_name, is_metal)
        status = "✓" if result == expected else "✗"
        metal_str = "(metal)" if is_metal else "(non-metal)"
        print(f"  {input_name:<5} {metal_str:<12} -> {result:<5} {status}")

if __name__ == "__main__":
    test_metal_atom_naming()