#!/usr/bin/env python
"""
Test the chain naming and sanitization logic
"""

def sanitize_chain_id(chain_id: str) -> str:
    """Sanitize chain IDs for Rosetta compatibility"""
    if '_' in chain_id:
        parts = chain_id.split('_')
        base = parts[0]
        try:
            num = int(parts[1])
            if num == 1:
                return base  # C_1 -> C (start from base name)
            else:
                return f"{base}{num-1}"  # C_2 -> C1, C_3 -> C2, etc.
        except (ValueError, IndexError):
            return chain_id.replace('_', '')

    return chain_id

def get_ligand_chain_id(ligand_index: int) -> str:
    """Generate chain IDs for ligands following X, Y, Z, Z1, Y1, X1, ... pattern"""
    base_chains = ['X', 'Y', 'Z']

    if ligand_index < 3:
        return base_chains[ligand_index]

    # For more than 3 ligands: Z1, Y1, X1, Z2, Y2, X2, ...
    cycle = (ligand_index - 3) // 3 + 1  # Which cycle (1, 2, 3, ...)
    chain_in_cycle = (ligand_index - 3) % 3  # Which chain in cycle (0, 1, 2)

    # Reverse order: Z1, Y1, X1
    base_chain = base_chains[2 - chain_in_cycle]  # Z, Y, X

    return f"{base_chain}{cycle}"

def test_chain_sanitization():
    """Test chain ID sanitization"""
    print("Testing chain ID sanitization:")
    test_cases = [
        "A_1", "A_2", "A_3", "A_10",
        "C_1", "C_2",
        "Z_1", "Z_2",
        "A", "B", "C",  # Already clean
        "X_abc",  # Invalid number
    ]

    for original in test_cases:
        sanitized = sanitize_chain_id(original)
        print(f"  {original:6} -> {sanitized}")

def test_ligand_chain_naming():
    """Test ligand chain naming sequence"""
    print("\nTesting ligand chain naming (X, Y, Z, Z1, Y1, X1, ...):")
    for i in range(10):
        chain_id = get_ligand_chain_id(i)
        print(f"  Ligand {i}: {chain_id}")

if __name__ == "__main__":
    test_chain_sanitization()
    test_ligand_chain_naming()