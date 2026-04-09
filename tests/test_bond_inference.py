#!/usr/bin/env python3
"""Tests for bond inference (both RDKit and distance-based fallback)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from rosetta_params_utils import MolfileAtom, MolfileBond, Bond
from ai_to_params import infer_bonds_rdkit, infer_bonds_distance


def _make_atom(name, elem, x, y, z):
    """Helper to create a MolfileAtom with coordinates."""
    a = MolfileAtom()
    a.name = name
    a.elem = elem
    a.x = float(x)
    a.y = float(y)
    a.z = float(z)
    a.partial_charge = None
    a.bonds = []
    a.heavy_bonds = []
    a.is_H = (elem == 'H')
    a.is_ring = False
    a.ring_size = 0
    return a


class TestDistanceBondInference:
    """Test distance-based bond inference (fallback method)."""

    def test_water_molecule(self):
        """H-O-H should have 2 bonds."""
        atoms = [
            _make_atom("O", "O", 0.0, 0.0, 0.0),
            _make_atom("H1", "H", 0.757, 0.586, 0.0),
            _make_atom("H2", "H", -0.757, 0.586, 0.0),
        ]
        bonds = infer_bonds_distance(atoms)
        assert len(bonds) == 2

    def test_no_bond_far_apart(self):
        """Atoms far apart should have no bonds."""
        atoms = [
            _make_atom("C1", "C", 0.0, 0.0, 0.0),
            _make_atom("C2", "C", 10.0, 0.0, 0.0),
        ]
        bonds = infer_bonds_distance(atoms)
        assert len(bonds) == 0

    def test_cc_single_bond(self):
        """Two carbons at ~1.54 A should be bonded."""
        atoms = [
            _make_atom("C1", "C", 0.0, 0.0, 0.0),
            _make_atom("C2", "C", 1.54, 0.0, 0.0),
        ]
        bonds = infer_bonds_distance(atoms)
        assert len(bonds) == 1

    def test_mirror_bonds(self):
        """Each bond should have a mirror bond."""
        atoms = [
            _make_atom("C", "C", 0.0, 0.0, 0.0),
            _make_atom("O", "O", 1.2, 0.0, 0.0),
        ]
        bonds = infer_bonds_distance(atoms)
        assert len(bonds) == 1
        assert bonds[0].mirror is not None
        assert bonds[0].mirror.mirror is bonds[0]
        assert bonds[0].a1 is bonds[0].mirror.a2
        assert bonds[0].a2 is bonds[0].mirror.a1

    def test_atom_bonds_populated(self):
        """After inference, atoms should have their bonds lists populated."""
        atoms = [
            _make_atom("C", "C", 0.0, 0.0, 0.0),
            _make_atom("O", "O", 1.2, 0.0, 0.0),
            _make_atom("H", "H", -0.5, 0.8, 0.0),
        ]
        infer_bonds_distance(atoms)
        # C should be bonded to O and H
        assert len(atoms[0].bonds) == 2
        # O should be bonded to C
        assert len(atoms[1].bonds) == 1
        # H should be bonded to C
        assert len(atoms[2].bonds) == 1

    def test_all_bonds_single(self):
        """Distance-based inference assigns all bonds as SINGLE."""
        atoms = [
            _make_atom("C", "C", 0.0, 0.0, 0.0),
            _make_atom("O", "O", 1.2, 0.0, 0.0),
        ]
        bonds = infer_bonds_distance(atoms)
        assert all(b.order == Bond.SINGLE for b in bonds)


class TestRDKitBondInference:
    """Test RDKit-based bond inference with bond order perception."""

    def test_ethane(self):
        """Ethane (C2H6): 1 C-C bond + 6 C-H bonds = 7 total bonds."""
        # Approximate ethane geometry
        atoms = [
            _make_atom("C1", "C", 0.000, 0.000, 0.000),
            _make_atom("C2", "C", 1.540, 0.000, 0.000),
            _make_atom("H1", "H", -0.36, 1.03, 0.000),
            _make_atom("H2", "H", -0.36, -0.51, 0.89),
            _make_atom("H3", "H", -0.36, -0.51, -0.89),
            _make_atom("H4", "H", 1.90, 1.03, 0.000),
            _make_atom("H5", "H", 1.90, -0.51, 0.89),
            _make_atom("H6", "H", 1.90, -0.51, -0.89),
        ]
        bonds = infer_bonds_rdkit(atoms)
        assert len(bonds) == 7

    def test_formaldehyde_connectivity(self):
        """Formaldehyde (CH2O): should detect C-O bond."""
        atoms = [
            _make_atom("C", "C", 0.000, 0.000, 0.000),
            _make_atom("O", "O", 1.200, 0.000, 0.000),
            _make_atom("H1", "H", -0.55, 0.94, 0.0),
            _make_atom("H2", "H", -0.55, -0.94, 0.0),
        ]
        bonds = infer_bonds_rdkit(atoms)

        # Should have 3 bonds: C-O, C-H, C-H
        assert len(bonds) == 3
        co_bonds = [b for b in bonds if
                    {b.a1.elem, b.a2.elem} == {'C', 'O'}]
        assert len(co_bonds) == 1

    def test_benzene_connectivity(self):
        """Benzene: should detect 12 bonds (6 C-C + 6 C-H)."""
        import math
        atoms = []
        for i in range(6):
            angle = i * math.pi / 3.0
            x = 1.40 * math.cos(angle)
            y = 1.40 * math.sin(angle)
            atoms.append(_make_atom(f"C{i+1}", "C", x, y, 0.0))
        for i in range(6):
            angle = i * math.pi / 3.0
            x = 2.48 * math.cos(angle)
            y = 2.48 * math.sin(angle)
            atoms.append(_make_atom(f"H{i+1}", "H", x, y, 0.0))

        bonds = infer_bonds_rdkit(atoms)

        assert len(bonds) == 12
        cc_bonds = [b for b in bonds if b.a1.elem == 'C' and b.a2.elem == 'C']
        assert len(cc_bonds) == 6

    def test_ring_detection(self):
        """Ring bonds should be marked with is_ring=True."""
        import math
        atoms = []
        for i in range(6):
            angle = i * math.pi / 3.0
            atoms.append(_make_atom(f"C{i+1}", "C", 1.40 * math.cos(angle), 1.40 * math.sin(angle), 0.0))
        for i in range(6):
            angle = i * math.pi / 3.0
            atoms.append(_make_atom(f"H{i+1}", "H", 2.48 * math.cos(angle), 2.48 * math.sin(angle), 0.0))

        bonds = infer_bonds_rdkit(atoms)
        cc_bonds = [b for b in bonds if b.a1.elem == 'C' and b.a2.elem == 'C']
        assert all(b.is_ring for b in cc_bonds), "All C-C bonds in benzene should be in a ring"

    def test_mirror_bonds(self):
        """RDKit inference should also produce proper mirror bonds."""
        atoms = [
            _make_atom("C", "C", 0.0, 0.0, 0.0),
            _make_atom("O", "O", 1.2, 0.0, 0.0),
            _make_atom("H1", "H", -0.55, 0.94, 0.0),
            _make_atom("H2", "H", -0.55, -0.94, 0.0),
        ]
        bonds = infer_bonds_rdkit(atoms)
        for bond in bonds:
            assert bond.mirror is not None
            assert bond.mirror.mirror is bond
            assert bond.a1 is bond.mirror.a2
