#!/usr/bin/env python3
"""Tests for the parameterization pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from rosetta_params_utils import (
    MolfileAtom, MolfileBond, Bond,
    add_fields_to_atoms, add_fields_to_bonds,
    find_virtual_atoms, uniquify_atom_names,
    assign_rosetta_types, assign_mm_types, assign_centroid_types,
    assign_partial_charges, assign_rotatable_bonds, assign_rigid_ids,
)
from ai_to_params import infer_bonds_rdkit


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


def _build_methanol():
    """Build a simple methanol molecule (CH3OH) for testing."""
    atoms = [
        _make_atom("C", "C", 0.000, 0.000, 0.000),
        _make_atom("O", "O", 1.430, 0.000, 0.000),
        _make_atom("H1", "H", -0.36, 1.03, 0.000),
        _make_atom("H2", "H", -0.36, -0.51, 0.89),
        _make_atom("H3", "H", -0.36, -0.51, -0.89),
        _make_atom("HO", "H", 1.79, 0.94, 0.00),
    ]
    bonds = infer_bonds_rdkit(atoms)
    return atoms, bonds


class TestRosettaTypeAssignment:
    """Test Rosetta atom type assignment."""

    def test_methanol_types(self):
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)

        type_map = {a.elem: a.ros_type.strip() for a in atoms if not a.is_H}
        assert type_map['C'] == 'CH3'
        assert type_map['O'] == 'OH'

    def test_hydrogen_types(self):
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)

        h_types = [a.ros_type.strip() for a in atoms if a.is_H]
        # HO bonded to O should be Hpol, others Hapo
        assert 'Hpol' in h_types
        assert 'Hapo' in h_types


class TestPartialCharges:
    """Test partial charge assignment."""

    def test_charges_sum_to_zero(self):
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)
        assign_partial_charges(atoms, net_charge=0.0)

        total_charge = sum(a.partial_charge for a in atoms)
        assert abs(total_charge) < 1e-3

    def test_all_charges_assigned(self):
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)
        assign_partial_charges(atoms, net_charge=0.0)

        assert all(a.partial_charge is not None for a in atoms)


class TestRotatableBonds:
    """Test rotatable bond detection."""

    def test_methanol_no_rotatable(self):
        """Methanol C-O is not rotatable (C has only 1 heavy neighbor)."""
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)
        assign_rotatable_bonds(bonds)

        rot_bonds = [b for b in bonds if b.can_rotate]
        # Methanol C has only 1 heavy neighbor -> not rotatable
        assert len(rot_bonds) == 0

    def test_ethanol_rotatable(self):
        """Ethanol C-C bond should be rotatable, C-O is proton chi."""
        atoms = [
            _make_atom("C1", "C", 0.000, 0.000, 0.000),
            _make_atom("C2", "C", 1.540, 0.000, 0.000),
            _make_atom("O",  "O", 2.400, 1.000, 0.000),
            _make_atom("H1", "H", -0.36, 1.03, 0.000),
            _make_atom("H2", "H", -0.36, -0.51, 0.89),
            _make_atom("H3", "H", -0.36, -0.51, -0.89),
            _make_atom("H4", "H", 1.90, -0.51, 0.89),
            _make_atom("H5", "H", 1.90, -0.51, -0.89),
            _make_atom("HO", "H", 3.30, 0.70, 0.00),
        ]
        bonds = infer_bonds_rdkit(atoms)
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)
        assign_rotatable_bonds(bonds)

        rot_bonds = [b for b in bonds if b.can_rotate]
        assert len(rot_bonds) >= 1


class TestRigidIds:
    """Test rigid ID assignment."""

    def test_all_atoms_assigned(self):
        atoms, bonds = _build_methanol()
        add_fields_to_atoms(atoms)
        add_fields_to_bonds(bonds)
        find_virtual_atoms(atoms)
        uniquify_atom_names(atoms)
        assign_rosetta_types(atoms)
        assign_rotatable_bonds(bonds)
        assign_rigid_ids(atoms)

        assert all(a.rigid_id > 0 for a in atoms)
