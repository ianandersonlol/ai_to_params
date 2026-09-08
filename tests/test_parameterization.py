#!/usr/bin/env python3
"""Tests for the parameterization pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io
import pytest
from rosetta_params_utils import (
    MolfileAtom, MolfileBond, Bond,
    add_fields_to_atoms, add_fields_to_bonds,
    find_virtual_atoms, uniquify_atom_names,
    assign_rosetta_types, assign_mm_types, assign_centroid_types,
    assign_partial_charges, assign_rotatable_bonds, assign_rigid_ids,
    build_fragment_trees, assign_internal_coords, write_param_file,
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


def _build_propanol():
    """Build 1-propanol (CH3-CH2-CH2-OH): one heavy-atom chi, one proton chi."""
    atoms = [
        _make_atom("C1", "C", 0.000, 0.000, 0.000),
        _make_atom("C2", "C", 1.530, 0.000, 0.000),
        _make_atom("C3", "C", 2.060, 1.440, 0.000),
        _make_atom("O1", "O", 3.480, 1.440, 0.000),
        _make_atom("H1", "H", -0.36, 1.03, 0.00),
        _make_atom("H2", "H", -0.36, -0.51, 0.89),
        _make_atom("H3", "H", -0.36, -0.51, -0.89),
        _make_atom("H4", "H", 1.89, -0.51, 0.89),
        _make_atom("H5", "H", 1.89, -0.51, -0.89),
        _make_atom("H6", "H", 1.70, 1.95, 0.89),
        _make_atom("H7", "H", 1.70, 1.95, -0.89),
        _make_atom("HO", "H", 3.80, 2.34, 0.00),
    ]
    bonds = infer_bonds_rdkit(atoms)
    return atoms, bonds


def _write_params(atoms, bonds, base_confs=1, max_confs=5000):
    """Run the full parameterization pipeline and return the params text."""
    class MockMolfile:
        def __init__(self, atoms, bonds):
            self.atoms = atoms
            self.bonds = bonds
            self.footer = []

    molfile = MockMolfile(atoms, bonds)
    add_fields_to_atoms(atoms)
    add_fields_to_bonds(bonds)
    find_virtual_atoms(atoms)
    uniquify_atom_names(atoms)
    assign_rosetta_types(atoms)
    assign_mm_types(atoms)
    assign_centroid_types(atoms)
    assign_partial_charges(atoms, net_charge=0.0)
    assign_rotatable_bonds(bonds)
    assign_rigid_ids(atoms)
    for atom in atoms:
        atom.fragment_id = 1
    build_fragment_trees(molfile)
    assign_internal_coords(molfile)

    buf = io.StringIO()
    write_param_file(buf, molfile, "LIG", 1, base_confs, max_confs, None)
    return buf.getvalue()


class TestChiLines:
    """Test that rotatable bonds are written as CHI / PROTON_CHI records."""

    def test_acetone_has_no_chi(self):
        """Acetone: two methyls and a C=O, nothing rotatable."""
        atoms = [
            _make_atom("C2", "C", 0.000, 0.000, 0.000),
            _make_atom("O1", "O", 0.000, 1.220, 0.000),
            _make_atom("C1", "C", 1.300, -0.760, 0.000),
            _make_atom("C3", "C", -1.300, -0.760, 0.000),
            _make_atom("H1", "H", 2.20, -0.15, 0.00),
            _make_atom("H2", "H", 1.35, -1.40, 0.89),
            _make_atom("H3", "H", 1.35, -1.40, -0.89),
            _make_atom("H4", "H", -2.20, -0.15, 0.00),
            _make_atom("H5", "H", -1.35, -1.40, 0.89),
            _make_atom("H6", "H", -1.35, -1.40, -0.89),
        ]
        bonds = infer_bonds_rdkit(atoms)
        text = _write_params(atoms, bonds)
        assert not [l for l in text.splitlines() if l.startswith("CHI ")]
        assert "PROTON_CHI" not in text

    def test_propanol_chi_lines(self):
        atoms, bonds = _build_propanol()
        text = _write_params(atoms, bonds)
        lines = text.splitlines()
        chi_lines = [l for l in lines if l.startswith("CHI ")]
        proton_lines = [l for l in lines if l.startswith("PROTON_CHI ")]

        # C2-C3 (heavy chi) and C3-O1 (proton chi); C1-C2 is a methyl, not rotatable
        assert len(chi_lines) == 2
        assert len(proton_lines) == 1

        # Chis are numbered sequentially starting at 1
        assert [l.split()[1] for l in chi_lines] == ["1", "2"]

        # Proton chis are written first so -ex1/-ex2 sample them
        assert chi_lines[0].split()[1:] == ["1", "C2", "C3", "O1", "HO"]
        # sp3 hydroxyl: three-fold sampling with extra samples when affordable
        assert proton_lines[0].split() == \
            ["PROTON_CHI", "1", "SAMPLES", "3", "60", "-60", "180", "EXTRA", "1", "20"]

        # Heavy-atom chi is defined root-to-tip along the C2-C3 axis
        assert chi_lines[1].split()[2:] == ["O1", "C3", "C2", "C1"]

        # CHI records must precede NBR_ATOM in the params file
        assert lines.index(chi_lines[0]) < lines.index(
            next(l for l in lines if l.startswith("NBR_ATOM")))

    def test_proton_chi_extra_sampling_disabled_when_too_many_confs(self):
        atoms, bonds = _build_propanol()
        # One sp3 proton chi multiplies conformer count by 9; cap below that
        text = _write_params(atoms, bonds, base_confs=1, max_confs=5)
        proton_lines = [l for l in text.splitlines() if l.startswith("PROTON_CHI ")]
        assert len(proton_lines) == 1
        assert proton_lines[0].endswith("EXTRA 0")


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
