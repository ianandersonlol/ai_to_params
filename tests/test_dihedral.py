#!/usr/bin/env python3
"""Tests for dihedral angle calculation.

Regression test for the abs(cos_dihedral) bug that folded dihedrals
into [0°, 90°], collapsing S-Fe-Fe-S angles in iron-sulfur clusters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import pytest
from rosetta_params_utils import r3


def _pt(x, y, z):
    return r3.Triple(x, y, z)


class TestDihedral:
    """Test dihedral angle calculation across full [-180°, 180°] range."""

    def test_zero_dihedral(self):
        """All four points in the same plane, cis: dihedral = 0°."""
        a = _pt(0, 1, 0)
        b = _pt(0, 0, 0)
        c = _pt(1, 0, 0)
        d = _pt(1, 1, 0)
        assert abs(r3.dihedral(a, b, c, d)) < 1e-6

    def test_180_dihedral(self):
        """Four points in trans configuration: dihedral = 180°."""
        a = _pt(0, 1, 0)
        b = _pt(0, 0, 0)
        c = _pt(1, 0, 0)
        d = _pt(1, -1, 0)
        result = r3.dihedral(a, b, c, d)
        assert abs(abs(result) - 180.0) < 1e-6, f"Expected ±180°, got {result}°"

    def test_90_dihedral(self):
        """90° dihedral."""
        a = _pt(0, 1, 0)
        b = _pt(0, 0, 0)
        c = _pt(1, 0, 0)
        d = _pt(1, 0, 1)
        result = r3.dihedral(a, b, c, d)
        assert abs(abs(result) - 90.0) < 1e-6, f"Expected ±90°, got {result}°"

    def test_sign_distinguishes_rotation_direction(self):
        """+90° and -90° must have opposite signs."""
        a = _pt(0, 1, 0)
        b = _pt(0, 0, 0)
        c = _pt(1, 0, 0)
        d_plus = _pt(1, 0, 1)
        d_minus = _pt(1, 0, -1)
        plus = r3.dihedral(a, b, c, d_plus)
        minus = r3.dihedral(a, b, c, d_minus)
        assert abs(plus + minus) < 1e-6, f"Expected opposite signs, got {plus}° and {minus}°"

    def test_s_fe_fe_s_trans(self):
        """Regression: S-Fe-Fe-S in [2Fe-2S] cluster rhombic ring.

        The abs() bug collapsed this 180° dihedral to 0°, causing
        sulfurs to stack on top of each other in the output PDB.
        WHYYYYYYYYYYY DOESN'T THIS WANT TO WORK??!?!!?!?!?!?!?!?!?!!?!?!
        """
        s1 = _pt(0.0, 1.1, 0.0)
        fe1 = _pt(0.0, 0.0, 0.0)
        fe2 = _pt(2.7, 0.0, 0.0)
        s2 = _pt(2.7, -1.1, 0.0)
        result = r3.dihedral(s1, fe1, fe2, s2)
        assert abs(abs(result) - 180.0) < 1e-6, f"Expected ±180°, got {result}°"
