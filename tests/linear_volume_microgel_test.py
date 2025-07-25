#
# This file is part of MeanFieldPB.
#
# Copyright (C) 2025 The MeanFieldPB Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import math
import unittest as ut
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PBequations import PBequation_volumeMicrogel_strong as PBstrong
from PBequations import PBequation_volumeMicrogel_weak as PBweak
from LinearPB.linear_volume_microgel_weak import LinearVolumeMicrogelWeak, variables
import volume_microgel
from volume_microgel_test import test_PBeq

def test_PBeq(r, phi, phi_prime, f_diff_eq, param):
    """

    """

    def residual(quantity):
        quantity = np.abs(quantity)
        mask_less = (r > 1e-2) & (r < 0.99)
        mask_large = r > 1.01
        residual_less = quantity[mask_less]
        residual_large = quantity[mask_large]
        res = np.max(np.append(residual_less, residual_large))
        return res
    
    residual_0 = residual(np.gradient(phi, r) - phi_prime) / np.max(np.abs(phi_prime))
    phi_double_prime = np.gradient(phi_prime, r)
    residual_1 = residual(phi_double_prime - f_diff_eq(r, phi, phi_prime)) / np.max(np.abs(phi_double_prime))

    return residual_0, residual_1



class TestLinearVolumeMicrogels(ut.TestCase):
    """
    """

    def test_solutionPBeq_selfConsistency_weak(self):

        # Parameters
        N_nodes = 100000
        a = 50 # [nm]
        Z_a = 100
        Z_b = 0
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]
        charge_type = 'weak'
        pKa = 4
        pKb = 4
        pH = 7

        my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type, pKa, pKb, pH)
        r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
        lin_elec_pot = my_suspension.lin_elec_pot(r)
        lin_elec_field = my_suspension.lin_elec_field(r)

        def f_diff_eq(r, y, y_prime):
            """
            Function that defines the second derivative in the linearized
            differential equation y'' = f(r, y, y'; kresa, k0_aa, xi_a, k0_ab, xi_b)

            All quantities should be in reduced units.
            """
            
            R, c_Hres, k_res, k_acid, h_acid, k_basic, h_basic = variables(a, lb, vol_frac, c_salt, Z_a, Z_b, pKa, pKb, pH)
            zeta2 = k_res**2 + h_acid * k_acid**2 + h_basic * k_basic**2 # [1/nm^2]
            zetaa2 = zeta2 * a**2 # dimensionless
            kresa = k_res * a # dimensionless

            return np.where(r<1, -2 * y_prime / r + zetaa2 * (y + (k_acid**2 - k_basic)/zeta2),
                                -2 * y_prime / r + kresa * kresa * y)
        
        # Self-consistency of first and second derivative
        param = (my_suspension.kresa, my_suspension.k0_aa, my_suspension.xi_a,
                 my_suspension.k0_ab, my_suspension.xi_b)
        residual_0, residual_1 = test_PBeq(r/a, lin_elec_pot, lin_elec_field, 
                                           f_diff_eq)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)