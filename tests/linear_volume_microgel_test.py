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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'meanfieldpb'))
from PBequations import PBequation_volumeMicrogel_strong as PBstrong
from PBequations import PBequation_volumeMicrogel_weak as PBweak
from LinearPB.linear_volume_microgel_weak import LinearVolumeMicrogelWeak, variables
import volume_microgel

def test_PBeq(r, a, phi, phi_prime, f_diff_eq):
    """

    """

    def residual(quantity):
        quantity = np.abs(quantity)
        mask_less = (r > 1e-2 * a) & (r < 0.99 * a)
        mask_large = r > 1.01 * a
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
    Test cases for the linear solution of the Poisson-Boltzmann equation for volume-charged
    microgels with weak charges.
    This class tests the self-consistency of the linearized Poisson-Boltzmann equation
    and its agreement with the nonlinear solution for weakly charged microgels.
    It uses the `unittest` framework for structured testing.
    The tests include:
    - `test_solutionPBeq_selfConsistency_weak`: Checks the self-consistency of the linearized Poisson-Boltzmann equation.
    - `test_agreement_with_nonlinear_solution`: Verifies that the linear solution agrees with the nonlinear solution for weakly charged microgels.
    """
    # Parameters
    N_nodes = 10000
    a = 50 # [nm]
    Z_b = 0
    lb = 0.71 # [nm]
    vol_frac = 0.01
    c_salt = 1e-3 # [M]
    charge_type = 'weak'
    pKa = 4
    pKb = 4
    pH = 7

    def test_solutionPBeq_selfConsistency_weak(self):

        # regular charge
        Z_a = 100

        # Set up the microgel suspension
        my_suspension = volume_microgel.VolumeMicrogel(self.a, Z_a, self.Z_b, self.lb, 
                                                       self.vol_frac, self.c_salt, self.charge_type,
                                                       self.pKa, self.pKb, self.pH)
        r = np.linspace(1e-8, my_suspension.R_cell, self.N_nodes)
        lin_elec_pot = my_suspension.lin_elec_pot(r)
        lin_elec_field = my_suspension.lin_elec_field(r)

        def f_diff_eq(r, y, y_prime):
            """
            Function that defines the second derivative in the linearized
            differential equation y'' = f(r, y, y'; kresa, k0_aa, xi_a, k0_ab, xi_b)

            All quantities should be in reduced units.
            """

            R, c_Hres, k_res, k_acid, h_acid, k_basic, h_basic = variables(self.a, self.lb, self.vol_frac, self.c_salt, Z_a, self.Z_b, self.pKa, self.pKb, self.pH)
            zeta2 = k_res**2 + h_acid * k_acid**2 + h_basic * k_basic**2 # [1/nm^2]

            return np.where(r<self.a, -2 * y_prime / r + zeta2 * (y + (k_acid**2 - k_basic)/zeta2),
                                -2 * y_prime / r + k_res * k_res * y)

        # Self-consistency of first and second derivative
        residual_0, residual_1 = test_PBeq(r, self.a, lin_elec_pot, lin_elec_field, f_diff_eq)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)


    def test_agreement_with_nonlinear_solution(self):

        # low-charge limit charge
        Z_a = 10

        # Set up the microgel suspension
        my_suspension = volume_microgel.VolumeMicrogel(self.a, Z_a, self.Z_b, self.lb, 
                                                       self.vol_frac, self.c_salt, self.charge_type,
                                                       self.pKa, self.pKb, self.pH)
        r = np.linspace(1e-8, my_suspension.R_cell, self.N_nodes)
        
        # Linear solution
        lin_elec_pot = my_suspension.lin_elec_pot(r)

        # Nonlinear solution
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)

        self.assertAlmostEqual(np.max(np.abs(lin_elec_pot - my_suspension.elec_pot)), 0, places=4, msg=None, delta=None)