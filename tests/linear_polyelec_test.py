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
import unittest as ut
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'meanfieldpb'))
sys.path.append(os.path.dirname(__file__))
from PBequations import PBequation_linearPolyelectrolyte_strong as PBlinPoly
import linear_polyelec


def test_PBeq(r, phi, phi_prime, param):
    residual_0 = np.max(np.abs(np.gradient(phi, r) - phi_prime))
    phi_double_prime = np.gradient(phi_prime, r)
    residual_1 = phi_double_prime - PBlinPoly.f_diff_eq(r, phi, phi_prime, param)
    residual_1 = np.max(np.abs(residual_1))
    return residual_0, residual_1

class TestLinearPolyelectrolyte(ut.TestCase):
    """
    Unit tests for the `LinearPolyelectrolyte` class.

    This test class contains methods to validate the behavior of the `LinearPolyelectrolyte` 
    class, including solving the nonlinear Poisson-Boltzmann equation and verifying ion 
    profiles and electroneutrality.

    Methods:
    --------
    test_solutionPBeq_selfConsistency():
        Tests the self-consistency of the Poisson-Boltzmann equation solution by verifying 
        the first and second derivatives of the electric potential.

    test_ionProfiles():
        Validates the ion profiles and electroneutrality of the system by comparing 
        calculated results with reference data from the literature.

    Description:
    ------------
    - `test_solutionPBeq_selfConsistency`:
        This test initializes a `LinearPolyelectrolyte` instance with exemplary parameters, 
        solves the nonlinear Poisson-Boltzmann equation, and checks the self-consistency 
        of the solution by comparing the first and second derivatives of the electric 
        potential with the expected values.

    - `test_ionProfiles`:
        This test compares the calculated ion profiles and cumulative charge density 
        with reference data from Deserno et al., Macromolecules 2000. It also verifies 
        the electroneutrality of the system by ensuring the net charge is approximately zero.
    """

    def test_solutionPBeq_selfConsistency(self):

        # Exemplary parameters
        N_nodes = 50000
        a = 2 # [nm]
        Z = 1/5
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]

        my_suspension = linear_polyelec.LinearPolyelectrolyte(a, Z, lb, vol_frac, c_salt)

        r = np.linspace(a, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        phi_0, phi_R = -2, 0.0
        y_init[0] = np.linspace(phi_0, phi_R, r.size)  # Initial guess for phi
        my_suspension.solve_nonlin_PB(r, y_init)

        # Self-consistency of first and second derivative
        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot,
                                           my_suspension.elec_field, my_suspension.kappaa)
        
        self.assertAlmostEqual(residual_0, 0, places=2, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=1, msg=None, delta=None)

    def test_ionProfiles(self):

        # Reference data
        # Deserno et al., Macromolecules 2000, 33, 1, 199–206
        # https://doi.org/10.1021/ma990897o
        ref_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), './ref_data.dat'))
        r_ref = ref_data[:,0]
        P_ref = ref_data[:,1]

        # Exemplary parameters
        N_nodes = 50000
        a = 0.71 # [nm]
        xi_ref = 0.96
        Z = xi_ref / a
        lb = 0.71 # [nm]
        R_red = 123.8 # cell radius over polymer radius
        vol_frac = 1 / R_red**2
        c_salt = 1e-6 # [M]

        my_suspension = linear_polyelec.LinearPolyelectrolyte(a, Z, lb, vol_frac, c_salt)

        self.assertAlmostEqual(my_suspension.R_cell/my_suspension.a, R_red, places=8, msg=None, delta=None)
        self.assertAlmostEqual(my_suspension.xi, 0.96, places=8, msg=None, delta=None)

        # Iteration 1
        r = np.linspace(a, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)
        # Iteration 2
        y_init = np.zeros((2, r.size))
        y_init[0] = my_suspension.elec_pot  # Initial guess for phi
        my_suspension.solve_nonlin_PB(r, y_init)

        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot,
                                           my_suspension.elec_field, my_suspension.kappaa)
        self.assertAlmostEqual(residual_0, 0, places=2, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=1, msg=None, delta=None)
    
        # Check electroneutrality
        net_charge = my_suspension.total_charge(r[-1])
        self.assertAlmostEqual(net_charge, 0, places=5, msg=None, delta=None)

        # Compare cummulative density
        r = r_ref[:-1] * my_suspension.a
        total_q = np.array([my_suspension.total_charge(r_val) for r_val in r])
        P_calc = 1-np.abs(total_q)
        residual = np.max(np.abs(P_ref[:-1] - P_calc))
        self.assertAlmostEqual(residual, 0, places=1, msg=None, delta=None)
        


if __name__ == '__main__':
    ut.main()
