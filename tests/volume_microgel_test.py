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

from meanfieldpb.PBequations import PBequation_volumeMicrogel_strong as PBstrong
from meanfieldpb.PBequations import PBequation_volumeMicrogel_weak as PBweak
from meanfieldpb import volume_microgel


def _test_PBeq(r, phi, phi_prime, f_diff_eq, param):
    """
    Computes the residuals to test the self-consistency of the numerical solution
    to the Poisson-Boltzmann (PB) equation for volume microgels.

    This function compares the numerical first and second derivatives of the electrostatic
    potential (phi) and field (phi_prime) with their analytical expressions given by the PB
    differential equation. It returns the maximum relative residuals for both the first and
    second derivative consistency checks.

    Parameters
    ----------
    r : ndarray
        Radial coordinate array (normalized).
    phi : ndarray
        Electrostatic potential profile.
    phi_prime : ndarray
        Electrostatic field profile (first derivative of phi).
    f_diff_eq : function
        Function representing the right-hand side of the PB differential equation.
    param : tuple
        Parameters required by the PB equation (length 2 for strong, 5 for weak charge cases).

    Returns
    -------
    residual_0 : float
        Maximum relative residual between numerical and provided first derivative.
    residual_1 : float
        Maximum relative residual between numerical and analytical second derivative.
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
    if len(param)==2:
        kresa, Zlba = param
        residual_1 = residual(phi_double_prime - f_diff_eq(r, phi, phi_prime, kresa, Zlba)) / np.max(np.abs(phi_double_prime))
    else:
        kresa, k0_aa, xi_a, k0_ab, xi_b = param
        residual_1 = residual(phi_double_prime - f_diff_eq(r, phi, phi_prime, kresa, k0_aa, xi_a, k0_ab, xi_b)) / np.max(np.abs(phi_double_prime))

    return residual_0, residual_1


class TestVolumeMicrogels(ut.TestCase):
    """
    Unit tests for the VolumeMicrogel class and its Poisson-Boltzmann equation solvers.

    This test class contains methods to verify the self-consistency of the numerical solutions
    to the nonlinear Poisson-Boltzmann (PB) equations for both 'strong' and 'weak' charge types
    in volume microgels. The tests check that the first and second derivatives of the electrostatic
    potential and field are consistent with the expected PB differential equations.

    Methods
    -------
    test_solutionPBeq_selfConsistency_strong():
        Tests the self-consistency of the PB solution for a strongly charged volume microgel.
        Asserts that the numerical derivatives match the analytical expressions within a specified tolerance.

    test_solutionPBeq_selfConsistency_weak():
        Tests the self-consistency of the PB solution for a weakly charged volume microgel.
        Asserts that the numerical derivatives match the analytical expressions within a specified tolerance.

    The tests use a helper function `_test_PBeq` to compute the residuals between numerical and analytical
    derivatives, and assert that these residuals are close to zero.
    """

    def test_solutionPBeq_selfConsistency_strong(self):

        # Parameter
        N_nodes = 5000
        a = 50 # [nm]
        Z_a = 100
        Z_b = 0
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]
        charge_type = 'strong'

        my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
        r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)

        # Self-consistency of first and second derivative
        param = (my_suspension.kresa, my_suspension.Zlba)
        residual_0, residual_1 = _test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
                                           PBstrong.f_diff_eq, param)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)

    
    def test_solutionPBeq_selfConsistency_weak(self):

        # Parameter
        N_nodes = 100000
        a = 50 # [nm]
        Z_a = 100
        Z_b = 0
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]
        charge_type = 'weak'

        my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
        r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)

        # Self-consistency of first and second derivative
        param = (my_suspension.kresa, my_suspension.k0_aa, my_suspension.xi_a,
                 my_suspension.k0_ab, my_suspension.xi_b)
        residual_0, residual_1 = _test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
                                           PBweak.f_diff_eq, param)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)
    

if __name__ == '__main__':
    ut.main()
