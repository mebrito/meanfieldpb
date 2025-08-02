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

from meanfieldpb.PBequations import PBequation_surfaceMicrogel_strong as PBstrong
from meanfieldpb import surface_microgel


def test_PBeq(r, phi, phi_prime, f_diff_eq, param):
    """
    Computes the residuals to test the self-consistency of the numerical solution
    to the Poisson-Boltzmann (PB) equation.

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
        Parameters required by the PB equation (length 3 for strong, 6 for weak charge cases).

    Returns
    -------
    residual_0 : float
        Maximum relative residual between numerical and provided first derivative.
    residual_1 : float
        Maximum relative residual between numerical and analytical second derivative.
    """

    def residual(quantity):
        quantity = np.abs(quantity)
        mask_less = (r > 1e-2) & (r < 0.9*param[-1])
        mask_mid = (r >= 1.01*param[-1]) & (r <= 0.99)
        mask_large = r > 1.01
        residual_less = quantity[mask_less]
        residual_mid = quantity[mask_mid]
        residual_large = quantity[mask_large]
        res = np.max(np.append(np.append(residual_less, residual_mid), residual_large))
        return res
    
    residual_0 = residual(np.gradient(phi, r) - phi_prime) / np.max(np.abs(phi_prime))
    phi_double_prime = np.gradient(phi_prime, r)
    if len(param)==3:
        kresa, Zlba, gamma = param
        residual_1 = residual(phi_double_prime - f_diff_eq(r, phi, phi_prime, kresa, Zlba, gamma)) / np.max(np.abs(phi_double_prime))
    else:
        kresa, k0_aa, xi_a, k0_ab, xi_b, gamma = param
        residual_1 = residual(phi_double_prime - f_diff_eq(r, phi, phi_prime, kresa, k0_aa, xi_a, k0_ab, xi_b, gamma)) / np.max(np.abs(phi_double_prime))

    return residual_0, residual_1


class TestSurfaceMicrogels(ut.TestCase):
    """
    Unit tests for the SurfaceMicrogel class and its Poisson-Boltzmann equation solver.

    This test class verifies the self-consistency of the numerical solution to the nonlinear
    Poisson-Boltzmann (PB) equation for a strongly charged surface microgel. The test checks
    that the first and second derivatives of the electrostatic potential and field are consistent
    with the expected PB differential equations.

    Methods
    -------
    test_solutionPBeq_selfConsistency_strong():
        Tests the self-consistency of the PB solution for a strongly charged surface microgel.
        Asserts that the numerical derivatives match the analytical expressions within a specified tolerance.

    The test uses a helper function `test_PBeq` to compute the residuals between numerical and analytical
    derivatives, and asserts that these residuals are close to zero.
    """

    def test_solutionPBeq_selfConsistency_strong(self):

        # Parameter
        N_nodes = 5000
        a = 50 # [nm]
        b = 40 # [nm]
        Z_a = 100
        Z_b = 0
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]
        charge_type = 'strong'

        my_suspension = surface_microgel.SurfaceMicrogel(a, b, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
        r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)

        # Self-consistency of first and second derivative
        param = (my_suspension.kresa, my_suspension.Zlba, my_suspension.gamma)
        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
                                           PBstrong.f_diff_eq, param)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)

    
    # def test_solutionPBeq_selfConsistency_weak(self):

    #     # Parameter
    #     N_nodes = 100000
    #     a = 50 # [nm]
    #     Z_a = 100
    #     Z_b = 0
    #     lb = 0.71 # [nm]
    #     vol_frac = 0.01
    #     c_salt = 1e-3 # [M]
    #     charge_type = 'weak'

    #     my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
    #     r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
    #     y_init = np.zeros((2, r.size))
    #     my_suspension.solve_nonlin_PB(r, y_init)

    #     # Self-consistency of first and second derivative
    #     param = (my_suspension.kresa, my_suspension.k0_aa, my_suspension.xi_a,
    #              my_suspension.k0_ab, my_suspension.xi_b)
    #     residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
    #                                        PBweak.f_diff_eq, param)

    #     self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
    #     self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)

    

if __name__ == '__main__':
    ut.main()
