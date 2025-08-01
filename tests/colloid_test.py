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
from PBequations import PBequation_colloid_strong as PBstrong
import colloid

# import matplotlib.pyplot as plt

def test_PBeq(r, phi, phi_prime, f_diff_eq, param):
    """
    Computes the residuals to test the self-consistency of the numerical solution
    to the Poisson-Boltzmann (PB) equation for colloidal systems.

    This function compares the numerical first and second derivatives of the electrostatic
    potential (phi) and field (phi_prime) with their analytical expressions given by the PB
    differential equation. It returns the maximum relative residuals for both the first and
    second derivative consistency checks, considering only the region where r > 1.01.

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
        mask = r > 1.01
        residual = quantity[mask]
        res = np.max(residual)
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


class TestColloid(ut.TestCase):
    """
    Unit tests for the Colloid class and its Poisson-Boltzmann equation solver.

    This test class verifies the self-consistency of the numerical solution to the nonlinear
    Poisson-Boltzmann (PB) equation for a strongly charged colloidal suspension. The test checks
    that the first and second derivatives of the electrostatic potential and field are consistent
    with the expected PB differential equations.

    Methods
    -------
    test_solutionPBeq_selfConsistency_strong():
        Tests the self-consistency of the PB solution for a strongly charged colloidal system.
        Asserts that the numerical derivatives match the analytical expressions within a specified tolerance.

    The test uses a helper function `test_PBeq` to compute the residuals between numerical and analytical
    derivatives, and asserts that these residuals are close to zero.
    """

    def test_solutionPBeq_selfConsistency_strong(self):

        # Parameter
        N_nodes = 5000
        a = 50 # [nm]
        Z = 100
        lb = 0.71 # [nm]
        vol_frac = 0.01
        c_salt = 1e-3 # [M]
        charge_type = 'strong'

        my_suspension = colloid.Colloid(a, Z, lb, vol_frac, c_salt, charge_type)
        r = np.linspace(a, my_suspension.R_cell, N_nodes)
        y_init = np.zeros((2, r.size))
        my_suspension.solve_nonlin_PB(r, y_init)

        # Self-consistency of first and second derivative
        param = (my_suspension.kresa, my_suspension.Zlba)
        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
                                           PBstrong.f_diff_eq, param)

        # print('residual_0', residual_0)
        # print('residual_1', residual_1)
        
        # # Plot electric potential
        # plt.figure(figsize=(10, 5))
        # plt.plot(r, my_suspension.elec_pot, label='Electric Potential (phi)')
        # plt.plot(r, my_suspension.elec_field, label='Electric Field (phi\')', color='orange')
        # plt.xlabel('r (in units of a)')
        # plt.ylabel('Electric Potential')
        # plt.title('Electric Potential vs Radial Distance')
        # plt.legend()
        # plt.grid()
        # plt.show()

        # # Plot second derivative
        # plt.figure(figsize=(10, 5))
        # plt.plot(r/a, np.gradient(my_suspension.elec_field, r/a), 'x', fillstyle='none', label='Second Derivative of Electric Field')
        # plt.plot(r/a, PBstrong.f_diff_eq(r/a, my_suspension.elec_pot, my_suspension.elec_field,
        #                                my_suspension.kresa, my_suspension.Zlba), 'o', fillstyle='none', label='Analytical Second Derivative', color='red')
        # plt.xlabel('r (in units of a)')
        # plt.ylabel('Second Derivative')
        # plt.title('Second Derivative of Electric Field vs Radial Distance')
        # plt.legend()
        # plt.grid()
        # plt.show()

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)