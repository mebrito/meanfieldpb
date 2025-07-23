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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PBequations import PBequation_volumeMicrogel_strong as PBstrong
from PBequations import PBequation_volumeMicrogel_weak as PBweak
import volume_microgel


def test_PBeq(r, phi, phi_prime, f_diff_eq, param):
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

    The tests use a helper function `test_PBeq` to compute the residuals between numerical and analytical
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
        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
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
        residual_0, residual_1 = test_PBeq(r/a, my_suspension.elec_pot, my_suspension.elec_field, 
                                           PBweak.f_diff_eq, param)

        self.assertAlmostEqual(residual_0, 0, places=4, msg=None, delta=None)
        self.assertAlmostEqual(residual_1, 0, places=4, msg=None, delta=None)


    # def test_solutionPBeq_only_acid_weak(self):

    #     # Reference solution 
    #     a = 50 # [nm]
    #     Z_a = 100
    #     Z_b = 0
    #     lb = 0.71 # [nm]
    #     vol_frac = 0.01
    #     c_salt = 1e-3 # [M]
    #     charge_type = 'strong'

    #     r_ref = np.array([1.000000e-08, 4.210111e-07, 8.320223e-07, 1.654045e-06,
    #     3.298089e-06, 6.586178e-06, 2.400405e-04, 4.734948e-04,
    #     6.155304e-02, 1.226326e-01, 2.939552e-01, 5.522017e-01,
    #     7.761009e-01, 1.000000e+00, 1.124145e+00, 1.248290e+00,
    #     1.372435e+00, 1.496580e+00, 1.620725e+00, 1.744870e+00,
    #     1.931088e+00, 2.117306e+00, 2.303523e+00, 2.489741e+00,
    #     2.666763e+00, 2.843784e+00, 3.041497e+00, 3.239209e+00,
    #     3.425427e+00, 3.611645e+00, 3.820852e+00, 4.030059e+00,
    #     4.237543e+00, 4.439566e+00, 4.641589e+00])
    #     phi_ref = np.array([-6.212661e-01, -6.212661e-01, -6.212661e-01, -6.212661e-01,
    #     -6.212661e-01, -6.212661e-01, -6.212660e-01, -6.212660e-01,
    #     -6.202327e-01, -6.171549e-01, -5.972970e-01, -5.328672e-01,
    #     -4.360829e-01, -2.894832e-01, -2.079398e-01, -1.512402e-01,
    #     -1.111115e-01, -8.230835e-02, -6.139625e-02, -4.606862e-02,
    #     -3.022536e-02, -2.001787e-02, -1.336217e-02, -8.979652e-03,
    #     -6.188152e-03, -4.285676e-03, -2.859896e-03, -1.921330e-03,
    #     -1.331322e-03, -9.326482e-04, -6.385837e-04, -4.541959e-04,
    #     -3.451174e-04, -2.894530e-04, -2.723604e-04])
    #     dphi_ref = np.array([0.000000e+00, 2.088797e-07, 4.480708e-07, 9.001317e-07,
    #     1.797277e-06, 3.589743e-06, 1.308353e-04, 2.580813e-04,
    #     3.360210e-02, 6.725514e-02, 1.659536e-01, 3.397099e-01,
    #     5.332615e-01, 7.885058e-01, 5.429584e-01, 3.813503e-01,
    #     2.720444e-01, 1.965185e-01, 1.434329e-01, 1.055947e-01,
    #     6.760151e-02, 4.385206e-02, 2.875235e-02, 1.901903e-02,
    #     1.292757e-02, 8.835294e-03, 5.805273e-03, 3.828279e-03,
    #     2.589036e-03, 1.746237e-03, 1.108768e-03, 6.814035e-04,
    #     3.865717e-04, 1.735477e-04, 0.000000e+00])
        
    #     my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)

    #     print('kresa ', my_suspension.kresa)
    #     print('Zlba ', my_suspension.Zlba)
    #     r = np.linspace(r_ref[0], a/vol_frac**(1./3.), 1000)
    #     y_init = np.zeros((2, r.size))
    #     # phi_0 = 0.0
    #     # phi_R = 0.0
    #     # y_init[0] = np.linspace(phi_0, phi_R, r.size)  # Initial guess for phi
    #     y_init[0] = y_init[0] * -0.5
    #     y_init[1] = y_init[1] * 1 
    #     print('y_init', y_init)
    #     my_suspension.solve_nonlin_PB(r, y_init)

    #     # --------------------------------------------------
    #     # Create a plot with two subplots
    #     fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    #     # First subplot: phi_ref vs r_ref
    #     axs[0].plot(r_ref, phi_ref, label='phi_ref vs r_ref', color='blue')
    #     axs[0].plot(r/a, my_suspension.elec_pot, '--', label='solution', color='blue')
    #     axs[0].set_xlabel('r_ref [nm]')
    #     axs[0].set_ylabel('phi_ref')
    #     axs[0].set_title('Electrostatic Potential vs Radius')
    #     axs[0].legend()
    #     axs[0].grid()

    #     # Second subplot: dphi_ref vs r_ref
    #     axs[1].plot(r_ref, dphi_ref, label='dphi_ref vs r_ref', color='red')
    #     axs[1].plot(r/a, my_suspension.elec_field, '--', label='solution', color='red')
    #     axs[1].set_xlabel('r_ref [nm]')
    #     axs[1].set_ylabel('dphi_ref')
    #     axs[1].set_title('Electric Field vs Radius')
    #     axs[1].legend()
    #     axs[1].grid()

    #     # Adjust layout and save the plot as a PDF
    #     plt.tight_layout()
    #     plt.savefig('phi_and_dphi_vs_r.pdf')
    #     plt.close()
    #     # --------------------------------------------------

    #     # --------------------------------------------------
    #     # Compute the second derivative of the solution
    #     second_derivative = np.gradient(dphi_ref, r_ref)

    #     # Create a new plot for the second derivative
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(r_ref, second_derivative, label='Second Derivative', color='green')
    #     plt.plot(r_ref, np.where(r_ref<1, -2 * dphi_ref / r_ref + my_suspension.kresa * my_suspension.kresa * np.sinh(phi_ref) - 3 * my_suspension.Zlba,
    #                         -2 * dphi_ref / r_ref + my_suspension.kresa * my_suspension.kresa * np.sinh(phi_ref)), '--', label='Analytical Second Derivative', color='orange')
    #     plt.xlabel('r')
    #     plt.ylabel('Second Derivative')
    #     plt.title('Second Derivative of Electric Field vs Radius')
    #     plt.legend()
    #     plt.grid()

    #     # Save the plot as a PDF
    #     plt.tight_layout()
    #     plt.savefig('second_derivative_vs_r_matlab.pdf')
    #     plt.close()
    #     # --------------------------------------------------

    #     # --------------------------------------------------
    #     # Compute the second derivative of the solution
    #     second_derivative = np.gradient(my_suspension.elec_field, r/a)

    #     # Create a new plot for the second derivative
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(r/a, second_derivative, label='Second Derivative', color='green')
    #     plt.plot(r/a, np.where(r/a<1, -2 * my_suspension.elec_field / r + my_suspension.kresa * my_suspension.kresa * np.sinh(my_suspension.elec_pot) - 3 * my_suspension.Zlba,
    #                         -2 * my_suspension.elec_field  / r + my_suspension.kresa * my_suspension.kresa * np.sinh(my_suspension.elec_pot)), '--', label='Analytical Second Derivative', color='orange')
    #     plt.xlabel('r/a')
    #     plt.ylabel('Second Derivative')
    #     plt.title('Second Derivative of Electric Field vs Radius')
    #     plt.legend()
    #     plt.grid()

    #     # Save the plot as a PDF
    #     plt.tight_layout()
    #     plt.savefig('second_derivative_vs_r.pdf')
    #     plt.close()
    #     # --------------------------------------------------
    #     # Self-consistency of first and second derivative
    #     np.testing.assert_almost_equal([my_suspension.elec_pot, my_suspension.elec_field], 
    #                                    [phi_ref, dphi_ref],
    #                                     decimal=2)
    

if __name__ == '__main__':
    ut.main()
