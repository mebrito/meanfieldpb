#
# MeanFieldPB is a package for modeling the electrostatics of charge-equilibrated
# colloidal and polymer suspensions using mean-field Poisson-Boltzmann equations.

# Copyright (C) 2025 The MeanFieldPB Project

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
This script computes and visualizes the electrostatic potential, electric field, and probability density
profiles for a infinite polyelectrolyte in a suspension using a mean-field Poisson-Boltzmann cell-model
approach.

The computed probability density profile is compared to reference data from:
Deserno et al., Macromolecules 2000, 33, 1, 199-206
https://doi.org/10.1021/ma990897o

Parameters:
- N_nodes: Number of spatial grid points for numerical solution.
- a: Polyelectrolyte radius [nm].
- xi_ref: reference linear charge density [1/nm].
- lb: Bjerrum length [nm].
- R_red: Reduced cell radius (over the polyelectrolyte radius).
- c_salt: Salt concentration [M].

Outputs:
- Plots of electrostatic potential and field.
- Plots of probability density compared to reference data.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import linear_polyelec

colors = ['tab:blue', 'tab:red']
font_size = 15

# Parameters
N_nodes = 50000
a = 0.71 # [nm]
xi_ref = 0.96
Z = xi_ref / a
lb = 0.71 # [nm]
R_red = 123.8 # cell radius over polymer radius
vol_frac = 1 / R_red**2
c_salt = 1e-6 # [M]

# Create suspension instance
my_suspension = linear_polyelec.LinearPolyelectrolyte(a, Z, lb, vol_frac, c_salt)

# Solve the non-linear Poisson-Boltzmann equation
r = np.linspace(a, my_suspension.R_cell, N_nodes)
y_init = np.zeros((2, r.size))
my_suspension.solve_nonlin_PB(r, y_init)

# Plotting electrostatic potential and field
plt.semilogx(my_suspension.r/a, my_suspension.elec_pot, label=r'electrostatic potential $\phi(r)$', lw=2, color=colors[0])
plt.semilogx(my_suspension.r/a, my_suspension.elec_field, label=r"electric field $\phi'(r)$", lw=2, color=colors[1])
plt.xlabel('$r/a_0$', fontsize=font_size)
plt.ylabel(r"$\phi(r)$, $\phi'(r)$", fontsize=font_size)
plt.legend()
plt.show()

# Compare cummulative probability density
# Plotting probability density profile from reference
ref_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), './ref_data/lin_polyelec.dat'), skip_header=7)
plt.semilogx(ref_data[:, 0], ref_data[:, 1], label=r'reference data', marker='o', markersize=4, lw=0, fillstyle='none', color=colors[1])

# Calculating and plotting cummulative probability density
# P(r) = 1 - |Q(r)|, where Q(r) is the total charge up to radius r
r = ref_data[:-1, 0] * my_suspension.a
total_q = np.array([my_suspension.total_charge(r_val) for r_val in r])
P_calc = 1-np.abs(total_q)
plt.semilogx(r/my_suspension.a, P_calc, label='MeanFieldPB', lw=2, color=colors[0])

# Setting labels, legends and show
plt.xlabel('$r/a_0$', fontsize=font_size)
plt.ylabel(r"$P(r)$", fontsize=font_size)
plt.legend()
plt.show()