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
This script computes and visualizes the electrostatic potential, electric field, and ion density profiles 
for a charged microgel particle in a suspension using a mean-field Poisson-Boltzmann cell-model approach.

The computed ion density profile is compared to reference data from:
Brito et al., J. Chem. Phys. 151, 224901 (2019)
https://doi.org/10.1063/1.5129575

Parameters:
- N_nodes: Number of spatial grid points for numerical solution.
- a0: Reference microgel radius [nm].
- alpha: Swelling factor.
- a: Swollen microgel radius [nm].
- Z_a, Z_b: Microgel backbone charge numbers.
- lb: Bjerrum length [nm].
- vol_frac: Microgel volume fraction.
- c_salt: Salt concentration [M].
- charge_type: Type of microgel charge ('strong' or other).

Outputs:
- Plots of electrostatic potential and field.
- Plots of ion density difference compared to reference data.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import volume_microgel

colors = ['tab:blue', 'tab:red']
font_size = 15

# Parameter
N_nodes = 5000
a0 = 10 # [nm]
alpha = 2.612
a =  alpha * a0 # [nm]
Z_a = 500
Z_b = 0
lb = 0.71 # [nm] (water at room temperature)
vol_frac = 0.005 * alpha**3
c_salt = 1e-4 # [M]
charge_type = 'strong'

# Create suspension instance
my_suspension = volume_microgel.VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
y_init = np.zeros((2, r.size))
# Solve the non-linear Poisson-Boltzmann equation
my_suspension.solve_nonlin_PB(r, y_init)

# Plotting electrostatic potential and field
plt.plot(my_suspension.r/a0, my_suspension.elec_pot, label=r'electrostatic potential $\phi(r)$', lw=2, color=colors[0])
plt.plot(my_suspension.r/a0, my_suspension.elec_field, label=r"electric field $\phi'(r)$", lw=2, color=colors[1])

# Setting reference microgel radius line
plt.axvline(x=alpha, color='gray', linestyle='--', label=r'microgel radius $a/a_0$', lw=1, alpha=0.5)
# Setting labels, legends and show
plt.xlabel('$r/a_0$', fontsize=font_size)
plt.ylabel(r"$\phi(r)$, $\phi'(r)$", fontsize=font_size)
plt.legend()
plt.show()

# Plotting ion density profile
n_diff = (my_suspension.cation_density() - my_suspension.anion_density()) * my_suspension.conversion_factor # Convert to [nm^-3]
plt.plot(my_suspension.r/a0, n_diff*a0**3, label=r'MeanFieldPB', lw=2, color=colors[0])
# Plotting ion density profile from reference
ref_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), './ref_data/microgel.dat'), skip_header=7)
plt.plot(ref_data[:, 0], ref_data[:, 1], label=r'reference data', marker='o', markersize=4, lw=0, fillstyle='none', color=colors[1])

# Setting reference microgel radius line
plt.axvline(x=alpha, color='gray', linestyle='--', label=r'microgel radius $a/a_0$', lw=1, alpha=0.5)
# Setting labels, legends and show
plt.xlabel('$r/a_0$', fontsize=font_size)
plt.ylabel(r"$[n_+(r) - n_-(r)]a_0^3$", fontsize=font_size)
plt.legend()
plt.show()