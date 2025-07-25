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
This script computes and visualizes the electrostatic potential, electric field, and counterion density profile
for a charged microgel particle in a suspension using a mean-field Poisson-Boltzmann cell-model approach.

The computed counterion density profile is compared to reference data from:
Alziyadi et al., J. Chem. Phys. 159, 184901 (2023)
https://doi.org/10.1063/5.0161027

Parameters:
- N_nodes: Number of spatial grid points for numerical solution.
- a0: Reference microgel radius [nm].
- alpha: Swelling factor.
- a: Swollen microgel radius [nm].
- b: Microgel inner radius [nm]. 
- Z_a, Z_b: Microgel backbone charge numbers.
- lb: Bjerrum length [nm].
- vol_frac: Microgel volume fraction.
- c_salt: Salt concentration [M].
- charge_type: Type of microgel charge ('strong' or other).

Outputs:
- Plots of electrostatic potential and field.
- Plots of counterion density difference compared to reference data.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import surface_microgel

colors = ['tab:blue', 'tab:red', 'black']
markers = ['o', 's', '^']
font_size = 15

# Parameter
N_nodes = 20000
a0 = 10 # [nm]
alpha = 2.5
a =  alpha * a0 # [nm]
b = 0.995 * a # [nm] (internal radius)
Z_a = 1000
Z_b = 0
lb = 0.714 # [nm] (water at room temperature)
vol_fracs_0 = [0.01, 0.02, 0.03] 
c_salt = 7e-5 # [M]
charge_type = 'strong'

# Plotting electrostatic potential and field in two subplots sharing x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 12))

for i, vol_frac_0 in enumerate(vol_fracs_0):
    vol_frac = vol_frac_0 * alpha**3
    
    # Create suspension instance
    my_suspension = surface_microgel.SurfaceMicrogel(a, b, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)
    r = np.linspace(1e-8, my_suspension.R_cell, N_nodes)
    y_init = np.zeros((2, r.size))
    # Solve the non-linear Poisson-Boltzmann equation
    my_suspension.solve_nonlin_PB(r, y_init)

    # Plotting electrostatic potential and electric field
    ax1.plot(my_suspension.r/a, my_suspension.elec_pot, label=str(vol_frac_0), lw=2, color=colors[i])
    ax2.plot(my_suspension.r/a, my_suspension.elec_field, label=str(vol_frac_0), lw=2, color=colors[i])

    # Plotting counterion density profile from MeanFieldPB
    n_diff = my_suspension.cation_density() * my_suspension.conversion_factor # Convert to [nm^-3]
    ax3.plot(my_suspension.r/a, n_diff*a**3, label=str(vol_frac_0), lw=3, color=colors[i])

    # Plotting MD-simulation counterion density profile from reference
    ref_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), f'./ref_data/surf-microgel_MD_{str(vol_frac_0)}.dat'), skip_header=7)
    plt.plot(ref_data[:, 0], ref_data[:, 1],
             marker=markers[i], markersize=8, lw=0, fillstyle='none', color=colors[i])

    # Plotting PB counterion density profile from reference
    if vol_frac_0 == 0.03:
        ref_data_pb = np.genfromtxt(os.path.join(os.path.dirname(__file__), f'./ref_data/surf-microgel_PB_{str(vol_frac_0)}.dat'), skip_header=7)
        ax3.plot(ref_data_pb[:, 0], ref_data_pb[:, 1],
                 marker='', markersize=5, lw=2, linestyle=(0, (5, 5)), color='lightgray')


# Setting reference microgel radius line
ax1.axvline(x=1, color='gray', linestyle='--', label=r'microgel radius', lw=1, alpha=0.5)
ax2.axvline(x=1, color='gray', linestyle='--', lw=1, alpha=0.5)
ax3.axvline(x=1, color='gray', linestyle='--', lw=1, alpha=0.5)

# Setting labels and legends
ax3.set_xlabel('$r/a$', fontsize=font_size)
ax1.set_ylabel(r"Electrostatic potential $\phi(r)$", fontsize=font_size)
ax2.set_ylabel(r"Electric field $\phi'(r)$", fontsize=font_size)
ax3.set_ylabel(r"Counterion density $n_+(r) a^3$", fontsize=font_size)
ax1.legend(title='dry volume fraction', fontsize=font_size-2)
ax2.legend(title='dry volume fraction', fontsize=font_size-2)
ax3.legend(title='dry volume fraction', fontsize=font_size-2)

ax3.text(0.02, 0.95, "Solid lines: MeanFieldPB\nMarkers: MD simulations from Ref.\nDashed line: PB solution from Ref.", transform=ax3.transAxes,
         fontsize=font_size-3, verticalalignment='top', horizontalalignment='left')
plt.tight_layout()
plt.show()
