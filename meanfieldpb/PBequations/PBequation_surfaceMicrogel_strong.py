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
import numpy as np


def f_diff_eq(r, y, y_prime, kresa, Zlba, gamma):
    """
    Function that defines the second derivative in the differential
    equation y'' = f(r, y, y'; kresa, Zlba, gamma)

    All quantities should be in reduced units.
    """

    return np.where(
        r <= gamma,
        -2 * y_prime / r + kresa * kresa * np.sinh(y),
        np.where(
            (r > gamma) & (r < 1),
            -2 * y_prime / r + kresa * kresa * np.sinh(y) - 3 * Zlba / (1 - gamma**3),
            -2 * y_prime / r + kresa * kresa * np.sinh(y),
        ),
    )


# Define the ODE system
def odes(r, y, p):
    """
    ODE for volume-charge microgels strongly charged

    r (float): grid in the radial axis, in units of a
    y (np.array(2)): y[0] electric potential, y[1] electric field.
    p (np.array(2)): equation parameters [kresa, Zlba].
    """
    kresa, Zlba, gamma = p
    dydr = np.zeros_like(y)
    dydr[0] = y[1]
    dydr[1] = f_diff_eq(r, y[0], y[1], kresa, Zlba, gamma)
    return dydr


# Define the boundary conditions
def boundary_conditions(ya, yb):
    """
    boundary conditions of the ODE for volume-charge microgels strongly charged

    ya (np.array(2)): ya[0] electric potential and ya[1] electric field at r[0].
    yb (np.array(2)): yb[0] electric potential and yb[1] electric field at r[-1].
    p (np.array(2)): equation parameters [kresa, Zlba].
    """
    phi_prime_0 = 0.0  # Boundary condition at r=0
    phi_prime_R = 0.0  # Boundary condition at r=R
    return np.array([ya[1] - phi_prime_0, yb[1] - phi_prime_R])
