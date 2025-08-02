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
import numpy as np


def vol(a):
    return (4 * math.pi * a**3) / 3


def inverseLangevin(x):
    return (
        x
        * (3 - 1.00651 * x**2 - 0.962251 * x**4 + 1.47353 * x**6 - 0.48953 * x**8)
        / ((1 - x) * (1 + 1.01524 * x))
    )


def AA(Nmonch, Nch, a0):
    """
    A factor calculation for tranformation from chain end-to-end distance
    to swelling parameter variation
    Nmonch: number of monomer per chain,
    Nch: number of chains.
    a0: dry radius

    Units: a0 should be in units of sigma.
    """
    Rmax = Nmonch + 1  # in units of sigma
    return a0 * (math.pi / Nch) ** (1 / 3) * 3 ** (1 / 6) / Rmax


def elastic(a, a0, Nch, Nmonch):
    """
    gel elastic pressure contribution for gaussian-chain spherical
    network of
    a: radius
    a0: dry radius
    Nch: number of chains. Default in [nm].

    Units: a and a0 in units of sigma
    """
    A = AA(Nmonch, Nch, a0)
    return (3 / (4 * math.pi * a**3)) * (-Nch * (a / a0) ** 2 * Nmonch * A**2)


def elastic_crosslinked(a, a0, Nch, Nmonch):
    """
    gel elastic pressure contribution for crosslinked gaussian
    spherical network of
    a: radius
    a0: dry radius
    Nch: number of chains. Default in [nm].

    Units: a and a0 in units of sigma
    """
    return (3 / (4 * math.pi * a**3)) * (-Nch * ((a / a0) ** 2 - 1 / 2))


def elastic_Langevin(a, Nmonch, Nch, N_m, a0):
    """
    gel elastic pressure contribution for finite-extensibility-chain spherical
    network of
    a: radius,
    a0: dry radius,
    Nch: number of chains,
    Nmonch: number of monomer per chain,
    N_ch: number of chains,
    sigma: monomer size [nm]

    Units: a and a0 in units of sigma
    """
    A = AA(Nmonch, Nch, a0)
    alpha = a / a0
    return (3 / (4 * math.pi * a**3)) * (
        -N_m * A * alpha * inverseLangevin(A * alpha) / 3
    )


def mixing_interaction_contrib(a, a0, N_m, chi):
    """
    mixing-entropy and interaction-energy pressure contributions for a spherical network
    in a solvent such that
    a: radius,
    a0: dry radius,
    N_m: total number of monomers in the netwrok,
    chi: Flory chi interaction parameter.

    Units: a and a0 in units of sigma
    """
    alpha = a / a0
    return (
        (3 / (4 * math.pi * a**3))
        * (-N_m)
        * (alpha**3 * np.log(1 - alpha ** (-3)) + chi / alpha**3 + 1)
    )


def pi_g(a, a0, N_m, Nch, Nmonch, chi):
    """
    gel intrinsic osmotic pressure for gaussian-chain network. Units: a and a0 in units of sigma
    """
    return mixing_interaction_contrib(a, a0, N_m, chi) + elastic(a, a0, Nch, Nmonch)


def pi_g_crosslinked(a, a0, N_m, Nch, Nmonch, chi):
    """
    gel intrinsic osmotic pressure for crosslinked gaussian-chain network. Units: a and a0 in units of sigma
    """
    return mixing_interaction_contrib(a, a0, N_m, chi) + elastic_crosslinked(
        a, a0, Nch, Nmonch
    )


def pi_g_Langevin(a, Nmonch, Nch, a0, N_m, chi):
    """
    gel intrinsic osmotic pressure for finite extensible chains (Langevin model).
    Distances in units of sigma
    """
    return mixing_interaction_contrib(a, a0, N_m, chi) + elastic_Langevin(
        a, Nmonch, Nch, N_m, a0
    )
