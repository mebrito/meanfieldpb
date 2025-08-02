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
"""
Utility functions for MeanFieldPB calculations.

This module provides helper functions for various calculations in colloidal
and polyelectrolyte systems, including reservoir concentration calculations
for charge-regulated systems and other physical property computations.
"""

import math


def reservoir_concent(c_Hres: float, c_salt: float, c_ref: float = 1.0) -> float:
    """
    Calculate the reservoir salt concentration for charge-regulated systems.

    This function computes the total reservoir concentration accounting for
    water autodissociation and charge regulation effects. It's particularly
    useful for systems with pH-dependent surface charge where the ionic
    strength must include contributions from both added salt and H⁺/OH⁻ ions.

    Parameters
    ----------
    c_Hres : float
        Hydrogen ion concentration in the reservoir [M].
    c_salt : float
        Added salt concentration in the reservoir [M].
    c_ref : float, optional
        Reference concentration for activity calculations [M]. Default is 1.0.

    Returns
    -------
    float
        Total reservoir ionic concentration [M], including contributions
        from H⁺, Na⁺, and the excess of either H⁺ or OH⁻ ions.

    Notes
    -----
    The calculation accounts for water autodissociation:

    .. math::
        K_w = [H^+][OH^-] = 10^{-14} \\text{ at 25°C}

    The hydroxide concentration is calculated as:

    .. math::
        [OH^-] = \\frac{K_w}{[H^+]}

    If [H⁺] ≥ [OH⁻] (acidic conditions):
        [Na⁺] = c_salt

    If [H⁺] < [OH⁻] (basic conditions):
        [Na⁺] = c_salt + ([OH⁻] - [H⁺])

    The total concentration is: c_res = [H⁺] + [Na⁺]

    Examples
    --------
    For neutral pH conditions:

    >>> c_Hres = 1e-7  # pH = 7
    >>> c_salt = 0.1   # 100 mM NaCl
    >>> c_total = reservoir_concent(c_Hres, c_salt)
    >>> print(f"Total concentration: {c_total:.6f} M")
    Total concentration: 0.100000 M

    For acidic conditions:

    >>> c_Hres = 1e-3  # pH = 3
    >>> c_salt = 0.01  # 10 mM NaCl
    >>> c_total = reservoir_concent(c_Hres, c_salt)
    >>> print(f"Total concentration: {c_total:.6f} M")
    Total concentration: 0.011000 M

    For basic conditions:

    >>> c_Hres = 1e-11  # pH = 11
    >>> c_salt = 0.01   # 10 mM NaCl
    >>> c_total = reservoir_concent(c_Hres, c_salt)
    >>> print(f"Total concentration: {c_total:.6f} M")
    Total concentration: 0.010999 M
    """
    K_w = 10 ** (-14)  # reaction constant autdissociation water
    c_OHres = c_ref**2 * K_w / c_Hres  # [M]
    if c_Hres >= c_OHres:
        c_Nares = c_salt
    else:
        c_Nares = c_salt + (c_OHres - c_Hres)

    return c_Hres + c_Nares


def KaCref(pK, c_ref):
    """
    return in units of c_ref, which is typically 1M
    """
    return 10.0 ** (-pK) * c_ref


def csch(x):
    return 1.0 / (math.sinh(x))


def coth(x):
    return 1.0 / (math.tanh(x))
