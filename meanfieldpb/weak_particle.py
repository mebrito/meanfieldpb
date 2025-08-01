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

class WeakParticle():
    """
    Base class for containing parameters relative to weak charges.

    Parameters:
    -----------
    Z_a : float
        Maximum particle charge from acid sites in units of the fundamental charge `e`.
    Z_b : float
        Maximum particle charge from basic sites in units of the fundamental charge `e`.
    pK_a : float
        Dissociation constant of acid sites.
    pK_b : float
        Dissociation constant of basic sites.
    pH_res : float
        Reservoir pH.

    Attributes:
    -----------
    c_ref : float
        Reference concentration in molars [M] (default is 1 M).
    c_Hres : float
        Reservoir hydrogen ion (H+) concentration in molars [M], calculated as `c_ref * 10^(-pH_res)`.
    Ka : float
        Acid dissociation constant, calculated as `c_ref * 10^(-pK_a)`, ( K = K_a,acid*c_ref ).
    Kb : float
        Basic dissociation constant, calculated as `c_ref * 10^(-pK_b)`, ( K = K_a,basic*c_ref ).
    xi_a : float
        Coupling parameter for acid sites.
    xi_b : float
        Coupling parameter for basic sites.

    Description:
    ------------
    The `WeakParticle` class provides a framework for modeling particles with weak charges. 
    It initializes key parameters such as the maximum charges from acid and basic sites, 
    dissociation constants, and reservoir pH. The class also calculates derived attributes 
    like the reservoir hydrogen ion concentration (`c_Hres`) and dissociation constants 
    (`Ka` and `Kb`) based on the input parameters.
    """

    c_ref = 1 # reference concentration [M]

    def __init__(self, Z_a, Z_b, pK_a, pK_b, pH_res) -> None:
        self.Z_a = Z_a
        self.Z_b = Z_b
        self.pK_a = pK_a
        self.pK_b = pK_b
        self.pH_res = pH_res

        self.c_Hres = self.c_ref * 10**(-pH_res)
        self.Ka = self.c_ref * 10^(-self.pK_a); # K = K_a,acid*c_ref
        self.Kb = self.c_ref * 10^(-self.pK_b); # K = K_a,basic*c_ref

        self.xi_a = self.c_Hres/self.Ka
        self.xi_b = self.c_Hres/self.Kb