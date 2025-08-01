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

from dataclasses import dataclass
import numpy as np
import math, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import reservoir_concent, KaCref, csch, coth
from suspension import Suspension

# REMOVE this function in the future to avoid hidden duplication of code
def variables(a, lb, vol_frac, c_salt, Zmax_a, Zmax_b, pKa, pKb, pH, c_ref=1):
    """
    Basic variables of the problem.

    Inputs:
    a : 
        microgel radius in [nm]
    lb :
        solvent Bjerrum length in [nm]
    vol_frac:
        microgel volume fraction
    c_salt : 
        reseroir salt concentration
    Zmax_a, Zmax_b :
        maximum charge of acidic and basic sites
    pKa, pKb :
        pK constants for acidic and basic sites
    pH : 
        reservoir pH

    """
    conversion_factor = Suspension._conversion_factor # If the quatity is in molar, multiplication by this factor gives density in number of particles per nm^3

    R = (a**3 / vol_frac)**(1./3.) # cell radius
    c_Hres = c_ref * 10**(-pH) # [M] # H^+ reservoir concentration 
    c_res = reservoir_concent(c_Hres, c_salt, c_ref) # total reservoir microion concentration
    c_tilde = c_res * conversion_factor # in number of particles per nm^3
    k_res = math.sqrt(8 * math.pi * lb * c_tilde)

    KaacidCref = KaCref(pKa, c_ref)
    c0acid_tilde = 3 * Zmax_a / (4 * math.pi * a**3) # concentration of acidic sites in number of particles per nm^3
    k_acid = np.sqrt(4 * math.pi * lb * c0acid_tilde * KaacidCref / (c_Hres + KaacidCref))
    h_acid = c_Hres / (c_Hres + KaacidCref)
    
    KabasicCref = KaCref(pKb, c_ref)
    c0basic_tilde = 3 * Zmax_b / (4 * math.pi * a**3) # concentration of acidic sites in number of particles per nm^3
    k_basic = np.sqrt(4 * math.pi * lb * c0basic_tilde * c_Hres / (c_Hres + KabasicCref))
    h_basic = KabasicCref / (c_Hres + KabasicCref)

    return R, c_Hres, k_res, k_acid, h_acid, k_basic, h_basic

@dataclass
class LinearVolumeMicrogelWeak():
    """
    Linear solution of the PB equation accounting for charge regulation with
    both acid and basic groups for spherical microgels. This corresponds to
    the called Debye-Hückel approximation, which is valid for low concentrations
    of microgels and low electrostatic potentials.

    Inputs:
    r :
        radial distance in [nm]
    a : 
        microgel radius in [nm]
    lb :
        solvent Bjerrum length in [nm]
    vol_frac:
        microgel volume fraction
    c_salt : 
        reservoir salt concentration
    Zmax_a, Zmax_b :
        maximum charge of acidic and basic sites
    pKa, pKb :
        pK constants for acidic and basic sites
    pH : 
        reservoir pH

    Output is dimensionless
    """
    a: float
    lb: float
    vol_frac: float
    c_salt: float
    Zmax_a: float
    Zmax_b: float
    pKa: float
    pKb: float
    pH: float

    def __post_init__(self):
        R, c_Hres, k_res, k_acid, h_acid, k_basic, h_basic = variables(self.a, self.lb, self.vol_frac, self.c_salt, self.Zmax_a, self.Zmax_b, self.pKa, self.pKb, self.pH)
        self.zeta = math.sqrt(k_res**2 + h_acid * k_acid**2 + h_basic * k_basic**2)

        self.c = (k_acid**2 - k_basic**2) / self.zeta**2

        self.AA = (self.c*(math.exp(2*R*k_res)*(1 + self.a*k_res)*(-1 + R*k_res) - math.exp(2*self.a*k_res)*(-1 + self.a*k_res)*(1 + R*k_res))*csch(self.a*self.zeta))/((-math.exp(2*self.a*k_res))*(1 + R*k_res)*(k_res - self.zeta*coth(self.a*self.zeta)) + math.exp(2*R*k_res)*(-1 + R*k_res)*(k_res + self.zeta*coth(self.a*self.zeta)))

        self.BB = (self.c*math.exp(self.a*k_res)*(1 + R*k_res)*(self.a*self.zeta*math.cosh(self.a*self.zeta) - math.sinh(self.a*self.zeta)))/(math.exp(2*self.a*k_res)*(1 + R*k_res)*((-self.zeta)*math.cosh(self.a*self.zeta) + k_res*math.sinh(self.a*self.zeta)) - math.exp(2*R*k_res)*(-1 + R*k_res)*
            (self.zeta*math.cosh(self.a*self.zeta) + k_res*math.sinh(self.a*self.zeta)))

        self.CC = -((self.c*math.exp((self.a + 2*R)*k_res)*(-1 + R*k_res)*(self.a*self.zeta*math.cosh(self.a*self.zeta) - math.sinh(self.a*self.zeta)))/(math.exp(2*self.a*k_res)*(1 + R*k_res)*(self.zeta*math.cosh(self.a*self.zeta) - k_res*math.sinh(self.a*self.zeta)) + math.exp(2*R*k_res)*(-1 + R*k_res)*
            (self.zeta*math.cosh(self.a*self.zeta) + k_res*math.sinh(self.a*self.zeta))))
        
        self.k_res = k_res
        
    def lin_elec_pot(self, r):
        """
        Linear electrostatic potential.
        """
        def inner(x):
            return self.AA * np.sinh(self.zeta * x) / x - self.c

        def outer(x):
            return self.BB * np.exp(self.k_res * x) / x + self.CC * np.exp(-self.k_res * x)/ x

        return np.piecewise(r, [r <= self.a, self.a < r], [lambda r: inner(r), lambda r: outer(r)])

    def lin_elec_field(self, r):
        """
        Linear electric field.
        """
        def inner(x):
            term1 = x * self.zeta * np.cosh(self.zeta * x)
            term2 = np.sinh(self.zeta * x)
            return self.AA * (term1 - term2) / (x**2)
        
        def outer(x):
            term1 = self.BB * np.exp(2 * self.k_res * x) * (self.k_res * x - 1)
            term2 = self.CC * (self.k_res * x + 1) 
            return np.exp(-self.k_res * x) * (term1 - term2) / (x**2)

        return np.piecewise(r, [r <= self.a, self.a < r], [lambda r: inner(r), lambda r: outer(r)])