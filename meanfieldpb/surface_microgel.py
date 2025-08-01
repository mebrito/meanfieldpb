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
from dataclasses import dataclass, field
from suspension import Suspension
from weak_particle import WeakParticle
import numpy as np
from scipy.integrate import solve_bvp
import numpy as np
import math
from util import reservoir_concent
from PBequations import PBequation_surfaceMicrogel_strong as PBstrong

class SurfaceMicrogel(Suspension, WeakParticle):
    """
    Child class of Suspension representing a microgel suspension of surface-charged
    microgels.
    a: float # Particle (external) radius in [nm]
    b: float # Particle (internal) radius in [nm] (b < a)
    """
    def __init__(self, a, b, Z_a, Z_b, lb, vol_frac, c_salt, charge_type, pK_a=4, pK_b=4, pH_res=7):
        self.b = b
        Suspension.__init__(self, a, Z_b-Z_a, lb, c_salt, charge_type)

        if self.charge_type=='strong':
            self._c_tilde = c_salt * self.conversion_factor # in number of particles per nm^3
            self._k0_aa = None
            self._k0_ab = None
        elif self.charge_type=='weak':
            WeakParticle.__init__(self, Z_a, Z_b, pK_a, pK_b, pH_res)
            self._c_tilde = reservoir_concent(self.c_Hres, self.c_salt, self.c_ref) * self.conversion_factor
            self._k0_aa = math.sqrt(3 * self.lb * self.Z_a / self.a)
            self._k0_ab = math.sqrt(3 * self.lb * self.Z_b / self.a)

        self._vol_frac = vol_frac
        self._R_cell = a / vol_frac**(1./3)
        self._kresa = self._screening_(self.c_tilde)
        self._gamma = self.b / self.a
        assert self._gamma < 1, "The internal radius should be smaller than the external radius."
        assert self._gamma > 0, "Internal and external radii should be larger than 0."

    @property
    def vol_frac(self):
        return self._vol_frac

    @property
    def R_cell(self):
        return self._R_cell

    @property
    def kresa(self):
        return self._kresa

    @property
    def c_tilde(self):
        return self._c_tilde

    @property
    def k0_aa(self):
        return self._k0_aa

    @property
    def k0_ab(self):
        return self._k0_ab
    
    @property
    def gamma(self):
        return self._gamma

    def set_grid(self, r: np.array):
        """
        Set grid r for solution of PB equation.
        """
        self.r = r

        # Assert if all components are increasing
        assert np.all(np.diff(r) > 0), "Not all components are increasing"

        # Assert if all components are larger or equal than 0
        assert self.r[0] >= 0, "First component of the grid larger than 0"

        # Assert if all components are smaller or equal than the cell radius
        assert self.r[-1] <= self.R_cell, "Last component of the grid smaller than R_coll"

    
    def solve_nonlin_PB(self, r, y_init):
        """
        """
        # For solving the PB equation, distances should be given in units of self.a
        r = r / self.a
        if self.charge_type=='strong':
            params = np.array([self.kresa, self.Zlba, self.gamma])
            solution = solve_bvp(lambda r, y: PBstrong.odes(r, y, params), 
                        lambda ya, yb: PBstrong.boundary_conditions(ya, yb), r, y_init)
        
        assert solution.p==None, "Parameters of the Poisson-Boltzmann equation have been fitted."
        assert solution.success==0, "The solution of the Poisson-Boltzmann equation was not successful."
        sol = solution.sol(r)
        self.elec_pot = sol[0]
        self.elec_field = sol[1]
        self.r = r * self.a

    def lin_elec_pot(self, r):
        return super().lin_elec_pot(r)

    def lin_elec_field(self, r):
        return super().lin_elec_field(r)