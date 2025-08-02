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
from meanfieldpb.suspension import Suspension
from meanfieldpb.weak_particle import WeakParticle
import numpy as np
from scipy.integrate import solve_bvp
import math
from meanfieldpb.PBequations import PBequation_volumeMicrogel_strong as PBstrong
from meanfieldpb.PBequations import PBequation_volumeMicrogel_weak as PBweak
from meanfieldpb.util import reservoir_concent
from meanfieldpb.LinearPB.linear_volume_microgel_weak import LinearVolumeMicrogelWeak

class VolumeMicrogel(Suspension,WeakParticle):
    """
    Represents a microgel suspension of volume-charged microgels.

    This class is a child of the `Suspension` and `WeakParticle` classes, designed to model 
    microgel suspensions with either strong or weakly charged particles. It includes methods 
    for setting up the system, solving the Poisson-Boltzmann equation, and calculating 
    reservoir concentrations.

    Parameters:
    -----------
    a : float
        Particle radius in nanometers [nm].
    Z_a : float
        Maximum charge of acid sites in units of the fundamental charge `e`.
    Z_b : float
        Maximum charge of basic sites in units of the fundamental charge `e`.
    lb : float
        Bjerrum length in nanometers [nm].
    vol_frac : float
        Suspension volume fraction.
    c_salt : float
        Reservoir salt concentration in molars [M].
    charge_type : str
        Type of particle charge, either 'strong' or 'weak'.
    pK_a : float, optional
        pK of the acid sites, required if `charge_type='weak'`. Default is 4.
    pK_b : float, optional
        pK of the basic sites, required if `charge_type='weak'`. Default is 4.
    pH_res : float, optional
        Reservoir pH, required if `charge_type='weak'`. Default is 7.

    Attributes:
    -----------
    vol_frac : float
        Suspension volume fraction.
    R_cell : float
        Cell radius, calculated based on the particle radius and volume fraction.
    c_tilde : float
        Reservoir concentration in number of particles per nm³.
    kresa : float
        Screening constant for the system.

    Methods:
    --------
    set_grid(r: np.array):
        Sets the radial grid for solving the Poisson-Boltzmann equation.
    reservoir_concent(c_Hres, c_salt):
        Determines the reservoir salt concentration for charge-regulated systems, 
        accounting for water autodissociation.
    solve_nonlin_PB(r, y_init):
        Solves the nonlinear Poisson-Boltzmann equation for the system.

    Description:
    ------------
    The `VolumeMicrogel` class models microgel suspensions with either strong or weak charges. 
    For strong charges, the reservoir concentration is directly calculated. For weak charges, 
    the class uses the `WeakParticle` parent class to account for charge regulation based on 
    pH and pK values. The class also provides methods to set up the radial grid, compute 
    reservoir concentrations, and solve the nonlinear Poisson-Boltzmann equation for the system.

    """

    def __init__(self, a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type, pK_a=4, pK_b=4, pH_res=7):
        Suspension.__init__(self, a, Z_b-Z_a, lb, c_salt, charge_type)

        self._vol_frac = vol_frac
        
        if self.charge_type=='strong':
            self._c_tilde = c_salt * self.conversion_factor # in number of particles per nm^3
            self._k0_aa = None
            self._k0_ab = None
        elif self.charge_type=='weak':
            WeakParticle.__init__(self, Z_a, Z_b, pK_a, pK_b, pH_res)
            self._c_tilde = reservoir_concent(self.c_Hres, self.c_salt, self.c_ref) * self.conversion_factor
            self._k0_aa = math.sqrt(3 * self.lb * self.Z_a / self.a)
            self._k0_ab = math.sqrt(3 * self.lb * self.Z_b / self.a)
            self.linear_solution = LinearVolumeMicrogelWeak(self.a, self.lb, self.vol_frac, c_salt, self.Z_a, self.Z_b, pK_a, pK_b, pH_res)

        self._R_cell = a / vol_frac**(1./3)
        self._kresa = self._screening_(self.c_tilde)


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
    
    def set_grid(self, r: np.array):
        """
        Set the radial grid for solving the Poisson-Boltzmann equation.

        Parameters:
        -----------
        r : np.array
            Radial grid for the solution of the Poisson-Boltzmann equation.

        Raises:
        -------
        AssertionError
            If the grid is not strictly increasing.
            If the first component of the grid is smaller than 0.
            If the last component of the grid is larger than the cell radius (`R_cell`).
        """
        
        self.r = r
        
        # Assert if all components are increasing
        assert np.all(np.diff(r) > 0), "Not all components are increasing"

        # Assert if all components are larger or equal than 0
        assert self.r[0] >= 0, "First component of the grid larger than 0"

        # Assert if all components are smaller or equal than the cell radius
        assert self.r[-1] <= self.R_cell, "Last component of the grid smaller than R_cell"

    
    def solve_nonlin_PB(self, r, y_init):
        """
        Solve the nonlinear Poisson-Boltzmann equation for the system.

        Parameters:
        -----------
        r : np.array
            Radial grid for the Poisson-Boltzmann solution, in the same units as the particle radius.
        y_init : np.array
            Initial guess for the solution of the differential equation.

        Returns:
        --------
        None
            The results are stored in the attributes `elec_pot` (electric potential), 
            `elec_field` (electric field), and `r` (radial grid in original units).

        Description:
        ------------
        This method solves the nonlinear Poisson-Boltzmann equation for the system. 
        The radial grid is normalized by the particle radius (`a`) before solving. 
        Depending on the `charge_type` of the system (`strong` or `weak`), the method 
        uses different parameter sets and equations:

        - For `strong` charges, the parameters include the screening constant (`kresa`) 
        and the linear charge density (`Zlba`). The equations and boundary conditions 
        are defined in the `PBstrong` module.
        - For `weak` charges, additional parameters such as the dissociation constants 
        (`Ka`, `Kb`) and coupling parameters (`xi_a`, `xi_b`) are used. The equations 
        and boundary conditions are defined in the `PBweak` module.

        The solution is obtained using the `solve_bvp` method from `scipy.integrate`. 
        The electric potential and electric field are extracted from the solution and 
        stored in the `elec_pot` and `elec_field` attributes, respectively. The radial 
        grid is converted back to its original units and stored in the `r` attribute.
        """

        # For solving the PB equation, distances should be given in units of self.a
        r = r / self.a
        if self.charge_type=='strong':
            params = np.array([self.kresa, self.Zlba])
            solution = solve_bvp(lambda r, y: PBstrong.odes(r, y, params), 
                        lambda ya, yb: PBstrong.boundary_conditions(ya, yb), r, y_init)
        elif self.charge_type=='weak':
            params = np.array([self.kresa, self.k0_aa, self.xi_a, self.k0_ab, self.xi_b])
            solution = solve_bvp(lambda r, y: PBweak.odes(r, y, params),
                        lambda ya, yb: PBweak.boundary_conditions(ya, yb), r, y_init)
        
        assert solution.p==None, "Parameters of the Poisson-Boltzmann equation have been fitted."
        assert solution.success==0, "The solution of the Poisson-Boltzmann equation was not successful."
        sol = solution.sol(r)
        self.elec_pot = sol[0]
        self.elec_field = sol[1]
        self.r = r * self.a

    def lin_elec_pot(self, r):
        return self.linear_solution.lin_elec_pot(r)

    def lin_elec_field(self, r):
        return self.linear_solution.lin_elec_field(r)