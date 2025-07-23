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
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
import math
from suspension import Suspension
from PBequations import PBequation_linearPolyelectrolyte_strong as PBlinPoly

def solve_nonlin_PB_limPoly(r: np.array, y_init: np.array, kappa: float, xi: float) -> 'scipy.integrate.OdeSolution':
    """
    Solve the nonlinear Poisson-Boltzmann equation for a linear polyelectrolyte.

    Parameters:
    -----------
    r : np.array
        Radial grid for the solution of the Poisson-Boltzmann equation.
    y_init : np.array
        Initial guess for the solution of the differential equation.
    kappa : float
        Screening parameter, related to the ionic strength of the solution.
    xi : float
        Coupling parameter, related to the charge density of the polyelectrolyte.

    Returns:
    --------
    scipy.integrate.OdeSolution
        The solution of the boundary value problem (BVP) for the nonlinear 
        Poisson-Boltzmann equation.

    Description:
    ------------
    This function uses the `solve_bvp` method from `scipy.integrate` to solve 
    the nonlinear Poisson-Boltzmann equation. The differential equations and 
    boundary conditions are defined in the `PBlinPoly` module. The solution 
    provides the electric potential and its derivative (electric field) as 
    functions of the radial distance.
    """
    return solve_bvp(lambda r, y: PBlinPoly.odes(r, y, kappa), 
                        lambda ya, yb: PBlinPoly.boundary_conditions(ya, yb, xi), r, y_init)
        
        
@dataclass
class LinearPolyelectrolyte(Suspension):
    """
    LinearPolyelectrolyte
    A class representing a linear polyelectrolyte suspension system. This class
    inherits from the Suspension class and provides methods to calculate various
    properties of the system, such as the number of ions, condensed particles, 
    and system salt concentration, as well as solving the nonlinear Poisson-Boltzmann 
    equation.
    Attributes:
        a (int): Polyelectrolyte radius in nanometers [nm].
        Z (int): Linear charge density in units of [1/a].
        lb (int): Bjerrum length in units of [a].
        surf_frac (int): Volume fraction of the suspension.
        c_salt (int): Reservoir salt concentration in molars [M].

        _surf_frac (float): Internal representation of the surface fraction.
        _R_cell (float): Cell radius derived from the surface fraction.
        c_tilde (float): Salt concentration in number of particles per nm^3.
        kappaa (float): Screening parameter.
        xi (float): Dimensionless charge parameter.
        _system_volume (float): Volume of the system.
        r (np.array): Radial grid for solving the Poisson-Boltzmann equation.
        elec_pot (np.array): Electric potential solution from the PB equation.
        elec_field (np.array): Electric field solution from the PB equation.

    Methods:
        surf_frac:
            Property to get the surface fraction of the suspension.
        R_cell:
            Property to get the cell radius.
        system_volume:
            Property to get the system volume.
        set_grid(r: np.array):
            Sets the radial grid for solving the Poisson-Boltzmann equation.
            Ensures the grid is valid and within the system's domain.
        solve_nonlin_PB(r: np.array, y_init: np.array):
            Solves the nonlinear Poisson-Boltzmann equation for the system.
            Requires a radial grid and an initial guess for the solution.
        _geometry_differentrial(x):
            Computes the differential geometry factor for a given radius or array of radii.
        N_cations():
            Calculates the total number of cations in the system.
        N_anions():
            Calculates the total number of anions in the system.
        _N_condensed(R_M: float, species_density: np.array):
            Calculates the number of condensed particles of a given species within the Manning radius.
        N_condensed_cations(R_M: float):
            Calculates the number of condensed cations within the Manning radius.
        N_condensed_anions(R_M: float):
            Calculates the number of condensed anions within the Manning radius.
        system_salt_conc():
            Calculates the system salt concentration in molars [M].
        total_charge(r: float):
            Calculates the total charge per unit length of the system.    
    """

    a: int # polyelectrolyte radius
    Z: int # linear charge density in units of [1/a]
    lb: int # Bjerrum length in units of [a]
    surf_frac: int # volume fraction
    c_salt: int # reservoir salt concentration in [M]
    
    def __init__(self, a, Z, lb, surf_frac, c_salt):
        Suspension.__init__(self, a, Z, lb, c_salt, 'strong')
        self._surf_frac = surf_frac
        self._R_cell = a / surf_frac**(1./2)
        self.c_tilde = c_salt * self.conversion_factor # in number of particles per nm^3
        self.kappaa = self._screening_(self.c_tilde)
        self.xi = self.Z * self.a
        self._system_volume = math.pi * (self.R_cell**2 - self.a**2)


    @property
    def surf_frac(self):
        return self._surf_frac
    

    @property
    def R_cell(self):
        return self._R_cell
    

    @property
    def system_volume(self):
        return self._system_volume
    

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
            If the first component of the grid is smaller than the polyelectrolyte radius (`a`).
            If the last component of the grid is larger than the cell radius (`R_cell`).
        """

        self.r = r

        # Assert if all components are increasing
        assert np.all(np.diff(r) > 0), "Not all components are increasing"

        # Assert if all components are larger or equal than 0
        assert self.r[0] >= self.a, "First component of the grid larger or equal to a"

        # Assert if all components are smaller or equal than the cell radius
        assert self.r[-1] <= self.R_cell, "Last component of the grid smaller than R_coll"


    def solve_nonlin_PB(self, r, y_init):
        """
        Solve the nonlinear Poisson-Boltzmann equation for the system.

        Parameters:
        -----------
        r : np.array
            Radial grid for the Poisson-Boltzmann solution, in the same units as the polymer radius.
        y_init : np.array
            Initial guess for the solution of the differential equation.

        Raises:
        -------
        ValueError
            If the screening parameter (`kappaa`) or the coupling parameter (`xi`) is not set.

        Description:
        ------------
        This method solves the nonlinear Poisson-Boltzmann equation for the system. 
        It first sets the radial grid using the `set_grid` method and normalizes the 
        grid by the polyelectrolyte radius (`a`). The `solve_nonlin_PB_limPoly` function 
        is then used to solve the equation, providing the electric potential and electric 
        field as solutions. The results are stored in the `elec_pot` and `elec_field` 
        attributes of the class.

        Notes:
        ------
        - The radial distances must be provided in units of the polyelectrolyte radius (`a`).
        - The screening parameter (`kappaa`) and coupling parameter (`xi`) must be set 
        before calling this method.

        Example:
        --------
        >>> r = np.linspace(1, 10, 100)
        >>> y_init = np.zeros((2, 100))
        >>> polyelectrolyte.solve_nonlin_PB(r, y_init)
        """

        # For solving the PB equation, distances should be given in units of self.a
        self.set_grid(r)
        r_ = r / self.a
        if self.kappaa!=None and self.xi!=None:
            solution = solve_nonlin_PB_limPoly(r_, y_init, self.kappaa, self.xi)
            sol = solution.sol(r_)
            self.elec_pot = sol[0]
            self.elec_field = sol[1]
        else:
            print('Error: kappaa and xi are equal to None. Set them.') 


    def _geometry_differentrial(self, x):
        """
        Compute the differential geometry factor for a given radius or array of radii.

        Parameters:
        -----------
        x : float or np.ndarray
            Radius or array of radii for which the geometry factor is computed.

        Returns:
        --------
        float or np.ndarray
            The geometry factor, calculated as `2 * π * x` for each radius.
        """
        if isinstance(x, np.ndarray):
            return np.array([2 * math.pi * x_val for x_val in x])
        else:
            return 2 * math.pi * x
        
    
    def N_cations(self):
        """
        Calculate the total number of cations in the system over the full range of radial distances.

        Returns:
        --------
        float
            The total number of cations in the system.
        """

        return self._N_syst(self.cation_density(), self._geometry_differentrial)


    def N_anions(self):
        """
        Calculate the total number of anions in the system over the full range of radial distances.

        Returns:
        --------
        float
            The total number of anios in the system.
        """

        return self._N_syst(self.anion_density(), self._geometry_differentrial)
    

    def _N_condensed(self, R_M, species_density):
        """
        Calculate the number of condensed particles of a given species within the Manning radius.

        Parameters:
        -----------
        R_M : float
            Manning radius in nanometers [nm].
        species_density : np.array
            Density profile of the species of interest.

        Returns:
        --------
        float
            The number of condensed particles of the given species.

        Raises:
        -------
        ValueError
            If the Manning radius (`R_M`) is outside the radial grid domain.

        Description:
        ------------
        This method computes the number of condensed particles of a given species 
        within the Manning radius (`R_M`). It integrates the species density profile 
        over the radial grid up to `R_M`, taking into account the differential geometry 
        factor and a conversion factor to ensure the result is in the correct units.
        """

        if self.r[0] <= R_M <= self.r[-1]:
            r_ = self.r[self.r<=R_M]
            dens_ = species_density[self.r<=R_M]
            N_condensed = simpson(self.conversion_factor * dens_ * self._geometry_differentrial(r_), r_)
            return N_condensed
        else:
            raise ValueError("Invalid R_M. It must fall within the system domain.")


    def N_condensed_cations(self, R_M):
        """
        Calculate the number of condensed cations within the Manning radius.

        Parameters:
        -----------
        R_M : float
            Manning radius in nanometers [nm].

        Returns:
        --------
        float
            The number of condensed cations.
        """

        return self._N_condensed(R_M, self.cation_density())
    

    def N_condensed_anions(self, R_M):
        """
        Calculate the number of condensed anions within the Manning radius.

        Parameters:
        -----------
        R_M : float
            Manning radius in nanometers [nm].

        Returns:
        --------
        float
            The number of condensed anions.
        """

        return self._N_condensed(R_M, self.anion_density())
    

    def system_salt_conc(self):
        """
        Calculate the system salt concentration in molars [M].

        Returns:
        --------
        float
            The system salt concentration in molars [M].

        Description:
        ------------
        This method computes the system salt concentration by dividing the total 
        number of anions (`N_anions`) by the system volume. The result is then 
        converted to molars using the `conversion_factor` attribute.
        """

        salt_conc = self.N_anions() / self.system_volume
        return salt_conc / self.conversion_factor
    

    def total_charge(self, r):
        """
        Calculate the total charge per unit length of the system.

        Parameters:
        -----------
        r : float
            Radial distance or radius up to which the total charge is calculated.

        Returns:
        --------
        float
            The total charge per unit length, normalized by the linear charge density (`Z`).

        Description:
        ------------
        This method computes the total charge per unit length of the system by considering 
        the contributions from the condensed cations and anions within the given radius (`r`). 
        The calculation is normalized by the linear charge density (`Z`) of the polyelectrolyte.
        """

        return (-self.Z + self._N_condensed(r, self.cation_density())
                - self._N_condensed(r, self.anion_density())) / self.Z