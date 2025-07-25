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
from scipy.integrate import simpson
import math
from abc import ABC, abstractmethod

class Suspension(ABC):
    """
    Base class for modeling electrostatic interactions in colloidal suspensions.
    
    This class provides the fundamental framework for solving Poisson-Boltzmann equations
    in various particle suspension systems. It implements the cell model approach where
    particles are arranged in a periodic structure, allowing calculation of electrostatic
    potential, electric field, and ion density distributions.
    
    The class handles both strong and weak electrolytes and provides methods for
    calculating key physical properties such as screening parameters, ion densities,
    and total particle numbers.
    
    Parameters
    ----------
    a : float
        Particle radius in nanometers [nm].
    Z : float  
        Particle charge in units of the fundamental charge e.
    lb : float
        Bjerrum length in nanometers [nm]. The distance at which thermal energy
        equals Coulomb interaction energy between two elementary charges.
    c_salt : float
        Reservoir salt concentration in molars [M].
    charge_type : str
        Type of particle charge - 'strong' or 'weak' electrolyte.
        
    Attributes
    ----------
    a : float
        Particle radius [nm].
    Z : float
        Particle charge [e].
    lb : float
        Bjerrum length [nm].
    c_salt : float
        Salt concentration [M].
    charge_type : str
        Electrolyte type.
    Zlba : float
        Dimensionless charge parameter Z*lb/a.
    conversion_factor : float
        Unit conversion factor from molars to particles per nm³.
    r : numpy.ndarray or None
        Radial grid points [nm].
    elec_pot : numpy.ndarray or None
        Dimensionless electric potential φ(r).
    elec_field : numpy.ndarray or None
        Dimensionless electric field φ'(r).
        
    Notes
    -----
    The Bjerrum length is defined as lb = e²/(4πε₀εᵣkᵦT), where:
    - e is the elementary charge
    - ε₀ is the vacuum permittivity
    - εᵣ is the relative permittivity of water (~80)
    - kᵦ is Boltzmann constant
    - T is temperature
    
    At room temperature in water, lb ≈ 0.71 nm.
    
    Examples
    --------
    This is an abstract base class. Use specific implementations like:
    
    >>> from src import colloid
    >>> suspension = colloid.Colloid(50, 100, 0.71, 0.01, 0.001, 'strong')
    >>> print(f"Charge parameter: {suspension.Zlba:.2f}")
    """

    def __init__(self, a, Z, lb, c_salt, charge_type):
        """
        Base class for Suspension systems.

        Parameters:
        a (float): particle radius in [nm].
        Z (float): particle (maximum) charged in units of the fundamental charge e.
        lb (float): Bjerrum length in [nm].
        c_salt (float): reservoir salt concentration in molars [M].
        charge_type (str): type of particle charge - 'strong' or 'weak'.
        
        Attributes:
        R_cell (float): cell radius in [nm].
        r (np.array): radial grid.
        elec_pot (np.array): Electric potential in the cell.
        elec_field (np.array): Electric field in the cell.
        """
        self._a = a
        self._Z = Z
        self._lb = lb
        self._c_salt = c_salt
        self._charge_type = charge_type
        
        self._Zlba = self.Z * self.lb / self.a

        self.r = None  # Initialize radial grid r to None
        self.elec_pot = None  # Initialize electric potential elec_pot to None
        self.elec_field = None  # Initialize electric field elec_field to None

        self._conversion_factor = 6.02214/10 # If the quatity is in molar, multiplication by this factor gives density in number of particles per nm^3


    @property
    def a(self):
        return self._a
    

    @property
    def Z(self):
        return self._Z
    

    @property
    def lb(self):
        return self._lb
    

    @property
    def c_salt(self):
        return self._c_salt
    

    @property
    def charge_type(self):
        return self._charge_type
    

    @property
    def Zlba(self):
        return self._Zlba
    

    @property
    def conversion_factor(self):
        return self._conversion_factor
    

    def get_r(self):
        """
        Retrieve the radial grid

        Returns:
        --------
        np.array
            The radial grid in the cell.
        """
        if self.r is not None:
            return self.pot
        else:
            raise ValueError("Grid r not set.")    


    def get_elec_pot(self):
        """
        Retrieve the electric potential of the system.

        Returns:
        --------
        np.array
            The electric potential in the cell.
        """
        if self.elec_pot is not None:
            return self.elec_pot
        else:
            raise ValueError("Undefined electric potential.")


    def get_elec_field(self):
        """
        Retrieve the electric field of the system.

        Returns:
        --------
        np.array
            The electric field in the cell.
        """
        if self.elec_field is not None:
            return self.elec_field
        else:
            raise ValueError("Undefined electric field.")


    def _screening_(self, c_tilde):
        """
        Calculate the screening constant for the system.

        Parameters:
        -----------
        c_tilde : float
            Reservoir concentration in number of particles per nm^3.

        Returns:
        --------
        float
            The screening constant, calculated using the Bjerrum length (`lb`),
            reservoir concentration (`c_tilde`), and particle radius (`a`).

        Description:
        ------------
        The screening constant is a measure of the electrostatic screening effect
        in the system. It is computed using the formula:
            sqrt(8 * π * lb * c_tilde) * a
        where `lb` is the Bjerrum length, `c_tilde` is the reservoir concentration,
        and `a` is the particle radius.
        """
        return math.sqrt(8 * math.pi * self.lb * c_tilde) * self.a
    

    def cation_density(self):
        """ 
        Calculate the cation density profile in the system.

        Returns:
        --------
        np.array
            The cation density profile in molars [M].

        Raises:
        -------
        ValueError
            If the electric potential (`elec_pot`) is not defined (i.e., None).

        Description:
        ------------
        This method computes the cation density profile based on the electric potential
        (`elec_pot`) and the reservoir salt concentration (`c_salt`). The cation density
        is calculated using the formula:
            c_salt * exp(-elec_pot)
        where `c_salt` is the reservoir salt concentration and `elec_pot` is the electric
        potential. If the electric potential is not defined, a `ValueError` is raised,
        indicating that the Poisson-Boltzmann equation needs to be solved to obtain it.
        """

        if self.elec_pot is not None:
            return self.c_salt * np.exp(- self.elec_pot)
        else:
            raise ValueError("Undefined electric potential. Obtain it by solving Poisson-Boltzmann equation.")
        

    def anion_density(self):
        """ 
        Calculate the anion density profile in the system.

        Returns:
        --------
        np.array
            The anion density profile in molars [M].

        Raises:
        -------
        ValueError
            If the electric potential (`elec_pot`) is not defined (i.e., None).

        Description:
        ------------
        This method computes the anion density profile based on the electric potential
        (`elec_pot`) and the reservoir salt concentration (`c_salt`). The anion density
        is calculated using the formula:
            c_salt * exp(elec_pot)
        where `c_salt` is the reservoir salt concentration and `elec_pot` is the electric
        potential. If the electric potential is not defined, a `ValueError` is raised,
        indicating that the Poisson-Boltzmann equation needs to be solved to obtain it.
        """

        if self.elec_pot is not None:
            return self.c_salt * np.exp(self.elec_pot)
        else:
            raise ValueError("Undefined electric potential. Obtain it by solving Poisson-Boltzmann equation.")

    
    def _N_syst(self, species_density, geometry_differentrial):
        """
        Calculate the total number of particles in the system over the full range of `r`.
        Parameters:
        ----------
        species_density : np.array
            The density profile of the species of interest as a function of `r`.
        geometry_differentrial : callable
            A function representing the differential volume element for the system's geometry.
            For example:
            - Cylindrical geometry: f(x) = x
            - Spherical geometry: f(x) = 4 * pi * x^2
        Returns:
        -------
        float
            The total number of particles in the system, calculated by integrating the product
            of the species density, the geometry differential, and a conversion factor over `r`.
        """

        N_system = simpson(self.conversion_factor * species_density * geometry_differentrial(self.r), self.r)

        return N_system
    
    @abstractmethod
    def lin_elec_pot(self, r):
        """
        Abstract method to compute the linearized electric potential.
        
        Parameters:
        -----------
        r : np.array
            Radial grid for the solution of the Poisson-Boltzmann equation.
        
        Returns:
        --------
        np.array
            Linearized electric potential.
        
        Description:
        ------------
        This method must be implemented in subclasses to compute the linearized electric potential
        based on the radial grid `r`.
        """
        pass

    @abstractmethod
    def lin_elec_field(self, r):
        """
        Abstract method to compute the linearized electric field.

        Parameters:
        -----------
        r : np.array
            Radial grid for the solution of the Poisson-Boltzmann equation.

        Returns:
        --------
        np.array
            Linearized electric field.

        Description:
        ------------
        This method must be implemented in subclasses to compute the linearized electric field
        based on the radial grid `r`.
        """
        pass
