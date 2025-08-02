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
import numpy as np
from scipy.integrate import solve_bvp
from meanfieldpb.PBequations import PBequation_colloid_strong as PBstrong


class Colloid(Suspension):
    """
    Represents a hard sphere colloidal suspension with surface charge.

    This class models impermeable charged spherical particles suspended in an
    electrolyte solution. The particles have a fixed surface charge and are
    treated using the cell model where each particle occupies the center of
    a spherical Wigner-Seitz cell.

    The class solves the nonlinear Poisson-Boltzmann equation in spherical
    coordinates to determine the electrostatic potential and ion distributions
    around the charged colloids.

    Parameters
    ----------
    a : float
        Particle radius in nanometers [nm].
    Z : float
        Particle surface charge in units of elementary charge e.
    lb : float
        Bjerrum length in nanometers [nm].
    vol_frac : float
        Volume fraction of particles in suspension (0 < vol_frac < 1).
    c_salt : float
        Reservoir salt concentration in molars [M].
    charge_type : str
        Type of electrolyte - 'strong' or 'weak'.
    pKa : float, optional
        pK value of surface groups (only used if charge_type='weak'). Default: 4.
    pH_res : float, optional
        Reservoir pH (only used if charge_type='weak'). Default: 7.

    Attributes
    ----------
    vol_frac : float
        Volume fraction of particles.
    R_cell : float
        Wigner-Seitz cell radius in [nm].
    kresa : float
        Dimensionless screening parameter κ*a.
    c_tilde : float
        Salt concentration in particles per nm³.

    Notes
    -----
    The cell radius is determined by the volume fraction as:
    R_cell = a * (vol_frac)^(-1/3)

    The screening parameter is calculated as:
    κ = sqrt(8πlb*c_tilde)

    Examples
    --------
    Create a colloid suspension and solve for electrostatic potential:

    >>> import numpy as np
    >>> from src import colloid
    >>>
    >>> # System parameters
    >>> colloid_susp = colloid.Colloid(
    ...     a=50,           # 50 nm radius
    ...     Z=100,          # 100 elementary charges
    ...     lb=0.71,        # Bjerrum length in water
    ...     vol_frac=0.01,  # 1% volume fraction
    ...     c_salt=0.001,   # 1 mM salt
    ...     charge_type='strong'
    ... )
    >>>
    >>> # Set up radial grid and solve
    >>> r = np.linspace(colloid_susp.a, colloid_susp.R_cell, 500)
    >>> y_init = np.zeros((2, len(r)))
    >>> y_init[0] = np.linspace(2.0, 0.0, len(r))  # initial potential guess
    >>>
    >>> colloid_susp.solve_nonlin_PB(r, y_init)
    >>> print(f"Surface potential: {colloid_susp.elec_pot[0]:.3f}")
    """

    def __init__(self, a, Z, lb, vol_frac, c_salt, charge_type, pKa=4, pH_res=7):
        super().__init__(a, Z, lb, c_salt, charge_type)
        if self.charge_type == "strong":
            self._c_tilde = (
                c_salt * self.conversion_factor
            )  # in number of particles per nm^3

        self._vol_frac = vol_frac
        self._R_cell = a / vol_frac ** (1.0 / 3)
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

    def set_grid(self, r: np.array):
        """
        Set grid r for solution of PB equation.
        """

        self.r = r

        # Assert if all components are increasing
        assert np.all(np.diff(r) > 0), "Not all components are increasing"

        # Assert if all components are larger or equal than the colloid radius
        assert self.r[0] >= self.a, "First component of the grid larger than a"

        # Assert if all components are smaller or equal than the cell radius
        assert self.r[-1] <= self.R_cell, (
            "Last component of the grid smaller than R_coll"
        )

    def solve_nonlin_PB(self, r, y_init):
        """
        Solve the nonlinear Poisson-Boltzmann equation for the colloid system.

        This method solves the nonlinear Poisson-Boltzmann equation in spherical
        coordinates for a charged colloid surrounded by electrolyte solution.
        The equation is solved using a boundary value problem (BVP) solver.

        Parameters
        ----------
        r : numpy.ndarray
            Radial grid points from particle surface to cell boundary [nm].
            Must satisfy: a ≤ r ≤ R_cell.
        y_init : numpy.ndarray
            Initial guess for the solution with shape (2, len(r)).
            - y_init[0]: Initial guess for electric potential φ(r)
            - y_init[1]: Initial guess for electric field φ'(r)

        Returns
        -------
        None
            Results are stored in instance attributes:
            - self.r: radial grid [nm]
            - self.elec_pot: dimensionless electric potential φ(r)
            - self.elec_field: dimensionless electric field φ'(r)

        Raises
        ------
        AssertionError
            If the radial grid is not monotonically increasing or extends
            outside the valid domain [a, R_cell].
        ValueError
            If the charge type is not supported or parameters are invalid.

        Notes
        -----
        The Poisson-Boltzmann equation in spherical coordinates is:

        .. math::
            \\frac{1}{r^2}\\frac{d}{dr}\\left(r^2\\frac{d\\phi}{dr}\\right) = \\kappa^2 \\sinh(\\phi)

        With boundary conditions:
        - At r = a: φ'(a) = -Z*lb/a² (surface charge condition)
        - At r = R_cell: φ'(R_cell) = 0 (zero field at cell boundary)

        The dimensionless variables are:
        - φ = eψ/(kᵦT) (reduced potential)
        - r̃ = r/a (reduced radius)
        - κ̃ = κa (reduced screening parameter)

        Examples
        --------
        Solve for a typical colloid system:

        >>> import numpy as np
        >>> colloid_susp = Colloid(50, 100, 0.71, 0.01, 0.001, 'strong')
        >>>
        >>> # Create radial grid
        >>> r = np.linspace(colloid_susp.a, colloid_susp.R_cell, 500)
        >>>
        >>> # Initial guess: linear decay from surface to bulk
        >>> y_init = np.zeros((2, len(r)))
        >>> y_init[0] = np.linspace(2.0, 0.0, len(r))  # φ(r)
        >>> y_init[1] = np.zeros(len(r))               # φ'(r)
        >>>
        >>> # Solve PB equation
        >>> colloid_susp.solve_nonlin_PB(r, y_init)
        >>>
        >>> # Access results
        >>> surface_potential = colloid_susp.elec_pot[0]
        >>> print(f"Surface potential: {surface_potential:.3f}")
        """
        # For solving the PB equation, distances should be given in units of self.a
        r = r / self.a
        if self.charge_type == "strong":
            params = np.array([self.kresa, self.Zlba])
            solution = solve_bvp(
                lambda r, y: PBstrong.odes(r, y, params),
                lambda ya, yb: PBstrong.boundary_conditions(ya, yb, params),
                r,
                y_init,
            )

        assert solution.p is None, (
            "Parameters of the Poisson-Boltzmann equation have been fitted."
        )
        assert solution.success == 0, (
            "The solution of the Poisson-Boltzmann equation was not successful."
        )
        sol = solution.sol(r)
        self.elec_pot = sol[0]
        self.elec_field = sol[1]
        self.r = r * self.a

    def lin_elec_pot(self, r):
        return super().lin_elec_pot(r)

    def lin_elec_field(self, r):
        return super().lin_elec_field(r)
