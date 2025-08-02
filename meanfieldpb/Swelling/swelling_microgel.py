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
from abc import ABC, abstractmethod

from meanfieldpb.surface_microgel import SurfaceMicrogel
from meanfieldpb.suspension import Suspension
from meanfieldpb.volume_microgel import VolumeMicrogel
from meanfieldpb.Swelling.osmotic_pressure import OsmoticPressureCalculator
from meanfieldpb.Swelling.polymer_network import PolymerNetwork


# Factories and factory abstraction for Suspension ----------------------------
class SuspensionFactory(ABC):
    """
    Factory abstraction for creating microgel suspensions.
    """

    @abstractmethod
    def get_microgel_suspension(self, microgel_type: str, **kwargs) -> Suspension:
        pass


class VolumeMicrogelFactory(SuspensionFactory):
    """ """

    def get_microgel_suspension(self, **kwargs) -> VolumeMicrogel:
        from src.volume_microgel import VolumeMicrogel

        return VolumeMicrogel(**kwargs)


class SurfaceMicrogelFactory(SuspensionFactory):
    """ """

    def get_microgel_suspension(self, **kwargs) -> SurfaceMicrogel:
        from src.surface_microgel import SurfaceMicrogel

        return SurfaceMicrogel(**kwargs)


# Factories and factory abstraction for PolymerNetwork ------------------------
class NetworkFactory(ABC):
    """
    Factory abstraction for creating polymer networks.
    """

    @abstractmethod
    def get_polymer_network(self, **kwargs) -> PolymerNetwork:
        pass


class CrosslinkedNetworkFactory(NetworkFactory):
    """ """

    def get_polymer_network(self, **kwargs):
        from polymer_network import CrosslinkedPolymerNetwork

        return CrosslinkedPolymerNetwork(**kwargs)


class UncrosslinkedNetworkFactory(NetworkFactory):
    """ """

    def get_polymer_network(self, **kwargs):
        from polymer_network import UncrosslinkedPolymerNetwork

        return UncrosslinkedPolymerNetwork(**kwargs)


class FiniteExtensibilityNetworkFactory(NetworkFactory):
    """ """

    def get_polymer_network(self, **kwargs):
        from polymer_network import FEPolymerNetwork

        return FEPolymerNetwork(**kwargs)


# Swelling Microgel Suspension ------------------------------------------------
@dataclass
class SwellingMicrogelSuspension:
    """ """

    microgel_type: str  # Type of microgel (e.g., "volume_microgel", "surface_microgel")
    network_type: str  # Type of polymer network (e.g., "crosslinked", "uncrosslinked", "finite_extensibility"
    Z_a: float
    Z_b: float
    lb: float
    vol_frac: float
    c_salt: float
    charge_type: str
    pK_a: float
    pK_b: float
    pH_res: float
    N_monomers: int
    N_crosslinks: int
    N_chains: int
    chi_parameter: float

    def read_suspension(self) -> SuspensionFactory:
        """ """
        factories = {
            "volume_microgel": VolumeMicrogelFactory(),
            "surface_microgel": SurfaceMicrogelFactory(),
        }

        return factories.get(self.microgel_type, None)

    def read_network(self) -> NetworkFactory:
        """ """
        factories = {
            "crosslinked": CrosslinkedNetworkFactory(),
            "uncrosslinked": UncrosslinkedNetworkFactory(),
            "finite_extensibility": FiniteExtensibilityNetworkFactory(),
        }

        return factories.get(self.network_type, None)

    def set_a_values(self, a_values: list[float]) -> None:
        """ """
        self.a_values = a_values

        if not hasattr(self, "a_values"):
            raise ValueError("a_values must be set before computing equilibrium size.")
        if not isinstance(self.a_values, list):
            raise TypeError("a_values must be a list of floats.")
        if not all(isinstance(a, (int, float)) for a in self.a_values):
            raise TypeError("All elements in a_values must be numeric (int or float).")
        if not all(a > 0 for a in self.a_values):
            raise ValueError("All elements in a_values must be greater than 0.")

    def compute_equilibrium_size(self) -> float:
        """ """
        susp = self.read_suspension()
        if susp is None:
            raise ValueError(f"Unknown microgel type: {self.microgel_type}")
        net = self.read_network()
        if net is None:
            raise ValueError(f"Unknown network type: {self.network_type}")

        for a_trial in self.a_values:
            microgel_suspension = susp.get_microgel_suspension(
                a=a_trial,
                Z_a=self.Z_a,
                Z_b=self.Z_b,
                lb=self.lb,
                vol_frac=self.vol_frac,
                c_salt=self.c_salt,
                charge_type=self.charge_type,
                pK_a=self.pK_a,
                pK_b=self.pK_b,
                pH_res=self.pH_res,
            )
            polymer_network = net.get_polymer_network(
                N_monomers=self.N_monomers,
                N_crosslinks=self.N_crosslinks,
                N_chains=self.N_chains,
                chi_parameter=self.chi_parameter,
            )

            op_calculator = OsmoticPressureCalculator(
                microgel_suspension, polymer_network
            )
            op_calculator.total_osmotic_pressure()

            # Here you would typically solve the PB equation or perform other calculations
