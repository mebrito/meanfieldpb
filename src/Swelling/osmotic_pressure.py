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

from src.surface_microgel import SurfaceMicrogel
from src.suspension import Suspension
from src.volume_microgel import VolumeMicrogel
from polymer_network import PolymerNetwork, CrosslinkedPolymerNetwork, UncrosslinkedPolymerNetwork, FEPolymerNetwork

# Strategies for calculating osmotic pressure contributions -------------------
def volume_self_contribution(microgel_suspension: VolumeMicrogel):
    # Implement the calculation logic for volume microgels
    pass

def surface_self_contribution(microgel_suspension: SurfaceMicrogel):
    # Implement the calculation logic for surface microgels
    pass

# Strategy for calculating polymer pressure contributions ---------------------
def crosslinked_polymer_contribution(network: CrosslinkedPolymerNetwork):
    # Implement the calculation logic for crosslinked polymer networks
    pass

def uncrosslinked_polymer_contribution(network: UncrosslinkedPolymerNetwork):
    # Implement the calculation logic for uncrosslinked polymer networks
    pass

def fe_polymer_contribution(network: FEPolymerNetwork):
    # Implement the calculation logic for finite extensibility polymer networks
    pass

# Osmotic Pressure Calculator -------------------------------------------------
@dataclass
class OsmoticPressureCalculator():
    """
    """
    microgel_suspension: Suspension
    network: PolymerNetwork

    def __post_init__(self):
        if not isinstance(self.microgel_suspension, (VolumeMicrogel, SurfaceMicrogel)):
            raise TypeError("microgel_suspension must be an instance of VolumeMicrogel or SurfaceMicrogel")
        
    def calculate_polymer_contribution(self):
        """
        """
        if isinstance(self.network, CrosslinkedPolymerNetwork):
            return crosslinked_polymer_contribution(self.network)
        elif isinstance(self.network, UncrosslinkedPolymerNetwork):
            return uncrosslinked_polymer_contribution(self.network)
        elif isinstance(self.network, FEPolymerNetwork):
            return fe_polymer_contribution(self.network)
        else:
            raise TypeError("network must be an instance of CrosslinkedPolymerNetwork, UncrosslinkedPolymerNetwork, or FEPolymerNetwork")

    def calculate_self_contribution(self):
        """
        """
        if isinstance(self.microgel_suspension, VolumeMicrogel):
            return volume_self_contribution(self.microgel_suspension)
        elif isinstance(self.microgel_suspension, SurfaceMicrogel):
            return surface_self_contribution(self.microgel_suspension)

    def calculate_microion_contribution(self):
        """
        """
        pass

    def total_osmotic_pressure(self):
        """
        """
        polymer_contribution = self.calculate_polymer_contribution()
        self_contribution = self.calculate_self_contribution()
        microion_contribution = self.calculate_microion_contribution()
        
        return polymer_contribution + self_contribution + microion_contribution