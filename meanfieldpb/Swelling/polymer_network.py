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
from abc import ABC


@dataclass
class PolymerNetwork(ABC):
    """
    Abstract base class for polymer networks.
    """

    a: float = 0.0  # Radius of the polymer network
    a0: float = 0.0  # Dry radius of the polymer network
    N_monomers: int = 0  # Number of monomers in the polymer chain
    N_chains: int = 0  # Number of chains in the polymer network
    N_crosslinks: int = 0  # Number of crosslinks in the polymer network
    chi_parameter: float = 0.0  # Interaction parameter
    # CHECK DEFINITIONS FOR N_m, Nch, Nmonch


class CrosslinkedPolymerNetwork(PolymerNetwork):
    """
    Represents a cross-linked polymer network.
    """

    def __init__(self, crosslink_density: float):
        self.crosslink_density = crosslink_density


class UncrosslinkedPolymerNetwork(PolymerNetwork):
    """
    Represents an uncross-linked polymer network.
    """

    def __init__(self, polymer_density: float):
        self.polymer_density = polymer_density


class FEPolymerNetwork(PolymerNetwork):
    """
    Represents a polymer network with chains with finite extensibility.
    """

    def __init__(self, element_size: float):
        self.element_size = element_size
