"""
MeanFieldPB: A Python package for solving Poisson-Boltzmann equations in colloidal suspensions.

This package provides tools for modeling electrostatics of
suspensions of colloids, surface microgels, volume microgels, and
linear polyelectrolytes using mean-field Poisson-Boltzmann theory.
"""

# Import particle-specific classes
try:
    from .colloid import Colloid
except ImportError:
    pass

try:
    from .surface_microgel import SurfaceMicrogel
except ImportError:
    pass

try:
    from .volume_microgel import VolumeMicrogel
except ImportError:
    pass

try:
    from .linear_polyelec import LinearPolyelectrolyte
except ImportError:
    pass

__version__ = "0.1.0"
__author__ = "Mariano E. Brito"
__email__ = "mbrito@icp.uni-stuttgart.de"

__all__ = [
    "Suspension",
    "Colloid",
    "SurfaceMicrogel",
    "VolumeMicrogel",
    "LinearPolyelectrolyte",
    "WeakParticle",
    "reservoir_concent",
]
