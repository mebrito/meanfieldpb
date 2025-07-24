# MeanFieldPB

A Python package for modeling the electrostatics of charge-equilibrated colloidal and polymer suspensions using mean-field Poisson-Boltzmann equations.

## Overview

MeanFieldPB provides computational tools for modeling the electrostatics in various particle systems including colloids, microgels (with volume- or surface backbone charge distribution), and linear polyelectrolytes. The package implements mean-field Poisson-Boltzmann equations to calculate electric potential, electric field, ion densities, and charge distributions in suspension systems within cell model approximation.

Within the Poisson-Boltzmann cell model (PBCM), the suspension of interest is modeled by a macroion (colloid, microgel, polyelectrolyte) coexisting with point-like microions (released counterions and salt ions) in a structureless solvent, which is only characterized by electric permittivity. The shape and size of the cell are determined by the macroion symmetries and the macroion concentration of the suspension, respectively. Microion density distributions are assumed to follow Boltzmann distribution. Interactions between microions are neglected, thus giving the mean-field character of the description. The electric potential and the electric field within the cell are determined by solving the Poisson equation. Exploiting cell symmetries, the Poisson equation is reduced to a stiff ordinary differential equation with Neumann boundary conditions.

## Features

- **Multiple Particle Types**: Support for different particle geometries and charge distributions
  - Colloids: Classical spherical charged particles
  - Surface Microgels: Ion-permeable colloids with surface-localized backbone charges
  - Volume Microgels: Ion-permeable colloids with charges distributed throughout their backbone volume
  - Linear Polyelectrolytes: Chain-like charged polymers

- **Electrostatic Modeling**: Solve nonlinear Poisson-Boltzmann equation within cell-model approximation

- **Charge Types**: Handle macroions -colloids and polyelectrolytes- with both strong and weak charges with different ionization behaviors

- **Physical Properties**: Calculate comprehensive electrostatic properties
  - Electric potential profiles
  - Electric field distributions
  - Ion density distributions
  - Particle charge distributions

- **Cell Model Approach**: Exploits symmetries of the macroions, implementing cell model approximation. Here, the cell shape is determined by the macroion symmetries and its size by the suspension concentration

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- SciPy 
- Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package

```bash
pip install -e .
```

## Quick Start

### Basic Colloid Suspension

```python
import numpy as np
from src import Colloid

# Define system parameters
a = 50          # particle radius [nm]
Z = 100         # particle charge [e]
lb = 0.71       # Bjerrum length [nm]
vol_frac = 0.001 # volume fraction
c_salt = 0.0001  # salt concentration [M]
charge_type = 'strong'

# Create colloid suspension
colloid_system = Colloid(a, Z, lb, vol_frac, c_salt, charge_type)

# Calculate electrostatic potential
r = np.linspace(1e-5, colloid_system.R_cell, 1000)
potential = colloid_system.elec_pot(r)
```

### Volume Microgel Suspension

```python
from src import VolumeMicrogel

# Microgel parameters
a0 = 108        # reference radius [nm]
alpha = 1.84    # swelling factor
Z_a = 1392      # backbone charge
Z_b = 0         # additional charge
lb = 0.71       # Bjerrum length [nm]
vol_frac = 0.02 # volume fraction
c_salt = 0.001  # salt concentration [M]

# Create microgel suspension
microgel_system = VolumeMicrogel(a0, alpha, Z_a, Z_b, lb, vol_frac, c_salt, 'strong')

# Calculate ion density profiles
r = np.linspace(1e-5, microgel_system.R_cell, 1000)
n_plus = microgel_system.n_plus(r)
n_minus = microgel_system.n_minus(r)
```

## Available Particle Types

### Colloid
Classical spherical particles with surface charges. Suitable for modeling hard spheres, silica particles, and other colloidal systems.

### SurfaceMicrogel
Particles with charges localized on the surface. Models microgels where ionizable groups are primarily at the particle interface.

### VolumeMicrogel
Particles with charges distributed throughout their volume. Accounts for the permeable nature of microgel particles with internal charge distributions.

### LinearPolyelectrolyte
Chain-like charged polymers. Models polyelectrolyte solutions and polymer-colloid mixtures.

## Physical Parameters

- **a**: Particle radius [nm]
- **Z**: Particle charge [e]
- **lb**: Bjerrum length [nm] - distance where thermal and Coulomb energies are equal
- **vol_frac**: Volume fraction of particles in suspension
- **c_salt**: Salt concentration [M]
- **charge_type**: 'strong' or 'weak' electrolyte behavior

## Key Methods

All particle classes inherit from the base `Suspension` class and provide:

- `elec_pot`: Calculate electrostatic potential at distance r
- `elec_field`: Calculate electric field at distance r
- `cation_density()`: Calculate positive ion density at distance r
- `anion_density()`: Calculate negative ion density at distance r

## Examples

The `samples/` directory contains example scripts demonstrating:

- Volume microgel electrostatics with comparison to reference data
- Basic usage patterns for different particle types
- Visualization of potential and density profiles

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/volume_microgel_test.py
```

Or use the provided test runner:

```bash
cd tests/
./run_tests.sh
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MeanFieldPB in your research, please cite:

```
Brito et al., J. Chem. Phys. 151, 224901 (2019)
https://doi.org/10.1063/1.5129575
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## Author

**Mariano E. Brito**  
Email: mbrito@icp.uni-stuttgart.de

## Acknowledgments

This package implements mean-field Poisson-Boltzmann theory for colloidal suspensions, building on established theoretical frameworks in colloid science and statistical mechanics.
