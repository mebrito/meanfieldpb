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
- NumPy (≥1.20.0)
- SciPy (≥1.7.0)
- Matplotlib (≥3.5.0)

### Install from PyPI (when available)

```bash
pip install meanfieldpb
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/mebrito/meanfieldpb.git
cd meanfieldpb

# Install in development mode
pip install -e .
```

### Install with Optional Dependencies

```bash
# Install with development tools
pip install -e .[dev]

# Install with documentation tools  
pip install -e .[docs]

# Install with testing tools
pip install -e .[test]
```

## Quick Start

### Basic Colloid Suspension

```python
import numpy as np
from meanfieldpb import Colloid

# Define system parameters
a = 50              # particle radius [nm]
Z = 100             # particle charge [e]
lb = 0.71           # Bjerrum length [nm] (water at room temperature)
vol_frac = 0.001    # volume fraction
c_salt = 0.0001     # salt concentration [M]
charge_type = 'strong'

# Create colloid suspension
colloid_system = Colloid(a, Z, lb, vol_frac, c_salt, charge_type)

# Set up spatial grid from particle surface to cell boundary
r = np.linspace(a, colloid_system.R_cell, 1000)

# Calculate electrostatic potential
y_init = np.zeros((2, r.size))
colloid_system.solve_nonlin_PB(r, y_init)
potential = colloid_system.elec_pot
```

### Volume Microgel Suspension

```python
import matplotlib.pyplot as plt
import numpy as np
from meanfieldpb import VolumeMicrogel

# Microgel parameters
N_nodes = 5000      # number of grid points
a0 = 10             # reference radius [nm]
alpha = 2.612       # swelling factor
a = alpha * a0      # swollen radius [nm]
Z_a = 500           # backbone charge
Z_b = 0             # additional charge
lb = 0.71           # Bjerrum length [nm] (water at room temperature)
vol_frac = 0.005 * alpha**3  # volume fraction
c_salt = 1e-4       # salt concentration [M]
charge_type = 'strong'

# Create microgel suspension
microgel_system = VolumeMicrogel(a, Z_a, Z_b, lb, vol_frac, c_salt, charge_type)

# Set up spatial grid
r = np.linspace(1e-8, microgel_system.R_cell, N_nodes)
y_init = np.zeros((2, r.size))

# Solve the nonlinear Poisson-Boltzmann equation
microgel_system.solve_nonlin_PB(r, y_init)

# Calculate electrostatic properties
potential = microgel_system.elec_pot
electric_field = microgel_system.elec_field
cation_density = microgel_system.cation_density()
anion_density = microgel_system.anion_density()

# Plot electrostatic potential and field
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(microgel_system.r/a0, potential, label='Electrostatic potential φ(r)', lw=2)
plt.plot(microgel_system.r/a0, electric_field, label="Electric field φ'(r)", lw=2)
plt.axvline(x=alpha, color='gray', linestyle='--', label='Microgel radius a/a₀', alpha=0.5)
plt.xlabel('r/a₀')
plt.ylabel('φ(r), φ\'(r)')
plt.legend()
plt.title('Electrostatic Properties')

# Plot ion density difference
plt.subplot(1, 2, 2)
n_diff = (cation_density - anion_density) * microgel_system.conversion_factor
plt.plot(microgel_system.r/a0, n_diff*a0**3, label='Ion density difference', lw=2)
plt.axvline(x=alpha, color='gray', linestyle='--', label='Microgel radius a/a₀', alpha=0.5)
plt.xlabel('r/a₀')
plt.ylabel('[n₊(r) - n₋(r)]a₀³')
plt.legend()
plt.title('Ion Density Profile')

plt.tight_layout()
plt.show()
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

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=meanfieldpb

# Run specific test file
pytest tests/volume_microgel_test.py
```

Or use the traditional unittest approach:

```bash
# Run all tests
python -m unittest discover -s tests -p "*.py"

# Run specific test
python -m unittest tests.volume_microgel_test
```

Or use the provided test runner:

```bash
cd tests/
./run_tests.sh
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/mebrito/meanfieldpb.git
cd meanfieldpb

# Install in development mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 meanfieldpb/
black meanfieldpb/

# Run type checking
mypy meanfieldpb/
```

## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MeanFieldPB in your research, please cite:

```bibtex
@article{brito2019meanfield,
  title={Mean-field theory for charged microgels at interfaces},
  author={Brito, Mariano E. and others},
  journal={Journal of Chemical Physics},
  volume={151},
  pages={224901},
  year={2019},
  doi={10.1063/1.5129575}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Links

- **Repository**: https://github.com/mebrito/meanfieldpb
- **Issues**: https://github.com/mebrito/meanfieldpb/issues
- **Documentation**: https://meanfieldpb.readthedocs.io

## Author

**Mariano E. Brito**  
Email: mbrito@icp.uni-stuttgart.de

## Acknowledgments

This package implements mean-field Poisson-Boltzmann theory for colloidal suspensions, building on established theoretical frameworks in colloid science and statistical mechanics.
