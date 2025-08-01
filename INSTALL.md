# Installation

Follow these steps to install and set up the MeanFieldPB package:

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- git (for cloning the repository)

## Installation Steps

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/mebrito/meanfieldpb.git
    cd meanfieldpb
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

4. **Install the package**:

    ```bash
    python3 -m pip install -e .
    ```

5. **Complete installation script**:
    ```bash
    git clone https://github.com/mebrito/meanfieldpb.git
    cd meanfieldpb
    python3 -m venv env
    source env/bin/activate
    python3 -m pip install -r requirements.txt
    python3 -m pip install -e .
    ```

## Verification

After installation, verify that the package is working correctly:

```python
# Test basic import
import meanfieldpb
print(f"MeanFieldPB version: {meanfieldpb.__version__}")

# Test a simple colloid creation
from meanfieldpb import Colloid
colloid = Colloid(a=50, Z=100, lb=0.71, vol_frac=0.001, c_salt=0.0001, charge_type='strong')
print(f"Cell radius: {colloid.R_cell:.2f} nm")
```

## Usage

After installation, you can use the package in your Python scripts:

```python
from meanfieldpb import Colloid, VolumeMicrogel, SurfaceMicrogel, LinearPolyelectrolyte
```

You can also run the example scripts:

```bash
cd samples/
python3 volume_microgel_sample.py
```

## Uninstallation

To remove the package, deactivate the virtual environment and delete the `env` directory:

```bash
deactivate
rm -rf env
```

To uninstall just the package (keeping the environment):

```bash
pip uninstall meanfieldpb
```

## Troubleshooting

- **Python Version**: Ensure you are using Python 3.7 or higher.
- **Permission Issues**: If you encounter permission issues, try using `python3 -m pip install --user ...`.
- **Virtual Environment**: Always activate your virtual environment before using the package.
- **Dependencies**: Make sure all dependencies (NumPy, SciPy, Matplotlib) are properly installed.
- **Import Errors**: If you get import errors, ensure the package was installed correctly with `pip list | grep meanfieldpb`.

### Common Issues

1. **ModuleNotFoundError**: Make sure you've activated the virtual environment and installed the package.
2. **Compilation Errors**: Ensure you have the latest versions of NumPy and SciPy.
3. **Plotting Issues**: Install matplotlib if visualization examples don't work.

For more information, refer to the [README](./README.md) or open an issue on the repository.

## Development Installation

If you plan to contribute to the project:

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Install in development mode with all dependencies:

```bash
git clone https://github.com/mebrito/meanfieldpb.git
cd meanfieldpb
python3 -m venv env
source env/bin/activate
pip install -e .[dev]
```

This will install the package in editable mode with development tools (pytest, black, flake8, etc.).
