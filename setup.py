#!/usr/bin/env python
"""Setup script for MeanFieldPB package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="meanfieldpb",
    version="0.1.0",
    
    # Author information
    author="Mariano E. Brito",
    author_email="mbrito@icp.uni-stuttgart.de",
    
    # Package description
    description="A Python package for modeling the electrostatics of charge-equilibrated colloidal and polymer suspensions using mean-field Poisson-Boltzmann equations.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # License
    license="GNU GPLv3",
    
    # Package discovery
    packages=find_packages(),
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=[
        "requests>=20.0",
        "numpy>=1.20.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.5.0"
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
        ]
    },
    
    # Package metadata
    url="https://github.com/mebrito/meanfieldpb",  # Update with actual repository URL
    project_urls={
        "Bug Reports": "https://github.com/mebrito/meanfieldpb/issues",
        "Source": "https://github.com/mebrito/meanfieldpb",
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    
    # Keywords
    keywords="poisson-boltzmann, electrostatics, colloids, microgels, polyelectrolytes, mean-field, physics, chemistry",
    
    # Include package data
    include_package_data=True,
    
    # Entry points (if you want to provide command-line scripts)
    # entry_points={
    #     "console_scripts": [
    #         "meanfieldpb=meanfieldpb.main:main",
    #     ],
    # },
    
    # Zip safe
    zip_safe=False,
)
