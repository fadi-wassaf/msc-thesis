# msc-thesis

This repository contains the TeX source and code used for my master's thesis ([arXiv:2406.04351](https://arxiv.org/abs/2406.04351)) titled "Efficiently Building and Characterizing Electromagnetic Models of Multi-Qubit Superconducting Circuits" that was completed at RWTH Aachen University.

## Python Setup
To run any of the examples in `analysis/`, or to use any of the methods implemented in `src/tools/`, install the local package using pip. I recommend doing this within a virtial environment.
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Directories
 - `analysis/` : Model analysis examples that generate the plots within `thesis/figures/`
 - `models/` : Ansys models and electromagnetic simulation data used in `analysis/`
 - `src/` : Implementations for the following:
   - Rational impedance interconnection
   - CL Cascade network analysis
   - Lossless reciprocal vector fitting
   - Z/S-parameter grid builder
   - Transmon network analysis
 - `thesis/` : TeX source and [PDF](./thesis/main.pdf) for the thesis document