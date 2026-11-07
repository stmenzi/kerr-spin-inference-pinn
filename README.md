
# Inverse Physics-Informed Neural Networks for Kerr Spin Inference

## Overview

This repository contains the Python source code, numerical results, validation scripts, and figures supporting the manuscript:

**Identifiability Limits of Inverse Physics-Informed Neural Networks for Reduced Black-Hole Spin Inference**

**Author:** Stella Menziltsidou  
**Affiliation:** Department of Informatics, Democritus University of Thrace, Greece

The study investigates the numerical reliability and parameter-recovery limitations of inverse physics-informed neural networks (PINNs) applied to synthetic gravitational Kerr angular profiles.

The physical model is the spin-weighted angular Teukolsky equation with:

- Spin weight: s = -2
- Angular indices: l = m = 2
- Prescribed real frequency: omega = 0.5
- Spin search interval: [0.001, 0.99]

The work concerns the angular eigenproblem only. It does not solve the radial Teukolsky equation, infer complex quasi-normal-mode frequencies, or estimate black-hole spins from observational data.

## 1. Numerical Methods

### 1.1 Angular reference solver

A Jacobi-Galerkin spectral method constructs numerical angular eigenpairs. A 22-function basis is used to generate the synthetic reference profiles, while comparisons with a 16-function basis provide a numerical convergence check.

Regular endpoint behavior and normalization at the equator, S(0) = 1, are incorporated into the formulation.

### 1.2 Inverse PINN

The inverse neural network simultaneously optimizes:

- Neural-network parameters
- The bounded Kerr spin parameter
- The angular separation eigenvalue

The baseline network contains two hidden layers with 24 neurons each and tanh activation.

Training uses 800 Adam epochs, optionally followed by up to 50 L-BFGS iterations.

The objective combines the angular differential-equation residual and the mismatch with synthetic observations.

### 1.3 Numerical comparisons

The main benchmark includes nine estimation and diagnostic configurations, covering five generating spins, five Gaussian-noise levels, and five random seeds.

The experiments assess spin recovery, optimization dependence, noise sensitivity, initialization effects, architecture choices, and fixed-spin loss profiles.

Direct spectral fitting serves as a reference-family numerical comparator, not an independent astrophysical measurement.

## 2. Experimental Data

The repository contains the following completed studies:

| Experiment | Number of runs |
| --- | ---: |
| Core nine-configuration benchmark | 1125 |
| Initialization sensitivity | 125 |
| Architecture and activation comparison | 125 |
| Fixed-spin neural loss profiles | 225 |
| SOAP optimizer comparison | 125 |
| Sigmoid-exponent sensitivity | 100 |

The sigmoid-exponent experiment examines k = 0.5, 1, 2, and 4 using the same initial physical spin for all configurations.

## 3. Independent Validation

Two additional numerical checks are provided.

**Independent boundary-value validation**

`validate_independent.py` uses SciPy's `solve_bvp` implementation to compare angular eigenvalues and normalized eigenfunctions with the Jacobi-Galerkin results.

**Third-party software validation**

`validate_external.py` compares the same angular mode with the independently maintained `spheroidal` Python package, using spectral and Leaver eigenvalue calculations.

These comparisons validate the specified angular numerical problem. They do not validate radial quasi-normal-mode solutions or observational inference.

## 4. Repository Files

### Core code and results

- `kerr_teukolsky_study.py` — Main gravitational angular solver, inverse PINN, experimental configurations, and figure generation.
- `kerr_core.csv` — Complete main benchmark.
- `kerr_summary.csv` — Aggregated main benchmark results.
- `kerr_consistency.csv` — Numerical consistency checks.
- `kerr_metadata.json` — Experimental metadata.

### Additional experiments

- `kerr_architecture.csv`
- `kerr_architecture_summary.csv`
- `kerr_starts.csv`
- `kerr_fixed_profiles.csv`

### SOAP optimizer

- `run_soap.py` — SOAP experiments.
- `soap.py` — SOAP implementation with the required float64 compatibility correction.
- `kerr_soap_core.csv` — SOAP results.
- `kerr_soap_environment.json` — SOAP environment and optimizer settings.

### Sigmoid-exponent sensitivity

- `sigmoid_k.py`
- `kerr_sigmoid_k.csv`
- `kerr_sigmoid_k_summary.csv`
- `kerr_sigmoid_k_metadata.json`

### Validation

- `validate_independent.py`
- `validate_external.py`
- `kerr_independent_bvp_validation.csv`
- `kerr_external_validation.csv`

### Figures

- `kerr_fig_recovery.pdf`
- `kerr_fig_noise.pdf`
- `kerr_fig_initialization.pdf`
- `kerr_fig_architecture.pdf`
- `kerr_fig_fixed_profile.pdf`
- `kerr_fig_profile.pdf`

The script `plot_kerr_profile_inset.py` regenerates the spectral-profile figure with its magnified diagnostic inset.

## 5. Software Requirements

The Python scripts use:

- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch
- spheroidal

The recorded SOAP and sigmoid-exponent experiments used PyTorch 2.8.0+cpu and float64 precision.

The exact historical versions of every other dependency were not preserved in the available environment metadata. The requirements file therefore documents installation dependencies rather than a fully frozen historical environment.

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

A Python virtual environment is recommended.

## 6. Reproducing the Experiments

Run the commands from the repository directory.

**Important:** Execute full reproductions in a separate working copy so that the published CSV files and figures are preserved.

### 6.1 Short functional check

```bash
python kerr_teukolsky_study.py --quick --mode core
```

This performs abbreviated functional testing only.

Results from this mode must not be mixed with the full experimental benchmark.

### 6.2 Main benchmark

```bash
python kerr_teukolsky_study.py --mode all
```

The default configuration uses the full set of generating spins, noise levels, seeds, and training budgets.

Additional modes are:

```bash
python kerr_teukolsky_study.py --mode core
python kerr_teukolsky_study.py --mode architecture
python kerr_teukolsky_study.py --mode starts
python kerr_teukolsky_study.py --mode fixed_profile
python kerr_teukolsky_study.py --mode figures
```

The main script can skip previously completed configurations if their records already exist. For an independent complete reproduction, begin with a clean working copy without saved output CSVs.

### 6.3 Independent validation

```bash
python validate_independent.py
```

### 6.4 Third-party validation

```bash
python validate_external.py
```

This command requires the `spheroidal` package.

### 6.5 SOAP optimizer

```bash
python run_soap.py
```

The committed `soap.py` contains a float64 compatibility correction for the optimizer's preconditioner accumulators.

The SOAP experiment uses 800 optimizer steps and is not an equal-runtime comparison against Adam followed by additional L-BFGS iterations.

### 6.6 Sigmoid-exponent sensitivity

```bash
python sigmoid_k.py
```

The complete experiment evaluates four sigmoid exponents across five generating spins and five seeds, producing 100 noise-free runs.

The script uses `TEST_ONLY = False` for the full experiment.

### 6.7 Regenerate the magnified spectral profile

```bash
python plot_kerr_profile_inset.py
```

Run this after the main figure-generation commands because it writes `kerr_fig_profile.pdf` with the magnified inset.

The inset uses a denser candidate-spin grid applied to the same synthetic noisy observations. It does not represent a separately trained neural model or a statistically calibrated confidence interval.

## 7. Main Numerical Findings

Under the tested noise-free gravitational angular configuration:

| Method | Mean absolute spin error |
| --- | ---: |
| Adam | 0.237322 |
| Adam + L-BFGS | 0.000864 |
| SOAP | 0.183075 |

The additional sigmoid-exponent experiment produced:

| Sigmoid exponent k | MAE |
| --- | ---: |
| 0.5 | 0.001000 |
| 1.0 | 0.000864 |
| 2.0 | 0.000483 |
| 4.0 | 0.000454 |

These results demonstrate sensitivity to optimizer choice and bounded-spin parameterization under the tested numerical conditions.

They do not establish universal optimizer superiority, global identifiability, or observational black-hole spin accuracy.

## 8. Reproducibility and Limitations

The repository provides the experimental scripts, saved numerical outputs, metadata, and figures.

The numerical experiments are limited to a prescribed real frequency and one gravitational angular mode.

The reference-family inversion shares the same theoretical angular model used to generate the synthetic observations.

Independent cross-discretization and third-party comparisons support the angular implementation but do not establish astrophysical observational validity.

The nominal spectral-profile cutoff is a numerical diagnostic and should not be interpreted as a coverage-calibrated confidence interval.

The published outputs have not been independently reproduced from scratch across every experiment in a separate environment.

## Citation

Stella Menziltsidou.

*Identifiability Limits of Inverse Physics-Informed Neural Networks for Reduced Black-Hole Spin Inference.*

Manuscript associated with this repository.
