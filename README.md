# Inverse PINNs for Kerr Spin Inference

Python code and numerical results for the manuscript:

**Identifiability Limits of Inverse Physics-Informed Neural Networks for Reduced Black-Hole Spin Inference**

**Author:** Stella Menziltsidou  
**Democritus University of Thrace, Greece**

## About

This study uses Physics-Informed Neural Networks (PINNs) to estimate Kerr black-hole spin from synthetic angular data using the spin-weighted Teukolsky equation.

The experiments compare Adam, L-BFGS, SOAP, different neural networks, and numerical validation methods.

## Files

- `kerr_teukolsky_study.py` — Main experiments
- `kerr_*.csv` — Numerical results
- `validate_independent.py` — Independent validation
- `validate_external.py` — External validation
- `run_soap.py` — SOAP optimizer
- `sigmoid_k.py` — Sigmoid sensitivity
- `kerr_fig_*.pdf` — Figures

## Installation

```bash
pip install -r requirements.txt
```

## Run

Main experiments:

```bash
python kerr_teukolsky_study.py --mode all
```

Additional experiments:

```bash
python run_soap.py
python sigmoid_k.py
```

## Results

| Optimizer | Spin MAE |
|---|---:|
| Adam | 0.237322 |
| Adam + L-BFGS | 0.000864 |
| SOAP | 0.183075 |

Results are based on synthetic gravitational angular profiles at a fixed real frequency. They do not represent observational black-hole spin measurements or guarantee accurate recovery for every initialization.

## Citation

Stella Menziltsidou. *Identifiability Limits of Inverse Physics-Informed Neural Networks for Reduced Black-Hole Spin Inference.*
