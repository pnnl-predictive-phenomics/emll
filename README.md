# Ensemble Modeling with Linear-Logarithmic Kinetics (emll)

This project is a maintained fork of the code [pstjohn/emll](https://github.com/pstjohn/emll) using PyTensor over Theano. 

Works using emll can be found:
- [Bayesian Inference of Metabolic Kinetics from Genome-Scale Multiomics Data](https://dx.plos.org/10.1371/journal.pcbi.1007424)
- [Bayesian Inference for Integrating Yarrowia lipolytica Multiomics Datasets with Metabolic Modeling](https://pubs.acs.org/doi/full/10.1021/acssynbio.1c00267)

## Installation

To install:

```shell
pip install git+https://github.com/PNNL-CompBio/emll.git
```

or to install in developer mode:
```python
git clone https://github.com/PNNL-CompBio/emll.git
cd emll
python -m pip install -e .
```

Test install by running:

```shell
python -c "import emll"
```

This code uses the intelpython distribution for some faster blas routines.

## Code

General code for solving for the steady-state metabolite and flux values as a function of elasticity parameters, enzyme expression, and external metabolite concentrations is found in `emll/linlog_model.py`. PyTensor code to perform the regularized linear regression (and integrate this operation into pymc3 models) is found in `emll/pytensor_utils.py`.



# How to Cite

If you use `emll` in your work, please cite:
```bibtex
@article{St.John2019,
author = {{St. John}, Peter C. and Strutz, Jonathan and Broadbelt, Linda J. and Tyo, Keith E. J. and Bomble, Yannick J.},
doi = {10.1371/journal.pcbi.1007424},
editor = {Maranas, Costas D.},
issn = {1553-7358},
journal = {PLOS Computational Biology},
month = {nov},
number = {11},
pages = {e1007424},
title = {{Bayesian inference of metabolic kinetics from genome-scale multiomics data}},
url = {https://dx.plos.org/10.1371/journal.pcbi.1007424},
volume = {15},
year = {2019}
}
```
