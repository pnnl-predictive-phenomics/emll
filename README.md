# Ensemble Modeling with Linear-Logarithmic Kinetics (emll)

This project is a maintained fork of the code [pstjohn/emll](https://github.com/pstjohn/emll) using PyTensor over Theano. 

Works using emll can be found:
- St John, P. C., Strutz, J., Broadbelt, L. J., Tyo, K. E. J., & Bomble, Y. J. (2019). [Bayesian inference of metabolic kinetics from genome-scale multiomics data.](https://dx.plos.org/10.1371/journal.pcbi.1007424) PLoS Computational Biology, 15(11), e1007424. 
- McNaughton, A. D., Bredeweg, E. L., Manzer, J., Zucker, J., Munoz Munoz, N., Burnet, M. C., Nakayasu, E. S., Pomraning, K. R., Merkley, E. D., Dai, Z., Chrisler, W. B., Baker, S. E., St John, P. C., & Kumar, N. (2021). [Bayesian Inference for Integrating Yarrowia lipolytica Multiomics Datasets with Metabolic Modeling.](https://pubs.acs.org/doi/full/10.1021/acssynbio.1c00267) ACS Synthetic Biology [Electronic Resource], 10(11), 2968–2981. 
- McNaughton, A., Pino, J., Mahserejian, S., George, A., Johnson, C., Bohutskyi, P., Petyuk, V., & Zucker, J. (2024). [Bayesian framework for predicting and controlling metabolic phenotypes in microbial system.](https://doi.org/10.2172/2466236) Pacific Northwest National Laboratory (PNNL).

## Installation

To install:

```shell
pip install git+https://github.com/pnnl-predictive-phenomics/emll.git
```

or to install in developer mode:
```python
git clone https://github.com/pnnl-predictive-phenomics/emll.git
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
