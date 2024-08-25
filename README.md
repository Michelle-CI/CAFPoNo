# Causal Discovery via Normalizing Flows for Post-Nonelinear model (CAF-PoNo)

This is the implementation for our paper: ["Enabling Causal Discovery in Post-Nonlinear Models
with Normalizing Flows"](https://arxiv.org/pdf/2407.04980), accepted at ECAI 2024.

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.8-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/pdf/2407.04980"><img src="https://img.shields.io/badge/arXiv-2307.07973-b31b1b.svg" alt="arXiv" height="18"></a>
</p>


<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#structure">Structure</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

## Setup

```bash
conda env create -n cafpono --file env.yml 
conda activate cafpono
```

- Clone and install dodiscover package at https://github.com/francescomontagna/dodiscover

## Structure

```bash
.
├── data                        # Data used in the paper
├── env.yml                     # For environment setup
├── experiments                 # Main dir for experiments
│   ├── bivariate
│   ├── multivariate_dim
│   ├── multivariate_real
│   └── multivariate_sample_size
├── README.md
└── src                         # Source code for the implementation of CAF-PoNo and other baseline methods
    ├── data
    ├── HSIC.py
    ├── methods
    ├── pruning.py
    └── utils.py
```

## Experiments 

### Experiment for the bivariate setting 

```bash
python experiments/bivariate/main.py
```

### Experiment for multivariate settings

- Run experiment with different sample sizes
    ```bash
    python experiments/multivariate_sample_size/main.py
    ```
- Run experiment with different data dimensions 
    ```bash
    python experiments/multivariate_dim/main.py
    ```
- Run experiment with real data
    ```bash
    python experiments/multivariate_real/main.py
    ```
- Run experiment for running time comparison
    ```bash
    python experiments/multivariate_dim/timing.py
    ```

## Citation

```
@inproceedings{hoang2024enabling,
      title={Enabling Causal Discovery in Post-Nonlinear Models with Normalizing Flows}, 
      author={Hoang, Nu and Duong, Bao and Nguyen, Thin},
      year={2024},
      booktitle = {European Conference on Aritificial Intelligence (ECAI)},
}
```
