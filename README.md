# AMCG: a graph dual atomic-molecular conditional molecular generator

This code reproduces the experimental results obtained with the AMCG model as presented in the paper

[AMCG: a graph dual atomic-molecular conditional molecular generator](https://iopscience.iop.org/article/10.1088/2632-2153/ad5bbf)\
C. Abate, S. Decherchi, A. Cavalli  


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install the `conda` environment used to run the scripts just run

```conda env create -f environment.yaml```

Download the data from [here](https://doi.org/10.5281/zenodo.10606528), extract the `.zip` file and copy the content of `AMCG_DATA` in the `data` folder. 

## Usage

Activate the environment via `conda activate amcg_env`

To train a model from scratch create a training configuration file and run 

```python train.py path/to/config/file```

All the possible options can be found in `configs/train_config_guide.ini`.


To sample new molecules create a sampling configuration file and run 

```python sample.py path/to/config/file```

All the possible options can be found in `configs/sample_config_guide.ini`.


To optimize existing molecules create an optimization configuration file and run

```python optimize.py path/to/config/file```

All the possible options can be found in `configs/optim_config_guide.ini`.

Working configuration files can be found in `configs` folder.

The notebook `eval.ipynb` contains the code used to generate tables and figures.


## License

AMCG is GPL-licensed. Please see `license.txt` file in the repository.
