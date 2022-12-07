# Alice dataset for Eelbrain

This repository contains scripts and instructions to reproduce the results from the manuscript [*Eelbrain: A toolkit for time-continuous analysis with temporal response functions*](https://doi.org/10.1101/2021.08.01.454687).


# Setup

## Download this repository

If you're familiar with git, clone this repository. If not, simply download it as a [zip file](https://github.com/Eelbrain/Alice/archive/refs/heads/main.zip).

## Create the Python environment

The easiest way to install all the required libraries is with [conda](https://docs.conda.io/), which comes with the [Anaconda Python distribution](https://www.anaconda.com/products/individual). Once `conda` is installed, simply run, from the directory in which this `README` file is located:

```bash
$ conda env create --file=environment.yml
```

This will install all the required libraries into a new environment called `eelbrain`. Activate the new environment with:

```bash
$ conda activate eelbrain
```

You will have to activate the environment every time you start a new shell session.


## Download the Alice dataset

Download the Alice EEG dataset. This repository comes with a script that can automatically download the required data from [UMD DRUM](https://drum.lib.umd.edu/handle/1903/27591) by running:

```bash
$ python download_alice.py
```

The default download location is ``~/Data/Alice``. The scripts in the Alice repository expect to find the dataset at this location. If you want to store the dataset at a different location, provide the location as argument for the download:

```bash
$ python download_alice.py download/directory
```

then either create a link to the dataset at ``~/Data/Alice``, or change the root path where it occurs in scripts (always near the beginning).

This data has been derived from the [original dataset](https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t) using the script at `import_dataset/convert-all.py`.


## Download TRFs

All TRFs used in the different figures can be computed and saved using the scripts in the `analysis` directory. However, this may require substantial computing time. To get started faster, the TRFs can also be downloaded from [here](https://www.dropbox.com/s/0mkgmii47iiq4fw/17-11-2022-TR-eLife-85012%20-%20TRFs.zip?dl=0). Just move the downloaded `TRFs` folder into the `~/Data/Alice` directory, i.e., as `~/Data/Alice/TRFs`.


## Notebooks

Many Python scripts in this repository are actually [Jupyter](https://jupyter.org/documentation) notebooks. They can be recognized as such because of their header that starts with:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
```

These scripts were converted to Python scripts with [Jupytext](http://jupytext.readthedocs.io) for efficient management with git. To turn such a script back into a notebook, run this command (assuming the script is called `notebook.py`):

```bash
$ jupytext --to notebook notebook.py
```

# Subdirectories

## Predictors

The `predictors` directory contains scripts for generating predictor variables. These should be created first, as they are used in many of the other scripts:

- `make_gammatone.py`: Generate high resolution gammatone spectrograms which are used by `make_gammatone_predictors.py`
- `make_gammatone_predictors.py`: Generate continuous acoustic predictor variables
- `make_word_predictors.py`: Generate word-level predictor variables consisting of impulses at word onsets


## Analysis

The `analysis` directory contains scripts used to estimate and save various mTRF models for the EEG dataset. These mTRF models are used in some of the figure scripts.


## Figures

The `figures` directory contains the code used to generate all the figures in the paper.


## Import_dataset

This directory contains the scripts that were used to convert the data from the original Alice EEG dataset to the format used here.


# Further resources

This tutorial and dataset:
 - [Ask questions](https://github.com/Eelbrain/Alice/discussions)
 - [Report issues](https://github.com/Eelbrain/Alice/issues)

Eelbrain:
 - [Command reference](https://eelbrain.readthedocs.io/en/stable/reference.html)
 - [Examples](https://eelbrain.readthedocs.io/en/stable/auto_examples/index.html)
 - [Ask questions](https://github.com/christianbrodbeck/Eelbrain/discussions)
 - [Report issues](https://github.com/christianbrodbeck/Eelbrain/issues)

Other libraries:
 - [Matplotlib](https://matplotlib.org)
 - [MNE-Python](https://mne.tools/)

