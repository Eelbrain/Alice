# Alice dataset for Eelbrain

This repository contains scripts and instructions to reproduce the results from [*Eelbrain, a toolkit for time-continuous analysis with temporal response functions*](https://doi.org/10.7554/eLife.85012).


# Setup

## Download this repository

If you're familiar with git, clone this repository. If not, simply download it as a [zip file](https://github.com/Eelbrain/Alice/archive/refs/heads/main.zip).

## Create the Python environment

The easiest way to install all the required libraries is using the environment file provided in this repository (`environment.yml`) as described in the [Instructions for installing Eelbrain: Full Setup](https://eelbrain.readthedocs.io/en/latest/installing.html#full-setup).

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

## Create (or download) predictors and TRFs

In order to create predictors used in the analysis (and for some plots in the figures), execute the scripts in the `predictors` directory (see [Subdirectories](#subdirectories) below).

All TRFs used in the different figures can be computed and saved using the scripts in the `analysis` directory. However, this may require substantial computing time. To get started faster, the TRFs can also be downloaded from the data repository ([TRFs.zip](https://drum.lib.umd.edu/bitstreams/c46d0bfe-3ca9-496d-b248-8f39d6772b61/download)). Just move the downloaded `TRFs` folder into the `~/Data/Alice` directory, i.e., as `~/Data/Alice/TRFs`.

> [!NOTE]  
> Replicability: Due to numerical issues, results can differ slightly between different operating systems and hardware used. 
> Similarly, implementation changes (e.g., optimization) can affect results, even if the underlying algorithms are mathematically equivalent. 
> Changes in the boosting implementation are noted in the Eelbrain [Version History](https://eelbrain.readthedocs.io/en/stable/changes.html#major-changes).


## Notebooks

Many Python scripts in this repository are actually [Jupyter](https://jupyter.org/documentation) notebooks. They can be recognized as such because of their header that starts with:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
```

These scripts were converted to Python scripts with [Jupytext](http://jupytext.readthedocs.io) for efficient management with git. 
To turn such a script back into a notebook, run this command (assuming the script is called `my_trf_analysis.py`):

```bash
$ jupytext --to notebook my_trf_analysis.py
```

# Subdirectories <a name="subdirectories"></a>

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


# Experimental pipeline

The `pipeline` directory contains instructions for using an experimental pipeline that simplifies and streamlines TRF analysis. For more information, see the [Pipeline](pipeline) Readme file.


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

