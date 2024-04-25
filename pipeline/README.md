# TRF-Experiment Pipeline

This folder contains an example of how to use the experimental TRF-Experiment pipeline. This pipeline assumes the same file system origanization as the main tutorial for EEG data and predictors. However, all intermediate files and results will be managed by the pipeline.

If starting from scratch, only the following scripts from the base repository would need to be executed to create the required files before using the pipeline:

 - `download_alice.py` to download the dataset
 - `predictors/make_gammatone.py` to create high resolution gammatone spectrograms
 - `predictors/make_gammatone_predictors.py` to create predictors derived from the spectrograms
 - `predictors/make_word_predictors.py` to create word-based predictors


The core of the pipeline is the TRF-Experiment specification in the [`alice.py`](alice.py) file. This experiment can then be imported and used from other Python scripts to access data and results. This is demonstrated in the [`Auditory-TRFs.py`](Auditory-TRFs.py) notebook in this folder, which demonstrates an analysis like the original [`figures/Auditory-TRFs.py`](https://github.com/Eelbrain/Alice/blob/main/figures/Auditory-TRFs.py), but using the pipeline instead of the individual TRFs created through the script in the base repository (see the [Alice readme](../#notebooks) on how to restore notebooks from `*.py` files).

The [`TRFExperiment`](https://github.com/christianbrodbeck/TRF-Tools/blob/main/trftools/pipeline/_experiment.py) pipeline is an extension of the Eelbrain [`MneExperiment`](https://eelbrain.readthedocs.io/en/stable/experiment.html) pipeline. Documentation for the functionality of `MneExperiment` is best found in Eelbrain [documentation](http://eelbrain.readthedocs.io/en/stable/). Documentation for the TRF-Experiment extension is currently most easily found in the [source code](https://github.com/christianbrodbeck/TRF-Tools/blob/main/trftools/pipeline/_experiment.py), i.e., it can also directly retrieved using `iPython`'s help functionality.


> [!WARNING]  
> The `TRFExperiment` checks its cache for consistency every time the pipeline object is initialized (Like its parent class, [`MneExperiment`](https://eelbrain.readthedocs.io/en/stable/experiment.html)). There is, however, one exception: The pipeline cannot currently detect changes in predictor *files*. Whenever you change a predictor file, you *must* call the `TRFExperiment.invalidate_cache()` method with the given predictor name. This will delete all cached files that depend on a given predictor.


# Installing

The pipeline is implemented in the [TRF-Tools](https://github.com/christianbrodbeck/TRF-Tools) GitHub repository, which can be installed and updated with `pip`:

```bash
$ pip install --upgrade https://github.com/christianbrodbeck/TRF-Tools/archive/refs/heads/main.zip
```

A suitable [environment](environment.yml) file can be found in this folder.


# Batch estimating TRFs

The pipeline computes and caches TRFs whenever they are requested through one of the methods for accessing results. However, sometimes one might want to estimate a large number of TRFs. This can be done by creating a list of TRF jobs, as in the [`jobs.py`](jobs.py) example file. These TRFs can then be pre-computed by running the following command in a terminal:

```bash
$ trf-tools-make-jobs jobs.py
```

When running this command, TRFs that have already been cached will be skipped automatically, so there is no need to remove previous jobs from `jobs.py`. For example, when adding new subjects to a dataset this command can be used to compuate all TRFs for the new subjects. The pipeline also performs a cache check for every TRF, so this is a convenient way to re-create all TRFs after, for example, changing a preprocessing parameter.
