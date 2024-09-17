# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup
# Import the pipeline and analysis parameters. On every import, the pipeline checks the TRFExperiment definition in `alice.py` for changes, and deletes cached files that have become outdated.

# +
from eelbrain import *
from alice import PARAMETERS, alice


# Define parameters commonly used for tests
TEST = {
    'pmin': 0.05,  # for threshold cluster based permutation tests
    'metric': 'det',  # Test model difference in % explained variability
}
# -

# # Model comparisons
# Model comparisons can be directly visualized using the `TRFExperiment.show_model_test` method. For an explanation of the model syntax see the [TRFExperiment documentation](https://trf-tools.readthedocs.io/doc/pipeline.html#models).
#
# Calling this function assumes that TRFs have already been estimated (see the `jobs.py` file in the same directory). Alternatively, to estimate (and cache) the TRFs on the fly, add another parameter: `make=True` (note that this may take a while).

alice.show_model_test('gammatone-1 +@ gammatone-on-1', **PARAMETERS, **TEST)

# Multiple model comparisons can be specified in a dictionary, where keys are names that are used in the table and plot. Use this to simultaneously show all comparisons from the figure:

alice.show_model_test({
    'Envelope only': 'gammatone-1 > 0',
    'Add Onsets': 'gammatone-1 +@ gammatone-on-1',
    'STRF > TRF': 'gammatone-8 + gammatone-on-8 > gammatone-1 + gammatone-on-1',
    'STRF onset': 'auditory-gammatone @ gammatone-on-8',
}, **PARAMETERS, **TEST)

# # TRFs
# Because models usually contain more than one term, the results are returned in a `ResultCollection` dictionary

trfs = alice.load_trf_test('gammatone-1', **PARAMETERS, pmin=0.05, make=True)
trfs

p = plot.TopoArray(trfs['gammatone-1'], t=[0.050, 0.100, 0.150, 0.400], clip='circle')
_ = p.plot_colorbar()

# Access information about the clusters
trfs['gammatone-1'].clusters

# ## mTRFs
# The same procedure applys for models with multiple TRFs.

trfs = alice.load_trf_test('gammatone-1 + gammatone-on-1', **PARAMETERS, pmin=0.05, make=True)

trfs

# TRFs can be plotted together (although topography times will be the same) ...

p = plot.TopoArray(trfs, t=[0.050, 0.100, 0.150, 0.400], clip='circle')
vmin, vmax = p.get_vlim()  # store colormap values for next plot
_ = p.plot_colorbar()

# ... or individually

p = plot.TopoArray(trfs['gammatone-on-1'], t=[0.060, 0.110, 0.180, None], clip='circle', vmax=vmax)
_ = p.plot_colorbar()

# ## Accessing single subject TRFs
# For more specific TRF analyses, the TRFs can be loaded as a `Dataset`. Note that the TRFs are listed as `NDVar`s at the bottom of the table, and that names have been normalize to valid Python variable names (i.e., `-` has been replaced with `_`):

data = alice.load_trfs(-1, 'gammatone-1 + gammatone-on-1', **PARAMETERS)
data.head()

data['gammatone_1']
