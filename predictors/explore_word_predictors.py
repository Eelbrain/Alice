# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Original Alice word predictors
# Generate word-level predictors from the original analysis.

# +
from pathlib import Path

import eelbrain
import seaborn


DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

word_table = eelbrain.load.tsv(STIMULUS_DIR / 'AliceChapterOne-EEG.csv')
# Add word frequency as variable that scales with the expected response: larger response for less frequent words 
word_table['InvLogFreq'] = 17 - word_table['LogFreq']

# Preview table
word_table.head(10)
# -

# Variables to process
VARIABLES = ['InvLogFreq', 'NGRAM', 'RNN', 'CFG', 'Position']
# Colors for plotting
colors = eelbrain.plot.colors_for_oneway(VARIABLES)

# # Word variable properties
# Explore some properties of the predictors

# Density plot with Seaborn
word_table_long = eelbrain.table.melt('value', VARIABLES, 'variable', word_table)
data = word_table_long.as_dataframe()
_ = seaborn.displot(data=data, x='value', hue='variable', kind='kde', clip=(0, None), palette=colors)

# pairwise scatter-plots
eelbrain.report.scatter_table(VARIABLES, ds=word_table)

eelbrain.test.pairwise_correlations(VARIABLES, ds=word_table)

# # Generate predictors
# Generate a table that can serve to easily generate predictor time series. Here only one stimulus is used. For actually generating the predictors, run `make_word_predictors.py`.

# +
segment_table = word_table.sub(f"Segment == 1")
# Initialize a Dataset to contain the predictors
ds = eelbrain.Dataset(
    {'time': segment_table['onset']}, # column with time-stamps
    info={'tstop': segment_table[-1, 'offset']},  # store stimulus end time for generating time-continuous predictor 
)
# add columns for predictor variables
for key in VARIABLES:
    ds[key] = segment_table[key]
# add masks for lexical and non-lexical words
ds['lexical'] = segment_table['IsLexical'] == True
ds['nlexical'] = segment_table['IsLexical'] == False            

# preview the result
ds.head(10)
