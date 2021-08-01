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

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
import mne
import re
import trftools


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
EEG_DIR = DATA_ROOT / 'eeg'
TRF_DIR = DATA_ROOT / 'TRFs'
SUBJECTS = [path.name for path in EEG_DIR.iterdir() if re.match(r'S\d*', path.name)]

# Where to save the figure
DST = DATA_ROOT / 'figures'
DST.mkdir(exist_ok=True)

# Configure the matplotlib figure style
FONT = 'Helvetica Neue'
FONT_SIZE = 8
RC = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
}
pyplot.rcParams.update(RC)
# -

# # Do brain responses differ between word class?
# Test whether adding predcitors that distinguish function and content words improves the predictive power of the TRF models.

# Load predictive power of all models
models = ['words', 'words+lexical', 'acoustic+words', 'acoustic+words+lexical']
rows = []
for model in models:
    for subject in SUBJECTS:
        trf = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}.pickle')
        rows.append([subject, model, trf.proportion_explained])
model_data = eelbrain.Dataset.from_caselist(['subject', 'model', 'det'], rows)
# For more interpretable numbers, express proportion explained in terms of the maximum explained variability of the most complete model
index = model_data['model'] == 'acoustic+words+lexical'
model_data['det'] *= 100 / model_data[index, 'det'].mean('case').max('sensor')

lexical_model_test = eelbrain.testnd.TTestRelated('det', 'model', 'words+lexical', 'words', match='subject', ds=model_data, tail=1, pmin=0.05)
p = eelbrain.plot.Topomap(lexical_model_test, ncol=3, title=lexical_model_test, axh=2, clip='circle')

# ## How do the responses differ?
# Compare the TRFs corresponding to content and function words.

# load the TRFs
rows = []
for subject in SUBJECTS:
    trf = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} words+lexical.pickle')
    rows.append([subject, model, *trf.h_scaled])
trfs = eelbrain.Dataset.from_caselist(['subject', 'model', *trf.x], rows)

word_difference = eelbrain.testnd.TTestRelated('non_lexical + word', 'lexical + word', ds=trfs, pmin=0.05)

p = eelbrain.plot.TopoArray(word_difference, t=[0.100, 0.220, 0.400], clip='circle')

# ## When controlling for auditory responses?
# Do the same test, but include predictors controlling for responses to acoustic features in both models

lexical_acoustic_model_test = eelbrain.testnd.TTestRelated('det', 'model', 'acoustic+words+lexical', 'acoustic+words', match='subject', ds=model_data, tail=1, pmin=0.05)
p = eelbrain.plot.Topomap(lexical_acoustic_model_test, ncol=3, title=lexical_acoustic_model_test)

# ## Acoustic responses?
# Do acoustic predictors have predictive power in the area that's affected?

acoustic_model_test = eelbrain.testnd.TTestRelated('det', 'model', 'acoustic+words', 'words', match='subject', ds=model_data, tail=1, pmin=0.05)
p = eelbrain.plot.Topomap(acoustic_model_test, ncol=3, title=acoustic_model_test)

# # Analyze spectrogram by word class
# If auditory responses can explain the difference in response to function and content words, then that suggests that acoustic properties differ between function and content words. We can analyze this directly with TRFs. 

trf_word = eelbrain.load.unpickle(TRF_DIR / 'gammatone~word.pickle')
trf_lexical = eelbrain.load.unpickle(TRF_DIR / 'gammatone~word+lexical.pickle')

# Test whether information about the lexical status of the words improves prediction of the acoustic signal. 

ds_word = trf_word.partition_result_data()
ds_lexical = trf_lexical.partition_result_data()

# Test and plot predictive power difference
res = eelbrain.testnd.TTestRelated(ds_lexical['det'], ds_word['det'], tail=1)
ds_word[:, 'model'] = 'word'
ds_lexical[:, 'model'] = 'word+lexical'
ds = eelbrain.combine([ds_word, ds_lexical], incomplete='drop')
p = eelbrain.plot.UTSStat('det', 'model', match='i_test', ds=ds, title=res, h=2)

# For a univariate test, average across frequency
eelbrain.test.TTestRelated("det.mean('frequency')", 'model', match='i_test', ds=ds)

# Compare TRFs
word_acoustics_difference = eelbrain.testnd.TTestRelated('word + non_lexical', 'word + lexical', ds=ds_lexical)
p = eelbrain.plot.Array(word_acoustics_difference, ncol=3, h=2)

# # Generate figure

# +
# Initialize figure
figure = pyplot.figure(figsize=(7.5, 5))
gridspec = figure.add_gridspec(4, 9, height_ratios=[2,2,2,2], left=0.05, right=0.95, hspace=0.3)
topo_args = dict(clip='circle')
det_args = dict(**topo_args, vmax=15, cmap='lux-a')
cbar_args = dict(label='%', ticks=3, h=.5)

# Add predictive power tests
axes = figure.add_subplot(gridspec[0,0])
p = eelbrain.plot.Topomap(lexical_model_test.masked_difference(), axes=axes, **det_args)
axes.set_title("Word class\nwithout acoustics", loc='left')
p.plot_colorbar(right_of=axes, **cbar_args)

axes = figure.add_subplot(gridspec[1,0])
p = eelbrain.plot.Topomap(lexical_acoustic_model_test.masked_difference(), axes=axes, **det_args)
axes.set_title("Word class\nwith acoustics", loc='left')
p.plot_colorbar(right_of=axes, **cbar_args)

det_args['vmax'] = 100
axes = figure.add_subplot(gridspec[2,0])
p = eelbrain.plot.Topomap(acoustic_model_test.masked_difference(), axes=axes, **det_args)
axes.set_title("Acoustics", loc='left')
p.plot_colorbar(right_of=axes, **cbar_args)

# Add TRFs
axes = [
    figure.add_subplot(gridspec[0,3:5]), 
    figure.add_subplot(gridspec[1,3]), 
    figure.add_subplot(gridspec[1,4]),
    figure.add_subplot(gridspec[0,5:7]), 
    figure.add_subplot(gridspec[1,5]), 
    figure.add_subplot(gridspec[1,6]),
    figure.add_subplot(gridspec[0,7:9]), 
    figure.add_subplot(gridspec[1,7]), 
    figure.add_subplot(gridspec[1,8]),
]
p = eelbrain.plot.TopoArray(word_difference, t=[0.120, 0.220], axes=axes, axtitle=False, **topo_args, xlim=(-0.050, 1.00), topo_labels='below')
axes[0].set_title('Function words', loc='left')
axes[3].set_title('Content words', loc='left')
axes[6].set_title('Function > Content', loc='left')
p.plot_colorbar(left_of=axes[1], ticks=3)

# Add acoustic patterns
axes = [
    figure.add_subplot(gridspec[3,3:5]), 
    figure.add_subplot(gridspec[3,5:7]), 
    figure.add_subplot(gridspec[3,7:9]), 
]
plots = [word_acoustics_difference.c1_mean, word_acoustics_difference.c0_mean, word_acoustics_difference.difference]
p = eelbrain.plot.Array(plots, axes=axes, axtitle=False)
axes[0].set_title('Function', loc='left')
axes[1].set_title('Content', loc='left')
axes[2].set_title('Function > Content', loc='left')
# Add a line to highlight difference
for ax in axes:
    ax.axvline(0.070, color='k', alpha=0.5, linestyle=':')

figure.text(0.01, 0.96, 'A) Predictive power', size=10)
figure.text(0.27, 0.96, 'B) Word class TRFs (without acoustics)', size=10)
figure.text(0.27, 0.37, 'C) Spectrogram by word class', size=10)

figure.savefig(DST / 'Word-class-acoustics.pdf')
figure.savefig(DST / 'Word-class-acoustics.png')
