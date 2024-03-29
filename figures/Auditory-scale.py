# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
import numpy
import re


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
FONT = 'Arial'
FONT_SIZE = 8
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'figure.labelsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,    
    'legend.fontsize': FONT_SIZE,
}
pyplot.rcParams.update(RC)
pyplot.rcParams['hatch.linewidth'] = 4
# -

# # Load the data

# +
# load cross-validated predictive power and TRFs of the different spectrogram models
SCALES = ['linear', 'power-law', 'log']
MODELS = {
    'linear': 'gammatone-lin',
    'power-law': 'gammatone-pow',
    'log': 'gammatone',
    'linear+log': 'gammatone-lin+log',
}
COLORS = {
    'linear': '#1f77b4',
    'power-law': '#ff7f0e',
    'log': '#d62728',
}
COLORS['linear+log'] = COLORS['linear']
datasets = {}
for scale, model in MODELS.items():
    rows = []
    for subject in SUBJECTS:
        trf = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}.pickle')
        rows.append([subject, scale, trf.proportion_explained, *trf.h])
        trf_names = trf.x
    data = eelbrain.Dataset.from_caselist(['subject', 'scale', 'det', *trf_names], rows, info={'trfs': trf_names})
    # Average predictive power across sensors for easier comparison
    data['det_mean'] = data['det'].mean('sensor')
    datasets[scale] = data

# Create an aggregated dataset for predictive power (TRFs can't be included 
# because there is a different number of predictors in the lin+log model)
data_det = eelbrain.combine([data['subject', 'scale', 'det_mean'] for data in datasets.values()])

# Verify the Dataset
data_det.head()
# -

# # Model comparisons

p = eelbrain.plot.Barplot('det_mean', 'scale', match='subject', cells=SCALES, data=data_det, h=3, w=2, xtick_rotation=-20)

# ## Pairwise tests

eelbrain.test.TTestRelated('det_mean', 'scale', 'power-law', 'linear', match='subject', data=data_det)

eelbrain.test.TTestRelated('det_mean', 'scale', 'log', 'power-law', match='subject', data=data_det)

eelbrain.test.TTestRelated('det_mean', 'scale', 'linear+log', 'log', match='subject', data=data_det)

# ## STRFs

for scale, data in datasets.items():
    p = eelbrain.plot.Array([f"{trf}.mean('sensor')" for trf in data.info['trfs']], ncol=1, data=data, title=scale, axh=2, axw=3)

# # Figure

STIMULUS = 1
gammatone_lin = eelbrain.load.unpickle(DATA_ROOT / 'stimuli' / f'{STIMULUS}-gammatone.pickle')
gammatone_pow = gammatone_lin ** 0.6
gammatone_log = (1 + gammatone_lin).log()

# +
# Figure layout
figure = pyplot.figure(figsize=(7.5, 4))
hs = [1, 1, 1, 1, 1, 1]
ws = [1, 1, 3, 1]
gridspec = figure.add_gridspec(len(hs), len(ws), top=0.92, bottom=0.15, left=0.09, right=0.99, hspace=2., wspace=0.1, height_ratios=hs, width_ratios=ws)
# Plotting parameters for reusing
topo_args = dict(clip='circle')
array_args = dict(xlim=(-0.050, 1.0), axtitle=False)
topo_array_args = dict(topo_labels='below', **array_args, **topo_args)
det_args = dict(**topo_args, vmax=1, cmap='lux-gray')
cbar_args = dict(h=.5)
t_envelope = [0.050, 0.100, 0.150, 0.400]
t_onset = [0.060, 0.110, 0.180]

# Log scale graph
figure.text(0.01, 0.96, 'A) Nonlinear scales', size=10)
ax = figure.add_subplot(gridspec[0:2, 0])
x = numpy.linspace(0, 100)
ax.plot(x, numpy.log(x+1), label='log', color=COLORS['log'], zorder=2.2)
ax.plot(x, x**0.6/3, label='power-law', color=COLORS['power-law'], zorder=2.1)
ax.plot(x, x * 0.05, label='linear', color=COLORS['linear'])
ax.set_ylabel('Brain response')
ax.set_xlabel('Acoustic power')
ax.set_xticks(())
ax.set_yticks(())
pyplot.legend(loc=(.8, .05))
        
# Spectrograms
figure.text(0.38, 0.96, 'B', size=10)
sgrams = [gammatone_lin, gammatone_pow, gammatone_log]
for i, sgram, scale in zip(range(3), sgrams, SCALES):
    ax = figure.add_subplot(gridspec[i*2:(i+1)*2, 2])
    x = sgram.sub(time=(0, 3))
    eelbrain.plot.Array(x, axes=ax, ylabel=i==2, xticklabels=i==2, xlabel=i==2)
    ax.set_yticks(range(0, 129, 32))
    ax.set_title(scale.capitalize(), loc='left')

# Predictive power topography
for i, scale in enumerate(SCALES):
    ax = figure.add_subplot(gridspec[i*2:(i+1)*2, 3])
    data = datasets[scale]
    p = eelbrain.plot.Topomap('det * 100', data=data, axes=ax, **det_args)
    if i == 2:
        p.plot_colorbar(below=ax, clipmin=0, ticks=5, label='% variability explained')

# Predictive power barplot
figure.text(0.01, 0.55, 'C) Predictive power', size=10)
ax = figure.add_subplot(gridspec[3:, 0])
p = eelbrain.plot.Barplot('det_mean * 100', 'scale', match='subject', data=data_det, cells=MODELS, axes=ax, test=False, ylabel='% variability explained', xlabel='Scale', frame=False, xtick_rotation=-30, top=.22, colors=COLORS)
res = eelbrain.test.TTestRelated('det_mean', 'scale', 'power-law', 'linear', 'subject', data=data_det)
p.mark_pair('linear', 'power-law', .2, mark=res.p)
res = eelbrain.test.TTestRelated('det_mean', 'scale', 'log', 'power-law', 'subject', data=data_det)
p.mark_pair('power-law', 'log', .23, mark=res.p)
# hatch for linear+log bar
p.bars.patches[-1].set_hatch('//')
p.bars.patches[-1].set_edgecolor(COLORS['log'])
p.bars.patches[-1].set_linewidth(0)

figure.savefig(DST / 'Auditory-Scale.pdf')
eelbrain.plot.figure_outline()
