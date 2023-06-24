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

# # Prerequisites
#
# This scripts requires the files generated by:
#  - `analysis/estimate_trfs.pyy`
#  - `analysis/estimate_trfs_reference_strategy.py`
#  - `analysis/make_erps.py`

# +
import os
from pathlib import Path

import eelbrain
from matplotlib import pyplot


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
STIMULI_DIR = DATA_ROOT / 'stimuli'
TRF_DIR = DATA_ROOT / 'TRFs'
EPOCH_DIR = DATA_ROOT / 'Epochs'

# Where to save the figure
DST = DATA_ROOT / 'figures'
DST.mkdir(exist_ok=True)

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 8
RC = {
    'figure.dpi': 150,
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

# Get all subjects
subjects = [subject for subject in os.listdir(TRF_DIR) if subject.startswith('S')]
# -

# # Load the TRFs for the different reference strategies

# +
cases = []
for subject in subjects:
    for reference, name in zip(['mastoids', 'cz', 'average'], ['', '_cz','_average']):
        trf = eelbrain.load.unpickle(TRF_DIR / subject / f"{subject} envelope{name}.pickle")
        prediction_accuracy = trf.proportion_explained * 100  # to %
        cases.append([subject, trf.h[0], prediction_accuracy, reference])

column_names = ['subject', 'trf', 'prediction_accuracy','reference']
data_trfs = eelbrain.Dataset.from_caselist(column_names, cases, random='subject')
# -

# # Figure

# +
# Initialize figure
figure = pyplot.figure(figsize=(7.5, 5))
gridspec = figure.add_gridspec(9, 9, left=0.01, right=0.95, hspace=1.5)
topo_args = dict(clip='circle')
det_args = dict(**topo_args, vmax=0.5, cmap='lux-gray')
trf_vmax = 0.007
reference_labels = {
    'mastoids': 'mastoids', 
    'cz': 'Cz', 
    'average': 'average',
}

# A) Prediction accuracies 
for reference_idx, reference in enumerate(reference_labels):
    axes = figure.add_subplot(gridspec[reference_idx*3: reference_idx*3+3, 0:3])
    p = eelbrain.plot.Topomap('prediction_accuracy', data=data_trfs[data_trfs['reference']==reference], axes=axes, **det_args)
    label = reference_labels[reference]
    axes.set_title(f"Referenced to {label}", loc='left', size=10)
p.plot_colorbar(below=axes, label="% explained", clipmin=0, ticks=5, h=2)

# B) TRFs 
times = [0.04, 0.14, 0.24]
time_labels = ['%d ms' % (time*1000) for time in times]
for reference_idx, reference in enumerate(reference_labels):
    axes = figure.add_subplot(gridspec[reference_idx*3: reference_idx*3+3, 3:9])
    reference_index = data_trfs['reference'] == reference
    trf = data_trfs[reference_index, 'trf']
    
    # Plot butterfly TRF
    kwargs = dict(vmin=-0.004, vmax=0.007, linewidth=0.5, color='#808080', ylabel='TRF weights [a.u.]', frame='t', yticklabels='none', xlim=(-0.050, 1.000), clip=True)
    if reference_idx != 2:
        kwargs.update(dict(xticklabels='none', xlabel=''))
    plot = eelbrain.plot.Butterfly(trf, axes=axes, **kwargs)
    for time in times:
        plot.add_vline(time, color='r', linestyle='--')

    # Plot topomaps
    axes = [
        figure.add_subplot(gridspec[reference_idx*3: reference_idx*3+2, 6]),
        figure.add_subplot(gridspec[reference_idx*3: reference_idx*3+2, 7]),
        figure.add_subplot(gridspec[reference_idx*3: reference_idx*3+2, 8]),
    ]
    plot_topo = eelbrain.plot.Topomap([trf.sub(time=time) for time in times], axes=axes, axtitle=time_labels, ncol=len(times), vmax=trf_vmax, **topo_args)
    plot_topo.plot_colorbar(right_of=axes[-1], label='', label_rotation=90, ticks={trf_vmax:'+', -trf_vmax:'-', 0:'0'})

figure.text(0.01, 0.96, 'A) Predictive power', size=10)
figure.text(0.30, 0.96, 'B) Envelope TRF', size=10)

figure.savefig(DST / 'Reference-strategy.pdf')
figure.savefig(DST / 'Reference-strategy.png')
eelbrain.plot.figure_outline()
