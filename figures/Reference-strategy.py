# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: tutorialAlice
#     language: python
#     name: tutorialalice
# ---

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
import mne
import re
import trftools
import os
# -

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

# get all subjects
subjects = [subject for subject in os.listdir(TRF_DIR) if subject.startswith('S')]

# +
cases = []
for subject in subjects:
    for reference, name in zip(['mastoids','cz','average'], ['envelope.pickle','envelope_cz.pickle','envelope_average.pickle']):
        mtrf = eelbrain.load.unpickle(TRF_DIR / subject / f"{subject} {name}")
        trf = mtrf.h[0]
        prediction_accuracy = mtrf.r
        cases.append([subject, trf, prediction_accuracy, reference])

column_names = ['subject', 'trf', 'prediction_accuracy','reference']
data_trfs = eelbrain.Dataset.from_caselist(column_names, cases, random='subject')
# -

# Initialize figure
figure = pyplot.figure(figsize=(7.5, 5))
gridspec = figure.add_gridspec(9,9, height_ratios=[2,2,2,2,2,2,2,2,2], left=0.05, right=0.95, hspace=1.5)
topo_args = dict(clip='circle')
det_args = dict(**topo_args, vmax=0.06, cmap='lux-a')
cbar_args = dict(label='Pearson\'s r', ticks=2, h=2)

# 1) plot the prediction accuracies 
for reference_idx, reference in enumerate(['mastoids','cz','average']): 
    axes = figure.add_subplot(gridspec[reference_idx*3:reference_idx*3+3,0:3])
    p = eelbrain.plot.Topomap('prediction_accuracy', ds=data_trfs[data_trfs['reference']==reference], axes=axes, **det_args)
    axes.set_title("Reference to %s" % reference, loc='left')
p.plot_colorbar(below=axes, **cbar_args)


# plot the TRFs 
times = [0.04, 0.14, 0.24]
for reference_idx, reference in enumerate(['mastoids','cz','average']): 
    axes = figure.add_subplot(gridspec[reference_idx*3:reference_idx*3+3,3:9])
    
    # plot butterfly TRF
    res = eelbrain.testnd.TTestOneSample('trf', ds=data_trfs[data_trfs['reference']==reference], pmin=0.05)
    
    if reference_idx != 2:
        plot = eelbrain.plot.Butterfly(res, axes=axes, vmin=-0.004, vmax=0.007, color = '#808080', ylabel='TRF weights [a.u.]', frame='t', yticklabels='none', xticklabels='none', xlabel='')
    else:
         plot = eelbrain.plot.Butterfly(res, axes=axes, vmin=-0.004, vmax=0.007, color = '#808080', ylabel='TRF weights [a.u.]', frame='t', yticklabels='none')   
    for time in times: 
        plot.add_vline(time, color='r',linestyle='--')

    # plot topomaps
    axes = [figure.add_subplot(gridspec[reference_idx*3:reference_idx*3+2,6]), 
           figure.add_subplot(gridspec[reference_idx*3:reference_idx*3+2,7]), 
           figure.add_subplot(gridspec[reference_idx*3:reference_idx*3+2,8])]
    time_labels=['%d ms' % (time*1000) for time in times]
    plot_topo = eelbrain.plot.Topomap([data_trfs[data_trfs['reference']==reference, 'trf'].sub(time=time) for time in times], axes=axes, axtitle=time_labels, ncol=len(times), head_radius=0.45, clip='circle')
    plot_topo.plot_colorbar(right_of=axes[-1], label='', label_rotation=90, ticks={0.006:'+', -0.006:'-', 0.000:'0'})
    

figure.text(0.08, 0.96, 'A) Prediction accuracy', size=10)
figure.text(0.35, 0.96, 'B) Envelope TRF', size=10)

figure

figure.savefig(DST / 'Reference-strategy.pdf')
figure.savefig(DST / 'Reference-strategy.png')


