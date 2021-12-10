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

import eelbrain
import os
from pathlib import Path
import copy
import numpy as np
from matplotlib import pyplot

# +
# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
STIMULI_DIR = DATA_ROOT / 'stimuli'
TRF_DIR = DATA_ROOT / 'TRFs'
EPOCH_DIR = DATA_ROOT / 'Epochs'

tstart = -0.1
tstop = 1
# -

# Where to save the figure
DST = DATA_ROOT / 'figures'
DST.mkdir(exist_ok=True)

# get all subjects
subjects = [subject for subject in os.listdir(TRF_DIR) if subject.startswith('S') ]

# (1) GET THE ERP RESPONSE TO A WORD ONSET
cases = []
for subject in subjects:
    epochs = eelbrain.load.unpickle(f"{EPOCH_DIR}/{subject}/{subject}_epoched_word.pickle")
    
    # get erp and normalize amplitude
    epochs['eeg'] -= epochs['eeg'].mean('time')
    epochs['eeg'] /= epochs['eeg'].std('time')
    erp = epochs['eeg'].mean('case') #- epochs.mean('case').sub(time=(tstart,0)).mean('time')
    
    cases.append([subject, erp])
column_names = ['subject', 'erp']
data_erp = eelbrain.Dataset.from_caselist(column_names, cases)

# (2) GET THE TRF TO WORD ONSETS WHEN CONTOLLED FOR ACOUSTIC REPRESENTATIONS
cases = []
for subject in subjects:
    mtrf = eelbrain.load.unpickle(f"{TRF_DIR}/{subject}/{subject} acoustic+words.pickle")
    trf = mtrf.h[-1]
    cases.append([subject, trf])
column_names = ['subject', 'trf']
data_trfs_controlled = eelbrain.Dataset.from_caselist(column_names, cases)

# +
# (3) PREPARE COMPARISON OF TRF AND ERP
ds_reshaped_erp = copy.deepcopy(data_erp)
ds_reshaped_erp.rename('erp', 'pattern')
ds_reshaped_erp['type'] = eelbrain.Factor(['ERP'], repeat=ds_reshaped_erp.shape[0])

ds_reshaped_trf_controlled_2 = copy.deepcopy(data_trfs_controlled)
ds_reshaped_trf_controlled_2.rename('trf', 'pattern')
ds_reshaped_trf_controlled_2['type'] = eelbrain.Factor(['TRF'], repeat=ds_reshaped_trf_controlled_2.shape[0])

ds_merged = eelbrain.combine([ds_reshaped_erp, ds_reshaped_trf_controlled_2], dim_intersection=True)


# -

def normalize_trf(trf):
    
    trf -= trf.mean('time')
    normalize_by = trf.std('time')
    normalize_by[normalize_by==0] = 1 # zero trf must remain zero trf
    
    norm_trf = trf / normalize_by
    
    return norm_trf


ds_merged['norm_pattern'] = eelbrain.combine([normalize_trf(pattern) for pattern in ds_merged['pattern']])

# is is just a matter of amplitude difference or is there really a difference in topography (check: when TRF is normalized)
res = eelbrain.testnd.TTestRelated('norm_pattern', 'type', match='subject', ds=ds_merged, pmin=0.05)

# (4) CREATE FIGURE 
# Initialize figure
figure = pyplot.figure(figsize=(7.5, 7.5))
gridspec = figure.add_gridspec(5,6, height_ratios=[3,2,2,2,2], left=0.1, right=0.95, hspace=0.2)
topo_args = dict(clip='circle')
det_args = dict(**topo_args, vmax=0.06, cmap='lux-a')
cbar_args = dict(label='Pearson\'s r', ticks=2, h=2)

axes = [
    figure.add_subplot(gridspec[0,0:2]),
    figure.add_subplot(gridspec[0,2:4]),
    figure.add_subplot(gridspec[0,4:6]),
    
    figure.add_subplot(gridspec[2,0]), 
    figure.add_subplot(gridspec[2,1]),
    figure.add_subplot(gridspec[2,2]),
    figure.add_subplot(gridspec[2,3]),
    figure.add_subplot(gridspec[2,4]),
    figure.add_subplot(gridspec[2,5]),

    figure.add_subplot(gridspec[3,0]), 
    figure.add_subplot(gridspec[3,1]),
    figure.add_subplot(gridspec[3,2]),
    figure.add_subplot(gridspec[3,3]),
    figure.add_subplot(gridspec[3,4]),
    figure.add_subplot(gridspec[3,5]),
    
    figure.add_subplot(gridspec[4,0]), 
    figure.add_subplot(gridspec[4,1]),
    figure.add_subplot(gridspec[4,2]),
    figure.add_subplot(gridspec[4,3]),
    figure.add_subplot(gridspec[4,4]),
    figure.add_subplot(gridspec[4,5]),
]

# +
plot = eelbrain.plot.Array(res, axes=axes[0:3], axtitle=['ERP', 'TRF','ERP - TRF'], vmax=2)

times=[0,0.05, 0.1, 0.22, 0.35, 0.8]
# add vertical lines on the times of the topographies
for time in times: 
    plot.add_vline(time, color='k', linestyle='--')
# -

time_labels = ['%d ms' % int(time*1000) for time in times]
for type_idx, c_type in enumerate(['ERP', 'TRF']): 
    topographies = [ds_merged[ds_merged['type']==c_type, 'norm_pattern'].sub(time=time) for time in times]
    
    if type_idx == 0: 
        c_axes = axes[3:9]
    else:
        c_axes = axes[9:15]

    if type_idx == 0:
        topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=time_labels)
        b = plot.plot_colorbar(left_of=c_axes[0], ticks={2:'+', -2:'-', 0.000:'0'}, label=c_type, label_rotation=90)
    else: 
        topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=None)
        b = plot.plot_colorbar(left_of=c_axes[0], ticks={2:'+', -2:'-', 0.000:'0'}, label='TRF', label_rotation=90)

c_axes = axes[15:21]
topographies = [res.masked_difference().sub(time=time) for time in times]
topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=None)
b = plot.plot_colorbar(left_of=c_axes[0], ticks={2:'+', -2:'-', 0.000:'0'}, label='ERP - TRF', label_rotation=90)

figure.text(0.08, 0.96, 'A) Comparison ERP and TRF', size=10)
figure.text(0.08, 0.6, 'B) Corresponding topographies', size=10)

figure

figure.savefig(DST / 'allChannels-ERP-TRF.png')
