# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import eelbrain
import os
from pathlib import Path
import copy
import numpy as np
from matplotlib import pyplot
import os

# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
STIMULI_DIR = DATA_ROOT / 'stimuli'
TRF_DIR = DATA_ROOT / 'TRFs'
ERP_DIR = DATA_ROOT / 'ERPs'


# Where to save the figure
DST = DATA_ROOT / 'figures'
DST.mkdir(exist_ok=True)

# get all subjects
subjects = [subject for subject in os.listdir(TRF_DIR) if subject.startswith('S') ]
assert os.path.exists(ERP_DIR), "ERP directory is not found. Please, run script analysis/make_erps.py to create the different ERPs per subject."

# Get the ERP response to a word onset
cases = []
for subject in subjects:
    erp = eelbrain.load.unpickle(ERP_DIR / subject / f'{subject}_erp_word.pickle')
    cases.append([subject, erp])
column_names = ['subject', 'erp']
data_erp = eelbrain.Dataset.from_caselist(column_names, cases)

# Get the TRF to word onsets when controlled for acoustic representations
cases = []
for subject in subjects:
    mtrf = eelbrain.load.unpickle(TRF_DIR/ subject / f'{subject} acoustic+words.pickle')
    trf = mtrf.h[-1]
    cases.append([subject, trf])
column_names = ['subject', 'trf']
data_trfs_controlled = eelbrain.Dataset.from_caselist(column_names, cases)

# +
# Prepare comparison of TRF and ERP
ds_reshaped_erp = copy.deepcopy(data_erp)
ds_reshaped_erp.rename('erp', 'pattern')
ds_reshaped_erp[:, 'type'] = 'ERP'

ds_reshaped_trf_controlled_2 = copy.deepcopy(data_trfs_controlled)
ds_reshaped_trf_controlled_2.rename('trf', 'pattern')
ds_reshaped_trf_controlled_2[:, 'type'] = 'TRF'

ds_merged = eelbrain.combine([ds_reshaped_erp, ds_reshaped_trf_controlled_2], dim_intersection=True)


# -
pattern_normalized = ds_merged['pattern'] - ds_merged['pattern'].mean('time')
normalize_by = pattern_normalized.std('time')
normalize_by[normalize_by==0] = 1  # Avoid division by 0
pattern_normalized /= normalize_by
ds_merged['norm_pattern'] = pattern_normalized
ds_merged['norm_pattern_Fz'] = ds_merged['norm_pattern'].sub(sensor='1')

# +
## figure 
figure = pyplot.figure(figsize=(8.5, 7.5))
gridspec = figure.add_gridspec(8,6, height_ratios=[2, 3, 3, 5,2,2,2,2], left=0.1, right=0.95, hspace=0.3)
topo_args = dict(clip='circle')
det_args = dict(**topo_args, vmax=0.06, cmap='lux-a')
cbar_args = dict(label='Pearson\'s r', ticks=2, h=2)

axes = [
    figure.add_subplot(gridspec[3,0:2]),
    figure.add_subplot(gridspec[3,2:4]),
    figure.add_subplot(gridspec[3,4:6]),
    
    figure.add_subplot(gridspec[5,0]), 
    figure.add_subplot(gridspec[5,1]),
    figure.add_subplot(gridspec[5,2]),
    figure.add_subplot(gridspec[5,3]),
    figure.add_subplot(gridspec[5,4]),
    figure.add_subplot(gridspec[5,5]),
    
    figure.add_subplot(gridspec[6,0]), 
    figure.add_subplot(gridspec[6,1]),
    figure.add_subplot(gridspec[6,2]),
    figure.add_subplot(gridspec[6,3]),
    figure.add_subplot(gridspec[6,4]),
    figure.add_subplot(gridspec[6,5]),
    
    figure.add_subplot(gridspec[7,0]), 
    figure.add_subplot(gridspec[7,1]),
    figure.add_subplot(gridspec[7,2]),
    figure.add_subplot(gridspec[7,3]),
    figure.add_subplot(gridspec[7,4]),
    figure.add_subplot(gridspec[7,5]),
]

res = eelbrain.testnd.TTestRelated('norm_pattern_Fz', 'type', match='subject', ds=ds_merged, pmin=0.05)

c_axes = figure.add_subplot(gridspec[0:2, 1:5])
plot = eelbrain.plot.UTSStat('norm_pattern_Fz', 'type', ds=ds_merged, axes=c_axes, frame='t', ylabel='Normalized pattern [a.u.]')
plot.set_clusters(res.clusters, pmax=0.05, ptrend=None, color='.5', y=0, dy=0.1)

c_axes = figure.add_subplot(gridspec[0,4])
sensormap = eelbrain.plot.SensorMap(ds_merged['pattern'], labels='none', head_radius=0.45, axes=c_axes)
sensormap.mark_sensors('1', c='r')

res = eelbrain.testnd.TTestRelated('norm_pattern', 'type', match='subject', ds=ds_merged, pmin=0.05)
plot = eelbrain.plot.Array(res, axes=axes[0:3], axtitle=['ERP', 'TRF','ERP - TRF'], vmax=2)

times = [0,0.05, 0.1, 0.25, 0.35, 0.8]
# add vertical lines on the times of the topographies
for time in times: 
    plot.add_vline(time, color='k', linestyle='--')

time_labels = ['%d ms' % int(time*1000) for time in times]
for type_idx, c_type in enumerate(['ERP', 'TRF']): 
    topographies = [ds_merged[ds_merged['type']==c_type, 'norm_pattern'].sub(time=time) for time in times]
    
    if type_idx == 0: 
        c_axes = axes[3:9]
    else:
        c_axes = axes[9:15]

    if type_idx == 0:
        topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=time_labels)
        b = topomaps.plot_colorbar(left_of=c_axes[0], ticks={-2:'-2', 0:'0', +2: '2'}, label=c_type, label_rotation=90)
    else: 
        topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=None)
        b = topomaps.plot_colorbar(left_of=c_axes[0], ticks={-2:'-2', 0:'0', +2: '2'}, label='TRF', label_rotation=90)

c_axes = axes[15:21]
topographies = [res.masked_difference().sub(time=time) for time in times]
topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, head_radius=0.45, clip='circle', axtitle=None)
b = topomaps.plot_colorbar(left_of=c_axes[0], ticks={-1:'-1', 0:'0', +1: '1'}, label='ERP - TRF', label_rotation=90)

figure.text(0, 0.96, 'A) Comparison of ERP and TRF at single channel', size=10)
figure.text(0, 0.63, 'B) Comparison of ERP and TRF across channels', size=10)
figure.text(0, 0.39, 'C) Corresponding topographies', size=10)

figure.savefig(DST / 'ERP-TRF.png')
figure.savefig(DST / 'ERP-TRF.pdf')
