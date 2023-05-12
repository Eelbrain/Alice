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
import os
from pathlib import Path

import eelbrain
from matplotlib import pyplot

# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
STIMULI_DIR = DATA_ROOT / 'stimuli'
TRF_DIR = DATA_ROOT / 'TRFs'
ERP_DIR = DATA_ROOT / 'ERPs'

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
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
}
pyplot.rcParams.update(RC)
# -

# get all subjects
subjects = [subject for subject in os.listdir(TRF_DIR) if subject.startswith('S') ]
assert os.path.exists(ERP_DIR), "ERP directory is not found. Please, run script analysis/make_erps.py to create the different ERPs per subject."

# Get the ERP response to a word onset
cases = []
for subject in subjects:
    erp = eelbrain.load.unpickle(ERP_DIR / subject / f'{subject}_erp_word.pickle')
    cases.append([subject, erp])
# Use the same column names for ERPs and TRFs so that ERP and TRF datasets can be merged
data_erp = eelbrain.Dataset.from_caselist(['subject', 'pattern'], cases)
data_erp[:, 'type'] = 'ERP'

# Get the TRF to word onsets when controlled for acoustic representations
cases = []
for subject in subjects:
    mtrf = eelbrain.load.unpickle(TRF_DIR/ subject / f'{subject} acoustic+words.pickle')
    trf = mtrf.h[-1]
    cases.append([subject, trf])
data_trfs_controlled = eelbrain.Dataset.from_caselist(['subject', 'pattern'], cases)
data_trfs_controlled[:, 'type'] = 'TRF'

# Merge ERP and TRF data
data = eelbrain.combine([data_erp, data_trfs_controlled], dim_intersection=True)


# +
pattern_normalized = data['pattern'] - data['pattern'].mean('time')
normalize_by = pattern_normalized.std('time')
normalize_by[normalize_by == 0] = 1  # Avoid division by 0
pattern_normalized /= normalize_by
data['norm_pattern'] = pattern_normalized
data['norm_pattern_Fz'] = data['norm_pattern'].sub(sensor='1')

# Tests to compare ERP and TRF
res_fz = eelbrain.testnd.TTestRelated('norm_pattern_Fz', 'type', match='subject', data=data, pmin=0.05)
res_topo = eelbrain.testnd.TTestRelated('norm_pattern', 'type', match='subject', data=data, pmin=0.05)


# +
# Compose the figure
figure = pyplot.figure(figsize=(7.5, 6.5))
gridspec = figure.add_gridspec(8,6, height_ratios=[2, 3, 3, 5, 2, 2, 2, 2], left=.10, right=.95, bottom=.02, top=.95, hspace=0.3)
vmax = 2
topo_args = dict(clip='circle', vmax=vmax)
cbar_args = dict(ticks=3, label_rotation=90)

axes = [
    figure.add_subplot(gridspec[3, 0:2]),
    figure.add_subplot(gridspec[3, 2:4]),
    figure.add_subplot(gridspec[3, 4:6]),
    
    figure.add_subplot(gridspec[5, 0]),
    figure.add_subplot(gridspec[5, 1]),
    figure.add_subplot(gridspec[5, 2]),
    figure.add_subplot(gridspec[5, 3]),
    figure.add_subplot(gridspec[5, 4]),
    figure.add_subplot(gridspec[5, 5]),
    
    figure.add_subplot(gridspec[6, 0]),
    figure.add_subplot(gridspec[6, 1]),
    figure.add_subplot(gridspec[6, 2]),
    figure.add_subplot(gridspec[6, 3]),
    figure.add_subplot(gridspec[6, 4]),
    figure.add_subplot(gridspec[6, 5]),
    
    figure.add_subplot(gridspec[7, 0]),
    figure.add_subplot(gridspec[7, 1]),
    figure.add_subplot(gridspec[7, 2]),
    figure.add_subplot(gridspec[7, 3]),
    figure.add_subplot(gridspec[7, 4]),
    figure.add_subplot(gridspec[7, 5]),
]

c_axes = figure.add_subplot(gridspec[0:2, 1:5])
plot = eelbrain.plot.UTSStat('norm_pattern_Fz', 'type', data=data, axes=c_axes, frame='t', ylabel='Normalized pattern [a.u.]', legend=(.81, .88))
plot.set_clusters(res_fz.clusters, pmax=0.05, ptrend=None, color='.5', y=0, dy=0.1)

c_axes = figure.add_subplot(gridspec[0,4])
sensormap = eelbrain.plot.SensorMap(data['pattern'], labels='none', head_radius=0.45, axes=c_axes)
sensormap.mark_sensors('1', c='r')

plot = eelbrain.plot.Array(res_topo, axes=axes[0:3], axtitle=['ERP', 'TRF','ERP - TRF'], vmax=vmax)

times = [0.000 ,0.050, 0.100, 0.250, 0.350, 0.800]
# Add vertical lines on the times of the topographies
for time in times: 
    plot.add_vline(time, color='k', linestyle='--')

# ERP/TRF Topographies
for type_idx, c_type in enumerate(['ERP', 'TRF']): 
    topographies = [data[data['type'] == c_type, 'norm_pattern'].sub(time=time) for time in times]
    
    if type_idx == 0: 
        c_axes = axes[3:9]
        axtitle = [f'{time*1000:g} ms' for time in times]
    else:
        c_axes = axes[9:15]
        axtitle = None
    topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, axtitle=axtitle, **topo_args)
    b = topomaps.plot_colorbar(left_of=c_axes[0], label=c_type, **cbar_args)

# Difference topographies
c_axes = axes[15:21]
topographies = [res_topo.masked_difference().sub(time=time) for time in times]
topomaps = eelbrain.plot.Topomap(topographies, axes=c_axes, axtitle=None, **topo_args)
b = topomaps.plot_colorbar(left_of=c_axes[0], label='ERP - TRF', **cbar_args)

figure.text(.01, .98, 'A) Comparison of ERP and TRF at single channel', size=10)
figure.text(.01, .63, 'B) Comparison of ERP and TRF across channels', size=10)
figure.text(.01, .33, 'C) Corresponding topographies', size=10)

figure.savefig(DST / 'ERP-TRF.png')
figure.savefig(DST / 'ERP-TRF.pdf')
