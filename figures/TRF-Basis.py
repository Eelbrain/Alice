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
import matplotlib
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

# +
# Load cross-validated predictive power and TRFs of the different basis widths
model = 'gammatone'
basis_values = [0, 0.050, 0.100]

datasets = {}
for basis in basis_values:
    rows = []
    for subject in SUBJECTS:
        trf_path = TRF_DIR / subject / f'{subject} {model} basis-{basis*1000:.0f}.pickle'
        trf = eelbrain.load.unpickle(trf_path)
        rows.append([subject, basis, trf.proportion_explained, *trf.h])
    data = eelbrain.Dataset.from_caselist(['subject', 'basis', 'det', 'gammatone'], rows)
    data[:, 'basis_ms'] = int(basis * 1000)
    datasets[basis] = data
# Combined dataset for explanatory power
data = eelbrain.combine([data['subject', 'basis', 'basis_ms', 'det'] for data in datasets.values()])
# Average predictive power across sensors for easier comparison
data['det_mean'] = data['det'].mean('sensor') * 100

# Verify the Dataset
data.head()
# -

# Plot predictive power by basis window
p = eelbrain.plot.Barplot('det_mean', 'basis', match='subject', data=data, bottom=0.199, corr=False, h=3)

# Plot the three TRFs
for basis, data_basis in datasets.items():
    p = eelbrain.plot.TopoButterfly("gammatone.sum('frequency')", data=data_basis, t=0.040, clip='circle')

# Plot the sensor map to determine which sensor to use in the figure
p = eelbrain.plot.SensorMap(datasets[0]['gammatone'])

# +
# Settings for the figure
SENSOR = '19'

# Extract TRF for SENSOR 
ys = []
for data_i in datasets.values():
    y = data_i['gammatone'].sub(sensor=SENSOR, time=(-0.050, 0.400)).sum('frequency')
    ys.append(y)
data['gammatone_sensor'] = eelbrain.combine(ys)

# +
# Figure
LABELS = {
    '0': '0 (impulse basis)',
    '50': '50 ms basis',
    '100': '100 ms basis',
}
COLORS = eelbrain.plot.colors_for_oneway(LABELS, unambiguous=True)

# Figure layout
figure = pyplot.figure(figsize=(7.5, 3.5))
hs = [1, 1, 1]
ws = [1, 1, 3]
gridspec = figure.add_gridspec(len(hs), len(ws), top=0.92, bottom=0.15, left=0.11, right=0.99, hspace=0.4, wspace=0.2, height_ratios=hs, width_ratios=ws)
# Plotting parameters for reusing
topo_args = dict(clip='circle')
array_args = dict(xlim=(-0.050, 1.0), axtitle=False)
topo_array_args = dict(topo_labels='below', **array_args, **topo_args)
det_args = dict(**topo_args, vmax=0.01, cmap='lux-a')
cbar_args = dict(h=.5)
t_envelope = [0.050, 0.100, 0.150, 0.400]
t_onset = [0.060, 0.110, 0.180]

# Predictive power comparison
figure.text(0.01, 0.96, 'A) Predictive power', size=10)
ax = figure.add_subplot(gridspec[:2, 0])
pos = ax.get_position()
pos.y0 -= 0.1
pos.y1 -= 0.1
ax.set_position(pos)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=3, symbol=''))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
p = eelbrain.plot.Barplot('det_mean', 'basis_ms', match='subject', data=data, axes=ax, corr=False, ylabel='% explained', xlabel='Basis [ms]', frame=False, bottom=.195, top=0.205, colors=COLORS)

# Sensor map
figure.text(0.31, 0.96, 'B', size=10)
ax = figure.add_subplot(gridspec[0, 1])
p = eelbrain.plot.SensorMap(datasets[0]['gammatone'], labels='none', axes=ax, mark=SENSOR, head_radius=0.45)

# TRFs - individuals
for i, subject in enumerate(['S04', 'S06']):
    figure.text(0.45, 0.96, 'C', size=10)
    ax = figure.add_subplot(gridspec[i, 2])
    s_data = data.sub(f"subject == '{subject}'")
    legend = (0.80, 0.82) if i == 0 else False
    eelbrain.plot.UTSStat('gammatone_sensor*1000', 'basis_ms', error=False, data=s_data, axes=ax, frame='t', xlabel=False, xticklabels=False, ylabel=False, labels=LABELS, colors=COLORS, legend=legend)
    ax.set_title(f'Subject {subject}', loc='left')

# Average TRF
ax = figure.add_subplot(gridspec[2, 2], sharey=ax)
eelbrain.plot.UTSStat('gammatone_sensor*1000', 'basis_ms', data=data, axes=ax, frame='t', legend=False, colors=COLORS, ylabel=r"V (normalized $\times 10^4$)")
ax.set_title('All subjects', loc='left')

# Add windows to TRF plot
figure.text(0.85, 0.35, 'D', size=10)
y0 = 5
window_time = eelbrain.UTS(0.280, 0.010, 12)
window = eelbrain.NDVar.zeros(window_time)
window[0.330] += 10
ax.plot(window.time, y0 + window.x, color=COLORS['0'])
ax.plot(window.time, y0 + window.smooth('time', 0.050).x, color=COLORS['50'])
ax.plot(window.time.times-0.005, y0 + window.smooth('time', 0.100).x, color=COLORS['100'])

figure.savefig(DST / 'TRF-Basis.pdf')
figure.savefig(DST / 'TRF-Basis.png')
eelbrain.plot.figure_outline()
