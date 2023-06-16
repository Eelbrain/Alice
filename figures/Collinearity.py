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

# Simulations comparing boosting with ridge regression in presence collinearity.

# +
from pathlib import Path

import numpy as np
import matplotlib.pyplot as pyplot
import eelbrain

from scipy.signal import windows
from pyeeg.models import TRFEstimator


STIMULI = [str(i) for i in range(1, 13)]
# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
TRF_DIR = DATA_ROOT / 'TRFs'
TRF_DIR.mkdir(exist_ok=True)

# Where to cache simulation results
SIMULATION_DIR = DATA_ROOT / 'simulations'
SIMULATION_DIR.mkdir(exist_ok=True)

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
# -

# # Load stimuli

# +
# Make sure to name the stimuli so that the TRFs can later be distinguished
# Load the gammatone-spectrograms; use the time axis of these as reference
gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [eelbrain.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]
# -

# # Simulate the EEG

# +
# two of the adjacent bands in the gammatone (band 3 and 4) are assumed to drive
# the auditory response with a spatiotemporally alternating pattern.
tstep = gammatone[0].time.tstep
frequency = gammatone[0].get_dim(('frequency'))
time = eelbrain.UTS(0, 0.001, 1000)
strf = eelbrain.NDVar.zeros((frequency, time), name='Gammatone TRF')
strf.x[3, :100] += - 0.5 * windows.gaussian(100, 12)
strf.x[3, :200] += 0.65 * windows.gaussian(200, 17)
strf.x[3, 150:450] += - 0.15 * windows.gaussian(300, 50)
strf.x[4, 20:120] += + 0.2 * windows.gaussian(100, 12)
strf.x[4, 20:220] += - 0.25 * windows.gaussian(200, 17)
strf.x[4, 170:470] += + 0.15 * windows.gaussian(300, 50)
strf.x *= 1e-8
strf = eelbrain.resample(strf, 1/tstep)
gammatone_response = [eelbrain.convolve(strf, x) for x in gammatone]

# Add pink noise to the auditory responses to simulate raw EEG data
eeg = []
for response in gammatone_response:
    response -= response.mean('time')
    noise = eelbrain.powerlaw_noise(response.dims, 1)
    # SNR ~ -5dB
    eeg.append(response + 1.7783 * noise * response.std() / noise.std())

# Since trials are of unequal length, we will concatenate them for the TRF 
# estimation
eeg_concatenated = eelbrain.concatenate(eeg)
predictors_concatenated = eelbrain.concatenate(gammatone)
# -

# # Learn TRFs via boosting

# selective_stopping controls one facet of regularization
cache_path = SIMULATION_DIR / 'boosting.pickle'
if cache_path.exists():
    boosting_trfs = eelbrain.load.unpickle(cache_path)
else:
    boosting_trfs = [eelbrain.boosting(eeg_concatenated, predictors_concatenated, -0.1, 1., basis=0.05, error='l1', partitions=10, selective_stopping=ii, test=1, partition_results=True) for ii in range(1, 15, 1)]
    eelbrain.save.pickle(boosting_trfs, cache_path)
# Select selective_stopping when explained variances in test data starts decreasing
explained_variances_in_test = [model.proportion_explained for model in boosting_trfs]
increments = np.diff(explained_variances_in_test, prepend=0)
best_stopping = np.where(increments < 0)[0][0] - 1
boosting_trf = boosting_trfs[best_stopping]

# # Learn TRFs via Ridge regression using pyEEG

cache_path = SIMULATION_DIR / 'ridge.pickle'
if cache_path.exists():
    ridge_trf = eelbrain.load.unpickle(cache_path)
else:
    x = predictors_concatenated.get_data(('time', 'frequency'))
    y = eeg_concatenated.get_data('time')[:, None]
    reg_param = [0.02, 0.05, 0.1, 0.2, 0.5, 1]  # Ridge parameters
    ridge_trf = TRFEstimator(tmin=-0.1, tmax=1., srate=1/eeg[0].time.tstep, alpha=reg_param)
    scores, alpha = ridge_trf.xfit(x, y, n_splits=10)
    params = ridge_trf.get_params()
    tt = eelbrain.UTS.from_range(params['tmin'], params['tmax'], 1 / params['srate'])
    ridge_trf.h_scaled = eelbrain.NDVar(ridge_trf.coef_[:, :, 0].T, (frequency, tt), name='gammatone')
    eelbrain.save.pickle(ridge_trf, cache_path)

# # Figure

# +
# Prepare mTRFs for ploting
# hs = eelbrain.combine((strf, boosting_trf.h_scaled, ridge_trf.h_scaled), dim_intersection=True)
hs = [strf, boosting_trf.h_scaled, ridge_trf.h_scaled]
titles = ('Ground truth', 'Boosting', 'Ridge')
vmax = 8e-9

# Initialize figure
figure = pyplot.figure(figsize=(7.5, 3.5))
gridspec = figure.add_gridspec(2, 3, left=0.1, right=0.9, hspace=1., bottom=0.1)

# Plot TRFs as arrays
axes = [figure.add_subplot(gridspec[0, idx]) for idx in range(3)]
p = eelbrain.plot.Array(hs, axes=axes, vmax=vmax, xlim=(-0.100, 1.000), axtitle=False)
p.plot_colorbar(right_of=axes[-1], label="TRF weights [a.u.]", ticks=2, w=2)
for ax, title in zip(axes, titles):
    ax.set_title(title, loc='left', size=10)

# Plot two active, and one of the inactive frequency TRFs
interesting_frequencies = frequency.values[[3, 4, 5]]
colors = {key: eelbrain.plot.unambiguous_color(color) + (0.70,) for key, color in zip(titles, ('black', 'orange', 'sky blue'))}
axes = [figure.add_subplot(gridspec[1, idx]) for idx in range(3)]
# plot.UTS takes a nested list
freq_hs = [[h.sub(frequency=freq, name=title) for h, title in zip(hs, titles)] for freq in interesting_frequencies]
p = eelbrain.plot.UTS(freq_hs, axtitle=False, axes=axes, legend=False, ylabel='TRF weights [a.u.]', colors=colors, xlim=(-0.100, 1.000), bottom=-vmax, top=vmax)
for ax, frequency_ in zip(axes, interesting_frequencies):
    ax.set_yticks([-vmax, 0, vmax])
    ax.set_title(f"{frequency_:.0f} Hz", loc='right', size=10)

figure.text(0.01, 0.96, 'A) Gammatone TRF', size=10)
figure.text(0.01, 0.47, 'B) TRF comparison', size=10)
p.plot_legend((0.53, 0.44), ncols=3)

figure.savefig(DST / 'Simulation boosting vs ridge.pdf')
figure.savefig(DST / 'Simulation boosting vs ridge.png')
eelbrain.plot.figure_outline()
