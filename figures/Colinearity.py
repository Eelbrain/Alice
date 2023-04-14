# Author: Proloy Das <pdas6@mgh.harvard.edu>
"""This script compares Ridge regression and boosting during colinearity"""
import os
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as pyplot
import eelbrain
import trftools

from scipy.signal import windows
from pyeeg.models import TRFEstimator


def plot_trf_ndvars(trfs, axes=None):
    if axes is None:
        fig, axes = pyplot.subplots(len(trfs))
    else:
        assert len(axes) == len(trfs)
        fig = axes[0].figure
    for h, ax in zip(trfs, axes):
        taxis = np.linspace(*(map(lambda x: getattr(h.time, x),
                                  ('tmin', 'tmax', 'nsamples'))))
        ax.plot(taxis, h.get_data('time'))
    return fig


STIMULI = [str(i) for i in range(1, 13)]
tempfile = os.path.realpath(os.path.join(__file__, '..',
                                         '..', ".temppath.pickled"))
DATA_ROOT = Path(eelbrain.load.unpickle(tempfile))
PREDICTOR_DIR = DATA_ROOT / 'predictors'
TRF_DIR = DATA_ROOT / 'TRFs'
TRF_DIR.mkdir(exist_ok=True)

# Where to save the figure
DST = DATA_ROOT / 'figures'
DST.mkdir(exist_ok=True)
 
# Load stimuli
# ------------
# Make sure to name the stimuli so that the TRFs can later be distinguished
# Load the gammatone-spectrograms; use the time axis of these as reference
gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]

# Simulate the source time-series
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

# Learning the TRFs via boosting
# slective_stopping and basis controls two facets of regularization.
boosting_trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated,
                                 -0.1, 1., basis=0.05, error='l1', partitions=10,
                                 selective_stopping=8)

# Learning TRFs via Ridge regression using pyEEG
x = predictors_concatenated.get_data(('time', 'frequency'))
y = eeg_concatenated.get_data('time')[:, None]
# reg_param = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # Ridge parameter
reg_param = [0.0001, 0.0002, 0.0005]  # Ridge parameter
ridge_trf = TRFEstimator(tmin=-0.1, tmax=1., srate=1/eeg[0].time.tstep, alpha=reg_param)
scores, alpha = ridge_trf.xfit(x, y, n_splits=10)
params = ridge_trf.get_params()
tt = eelbrain.UTS.from_range(params['tmin'], params['tmax'], 1 / params['srate'])
ridge_trf.h_scaled = eelbrain.NDVar(ridge_trf.coef_[:, :, 0].T, (frequency, tt), name='gammatone')

hs = eelbrain.combine((strf, boosting_trf.h_scaled.sub(), ridge_trf.h_scaled), dim_intersection=True)
titles = ('Ground Truth', 'Boosting', 'Ridge')
vmax = hs.max()
vmin = hs.min()

# Initialize figure
figure = pyplot.figure(figsize=(7.5, 5))
gridspec = figure.add_gridspec(2, 3, left=0.1, right=0.85, hspace=1.5)

# plot TRFs as arrays
for idx, (h, title) in enumerate(zip(hs, titles)):
    axes = figure.add_subplot(gridspec[0, idx])
    p = eelbrain.plot.Array(h, axes=axes, vmin=vmin, vmax=vmax, )
    axes.set_title(title, loc='left', size=10)
    if idx > 0:
        p.axes[0].set_yticklabels('')
        p.axes[0].set_ylabel('')
p.plot_colorbar(right_of=axes, label="TRF weights [a.u.]", ticks=2, w=2)

# plot two active, and one of the inactive frequency TRFs
interesting_frequencies = frequency.values[np.array([3, 4, 5])]
ds = []
axes = []
for idx, frequency in enumerate(interesting_frequencies):
    axes.append(figure.add_subplot(gridspec[1, idx]))
    axes[-1].set_title(f"frequnecy={frequency:.0f}", loc='left', size=10)
    axes[-1].set_xlabel('Time[ms]') 
    h = hs.sub(frequency=frequency)
    ds.extend([(frequency, *i) for i in zip(h, titles)])
ds = eelbrain.Dataset.from_caselist(['frequency', 'TRF', 'cond'], ds,)
colors = dict((key, eelbrain.plot.unambiguous_color(color) + (0.70,))  for key, color in zip(titles, ('black', 'orange', 'sky blue')))
p = eelbrain.plot.UTSStat('TRF', x='cond', xax='frequency', error=None, ds=ds,
                          axtitle=False, yticklabels='left', axes=axes, legend=False,
                          xlabel=False, colors=colors)
legendfig = p.plot_legend()

figure.text(0.01, 0.96, 'A) Gammatone TRF', size=10)
figure.text(0.01, 0.49, 'B) TRF comparison', size=10)

figure.savefig(DST / 'TRF comparison.pdf')
figure.savefig(DST / 'TRF comparison.png')
legendfig.savefig(DST / 'TRF comparison legend.pdf')
legendfig.savefig(DST / 'TRF comparison legend.png')
