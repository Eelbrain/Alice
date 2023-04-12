# Author: Proloy Das <pdas6@mgh.harvard.edu>
"""This script compares Ridge regression and boosting during colinearity"""
import os
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import eelbrain
import mne
import trftools

from scipy.signal import windows
from pyeeg.models import TRFEstimator


def plot_trf_ndvars(trfs, axes=None):
    if axes is None:
        fig, axes = plt.subplots(len(trfs))
    else:
        assert len(axes) == len(trfs)
        fig = axes[0].figure
    for h, ax in zip(trfs, axes):
        taxis = np.linspace(*(map(lambda x: getattr(h.time, x),
                                  ('tmin', 'tmax', 'nsamples'))))
        ax.plot(taxis, h.get_data('time'))
        ax.set_title(f'{h.name}')
        ax.set_xlabel('time')
        ax.set_ylabel('TRF')
    return fig


STIMULI = [str(i) for i in range(1, 13)]
tempfile = os.path.realpath(os.path.join(__file__, '..',
                                         '..', ".temppath.pickled"))
DATA_ROOT = Path(eelbrain.load.unpickle(tempfile))
PREDICTOR_DIR = DATA_ROOT / 'predictors'
# EEG_DIR = DATA_ROOT / 'eeg'
# SUBJECTS = [path.name for path in EEG_DIR.iterdir() if re.match(r'S\d*', path.name)]
# Define a target directory for TRF estimates and make sure the directory is created
TRF_DIR = DATA_ROOT / 'TRFs'
TRF_DIR.mkdir(exist_ok=True)

# Load stimuli
# ------------
# Make sure to name the stimuli so that the TRFs can later be distinguished
# Load the gammatone-spectrograms; use the time axis of these as reference
gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]
# Load the broad-band envelope and process it in the same way
envelope = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-1.pickle') for stimulus in STIMULI]
envelope = [x.bin(0.01, dim='time', label='start') for x in envelope]
envelope = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='envelope') for x in envelope]
onset_envelope = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-on-1.pickle') for stimulus in STIMULI]
onset_envelope = [x.bin(0.01, dim='time', label='start') for x in onset_envelope]
onset_envelope = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='onset') for x in onset_envelope]
# Load onset spectrograms and make sure the time dimension is equal to the gammatone spectrograms
gammatone_onsets = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-on-8.pickle') for stimulus in STIMULI]
gammatone_onsets = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets]
gammatone_onsets = [eelbrain.set_time(x, gt.time, name='gammatone_on') for x, gt in zip(gammatone_onsets, gammatone)]
# # Load word tables and convert tables into continuous time-series with matching time dimension
word_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~word.pickle') for stimulus in STIMULI]
word_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='word') for gt, ds in zip(gammatone, word_tables)]
# # Function and content word impulses based on the boolean variables in the word-tables
# word_lexical = [eelbrain.event_impulse_predictor(gt.time, value='lexical', ds=ds, name='lexical') for gt, ds in zip(gammatone, word_tables)]
# word_nlexical = [eelbrain.event_impulse_predictor(gt.time, value='nlexical', ds=ds, name='non_lexical') for gt, ds in zip(gammatone, word_tables)]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]

# Simulate the source time-series
tstep = envelope[0].time.tstep
# t0 = 25
# trf_time_series = np.exp(-(((np.arange(-20, 100) - t0) * tstep)**2 / 0.004)) \
#             * np.sin(2. * np.pi * 4. * np.arange(-20, 100) * tstep) * 10e-9
trf_ts = np.zeros(1000)
trf_ts[:100] += 0.5 * windows.gaussian(100, 12)
trf_ts[:200] += - 0.75 * windows.gaussian(200, 17)
trf_ts[150:450] += 0.15 * windows.gaussian(300, 50)
trf_time_series = trf_ts * 1e-8
envelope_trf = eelbrain.NDVar(trf_time_series,
                              eelbrain.UTS(0, 0.001, trf_time_series.shape[-1]),
                              name='Envelope')
envelope_trf = eelbrain.resample(envelope_trf, 100)
envelope_responses = [eelbrain.convolve(envelope_trf, x[415.61543108, :] - x[415.61543108, :].mean('time'))
                      for x in gammatone]

# Simulate the source time-series
tstep = gammatone[0].time.tstep

strf_ts = np.zeros((8, 1000))
strf_ts[3, :100] += - 0.5 * windows.gaussian(100, 12)
strf_ts[3, :200] += 0.65 * windows.gaussian(200, 17)
strf_ts[3, 150:450] += - 0.15 * windows.gaussian(300, 50)
strf_ts[4, 20:120] += + 0.2 * windows.gaussian(100, 12)
strf_ts[4, 20:220] += - 0.25 * windows.gaussian(200, 17)
strf_ts[4, 170:470] += + 0.15 * windows.gaussian(300, 50)
strf_time_series = strf_ts * 1e-8
strf = eelbrain.NDVar(strf_time_series,
                      (gammatone[0].dims[0], eelbrain.UTS(0, 0.001, strf_time_series.shape[-1])),
                      name='Gammatone')
strf = eelbrain.resample(strf, 1/tstep)
p_true = eelbrain.plot.Array(strf, title='Ground Truth')
gammatone_response = [eelbrain.convolve(strf, x) for x in gammatone]
# t0 = 25
# trf_time_series = np.exp(-(((np.arange(-20, 100) - t0) * tstep)**2 / 0.004)) \
#             * np.sin(2. * np.pi * 4. * np.arange(-20, 100) * tstep) * 10e-9
trf_ts = np.zeros(1000)
trf_ts[:100] += - 0.5 * windows.gaussian(100, 12)
trf_ts[:200] += 0.65 * windows.gaussian(200, 17)
trf_ts[150:450] += - 0.15 * windows.gaussian(300, 50)
trf_time_series = trf_ts * 1e-8
envelope_trf = eelbrain.NDVar(trf_time_series,
                              eelbrain.UTS(0, 0.001, trf_time_series.shape[-1]),
                              name='Envelope')
envelope_trf = eelbrain.resample(envelope_trf, 100)
word_onset_responses = [eelbrain.convolve(envelope_trf, x[211.59617652, :] - x[211.59617652, :].mean('time'))
                      for x in gammatone]

# t1 = 35
# trf_time_series = - np.sqrt(np.exp(-(((np.arange(100) - t1) * tstep)**2 / 0.001)) * 10e-13)
trf_ts = np.zeros(1000)
trf_ts[:240] += 0.5 * windows.gaussian(240, 20)
trf_ts[30:410] -= 0.75 * windows.gaussian(380, 30)
trf_time_series = trf_ts * 3e-6
word_onset_trf = eelbrain.NDVar(trf_time_series,
                                eelbrain.UTS(0, 0.001, trf_time_series.shape[-1]),
                                name='Word-Onset')
word_onset_trf = eelbrain.resample(word_onset_trf, 100)
# word_onset_responses = [eelbrain.convolve(word_onset_trf, x - x.mean('time'))
#                         for x in word_onsets]

eeg = []
# for response in zip(envelope_responses, word_onset_responses):
#     response = eelbrain.combine(response)
#     response -= response.mean('time')
#     # print(response.std('time'))
#     response = response.sum('case')
#     noise = eelbrain.powerlaw_noise(response.dims, 1)
#     # SNR ~ -5dB
#     eeg.append(response + 1.7783 * noise * response.std() / noise.std())
for response in gammatone_response:
    response -= response.mean('time')
    noise = eelbrain.powerlaw_noise(response.dims, 1)
    # SNR ~ -5dB
    eeg.append(response + 1.7783 * noise * response.std() / noise.std())

interesting_frequencies = strf.dims[0].values[np.array([3, 4])]
fig, axes = plt.subplots(2)
fig = plot_trf_ndvars(strf.sub(frequency=interesting_frequencies), axes=axes)

# Since trials are of unequal length, we will concatenate them for the TRF 
# estimation
eeg_concatenated = eelbrain.concatenate(eeg)

## With controlling the acoustics
# predictors_concatenated = [eelbrain.concatenate(predictor) for predictor in 
#                            (gammatone, word_onsets)]
predictors_concatenated = [eelbrain.concatenate(predictor) for predictor in 
                           (gammatone, )]

# Learning the TRFs via boosting
boosting_trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated,
                                 -0.1, 1., basis=0.05, error='l1', partitions=10,
                                 selective_stopping=8)
# slective_stopping and basis controls two facets of regularization.
fig = plot_trf_ndvars(boosting_trf.h_scaled[0].sub(frequency=interesting_frequencies), axes=axes)
p_boosting = eelbrain.plot.Array(boosting_trf.h_scaled[0], title='Boosting')


# Learning TRFs via Ridge regression
# x = eelbrain.combine(predictors_concatenated).get_data(('time', 'case'))
# x = np.hstack((predictors_concatenated[0].get_data(('time', 'frequency')), predictors_concatenated[1].get_data(('time'))[:, None]))
x = eelbrain.combine(predictors_concatenated).get_data(('time', 'frequency'))
# Getting EEG data
y = eeg_concatenated.get_data('time')[:, None]

# TRF instance
reg_param = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # Ridge parameter
ridge_trf = TRFEstimator(tmin=-0.1, tmax=1., srate=1/eeg[0].time.tstep,
                         alpha=reg_param)

# Fit our model
# scores, alpha = ridge_trf.xfit(x, y, n_splits=10, feat_names=['Envelope', "Word-Onset"])
scores, alpha = ridge_trf.xfit(x, y, n_splits=10)

params = ridge_trf.get_params()
tt = eelbrain.UTS.from_range(params['tmin'], params['tmax'], 1 / params['srate'])
ff = predictors_concatenated[0].get_dim('frequency')
# ridge_trf.h_scaled = [eelbrain.NDVar(ridge_trf.coef_[:, :8, 0], (tt, ff), name='gammatone'),
#                       eelbrain.NDVar(ridge_trf.coef_[:, 8, 0], (tt), name='word')]
# fig = ridge_trf.plot(feat_id=[1, 2, 3, 8], ax=[axes[0], axes[0], axes[0], axes[1]])
ridge_trf.h_scaled = (eelbrain.NDVar(ridge_trf.coef_[:, :, 0].T, (ff, tt), name='gammatone'),)
fig = plot_trf_ndvars(ridge_trf.h_scaled[0].sub(frequency=interesting_frequencies), axes=axes)
p_ridge = eelbrain.plot.Array(ridge_trf.h_scaled[0], title='Ridge')

# ## Without controlling the acoustics
# predictors_concatenated = [eelbrain.concatenate(word_onsets)]
# # Learning the TRFs via boosting
# boosting_trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated,
#                                  -0.1, 1., basis=0.05, error='l1', partitions=10)
# fig = plot_trf_ndvars(boosting_trf.h_scaled, axes=[axes[1]])

# # Learning TRFs via Ridge regression
# # x = eelbrain.combine(predictors_concatenated).get_data(('time', 'case'))
# x = eelbrain.combine(predictors_concatenated).get_data('time')[:, None]
# # Getting EEG data
# y = eeg_concatenated.get_data('time')[:, None]

# # TRF instance
# reg_param = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]  # Ridge parameter
# ridge_trf = TRFEstimator(tmin=-0.1, tmax=1, srate=1/eeg[0].time.tstep,
#                          alpha=reg_param)

# # Fit our model
# scores, alpha = ridge_trf.xfit(x, y, n_splits=10, feat_names=["Word-Onset"])
# fig = ridge_trf.plot(feat_id=[0], ax=axes[1])

# axes[0].legend(('true', 'boosting (a+w)', 'ridge (a+w)'))
# axes[1].legend(('true', 'boosting (a+w)', 'ridge (a+w)', 'boosting (w)', 'ridge (w)'))
axes[0].legend(('true', 'boosting (g)', 'ridge (g)'))
axes[1].legend(('true', 'boosting (g)', 'ridge (g)',))
fig.show()
fig.tight_layout()
