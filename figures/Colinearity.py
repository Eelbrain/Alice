# Author: Proloy Das <pdas6@mgh.harvard.edu>
"""This script compares Ridge regression and boosting during colinearity"""
from pathlib import Path
import re

import numpy as np
import eelbrain
import mne
import trftools


STIMULI = [str(i) for i in range(1, 13)]
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
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
# word_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~word.pickle') for stimulus in STIMULI]
# word_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='word') for gt, ds in zip(gammatone, word_tables)]
# # Function and content word impulses based on the boolean variables in the word-tables
# word_lexical = [eelbrain.event_impulse_predictor(gt.time, value='lexical', ds=ds, name='lexical') for gt, ds in zip(gammatone, word_tables)]
# word_nlexical = [eelbrain.event_impulse_predictor(gt.time, value='nlexical', ds=ds, name='non_lexical') for gt, ds in zip(gammatone, word_tables)]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]


# Simulate the source time-series
tstep = envelope[0].time.tstep
t0 = 16
trf_time_series = np.exp(-(((np.arange(100) - t0) * tstep)**2 / 0.01)) \
            * np.sin(2. * np.pi * 4. * np.arange(100) * tstep) * 10e-9
envelope_trf = eelbrain.NDVar(trf_time_series, eelbrain.UTS(0, tstep, trf_time_series.shape[-1]))

envelope_response = eelbrain.convolve(envelope_trf, envelope[0])

t1 = 20
trf_time_series = np.exp(-(((np.arange(100) - t0) * tstep)**2 / 0.001)) \
            * np.sin(2. * np.pi * 4. * np.arange(100) * tstep) * 10e-5
onset_envelope_trf = eelbrain.NDVar(trf_time_series, eelbrain.UTS(0, tstep, trf_time_series.shape[-1]))

onset_envelope_response = eelbrain.convolve(onset_envelope_trf, onset_envelope[0])

response = eelbrain.combine([envelope_response, onset_envelope_response])
response = response.sum('case')
noise = eelbrain.powerlaw_noise(response.dims, 1)
eeg = response + noise * response.var() / noise.var()  # SNR = 0dB
# eelbrain.plot.LineStack(eelbrain.combine((response, eeg)))

# Learning the TRFs via boosting
trfs = eelbrain.boosting(eeg, [envelope[0], onset_envelope[0]], -0.2, 0.6)
