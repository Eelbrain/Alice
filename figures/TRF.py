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

# # Deconvolution example figure

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
from matplotlib.patches import ConnectionPatch
import mne


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
SUBJECT, SENSOR = 'S15', '19'
STIMULUS = '1'

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
# -

# ## Load data and estimate deconvolution 

# Load and pre-process the EEG data
raw = mne.io.read_raw_fif(DATA_ROOT / 'eeg' / SUBJECT / f'{SUBJECT}_alice-raw.fif', preload=True)
raw = raw.filter(1, 8, n_jobs=1)
# Extract the events from the EEG data, and select the trial corresponding to the stimulus
events = eelbrain.load.fiff.events(raw)
# Define one second of silence to pad stimuli
silence = eelbrain.NDVar.zeros(eelbrain.UTS(0, 1/100, 100))
# Load the stimuli coresponding to the events
stimuli = []
envelopes = []
for stimulus in events['event']:
    wave = eelbrain.load.wav(STIMULUS_DIR / f'{stimulus}.wav')
    # Stimulus for plotting
    stimulus_wave = eelbrain.resample(wave, 2000)
    stimuli.append(stimulus_wave)
    # Envelope predictors
    envelope = wave.envelope()
    envelope = eelbrain.resample(envelope, 100).clip(0)
    envelope = eelbrain.concatenate([envelope, silence])
    # Log transform to approximate auditory system response characteristics
    envelope = (envelope + 1).log()
    # Apply the same filter as for the EEG data
    envelope = eelbrain.filter_data(envelope, 1, 8)
    envelopes.append(envelope)
events['stimulus'] = stimuli
events['envelope'] = envelopes
# Find the stimulus duration based on the envelopes
durations = [envelope.time.tstop for envelope in envelopes]
# Load the EEG data corresponding to this event/stimulus
events['eeg'] = eelbrain.load.fiff.variable_length_epochs(events, tmin=0, tstop=durations, decim=5)

events.summary()

# Concatenate the first 11 trials for estimating the deconvolution
eeg = eelbrain.concatenate(events[:11, 'eeg'])
envelope = eelbrain.concatenate(events[:11, 'envelope'])
# Estimate the TRFs (one for each EEG sensor)
trf = eelbrain.boosting(eeg, envelope, -0.100, 0.500, basis=0.050, partitions=4)

# Visualize the TRFs
p = eelbrain.plot.TopoArray(trf.h, t=[0.040, 0.150, 0.380], clip='circle')

# Predict response to the 12th stimulus
envelope_12 = events[11, 'envelope']
eeg_12 = events[11, 'eeg']
eeg_12_predicted = eelbrain.convolve(trf.h_scaled, envelope_12)

# Evaluate cross-validated predictions
ss_total = eeg_12.abs().sum('time')
ss_residual = (eeg_12 - eeg_12_predicted).abs().sum('time')
proportion_explained_12 = 1 - (ss_residual / ss_total)
# Plot correlation on estimation and testing data
titles = [f'Training data\nMax explained = {trf.proportion_explained.max():.2%}$', f'Testing data\nMax explained = {proportion_explained_12.max():.2%}']
p = eelbrain.plot.Topomap([trf.proportion_explained, proportion_explained_12], sensorlabels='name', clip='circle', nrow=1, axtitle=titles)
p_cb = p.plot_colorbar(width=.1, w=2)

# # Figure

# +
# Initialize figure
figure = pyplot.figure(figsize=(7.5, 4))
pyplot.subplots_adjust(left=.05, right=.99, hspace=.1, wspace=.1)
ax_args = dict()
uts_args = dict(yticklabels='none', clip=True, frame='none')
continuous_args = dict(**uts_args, xlim=(11, 16))

# Define a function to format axes
def decorate(ax):
    ax.set_yticks(())
    ax.tick_params(bottom=False)
    ax.set_clip_on(False)

# Function to normalize before plotting
def normalize(y):
    return (y - y.mean()) / y.std()

# Stimulus
args = dict(color='k', **continuous_args)
ax_sti1 = ax = pyplot.subplot2grid((4, 9), (3, 0), colspan=4, **ax_args)
eelbrain.plot.UTS(events[0, 'stimulus'], axes=ax, **args, ylabel='Stimulus')
decorate(ax)
# held-out
ax_sti2 = ax = pyplot.subplot2grid((4, 9), (3, 5), colspan=4, **ax_args)
eelbrain.plot.UTS(events[11, 'stimulus'], axes=ax, **args, ylabel=False)
decorate(ax)

# Envelope
args = dict(color='b', xlabel=False, xticklabels=False, **continuous_args)
ax_env1 = ax = pyplot.subplot2grid((4, 9), (2, 0), colspan=4, **ax_args)
eelbrain.plot.UTS(envelope, axes=ax, **args, ylabel='Predictor')
decorate(ax)
# held-out
ax_env2 = ax = pyplot.subplot2grid((4, 9), (2, 5), colspan=4, **ax_args)
eelbrain.plot.UTS(envelope_12, axes=ax, **args, ylabel=False)
decorate(ax)

# EEG
args = dict(color='k', xlabel=False, xticklabels=False, **continuous_args)
ax_eeg1 = ax = pyplot.subplot2grid((4, 9), (0, 0), colspan=4, **ax_args)
ax.set_title("Training data", loc='left')
eelbrain.plot.UTS(normalize(eeg.sub(sensor=SENSOR)), axes=ax, **args, ylabel=f'EEG-{SENSOR}')
decorate(ax)
# held-out
ax_eeg2 = ax = pyplot.subplot2grid((4,9), (0, 5), colspan=4, sharey=ax, **ax_args)
ax.set_title("Held-out testing data", loc='left')
eelbrain.plot.UTS(normalize(eeg_12.sub(sensor=SENSOR)), axes=ax, **args, ylabel=False)
args['color'] = eelbrain.plot.Style('red', linestyle='-')
eelbrain.plot.UTS(normalize(eeg_12_predicted.sub(sensor=SENSOR)), axes=ax, **args, ylabel=False)
decorate(ax)

# TRF
ax_trf = ax = pyplot.subplot2grid((4, 9), (1, 4), **ax_args)
eelbrain.plot.UTS(trf.h.sub(sensor=SENSOR), axes=ax, **uts_args, color='purple', ylabel='TRF', xlabel='Lag $\\tau$ (ms)')
decorate(ax)

# Predictive power
ax_power = ax = pyplot.subplot2grid((4, 9), (1, 7))
eelbrain.plot.Topomap(proportion_explained_12, axes=ax, clip='circle', cmap='lux-gray', mark=[SENSOR], mcolor='#009E73', msize=1)
ax.set_title('Predictive power\n(% variability explained)')

pyplot.tight_layout()

# Arrows
args = dict(color='red', arrowstyle="fancy, tail_width=0.2, head_width=0.5, head_length=0.5")
con = ConnectionPatch(
    xyA=(0.5, 0), coordsA=ax_eeg1.transAxes,
    xyB=(-0.5, 0.5), coordsB=ax_trf.transAxes,
    connectionstyle='arc3,rad=.2', **args)
ax_eeg1.add_artist(con)
con = ConnectionPatch(
    xyA=(0.5, 1), coordsA=ax_env1.transAxes,
    xyB=(-0.6, 0.3), coordsB=ax_trf.transAxes,
    connectionstyle='arc3,rad=-.2', **args)
ax_eeg1.add_artist(con)
figure.text(-2, 0.4, 'B', transform=ax_trf.transAxes, size=10)
con = ConnectionPatch(
    xyA=(1.2, 0.5), coordsA=ax_trf.transAxes,
    xyB=(0.2, 0), coordsB=ax_eeg2.transAxes,
    connectionstyle='arc3,rad=.3', **args)
ax_eeg2.add_artist(con)
con = ConnectionPatch(
    xyA=(0.22, 1), coordsA=ax_env2.transAxes,
    xyB=(0.22, -0.3), coordsB=ax_eeg2.transAxes,
    connectionstyle='arc3,rad=.0', **args)
ax_eeg2.add_artist(con)
figure.text(2.4, 1.2, 'C', transform=ax_trf.transAxes, size=10)
# Prediction -> predictive power
con = ConnectionPatch(
    xyA=(0.6, 0), coordsA=ax_eeg2.transAxes,
    xyB=(0.35, 1.9), coordsB=ax_power.transAxes,
    connectionstyle='arc3,rad=.0', **args)
ax_eeg2.add_artist(con)
figure.text(0.55, -0.4, 'D', transform=ax_eeg2.transAxes, size=10)
# Stimulus -> predictor
args['color'] = (.5, .5, .5)
con = ConnectionPatch(
    xyA=(0.5, 1), coordsA=ax_sti1.transAxes,
    xyB=(0.5, -0.1), coordsB=ax_env1.transAxes,
    connectionstyle='arc3,rad=.0', **args)
ax_env1.add_artist(con)
figure.text(0.43, 1.4, 'A', transform=ax_sti1.transAxes, size=10)
con = ConnectionPatch(
    xyA=(0.5, 1), coordsA=ax_sti2.transAxes,
    xyB=(0.5, -0.1), coordsB=ax_env2.transAxes,
    connectionstyle='arc3,rad=.0', **args)
ax_env2.add_artist(con)
figure.text(0.43, 1.4, 'A', transform=ax_sti2.transAxes, size=10)

figure.savefig(DST / 'Deconvolution.pdf')
figure.savefig(DST / 'Deconvolution.png')
