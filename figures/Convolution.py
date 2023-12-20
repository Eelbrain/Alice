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

# # Convolution model figure

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
from matplotlib.patches import ConnectionPatch


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'

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
response_time = eelbrain.UTS(0, 0.010, 80)
response = eelbrain.gaussian(0.150, 0.040, response_time) - eelbrain.gaussian(0.350, 0.100, response_time) * .6 + eelbrain.gaussian(0.400, 0.200, response_time) * .1
response_2 = eelbrain.gaussian(0.150, 0.050, response_time)

time = eelbrain.UTS(0, 0.010, 5*100)
recording = eelbrain.NDVar.zeros(time)
event_times = [0.2, 1.1, 2.0, 3.2, 4.1]
impulse_times = [0.2, 1.1, 2.0, 3.2, 3.4, 3.7, 4.1, 4.2, 4.3]
impulse_value = [1.0, 0.6, 0.4, 1.2, 0.9, 0.3, 1.1, 0.6, 1.1]
for t in event_times:
    recording[t:t+0.800] += response

stimulus = eelbrain.NDVar.zeros(time)
for t, v in zip(impulse_times, impulse_value):
    stimulus[t] = v
    
wav = eelbrain.load.wav(STIMULUS_DIR / '1.wav')
wav_envelope = wav.sub(time=(0, 5)).envelope()
stimulus_envelope = eelbrain.resample(eelbrain.filter_data(wav_envelope, 0, 10), 100)
stimulus_envelope *= 4e-5

# +
# initialize figure
figure = pyplot.figure(figsize=(7.5, 7), facecolor='w')
shape = (11, 8)
ax_args = dict(frame_on=False)
uts_args = dict(xlabel=False, yticklabels='none', ylabel=False, clip=False)

def decorate(ax):
    ax.set_yticks(())
    ax.set_xticks(())
    ax.tick_params(bottom=False)
    ax.set_clip_on(False)

# Average-based
ax = pyplot.subplot2grid(shape, (0, 0), colspan=7, **ax_args)
ax.set_title('A) Traditional average model: response at discrete time points', loc='left', size=10)
eelbrain.plot.UTS(recording, axes=ax, **uts_args)
for t in event_times:
    box = pyplot.Rectangle((t, -1), 0.800, 2.2, ec='k', fill=False, alpha=0.5)
    ax.add_artist(box)
    ax.arrow(t+0.050, -1.8, 0, 1, color='b', head_width=0.05, head_length=0.5, clip_on=False)
decorate(ax)
ax.set_ylim(-1.1, 1.3)
# Average
ax = pyplot.subplot2grid(shape, (0, 7), **ax_args)
eelbrain.plot.UTS(response, axes=ax, **uts_args)
ax.set_title('Average')
decorate(ax)

# TRF impulse
ax_b1 = ax = pyplot.subplot2grid(shape, (2, 0), colspan=7, **ax_args)
ax.set_title('B) TRF to discrete events: each impulse elicits a response', loc='left', size=10)
eelbrain.plot.UTS(stimulus, axes=ax, colors='b', stem=True, **uts_args)
decorate(ax)
# TRF
ax = pyplot.subplot2grid(shape, (2, 7), **ax_args)
eelbrain.plot.UTS(response, axes=ax, **uts_args)
ax.set_title('TRF 1')
decorate(ax)
# TRF response
response_impulse = eelbrain.convolve(response, stimulus)
ax_b2 = ax = pyplot.subplot2grid(shape, (3, 0), colspan=7, **ax_args)
eelbrain.plot.UTS(response_impulse, axes=ax, **uts_args)
decorate(ax)

# TRF continuous
ax_c1 = ax = pyplot.subplot2grid(shape, (5, 0), colspan=7, **ax_args)
ax.set_title('C) TRF with a continuous predictor: each time point elicits a response', loc='left', size=10)
plot = eelbrain.plot.UTS(stimulus_envelope, axes=ax, colors='b', **uts_args)
decorate(ax)
stimulus_handle = plot.plots[0].legend_handles['1.wav']
# TRF
ax = pyplot.subplot2grid(shape, (5, 7), **ax_args)
eelbrain.plot.UTS(response, axes=ax, **uts_args)
ax.set_title('TRF 2')
decorate(ax)
# TRF response
response_continuous = eelbrain.convolve(response, stimulus_envelope, name='response')
ax_c2 = ax = pyplot.subplot2grid(shape, (6, 0), colspan=7, **ax_args)
plot = eelbrain.plot.UTS(response_continuous, axes=ax, **uts_args)
decorate(ax)
response_handle = plot.plots[0].legend_handles['response']

# mTRF: continuous stimulus
style = eelbrain.plot.Style('C1', linestyle='--')
trf_response = eelbrain.convolve(response, stimulus_envelope, name='response')
ax = pyplot.subplot2grid(shape, (8, 0), colspan=7, **ax_args)
ax.set_title('D) mTRF: simultaneous additive responses to multiple predictors', loc='left', size=10)
eelbrain.plot.UTS(stimulus_envelope, axes=ax, colors='b', **uts_args)
plot = eelbrain.plot.UTS(trf_response * .1, axes=ax, colors=style, **uts_args)
decorate(ax)
partial_response_handle = plot.plots[0].legend_handles['response * 0.1']
# TRF
ax = pyplot.subplot2grid(shape, (8, 7), **ax_args)
eelbrain.plot.UTS(response, axes=ax, **uts_args)
ax.set_title('mTRF')
decorate(ax)
# mTRF: impulse stimulus
trf_response_2 = eelbrain.convolve(response_2, stimulus)
ax = pyplot.subplot2grid(shape, (9, 0), colspan=7, **ax_args)
eelbrain.plot.UTS(stimulus, axes=ax, colors='b', stem=True, **uts_args)
eelbrain.plot.UTS(trf_response_2, axes=ax, colors=style, **uts_args)
decorate(ax)
# TRF
ax = pyplot.subplot2grid(shape, (9, 7), **ax_args)
eelbrain.plot.UTS(response_2, axes=ax, **uts_args)
decorate(ax)
# TRF response
trf_response = trf_response + trf_response_2
ax = pyplot.subplot2grid(shape, (10, 0), colspan=7, **ax_args)
eelbrain.plot.UTS(trf_response, axes=ax, **uts_args)
decorate(ax)

#add legend
handles = [stimulus_handle, response_handle, partial_response_handle]
labels = ['Stimulus', 'Response', 'Partial response']
pyplot.figlegend(handles, labels, loc='lower right')

pyplot.tight_layout()
ax_b2.set_ylim(0, 1)

# Arrows
args = dict(color='0.5', arrowstyle="->, head_width=0.25, head_length=0.5", linestyle=':')
for t in impulse_times:
    con = ConnectionPatch(
        xyA=(t, -0.2), coordsA=ax_b1.transData,
        xyB=(t, response_impulse[t] + 0.2), coordsB=ax_b2.transData,
        **args)
    ax_b1.add_artist(con)
t = 0.01
con = ConnectionPatch(
    xyA=(t, stimulus_envelope[t] - 0.08), coordsA=ax_c1.transData,
    xyB=(t, response_continuous[t] + 0.5), coordsB=ax_c2.transData,
    **args)
ax_c1.add_artist(con)
ax_c1.text(t+0.04, -0.3, '...', ha='left', color='0.5', size=12)

figure.savefig(DST / 'Convolution.pdf')
eelbrain.plot.figure_outline()
