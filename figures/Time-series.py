# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from pathlib import Path

import eelbrain
from matplotlib import pyplot
import mne
import trftools


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
SUBJECT = 'S01'
STIMULUS = '1'
# Crop the data for this demonstration 
TSTOP = 5.001  # Python indexing is exclusive of the specified stop sample 

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
}
pyplot.rcParams.update(RC)
# -

# Create a raw object for the EEG data - does not yet load the data itself 
raw = mne.io.read_raw_fif(DATA_ROOT / 'eeg' / SUBJECT / f'{SUBJECT}_alice-raw.fif')
# Extract only the events form the EEG data, and show the events table
events = eelbrain.load.fiff.events(raw)
events.head()

# For this illustration, pick only the first stimulus
events = events.sub(f"event == {STIMULUS!r}")
# Load the EEG data corresponding to this event; load only the first 5 seconds of data here
eeg = eelbrain.load.fiff.epochs(events, tmin=0, tstop=TSTOP)
# Filter the data and resample it to 100 Hz
eeg = eelbrain.filter_data(eeg, 1, 20)
eeg = eelbrain.resample(eeg, 100)

# +
# Load the stimulus wave file
wave = eelbrain.load.wav(STIMULUS_DIR / f'{STIMULUS}.wav')
# Load the high-resolution gammatone spectrogram from disk (run predictors/make_gammatone.py to generate this file)
gammatone = eelbrain.load.unpickle(STIMULUS_DIR / f'{STIMULUS}-gammatone.pickle')
# Remove artifacts and apply a log transform to simulate compression in the auditory systems
gammatone = gammatone.clip(0)
gammatone = (gammatone + 1).log()

# Crop the data
wave = wave.sub(time=(0, TSTOP))
gammatone = gammatone.sub(time=(0, TSTOP))
# -

# For discrete events, load the word table
word_table = eelbrain.load.tsv(STIMULUS_DIR / 'AliceChapterOne-EEG.csv')
# Restrict to the stimulus
word_table = word_table.sub(f"Segment == {STIMULUS}")
# As with the sounds, remove words that occured after TSTOP 
word_table = word_table.sub(f"onset < {TSTOP}")
word_table.head()

# +
# Setup the figure layout
fig, axes = pyplot.subplots(9, figsize=(7.5, 7), sharex=True, subplot_kw=dict(frame_on=False))

# Plot the EEG data
eelbrain.plot.Butterfly(eeg, axes=axes[0], ylabel='EEG')

# plot the sound wave; downsample it first to make plotting faster
wave_rs = eelbrain.resample(wave, 1000)
eelbrain.plot.UTS(wave_rs, colors='k', axes=axes[1], ylabel='Sound\nWave')

# plot the full gammatone spectrogram
eelbrain.plot.Array(gammatone, axes=axes[2], ylabel='Gammatone\nSpectrogram', vmax=25)

# Generate an 8-band version of the spectrogram by averaging in frequency bins
gammatone_8 = gammatone.bin(dim='frequency', nbins=8)
# Resample it to match the EEG sampling rate
gammatone_8 = eelbrain.resample(gammatone_8, 100)
eelbrain.plot.Array(gammatone_8, axes=axes[3], ylabel='Gammatone\n8 Bands', interpolation='none', vmax=25)

# Generate an envelope representation by summing across all frequency bands
gammatone_envelope = gammatone.sum('frequency')
gammatone_envelope = eelbrain.resample(gammatone_envelope, 100)
eelbrain.plot.UTS(gammatone_envelope, colors='red', axes=axes[4], ylabel='Gammatone\nEnvelope')

# Use an edge detector model to extract acoustic onsets from the gammatone spectrogram
gammatone_onsets = trftools.neural.edge_detector(gammatone, 30)
eelbrain.plot.Array(gammatone_onsets, axes=axes[5], ylabel='Acoustic\nOnsets', interpolation='none')

# Generate an impulse at every word onset. Use a time dimension with the same properties as the EEG data.
eeg_time = eelbrain.UTS(0, 1/100, 501)
words = eelbrain.NDVar.zeros(eeg_time)
for time in word_table['onset']:
    words[time] = 1
eelbrain.plot.UTS(words, stem=True, top=1.5, bottom=-0.5, colors='blue', axes=axes[6], ylabel='Words')

# For illustration, add the words to the plot
for time, word in word_table.zip('onset', 'Word'):
    axes[6].text(time, -0.1, word, va='top', fontsize=8)

# Generate an impulse at every word onset, scaled by a variable
ngram_surprisal = eelbrain.NDVar.zeros(eeg_time)
for time, value in word_table.zip('onset', 'NGRAM'):
    ngram_surprisal[time] = value
eelbrain.plot.UTS(ngram_surprisal, stem=True, colors='blue', axes=axes[7], ylabel='N-Gram')

# Generate an alternative impulse, only at content word onsets, and scaled by a different variable
cfg_surprisal = eelbrain.NDVar.zeros(eeg_time)
for time, value, is_lexical in word_table.zip('onset', 'NGRAM', 'IsLexical'):
    if is_lexical:
        cfg_surprisal[time] = value
eelbrain.plot.UTS(cfg_surprisal, stem=True, colors='blue', axes=axes[8], ylabel='N-Gram\nLexical')

# Fine-tune layout
LAST = 8
for i, ax in enumerate(axes):
    if i == LAST:
        pass
    else:
        ax.set_xlabel(None)
    ax.set_yticks(())
    ax.tick_params(bottom=False)

fig.tight_layout()
fig.savefig(DST / 'Time series.pdf')
fig.savefig(DST / 'Time series.png')
