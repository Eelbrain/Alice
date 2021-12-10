# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: tutorialAlice
#     language: python
#     name: tutorialalice
# ---

# +
from pathlib import Path
import re

import eelbrain
import mne
import trftools
# -

STIMULI = [str(i) for i in range(1, 13)]
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
EEG_DIR = DATA_ROOT / 'eeg'
SUBJECTS = [path.name for path in EEG_DIR.iterdir() if re.match(r'S\d*', path.name)]
# Define a target directory for epoched data and make sure the directory is created
EPOCH_DIR = DATA_ROOT / 'Epochs'
EPOCH_DIR.mkdir(exist_ok=True)

tstart = -0.1
tstop = 1

# +
# Load stimuli
# ------------
# Make sure to name the stimuli so that the TRFs can later be distinguished
# Load the gammatone-spectrograms; use the time axis of these as reference
gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]

subtlex_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~subtlex.pickle') for stimulus in STIMULI]
word_surprisal = [eelbrain.event_impulse_predictor(gt.time, value='surprisal_5gram', ds=ds, name='surprisal') for gt, ds in zip(gammatone, subtlex_tables)]

word_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~word.pickle') for stimulus in STIMULI]
word_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='word') for gt, ds in zip(gammatone, word_tables)]
# Get syntactical surprisal values 
word_NGRAM = [eelbrain.event_impulse_predictor(gt.time, value='NGRAM', ds=ds, name='ngram') for gt, ds in zip(gammatone, word_tables)]
word_RNN = [eelbrain.event_impulse_predictor(gt.time, value='RNN', ds=ds, name='rnn') for gt, ds in zip(gammatone, word_tables)]
word_CFG = [eelbrain.event_impulse_predictor(gt.time, value='CFG', ds=ds, name='cfg') for gt, ds in zip(gammatone, word_tables)]

word_lexical = [eelbrain.event_impulse_predictor(gt.time, value='lexical', ds=ds, name='lexical') for gt, ds in zip(gammatone, word_tables)]
# -

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]

# Loop through subjects to get the epoched data
for subject in SUBJECTS:
    subject_epoch_dir = EPOCH_DIR / subject
    subject_epoch_dir.mkdir(exist_ok=True)
    # Generate all epoch paths so we can check whether any new TRFs need to be estimated
    epoch_path = subject_epoch_dir / f'{subject}_epoched_word.pickle'
    # Skip this subject if all files already exist
    if epoch_path.exists():
        continue
    
    # Load the EEG data
    raw = mne.io.read_raw(EEG_DIR / subject / f'{subject}_alice-raw.fif', preload=True)
    # Band-pass filter the raw data between 0.2 and 20 Hz
    raw.filter(0.5, 20)
    # Interpolate bad channels
    raw.interpolate_bads()
    # Extract the events marking the stimulus presentation from the EEG file
    events = eelbrain.load.fiff.events(raw)
    # Not all subjects have all trials; determine which stimuli are present
    trial_indexes = [STIMULI.index(stimulus) for stimulus in events['event']]
    # Extract the EEG data segments corresponding to the stimuli
    trial_durations = [durations[i] for i in trial_indexes]
    eeg = eelbrain.load.fiff.variable_length_epochs(events, -0.100, trial_durations, decim=5, connectivity='auto')
    # Since trials are of unequal length, we will concatenate them for the TRF estimation.
    eeg_concatenated = eelbrain.concatenate(eeg)
    
    # Get corresponding word onsets
    word_onset_predictor = eelbrain.concatenate([word_onsets[i] for i in trial_indexes])
    word_surprisal_predictor = eelbrain.concatenate([word_surprisal[i] for i in trial_indexes])
    
    rnn_predictor = eelbrain.concatenate([word_RNN[i] for i in trial_indexes])
    cfg_predictor = eelbrain.concatenate([word_CFG[i] for i in trial_indexes])
    ngram_preditor = eelbrain.concatenate([word_NGRAM[i] for i in trial_indexes])
    word_surprisal_predictor = eelbrain.concatenate([word_surprisal[i] for i in trial_indexes])
    
    content_word_predictor = eelbrain.concatenate([word_lexical[i] for i in trial_indexes])
    
    if word_onset_predictor.time.tstop > eeg_concatenated.time.tstop: 
        print('error with subject %s' % subject)

    # Make epoched data
    rows = []
    for trial_idx, onset_time in enumerate(word_onset_predictor.flatnonzero()):
        
        # remark: tstart is negative!
        if onset_time + tstart < 0: 
            continue 
            
        if onset_time + tstop > eeg_concatenated.time.tstop: 
            continue
        
        current_epoch = eeg_concatenated.sub(time=(onset_time+tstart, onset_time+tstop))
        # change dimension (tmin to tstart)
        current_epoch = eelbrain.set_tmin(current_epoch, tmin = tstart)
        rows.append([current_epoch, word_surprisal_predictor.sub(time=onset_time), trial_idx + 1, 
                    rnn_predictor.sub(time=onset_time), ngram_preditor.sub(time=onset_time), cfg_predictor.sub(time=onset_time), 
                    content_word_predictor.sub(time=onset_time)])

    column_names = ['eeg', 'surprisal', 'trial_idx', 'rnn','ngram','cfg', 'content_word']
    ds = eelbrain.Dataset.from_caselist(column_names, rows)
    eelbrain.save.pickle(ds, epoch_path)


