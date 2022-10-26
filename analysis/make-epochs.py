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
from operator import itemgetter 

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
# load word information (including the onset times)
word_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~word.pickle') for stimulus in STIMULI]
durations = [word_table['time'][-1]+tstop for word_table in word_tables]
word_onsets = [word_table['time'] for word_table in word_tables]
# -

# Loop through subjects to get the epoched data
for subject in SUBJECTS:
    subject_epoch_dir = EPOCH_DIR / subject
    subject_epoch_dir.mkdir(exist_ok=True)
    # Generate epoch path so we can check whether it already exists
    epoch_path = subject_epoch_dir / f'{subject}_erp_word.pickle'
    # Skip this subject if the file already exists
    if epoch_path.exists():
        continue
    
    # Load the EEG data
    raw = mne.io.read_raw(EEG_DIR / subject / f'{subject}_alice-raw.fif', preload=True)
    # Band-pass filter the raw data between 0.5 and 20 Hz
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
    current_word_onsets = [word_onsets[i] for i in trial_indexes]

    # Make epoched data
    rows = []
    for eeg_segment, matched_word_onsets in zip(eeg, current_word_onsets):
        for onset_time in matched_word_onsets:
            # Note: tstart is negative!
            if onset_time + tstart < 0: 
                continue 
                
            if onset_time + tstop > eeg_segment.time.tstop: 
                continue
            
            current_epoch = eeg_segment.sub(time=(onset_time+tstart, onset_time+tstop))
            # Update the epoch's time to be relative to word onset
            current_epoch = eelbrain.set_tmin(current_epoch, tmin=tstart)
            rows.append([current_epoch])

    column_names = ['eeg']
    ds = eelbrain.Dataset.from_caselist(column_names, rows)

    # take average
    ERP = ds['eeg'].mean('case')
    # do baseline correction: subtract from each sensor its average
    baseline_corrected_ERP = ERP - ERP.mean('time')
    eelbrain.save.pickle(baseline_corrected_ERP, epoch_path)


