"""This script estimates TRFs for comparing different basis windows"""
from pathlib import Path
import re

import eelbrain
import mne


STIMULI = [str(i) for i in range(1, 13)]
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
EEG_DIR = DATA_ROOT / 'eeg'
SUBJECTS = [path.name for path in EEG_DIR.iterdir() if re.match(r'S\d*', path.name)]
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
gammatone = [eelbrain.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]
# Filter the predictor with the same parameters as we will filter the EEG data
gammatone = [eelbrain.filter_data(x, 0.5, 20) for x in gammatone]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, gammatone)]

# Models
# ------
# Pre-define models here to have easier access during estimation. In the future, additional models could be added here and the script re-run to generate additional TRFs.
model, predictors = 'gammatone', [gammatone]
basis_values = [0, 0.050, 0.100]

# Estimate TRFs
# -------------
# Loop through subjects to estimate TRFs
for subject in SUBJECTS:
    subject_trf_dir = TRF_DIR / subject
    subject_trf_dir.mkdir(exist_ok=True)
    # Generate all TRF paths so we can check whether any new TRFs need to be estimated
    trf_paths = {basis: subject_trf_dir / f'{subject} {model} basis-{basis*1000:.0f}.pickle' for basis in basis_values}
    # Skip this subject if all files already exist
    if all(path.exists() for path in trf_paths.values()):
        continue
    # Load the EEG data
    raw = mne.io.read_raw(EEG_DIR / subject / f'{subject}_alice-raw.fif', preload=True)
    # Band-pass filter the raw data between 0.5 and 20 Hz
    raw.filter(0.5, 20, n_jobs=1)
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
    # Select and concetenate the predictors corresponding to the EEG trials
    predictors_concatenated = []
    for predictor in predictors:
        predictors_concatenated.append(eelbrain.concatenate([predictor[i] for i in trial_indexes]))

    for basis, path in trf_paths.items():
        # Skip if this file already exists
        if path.exists():
            continue
        print(f"Estimating: {subject} ~ {basis}")
        # Fit the mTRF
        trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated, -0.100, 1.000, error='l1', basis=basis, partitions=5, test=1, selective_stopping=True)
        eelbrain.save.pickle(trf, path)
