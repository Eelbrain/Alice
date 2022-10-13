"""This script estimates TRFs for several models and saves them"""
from pathlib import Path
import re

import eelbrain
import mne
import trftools
import numpy as np


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
# Load the broad-band envelope 
envelope = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-1.pickle') for stimulus in STIMULI]
envelope = [x.bin(0.01, dim='time', label='start') for x in envelope]
envelope = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='envelope') for x in envelope]

# Extract the duration of the stimuli, so we can later match the EEG to the stimuli
durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, envelope)]

# Models
# ------
# Pre-define models here to have easier access during estimation. In the future, additional models could be added here and the script re-run to generate additional TRFs.
models = {
    # Acoustic models
    'envelope': [envelope],
}

# Estimate TRFs
# -------------
# Loop through subjects to estimate TRFs
for subject in SUBJECTS:
    subject_trf_dir = TRF_DIR / subject
    subject_trf_dir.mkdir(exist_ok=True)

    for reference in ['cz', 'average']:
        # Generate all TRF paths so we can check whether any new TRFs need to be estimated
        trf_paths = {model: subject_trf_dir / f'{subject} {model}_{reference}.pickle' for model in models}
        # Skip this subject if all files already exist
        if all(path.exists() for path in trf_paths.values()):
            continue
        # Load the EEG data
        raw = mne.io.read_raw(EEG_DIR / subject / f'{subject}_alice-raw_{reference}.fif', preload=True)
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
        # Since trials are of unequal length, we will concatenate them for the TRF estimation.
        eeg_concatenated = eelbrain.concatenate(eeg)

        if reference in ['cz']: 
            # As the Cz-channel was used for reference, the channel contains zeros (which cannot be used for TRF estimation)
            # Therefore, this channel is replaced with random noise to preserve the 64-sensor dimension. 
            Cz_location = [label_idx for label_idx, label in enumerate(eeg_concatenated.sensor.names) if label == '33']
            eeg_concatenated.x[Cz_location,] = np.random.randn(eeg_concatenated[Cz_location,].x.shape[0], eeg_concatenated[Cz_location,].x.shape[1])*np.mean(eeg_concatenated.x)

        for model, predictors in models.items():
            path = trf_paths[model]
            # Skip if this file already exists
            if path.exists():
                continue
            print(f"Estimating: {subject} ~ {model} ~ {reference}")
            # Select and concetenate the predictors corresponding to the EEG trials
            predictors_concatenated = []
            for predictor in predictors:
                predictors_concatenated.append(eelbrain.concatenate([predictor[i] for i in trial_indexes]))
            # Fit the mTRF
            trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated, -0.100, 1.000, error='l1', basis=0.050, partitions=5, test=1, selective_stopping=True)
            # Save the TRF for later analysis
            eelbrain.save.pickle(trf, path)
