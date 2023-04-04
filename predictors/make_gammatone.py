"""Generate high-resolution gammatone spectrograms"""
import os
from pathlib import Path

import eelbrain


# Define paths to data
tempfile = os.path.realpath(os.path.join(__file__, '..',
                                         '..', ".temppath.pickled"))
DATA_ROOT = Path(eelbrain.load.unpickle(tempfile))
STIMULUS_DIR = DATA_ROOT / 'stimuli'

# Loop through the stimuli
for i in range(1, 13):
    # Define a filename for the gammatone spectrogram corresponding to this predictor.
    dst = STIMULUS_DIR / f'{i}-gammatone.pickle'
    # If the file already exists, we can skip it
    if dst.exists():
        continue
    # Load the sound file corresponding to the predictor
    wav = eelbrain.load.wav(STIMULUS_DIR / f'{i}.wav')
    # Apply a gammatone filterbank, producing a high resolution spectrogram
    gt = eelbrain.gammatone_bank(wav, 80, 15000, 128, location='left', tstep=0.001)
    # Save the gammatone spectrogram at the intended destination
    eelbrain.save.pickle(gt, dst)
