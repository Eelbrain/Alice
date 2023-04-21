"""
Predictors based on gammatone spectrograms

Assumes that ``make_gammatone.py`` has been run to create the high resolution
spectrograms.
"""
from pathlib import Path

import eelbrain


# Define paths to data, and destination for predictors
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

# If the directory for predictors does not exist yet, create it
PREDICTOR_DIR.mkdir(exist_ok=True)
# Loop through stimuli
for i in range(1, 13):
    # Load the high resolution gammatone spectrogram
    gt = eelbrain.load.unpickle(STIMULUS_DIR / f'{i}-gammatone.pickle')

    # Apply a log transform to approximate peripheral auditory processing
    gt_log = (gt + 1).log()
    # Apply the edge detector model to generate an acoustic onset spectrogram
    gt_on = eelbrain.edge_detector(gt_log, c=30)

    # Create and save 1 band versions of the two predictors (i.e., temporal envelope predictors)
    eelbrain.save.pickle(gt_log.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-1.pickle')
    eelbrain.save.pickle(gt_on.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-on-1.pickle')
    # Create and save 8 band versions of the two predictors (binning the frequency axis into 8 bands)
    x = gt_log.bin(nbins=8, func='sum', dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-8.pickle')
    x = gt_on.bin(nbins=8, func='sum', dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-on-8.pickle')

    # Create gammatone spectrograms with linear scale, only 8 bin versions
    x = gt.bin(nbins=8, func='sum', dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-lin-8.pickle')
    # Powerlaw scale
    gt_pow = gt ** 0.6
    x = gt_pow.bin(nbins=8, func='sum', dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-pow-8.pickle')
