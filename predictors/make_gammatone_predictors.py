"""Predictors based on gammatone spectrograms"""
from pathlib import Path

import numpy as np
import eelbrain

DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

PREDICTOR_DIR.mkdir(exist_ok=True)
for i in range(1, 13):
    gt = eelbrain.load.unpickle(STIMULUS_DIR / f'{i}-gammatone.pickle')

    # Remove resampling artifacts
    gt = gt.clip(0, out=gt)
    # apply log transform
    gt = (gt + 1).log()
    # generate onset detector model
    gt_on = eelbrain.edge_detector(gt, c=30)

    # 1 band predictors
    eelbrain.save.pickle(gt.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-1.pickle')
    eelbrain.save.pickle(gt_on.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-on-1.pickle')
    # 8 band predictors
    x = gt.bin(nbins=8, func=np.sum, dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-8.pickle')
    x = gt_on.bin(nbins=8, func=np.sum, dim='frequency')
    eelbrain.save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-on-8.pickle')
