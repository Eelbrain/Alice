"""Generate high-resolution gammatone spectrograms"""
from pathlib import Path

import eelbrain


DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'

for i in range(1, 13):
    dst = STIMULUS_DIR / f'{i}-gammatone.pickle'
    if dst.exists():
        continue
    wav = eelbrain.load.wav(STIMULUS_DIR / f'{i}.wav')
    gt = eelbrain.gammatone_bank(wav, 20, 5000, 256, location='left', integration_window=0.010, tstep=0.001)
    eelbrain.save.pickle(gt, dst)
