"""Generate high-resolution gammatone spectrograms"""
from pathlib import Path

from eelbrain import *
from trftools import gammatone_bank


DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'

for i in range(1, 13):
    dst = STIMULUS_DIR / f'{i}-gammatone.pickle'
    if dst.exists():
        continue
    wav = load.wav(STIMULUS_DIR / f'{i}.wav')
    gt = gammatone_bank(wav, 20, 5000, 256, location='left', pad=False)
    gt = resample(gt, 1000)
    save.pickle(gt, dst)
