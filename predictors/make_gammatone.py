"""Generate high-resolution gammatone spectrograms"""
import os
from pathlib import Path

from eelbrain import *
from trftools import gammatone_bank


tempfile = os.path.realpath(os.path.join(__file__, '..',
                                         '..', ".temppath.pickled"))
DATA_ROOT = Path(load.unpickle(tempfile))
STIMULUS_DIR = DATA_ROOT / 'stimuli'

for i in range(1, 13):
    dst = STIMULUS_DIR / f'{i}-gammatone.pickle'
    if dst.exists():
        continue
    wav = load.wav(STIMULUS_DIR / f'{i}.wav')
    gt = gammatone_bank(wav, 20, 5000, 256, location='left', pad=False, tstep=0.001)
    save.pickle(gt, dst)
