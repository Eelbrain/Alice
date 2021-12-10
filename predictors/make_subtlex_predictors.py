"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
from pathlib import Path

import eelbrain


DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

word_table = eelbrain.load.tsv(STIMULUS_DIR / 'subtlex_table.txt')

for segment in range(1, 13):
    segment_table = word_table.sub(f"Segment == {segment}")
    ds = eelbrain.Dataset({'time': segment_table['onset']})
    # add predictor variables
    for key in ['surprisal_5gram', 'surprisal_1gram']:
        ds[key] = segment_table[key]

    # save
    eelbrain.save.pickle(ds, PREDICTOR_DIR / f'{segment}~subtlex.pickle')
