"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
from pathlib import Path

import eelbrain


DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

word_table = eelbrain.load.tsv(STIMULUS_DIR / 'AliceChapterOne-EEG.csv')
# Add word frequency as variable that scales with the expected response: larger response for less frequent words
word_table['InvLogFreq'] = 17 - word_table['LogFreq']

for segment in range(1, 13):
    segment_table = word_table.sub(f"Segment == {segment}")
    ds = eelbrain.Dataset({'time': segment_table['onset']}, info={'tstop': segment_table[-1, 'offset']})
    # add predictor variables
    ds['LogFreq'] = segment_table['InvLogFreq']
    for key in ['NGRAM', 'RNN', 'CFG', 'Position']:
        ds[key] = segment_table[key]
    # create masks for lexical and non-lexical words
    ds['lexical'] = segment_table['IsLexical'] == True
    ds['nlexical'] = segment_table['IsLexical'] == False
    # save
    eelbrain.save.pickle(ds, PREDICTOR_DIR / f'{segment}~word.pickle')
