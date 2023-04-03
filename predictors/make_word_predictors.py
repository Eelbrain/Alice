"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
from pathlib import Path

import eelbrain


# Define paths to source data, and destination for predictors
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
STIMULUS_DIR = DATA_ROOT / 'stimuli'
PREDICTOR_DIR = DATA_ROOT / 'predictors'

# Load the text file with word-by-word predictor variables
word_table = eelbrain.load.tsv(STIMULUS_DIR / 'AliceChapterOne-EEG.csv')
# Add word frequency as variable that scales with the expected response (in
# impulse-based continuous predictor variables, impulses quantify the difference
# from the baseline, 0, i.e. larger magnitude impulses always predict larger
# magnitude of responses; however, based on previous research, we expect larger
# responses to less frequent words)
word_table['InvLogFreq'] = 17 - word_table['LogFreq']

# Loop through the stimuli
for segment in range(1, 13):
    # Take the subset of the table corresponding to the current stimulus
    segment_table = word_table.sub(f"Segment == {segment}")
    # Initialize a new Dataset with just the time-stamp of the words; add an
    # info dictionary with the duration of the stimulus ('tstop')
    ds = eelbrain.Dataset({'time': segment_table['onset']}, info={'tstop': segment_table[-1, 'offset']})
    # Add predictor variables to the new Dataset
    ds['LogFreq'] = segment_table['InvLogFreq']
    for key in ['NGRAM', 'RNN', 'CFG', 'Position']:
        ds[key] = segment_table[key]
    # Create and add boolean masks for lexical and non-lexical words
    ds['lexical'] = segment_table['IsLexical'] == True
    ds['nlexical'] = segment_table['IsLexical'] == False
    # Save the Dataset for this stimulus
    eelbrain.save.pickle(ds, PREDICTOR_DIR / f'{segment}~word.pickle')
