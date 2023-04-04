"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
import os
from pathlib import Path

import eelbrain


tempfile = os.path.realpath(os.path.join(__file__, '..',
                                         '..', ".temppath.pickled"))
DATA_ROOT = Path(eelbrain.load.unpickle(tempfile))
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
    data = eelbrain.Dataset({'time': segment_table['onset']}, info={'tstop': segment_table[-1, 'offset']})
    # Add predictor variables to the new Dataset
    data['LogFreq'] = segment_table['InvLogFreq']
    for key in ['NGRAM', 'RNN', 'CFG', 'Position']:
        data[key] = segment_table[key]
    # Create and add boolean masks for lexical and non-lexical words
    data['lexical'] = segment_table['IsLexical'] == True
    data['nlexical'] = segment_table['IsLexical'] == False
    # Save the Dataset for this stimulus
    eelbrain.save.pickle(data, PREDICTOR_DIR / f'{segment}~word.pickle')
