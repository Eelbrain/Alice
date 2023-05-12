"""
Estimate deconvolution of the spectrogram depending on function and content words. Keep results for each test-partition, so as to be able to statistically assess the models.
"""
from pathlib import Path

import eelbrain


# Data locations
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Alice'
PREDICTOR_DIR = DATA_ROOT / 'predictors'
TRF_DIR = DATA_ROOT / 'TRFs'

# Collect stimulus data from all trials
gammatone_trials = []
word_trials = []
lexical_trials = []
non_lexical_trials = []
# loop through trials to load all stimuli
for trial in range(1, 13):
    gammatone = eelbrain.load.unpickle(PREDICTOR_DIR / f'{trial}~gammatone-8.pickle')
    gammatone = gammatone.bin(0.01)
    events = eelbrain.load.unpickle(PREDICTOR_DIR / f'{trial}~word.pickle')
    # turn categorial predictors into time-series matching the spectrogram
    word = eelbrain.event_impulse_predictor(gammatone, time='time', value=1, data=events, name='word')
    lexical = eelbrain.event_impulse_predictor(gammatone, time='time', value='lexical', data=events, name='lexical')
    non_lexical = eelbrain.event_impulse_predictor(gammatone, time='time', value='nlexical', data=events, name='non_lexical')
    # store ndvars in lists
    gammatone_trials.append(gammatone)
    word_trials.append(word)
    lexical_trials.append(lexical)
    non_lexical_trials.append(non_lexical)
# concatenate trials
gammatone = eelbrain.concatenate(gammatone_trials)
word = eelbrain.concatenate(word_trials)
lexical = eelbrain.concatenate(lexical_trials)
non_lexical = eelbrain.concatenate(non_lexical_trials)

trf_word = eelbrain.boosting(gammatone, word, -0.100, 1.001, partitions=15, partition_results=True, test=True)
eelbrain.save.pickle(trf_word, TRF_DIR / 'gammatone~word.pickle')

trf_lexical = eelbrain.boosting(gammatone, [word, lexical, non_lexical], -0.100, 1.001, partitions=15, partition_results=True, test=True)
eelbrain.save.pickle(trf_lexical, TRF_DIR / 'gammatone~word+lexical.pickle')
