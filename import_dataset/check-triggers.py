# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Check alignments
# Check alignemnts of stimuli with the EEG data. The EEG recording contains a record of the acoustic stimulus, which can be compare with the stimulus itself. This loads the events through the pipeline in `alice.py`, i.e. the trigger correction is already applied and all subjects should have the correct alignment.

# +
# %matplotlib inline
from eelbrain import *

from alice import alice


# load the acoustic envelope predictor for each stimulus
gt = {f'{i}': alice.load_predictor(f'{i}~gammatone-1', 0.002, 1000, name='WAV') for i in range(1, 13)}
for y in gt.values():
    y /= y.std()
# -

for subject in alice:
    events = alice.load_events(raw='raw', data_raw=True)
    raw = events.info['raw']
    raw.load_data()
    # S16, S22 have broken AUX channels
    if subject in ['S05', 'S38']:
        continue  # no AUD channel
    for name in ['AUD', 'Aux5']:
        if name in raw.ch_names:
            ch = raw.ch_names.index(name)
            break
    else:
        print(subject, raw.ch_names)
        raise
    xs = []
    # extract audio from EEG
    for segment, i0 in events.zip('event', 'i_start'):
        x = NDVar(raw._data[ch, i0:i0+1000], UTS(0, 0.002, 1000), name='EEG')
        x -= x.min()
        x /= x.std()
        xs.append([x, gt[segment]])
    p = plot.UTS(xs, axh=2, w=10, ncol=1, title=subject, axtitle=events['trigger'])
    # display and close to avoid having too many open figures
    display(p)
    p.close()

