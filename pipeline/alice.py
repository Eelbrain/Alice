# This file contains a pipeline specification. The pipeline is defined as a subclass of TRFExperiment, and adds information about the specific paradigm and data. TRFExperiment is a subclass of eelbrain.MneExperiment, which is documented extensively on the Eelbrain website.
from eelbrain.pipeline import *
from trftools.pipeline import *


# This is the root directory where the pipeline expects the experiment's data. The directory used here corresponds to the default download location when using the download_alic.py script in this repository. Generally, the file locations used in the example analysis scripts is consistent with the location and naming convention for this pipeline.
DATA_ROOT = "~/Data/Alice"
# Since each of the audio files used as stimuli had a different length, we define that information here
SEGMENT_DURATION = {
    '1': 57.541,
    '2': 60.845,
    '3': 63.259,
    '4': 69.989,
    '5': 66.273,
    '6': 63.778,
    '7': 62.897,
    '8': 57.311,
    '9': 57.226,
    '10': 61.27,
    '11': 56.17,
    '12': 46.983,
}

# One may also want to define parameters used for estimating TRFs here, as they are often re-used along with the pipeline in multiple location (see notebooks in this directory and jobs.py for examples)
PARAMETERS = {
    'raw': '0.5-20',
    'samplingrate': 50,
    'data': 'eeg',
    'tstart': -0.100,
    'tstop': 1.00,
    'filter_x': 'continuous',
    'error': 'l1',
    'basis': 0.050,
    'partitions': -5,
    'selective_stopping': 1,
}


# Start by defining a subclass of the pipeline
class Alice(TRFExperiment):

    # This is the directory withing the root directory that contains the EEG data
    data_dir = 'eeg'
    # This is how subject names are identified ("S" followed by two digits). See the documentation of the builtin Python regular expression (re) module for details on how to build patterns
    subject_re = r'S\d\d'

    # This is used to identify the *-raw.fif file in the eeg directory (some experiments contain more than one session per participant)
    sessions = ['alice']

    # This defines the preprocessing pipeline. For details see https://eelbrain.readthedocs.io/en/stable/experiment.html
    raw = {
        'raw': RawSource(connectivity='auto'),
        '0.5-20': RawFilter('raw', 0.5, 20, cache=False),
    }

    # This adds the segment duration (plus 1 second) to the events marking stimulus onset in the eeg files. For details see https://eelbrain.readthedocs.io/en/stable/experiment.html
    variables = {
        'duration': LabelVar('event', {k: v + 1 for k, v in SEGMENT_DURATION.items()}),
    }

    # This defines a data "epoch" to extract the event-related data segments from the raw EEG data during which the story was presented. For details see https://eelbrain.readthedocs.io/en/stable/experiment.html
    epochs = {
        'chapter-1': PrimaryEpoch('alice', tmin=0, tmax='duration', samplingrate=50),
    }

    # This defines which variable (among the variables assigned to the events in the EEG recordings) designates the stimulus that was presented. This is used to identify the predictor file corresponding to each event. Here we use the segment number for this purpose, as the EEG recordings already contains events labeled 1-12, according to which of the stimulus wave files was presented. This name is also the first part of each predictor filename, followed by ``~``. For example, ``1~gammatone-8.pickle`` is the ``gammatone-8`` predictor for segment ``1``. The value used here, 'event', is a variable that is already contained in the EEG files, but any variable added by the user could also be used (see the *Events* section in https://eelbrain.readthedocs.io/en/stable/experiment.html)
    stim_var = 'event'

    # This specifies how the pipeline will find and handle predictors. The key indicates the first part of the filename in the <root>/predictors directory. The FilePredictor object allows specifying some options on how the predictor files are used (see the FilePredictor documentation for details).
    predictors = {
        'gammatone': FilePredictor(resample='bin'),
        'word': FilePredictor(columns=True),
    }

    # Models are shortcuts for invoking multiple predictors
    models = {
        'auditory-gammatone': 'gammatone-8 + gammatone-on-8',
    }


# This creates an instance of the pipeline. Doing this here will allow other scripts to import the instance directly.
alice = Alice(DATA_ROOT)
