# This file lists batch jobs. Run the batch jobs with::
#
#   $ trf-tools-make-jobs jobs.py
#
from alice import PARAMETERS, alice


JOBS = [
    # Batch-compute TRFs for all subjects:
    alice.trf_job('gammatone-1', **PARAMETERS),
    alice.trf_job('gammatone-1 + gammatone-on-1', **PARAMETERS),
    alice.trf_job('gammatone-8 + gammatone-on-8', **PARAMETERS),
    # Batch compute TRFs for both models in a model comparison:
    alice.trf_job('auditory-gammatone @ gammatone-on-8', **PARAMETERS),
]
