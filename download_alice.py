# Author: Proloy Das <pdas6@mgh.harvard.edu>
# License: MIT
"""
Usage: run the script without any arguments (downloaded at default location)::

    $ python download_alice.py

Or with a valid pathname argument (downloaded at pathname)::

    $ python download_alice.py /home/xyz/alice_data

Disclaimer: The following functions are heavily inspired by the namesake
functions in mne/datasets/utils.py and mne/utils/check.py which are
distributed under following license terms:

Copyright © 2011-2019, authors of MNE-Python
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import os.path as op
import sys
import shutil
import tarfile
import stat
import zipfile
from urllib.request import urlretrieve

from mne.utils import logger
from mne.utils.numerics import hashfunc


_alice_license_text = """
This dataset accompanies "Eelbrain: A Python toolkit for time-continuous
analysis with temporal response functions" (Brodbeck et al., 2021) and is a
derivative of the Alice EEG datasets collected at the University of Michigan
Computational Neurolinguistics Lab (Bhattasali et al., 2020). The files were
converted from the original matlab format to fif format in order to be
compatible with Eelbrain.


The original dataset is licensed under CC BY
(https://creativecommons.org/licenses/by/4.0/) and by downloading the dataset
you pledge to comply with all relevant rules and regulations imposed by the
above-mentioned license terms.

The original work can be found at DOI: 10.7302/Z29C6VNH.
"""


def _get_path(path, name):
    """Get a dataset path."""
    # 1. Input
    if path is not None:
        if not isinstance(path, str):
            raise ValueError('path must be a string or None')
        return path
    logger.info('Using default location ~/Data/Alice for %s...' % name)
    path = op.join(os.getenv('_ALICE_FAKE_HOME_DIR', op.expanduser("~")), 'Data', 'Alice')
    if not op.exists(path):
        logger.info(f'Creating {path}')
        try:
            os.makedirs(path)
        except OSError:
            raise OSError("User does not have write permissions "
                          "at '%s', try giving the path as an "
                          "argument to data_path() where user has "
                          "write permissions, for ex:data_path"
                          "('/home/xyz/me2/')" % (path,))
    return path


def _data_path(path=None, force_update=False, update_path=True, download=True,
               name=None, check_version=False, return_version=False,
               archive_name=None, accept=False):
    """Aux function."""
    path = _get_path(path, name)

    # try to match url->archive_name->folder_name
    urls = dict(  # the URLs to use
        alice=[
            'https://drum.lib.umd.edu/bitstream/handle/1903/27591/stimuli.zip',
            'https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.0.zip',
            'https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.1.zip',
            'https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.2.zip',
        ],
        mtrfs="unknown",
    )
    # filename of the resulting downloaded archive (only needed if the URL
    # name does not match resulting filename)
    archive_names = dict(
        alice=['stimuli.zip',
               'eeg.0.zip',
               'eeg.1.zip',
               'eeg.2.zip'],
        mtrfs='whoknows.zip',
    )
    # original folder names that get extracted (only needed if the
    # archive does not extract the right folder name; e.g., usually GitHub)
    folder_origs = dict(  # not listed means None (no need to move)
    )
    # finally, where we want them to extract to (only needed if the folder name
    # is not the same as the last bit of the archive name without the file
    # extension)
    folder_names = dict(
        brainstorm='MNE-brainstorm-data',
    )
    md5_hashes = dict(
        alice=[
            '4336a47bef7d3e63239c40c0623dc186',
            'd63d96a6e5080578dbf71320ddbec0a0',
            'bdc65f168db4c0f19bb0fed20eae129b',
            '3fb33ca1c4640c863a71bddd45006815',
            ],
        mtrfs='3194e9f7b46039bb050a74f3e1ae9908',
    )
    assert set(md5_hashes) == set(urls)
    url = urls[name]
    hash_ = md5_hashes[name]
    folder_orig = folder_origs.get(name, None)
    url = [url] if not isinstance(url, list) else url
    hash_ = [hash_] if not isinstance(hash_, list) else hash_
    archive_name = archive_names.get(name)
    if archive_name is None:
        archive_name = [u.split('/')[-1] for u in url]
    if not isinstance(archive_name, list):
        archive_name = [archive_name]
    folder_path = [
       op.join(path, folder_names.get(name, a.split('.')[0]))
       for a in archive_name
       ]
    if not isinstance(folder_orig, list):
        folder_orig = [folder_orig] * len(url)
    folder_path = [op.abspath(f) for f in folder_path]
    assert hash_ is not None
    assert all(isinstance(x, list) for x in (url, archive_name,
                                             folder_path))
    assert len(url) == len(archive_name) == len(folder_path)
    logger.debug('URL:          %s' % (url,))
    logger.debug('archive_name: %s' % (archive_name,))
    logger.debug('hash:         %s' % (hash_,))
    logger.debug('folder_path:  %s' % (folder_path,))

    need_download = any(not op.exists(f) for f in folder_path)
    if need_download and not download:
        return ''

    if need_download or force_update:
        logger.debug('Downloading: need_download=%s, force_update=%s'
                     % (need_download, force_update))
        for f in folder_path:
            logger.debug('  Exists: %s: %s' % (f, op.exists(f)))
        # # License
        if name == 'alice':
            if accept or '--accept-alice-license' in sys.argv:
                answer = 'y'
            else:
                # If they don't have stdin, just accept the license
                # https://github.com/mne-tools/mne-python/issues/8513#issuecomment-726823724  # noqa: E501
                answer = _safe_input(
                    '%sAgree (y/[n])? ' % _alice_license_text, use='y')
            if answer.lower() != 'y':
                raise RuntimeError('You must agree to the license to use this '
                                   'dataset')
        assert len(url) == len(hash_)
        assert len(url) == len(archive_name)
        assert len(url) == len(folder_orig)
        assert len(url) == len(folder_path)
        assert len(url) > 0
        # 1. Get all the archives
        full_name = list()
        for u, an, h in zip(url, archive_name, hash_):
            remove_archive, full = _download(path, u, an, h)
            full_name.append(full)
        del archive_name
        # 2. Extract all of the files
        remove_dir = True
        for u, fp, an, h, fo in zip(url, folder_path, full_name, hash_,
                                    folder_orig):
            _extract(path, name, fp, an, None, remove_dir)
            remove_dir = False  # only do on first iteration
        # 3. Remove all of the archives
        if remove_archive:
            for an in full_name:
                os.remove(op.join(path, an))

        logger.info('Successfully extracted to: %s' % path)

    return path


def _download(path, url, archive_name, hash_, hash_type='md5'):
    """Download and extract an archive, completing the filename."""
    full_name = op.join(path, archive_name)
    remove_archive = True
    fetch_archive = True
    if op.exists(full_name):
        logger.info('Archive exists (%s), checking hash %s.'
                    % (archive_name, hash_,))
        fetch_archive = False
        if hashfunc(full_name, hash_type=hash_type) != hash_:
            if input('Archive already exists but the hash does not match: '
                     '%s\nOverwrite (y/[n])?'
                     % (archive_name,)).lower() == 'y':
                os.remove(full_name)
                fetch_archive = True
    if fetch_archive:
        logger.info('Downloading archive %s to %s' % (archive_name, path))
        try:
            temp_file_name, header = urlretrieve(url)
            # check hash sum eg md5sum
            if hash_ is not None:
                logger.info('Verifying hash %s.' % (hash_,))
                hashsum = hashfunc(temp_file_name, hash_type=hash_type)
                if hash_ != hashsum:
                    raise RuntimeError('Hash mismatch for downloaded file %s, '
                                       'expected %s but got %s'
                                       % (temp_file_name, hash_, hashsum))
            shutil.move(temp_file_name, full_name)
        except Exception:
            logger.error('Error while fetching file %s.'
                         ' Dataset fetching aborted.' % url)
            raise
        # _fetch_file(url, full_name, print_destination=False,
        #             hash_=hash_, hash_type=hash_type)
    return remove_archive, full_name


def _extract(path, name, folder_path, archive_name, folder_orig, remove_dir):
    if op.exists(folder_path) and remove_dir:
        logger.info('Removing old directory: %s' % (folder_path,))

        def onerror(func, path, exc_info):
            """Deal with access errors (e.g. testing dataset read-only)."""
            # Is the error an access error ?
            do = False
            if not os.access(path, os.W_OK):
                perm = os.stat(path).st_mode | stat.S_IWUSR
                os.chmod(path, perm)
                do = True
            if not os.access(op.dirname(path), os.W_OK):
                dir_perm = (os.stat(op.dirname(path)).st_mode |
                            stat.S_IWUSR)
                os.chmod(op.dirname(path), dir_perm)
                do = True
            if do:
                func(path)
            else:
                raise exc_info[1]
        shutil.rmtree(folder_path, onerror=onerror)

    logger.info('Decompressing the archive: %s' % archive_name)
    logger.info('(please be patient, this can take some time)')
    if archive_name.endswith('.zip'):
        with zipfile.ZipFile(archive_name, 'r') as ff:
            ff.extractall(path)
    else:
        if archive_name.endswith('.bz2'):
            ext = 'bz2'
        else:
            ext = 'gz'
        with tarfile.open(archive_name, 'r:%s' % ext) as tf:
            tf.extractall(path=path)

    if folder_orig is not None:
        shutil.move(op.join(path, folder_orig), folder_path)


def _safe_input(msg, *, alt=None, use=None):
    "copied from mne/utils/check.py"
    try:
        return input(msg)
    except EOFError:  # MATLAB or other non-stdin
        if use is not None:
            return use
        raise RuntimeError(
            f'Could not use input() to get a response to:\n{msg}\n'
            f'You can {alt} to avoid this error.')


def data_path(path=None, force_update=False, update_path=True, download=True,
              accept=False, verbose=None):
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='alice',
                      download=download, accept=accept)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        path = data_path()
    elif isinstance(sys.argv[1], str):
        path = data_path(path=sys.argv[1])
    else:
        raise ValueError(f'{sys.argv[1]} is not a valid pathname.'
                         f'Run script either with a valid path argument or'
                         f'or without any argument.')
