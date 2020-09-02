"""Class for downloading and ingestion of data for running regression tests.

The output of these classes is a directory of TFRecord files ready for
decoding experiments.

Note, this code assumes the following data flow
  Internet data file -> Download -> cache_dir (local copy of original data)
  cache_dir -> Ingestion -> tf_dir (tfrecord files ready for decoding exps)

"""


import os
import re
import sys
import tempfile
import typing
import urllib

from absl import app
from absl import flags
from absl import logging
import attr
import numpy as np
import pandas as pd
import pyedflib
import requests
import scipy.io as spio

from telluride_decoding import brain_data
from telluride_decoding import ingest
from telluride_decoding import preprocess

import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


regression_data_print = sys.stdout  # Feel free to redirect prints elsewhere.


@attr.s
class DataLocation(object):
  internet = attr.ib(type=typing.Text)   # Original source of data on Internet
  cache_dir = attr.ib(type=typing.Text)   # Loc. of local copy of original data
  tf_dir = attr.ib(type=typing.Text)  # Location of local ingested TFRecord copy
  desired_frame_rate = attr.ib(type=float)  # Frame rate for the data
  data_type = attr.ib()    # Which Regression class provides default params


# These flags allow these parameters to override the defaults defined in the
# DataLocation objects defined at the end of this file.

flags.DEFINE_string('internet', None,
                    'Location of data on the Internet.')
flags.DEFINE_string('cache_dir', None,
                    'Where to cache datasets downloaded from the Internet.')
flags.DEFINE_string('tf_output_dir', None,
                    'The base directory for storing ingested TFRecords.')
flags.DEFINE_float('desired_frame_rate', 0,
                   'Desired frame rate after ingestion of the TFRecords.')
flags.DEFINE_bool('force', False,
                  'Ignore existing files and force new download & ingestion.')

# Note: One additional flag (type) is defined below, after we know which
# RegressionData classes are defined by this file.

FLAGS = flags.FLAGS

# Get the location of a directory for temporary files.
#  https://unix.stackexchange.com/questions/352107/generic-way-to-get-temp-path
_tmp_dir = os.environ.get('TMPDIR') or '/tmp'


def loadmat(filename):
  """Read a matlab file.

  This function reads a Matlab file from disk, parses it, and returns it in
  a simple dictionary. Code from https://stackoverflow.com/a/8832212.

  Args:
    filename: Full path to mat file from which to load data

  Returns:
    out_dict: Dictionary with the contents of the mat file
  """

  def _check_keys(key_dict):
    # checks if entries in dictionary are mat-objects.
    # If yes todict is called to change them to nested dictionaries.
    for key in key_dict:
      if isinstance(key_dict[key], spio.matlab.mio5_params.mat_struct):
        key_dict[key] = _todict(key_dict[key])
    return key_dict

  def _todict(matobj):
    """A recursive function converting matobjects to nested dictionaries."""
    key_dict = {}
    # pylint: disable=protected-access
    for strg in matobj._fieldnames:
      elem = matobj.__dict__[strg]
      if isinstance(elem, spio.matlab.mio5_params.mat_struct):
        key_dict[strg] = _todict(elem)
      else:
        key_dict[strg] = elem
    return key_dict

  with tf.io.gfile.GFile(filename, 'rb') as fp:
    data = spio.loadmat(fp, struct_as_record=False, squeeze_me=True)
  out_dict = _check_keys(data)
  return out_dict


def download_from_gdrive(url, output, debug=False):
  """Download the file off the drive URL and store it locally.

  Code from: https://github.com/wkentaro/gdown/blob/master/gdown/download.py
  Args:
    url: Google drive URL to download data from
    output: Location to store output file
    debug: Displays detailed logs if True

  Returns:
    output: Location of output file when downloaded correctly, None otherwise.
  """

  def parse_url(url):
    """Parse URL for gdrive ID."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname == 'drive.google.com'
    is_download_link = parsed.path.endswith('/uc')

    file_id = None
    if is_gdrive and 'id' in query:
      file_ids = query['id']
      if len(file_ids) == 1:
        file_id = file_ids[0]
    match = re.match(r'^/file/d/(.*?)/view$', parsed.path)
    if match:
      file_id = match.groups()[0]
    if is_gdrive and not is_download_link:
      logging.warn(
          'You specified a Google Drive Link but it is not the correct link '
          'to download the file. Maybe you should try: %s',
          'https://drive.google.com/uc?id=%s' % file_id)

    return file_id, is_download_link

  def get_url_from_gdrive_confirmation(contents):
    """This function appends gdrive confirmation postfix to the url."""
    url = ''
    for line in contents.splitlines():
      m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
      if m:
        url = 'https://docs.google.com' + m.groups()[0]
        url = url.replace('&amp;', '&')
        return url
      m = re.search('confirm=([^;&]+)', line)
      if m:
        confirm = m.groups()[0]
        url = re.sub(r'confirm=([^;&]+)', r'confirm={}'.format(confirm), url)
        return url
      m = re.search('"downloadUrl":"([^"]+)', line)
      if m:
        url = m.groups()[0]
        if url:
          url = url.replace('\\u003d', '=')
          url = url.replace('\\u0026', '&')
        return url

  url_origin = url
  sess = requests.session()
  file_id, is_download_link = parse_url(url)
  while True:
    res = sess.get(url, stream=True)
    if 'Content-Disposition' in res.headers:
      # This is the file
      break
    if not (file_id and is_download_link):
      break
    # Need to redirect with confirmation
    url = get_url_from_gdrive_confirmation(res.text)
    if url is None:
      logging.error('Permission denied: %s.', url_origin)
      logging.error('Maybe you need to change permission over '
                    "'Anyone with the link'?")
      return

  if output is None:
    if file_id and is_download_link:
      m = re.search('filename="(.*)"', res.headers['Content-Disposition'])
      if m:
        output = m.groups()[0]
      else:
        raise ValueError('Can not parse headers: %s' % res)
    else:
      output = os.path.basename(url)
  output_is_path = isinstance(output, str)

  if debug:
    logging.info('Downloading...')
    logging.info('From: %s', url_origin)
    logging.info('To: %s',
                 os.path.abspath(output) if output_is_path else output)

  if output_is_path:
    tmp_file = tempfile.mktemp(
        suffix=tempfile.template,
        prefix=os.path.basename(output),
        dir=os.path.dirname(output),
    )
    f = tf.io.gfile.GFile(tmp_file, 'wb')
  else:
    tmp_file = None
    f = output

  if not f:
    raise ValueError('Can not open output file: %s' % output)
  try:
    total = res.headers.get('Content-Length')
    chunk_size = 512 * 1024  # 512KB
    for chunk in res.iter_content(chunk_size=chunk_size):
      f.write(chunk)
    if total is not None:
      total = int(total)

    if tmp_file:
      f.close()
      tf.io.gfile.rename(tmp_file, output, overwrite=True)
  except IOError as e:
    print(e, file=regression_data_print)
    return
  finally:
    try:
      if tmp_file:
        tf.io.gfile.remove(tmp_file)
    except OSError as e:
      print(e, file=regression_data_print)
    except tf.errors.NotFoundError:
      pass
  return output


def make_if_not_exists(directory):
  if not os.path.exists(directory):
    tf.io.gfile.makedirs(directory)


class RegressionData(object):
  """Base for data download & to ingest data for regression experiments."""

  def download_data(self, url, cache_dir, debug=False):
    # Be sure to specialize this routine and call this superclass last.
    del debug
    readme_file = os.path.join(cache_dir, 'README.txt')
    with tf.io.gfile.GFile(readme_file, 'w') as fp:
      fp.write('These files were downloaded\nFrom %s\nTo %s\nUsing: %s\n' %
               (url, cache_dir, sys.argv))



class RegressionDataTelluride4(RegressionData):
  """Class for downloading and ingesting Telluride4 data."""

  def is_data_local(self, cache_dir):
    return tf.io.gfile.exists(os.path.join(cache_dir, 'Telluride2015.mat'))

  def download_data(self, url, cache_dir, debug=False):
    """Download the Telluride4 data from the published location.

    Need to copy to local /tmp file because gdrive code doesn't support writing
    to tf.io.gfile object.

    The normal location is:
      https://drive.google.com/uc?id=0ByZjGXodIlspWmpBcUhvenVQa1k

    Args:
      url: Location of the original data from the Telluride4 experiment
      cache_dir: Where to store the local copy.
      debug: Get debugging information from downloading process.
    """
    tmp_file = os.path.join(_tmp_dir, 'Telluride2015.mat')
    download_from_gdrive(
        url,
        tmp_file,
        debug=debug)
    cache_file = os.path.join(cache_dir, os.path.split(tmp_file)[1])
    with tf.io.gfile.GFile(tmp_file, 'rb') as old_fp:
      logging.info('Copying %s to %s...', tmp_file, cache_file)
      with tf.io.gfile.GFile(cache_file, 'wb') as new_fp:
        new_fp.write(old_fp.read())

    super(RegressionDataTelluride4, self).download_data(url, cache_dir)

  def is_data_ingested(self, tf_dir, num_files=32):
    file_count = len(tf.io.gfile.glob(os.path.join(tf_dir, '*.tfrecords')))
    return file_count == num_files

  def ingest_data(self, cache_dir, tf_dir, desired_frame_rate):
    """Reads the Telluride4 data from mat file and ingest TFrecords.

    Args:
      cache_dir: Directory with the local copy of the original data
      tf_dir: Where to store the TFRecord files.
      desired_frame_rate: Desired frame rate after ingestion.
    """
    mat_data = loadmat(os.path.join(cache_dir, 'Telluride2015.mat'))
    mat_objects = mat_data['data']

    eeg_signals = mat_objects['eeg']
    audio_signals = mat_objects['wav']
    logging.info('Audio and EEG data shapes: %s, %s',
                 ''.join(str(audio_signals.shape)),
                 ''.join(str(eeg_signals.shape)))
    if audio_signals.shape[0] != 4:  # Number of audio files.
      raise ValueError('Incorrect shapes for audio_signals (%s)' %
                       audio_signals.shape)
    if eeg_signals.shape[0] != 32:  # Number of trials.
      raise ValueError('Incorrect shapes for eeg_signals (%s)' %
                       eeg_signals.shape)
    eeg_signals_count = eeg_signals.shape[0]

    make_if_not_exists(tf_dir)
    trial_dict = {}
    all_ingested_files = []
    for i in range(eeg_signals_count):
      sound_dict = {
          'intensity':
              audio_signals[i % 4],
          'ones':
              np.ones(audio_signals[i % 4].shape,
                      dtype=audio_signals[i % 4].dtype)
      }
      eeg_dict = {'eeg_data': eeg_signals[i]}
      trial_dict['trial_{:02d}'.format(i + 1)] = [
          sound_dict,
          ingest.MemoryBrainDataFile(eeg_dict)
      ]

    eeg_dir = '.'
    sound_dir = '.'
    exp = ingest.BrainExperiment(
        trial_dict, sound_dir, eeg_dir, frame_rate=desired_frame_rate)
    exp.load_all_data(sound_dir, eeg_dir)
    exp.z_score_all_data()
    for trial in exp.iterate_trials():
      trial.assemble_brain_data('eeg_data')
      for k in trial.model_features:
        logging.info('Trial#, Audio shape: %s, %s', str(k),
                     ''.join(str(trial.model_features[k].shape)))
    make_if_not_exists(tf_dir)
    all_ingested_files = exp.write_all_data(tf_dir)

    write_summary(cache_dir, tf_dir, desired_frame_rate, all_ingested_files)


class RegressionDataJensMemory(RegressionData):
  """Class for downloading & ingesting Jens data."""

  @property
  def name(self):
    return 'Jens'

  def is_data_local(self, cache_dir, num_subjects=22):
    if tf.io.gfile.exists(cache_dir):
      all_files = [f for f in tf.io.gfile.listdir(cache_dir)
                   if f.endswith('mat')]
      return len(all_files) == num_subjects
    return False

  def download_data(self, url, cache_dir):
    """Downloads the Jens data by COCOHA from the published location.

    Normal location is:
      https://zenodo.org/record/1158410/files/DATA.zip  # 128Hz Data, 64 chan

    Args:
      url: Location of the original data from Jens' experiment
      cache_dir: Where to store the local copy.
    """
    tmp_jens_dir = os.path.join(_tmp_dir, 'jens_raw_data')
    tf.io.gfile.makedirs(tmp_jens_dir)
    os.system('wget -c ' + url + ' -P ' + tmp_jens_dir + '/')
    os.system('unzip -o ' + os.path.join(tmp_jens_dir, 'DATA.zip') + ' -d ' +
              tmp_jens_dir)

    all_files = sorted(tf.io.gfile.listdir(tmp_jens_dir))
    mat_files = [f for f in all_files if f.endswith('.mat')]
    logging.info(mat_files)
    make_if_not_exists(cache_dir)
    for mat_file in mat_files:
      old_file = os.path.join(tmp_jens_dir, mat_file)
      new_file = os.path.join(cache_dir, os.path.split(old_file)[1])
      with tf.io.gfile.GFile(old_file, 'rb') as old_fp:
        logging.info('Copying %s to %s...', old_file, new_file)
        with tf.io.gfile.GFile(new_file, 'wb') as new_fp:
          new_fp.write(old_fp.read())

    super(RegressionDataJensMemory, self).download_data(url, cache_dir)

  def is_data_ingested(self, tf_dir, num_subjects=22, num_trials=40):
    """Checks to make sure all the data has been ingested into TFRecords."""
    if tf.io.gfile.exists(tf_dir):
      return sum([
          len(tf.io.gfile.glob(os.path.join(tf_dir, sdir, '*.tfrecords')))
          for sdir in tf.io.gfile.glob(os.path.join(tf_dir, 'subject_*'))
      ]) >= num_trials * num_subjects
    return False

  def ingest_data(self, cache_dir, tf_dir, desired_frame_rate):
    """Reads the Jens data from mat files and ingest it to TFrecords.

    Args:
      cache_dir: Local copy of the original archive file from the Internet
      tf_dir: Folder where tfrecord files are written out
      desired_frame_rate: Desired frame rate after ingestion.
    """
    mat_files_list = sorted(tf.io.gfile.glob(os.path.join(cache_dir,
                                                          '*.mat')))
    eeg_dir = '.'
    sound_dir = '.'
    make_if_not_exists(tf_dir)

    print('Ingesting %d files of Jens data.' % len(mat_files_list),
          file=regression_data_print)
    all_ingested_files = []
    for sid, mat_file in enumerate(mat_files_list):
      print('Ingesting %s' % mat_file,
            file=regression_data_print)
      tf_dir_subject = os.path.join(tf_dir,
                                    'subject_{:02d}'.format(sid + 1))
      mat_data = loadmat(mat_file)
      mat_object = mat_data['data']
      # Both framte rates should be 128Hz according to:
      #   https://zenodo.org/record/1158410/#.XvqtpZNKjVs
      wav_fs = mat_object['fsample']
      eeg_fs = mat_object['fsample']
      trial_dict = {}
      for trial_idx, trial in enumerate(mat_object['trial']):
        eeg_signal = trial[:69, :].T
        audio_signal = trial[69:70, :].T
        p_eeg = preprocess.Preprocessor('eeg', eeg_fs, desired_frame_rate)
        ds_eeg_signal = p_eeg.resample(eeg_signal)
        p_audio = preprocess.Preprocessor('audio', wav_fs, desired_frame_rate)
        ds_audio_signal = p_audio.resample(audio_signal)
        eeg_dict = {'eeg_data': ds_eeg_signal}
        audio_files_dict = {'intensity': ds_audio_signal}
        trial_key = 'trial_{:02d}'.format(trial_idx + 1)
        trial_dict[trial_key] = [
            audio_files_dict,
            ingest.MemoryBrainDataFile(eeg_dict, sr=desired_frame_rate)
        ]
        logging.info('Audio and EEG data shapes: %s, %s',
                     ''.join(str(audio_signal.shape)),
                     ''.join(str(eeg_signal.shape)))
      exp = ingest.BrainExperiment(
          trial_dict, sound_dir, eeg_dir, frame_rate=desired_frame_rate)
      exp.load_all_data(sound_dir, eeg_dir)
      exp.z_score_all_data()
      for trial in exp.iterate_trials():
        trial.assemble_brain_data('eeg_data')
        for k in trial.model_features:
          logging.info('Trial # %s, audio shapes %s', str(k),
                       ''.join(str(trial.model_features[k].shape)))
      make_if_not_exists(tf_dir_subject)
      all_ingested_files.extend(exp.write_all_data(tf_dir_subject))

    write_summary(cache_dir, tf_dir, desired_frame_rate, all_ingested_files)


class RegressionDataJensImpaired(RegressionData):
  """Class for downloading & ingesting Jens data."""

  @property
  def name(self):
    return 'JensImpaired'

  def is_data_local(self, cache_dir, num_subjects=44):
    if tf.io.gfile.exists(cache_dir):
      all_files = tf.io.gfile.listdir(cache_dir)
      all_files_sub = [f.startswith('sub-') for f in all_files]
      return len(all_files_sub) == num_subjects
    return False

  def download_data(self, url, cache_dir):
    """Dummy function.  Too big to download automatically.

    This is too bulky to fetch each time.  Just tell user how to download it
    by hand.  Normal location is
      https://zenodo.org/record/3618205/files/ds-eeg-snhl.tar?download=1

    Args:
      url: Location of the original data from Jens' experiment
      cache_dir: Where to store the local copy.
    """
    super(RegressionDataJensImpaired, self).download_data(url, _tmp_dir)
    print('To download manually, use command: wget -c {} {}'.format(
        url, cache_dir))

  def is_data_ingested(self, tf_dir, num_subjects=44, num_trials=48):
    """Checks to make sure all the data has been ingested into TFRecords."""
    if tf.io.gfile.exists(tf_dir):
      return sum([
          len(tf.io.gfile.glob(os.path.join(tf_dir, sdir, '*.tfrecords')))
          for sdir in tf.io.gfile.listdir(tf_dir)
      ]) >= num_trials * num_subjects
    return False

  def ingest_data(self, cache_dir, tf_dir, desired_frame_rate):
    """Reads the Jens data from mat files and ingest it to TFrecords.

    Args:
      cache_dir: Local copy of the original archive file from the Internet
      tf_dir: Folder where tfrecord files are written out
      desired_frame_rate: Desired frame rate after ingestion.
    """
    eeg_dir = '.'
    sound_dir = '.'

    frame_rate = 512
    make_if_not_exists(tf_dir)

    # All subject directories from the cache directory
    all_dirs = tf.io.gfile.listdir(cache_dir)
    all_dirs_sub = sorted([f for f in all_dirs if f.startswith('sub-')])
    print('Ingesting {} subject directories of Jens Hearing impaired data.'
          .format(len(all_dirs_sub)))

    for sid, subject_dir in enumerate(all_dirs_sub):
      # There is a single EEG and events file per subject
      eeg_file = os.path.join(
          cache_dir, subject_dir,
          'eeg/{}_task-selectiveattention_eeg.bdf'.format(subject_dir))
      events_file = os.path.join(
          cache_dir, subject_dir,
          'eeg/{}_task-selectiveattention_events.tsv'.format(subject_dir))

      # Read in events file and load the start times of attended and
      # unattended audio for all trials (48 trials, 32 with dual audio)
      # Loading the events tsv file into a pandas Dataframe object:
      # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
      # - a tabular format. Each column in the table, accessed using a string
      # key holds an array of objects of a fixed type. Keys are the column
      # headers in the tsv file.
      events_df = pd.read_csv(events_file, sep='\t')
      # Subject 24 has events described in 2 parts hence the if case below
      # which pulls the remaining data from the second file.
      if sid == 23:
        events_file_part = os.path.join(
            cache_dir, subject_dir,
            'eeg/{}_task-selectiveattention_run-2_events.tsv'.format(
                subject_dir))
        events_df_part = pd.read_csv(events_file_part, sep='\t')
        events_df = pd.concat([events_df, events_df_part])

      # Attended audio is called target and unattended audio called masker.
      # target_onset and marker_onset events indicate the start of both of those
      # audio streams. stim_file contains the filename storing the audio stream
      # and sample indicates the EEG sample number at which the audio stream
      # starts.
      start_samples = events_df[events_df['trigger_type'] ==
                                'targetonset']['sample'].values
      masker_df = events_df[events_df['trigger_type'] == 'maskeronset'][[
          'sample', 'stim_file'
      ]]
      # Get ids of experiments with masker audio from file names
      masker_df['stim_file'] = masker_df['stim_file'].apply(
          lambda x: int(x.split('/')[-1][1:-4]))

      # Sanity check
      if len(start_samples) != 48 or len(masker_df) != 32:
        raise ValueError(
            'Incorrect event counts for subject %s: %d/48 and %d/32' %
            (subject_dir, len(start_samples), len(masker_df)))

      # Load EEG file, which is in bdf format.
      print('Reading in {}'.format(eeg_file))
      f = pyedflib.EdfReader(eeg_file)
      num_channels = f.signals_in_file

      sigbufs = np.zeros((f.getNSamples()[0], num_channels))
      for i in np.arange(num_channels):
        sigbufs[:, i] = f.readSignal(i)

      # Stuff for trials
      tf_dir_subject = os.path.join(tf_dir,
                                    'subject_{:02d}'.format(sid + 1))
      trial_dict = {}
      target_audio_signals = []  # List storing target audios
      masker_audio_signals = []  # List storing masker audios
      chopped_sigbufs = []  # List storing chopped signals based on audio timing
      all_ingested_files = []
      for trial_idx in range(1, 49):
        # Load the target audio file
        target_audio_signal = loadmat(
            os.path.join(
                cache_dir,
                'derivatives/stimuli/sub{:03d}/target/t{:03d}.mat'.format(
                    sid + 1, trial_idx)))['dat']['feat']
        target_audio_signals.append(target_audio_signal)
        # Load EEG portion based on audio start time
        target_start_time = start_samples[trial_idx - 1]
        current_chopped_signal = sigbufs[
            target_start_time:target_start_time +
            target_audio_signal.shape[0], :]
        chopped_sigbufs.append(current_chopped_signal)

        # Only trials having masker data will have start times. This is used to
        # distinguish between trials having dual speaker data v/s ones having
        # single speaker data.
        masker_start_time = masker_df[masker_df['stim_file'] ==
                                      trial_idx]['sample'].values
        if masker_start_time:
          # Masker is offset from target audio by jitter seconds
          # Align accordingly
          masker_audio_file = os.path.join(
              cache_dir,
              'derivatives/stimuli/sub{:03d}/masker/m{:03d}.mat'.format(
                  sid + 1, trial_idx))
          masker_audio_signal = loadmat(masker_audio_file)['dat']['feat']
          start_time_diff = int(masker_start_time - target_start_time)
          masker_audio_signal = np.concatenate(
              (np.zeros(start_time_diff),
               masker_audio_signal[:-1 * start_time_diff]))
          trial_key = 'trial_{:02d}_dual_speaker'.format(trial_idx)
        else:
          # Trials with single audio - dummy masked
          masker_audio_signal = np.zeros_like(target_audio_signal)
          trial_key = 'trial_{:02d}_single_speaker'.format(trial_idx)
        masker_audio_signals.append(masker_audio_signal)

        assert len(masker_audio_signal) == len(target_audio_signal)
        eeg_dict = {'eeg_data': current_chopped_signal}
        audio_files_dict = {
            'attended_intensity': target_audio_signal,
            'unattended_intensity': masker_audio_signal
        }
        trial_dict[trial_key] = [
            audio_files_dict,
            ingest.MemoryBrainDataFile(eeg_dict, sr=desired_frame_rate)
        ]
        logging.info('Audio and EEG data shapes: %s, %s',
                     ''.join(str(target_audio_signal.shape)),
                     ''.join(str(current_chopped_signal.shape)))
      chopped_sigbufs = np.vstack(chopped_sigbufs)
      target_audio_signal_arr = np.hstack(target_audio_signals)
      masker_audio_signal_arr = np.hstack(masker_audio_signals)
      print('Raw EEG shape:{}'.format(sigbufs.shape))
      print('Cut EEG shape:{}'.format(chopped_sigbufs.shape))
      print('Target Audio shape: {}'.format(target_audio_signal_arr.shape))
      print('Masker audio shape: {}'.format(masker_audio_signal_arr.shape))
      exp = ingest.BrainExperiment(
          trial_dict, sound_dir, eeg_dir, frame_rate=frame_rate)
      exp.load_all_data(sound_dir, eeg_dir)
      exp.z_score_all_data()
      for trial in exp.iterate_trials():
        trial.assemble_brain_data('eeg_data')
        for k in trial.model_features:
          logging.info('Trial # %s, audio shapes %s', str(k),
                       ''.join(str(trial.model_features[k].shape)))
      make_if_not_exists(tf_dir_subject)
      exp.write_all_data(tf_dir_subject)
      all_ingested_files = os.listdir(tf_dir_subject)
      all_ingested_files = [
          os.path.join(tf_dir_subject, f) for f in all_ingested_files
      ]
      write_summary(cache_dir, tf_dir_subject, desired_frame_rate,
                    all_ingested_files)


def write_summary(cache_dir, tf_dir, frame_rate, all_ingested_files=None):
  """Write a summary of the experiment into the directory's readme file.

  The README.txt file contains the source directory and the frame rate, as
  well as the feature names, and the length of each file.

  Args:
    cache_dir: where the original data was cached locally.
    tf_dir: Output directory for the ingested TFRecord files.
    frame_rate: The ingested data file frame rate.
    all_ingested_files: A list of files that were ingested.
  """
  readme_file = os.path.join(tf_dir, 'README.txt')
  with tf.io.gfile.GFile(readme_file, 'w') as fp:
    print('These files were ingested from:', cache_dir, file=fp)
    print('Using:', sys.argv, file=fp)
    print('With a output frame rate of %gHz' % frame_rate, file=fp)

    if all_ingested_files:
      features = brain_data.discover_feature_shapes(all_ingested_files[0])
      print('\nFeature shapes are:', file=fp)
      for k, v in features.items():
        print('\t%s: %s' % (k, v), file=fp)

      print('\nAll ingested files:', file=fp)
      for filename in all_ingested_files:
        count, error = brain_data.count_tfrecords(filename)
        error_string = ''
        if error:
          error_string = 'READ ERROR'
        print('\t%s: %d records (%s seconds) %s' %
              (filename, count, count/float(frame_rate), error_string),
              file=fp)

locations = {}

locations['telluride4'] = DataLocation(
    'https://drive.google.com/uc?id=0ByZjGXodIlspWmpBcUhvenVQa1k',
    'test_data/local_cache/telluride4',
    'test_data/tf_dir/telluride4_64Hz',
    64,   # Original is 128Hz frame rate
    RegressionDataTelluride4
    )

locations['jens_memory'] = DataLocation(
    'https://zenodo.org/record/1158410/files/DATA.zip',  # 128Hz Data, 64 chan
    'test_data/local_cache/jens_memory',
    'test_data/tf_dir/jens_memory_64Hz',
    64,    # Original data starts at 128Hz, download to 64 for these Exps.
    RegressionDataJensMemory
    )

locations['jens_impaired'] = DataLocation(
    'https://zenodo.org/record/3618205/files/ds-eeg-snhl.tar?download=1',
    'test_data/DTU_hearing_loss',
    'test_data/tf_dir/jens_impaired_64Hz/',
    64,    # Original is 512 Hz
    RegressionDataJensImpaired
    )

flags.DEFINE_enum('type', 'telluride4',
                  list(locations.keys()),
                  'Which type of data to ingest.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  database = locations[FLAGS.type]
  data_object = database.data_type()

  # Get default locations for this type of data
  url = FLAGS.internet or database.internet
  cache_dir = FLAGS.cache_dir or database.cache_dir
  tf_dir = FLAGS.tf_output_dir or database.tf_dir
  desired_frame_rate = FLAGS.desired_frame_rate or database.desired_frame_rate
  # Get a local copy of the data
  if FLAGS.force or not data_object.is_data_local(cache_dir):
    print(
        'Downloading data from Internet to cache_dir:',
        cache_dir,
        file=regression_data_print)
    data_object.download_data(url, cache_dir)
  else:
    print(
        'No need to download data since it is here:',
        cache_dir,
        file=regression_data_print)

  # Convert the original data archive into TFRecords
  if FLAGS.force or not data_object.is_data_ingested(tf_dir):
    print('Ingesting data into tf_dir:', tf_dir, file=regression_data_print)
    data_object.ingest_data(cache_dir, tf_dir, desired_frame_rate)
  else:
    print(
        'No need to ingest data since it is here:',
        tf_dir,
        file=regression_data_print)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
