# Copyright 2019 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code to facilitate ingesting data into the decoding framework.

The classes in this file facilitate ingesting data from EEG/MEG/ECoG
experiments, preprocessing it, and outputting it in tfrecord format appropriate
for the decoding software.

This file contains four interdependent classes to describe an experiment. They
are:

BrainSignal: One named brain signal, from one electrode.

BrainTrial: Data from one trial, the result of playing one sound and recording
a number of electrodes (stored as BrainSignals). Each trial will be written to
one TFRecord file. A trial contains one sound file and all the brain signals, as
well as the audio features calculated from the waveform.

BrainDataFile: Where data about one trial is stored. The most common format is
EDF, but an in-memory class allows for other kinds of data. A BrainTrial
contains a pointer to a specific kind of BrainDataFile, depending on how the
data is stored. This is read by the BrainTrial code.

BrainExperiment: All the data about a number of trials, allowing one to grab
all the data, z-score the data, and summarize the experiment.

This code provides tools to read directy from recorded EDF files, associating
them with sound files, and then writing everything out as tfrecord files.

A simpler path uses your own python code to accumulate the different signals,
and then uses the following template to write out the overall tfrecord files. In
this example, the feature data (eeg or audio) are stored in arrays called
f1_data and f2_data, which are num_samples x num_feature_dimensions.  Any
number of feature maps can be used (only 2 are used here.)

  trial_dict = {}
  for i in range(len(f1_data)):
    # Each trial is described in a list of a sound and zero+ EEG recordings.
    trial_dict[f'{i}'] = [{},  # Empty dictionary indicating no sound so far.
                         ]

  experiment = ingest.BrainExperiment(trial_dict, '/tmp', '/tmp')
  experiment.load_all_data()

  for i in range(len(f1_data)):
    experiment.trial_data(f'{i}').add_model_feature('f1', f1_data[i])
    experiment.trial_data(f'{i}').add_model_feature('f2', f2_data[i])

  print(experiment.summary())

  experiment.z_score_all_data()

  !mkdir -p /tmp/tfdir
  experiment.write_all_data('/tmp/tfdir')
"""

import collections
import os
import pickle
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from absl import app
from absl import logging

import numpy as np
import pyedflib
import scipy.io.wavfile
import scipy.signal
import scipy.stats
import tensorflow as tf


def assert_type(var_name: str, var: Any,
                expected_type: Type[Any]) -> None:
  if not isinstance(var, expected_type):
    raise TypeError(f'{var_name} must be of type {expected_type}, ' +
                    f'but got value {var} of type {type(var)}')


class BrainSignal(object):
  """Store a single channel of brain data.

  This can be any 1D or 2D signal with a sample rate.  Only thing special is a
  routine (fix_offset) which allows the front of an EEG signal to be removed
  so the audio and brain signals are in sync. All data is stored as a
  two-dimensional array:
    num_times x num_channels

  TODO: change name data_type to data_source
  """

  def __init__(self, name: str, signal: Union[np.ndarray, List[float]],
               sample_rate: float, data_type: Optional[str] = None):
    # Signal has size of num_times x num_channels.
    assert_type('name', name, str)
    signal = np.asarray(signal)
    if not sample_rate > 0.0:
      raise ValueError('Signal\'s sample rate must be greater than 0.')
    self._name = name
    if len(signal.shape) == 1:
      signal = np.reshape(signal, (-1, 1))
    self._signal = signal
    self._sr = float(sample_rate)
    self._time_zero = 0.0
    self._data_type = data_type

  @property
  def signal(self) -> np.ndarray:
    """Returns the signals value (a np.ndarray)."""
    return self._signal

  @property
  def data_type(self) -> str:
    return self._data_type

  @property
  def sr(self) -> float:
    """Return the signal's sample rate."""
    return self._sr

  @property
  def name(self):
    """Return the channel's name."""
    return self._name

  def fix_offset(self, offset_seconds: float):
    """Fix the offset in a brain signal by removing the first offset-seconds.

    Data in an experiment often has the audio and the brain data out of sync,
    usually because the brain data starts recording before we start the audio
    playback. Code elsewhere figures out what the offset is (in seconds), and
    this routine chops off the first offset-seconds so everything will be lined
    up.

    Args:
      offset_seconds: time (in seconds) to remove from the start of this signal.
    """
    if offset_seconds < 0:
      raise ValueError('Offset_seconds to remove must be >= 0.')
    samples = int(offset_seconds * self._sr)
    if samples > 0:
      self._signal = self._signal[samples:,]

# From:
#  https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.mstats.theilslopes.html
# From a scatter plot (audio event times versus eeg event times) compute a
# robust estimate of the slope, in order to estimate the intercept, and thus the
# temporal offset.
# An alternative, from Alain, is to look at the time difference between every
# event in signal A, compared to every event in signal B.  Most will be random,
# but there should be a big peak at the one-true offset.


def find_temporal_offset_via_linear_regression(
    audio_trigger_times: np.ndarray,
    eeg_trigger_times: np.ndarray,
    verbose: bool = True) -> Tuple[float, int]:
  """Find offset between audio and eeg using linear regression over times.

  Given x and y data, find the offset to subtract from the y times to make
  the triggers align.
  Args:
    audio_trigger_times: An array of trigger times embedded in the audio.
    eeg_trigger_times: An array of trigger times from the EEG recording
    verbose: Show debugging information
  Returns:
    A tuple consisting of:
      the amount that the eeg data leads the audio,
      the number of data points that are outliers
  """
  num_points = min(len(audio_trigger_times), len(eeg_trigger_times))
  x = audio_trigger_times[0:num_points]
  y = eeg_trigger_times[0:num_points]
  res = scipy.stats.theilslopes(y, x, 0.90)
  if verbose and abs(res[0]-1.0) > 0.01:
    logging.warning('WARNING: Theil-Sen slope result is: %s', res)
  intercept = res[1]
  yp = (x+intercept)
  outliers = abs(y-yp) > 0.1
  outlier_indices = np.nonzero(outliers)[0]
  if any(outlier_indices):
    logging.info('Outlier indices: %s, Intercept is %g',
                 outlier_indices, intercept)
  if len(outlier_indices) > 1:
    logging.info('audio_trigger_times=%s', audio_trigger_times)
    logging.info('eeg_trigger_times=%s', eeg_trigger_times)
  return res[1], len(outlier_indices)  # The intercept


def find_temporal_offset_via_mode_histogram(audio_triggers: List[float],
                                            eeg_triggers: List[float],
                                            max_time: float = 0,
                                            fs: float = 0) -> float:
  """Use histogram of temporal event differences to compute offset.

  An alternative, from Alain, is to look at the time difference between every
  event in signal A, compared to every event in signal B.  Most will be random,
  but there should be a big peak at the one-true offset.

  Args:
    audio_triggers: Times (in seconds) for the first set of events.
    eeg_triggers: Times (in seconds) for the second set of events.
    max_time: maximum time difference in the histrogram computation.
    fs: sample rate for the trigger waveform.
  Returns:
    The time interval (in seconds if fs is supplied) by which the audio and eeg
    differ. (Without fs, it is returned in samples.)
  """
  audio_triggers = np.asarray(audio_triggers)
  eeg_triggers = np.asarray(eeg_triggers)
  differences = []
  if fs > 0:
    audio_triggers = (audio_triggers*fs).astype(np.int32)
    eeg_triggers = (eeg_triggers*fs).astype(np.int32)
  for a in audio_triggers:
    for e in eeg_triggers:
      if max_time == 0 or abs(e - a) < max_time*fs:
        differences.append(e - a)
  mode, _ = scipy.stats.mode(differences, axis=None)
  mode = int(mode)
  logging.info('find_temporal_offset_via_mode_histogram: mode is %g, '
               'mean is %g', mode, np.mean(differences))
  if fs > 0:
    mode = mode / float(fs)
  return mode


def remove_close_times(times: List[float],
                       min_time: float = 0.06) -> List[float]:
  """Remove close trigger times to get trigger onsets.

  Remove trigger times that are within some specified interval of the
  accepted trigger time. The time interval is usually the duration of the
  trigger pulse.

  Args:
    times: An np array of trigger times in seconds.
    min_time: The minimum expected time between each trigger.

  Returns:
    new_times: An np array of accepted onset times in seconds.
  """
  times = sorted(times)
  idx = 1
  last_time = times[idx-1]
  new_times = np.zeros(np.shape(times))
  new_times[idx-1] = last_time
  for t in times[idx:]:
    if t > last_time + min_time:
      new_times[idx] = t
      idx += 1
    last_time = t
  idx = range(idx, len(times))
  new_times = np.delete(new_times, idx)
  return new_times

###################  Everything about one trial  ###############################


class BrainTrial(object):
  """Store everything we know about one trial.

  A trial consists of an audio file and an arbitrary number of EEG channels.
  This class provides methods to:
  1) Load the sound and EEG data.  EEG data comes from a BrainDataFile class.
  2) Compute features of the audio (right now just intensity and spectrogram)
  3) Find the triggers in the data, either from a trigger event channel (NATUS)
     or a copy of the audio (CGX)
  4) Prepare data for export by finding the trigger signals, removing the
     temporal offset, assemble the desired channels, and write out all the data
     for this trial into a TFRecord file.
  """

  def __init__(self, trial_name: str):
    self._sound_data = None
    self._sound_fs = None
    self._brain_data = collections.OrderedDict()   # Keyed by signal name
    self._model_features = {}   # Keyed by feature name
    if trial_name.endswith('.wav'):
      trial_name = trial_name.replace('.wav', '')
    self._trial_name = trial_name

  @property
  def model_features(self) -> Dict[str, np.ndarray]:
    return self._model_features

  @property
  def brain_data(self) -> Dict[str, np.ndarray]:
    return self._brain_data

  @model_features.setter
  def model_features(self, new_dict: Dict[str, np.ndarray]):
    assert_type('audio features for trial (new_dict)', new_dict, dict)
    self._model_features = new_dict

  @property
  def sound_fs(self) -> float:
    return self._sound_fs

  @property
  def sound_data(self) -> np.ndarray:
    return self._sound_data

  @sound_data.setter
  def sound_data(self, new_sound: np.ndarray):
    self._sound_data = new_sound

  @property
  def filename(self) -> str:
    return 'dummy_brain_trial'

  def add_model_feature(self, name: str, data: Union[List[float], np.ndarray]):
    """Adds a feature (np array) to the dictionary of features for this trial.

    Args:
      name: The name of the feature.
      data: Data which can be transformed into a np array.
    """
    assert_type('name', name, str)
    if not self._model_features:
      self._model_features = {}
    self._model_features[name] = np.asarray(data)

  @property
  def trial_name(self) -> str:
    return self._trial_name

  def summary_string(self) -> str:
    """Construct string to summarize one trial's data."""
    summary = '%d EEG channels' % len(self._brain_data)
    if self._brain_data:
      eeg_sample = self._brain_data[list(self._brain_data.keys())[0]]
      if isinstance(eeg_sample.signal, np.ndarray):
        summary += ' with %gs of eeg data' % (eeg_sample.signal.shape[0]/
                                              float(eeg_sample.sr))
      else:
        summary += 'No EEG data'
      if self._sound_data is not None:
        summary += ', %gs of audio data' % (self._sound_data.shape[0]/
                                            float(self._sound_fs))
      for k in self._model_features:
        summary += ', %s samples of %s data' % (self._model_features[k].shape,
                                                k)
    summary += '.'
    return summary

  def load_sound(self,
                 sound_data: Union[str, List[float], np.ndarray],
                 sound_fs: Optional[float] = None,
                 sound_dir: Optional[str] = None):
    """Load the sound file for this trial.

    Args:
      sound_data: A file name (in sound_dir) from which to read the sound
        waveform (in .wav format).  Or an np.ndarray with the actual data, in
        which case you need to specify the sample rate via sound_fs.
      sound_fs: The sound's sampling rate if sound_data is an np.ndarray.
      sound_dir: Directory from which to read the sound_data.
    """
    if isinstance(sound_data, str):
      sound_filename = os.path.join(sound_dir, sound_data)
      if not sound_filename.endswith('.wav'):
        sound_filename += '.wav'
      try:
        with LocalCopy(sound_filename) as fp:
          [self._sound_fs, self._sound_data] = scipy.io.wavfile.read(fp)
          num_frames = self._sound_data.shape[0]
          self._sound_data = self._sound_data.reshape(num_frames, -1)
          self._sound_data = self._sound_data.astype(np.float32) / 32767.0
      except FileNotFoundError:
        raise ValueError(
            f'Can not open {sound_filename} to read audio waveform.')
      except:
        print(f'Can not read sound data from {sound_filename}')
        raise
    else:
      # If sound_data is not a filename, it must be convertible to an ndarray.
      sound_data = np.asarray(sound_data)
      if sound_fs <= 0:  # pytype: disable=unsupported-operands
        raise ValueError('sound sample rate must be greater than 0.')
      self._sound_data = sound_data.reshape(sound_data.shape[0], -1)
      self._sound_fs = sound_fs

  def load_brain_data(self, eeg_dir: str, brain_data: 'BrainDataFile'):
    """Load the brain_data from one file.

    Note: if there are more than one brain recording of each sound file, then
    this function will be called for each recording file and the results
    are merged. (The labels should probably not overlap between the two files.)

    Args:
      eeg_dir: What folder contains the data
      brain_data: A BrainDataFile object that describes where the data is.
    Raises:
      IOError: if eeg_dir doesn't exist.
      TypeError: for bad parameter values.
    """
    assert_type('brain_data', brain_data, BrainDataFile)
    if not tf.io.gfile.exists(eeg_dir):
      raise IOError('brain data director %s does not exist.' % eeg_dir)
    brain_data.load_all_data(eeg_dir)
    labels = brain_data.signal_names
    data_type = brain_data.data_type
    for name in labels:
      signal = brain_data.signal_values(name)
      sr = brain_data.signal_fs(name)
      logging.info('load_brain_data: loading brain signal named %s with '
                   'fs %gHz', name, sr)
      self._brain_data[name] = BrainSignal(name, signal, sr,
                                           data_type=data_type)

  def iterate_brain_channels(self, data_type: Optional[str] = None):
    for a_brain_signal in self._brain_data.values():
      assert_type('a_brain_signal', a_brain_signal, BrainSignal)
      if data_type is None or a_brain_signal.data_type == data_type:
        yield a_brain_signal

  def adjust_data_sizes(
      self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Adjust data sizes in all the dicts to have the same number of frames.

    This is needed because the different kinds of data (eeg, intensity, etc)
    might have slightly different number of frames. But in the end the TFRecord
    file needs everything to have the same number of frames.

    Each data set has size num_frames x num_dimensions

    Args:
      data_dict: A dictionary pointing to a data array for each signal type.

    Returns:
      The corrected data dictionary.

    Raises:
      ValueError for bad parameter values.
    """
    if not isinstance(data_dict, dict):
      raise ValueError('data supplied to adjust_data_sizes must be a dict.')
    min_size = 1 << 31   # Big number
    for k in data_dict:
      data_shape = data_dict[k].shape
      logging.info('Adjust_data_sizes: original %s.shape is %s', k, data_shape)
      if len(data_shape) == 1:
        data_dict[k] = np.reshape(data_dict[k], (-1, 1))
        data_shape = data_dict[k].shape
      min_size = min(min_size, data_shape[0])
    logging.info('adjust_data_sizes: Min size for data is %d', min_size)
    for k in data_dict:
      if data_dict[k].shape[0] != min_size:
        data_dict[k] = data_dict[k][0:min_size, :]
    return data_dict

  def find_audio_trigger_times(
      self, channel_with_trigger: int = 1) -> List[int]:
    """Find the locations of leading edges of a pulse that indicate a trigger.

    Given a stereo audio signal, with non-zero values at the trigger
    locations, find the actual times (in seconds).  The event impulses are in
    the second channel (index=1).

    Args:
      channel_with_trigger: Where to look for the trigger information

    Returns:
      A list of the times (in sample #) of the trigger starts.
    """
    assert_type('self._sound_data', self._sound_data, np.ndarray)
    if channel_with_trigger > self._sound_data.shape[1]:
      raise ValueError('Trigger channel (%d) too high.' % channel_with_trigger)
    trigger_signal = self._sound_data[:, channel_with_trigger]
    trigger_signal = np.hstack((np.zeros((1)), trigger_signal))
    trigger_times = np.nonzero(np.logical_and(trigger_signal[0:-1] == 0,
                                              trigger_signal[1:] > 0))
    trigger_times = trigger_times[0]/float(self._sound_fs)
    return trigger_times

  # Note: Note: we should apply this fix from NATUS to the trigger values in the
  # EDF file. Magic constants provided by NATUS to fix their code and transform
  # their EDF files into normal byte codes.
  #   TRIGINFIX = INT(-0.0063606452364314*( TRIGINOLD -5151600)+(-32768) +0.5)
  def find_eeg_trigger_times(
      self,
      channel_name: str = 'TRIG') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the times of the EEG trigger events.

    Look for the trigger channel, and detect the changes that occur when an
    event in channel #1 is detected. Return a list of these times.

    Args:
      channel_name: Which channel of the EEG data contains the event trigger.

    Returns:
      A tuple consisting of:
        a list of trigger times
        the actual trigger signal as recovered from the Natus software
        the corrected trigger signal (with Natus level fix)

    Raises:
      ValueError for bad parameter values.
    """
    if channel_name not in self._brain_data:
      raise ValueError('channel name %s not in brain data %s.' %
                       (channel_name, list(self._brain_data.keys())))
    trigger_signal = self._brain_data[channel_name].signal
    def natus_trigger_fix(x):
      return np.floor(-0.0063606452364314*(x - 5151600) + (-32768) + 0.5)
    trigger_signal2 = natus_trigger_fix(trigger_signal)
    trigger_logical = trigger_signal2 % 2
    trigger_edges = np.logical_and(np.logical_not(trigger_logical[0:-1]),
                                   trigger_logical[1:])
    trigger_times = np.nonzero(trigger_edges)[0]
    trigger_times = trigger_times / float(self._brain_data[channel_name].sr)
    return trigger_times, trigger_signal, trigger_signal2

  def find_cognionix_trigger_time(self, channel_name: str = 'EXP32',
                                  level: float = 8000) -> Optional[float]:
    if channel_name not in self._brain_data:
      raise ValueError('channel name %s not in brain data %s.' %
                       (channel_name, self._brain_data))
    times = np.nonzero(self._brain_data[channel_name].signal > level)
    if times:
      return times[0//float(self._brain_data[channel_name].sr)]
    return None

  def fix_eeg_offset(self, offset_seconds: float):
    """Find the offset between audio and EEG using robust linear regression.

    Remove the initial parts of the EEG signal that is caused because the EEG
    recording starts first, and then the audio is played.

    Use the find_audio_trigger_times and find_eeg_trigger_times to find the raw
    event times.  Then use find_temporal_offset_via_linear_regression or
    find_temporal_offset_via_mode_histogram to robustly estimate the offset
    time given the event times in the audio and EEG streams.

    Args:
      offset_seconds: How many seconds, a float, to remove from the beginning
      of each EEG signal in this trial.
    """
    logging.info('fix_eeg_offset: Removing %gs at the start of the %s '
                 'EEG signals', offset_seconds, self._trial_name)
    for signal_name in self._brain_data:
      self._brain_data[signal_name].fix_offset(offset_seconds)

  def assemble_brain_data(self, eeg_channel_names: Union[List[str], str]):
    """Assemble the channels of EEG data for one trial.

    Given the trial_data dictionary, extract the eeg data we care about for a
    decoding experiment, and downsample it if necessary.  Store it as an audio
    feature for later output. Note: fix_eeg_offset should be called before this
    function since this function grabs the eeg data and turns it in an
    audio_feature for output soon. This new brain data is added to the
    audio_feature dictionary with the key 'eeg', and nothing is returned.

    Args:
      eeg_channel_names: a list of channel names, or alternative, a CSV string

    Raises:
      TypeError for bad parameter values.
    """
    if not (isinstance(eeg_channel_names, str) or
            isinstance(eeg_channel_names, list)):
      raise TypeError('eeg_channel_names must be a string or a list of '
                      'strings.')

    if isinstance(eeg_channel_names, str):
      eeg_channel_names = [s.strip() for s in eeg_channel_names.split(',')]
    if len(set(eeg_channel_names)) != len(eeg_channel_names):
      # Check for dupes since it will cause array assembly to fail.
      raise ValueError('Looks like duplicate channel names in request: %s' %
                       eeg_channel_names)
    # Find out how much space we need for the EEG data
    frame_width = 0
    frame_len = 1 << 31   # A really big number
    for k in eeg_channel_names:
      if k not in self._brain_data:
        raise ValueError('Missing feature %s' % k)
      signal = self._brain_data[k].signal
      frame_width += signal.shape[1]
      frame_len = min(frame_len, signal.shape[0])

    # Fill in the EEG data
    logging.info('Trial %s: adjusting EEG data starting with size of %dx%d.',
                 self.trial_name, frame_len, frame_width)
    eeg_data = np.zeros((frame_len, frame_width), dtype=np.float32)
    c = 0
    for k in self._brain_data:    # Iterate over channel names in BV order
      if k in eeg_channel_names:
        signal = self._brain_data[k].signal
        signal_width = signal.shape[1]
        c_end = c + signal_width
        eeg_data[:, c:c_end] = signal[0:frame_len, :]
        c += signal_width
    if c != frame_width:
      raise ValueError('Width mismatch: %d vs %d' % (c, frame_width))
    self._model_features['eeg'] = eeg_data

  def write_data_as_tfrecords(self, tf_dir: str,
                              reverse_data_for_test: bool = False):
    """Given the features we care about, and optionally z-score the data.

    Write it all out as TFRecord files.

    Args:
      tf_dir: Where to put the data (the file will be named by the trial name
        plus .tfrecords)
      reverse_data_for_test: whether to randomize the input and output data by
        reversing the eeg data, so we can test random data.

    Returns:
      The actual filename from where the data was found (for debugging)

    Raises:
      TypeError for bad parameter values.
    """
    assert_type('tf_dir', tf_dir, str)

    # Add in all the available audio features. Assemble_brain_data adds the
    # eeg data before we get here.
    new_data = {}
    for k, v in self._model_features.items():
      new_data[k] = v
    new_data = self.adjust_data_sizes(new_data)

    if reverse_data_for_test:
      logging.info('write_data_as_tfrecords: Reversing %s data for test!',
                   self._trial_name)
      new_data['eeg'] = np.flipud(new_data['eeg'])

    # Write out the data
    filename = os.path.join(tf_dir, self._trial_name + '.tfrecords')
    convert_data_to_tfrecords(filename, new_data)
    return filename


###################  BrainDataFile for data  ###############################


class BrainDataFile(object):
  """Virual class that describes how to read one kind of brain data.

  Query this class to get the signal names and values.
  """

  def __init__(self, data_filename: str, data_type: Optional[str] = None):
    self._data_filename = data_filename
    self._data_type = data_type

  @property
  def filename(self) -> str:
    return self._data_filename

  @property
  def data_type(self) -> str:
    return self._data_type

  def __str__(self) -> str:
    return type(self).__name__ + '(\'' + self._data_filename + '\')'

  @property
  def signal_names(self) -> List[str]:
    raise NotImplementedError

  def signal_values(self, name: str) -> Optional[np.ndarray]:
    raise NotImplementedError

  def signal_fs(self, _) -> float:
    raise NotImplementedError

  def load_all_data(self, _):
    pass


class MemoryBrainDataFile(BrainDataFile):
  """A generic in-memory data file specification.

  The trial data is passed in with a dict, each item giving the channel name
  and then a NP array of data. Use this sub-class for testing, and one-off
  data formats.
  """

  def __init__(self, trial_dict, sr=64, data_type: Optional[str] = None,
               name: str = 'in_memory'):
    assert_type('trial_dict', trial_dict, dict)
    if sr <= 0.0:
      raise ValueError('Sample rate must be > 0.')
    for channel_name, channel_data in trial_dict.items():
      assert_type('channel_name', channel_name, str)
      channel_data = np.asarray(channel_data)
      if len(channel_data.shape) > 2:
        raise ValueError('Bad MemoryBrainDataFile shape for %s(%s)'
                         % (channel_name, str(channel_data.shape)))
    self._my_data_dict = trial_dict
    self._my_sr = sr
    BrainDataFile.__init__(self, name, data_type=data_type)

  @property
  def signal_names(self) -> List[str]:
    return list(self._my_data_dict.keys())

  def signal_values(self, name: str) -> Optional[np.ndarray]:
    if name in self._my_data_dict:
      return self._my_data_dict[name]

  def signal_fs(self, _) -> float:
    return self._my_sr


class LocalCopy(object):
  """Create a local (temporary) copy of a file for software.

  This is a context manager which is an important workaround for (Matlab or
  EDF reading) software that doesn't know how to read from the Google file
  systems. Use this in a with clause to get a local filename that can be
  passed to EDF or Matlab reading code.
  """

  def __init__(self, remote_filename: str):
    self._remote_filename = remote_filename

  def __enter__(self):
    _, suffix = os.path.splitext(self._remote_filename)
    self._fp = tempfile.NamedTemporaryFile(suffix=suffix)
    self._name = self._fp.name
    tf.io.gfile.copy(self._remote_filename, self._name, overwrite=True)
    return self._name

  def __exit__(self, exception_type, exception_value, traceback):
    self._fp.close()


def parse_edf_file(sample_edf_file: str) -> Dict[str, Any]:
  """Parse the content of an EDF file, and return a dict with relevant parts.

  Args:
    sample_edf_file: From where to read the data.

  Returns:
    A dictionary with the parts we care about.
  """
  with pyedflib.EdfReader(sample_edf_file) as f:
    if not f:
      logging.error('Can not read EDF data from %s', sample_edf_file)
      return None  # pytype: disable=bad-return-type  # gen-stub-imports
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    fs_list = f.getSampleFrequencies()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
      sigbufs[i, :] = f.readSignal(i)
    header = f.getHeader()
    signal_headers = f.getSignalHeaders()
  return {'labels': signal_labels,
          'signals': sigbufs,
          'sample_rates': np.array(fs_list),
          'header': header,
          'signal_headers': signal_headers,
         }


class EdfBrainDataFile(BrainDataFile):
  """Code to read the EDF brain-signal file format."""

  def __init__(self, filename, data_type: Optional[str] = None, **kwds):
    self._edf_dict = {}
    super(EdfBrainDataFile, self).__init__(filename, data_type=data_type,
                                           **kwds)

  def load_all_data(self, data_dir: str):
    if not tf.io.gfile.exists(data_dir):
      raise IOError('Data_dir does not exist:', data_dir)
    data_filename = os.path.join(data_dir, self._data_filename)
    if not data_filename.endswith('.edf'):
      data_filename += '.edf'
    if not tf.io.gfile.exists(data_filename):
      raise IOError('Can not open %s for reading' % data_filename)
    with LocalCopy(data_filename) as local_filename:
      # Parse this with local file copy because EDF routine doesn't grok Google
      # file systems.
      self._edf_dict = parse_edf_file(local_filename)

  @property
  def signal_names(self) -> List[str]:
    return self._edf_dict['labels']

  def signal_values(self, name: str) -> np.ndarray:
    assert_type('name', name, str)
    channel_number = self.find_channel_index(name)
    return self._edf_dict['signals'][channel_number]

  def signal_fs(self, name: str) -> float:
    assert_type('name', name, str)
    channel_number = self.find_channel_index(name)
    return self._edf_dict['sample_rates'][channel_number]

  def find_channel_index(self, desired_label: str = 'TRIG') -> Optional[int]:
    """Look through the EDF channel names for the desired channel index.

    Args:
      desired_label: what is the name of the channel we want to find.

    Returns:
      Return the index number in the data array.
    """
    if 'labels' not in self._edf_dict:
      raise ValueError('Can not find labels among: %s' % self._edf_dict.keys())
    for index, label in enumerate(self._edf_dict['labels']):
      if label == desired_label:
        return index
    return None


###################  BrainExperiment for everything  ###########################
BrainTrialDict = Dict[str,
                      List[Union[str, Dict[str, Any], BrainDataFile]]]


class BrainExperiment(object):
  """Everything we know about one experiment.

  This provides pointers to the
  different trials (which are indexed by the sound-file name).  Among other
  things to be done across trials, this routine also does the z-score
  computation, which is necessarily done across all the trials.

  Note: the trial_dict maps from a sound file to BrainDataFile(s). Normally
  there is just one BrainDataFile per sound, but in case of simultaneous
  recordings we will have multiple BrainDataFile recordings.  They get merged
  when loaded into just one BrainTrial object.
  """

  @staticmethod
  def delete_suffix(filename: str, suffix: str) -> str:
    if filename.endswith(suffix):
      filename = filename.replace(suffix, '')
    return filename

  def __init__(self,
               trial_dict: BrainTrialDict,
               sound_dir: Optional[str] = None,
               eeg_dir: Optional[str] = None,
               frame_rate: float = 64):
    """Create a brain experiment from audio and EEG data.

    Args:
      trial_dict: A dictionary that maps trial name into sound and MEG signals.
      sound_dir: Where to find the sound data, if trial_dict has strings (file
        names)
      eeg_dir: Where to find the brain data, if trial_dict has strings (file
        names)
      frame_rate: Sample rate (per second) of the data in this experiment.
    """
    if not isinstance(trial_dict, dict):
      raise TypeError('trial is specified with a dictionary of data not %s' %
                      trial_dict)
    if sound_dir:
      assert_type('sound_dir', sound_dir, str)
    if eeg_dir:
      assert_type('eeg_dir', eeg_dir, str)
    self._sound_dir = sound_dir
    self._eeg_dir = eeg_dir
    self._frame_rate = frame_rate

    self._trial_dict = trial_dict
    for k, v in self._trial_dict.items():
      # Check up front to make we have the right kind of data.
      assert_type('Trial name', k, str)
      assert_type('Trial data', v, list)
    self._data_dict = {}
    self._feature_mean = {}
    self._feature_std = {}

  def trial_data(self, key: str) -> Optional[BrainTrial]:
    if key in self._data_dict:
      return self._data_dict[key]
    return None

  def add_sound_data(self,
                     sound_dict: Dict[str, Union[float, np.ndarray]],
                     trial: BrainTrial):
    """Add the sound data to this trial from a sound_dict.

    Args:
      sound_dict: a dictionary of sound features.  The special keys
        audio_data and audio_sr indicate waveform data.  All the other keys
        are used for audio features.
      trial: A BrainTrial class to which we want to add the sound data.
    """
    assert_type('Sound dictionary', sound_dict, dict)
    assert_type('Trial argument', trial, BrainTrial)
    if 'audio_data' in sound_dict and 'audio_sr' in sound_dict:
      logging.info('Adding sound data of size %s and %gHz to trial %s',
                   len(sound_dict['audio_data']), sound_dict['audio_sr'],
                   trial.trial_name)
      trial.load_sound(sound_dict['audio_data'], sound_dict['audio_sr'])
      del sound_dict['audio_data']
      del sound_dict['audio_sr']
    if sound_dict:
      trial.model_features = sound_dict

  def iterate_trials(self):
    for trial in self._data_dict.values():
      yield trial

  def load_all_data(self, verbose: bool = False):
    """Load all the sound and EEG data for this experiment.

    Args:
      verbose: Print the location of each file we are reading.
    Raises:
      IOError: if sound or eeg directories doesn't exist.
      TypeError for bad parameter values.
    """
    for trial_name, all_data in self._trial_dict.items():
      assert_type('trial_name', trial_name, str)
      this_trial = BrainTrial(trial_name)
      sound_data = all_data[0]
      if isinstance(sound_data, str):
        if verbose:
          logging.info('load_all_data %s: Reading sound data from %s...',
                       trial_name, sound_data)
        this_trial.load_sound(sound_data, sound_dir=self._sound_dir)
      elif isinstance(sound_data, dict):
        self.add_sound_data(sound_data, this_trial)
      else:
        raise TypeError('Can not process %s for sounds.' % type(sound_data))
      for eeg_data_item in all_data[1:]:
        logging.info('load_all_data %s: Reading EEG data from %s...',
                     trial_name, eeg_data_item)
        this_trial.load_brain_data(self._eeg_dir, eeg_data_item)
      self._data_dict[trial_name] = this_trial

  def check_sound_eeg_files(self):
    """Make sure we have all the sound and eeg data.

    Good to do before continuing down the analysis pathway.

    Raises:
      IOError: if sound file doesn't exist.
    """
    assert_type('self._trial_dict', self._trial_dict, BrainTrialDict)
    for (trial_name, trial_data) in self._trial_dict.items():
      sound_loc = os.path.join(self._sound_dir, trial_name + '.wav')
      if not tf.io.gfile.exists(sound_loc):
        raise IOError('Can not find %s in %s' % (trial_name, self._sound_dir))
      if isinstance(trial_data, list):
        trial_list = trial_data
      else:
        trial_list = [trial_data,]
      for data in trial_list:
        if isinstance(data, BrainTrial):
          e = data.filename
          eeg_loc = os.path.join(self._eeg_dir, e + '.edf')
          if not tf.io.gfile.exists(eeg_loc):
            raise IOError('Can not find %s in %s' % (e+'.edf', self._eeg_dir))

  def summary(self) -> str:
    summary = 'Experiment summary:\n'
    summary = summary + ('  Reading sound from: %s\n' % self._sound_dir)
    summary = summary + ('  Reading EEG data from: %s\n' % self._eeg_dir)
    summary = summary + ('  Found %d trials\n' % len(self._trial_dict))
    for trial_name, trial_data in self._data_dict.items():
      summary = summary + ('    Trial %s: %s\n' % (trial_name,
                                                   trial_data.summary_string()))
    return summary

  def get_all_feature_data(self, feature_name: str) -> List[np.ndarray]:
    features = []
    for trial_data in self._data_dict.values():
      model_features = trial_data.model_features
      if feature_name in model_features:
        features.append(model_features[feature_name])
    return features

  def zscore_all_features(self,
                          feature_name: str,
                          mean: Union[float, np.ndarray],
                          std: Union[float, np.ndarray]):
    """Remove mean and normalize variance of all features.

    Args:
      feature_name: Feature to normalize
      mean: Current mean of the data to remove
      std: Standard deviation of the data.
    """
    if abs(std) == 1e-10:  # Don't bother scaling if standard deviation is zero
      std = 1.0
    for trial_data in self._data_dict.values():
      model_features = trial_data.model_features
      if feature_name in model_features:
        new_data = normalize_data(model_features[feature_name], mean, std)
        model_features[feature_name] = new_data
      trial_data.model_features = model_features

  def z_score_all_data(self):
    """Remove the mean and normalize a dictionary of data.

    For the features, which are stored as entries in a dictionary, normalize the
    data so the mean is zero and standard deviation is 1. Arrays in the
    dictionary are modified in place.
    """
    trial_data_key = list(self._data_dict.keys())[0]
    trial_data = self._data_dict[trial_data_key]
    # TODO Should probably define BrainTrial methods for these
    feature_keys = list(trial_data.model_features.keys())
    logging.info('zscore_all_data feature_keys: %s', feature_keys)
    for data_type in feature_keys:
      if data_type == 'ones':
        continue
      all_data = self.get_all_feature_data(data_type)
      mean, std = find_mean_std(all_data)
      self._feature_mean[data_type] = mean
      self._feature_std[data_type] = std
      self.zscore_all_features(data_type, mean, std)

  def save_zscore_data(self, filename: str):
    """Save the experiments's mean and standard deviation used for z-scoring.

    This is needed so we can adjust the data we use for inference. Note: The
    zscoring is done by data type, so the mean and standard deviation are unique
    for each data type (eeg, intensity, etc).  But really they should be
    independent for each column of each feature.

    Args:
      filename: where to put the pickle file with the saved parameters.
    """
    with tf.io.gfile.GFile(filename, 'w') as fp:
      pickle.dump({'mean': self._feature_mean,
                   'std': self._feature_std},
                  fp)

  def write_all_data(self, tf_dir):
    """Write out all the trial data for this experiment.

    Args:
      tf_dir: A directory where the TFRecord files will be stored.

    Returns:
      A list of all the data files that have been created.
    """
    all_files = []
    for trial in self.iterate_trials():
      all_files.append(trial.write_data_as_tfrecords(tf_dir))
    return all_files


def find_mean_std(data_list: List[np.ndarray],
                  columnwise: bool = False) -> Tuple[Union[np.ndarray, float],
                                                     Union[np.ndarray, float]]:
  """Given a list of np arrays, find their joint mean and standard deviation.

  Args:
    data_list: a list of np arrays with the data to normalize.
    columnwise: Flag to indicate whether to compute mean and std along
                columns or across the entire matrix.
  Returns:
    A tuple with the data's mean and standard deviation.
  """
  data_sum = 0.0
  count = 0
  for d in data_list:
    if columnwise:
      data_sum += np.sum(d, axis=0, keepdims=True)
      count += d.shape[0]
    else:
      data_sum += np.sum(d)
      count += np.prod(d.shape)
  data_mean = data_sum / count
  sum2 = 0.0
  for d in data_list:
    d = d - data_mean
    if columnwise:
      sum2 += np.sum(d*d, axis=0, keepdims=True)
    else:
      sum2 += np.sum(d*d)
  data_std = np.sqrt(sum2/count)
  return data_mean, data_std


def normalize_data(a: np.ndarray,
                   data_mean: Union[float, np.ndarray],
                   data_std: Union[float, np.ndarray]) -> np.ndarray:
  """Remove the mean from an np array and normalize by the standard deviation.

  Make sure we don't divide by zero, which might happen with artificial data.

  Args:
    a: the array of data
    data_mean: the mean we want to remove (perhaps computed from different data)
    data_std: the current standard deviation to divide the data by.

  Returns:
    A new array, with the mean subtracted and divided by the data_std.
  """
  centered = a - data_mean
  if np.max(np.abs(data_std)) > 0.0:
    return centered / data_std
  return centered


#################   Write out the data in TFRecords format  ####################


def convert_data_to_tfrecords(filename: str, data_dict: Dict[str, np.ndarray]):
  """Convert a dictionary of audio and brain signals into TFRecord format.

  The dictionary translates between feature type, and points to an array per
  feature with the data.  Each data is num_frames x num_features in size, and
  all data dictionaries must have the same type. Each TFRecord has a list of
  feature names and one frame of data. There are num_frames records in the
  output file.

  Args:
    filename: Where to dump the data in TFRecord format
    data_dict: A dictionary of data
  """
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  assert_type('Input data_dict', data_dict, dict)
  first_key = list(data_dict.keys())[0]
  first_data = data_dict[first_key]
  num_examples = first_data.shape[0]
  for k in data_dict:
    if data_dict[k].shape[0] != num_examples:
      raise ValueError('Inconsistent shapes: %s %s vs %s %s' %
                       (k, data_dict[k].shape, first_key, first_data.shape))
    if len(data_dict[k].shape) != 2:
      raise ValueError('Not 2d shape for key %s: %s' % (k, data_dict[k].shape))

  logging.info('Writing %d TFRecords data to %s', num_examples, filename)
  with tf.io.TFRecordWriter(filename) as writer:
    for row in range(num_examples):
      feature_dict = {}
      for k in data_dict:
        data = data_dict[k]
        feature = None
        # if type(data[row, 0]) == np.str or type[data[row, 0]:
        if data.dtype == str or data.dtype == '|S1':
          feature = _bytes_feature(data[row])
        elif isinstance(data, np.ndarray):
          if data.dtype == np.float64 or data.dtype == np.float32:
            feature = _float_feature(data[row, :].tolist())
          elif data.dtype == np.int64 or data.dtype == np.int32:
            feature = _int64_feature(data[row, :].tolist())
        if not feature:
          raise ValueError('Can\'t convert %s data to TFRecord: %s %s' %
                           (k, type(data), data.dtype))
        feature_dict[k] = feature
      example = tf.train.Example(
          features=tf.train.Features(feature=feature_dict))
      writer.write(example.SerializeToString())


def discover_feature_shapes(
    tfrecord_file_name: str) -> Dict[str, tf.io.FixedLenFeature]:
  """Read a TFRecord file, parse one TFExample, and return the structure.

  Args:
    tfrecord_file_name: Where to read the data (just one needed)

  Returns:
    A dictionary of names and tf.io.FixedLenFeatures suitable for
    tf.parse_example.

  Raises:
      ValueError or TypeError for bad parameter values.
  """
  assert_type('tfrecord_file_name', tfrecord_file_name, str)

  dataset = tf.data.TFRecordDataset(tfrecord_file_name)

  for next_record in dataset.take(1):
    a_record = next_record
  if not a_record:
    raise ValueError('Could not read any data from tfrecord file.')
  an_example = tf.train.Example.FromString(a_record.numpy())

  assert_type('an_example', an_example, tf.train.Example)

  feature_keys = list(an_example.features.feature.keys())

  shapes = {}
  for k in feature_keys:
    feature_list = an_example.features.feature[k]
    if feature_list.float_list.value:
      dimensionality = len(feature_list.float_list.value)
      feature_type = tf.float32
    elif feature_list.int64_list.value:
      dimensionality = len(feature_list.int64_list.value)
      feature_type = tf.int64
    elif feature_list.bytes_list.value:
      dimensionality = len(feature_list.byte_list.value)
      feature_type = tf.str
    shapes[k] = tf.io.FixedLenFeature([dimensionality,], feature_type)
  return shapes


def count_tfrecords(tfrecord_file_name: str) -> Tuple[int, bool]:
  """Count the number of records in a TFRecord file.

  Args:
    tfrecord_file_name: Where to read the tfrecord data.

  Returns:
    A tuple consisting of record_count, error_flag.  The error flag tells if
    the reading failed.
  """
  assert_type('tfrecord_file_name', tfrecord_file_name, str)

  dataset = tf.data.TFRecordDataset(tfrecord_file_name)

  record_count = 0
  for a_record in dataset:
    try:
      an_example = tf.train.Example.FromString(a_record.numpy())
      assert_type('an_example', an_example, tf.train.Example)
      record_count += 1
    except Exception as e:  # pylint: disable=broad-except
      logging.info('Got an error in count_tfrecords: %s', e)
      return record_count, True
  return record_count, False


def read_tfrecords(tfrecord_file_name: str,
                   start_frame: int = 0,
                   frame_count: int = 512) -> Dict[str, np.ndarray]:
  """Read in some of the TFRecord data and return the data in np arrays.

  Args:
    tfrecord_file_name: Where to read the data from
    start_frame: Which frame number to start reading the data.
      (Ignore precedding.)
    frame_count: How many frames to read

  Returns:
    A dictionary, keyed by the data names in the tfrecord file, with
      np.ndarrays for the data.

  Raises:
    TypeError for bad parameter values.
  """
  assert_type('tfrecord_file_name', tfrecord_file_name, str)

  records = {}
  shapes = discover_feature_shapes(tfrecord_file_name)
  for k in shapes:
    records[k] = np.zeros((frame_count, shapes[k].shape[0]), dtype=np.float32)

  dataset = tf.data.TFRecordDataset(tfrecord_file_name)
  current_frame_count = 0
  for a_record in dataset:
    try:
      an_example = tf.train.Example.FromString(a_record.numpy())
      assert_type('an_example', an_example, tf.train.Example)
      feature_keys = list(an_example.features.feature.keys())
      if current_frame_count >= start_frame:
        for k in feature_keys:
          data = an_example.features.feature[k].float_list.value
          records[k][current_frame_count, :] = data
      current_frame_count += 1
      if current_frame_count >= start_frame + frame_count:
        break
    except tf.errors.OutOfRangeError:
      break
  if current_frame_count != frame_count:   # Downsize if not enough data.
    for k in records:
      records[k] = records[k][:current_frame_count, :]
  return records


def transform_tfrecords(
    input_file: str,
    new_tf_dir: str,
    trial_name: str,
    transforms: List[Callable[[Dict[str, np.ndarray],],
                              Tuple[str, np.ndarray]]]) -> str:
  """Transforms a TFRecord file by adding more computed fields.

  Args:
    input_file: A filename pointing to tfrecord data.
    new_tf_dir: Where to put the new data (in tfrecord format).
    trial_name: Name of this trial, which then becomes the filename (after
      adding .tfrecord) for the new data in new_tf_dir.
    transforms: a list of functions which take the data dictionary with all
      the data, and returns a tuple consisting of a new data field name, and
      a new set of data with the same first dimension as the input data.

  Returns:
    The new file name
  """
  record_count, errors = count_tfrecords(input_file)
  if errors:
    raise ValueError('Found errors after reading %d records from %s.' %
                     (record_count, input_file))
  data_dict = read_tfrecords(input_file, frame_count=record_count)
  for transform_lambda in transforms:
    new_name, new_data = transform_lambda(data_dict)
    data_dict[new_name] = new_data

  brain_trial = BrainTrial(trial_name)
  for k, v in data_dict.items():
    brain_trial.add_model_feature(k, v)

  return brain_trial.write_data_as_tfrecords(new_tf_dir)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  app.run(main)
