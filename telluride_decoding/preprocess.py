# Copyright 2020 Google Inc.
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

"""Preprocesses multivariate stimulus/brain data prior to shuffling.

This code generates classes that implement several data preprocessing steps
prior to shuffling via the tf.data.dataset operators (in preparation for
stochastic gradient descent).  All preprocessing steps are optional and are
executed in the following order:
  1. high-pass filtering
  2. low-pass filtering
  3. resampling (offline only)
  3. re-referencing
  4. channel selection
  4. normalization
  5. temporal context addition

Note, resampling should only be used for offline analysis. Other functions like
glitch removal will be included later.

  Typical usage example:

  p = preprocess.Preprocessor('cgrid', fs_in=500, fs_out=100,
                              highpass_cutoff=0.5, highpass_order=3,
                              ref_channels = [[11], [4]],
                              channels_to_ref = [range(7), range(7, 14)],
                              pre_context=0, post_context=300)

  processed_eeg = p.process(eeg)
"""

import re

from absl import flags
from absl import logging
import numpy as np
import scipy.signal

FLAGS = flags.FLAGS


class Preprocessor(object):
  """Routines to implement data preprocessing.

  Data preprocessing steps implemented prior to shuffling.  Includes high/low-
  pass filtering, re-referencing, channel selection, normalization, temporal
  context addition.

  Attributes:
    name: A string indicating the object name.
    fs_in: A float specifying the original sample rate of the data.
    fs_out: A float specifying the final sample rate of the data.
    highpass_cutoff: A float specifying the high-pass filter cutoff.
    highpass_order: An integer specifying the high-pass filter order.
    lowpass_cutoff: A float specifying the low-pass filter cutoff.
    lowpass_order: An integer specifying the low-pass filter order.
    ref_channels: A list containing a list(s) of reference channels that
      correspond to the parallel lists in channels_to_ref.
    channels_to_ref: A list containing a list(s) of channels to be referenced
      that correspond to the parallel lists in ref_channels.
    channel_numbers: A list or string of channels to be retained.
    data_mean: A float containing the value to be used for demeaning.
    data_std: A float containing the value to be used for normalization.
    pre_context: An integer specifying the pre-stimulus lags in samples.
    post_context: An integer specifying the post-stimulus lags in samples.
  """
  # TODO Add glitch removal

  def __init__(self,
               name,
               fs_in,
               fs_out,
               highpass_cutoff=0,
               highpass_order=4,
               lowpass_cutoff=0,
               lowpass_order=4,
               ref_channels=None,
               channels_to_ref=None,
               channel_numbers=None,
               data_mean=0,
               data_std=1,
               pre_context=0,
               post_context=0):
    """Specifies desired parameters up front.  Enter 0 or None to disable."""
    self.check_params(name, fs_in, fs_out, highpass_cutoff, highpass_order,
                      lowpass_cutoff, lowpass_order, ref_channels,
                      channels_to_ref, channel_numbers, data_std, pre_context,
                      post_context)
    self._fs_in = fs_in
    if '(' in name:
      self.init_from_string(fs_in, name)
    self._name = name
    self._fs_out = fs_out
    self.init_highpass(highpass_cutoff, highpass_order)
    self.init_lowpass(lowpass_cutoff, lowpass_order)
    self._ref_channels = ref_channels
    self._channels_to_ref = channels_to_ref
    self.init_channel_numbers(channel_numbers)
    self._data_mean = data_mean
    self._data_std = data_std
    self._pre_context = pre_context
    self._post_context = post_context
    self.context_reset()
    self._next_frame_idx = 0

  def init_highpass(self, highpass_cutoff, highpass_order):
    """Initializes the high-pass filter coefficients."""
    if highpass_cutoff > 0:
      self._highpass_cutoff = highpass_cutoff
      self._highpass_order = highpass_order
      logging.info('High-pass filtering the data with the 3dB point at %gHz.',
                   highpass_cutoff)
      self._highpass_sos = scipy.signal.butter(highpass_order, highpass_cutoff,
                                               'hp', output='sos',
                                               fs=self.fs_in)
      self._highpass_state = None  # to be created later when we know sizes
    else:
      self._highpass_sos = None

  def init_lowpass(self, lowpass_cutoff, lowpass_order):
    """Initializes the low-pass filter coefficients."""
    if lowpass_cutoff > 0 or self._fs_out < self._fs_in:
      nyquist = self._fs_out / 2
      if lowpass_cutoff > nyquist or (self._fs_out < self._fs_in and
                                      lowpass_cutoff == 0):
        lowpass_cutoff = 0.75 * nyquist
        lowpass_order = 10
        print('Using %gHz low-pass filter to prevent aliasing'
              % lowpass_cutoff)
      self._lowpass_cutoff = lowpass_cutoff
      self._lowpass_order = lowpass_order
      logging.info('Low-pass filtering the data with the 3dB point at %gHz.',
                   lowpass_cutoff)
      self._lowpass_sos = scipy.signal.butter(lowpass_order, lowpass_cutoff,
                                              'lp', output='sos', fs=self.fs_in)
      self._lowpass_state = None  # to be created later when we know sizes
    else:
      self._lowpass_sos = None

  def init_channel_numbers(self, channel_numbers):
    """Parses the channel specification string."""
    if isinstance(channel_numbers, int):
      self._channel_numbers = [channel_numbers]

    elif isinstance(channel_numbers, list):
      self._channel_numbers = channel_numbers

    elif isinstance(channel_numbers, str):
      if ',' in channel_numbers:
        channel_numbers = channel_numbers.split(',')
      else:
        channel_numbers = [channel_numbers,]

      def expand_number_range(range_list):
        """Expands any requests that include a range (x-y).  Inclusive range."""
        if '-' in range_list:
          range_list = range_list.split('-')
          assert len(range_list) == 2
          range_list = list(range(int(range_list[0]), int(range_list[1])+1))
        else:  # Not a range, just return the number as a list.
          range_list = [int(range_list),]
        return range_list

      # Squash list of lists to a 1-D numpy array.
      channel_numbers = np.concatenate([expand_number_range(r) for r
                                        in channel_numbers])
      self._channel_numbers = np.unique(channel_numbers).tolist()
      print('channel numbers: ', self._channel_numbers)
    else:
      self._channel_numbers = None

  @property
  def name(self):
    return self._name

  @property
  def fs_in(self):
    return self._fs_in

  @property
  def fs_out(self):
    return self._fs_out

  @property
  def highpass_cutoff(self):
    return self._highpass_cutoff

  @property
  def highpass_order(self):
    return self._highpass_order

  @property
  def lowpass_cutoff(self):
    return self._lowpass_cutoff

  @property
  def lowpass_order(self):
    return self._lowpass_order

  @property
  def ref_channels(self):
    return self._ref_channels

  @property
  def channels_to_ref(self):
    return self._channels_to_ref

  @property
  def channel_numbers(self):
    return self._channel_numbers

  @property
  def data_mean(self):
    return self._data_mean

  @property
  def data_std(self):
    return self._data_std

  @property
  def pre_context(self):
    return self._pre_context

  @property
  def post_context(self):
    return self._post_context

  def __repr__(self):
    return ('Preprocessor(name={}, fs_in={}, fs_out={}, highpass_cutoff={}, ' +
            'highpass_order={}, lowpass_cutoff={}, lowpass_order={}, ' +
            'ref_channels={}, channels_to_ref={}, channel_numbers={} ' +
            'data_mean={}, data_std={}, pre_context={}, post_context={})'
           ).format(self.name, self.fs_in, self.fs_out, self.highpass_cutoff,
                    self.highpass_order, self.highpass_cutoff,
                    self.highpass_order, self._ref_channels,
                    self.channels_to_ref, self.channel_numbers, self.data_mean,
                    self.data_std, self.pre_context, self.post_context)

  def check_params(self, name, fs_in, fs_out, highpass_cutoff, highpass_order,
                   lowpass_cutoff, lowpass_order, ref_channels, channels_to_ref,
                   channel_numbers, data_std, pre_context, post_context):
    """Checks correctness of parameters passed as input."""
    if not isinstance(name, str):
      raise TypeError('name must be a string, not %s' % name)
    if fs_in <= 0:
      raise ValueError('fs_in should not be less than 0.')
    if fs_out <= 0:
      raise ValueError('fs_out should not be less than 0.')
    if highpass_cutoff < 0:
      raise ValueError('highpass_cutoff should not be less than 0.')
    if highpass_order < 0:
      raise ValueError('highpass_order should not be less than 0.')
    if lowpass_cutoff < 0:
      raise ValueError('lowpass_cutoff should not be less than 0.')
    if lowpass_order < 0:
      raise ValueError('lowpass_order should not be less than 0.')
    if not isinstance(ref_channels, list) and ref_channels is not None:
      raise ValueError('ref_channels must be a list.')
    if not isinstance(channels_to_ref, list) and channels_to_ref is not None:
      raise ValueError('channels_to_ref must be a list.')
    if not isinstance(channel_numbers, (list, str)) \
    and channel_numbers is not None:
      raise ValueError('c    hannel_numbers must be a list.')
    if data_std <= 0:
      raise ValueError('data_std must be greater than 0.')
    if pre_context < 0:
      raise ValueError('pre_context should not be less than 0.')
    if post_context < 0:
      raise ValueError('post_context should not be less than 0.')

  def check_dims(self, data):
    """Checks that the data is two dimensional.

    Args:
      data: A numpy array of size num_frames x num_channels.
    """
    if np.ndim(data) != 2:
      raise ValueError('Input data must be a two dimensional numpy array. '
                       'Data received has shape (%g, %g).' % data.shape)

  def highpass_filter_reset(self, data):
    """Resets the saved state of the high-pass filter to initial value.

    Args:
      data: A numpy array of size num_frames x num_channels.
    """
    zi = scipy.signal.sosfilt_zi(self._highpass_sos)
    print('Zi shape: ', zi.shape, data.shape)
    self._highpass_state = data[0, :] * np.repeat(zi[:, :, np.newaxis],
                                                  data.shape[1], axis=2)
    logging.info('Resetting the high-pass filter state.')

  def highpass_filter(self, data, reset=False):
    """High-pass filters the data along each channel.

    Args:
      data: A numpy array of size num_frames x num_channels.
      reset: A flag indicating whether to reset the state before processing.

    Returns:
      The filtered data.
    """
    data = np.asarray(data)
    if self._highpass_sos is not None:
      if self._highpass_state is None or reset:
        self.highpass_filter_reset(data)
      data, self._highpass_state = scipy.signal.sosfilt(
          self._highpass_sos, data, zi=self._highpass_state, axis=0)
      return data
    return data

  def lowpass_filter_reset(self, data):
    """Resets the saved state of the low-pass filter to initial value.

    Args:
      data: A numpy array of size num_frames x num_channels.
    """
    zi = scipy.signal.sosfilt_zi(self._lowpass_sos)
    self._lowpass_state = data[0, :] * np.repeat(zi[:, :, np.newaxis],
                                                 data.shape[1], axis=2)
    logging.info('Resetting the low-pass filter state.')

  def lowpass_filter(self, data, reset=False):
    """Low-pass filters the data along each channel.

    Args:
      data: A numpy array of size num_frames x num_channels.
      reset: A flag indicating whether to reset the state before processing.

    Returns:
      The filtered data.
    """
    data = np.asarray(data)
    if self._lowpass_sos is not None:
      if self._lowpass_state is None or reset:
        self.lowpass_filter_reset(data)
      data, self._lowpass_state = scipy.signal.sosfilt(
          self._lowpass_sos, data, zi=self._lowpass_state, axis=0)
      return data
    return data

  def resample(self, data):
    """Resample data by a non-integer factor.

    Resamples uniformly-sampled multivariate data by a non-integer factor by
    recalculating the time points and rounding to the nearest neighbor. When
    downsampling, the data is automatically low-pass filtered below 75% of the
    Nyquist frequency to prevent aliasing.  Upsampling results in repetition of
    certain samples.

    Args:
      data: A numpy array of size num_frames x num_channels.

    Returns:
      New data array with a lower sample rate.

    Raises:
      ValueError for bad parameter values.
    """
    # Resample signal
    if self._fs_out != self._fs_in:

      # Check that new sample rate is compatable with batch size
      if self._next_frame_idx != 0:
        raise ValueError('New sample rate incompatable with batch size.')

      # Input dimensions
      frames_in = data.shape[0]  # samples
      len_data = float(frames_in) / self._fs_in  # seconds

      # Output dimensions
      frames_out = int(np.round(len_data*self._fs_out))  # samples
      delta_out = 1.0 / self._fs_out  # seconds

      # Predict next frame
      self._next_frame_idx = int(np.round(frames_out*delta_out*self._fs_in)) - \
                         frames_in

      # Compute indices to output
      idx_out = np.round(np.arange(frames_out)*delta_out*self._fs_in)

      # Sample data using output indices
      data_out = np.zeros((frames_out, data.shape[1]))
      for i in range(frames_out):
        idx = min(frames_in - 1, idx_out[i])
        data_out[i, :] = data[int(idx), :]

    else:

      # No resampling
      data_out = data

    return data_out

  def reref_data(self, data):
    """Re-references the data to the average of a group of channels.

    Reference channels and channels to be referenced must be input as a list of
    lists so that different groups of channels can be referenced independently.
    For example, we may wish to re-reference the left ear channels to the right
    mastoid and vice versa. Thus, the outer list indicates the ear (left, right)
    and the inner list indicates the corresponding channel numbers.

    Args:
      data: A numpy array with shape num_frames x num_channels.

    Returns:
      The re-referenced data.
    """
    if self._ref_channels is not None or self._channels_to_ref is not None:
      if self._ref_channels is None:  # Re-reference to global average.
        self._ref_channels = [range(data.shape[1])]
      if self._channels_to_ref is None:  # Re-reference all channels.
        self._channels_to_ref = [range(data.shape[1])]
      d = np.copy(data)  # create copy to avoid using re-referenced data
      for ref, chans in zip(self._ref_channels, self._channels_to_ref):
        data[:, list(chans)] -= np.mean(d[:, list(ref)], axis=1, keepdims=True)
    return data

  def select_channels(self, data):
    """Retains the desired channels.

    Args:
      data: A numpy array with shape num_frames x num_channels.

    Returns:
      Just the desired channels of the data.
    """
    if self._channel_numbers:
      return data[:, self._channel_numbers]
    return data

  def find_mean_std(self, data):
    """Finds the mean and standard deviation of the data across all channels.

    Args:
      data: A numpy array of size num_frames x num_channels.
    """
    if self._data_mean is None:
      self._data_mean = np.mean(data)
    if self._data_std is None:
      self._data_std = np.std(data)

  def normalize_data(self, data):
    """Removes the mean and normalizes by the standard deviation.

    Args:
      data: A numpy array of size num_frames x num_channels.

    Returns:
      The normalized data.
    """
    self.find_mean_std(data)
    return (data - self._data_mean) / self._data_std

  def shift(self, arr, shift_amt, pre_context, post_context):
    """Shifts the array by the amount specified.

    Returns the exact size which will be used in the context filled array
    snipping off pre and post context.

    Args:
      arr: The array to be shifted.
      shift_amt: Amount by which to shift the data array.
      pre_context: Pre-context lags to be prepended to the data.
      post_context: Post-context lags to be appended to the data.

    Returns:
      result: Array shifted by given amount.
    """
    result = arr[pre_context - shift_amt:arr.shape[0] - post_context -
                 shift_amt, :]
    return result

  def add_context(self, data):
    """Add pre and post temporal context to data.

    Args:
      data: Input data as 2d numpy array.

    Returns:
      context_filled_data: Data with pre and post context. The last post_context
      number of samples appear in the next batch.
    """
    pre_context = self._pre_context
    post_context = self._post_context
    if pre_context == 0 and post_context == 0:
      return data
    num_features = data.shape[1]
    if self._context_state is None:
      self._context_state = np.zeros((pre_context, num_features))
    data = np.concatenate((self._context_state, data))
    self._context_state = data[-(pre_context + post_context):, :]
    num_rows = data.shape[0] - pre_context - post_context
    num_columns = num_features * (pre_context + post_context + 1)
    context_filled_data = np.empty((num_rows, num_columns), dtype=float)
    context_filled_data[:, (pre_context) * num_features:(pre_context + 1) *
                        num_features] = data[pre_context:data.shape[0] -
                                             post_context, :]
    for shift_amt in range(1, pre_context + 1):
      result = self.shift(data, shift_amt, pre_context, post_context)
      context_filled_data[:, (pre_context - shift_amt) *
                          num_features:(pre_context - shift_amt + 1) *
                          num_features] = result
    for shift_amt in range(1, post_context + 1):
      result = self.shift(data, -(shift_amt), pre_context, post_context)
      context_filled_data[:, (pre_context + shift_amt) *
                          num_features:(pre_context + shift_amt + 1) *
                          num_features] = result
    return context_filled_data

  def context_reset(self):
    """Resets the saved state of the context."""
    self._context_state = None
    logging.info('Resetting the context')

  def process(self, data, reset=False):
    """Implements all preprocessing steps for a batch of data.

    Args:
      data: A Numpy array of data of size num_frames x num_channels.
      reset: A boolean indicating whether to reset filter states.

    Returns:
      The preprocessed data.
    """
    data = np.asarray(data)
    self.check_dims(data)
    data = self.highpass_filter(data, reset=reset)
    data = self.lowpass_filter(data, reset=reset)
    data = self.resample(data)
    data = self.reref_data(data)
    data = self.select_channels(data)
    data = self.normalize_data(data)
    data = self.add_context(data)
    return data

  def init_from_string(self, fs_in, param_string):
    """Initializes this object from a parameter string.

    The parameter string has the form:
      feature_name(key=val;key=val;key=val;*)
    This function is called if the normal init function is called with a feature
    name that includes parameters (indicated by parenthesis).

    Args:
      fs_in: Mandatory frame rate for the feature.
      param_string: A parameter string as described above.
    """
    if '(' in param_string:
      name_params_re = re.compile(r'(\w*)\((.*)\)$')
      pieces = name_params_re.match(param_string)
      name = pieces.group(1)
      params = pieces.group(2)
      param_list = params.split(';')
      param_dict = {}
      for param in param_list:
        if '=' not in param:
          raise ValueError('preprocess param %s missing a value.' % param)
        k, v = param.split('=', 1)
        if v.isdigit():
          v = int(v)
        else:
          try:
            v = float(v)
          except ValueError:
            pass
        param_dict[k] = v
      self._name = name
      self.init_highpass(param_dict['highpass_cutoff'],
                         param_dict['highpass_order'])
      self.init_channel_numbers(param_dict['channel_numbers'])
    else:
      self.__init__(self, fs_in, param_string)


class AudioFeatures(object):
  """Routines to implement audio feature extraction.

  Audio feature extraction steps implemented as part of audio preprocessing.
  Includes RMS intensity calculation, non-integer resampling and dynamic range
  compression (optional).

  Attributes:
    name: A string indicating the object name.
    fs_in: A float specifying the original sample rate of the data.
    fs_out: A float specifying the final sample rate of the data.
    window: A float specifying the number of neighboring frames to average when
      resampling the audio intensity.  Values>1 result in overlap between
      neighboring frames and increased smoothing.
    exponent: A float specifying the exponent used to scale the RMS intensity.
      Values<1 result in dynamic range compression. Suggested value: log10(2).
    buff: A numpy array used as a buffer for averaging the initial values.
  """

  def __init__(self, name, fs_in, fs_out, window=1, exponent=1, buff=None):
    """Specifies desired parameters up front."""
    self.check_params(name, fs_in, fs_out, window)
    self._name = name
    self._fs_in = fs_in
    self._fs_out = fs_out
    self._window = window
    self._exponent = exponent
    self._buff = buff

  def check_params(self, name, fs_in, fs_out, window):
    """Checks correctness of parameters passed as input."""
    if not isinstance(name, str):
      raise TypeError('name must be a string, not %s' % name)
    if fs_in <= 0:
      raise ValueError('fs_in should not be less than 0.')
    if fs_out <= 0:
      raise ValueError('fs_out should not be less than 0.')
    if window <= 0:
      raise ValueError('window must be greater than than 0.')

  def audio_resample(self, data):
    """Resample audio by a non-integer factor by averaging neighboring samples.

    Given audio data at a sample rate of fs_in, compute the moving average over
    a specified window to resample to a rate of fs_out.  If window > 1, the
    coutput frames are computed using overlapping samples, resulting in greater
    smoothing.  This can help reduce the effects of aliasing.

    This routine was adapted from the algorithm CreateLoudnessFeature from the
    Telluride Decoding Toolbox (http://www.ine-web.org/software/decoding/).

    Args:
      data: A numpy array of size frames_in x num_channels.

    Returns:
      data_out: The resampled data as a numpy array.
    """

    # Get half window size in seconds.
    half_window_size = 0.5 * self._window / self._fs_out

    # Concatenate and update buffer.
    if self._buff is not None:
      data = np.concatenate((self._buff, data), axis=0)
      tau = self._buff.shape[0]
    else:
      tau = 0
    self._buff = data[-int(self._fs_in * half_window_size):, :]

    # Get i/o data dimensions.
    frames_in = data.shape[0]
    frames_out = int(round((frames_in - tau) / self._fs_in * self._fs_out))

    # Resample data via moving average.
    data_out = np.zeros((frames_out, data.shape[1]))
    if self._fs_out < self._fs_in or self._window > 1:
      for i in range(frames_out):
        t = float(i) / self._fs_out  # center of window in seconds
        t1 = int(max(0, round(self._fs_in * (t - half_window_size)) + tau))
        t2 = int(min(frames_in,
                     round(self._fs_in * (t + half_window_size)) + tau))
        data_out[i, :] = np.mean(data[t1:t2, :], axis=0)

    else:

      data_out = data

    return data_out

  def compute_intensity(self, data):
    """Compute the RMS intensity of an audio signal.

    Computes the RMS intensity of an audio signal by averaging the signal power
    over a moving window and taking the square root of the resampled signal.
    Optionally, dynamic range compression can applied by raising the intensity
    to the power of some .  It is suggested to use an exponent of log10(2) to
    model human auditory perception (Stevens, 1955).

    Args:
      data: A numpy array of size frames_in x num_channels.

    Returns:
      The intensity feature as a numpy array.
    """
    # Compute RMS intensity.
    data = self.audio_resample(data**2)**0.5

    # Apply dynamic range compression.
    return data**self._exponent

  # Get the spectrogram code from this colab
  # TODO: Move this to external colab after release.
  def compute_spectrogram(self, wave, segment_size=128, n_overlap=8, n_trans=4,
                          smoothing_filter=(.2, 1, .2)):
    """Compute an auditory spectrogram of a waveform like the Auditory Toolbox.

    This spectrogram is vaguely auditory-like.  It includes a preemphasis filter
    to raise the level of the high-frequencies, and a simple loudness mapping to
    compress the dynamic range of the output.  Auditory Toolbox is defined here:
    https://engineering.purdue.edu/~malcolm/interval/1998-010/

    Args:
      wave: the 1-D audio waveform in numpy format
      segment_size: number of data points to grab for each frame.
      n_overlap: n_overlap: number of frames overlapping a point.  Frame step is
        segment_size / nlap samples per frame (probably 2 or 4).
      n_trans: factor by which the FFT transform is bigger than a segment
      smoothing_filter: Filter by which to average local result and give a
        smoother_result. Defaults to [.2, 1, .2]. Set to [1] for no smoothing.

    Returns:
      A spectrogram 'array' with fourth root of power, filter smoothed and
      formatted for display.  Along with an array of frequencies that are the
      center of each spectrogram row (from 0 to 0.5, units of pi).
    """
    wave = np.squeeze(wave).astype(np.float32)
    if len(wave.shape) != 1:
      raise ValueError('Wave.shape wrong:' + str(wave.shape))
    premph_wave = scipy.signal.lfilter([1, -0.95], [1], wave)

    f, _, spectrum = scipy.signal.stft(premph_wave, fs=1.0, window='hamming',
                                       nperseg=segment_size,
                                       noverlap=segment_size - (segment_size/
                                                                n_overlap),
                                       nfft=segment_size*n_trans,
                                       return_onesided=True)
    spectrum = np.real(spectrum * np.conj(spectrum))  # back to power to smooth
    spectrum = scipy.signal.lfilter(smoothing_filter, [1], spectrum, axis=0)
    spectrum = scipy.signal.lfilter(smoothing_filter, [1], spectrum, axis=1)

    # Finally, compress with square root of amplitude (fourth root of power)
    off = 0.0001*np.max(spectrum)   # low end stabilization offset,
    spectrum = (off+spectrum)**0.25 - off**0.25  # better than a threshold hack!
    spectrum = 255/np.max(spectrum)*spectrum
    return spectrum, f
