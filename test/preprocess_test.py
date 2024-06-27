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

"""Test for telluride_decoding.preprocess."""

from absl.testing import absltest
from absl.testing import parameterized
from matplotlib import pyplot as plt
import numpy as np
import scipy
from telluride_decoding import preprocess
from telluride_decoding.brain_data import TestBrainData
import tensorflow as tf


class PreprocessTest(parameterized.TestCase):
  """Class to test the preprocessing objects."""

  def test_highpass_freq_resp(self):
    """Test the frequency response of the filter."""
    fs_in = 100.0
    fs_out = 100.0
    highpass_cutoff = 2
    highpass_order = 2
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                highpass_cutoff=highpass_cutoff,
                                highpass_order=highpass_order)

    num_frames = 512
    impulse = np.zeros((num_frames, 1))
    # Don't put impulse at zero because of filter initialization
    impulse[1, 0] = 1
    response = p.highpass_filter(impulse, reset=True)
    freq_resp = 20*np.log10(np.abs(np.fft.fft(response, axis=0,
                                              n=2*num_frames)))
    freqs = np.fft.fftfreq(2*num_frames)*fs_in

    with tf.io.gfile.GFile('/tmp/test_highpass_freq_resp.png', mode='w') as fp:
      plt.clf()
      plt.semilogx(freqs[0:num_frames//2], freq_resp[0:num_frames//2])
      plt.ylim([-20, 0])
      plt.plot(highpass_cutoff, -3.02, 'x')
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Response (dB)')
      plt.grid(True, which='both')
      plt.title('%gHz Highpass Filter Test' % highpass_cutoff)
      plt.savefig(fp)

    three_db_point = np.abs(freqs - highpass_cutoff).argmin()
    # Check gain at cutoff frequency
    self.assertAlmostEqual(freq_resp[three_db_point], -3.02, delta=.3)
    # Make sure gain is less than -3dB below the cutoff frequency
    np.testing.assert_array_less(freq_resp[0:three_db_point], -3.02)
    # Make sure gain is better than -3dB above the cutoff frequency
    np.testing.assert_array_less(-3.02, freq_resp[three_db_point+1:num_frames])

  def test_high_pass_by_frame(self):
    """Make sure we get the same result all at once, and sample by sample."""
    fs_in = 100.0
    fs_out = 100.0
    highpass_cutoff = 20
    highpass_order = 2
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                highpass_cutoff=highpass_cutoff,
                                highpass_order=highpass_order)

    num_frames = 1000
    input_data = np.random.rand(num_frames, 1)
    input_data[num_frames//2:] += 1.0   # So there is a discontinuity

    full_result = p.highpass_filter(input_data, reset=True)

    stepwise_result = full_result * 0.0
    p.highpass_filter_reset(input_data)
    for i in range(num_frames):
      stepwise_result[i] = p.highpass_filter(input_data[i:i+1, :])

    with tf.io.gfile.GFile('/tmp/test_highpass_by_frame.png', mode='w') as fp:
      plt.clf()
      plt.plot(full_result[0:40])
      plt.plot(stepwise_result[0:40])
      plt.title('Stepwise Highpass Filter Test')
      plt.savefig(fp)

    np.testing.assert_allclose(full_result, stepwise_result, rtol=1e-07)

  def test_resample(self):
    """Test the resampling code."""
    # Generate a 10Hz sinusoidal signal, sampled at 1kHz.
    fs_in = 1000.0
    fs_out = 100.0
    sig_len = 2  # seconds
    t_in = np.reshape(np.linspace(0, sig_len, int(sig_len*fs_in), False),
                      (-1, 1))
    t_out = np.reshape(np.linspace(0, sig_len, int(sig_len*fs_out), False),
                       (-1, 1))
    sig_in = np.sin(2*np.pi*10*t_in)
    sig_out = np.sin(2*np.pi*10*t_out)
    # Make it 2 channels for better test.
    sig_in = np.hstack((sig_in, -sig_in))
    sig_out = np.hstack((sig_out, -sig_out))
    # Downsample signal to 100Hz.
    p = preprocess.Preprocessor('test', fs_in, fs_out)
    sig_resamp = p.resample(sig_in)
    # Test for equal signal length.
    self.assertEqual(sig_resamp.shape[0], sig_out.shape[0])

  @parameterized.named_parameters(
      ('even', 500, 100, 100, 500),
      ('single', 500, 100, 100, 100))
  def test_downsample_good(self, fs_in, fs_out, batch_size, data_size):
    data = np.reshape(np.arange(data_size), [data_size, 1])
    ds_rate = fs_in/fs_out
    p = preprocess.Preprocessor('test', fs_in, fs_out)
    results = np.empty((0, 1))
    for i in range(0, data_size, batch_size):
      r = p.resample(data[i:(i+batch_size), :])
      results = np.concatenate((results, r), axis=0)
    idx = range(0, data_size, int(round(ds_rate)))
    np.testing.assert_allclose(results, data[idx, :])

  @parameterized.named_parameters(
      ('odd', 500, 100, 97, 500))
  def test_downsample_bad(self, fs_in, fs_out, batch_size, data_size):
    data = np.reshape(np.arange(data_size), [data_size, 1])
    p = preprocess.Preprocessor('test', fs_in, fs_out)
    results = np.empty((0, 1))
    with self.assertRaisesRegex(
        ValueError, 'New sample rate incompatable with batch size.'):
      for i in range(0, data_size, batch_size):
        r = p.resample(data[i:(i+batch_size), :])
        results = np.concatenate((results, r), axis=0)

  def test_reref(self):
    """Test the re-referencing code."""
    # Generate 14-channel synthetic EEG data.
    fs_in = 100.0
    fs_out = 100.0
    num_frames = 500
    num_channels = 14
    input_data = np.random.randn(num_frames, num_channels)
    # Create preprocessor object.
    ref_channels = [[11], [4]]
    channels_to_ref = [range(7), range(7, 14)]
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                ref_channels=ref_channels,
                                channels_to_ref=channels_to_ref)
    # Re-reference the data.
    output_data = p.reref_data(np.copy(input_data))
    # Test that re-referenced data are close to expected values.
    np.testing.assert_allclose(output_data[:, :7],
                               input_data[:, :7]-input_data[:, [11]])
    np.testing.assert_allclose(output_data[:, 7:],
                               input_data[:, 7:]-input_data[:, [4]])

  def test_channel_selector_parsing(self):
    fs_in = 100.0
    fs_out = 100.0
    channel_numbers = '1,3,42,23,30-33'
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                channel_numbers=channel_numbers)
    self.assertEqual(p.channel_numbers, [1, 3, 23, 30, 31, 32, 33, 42])

  def test_channel_selection(self):
    """Test the channel selecting parsing code."""
    fs_in = 100.0
    fs_out = 100.0
    num_frames = 1000
    channel_numbers = '1,3,42,23,30-33'
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                channel_numbers=channel_numbers)

    data = np.ones((num_frames, 64), dtype=np.int32)
    data = np.cumsum(data, axis=1) - 1

    new_data = p.select_channels(data)
    self.assertEqual(list(new_data[0, :]), [1, 3, 23, 30, 31, 32, 33, 42])
    self.assertEqual(list(new_data[-1, :]), [1, 3, 23, 30, 31, 32, 33, 42])

  def test_processing(self):
    """Test preprocessing pipeline."""
    fs_in = 100.0
    fs_out = 100.0
    num_frames = 1000
    highpass_cutoff = 10
    channel_numbers = '1,3,42,23,30-33'
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                channel_numbers=channel_numbers,
                                highpass_cutoff=highpass_cutoff)

    good_channel = 42
    input_data = np.random.rand(num_frames, 64)
    input_data[:, good_channel] = 1
    output_data = p.process(input_data)
    with tf.io.gfile.GFile('/tmp/test_processing.png', mode='w') as fp:
      plt.clf()
      plt.plot(output_data)
      plt.title('Full Processing Test')
      plt.savefig(fp)

    # This checks the output of channel 42, which is the last channel (-1) of
    # the channels selected above.
    np.testing.assert_array_less(np.abs(output_data[100:, -1]), 0.01)
    self.assertEqual(output_data.shape[1], 8)

  def test_decimation(self):
    """Test the decimation code.
    """
    data = np.reshape(np.arange(100), (-1, 1))
    factor = 3  # How much to decimate input data by
    p = preprocess.Preprocessor('decimation', fs_in=16, fs_out=16, 
                                decimate=factor)
    start_frame = 0
    num_frames_to_process = 3
    results = []
    while start_frame < data.shape[0]:
      results.append(p.process(data[start_frame:
                                    start_frame+num_frames_to_process]))
      start_frame += num_frames_to_process

    results = np.concatenate(results, axis=0)
    np.testing.assert_equal(results, 
                            np.reshape(np.arange(0, data.shape[0], factor,
                                                 dtype=float),
                                       (-1, 1)))

  def test_processing_add_context(self):
    """Test case for adding context as we would in live data.

    This assumes we're passing off each second of data to the preprocessing
    mechanism as it arrives. It'll add precontext from the previous frames.
    Post context would not be possible for the last post_context frames.
    """
    fs_in = 100.0
    fs_out = 100.0
    num_secs = 10
    pre_context = 10
    post_context = 5
    num_features = 64
    highpass_cutoff = 0
    total_context = pre_context + 1 + post_context
    all_data = np.random.rand(num_secs * int(fs_in), num_features)
    # Just do the context addition in preprocessing step
    p = preprocess.Preprocessor('test', fs_in, fs_out,
                                highpass_cutoff=highpass_cutoff,
                                pre_context=pre_context,
                                post_context=post_context)
    c_out = np.empty((0, num_features*total_context))
    # Passing in multiple timesteps (batches) of data to ensure that edge
    # effects are handled correctly.
    for i in range(num_secs):
      input_data = all_data[i * int(fs_in):(i + 1) * int(fs_in), :]
      context_filled_data = p.add_context(input_data)
      self.assertEqual(context_filled_data.shape[1],
                       num_features * total_context)
      c_out = np.concatenate([c_out, context_filled_data], axis=0)
    np.testing.assert_array_equal(c_out[pre_context, :],
                                  all_data[:total_context, :].flatten())
    # Test that the pre context and post context using preprocess.py matches
    # with what we get from TestBrainData which uses the tf.signal.frame to
    # automate the addition of pre and post context
    test_brain_data = TestBrainData('input', 'output', fs_in,
                                    repeat_count=1,
                                    pre_context=pre_context,
                                    post_context=post_context)
    test_brain_data.preserve_test_data(all_data, all_data[:, :1], None)
    test_dataset = test_brain_data.create_dataset(mode='program_test')
    for i, _ in test_dataset.take(1):
      input_data_td = i
    np.testing.assert_array_equal(c_out,
                                  input_data_td['input_1'][:-post_context, :])

  def test_parsing(self):
    """Test the preprocessing parameter parsing by creating a preprocessor.

    Make sure that the string representation contains the right parameters.
    """
    fs_in = 100.0
    fs_out = 100.0
    feature_name = 'eeg'
    param_dict = {'channel_numbers': 2,
                  'highpass_order': 6,
                  'highpass_cutoff': 42,
                 }
    param_list = ['{}={}'.format(k, param_dict[k]) for k in param_dict]
    name_string = '{}({})'.format(feature_name, ';'.join(param_list))
    p = preprocess.Preprocessor(name_string, fs_in, fs_out)
    self.assertIn(feature_name, str(p))
    for k, v in param_dict.items():
      if k in ['channel_numbers', 'ref_channels', 'channels_to_ref']:
        self.assertEqual([[2]], getattr(p, f'_{k}'))
      else:
        self.assertEqual(v, getattr(p, f'_{k}'), f'Wrong value {v} for {k}')

  def test_audio_intensity(self):
    fs_in = 16000  # Samples per second
    fs_out = 100  # Samples per second
    f0 = 440  # Hz
    # Apply a Gaussian window to the sinusoid, and make sure the intensity
    # comes out with the same shape.
    window = scipy.signal.windows.gaussian(fs_in, std=fs_in/4.0).reshape(-1, 1)
    t = np.linspace(0, 1, fs_in).reshape(-1, 1)
    audio_data = np.sin(2*np.pi*t*f0) * window

    p = preprocess.AudioFeatures('test', fs_in, fs_out,
                                 window=1, exponent=np.log10(2), buff=None)

    loudness = p.compute_intensity(audio_data)
    self.assertLen(loudness, fs_out)
    loudness = loudness / np.max(loudness)
    expected_loudness = window[np.arange(0, len(window),
                                         fs_in/fs_out,
                                         dtype=np.int32)]**np.log10(2)
    self.assertLess(np.max(np.abs(expected_loudness-loudness)), 0.015)

  def test_audio_spectrogram(self):
    fs_in = 16000
    fs_out = 16000
    f0 = 6000  # Hz
    window = scipy.signal.windows.gaussian(fs_in, std=fs_in/4.0)
    t = np.linspace(0, 1, fs_in)
    audio_data = np.sin(2*np.pi*t*f0) * window

    segment_size = 128
    n_overlap = 2
    n_trans = 2
    p = preprocess.AudioFeatures('test', fs_in, fs_out,
                                 window=1, exponent=np.log10(2), buff=None)
    spectrogram, _ = p.compute_spectrogram(audio_data,
                                           segment_size=segment_size,
                                           n_overlap=n_overlap,
                                           n_trans=n_trans,
                                           smoothing_filter=[1])
    # Note: turning on the smoothing filter (the default) moves the peak one
    # bin to the right.
    self.assertEqual(spectrogram.shape[0], 129)
    self.assertEqual(spectrogram.shape[1], 251)
    self.assertEqual(np.argmax(spectrogram[:, 125]),
                     round(f0/(fs_in/(n_trans*segment_size))))

  def test_init_from_string(self):
    params = {'lowpass_cutoff': 2,
              'lowpass_order': 4}
    fs = 16000
    param_string = ';'.join([f'{k}={params[k]}' for k in params])
    p = preprocess.Preprocessor(f'test({param_string})', fs, fs)

    self.assertEqual(p.name, 'test')
    # Now make sure all the parameters we specified are correctly set in the
    # object.
    for k in params:
      self.assertEqual(getattr(p, f'_{k}'), params[k])

  def test_all(self):
    """A test which shows how to use this class to preprocess data.
    Load three signals: a ground (sine wave), a signal with two components,
    and an impulse.  Then specify the reference channel (the ground) and lowpass
    filter at 2Hz to see the response."""
    fs = 32
    total_frames = 100*fs
    num_dims = 3

    f1 = 0.5
    f2 = 1
    f3 = 2
    signals = np.zeros((total_frames, num_dims))
    t = np.arange(total_frames)/fs

    signals[:, 0] = np.sin(t*2*np.pi*f1)
    signals[:, 1] = signals[:, 0] + np.sin(t*2*np.pi*f2) + np.sin(t*2*np.pi*f3)
    signals[2, 2] += 1  # Impulse, but not at zero, to avoid startup problems.
    
    p =  preprocess.Preprocessor('test(ref_channels=0;channels_to_ref=1;' 
                                 f'lowpass_cutoff={f2};lowpass_order=2)', 
                                 fs, fs)
    
    frames_sent = 0
    result_data = []
    while frames_sent < total_frames:
      # Process 5 frames at a time, to make sure we don't have problems for
      # arbitrary block processing sizes.
      num = min(5, total_frames - frames_sent) 
      result = p.process(signals[frames_sent: frames_sent+num, :])
      result_data.append(result)
      frames_sent += num

    # Assemble all the results to compute the resulting spectrum for testing
    results = np.concatenate(result_data, axis=0)

    freqs = np.fft.fftfreq(total_frames)*fs
    def find_freq(freqs: np.ndarray, f: float):
      """Find the array index corresponding to the desired frequency.
      Args:
        freqs: The frequency of each (FFT) bin
        f: The desired frequency
      Returns:
        The bin number that best corresponds to the desired (FFT) Frequency
      """
      i = np.argmin((freqs-f)**2)
      print(f'Freq {f} is in bin {i} which corresponds to {freqs[i]}Hz')
      return i
    f1_index = find_freq(freqs, f1)
    f2_index = find_freq(freqs, f2)
    f3_index = find_freq(freqs, f3)
    f4_index = find_freq(freqs, 2*f3)

    freq_resp = 20*np.log10(np.abs(np.fft.fft(results, axis=0)))
    freq_resp = freq_resp[:total_frames//2, :]    # Keep positive freqs only
    # Normalize the maximum of each channel's frequency response
    freq_resp -= np.max(freq_resp, axis=0)
      
    if False:
      with open('/tmp/filter_resp.txt', 'w') as fp:
        for i in range(results.shape[0]):
          print(f'{i} {results[i, 0]}, {results[i,1]}, {results[i, 2]}', 
                file=fp)

      with open('/tmp/freq_resp.txt', 'w') as fp:
        for i in range(freq_resp.shape[0]):
          print(f'{i} {freqs[i]}Hz: {freq_resp[i, 0]}, {freq_resp[i,1]}, '
                f'{freq_resp[i, 2]}', 
                file=fp)

    with tf.io.gfile.GFile('/tmp/test_full_response.png', mode='w') as fp:
      # Plot the preprocessor output, in the time domain
      plt.clf()
      plt.plot(results[:200, :])
      plt.savefig(fp)

    with tf.io.gfile.GFile('/tmp/test_full_spectrum.png', mode='w') as fp:
      # Plot the frequency spectrum of each processed signal.
      plt.clf()
      plt.semilogx(freqs[:total_frames//2], freq_resp[:total_frames//2, :])
      plt.ylim([-80, 0])
      plt.plot(f2, -3.02, 'x')
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Response (dB)')
      plt.grid(True, which='both')
      plt.title(f'{f2}Hz Losspass Filter Test')
      plt.legend(('Reference (Gnd)', 'Filtered Signal', 'Impulse Response'))
      plt.savefig(fp)

    # Make sure that ground signal (at f1 Hz) is there, an impulse at f1 bin
    self.assertAlmostEqual(freq_resp[f1_index, 0], 0.00, 0.01)
    # Make sure rest of groound signal is zero (after zeroing out the f1 component)
    freq_resp[f1_index, 0] = -100
    np.testing.assert_array_less(freq_resp[:, 0], -40)

    # Make sure that the peak in the main EEG signal (channel 1) is there
    self.assertAlmostEqual(freq_resp[f2_index, 1], 0)
    freq_resp[f2_index, 1] = -100
    self.assertGreater(freq_resp[f3_index, 1], -10)
    freq_resp[f3_index, 1] = -100
    np.testing.assert_array_less(freq_resp[:, 1], -40)

    # Make sure the filter's frequency response from the impulse dies out as 
    # expected, 3dB at the filter's desired frequency, and then 12dB per octave.
    self.assertAlmostEqual(freq_resp[f2_index, 2], -3.01, places=2)
    # Expect 12dB per octave fall off.
    self.assertAlmostEqual(freq_resp[f3_index, 2], -12.46, places=2)


if __name__ == '__main__':
  absltest.main()
