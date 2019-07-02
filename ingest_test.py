# Lint as: python2, python3
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

"""Tests for ingest.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tempfile

from absl import flags
from absl.testing import absltest

from telluride_decoding import  ingest

import numpy as np
import scipy.io.wavfile
import scipy.signal


class IngestTest(absltest.TestCase):

  def setUp(self):
    super(IngestTest, self).setUp()
    self._test_dir = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/py/telluride_decoding/test_data/')

  def test_brain_signal(self):
    # Test to make sure fix_offset works with 1d signals.
    test_name = 'test_name'
    test_source = 'test_source'
    test_sr = 4
    test_data = np.arange(10)
    s = ingest.BrainSignal(test_name, test_data, test_sr, test_source)
    self.assertEqual(s.name, test_name)
    self.assertEqual(s.data_type, test_source)
    self.assertEqual(s.sr, test_sr)
    self.assertTrue(np.all(np.reshape(test_data, (-1, 1)) == s.signal))

    s.fix_offset(1)
    self.assertEqual(s.signal[0], 4)
    self.assertEqual(s.signal[-1], 9)

    # Test to make sure fix_offset works with 2d signals.
    s = ingest.BrainSignal('test', np.reshape(np.arange(20), (10, -1)), 4)
    s.fix_offset(1)
    self.assertLen(s.signal.shape, 2)
    self.assertEqual(s.signal[0, 0], 8)
    self.assertEqual(s.signal[0, 1], 9)

    # Test some of the parameter checking.
    with self.assertRaises(ValueError):
      s = ingest.BrainSignal(42, test_data, test_sr, test_source)

  def test_memory_brain_data_file(self):
    one_data = np.arange(10) + 100
    two_data = np.arange(10) + 200
    channel_data = {'one': one_data,
                    'two': two_data}
    test_sr = 4
    df = ingest.MemoryBrainDataFile(channel_data, test_sr)
    self.assertEqual(set(df.signal_names), set(channel_data.keys()))
    self.assertEqual(df.signal_fs('one'), test_sr)
    self.assertEqual(df.signal_fs('two'), test_sr)
    self.assertTrue(np.all(df.signal_values('one') == one_data))
    self.assertTrue(np.all(df.signal_values('two') == two_data))

  def test_brain_data_resample(self):
    # Test the downsampling code.
    # Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    fs1000 = 1000
    fs100 = 100
    signal_length = 2
    t1000 = np.reshape(np.linspace(0, signal_length,
                                   signal_length*fs1000, False), (-1, 1))
    t100 = np.reshape(np.linspace(0, signal_length,
                                  signal_length*fs100, False), (-1, 1))
    sig1000 = np.sin(2*np.pi*10*t1000) + 0*np.sin(2*np.pi*20*t1000)
    sig100 = np.sin(2*np.pi*10*t100) + 0*np.sin(2*np.pi*20*t100)
    sig = np.hstack((sig1000, 0*sig1000))   # Make it 2 channels for better test
    sig = np.reshape(sig1000, (-1, 1))

    filtered = ingest.BrainDataFile.resample(sig, fs1000, fs100, antialias=True)
    # alt = resampy.resample(sig, fs1000, fs100, axis=0)

    self.assertLen(t100, filtered.shape[0])
    # We should test over the entire signal length, but perhaps ok to drop the
    # first few samples.  Instead we drop the last second of data.   See this
    # bug: 133091105
    # TODO(malcolmslaney): After bug is fixed, make test longer
    test_range = np.arange(3, len(sig100)-100)
    self.assertTrue(np.allclose(filtered[test_range, 0],
                                sig100[test_range, 0],
                                atol=0.01))

  def test_brain_data_file_edf_ingest(self):
    test_file_name = 'sample.edf'
    data_file = ingest.EdfBrainDataFile(test_file_name)
    self.assertEqual(data_file.filename, test_file_name)

    data_file.load_all_data(self._test_dir)
    self.assertLen(data_file.signal_names, 103)
    test_channel_name = u'Snore'  # Just a random channel
    self.assertIn(test_channel_name, data_file.signal_names)

    self.assertEqual(data_file.signal_fs('TRIG'), 512.0)
    self.assertEqual(data_file.signal_values('TRIG').shape[0], 33792)

  def test_brain_trial(self):
    trial_name = 'subj01_1ksamples.wav'
    trial = ingest.BrainTrial(trial_name)
    self.assertEqual(trial.trial_name, trial_name.split('.')[0])

    trial.load_sound(trial_name, sound_dir=self._test_dir)
    brain_data_file_object = ingest.EdfBrainDataFile('sample.edf')
    trial.load_brain_data(self._test_dir, brain_data_file_object)

    summary = trial.summary_string()
    self.assertIn('103 EEG channels', summary)
    self.assertIn('with 66s of eeg data', summary)
    self.assertIn('1.00006s of audio data', summary)

    channels = [c for c in trial.iterate_brain_channels()]
    self.assertLen(channels, 103)
    self.assertIn('TRIG', [c.name for c in channels])

    trial.compute_intensity()
    trial.compute_spectrogram()
    scalp_channel_names = ('Fp1, Fp2, F3, F4, F7, F8, C3, C4, T7, T8, P3, P4, '
                           'P7, P8, O1, O2').split(', ')
    trial.assemble_brain_data(scalp_channel_names)

    tf_dir = tempfile.mkdtemp()
    tf_file = trial.write_data_as_tfrecords(tf_dir)
    count = ingest.count_tfrecords(tf_file)
    print('Found tfrecords:', count)
    feature_dict = ingest.discover_feature_shapes(tf_file)
    print('Feature dict is:', feature_dict)
    self.assertIn('spectrogram', feature_dict)
    self.assertIn('loudness', feature_dict)
    self.assertIn('eeg', feature_dict)
    self.assertEqual(feature_dict['eeg'].shape, [len(scalp_channel_names),])

    (count, errors) = ingest.count_tfrecords(tf_file)
    print('count_tfrecords:', count, errors)
    self.assertEqual(count, 64)
    self.assertEqual(errors, 0)

  def test_audio_intensity(self):
    audio_fs = 16000  # Samples per second
    loudness_fs = 100  # Samples per second
    f0 = 440  # Hz
    # Apply a Gaussian window to the sinusoid, and make sure the intensity
    # comes out with the same shape.
    window = scipy.signal.gaussian(audio_fs, std=audio_fs/4.0)
    t = np.linspace(0, 1, audio_fs)
    audio_data = np.sin(2*np.pi*t*f0) * window

    loudness = ingest.create_loudness_feature(audio_data, audio_fs, loudness_fs)
    self.assertLen(loudness, loudness_fs)
    loudness = loudness / np.max(loudness)
    expected_loudness = window[np.arange(0, len(window),
                                         audio_fs/loudness_fs,
                                         dtype=np.int32)]**2
    self.assertLess(np.max(np.abs(expected_loudness-loudness)), 0.015)

  def test_audio_spectrogram(self):
    audio_fs = 16000  # Samples per second
    f0 = 6000  # Hz
    window = scipy.signal.gaussian(audio_fs, std=audio_fs/4.0)
    t = np.linspace(0, 1, audio_fs)
    audio_data = np.sin(2*np.pi*t*f0) * window

    segment_size = 128
    n_overlap = 2
    n_trans = 2
    spectrogram, _ = ingest.auditory_spectrogram(audio_data,
                                                 segment_size=segment_size,
                                                 n_overlap=n_overlap,
                                                 n_trans=n_trans,
                                                 smoothing_filter=[1],
                                                )
    # Note: turning on the smoothing filter (the default) moves the peak one
    # bin to the right.
    self.assertEqual(spectrogram.shape[0], 129)
    self.assertEqual(spectrogram.shape[1], 251)
    self.assertEqual(np.argmax(spectrogram[:, 125]),
                     round(f0/(audio_fs/(n_trans*segment_size))))

  def test_mean_std(self):
    a = np.random.randn(3, 5)
    b = np.random.randn(3, 5)
    data_list = [a, b]
    mean, std = ingest.find_mean_std(data_list, columnwise=False)
    both_arrays = np.concatenate((np.reshape(a, (-1,)),
                                  np.reshape(b, (-1,))), axis=0)
    self.assertAlmostEqual(mean, np.mean(both_arrays))
    self.assertAlmostEqual(std, np.std(both_arrays))

    data_list = [ingest.normalize_data(a, mean, std),
                 ingest.normalize_data(b, mean, std)]
    mean, std = ingest.find_mean_std(data_list)
    self.assertAlmostEqual(mean, 0.0)
    self.assertAlmostEqual(std, 1.0)

  def test_mean_std_columnwise(self):
    a = np.random.randn(3, 5)
    b = np.random.randn(3, 5)
    data_list = [a, b]
    mean, std = ingest.find_mean_std(data_list, columnwise=True)
    both_arrays = np.concatenate((a, b), axis=0)
    true_mean = np.mean(both_arrays, axis=0, keepdims=True)
    true_std = np.std(both_arrays, axis=0, keepdims=True)
    np.testing.assert_allclose(true_mean[0], mean[0])
    np.testing.assert_allclose(true_std[0], std[0])
    data_list = [ingest.normalize_data(a, mean, std),
                 ingest.normalize_data(b, mean, std)]
    mean, std = ingest.find_mean_std(data_list, columnwise=True)
    np.testing.assert_allclose(mean[0], np.zeros_like(mean[0]), atol=1e-8)
    np.testing.assert_allclose(std[0], np.ones_like(std[0]))

  def test_find_temporal_offset_via_linear_regression(self):
    test_shift = 1.3
    audio_times = np.arange(0, 5, 1)
    eeg_times = audio_times + test_shift
    eeg_times[0] = math.pi   # Screw up time for first data point
    estimated_time, _ = ingest.find_temporal_offset_via_linear_regression(
        audio_times, eeg_times)
    self.assertAlmostEqual(estimated_time, test_shift, places=5)

  def test_find_temporal_offset_via_histogram(self):
    # Generate a bunch of random triggers, shift them, and see if the histogram
    # algorithm produces the right answer.
    num_triggers = 10
    test_shift = 1.42
    atriggers = np.random.random(num_triggers)
    etriggers = atriggers + test_shift
    atriggers[0] = math.pi
    num_triggers = 10
    atriggers = np.random.random(num_triggers)
    etriggers = atriggers + 1.42
    mode = ingest.find_temporal_offset_via_mode_histogram(atriggers, etriggers,
                                                          fs=100)
    self.assertAlmostEqual(mode, test_shift, delta=0.01)

  def test_brain_experiment(self):
    one_data = np.arange(10) + 100
    two_data = np.arange(10) + 200
    channel_data = {'one': one_data,
                    'two': two_data}
    test_sr = 4
    df = ingest.MemoryBrainDataFile(channel_data, test_sr)
    sound_filename = 'subj01_1ksamples.wav'
    trial_name = ingest.BrainExperiment.delete_suffix(sound_filename, '.wav')
    trial_dict = {trial_name: [sound_filename, df]}
    experiment = ingest.BrainExperiment(trial_dict,
                                        self._test_dir, self._test_dir)
    experiment.load_all_data(self._test_dir, self._test_dir)
    summary = experiment.summary()
    self.assertIn('Found 1 trials', summary)
    self.assertIn('Trial subj01_1ksamples: 2 EEG channels with 2.5s of '
                  'eeg data', summary)
    experiment.z_score_all_data()

  def test_brain_memory_experiment(self):
    fs = 16000
    audio_len = fs
    audio_data = np.random.randn(audio_len)

    frame_sr = 100
    channel_one = np.arange(2*frame_sr)    # Use ints for easier debugging
    channel_two = np.arange(2*frame_sr) + 200
    eeg_data = {'C1': channel_one,
                'C2': channel_two}
    df = ingest.MemoryBrainDataFile(eeg_data, frame_sr)

    trial_two_name = 'trial_2'
    experiment_dict = {trial_two_name:
                           [{'audio_data': audio_data, 'audio_sr': fs}, df],
                      }
    experiment = ingest.BrainExperiment(experiment_dict,
                                        self._test_dir, self._test_dir,
                                        frame_rate=frame_sr)
    self.assertTrue(experiment)
    experiment.load_all_data(self._test_dir, self._test_dir)
    summary = experiment.summary()
    self.assertIn('Found 1 trials', summary)
    self.assertIn('Trial trial_2: 2 EEG channels with 2s of eeg data', summary)
    for trial in experiment.iterate_trials():
      trial.compute_intensity()
      trial.assemble_brain_data(list(eeg_data.keys()))
      # Master copy of EEG data has moved from brain_data to audio_features dict
      brain_data = trial.audio_features['eeg']
      self.assertEqual(brain_data.shape, (2*frame_sr, 2))
    tmp_dir = '/tmp'
    experiment.write_all_data(tmp_dir)
    tf_file = os.path.join(tmp_dir, trial_two_name + '.tfrecords')

    (count, error) = ingest.count_tfrecords(tf_file)
    self.assertEqual(error, 0)
    self.assertEqual(count, frame_sr)

    file_data = ingest.read_tfrecords(tf_file)
    self.assertIn('eeg', file_data)
    self.assertIn('loudness', file_data)

    np.testing.assert_allclose(file_data['eeg'],
                               np.hstack((np.reshape(channel_one[:frame_sr],
                                                     (-1, 1)),
                                          np.reshape(channel_two[:frame_sr],
                                                     (-1, 1)))))

  # Test like above, but include the eeg offset correction.
  def test_brain_memory_experiment2(self):
    fs = 16000
    audio_len = fs
    audio_data = np.random.randn(audio_len)

    frame_sr = 100
    channel_one = np.arange(2*frame_sr)    # Use ints for easier debugging
    channel_two = np.arange(2*frame_sr) + 200
    eeg_data = {'C1': channel_one,
                'C2': channel_two}
    df = ingest.MemoryBrainDataFile(eeg_data, frame_sr)

    trial_two_name = 'trial_2'
    experiment_dict = {trial_two_name:
                           [{'audio_data': audio_data, 'audio_sr': fs}, df],
                      }
    experiment = ingest.BrainExperiment(experiment_dict,
                                        self._test_dir, self._test_dir,
                                        frame_rate=frame_sr)
    self.assertTrue(experiment)
    experiment.load_all_data(self._test_dir, self._test_dir)
    summary = experiment.summary()
    self.assertIn('Found 1 trials', summary)
    self.assertIn('Trial trial_2: 2 EEG channels with 2s of eeg data', summary)
    for trial in experiment.iterate_trials():
      trial.compute_intensity()
      trial.fix_eeg_offset(1.0)
      trial.assemble_brain_data(list(eeg_data.keys()))
      # Master copy of EEG data has moved from brain_data to audio_features dict
      brain_data = trial.audio_features['eeg']
      # Now the eeg size is shorter, due to fix_eeg_offset above.
      self.assertEqual(brain_data.shape, (frame_sr, 2))
    tmp_dir = '/tmp'
    experiment.write_all_data(tmp_dir)
    tf_file = os.path.join(tmp_dir, trial_two_name + '.tfrecords')

    (count, error) = ingest.count_tfrecords(tf_file)
    self.assertEqual(error, 0)
    self.assertEqual(count, frame_sr)

    file_data = ingest.read_tfrecords(tf_file)
    print('Read in data and found keys:', list(file_data.keys()))
    self.assertIn('eeg', file_data)
    self.assertIn('loudness', file_data)

    np.testing.assert_allclose(file_data['eeg'],
                               np.hstack((np.reshape(channel_one[frame_sr:],
                                                     (-1, 1)),
                                          np.reshape(channel_two[frame_sr:],
                                                     (-1, 1)))))

  def test_local_file_copy(self):
    sound_filename = 'tapestry.wav'
    full_filename = os.path.join(self._test_dir, sound_filename)
    with ingest.LocalCopy(full_filename) as fn:
      sound_fs, sound_data = scipy.io.wavfile.read(fn)
    self.assertEqual(sound_fs, 16000)
    self.assertEqual(sound_data.shape[0], 50381)


if __name__ == '__main__':
  absltest.main()
