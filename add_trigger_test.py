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

"""Tests for add_trigger.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import google3

from absl import flags
from absl.testing import absltest

from telluride_decoding import add_trigger

import numpy as np
from six.moves import range


class AddTriggerTest(absltest.TestCase):

  def setUp(self):
    super(AddTriggerTest, self).setUp()
    self._test_data = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/py/telluride_decoding/test_data/')

  def test_intervals(self):
    def interval_test(duration=10, minimum_interval=0.5, number=8,
                      include_zero=True):
      random_times = add_trigger.random_times(duration, number,
                                              minimum_interval, include_zero)
      sorted_times = np.sort(random_times[:number])
      if include_zero:
        self.assertEqual(sorted_times[0], 0.0)
      intervals = sorted_times[1:number] - sorted_times[0:(number-1)]
      self.assertTrue(np.all(intervals > minimum_interval))
      if include_zero:
        self.assertEqual(random_times[0], 0.0)
      self.assertTrue(np.all((random_times[1:] - random_times[0:-1]) >
                             minimum_interval))

    for _ in range(100):
      interval_test()

    with self.assertRaises(ValueError):
      duration = 10
      minimum_interval = 1.0
      number = 20
      include_zero = True
      add_trigger.random_times(duration, number, minimum_interval,
                               include_zero=include_zero)

  def test_add_triggers(self):
    audio_file = os.path.join(self._test_data, 'tapestry.wav')
    [audio_fs, audio_signal] = add_trigger.read_audio_wave_file(audio_file)
    self.assertLen(audio_signal.shape, 1)
    self.assertEqual(audio_signal.shape[0], 50381)
    self.assertEqual(audio_fs, 16000.0)
    self.assertEqual(audio_signal.dtype, np.int16)
    audio_seconds = audio_signal.shape[0]/float(audio_fs)

    minimum_interval = 0.5
    number = int(audio_seconds)
    include_zero = True
    pulse_length = 0.1
    event_times = add_trigger.random_times(audio_seconds, number,
                                           minimum_interval, include_zero)
    print('Event_times are:', event_times)
    triggered_audio = add_trigger.add_events_to_audio(audio_signal, event_times,
                                                      fs=audio_fs,
                                                      pulse_length=pulse_length)
    self.assertLen(triggered_audio.shape, 2)
    self.assertEqual(triggered_audio.shape[0], 50381)
    self.assertEqual(triggered_audio.shape[1], 2)
    self.assertLen(set(triggered_audio[:, 1]), 2)

    self.assertTrue(np.all(triggered_audio[:, 0] == audio_signal))
    event_start = int(event_times[0]*audio_fs) + 1
    event_end = event_start + int(pulse_length*audio_fs) - 1
    self.assertTrue(np.all(triggered_audio[event_start:event_end, 1] == 32767))

  def test_add_triggers_freq(self):
    audio_file = os.path.join(self._test_data, 'tapestry.wav')
    [audio_fs, audio_signal] = add_trigger.read_audio_wave_file(audio_file)
    audio_seconds = audio_signal.shape[0]/float(audio_fs)

    minimum_interval = 0.5
    number = int(audio_seconds)
    include_zero = True
    pulse_length = 0.1
    tone_freq = 440
    event_times = add_trigger.random_times(audio_seconds, number,
                                           minimum_interval, include_zero)
    triggered_audio = add_trigger.add_events_to_audio(audio_signal, event_times,
                                                      fs=audio_fs,
                                                      pulse_length=pulse_length,
                                                      pulse_freq=tone_freq)
    def find_peaks(a):
      peaks = np.logical_and(a[1:-1] > a[0:-2],
                             a[1:-1] > a[2:])
      return np.nonzero(peaks)[0] + 1
    peak_times = find_peaks(triggered_audio[:, 1])/float(audio_fs)
    estimated_freq = 1.0/(peak_times[3] - peak_times[2])
    self.assertLess(abs(tone_freq - estimated_freq), 5.0)

  def test_read_write(self):
    audio_filename = os.path.join(self._test_data, 'tapestry.wav')
    [fs, audio_data] = add_trigger.read_audio_wave_file(audio_filename)
    tmp_filename = os.path.join('/tmp', 'tapestry2.wav')
    add_trigger.write_audio_wave_file(tmp_filename, audio_data, fs)
    [fs2, audio_data2] = add_trigger.read_audio_wave_file(tmp_filename)
    self.assertEqual(fs, fs2)
    self.assertEqual(audio_data.shape, audio_data2.shape)
    self.assertEqual(np.max(audio_data - audio_data2), 0)

  def test_stereo_processing(self):
    """Tests for multi-channel input, with averaging."""
    num_secs = 1
    num_channels = 3
    fs = 16000  # Sampling frequency
    f0 = 440  # Tone frequency
    signal_max = 30000
    t = np.arange(fs*num_secs)/float(fs)
    signal = np.zeros((len(t), num_channels))
    signal[:, 0] = np.sin(2*np.pi*t*f0)*signal_max
    self.assertEqual(np.max(signal), signal_max)
    triggered_audio = add_trigger.add_events_to_audio(signal, [0, 0.25, 0.5])
    self.assertEqual(triggered_audio.shape[1], 2)
    self.assertAlmostEqual(np.max(triggered_audio[:, 0]),
                           signal_max/num_channels)  # After average
    self.assertGreater(np.max(triggered_audio[:, 1]), 30000)

if __name__ == '__main__':
  absltest.main()
