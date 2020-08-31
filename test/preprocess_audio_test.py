# Lint as: python2 python3
"""Tests for telluride_decoding.preprocess_audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import absltest
import numpy as np

from telluride_decoding import preprocess_audio


class AudioIntensityStoreTest(absltest.TestCase):

  def test_audio_intensity(self):
    fs = 1000
    t = np.arange(fs)

    window_step = 10
    half_window_width = int(window_step * 1.5)
    window_width = 2*half_window_width + 1
    storage = preprocess_audio.AudioIntensityStore(
        window_step=window_step, window_width=window_width,
        pre_context=half_window_width)

    input_pos = 0
    output_count = 0
    data_width = 34  # Arbitrary
    while input_pos < len(t):
      e = min(input_pos + data_width, len(t))
      storage.add_data(t[input_pos:e])
      input_pos = e
      for data in storage.next_window():
        self.assertIsInstance(data, float)
        b = -1.5*window_step + output_count*window_step
        e = 1.5*window_step + output_count*window_step + 1
        expected = np.arange(b, e, dtype=np.int32)
        if output_count == 0:
          expected[0:int(half_window_width)] = 0
        elif output_count == 1:
          expected[0:int(half_window_width - window_step)] = 0
        self.assertEqual(data, np.mean(np.square(expected)))
        output_count += 1

    self.assertEqual(output_count, int((len(t) - half_window_width)/
                                       float(window_step) + 1))


if __name__ == '__main__':
  absltest.main()
