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

"""Test for telluride_decoding.infer_decoder."""

from absl.testing import absltest

import numpy as np

from telluride_decoding import result_store


class NumpyStoreTest(absltest.TestCase):

  def test_storage(self):
    """Makes sure our storage class doesn't lose data."""
    num_points = 10000
    num_dims = 5
    data = np.random.randn(num_points, num_dims)
    storage = result_store.NumpyStore(init_frame_count=100)
    frame_chunk_size = 33
    count = 0
    while count < num_points:
      e = min(num_points, count + frame_chunk_size)
      storage.add_data(data[count:e, :])
      count = e
    np.testing.assert_array_equal(storage.all_data, data)
    self.assertEqual(storage.count, num_points)

  def test_storage_errors(self):
    with self.assertRaisesRegex(ValueError,
                                'Initial frame count must be greater than 0'):
      storage = result_store.NumpyStore(init_frame_count=0)

    storage = result_store.NumpyStore()
    self.assertFalse(storage.all_data)

    with self.assertRaisesRegex(TypeError, 'data must be a 2D numpy array'):
      storage = result_store.WindowedDataStore()
      storage.add_data(np.zeros((2, 3, 4)))

    with self.assertRaisesRegex(
        ValueError, 'Data\'s shape has changed, and this is not allowed.'):
      storage = result_store.WindowedDataStore()
      storage.add_data(np.zeros((42, 3)))
      storage.add_data(np.zeros((42, 2)))

  def test_online_storage(self):
    """Trying out online data fetching.

    Data is stored in the buffer one sample at a time and fetched in fixed size
    chunks. This is similar to the functionality used in the real time tool.
    """
    num_points = 100
    num_dims = 5
    data = np.random.randn(num_points, num_dims)
    storage = result_store.NumpyStore(init_frame_count=1000)
    frame_chunk_size = 1
    count = 0
    num_points_less_chunk = num_points - frame_chunk_size
    while count <= num_points_less_chunk:
      storage.add_data(data[count:count+frame_chunk_size, :])
      count += frame_chunk_size
    data_chunk = next(storage.next_window(5))
    np.testing.assert_array_equal(storage.all_data, data[5:])
    np.testing.assert_array_equal(data_chunk, data[:5])
    data_chunk = next(storage.next_window(5))
    np.testing.assert_array_equal(storage.all_data, data[10:])
    np.testing.assert_array_equal(data_chunk, data[5:10])
    data_chunk = next(storage.next_window(95))
    self.assertIsNone(data_chunk)


class WindowedDataStoreTest(absltest.TestCase):

  def test_online_storage(self):
    """Trying out online data fetching.

    Data is stored in the buffer one sample at a time and fetched in fixed size
    chunks. This is similar to the functionality used in the real time tool.
    """
    num_points = 100
    num_dims = 5
    data = np.random.randn(num_points, num_dims)
    storage = result_store.WindowedDataStore(
        initial_frame_count=1000, window_step=5, window_width=5, pre_context=0)
    frame_chunk_size = 1
    count = 0
    num_points_less_chunk = num_points - frame_chunk_size
    while count <= num_points_less_chunk:
      storage.add_data(data[count:count+frame_chunk_size, :])
      count += frame_chunk_size
    data_chunk = next(storage.next_window())
    np.testing.assert_array_equal(storage.all_data, data[5:])
    np.testing.assert_array_equal(data_chunk, data[:5])
    data_chunk = next(storage.next_window())
    np.testing.assert_array_equal(storage.all_data, data[10:])
    np.testing.assert_array_equal(data_chunk, data[5:10])

  def test_windowed_data_store_simple(self):
    fs = 2000
    t = np.reshape(np.arange(fs), (-1, 1))
    # Create two columns of data, one the opposite of the other to make sure
    # two-dimensional features work.
    t = np.concatenate((t, -t), axis=1)

    window_step = 10
    window_width = int(window_step * 3) + 1
    half_window_width = window_width//2
    storage = result_store.WindowedDataStore(
        window_step=window_step, window_width=window_width,
        pre_context=window_width//2)

    input_pos = 0
    output_count = 0
    data_width = 34  # Arbitrary
    while input_pos < len(t):
      e = min(input_pos + data_width, len(t))
      storage.add_data(t[input_pos:e, :])
      input_pos = e

      for data in storage.next_window():
        self.assertEqual(data.shape[0], window_width)
        b = -half_window_width + output_count*window_step
        e = half_window_width + output_count*window_step + 1
        expected = np.arange(b, e, dtype=np.int32)
        if output_count == 0:
          expected[0:int(half_window_width)] = 0
        elif output_count == 1:
          expected[0:int(half_window_width - window_step)] = 0
        np.testing.assert_array_equal(data[:, 0], expected)
        np.testing.assert_array_equal(data[:, 1], -expected)
        output_count += 1
    self.assertEqual(output_count, int((len(t) - half_window_width)/
                                       float(window_step) + 1))

  def test_audio_data_store_count(self):
    """Make sure that the all_data property returns the right amount of data."""
    fs = 2000
    t = np.reshape(np.arange(fs), (-1, 1))
    # Create two columns of data, one the opposite of the other to make sure
    # two-dimensional features work.
    t = np.concatenate((t, -t), axis=1)

    window_step = 10
    window_width = int(window_step * 3)+1
    storage = result_store.WindowedDataStore(
        window_step=window_step, window_width=window_width,
        pre_context=(window_width)//2, initial_frame_count=1000)

    input_pos = 0
    data_width = 34  # Arbitrary
    while input_pos < t.shape[0]:
      e = min(input_pos + data_width, t.shape[0])
      storage.add_data(t[input_pos:e, :])
      input_pos = e
    # Should get the amount we put in, plus the first half-window of zeros that
    # are added to the beginning of the data.
    self.assertEqual(storage.all_data.shape[0], t.shape[0] + (window_width)//2)

  def test_audio_data_errors(self):
    window_step = 10.5
    with self.assertRaisesRegex(ValueError,
                                'Must be an integer window_step for now'):
      result_store.WindowedDataStore(window_step=window_step)

    window_step = 10
    window_width = 2
    with self.assertRaisesRegex(
        ValueError, 'window_step .* must be less than or equal to '):
      result_store.WindowedDataStore(window_width=window_width,
                                     window_step=window_step)

  def test_audio_data_store_too_much(self):
    """Make sure we don't fail if we add lots of data."""
    window_step = 10

    storage = result_store.WindowedDataStore(window_step=window_step)
    t = np.ones(100)
    for _ in range(1000):
      storage.add_data(t)


class TwoResultStoreTest(absltest.TestCase):

  def test_two_results(self):
    num_points = 1000
    data = np.reshape(np.arange(num_points, dtype=np.float32), (-1, 1))

    window_width = 201
    window_step = 100
    two_results = result_store.TwoResultStore(
        window_width=window_width, window_step=window_step)
    two_results.add_data(data, -data)

    r1, r2 = two_results.all_data
    np.testing.assert_array_equal(r1, data)
    np.testing.assert_array_equal(r2, -data)

    num_batches = 0
    for r1, r2 in two_results.next_window():
      self.assertEqual(r1.shape[0], window_width)
      np.testing.assert_array_equal(r1, -r2)
      num_batches += 1
    self.assertEqual(num_batches,
                     (num_points-(window_width))//window_step + 1)

    with self.assertRaisesRegex(
        ValueError, 'Both data must have the same # frames.'):
      storage = result_store.TwoResultStore()
      storage.add_data(np.zeros((42, 3)), np.zeros((4, 2)))


if __name__ == '__main__':
  absltest.main()
