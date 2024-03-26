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

"""Tests for the decoding part of the library.

"""
import os
import subprocess

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

import numpy as np
from telluride_decoding.brain_data import BrainData
from telluride_decoding.brain_data import count_tfrecords
from telluride_decoding.brain_data import create_brain_dataset
from telluride_decoding.brain_data import discover_feature_shapes
from telluride_decoding.brain_data import mismatch_batch_randomization
from telluride_decoding.brain_data import TestBrainData
from telluride_decoding.brain_data import TFExampleData

import tensorflow as tf


# These flags are defined in decoding.py, but we add them here so we can test
# the brain_data classes in a standalone fashion.
flags.DEFINE_enum('context_method', 'new', ['old', 'new'],
                  'Switch to control temporal window approach.')
flags.DEFINE_bool('random_mixup_batch',
                  False,
                  'Mixup the data, so labels are random, for testing.')

flags.DEFINE_string('train_file_pattern', '',
                    'A regular expression for picking training files.')
flags.DEFINE_string('test_file_pattern', '',
                    'A regular expression for picking testing files.')
flags.DEFINE_string('validate_file_pattern', '',
                    'A regular expression for picking validation files.')
flags.DEFINE_string('tfexample_dir',
                    None,
                    'location of generic TFRecord data')
flags.DEFINE_string('input_field', 'mel_spectrogram',
                    'Input field to use for predictions.')
flags.DEFINE_string('output_field', 'envelope',
                    'Output field to predict.')


class MismatchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('even', 100000, 100, 150),
      ('odd',  100001, 100, 150),  # pylint: disable=bad-whitespace
      )
  def test_mismatch(self, n, reps, expected):
    x = tf.constant(np.reshape(np.arange(n), (-1, 1)))
    y = 10 + tf.constant(np.reshape(np.arange(n), (-1, 1)))
    not_different = 0
    for _ in range(reps):
      new_x, new_x2, new_y, new_a = mismatch_batch_randomization(x, y, x, x)
      np.testing.assert_array_equal(x, new_x)
      np.testing.assert_array_equal(new_x.shape[0], new_x2.shape[0])
      np.testing.assert_array_equal(new_x.shape[0], new_y.shape[0])
      np.testing.assert_array_equal(new_x.shape[0], new_a.shape[0])
      mid = (n+1)//2  # The first mid points are match, the rest are mismatch.
      np.testing.assert_array_equal(new_y[:mid, :], 0)
      np.testing.assert_array_equal(new_x2[:mid, :], y[0::2, :])  # Even #'ed

      np.testing.assert_array_equal(new_y[mid:, :], 1)
      # Hopefully not many matches in random part.
      not_different += np.sum(new_x2[mid:, :] == y[1::2, :])
      self.assertLess(not_different, expected)


class BrainDataTest(absltest.TestCase):

  def setUp(self):
    super(BrainDataTest, self).setUp()
    self._test_data_dir = os.path.join(
        flags.FLAGS.test_srcdir, '_main', 
        'test_data/',
        'meg')
    if not os.path.exists(self._test_data_dir):
      # Debugging: If not here, where.
      subprocess.run(['ls', flags.FLAGS.test_srcdir])
      subprocess.run(['ls', os.path.join(flags.FLAGS.test_srcdir, '_main')])
      self.assertTrue(os.path.exists(self._test_data_dir),
                      f'Test data dir does not exist: {self._test_data_dir}')

  ################## Linear data for testing ################################
  # Just a list of consecutive integers, to make it easier to debug batching
  # issues and the like.

  def create_linear_dataset(self,
                            dataset: BrainData,
                            mode: str = 'program_test',
                            num_samples: int = 1000, offset: int = 1000,
                            mixup_batch: bool = False):
    """Create a dataset of consecutive numbers for testing batching.

    Create a two column matrix for x, a one column matrix for x2, and a
    one column matrix for y.  The columns start with [0, 1000], [2000], and
    [3000] (assuming offset is 1000).

    Args:
      dataset: The BrainDecoding object for which we want to fill in the data.
      mode: The type of TF dataset to create, one of train, test, program_test
        to indicate the type of shuffling, etc. to do.
      num_samples: How many samples to create.
      offset: The offset between columns (features).
      mixup_batch: Whether to mixup the input and outputs.

    Returns:
      Two items in a tuple: the dataset iterator, and the actual tf.data.dataset
      object.
    """
    self.assertIsInstance(dataset, TestBrainData)
    print('Creating a linear dataset with %d and %d.' % (num_samples, offset))
    base_data = np.arange(num_samples).reshape((num_samples, 1))

    input_data = np.concatenate((base_data, base_data + offset), axis=1)
    input2_data = base_data + 2*offset
    output_data = base_data + 3*offset
    attention_data = np.ones((num_samples, 1))

    dataset.preserve_test_data(input_data, output_data,
                               input2_data, attention_data)
    return dataset.create_dataset(mode=mode, mixup_batch=mixup_batch)

  def test_create_dataset_notimplemented(self):
    brain_data = BrainData('in', 'out', 100)
    with self.assertRaises(NotImplementedError):
      brain_data.create_dataset()

  @flagsaver.flagsaver
  def test_linear_batching(self):
    """Test the batching (pre and post context).

    This is key to the decoding datasets. Make sure the resulting dataset
    object is returning the right data.  Test with linearly increasing data, so
    we know to expect, especially as we add context.
    """
    print('********** test_linear_batching **************')
    batch_size = 900   # Big batch to minimize chance of random alignment.
    frame_rate = 100.0
    # No shift
    def create_test_dataset(pre_context, post_context, mode, input_offset=0,
                            mixup_batch=False):
      """Create a test dataset, ensuring basic data shapes are correct.

      Args:
        pre_context: # of frames before the current frame to add for context.
        post_context: # of frames after the current frame to add for context.
        mode: Data generation mode, program test means don't shuffle for tests.
        input_offset: How much temporal offset to introduce between data.
        mixup_batch: Shuffle the batch if true.

      Returns:
        Tuple of input data (np), input2_data (np), output (np) and the
        BrainData object.
      """
      brain_data = TestBrainData('input_1', 'output', frame_rate,
                                 in2_fields='input_2',
                                 final_batch_size=batch_size,
                                 pre_context=pre_context,
                                 post_context=post_context,
                                 input_offset=input_offset,
                                 repeat_count=10)
      dataset = self.create_linear_dataset(brain_data, mode,
                                           mixup_batch=mixup_batch)
      features = -1  # Unused value to test for next step
      for next_element in dataset:
        (features, output) = next_element
        break
      self.assertNotEqual(features, -1)  # Test for not empty dataset

      input_data = features['input_1']
      input2_data = features['input_2']
      print('Input_data.shape', input_data.shape)
      print('Input2_data.shape', input2_data.shape)
      self.assertLen(input_data.shape, 2)
      self.assertGreater(input_data.shape[0], 0)
      self.assertGreater(input_data.shape[1], 0)
      print('test_linear_batch input:', input_data[0:3, :])
      print('test_linear_batch input2:', input2_data[0:3, :])
      print('test_linear_batch output:', output[0:3, :])
      return input_data, input2_data, output, brain_data

    def all_in_order(data):
      return np.all(data[1:, :] > data[0:-1, :])

    # First test without context, using test data so there is no shuffling.
    pre_context = 0
    post_context = 0
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, mode='program_test')
    num_input_channels = brain_data.num_input_channels

    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 2)
    # pylint: disable=bad-whitespace
    self.assertTrue(np.all(np.equal(input_data[0:3, :], [[0, 1000],
                                                         [1, 1001],
                                                         [2, 1002]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :], [[2000],
                                                          [2001],
                                                          [2002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :], [[3000],
                                                     [3001],
                                                     [3002]])))
    self.assertEqual(brain_data.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertEqual(brain_data.output_field_width(), 1)
    self.assertTrue(all_in_order(input_data))
    self.assertTrue(all_in_order(output))

    # Now test with an input_offset.
    print('********** test_linear_batching - Input Offset **************')
    # Input offset of +1
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, 'program_test', input_offset=1)
    print(f'input_data: {input_data[0:3, :]}')
    print(f'input2_data: {input2_data[0:3, :]}')
    print(f'output: {output[0:3, :]}')
    self.assertTrue(np.all(np.equal(input_data[0:3, :], [[1, 1001],
                                                         [2, 1002],
                                                         [3, 1003]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :], [[2000],
                                                          [2001],
                                                          [2002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :], [[3000],
                                                     [3001],
                                                     [3002]])))

    # Input offset of -1
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, 'program_test', input_offset=-1)
    print(f'input_data: {input_data[0:3, :]}')
    print(f'input2_data: {input2_data[0:3, :]}')
    print(f'output: {output[0:3, :]}')
    self.assertTrue(np.all(np.equal(input_data[0:3, :], [[0, 1000],
                                                         [1, 1001],
                                                         [2, 1002]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :], [[2001],
                                                          [2002],
                                                          [2003]])))
    self.assertTrue(np.all(np.equal(output[0:3, :], [[3001],
                                                     [3002],
                                                     [3003]])))

    # Input offset of 2
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, 'program_test', input_offset=2)
    print(f'input_data: {input_data[0:3, :]}')
    print(f'input2_data: {input2_data[0:3, :]}')
    print(f'output: {output[0:3, :]}')
    self.assertTrue(np.all(np.equal(input_data[0:3, :], [[2, 1002],
                                                         [3, 1003],
                                                         [4, 1004]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :], [[2000],
                                                          [2001],
                                                          [2002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :], [[3000],
                                                     [3001],
                                                     [3002]])))

    # Now test with training data, to make sure batches are randomized.
    print('********** test_linear_batching - Randomized Batches **************')
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, 'train')
    self.assertFalse(all_in_order(input_data))
    self.assertFalse(all_in_order(output))
    self.assertTrue(np.all((input_data[:, 0] + 3000) == output[:, 0]))

    # Now test with just two samples of pre context
    pre_context = 2
    post_context = 0
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, mode='program_test')
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 6)
    # Last row of input and output should match, with extra context before.
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0,    0,   0,    0,   0, 1000],
                                     [0,    0,   0, 1000,   1, 1001],
                                     [0, 1000,   1, 1001,   2, 1002]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :],
                                    [[2000],
                                     [2001],
                                     [2002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[3000],
                                     [3001],
                                     [3002]])))
    self.assertEqual(brain_data.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data[2:, :]))
    self.assertTrue(all_in_order(output))

    # Now test with just two samples of post context
    pre_context = 0
    post_context = 2
    input_data, input2_data, output, brain_data = create_test_dataset(
        pre_context, post_context, mode='program_test')
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 6)
    # First row of input and output should match, with extra context before.
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0, 1000, 1, 1001, 2, 1002],
                                     [1, 1001, 2, 1002, 3, 1003],
                                     [2, 1002, 3, 1003, 4, 1004]])))
    self.assertTrue(np.all(np.equal(input2_data[0:3, :],
                                    [[2000],
                                     [2001],
                                     [2002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[3000],
                                     [3001],
                                     [3002]])))
    self.assertEqual(brain_data.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data[:-2, :]))
    self.assertTrue(all_in_order(output))

    # Now test with the data NOT mixed up.
    pre_context = 0
    post_context = 0
    input_data, input2_data, output, _ = create_test_dataset(
        pre_context, post_context, 'train', mixup_batch=False)
    self.assertFalse(all_in_order(input_data))
    self.assertFalse(all_in_order(input2_data))
    self.assertFalse(all_in_order(output))
    matches = np.sum((input_data[:, 0] + 3000) == output[:, 0])
    print('Found %d matches in %d frames.' % (matches, input_data.shape[0]))
    self.assertEqual(matches, input_data.shape[0])

    # Now test with the data mixed up. Exact same test as before except for
    # matches.
    pre_context = 0
    post_context = 0
    input_data, input2_data, output, _ = create_test_dataset(
        pre_context, post_context, 'train', mixup_batch=True)
    self.assertFalse(all_in_order(input_data))
    self.assertFalse(all_in_order(input2_data))
    self.assertFalse(all_in_order(output))
    matches = np.sum((input_data[:, 0] + 3000) == output[:, 0])
    print('Found %d matches in %d frames.' % (matches, input_data.shape[0]))
    self.assertLess(matches, input_data.shape[0]/64)
    # pylint: enable=bad-whitespace

  ################## Simply scaled data ################################
  # Simple data for testing the machine learning.  Output is a simple transform
  # of the input data (sin in this case.) Use this data to test simple
  # regression, and data offsets.

  def simply_scaled_transform(self, input_data):
    """Transform to apply to the data.

    Kept as a separate function for easier testing. In this case, just the sin()
    of the first column.

    Args:
      input_data: The input data.

    Returns:
      the input_data transformed by this function.
    """
    self.assertNotEmpty(input_data.shape)
    return np.sin(input_data[:, 0:1] * 2 * np.pi)

  def create_simply_scaled_dataset(self, dataset, data_offset=0,
                                   num_input_channels=2, mode='program_test'):
    """A dataset where the output is a simple scalar function of the input."""
    self.assertIsInstance(dataset, TestBrainData)

    num_samples = 100000
    input_data = np.random.randn(num_samples + 2*abs(data_offset),
                                 num_input_channels).astype(np.float32)
    output_data = self.simply_scaled_transform(input_data)
    if data_offset >= 0:
      input_data = input_data[0:num_samples, :]
      output_data = output_data[data_offset:data_offset + num_samples, :]
    else:
      input_data = input_data[-data_offset:
                              -data_offset + num_samples, :]
      output_data = output_data[0:num_samples, :]
    dataset.preserve_test_data(input_data, output_data, None)
    return dataset.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_simple_shift(self):
    """Test the batching as we shift the data.

    Make sure that the testdata is properly formatted and that the output
    is computed from the correct column, as we shift the context around.
    """
    batch_size = 128
    print('********** test_testdata **************')
    pre_context = 2
    post_context = 2
    frame_rate = 100.0

    def create_shifted_test(data_offset):
      brain_data = TestBrainData('input', 'output', frame_rate,
                                 repeat_count=10,
                                 final_batch_size=batch_size,
                                 pre_context=pre_context,
                                 post_context=post_context)
      test_dataset = self.create_simply_scaled_dataset(brain_data,
                                                       data_offset)
      for next_element in test_dataset:
        (input_data, output) = next_element
      input_data = input_data['input_1']

      # Select the right column of the shifted input data to compute the
      # expect output.
      num_channels = brain_data.num_input_channels
      index = num_channels * (pre_context + data_offset)
      expected_output = self.simply_scaled_transform(
          input_data[:, index:index+num_channels])
      if data_offset < 0:
        # Can't predict what we can't see, so force the first few samples to be
        # equal.
        expected_output[0:-data_offset, :] = output[0:-data_offset, :]
      return input_data, output, expected_output, brain_data

    input_data, output, expected_output, brain_data = create_shifted_test(0)
    num_channels = brain_data.num_input_channels

    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     (pre_context + 1 + post_context)*num_channels)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(output.shape[1], 1)
    self.assertEqual(expected_output.shape[0], batch_size)
    self.assertEqual(expected_output.shape[1], 1)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    input_data, output, expected_output, brain_data = create_shifted_test(2)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     (pre_context + 1 + post_context)*num_channels)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(output.shape[1], 1)
    self.assertEqual(expected_output.shape[0], batch_size)
    self.assertEqual(expected_output.shape[1], 1)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    input_data, output, expected_output, brain_data = create_shifted_test(-2)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     (pre_context + 1 + post_context)*num_channels)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(output.shape[1], 1)
    self.assertEqual(expected_output.shape[0], batch_size)
    self.assertEqual(expected_output.shape[1], 1)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

  @flagsaver.flagsaver
  def test_data_count(self):
    print('********** test_data_count **************')
    repeat_count_request = 3
    batch_size_request = 128
    frame_rate = 100.0
    brain_data = TestBrainData('input', 'output', frame_rate,
                               repeat_count=repeat_count_request,
                               final_batch_size=batch_size_request)
    test_dataset = self.create_linear_dataset(brain_data, mode='train')
    batch_count = 0
    try:
      for vals in test_dataset:
        (inputs, output) = vals
        batch_count += 1
    except tf.errors.OutOfRangeError:
      pass

    print('batch_count is %d.' % batch_count)
    print('Impulse dataset returned %d items of size:' % batch_count,
          inputs['input_1'].shape,
          output.shape)

    self.assertEqual(inputs['input_1'].shape,
                     (batch_size_request, brain_data.num_input_channels))
    self.assertEqual(output.shape, (batch_size_request, 1))
    self.assertEqual(batch_count,
                     (repeat_count_request * 1000)//batch_size_request)

    # Check the attended speaker feature to make sure it exists in the file.
    self.assertEqual(inputs['attended_speaker'].shape, (batch_size_request, 1))
    np.testing.assert_equal(inputs['attended_speaker'].numpy()[0], 1)

  def test_count_tfrecords(self):
    """Make sure we can count the records in our test file."""
    all_files = []
    for (path, _, files) in tf.io.gfile.walk(self._test_data_dir):
      all_files += [path + '/' + f for f in files if f.endswith('.tfrecords')]
    self.assertLen(all_files, 3)
    num_records, found_errors = count_tfrecords(all_files[0])
    print('test_count_tfrecords got back:', num_records, found_errors)
    self.assertEqual(num_records, 1001)
    self.assertFalse(found_errors)

  @flagsaver.flagsaver
  def test_discover_feature_shapes(self):
    """Make sure we get the right kinds of records in our test file."""
    all_files = []
    for (path, _, files) in tf.io.gfile.walk(self._test_data_dir):
      all_files += [path + '/' + f for f in files if f.endswith('.tfrecords')]
    self.assertLen(all_files, 3)
    feature_dict = discover_feature_shapes(all_files[0])
    print('test_read_tfrecords: Record dictionary is', feature_dict)
    expected_feature_widths = {'phonetic_features': 19,
                               'mel_spectrogram': 64,
                               'meg': 148,
                               'phonemes': 38,
                               'envelope': 1,
                              }
    for k in expected_feature_widths:
      self.assertIn(k, feature_dict)
      self.assertIsInstance(feature_dict[k], tf.io.FixedLenFeature)
      self.assertEqual(feature_dict[k].shape[0], expected_feature_widths[k])

  def get_feature_shapes_from_file(self):
    all_files = []
    for (path, _, files) in tf.io.gfile.walk(self._test_data_dir):
      all_files += [path + '/' + f for f in files if f.endswith('.tfrecords')]
    self.assertNotEmpty(all_files)

    return discover_feature_shapes(all_files[0])

  @flagsaver.flagsaver
  def test_generic_read(self):
    """Test to see if we can parse a specific TFRecord data file."""
    self.longMessage = True     # For the asserts  pylint: disable=invalid-name

    features = self.get_feature_shapes_from_file()

    mandatory_feature_sizes = {'envelope': (1, tf.float32),
                               'phonetic_features': (19, tf.float32),
                               'phonemes': (38, tf.float32),
                               'meg': (148, tf.float32),
                               'mel_spectrogram': (64, tf.float32),
                              }

    for k in mandatory_feature_sizes:
      expected_size, expected_type = mandatory_feature_sizes[k]
      self.assertIn(k, features, 'looking for feature ' + k)
      self.assertEqual(expected_size, features[k].shape[0],
                       'testing size of feature ' + k)
      self.assertEqual(expected_type, features[k].dtype,
                       'testing type of feature ' + k)

  def test_tfexample_read(self):
    """Test reading the Generic TFRecord data with the dataset path.
    """
    print('********** test_generic_processing **************')
    batch_size = 128
    pre_context = 0
    post_context = 0
    frame_rate = 100.0

    flags.FLAGS.tfexample_dir = self._test_data_dir

    def get_one_element(input_feature, output_feature):
      test_brain_data = TFExampleData(input_feature, output_feature, frame_rate,
                                      final_batch_size=batch_size,
                                      pre_context=pre_context,
                                      post_context=post_context,
                                      data_dir=self._test_data_dir)
      test_dataset = test_brain_data.create_dataset('program_test')
      for vals in test_dataset.take(1):
        (input_data, output_data) = vals
      input_data = input_data['input_1']
      return input_data, output_data

    features = self.get_feature_shapes_from_file()
    input_feature = 'mel_spectrogram'
    output_feature = 'envelope'
    input_data, output_data = get_one_element(input_feature, output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], features[input_feature].shape[0])
    self.assertTrue(np.all(input_data >= 0.0))

    self.assertEqual(output_data.shape[0], batch_size)
    self.assertEqual(output_data.shape[1], features[output_feature].shape[0])
    self.assertTrue(np.all(output_data >= 0.0))

    input_feature = 'phonemes'
    output_feature = 'phonetic_features'
    input_data, output_data = get_one_element(input_feature, output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], features[input_feature].shape[0])
    # Make sure input data (phonemes) is binary
    self.assertTrue(np.all(np.logical_or(input_data == 0, input_data == 1)))

    self.assertEqual(output_data.shape[0], batch_size)
    self.assertEqual(output_data.shape[1], features[output_feature].shape[0])
    # Make sure output data (phonetic features) is binary
    self.assertTrue(np.all(np.logical_or(output_data == 0, output_data == 1)))

    input_feature = 'meg'
    output_feature = 'envelope'
    input_data, output_data = get_one_element(input_feature, output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], features[input_feature].shape[0])
    # Already checked envelope, so we don't need to do it again.

  @flagsaver.flagsaver
  def test_offset_shifts(self):
    """Simple tests to make sure we can shift the output in time."""
    frame_rate = 100.0
    def create_shifted_test(data_offset):
      test_dataset = TestBrainData('input_1', 'output', frame_rate,
                                   repeat_count=1,
                                   pre_context=max(-data_offset, 0),
                                   post_context=max(data_offset, 0))
      test_dataset = self.create_simply_scaled_dataset(test_dataset,
                                                       data_offset)
      for next_element in test_dataset.take(1):
        (input_data, output) = next_element
      input_data = input_data['input_1']

      if data_offset == 0:
        expected_output = self.simply_scaled_transform(input_data)
        output = output[:, 0]
      elif data_offset > 0:
        expected_output = self.simply_scaled_transform(
            input_data[data_offset:, :])
        output = output[0:-data_offset, 0]
      else:
        expected_output = self.simply_scaled_transform(
            input_data[0:data_offset, :])
        expected_output = expected_output[-data_offset:, :]
        output = output[-data_offset:data_offset, 0]
      expected_output = expected_output[:, 0]
      print('output:', output.shape, output[0:12])
      print('expected_output:', expected_output.shape, expected_output[0:12])
      return output, expected_output

    print('\n\n**********test_offset_shifts starting... *******')
    output, expected_output = create_shifted_test(0)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    output, expected_output = create_shifted_test(2)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    output, expected_output = create_shifted_test(-2)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

  def test_tfrecord_open(self):
    print('********** test_tfrecord_open **************')
    batch_size = 512
    pre_context = 0
    post_context = 0
    frame_rate = 100.0
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    final_batch_size=batch_size,
                                    pre_context=pre_context,
                                    post_context=post_context,
                                    repeat_count=1,
                                    data_dir=self._test_data_dir)
    print('Finished creatiung the TFExampleData')
    test_dataset = test_brain_data.create_dataset(mode='program_test')
    print('Finished creating the dataset.')
    for an_element in test_dataset.take(1):
      input_dict, output = an_element
      print('Read an element from the dataset.')
      break
    for k in input_dict:
      print('test_tfrecord_open found %s:' % k, input_dict[k].shape)
    print('test_tfrecord_open found %s:' % 'output', output.shape)

    self.assertEqual(input_dict['input_1'].shape[0], batch_size)
    # TFExample data has a 148-dimensional MEG signal
    self.assertEqual(input_dict['input_1'].shape[1],
                     148*(pre_context + post_context + 1))
    # Ignore input_dict['x2'] because it is dummy (unused) data
    self.assertEqual(output.shape[0], batch_size)

    self.assertEqual(output.shape[1], 1)  # Since it is envelope data

    filename = os.path.join(self._test_data_dir, 'subj01_1ksamples.tfrecords')
    for example in tf.compat.v1.python_io.tf_record_iterator(filename):
      # print(type(tf.train.Example.FromString(example)))
      # print(tf.train.Example.FromString(example))
      ex_proto = tf.train.Example.FromString(example)
      meg_data = ex_proto.features.feature['meg'].float_list.value
      break

    np.testing.assert_allclose(input_dict['input_1'][0, :], meg_data, rtol=0.01)

  def test_brain_data_creator(self):
    input_type = 'meg'
    output_type = 'envelope'
    frame_rate = 100.0
    self.assertIsInstance(create_brain_dataset('tfrecord', input_type,
                                               output_type, frame_rate,
                                               data_dir=self._test_data_dir),
                          TFExampleData)

    self.assertIsInstance(create_brain_dataset('test', input_type, output_type,
                                               frame_rate),
                          TestBrainData)

    with self.assertRaisesRegex(TypeError, 'must be a string'):
      create_brain_dataset(42, input_type, output_type, frame_rate)
    with self.assertRaisesRegex(TypeError, 'unknown data type'):
      create_brain_dataset('foobar', input_type, output_type, frame_rate)
    with self.assertRaisesRegex(ValueError, 'Missing data_dir'):
      create_brain_dataset('tfrecord', input_type, output_type, frame_rate)

  def test_file_randomness(self):
    """Tests to make sure we get randomness when listing files for allbut.
    """
    frame_rate = 100.0

    # Test randomization... about 2/3 the times the first file of the returned
    # file list should be a different file name.
    first_file = None
    mismatch_count = 0
    test_count = 100
    for _ in range(test_count):
      test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                      data_dir=self._test_data_dir)
      all_file_list = test_brain_data.all_files()
      self.assertLen(all_file_list, 3)
      if not first_file:
        first_file = all_file_list[0]
      if first_file != all_file_list[0]:
        mismatch_count += 1
    self.assertAlmostEqual(2*test_count/3, mismatch_count, delta=test_count//4)

  def test_file_allbut(self):
    """Various tests when using allbut to generate training data."""
    frame_rate = 100.0

    # Test randomization... about half the times the first file of the returned
    # file list should be a different file name.
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    data_dir=self._test_data_dir)
    all_file_list = test_brain_data.all_files()
    self.assertLen(all_file_list, 3)

    for _ in range(100):
      test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                      train_file_pattern='allbut_1',
                                      test_file_pattern='subj01',
                                      validate_file_pattern='subj01',
                                      data_dir=self._test_data_dir)
      filtered = test_brain_data.filter_file_names('train')
      self.assertLen(filtered, 1)   # Only get one result with allbut_1
      self.assertNotIn('subj01', filtered[0])   # Make sure never get subj01

  def test_missing_validate_field(self):
    """Tests for missing validate specification (only test)."""
    frame_rate = 100.0
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    train_file_pattern='allbut_1',
                                    test_file_pattern='subj01',
                                    data_dir=self._test_data_dir)
    with self.assertRaisesRegex(ValueError,
                                'Both test and validate must be specified'):
      test_brain_data.filter_file_names('train')

  def test_create_before_preserve(self):
    """Tests that the test data is preserved before calling create_dataset."""
    data = TestBrainData('in', 'out', 100)
    with self.assertRaisesRegex(ValueError,
                                'Must call preserve_test_data before create_'):
      data.create_dataset('train')

  def test_filter_file_names_missing(self):
    """Makes sure we have valid data for serving."""
    with self.assertRaisesRegex(ValueError,
                                'Missing data_dir in TFExampleData init'):
      TFExampleData('in', 'out', 100)

  def test_filter_create_dataset_bad_mode(self):
    data = TFExampleData('in', 'out', 100, data_dir=self._test_data_dir)
    with self.assertRaisesRegex(ValueError,
                                'mode must be one of test, validate or train'):
      data.create_dataset('fubar')

  def test_filter_file_names_both_types(self):
    data = BrainData('in', 'out', 100, train_file_pattern='allbut_fubar')
    with self.assertRaisesRegex(ValueError,
                                'Both test and validate must be specified'):
      data.filter_file_names('train')

  def test_filter_file_allbut_number(self):
    with self.assertRaisesRegex(ValueError,
                                'allbut_ spec must be an integer, not fubar.'):
      data = BrainData('in', 'out', 100, train_file_pattern='allbut_fubar',
                       test_file_pattern='whatever',
                       validate_file_pattern='whatever')
      data.filter_file_names('train')

  def test_filter_file_names(self):
    """Various tests to make sure we can filter the file names as needed."""
    def get_one_data(mode):
      filtered = test_brain_data.filter_file_names(mode)
      return [f.split('/')[-1] for f in filtered]

    frame_rate = 100.0
    # No specification ('') means all available files.
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    test_file_pattern='subj01',
                                    validate_file_pattern='subj01',
                                    data_dir=self._test_data_dir)
    filtered = get_one_data('train')
    self.assertLen(filtered, 3)  # All of them

    filtered = get_one_data('test')
    self.assertEqual(filtered, ['subj01_1ksamples.tfrecords'])

    filtered = get_one_data('validate')
    self.assertEqual(filtered, ['subj01_1ksamples.tfrecords'])

    # Make sure a bad pattern returns an empty train set
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    data_dir=self._test_data_dir,
                                    train_file_pattern='NotFoundPattern')

    filtered = get_one_data('train')
    self.assertCountEqual(filtered, [])

    filtered = get_one_data('test')
    self.assertCountEqual(filtered, ['subj01_1ksamples.tfrecords',
                                     'subj02_1ksamples.tfrecords',
                                     'subj03_1ksamples.tfrecords'])

    filtered = get_one_data('validate')
    self.assertCountEqual(filtered, ['subj01_1ksamples.tfrecords',
                                     'subj02_1ksamples.tfrecords',
                                     'subj03_1ksamples.tfrecords'])

    # Make sure allbut works... here where the validate and test list are empty
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    data_dir=self._test_data_dir,
                                    train_file_pattern='allbut',
                                    validate_file_pattern='NotFoundPattern',
                                    test_file_pattern='NotFoundPattern')

    filtered = get_one_data('train')
    self.assertCountEqual(filtered, ['subj01_1ksamples.tfrecords',
                                     'subj02_1ksamples.tfrecords',
                                     'subj03_1ksamples.tfrecords'])

    filtered = get_one_data('test')
    self.assertCountEqual(filtered, [])

    filtered = get_one_data('validate')
    self.assertCountEqual(filtered, [])

    # Normal case, where we have validate and test patterns, thus in this case
    # the train set will be empty.
    test_brain_data = TFExampleData('meg', 'envelope', frame_rate,
                                    data_dir=self._test_data_dir,
                                    train_file_pattern='allbut',
                                    validate_file_pattern='subj01',
                                    test_file_pattern='subj01')

    filtered = get_one_data('train')
    self.assertLen(filtered, 2)  # All but subj01

    filtered = get_one_data('test')
    self.assertEqual(filtered, ['subj01_1ksamples.tfrecords'])

    filtered = get_one_data('validate')
    self.assertEqual(filtered, ['subj01_1ksamples.tfrecords'])

if __name__ == '__main__':
  absltest.main()
