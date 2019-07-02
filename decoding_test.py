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

"""Tests for the decoding part of the library.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import shutil

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

from telluride_decoding import decoding

import numpy as np
import numpy.matlib
import scipy
from six.moves import range
import tensorflow as tf


class DecodingTest(absltest.TestCase):

  def setUp(self):
    super(DecodingTest, self).setUp()
    self._test_data = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/py/telluride_decoding/test_data/')

  def clear_model(self):
    model_dir = '/tmp/tf'
    try:
      # Clear out the model directory, if it exists, because the TF save model
      # code wants an empty directory.
      shutil.rmtree(model_dir)
    except OSError:
      pass

  ################## Linear data for testing ################################
  # Just a list of consecutive integers, to make it easier to debug batching
  # issues and the like.

  def create_linear_dataset(self, dataset, mode='test'):
    def transform(input_data):
      return input_data + 3000
    assert isinstance(dataset, decoding.BrainData)
    num_samples = 1000
    input_data = np.arange(num_samples).reshape((num_samples, 1))
    input_data = np.concatenate((input_data, num_samples + input_data), axis=1)
    output_data = transform(input_data[:, 0:1])
    dataset.preserve_test_data(input_data, output_data)
    return dataset.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_linear_batching(self):
    """Test the batching (pre and post context).

    This is key to the decoding datasets. Make sure the resulting dataset
    object is returning the right data.  Test with linearly increasing data, so
    we know to expect, especially as we add context.
    """
    print('********** test_linear_batching **************')
    batch_size = 999   # Big batch to minimize chance of random alignment.
    # No shift
    def create_test_dataset(pre_context, post_context, mode):
      test_dataset = decoding.BrainData('input', 'output',
                                        final_batch_size=batch_size,
                                        pre_context=pre_context,
                                        post_context=post_context,
                                        repeat_count=10)
      next_iterator, _ = self.create_linear_dataset(test_dataset, mode)
      with tf.compat.v1.Session() as sess:
        (input_data, output) = sess.run(next_iterator)
      input_data = input_data['x']
      print('Input_data.shape', input_data.shape)
      self.assertLen(input_data.shape, 2)
      self.assertGreater(input_data.shape[0], 0)
      self.assertGreater(input_data.shape[1], 0)
      print('test_linear_batch input:', input_data[0:3, :])
      print('test_linear_batch output:', output[0:3, :])
      return input_data, output, test_dataset

    def all_in_order(data):
      return np.all(data[1:, :] > data[0:-1, :])

    # First test without context, using test data so there is no shuffling.
    pre_context = 0
    post_context = 0
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           mode='test')
    num_input_channels = test_dataset.num_input_channels

    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 2)
    # pylint: disable=bad-whitespace
    self.assertTrue(np.all(np.equal(input_data[0:3, :], [[0, 1000],
                                                         [1, 1001],
                                                         [2, 1002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :], [[3000],
                                                     [3001],
                                                     [3002]])))
    self.assertEqual(test_dataset.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data))
    self.assertTrue(all_in_order(output))

    # Now test with training data, to make sure batches are randomized.
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           'train')
    self.assertFalse(all_in_order(input_data))
    self.assertFalse(all_in_order(output))
    self.assertTrue(np.all((input_data[:, 0] + 3000) == output[:, 0]))

    # Now test with just two samples of pre context
    pre_context = 2
    post_context = 0
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           mode='test')
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 6)
    # Last row of input and output should match, with extra context before.
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0,    0,   0,    0,   0, 1000],
                                     [0,    0,   0, 1000,   1, 1001],
                                     [0, 1000,   1, 1001,   2, 1002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[3000],
                                     [3001],
                                     [3002]])))
    self.assertEqual(test_dataset.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data[2:, :]))
    self.assertTrue(all_in_order(output))

    # Now test with just two samples of post context
    pre_context = 0
    post_context = 2
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           mode='test')
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 6)
    # First row of input and output should match, with extra context before.
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0, 1000, 1, 1001, 2, 1002],
                                     [1, 1001, 2, 1002, 3, 1003],
                                     [2, 1002, 3, 1003, 4, 1004]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[3000],
                                     [3001],
                                     [3002]])))
    self.assertEqual(test_dataset.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data[:-2, :]))
    self.assertTrue(all_in_order(output))

    # Now test with the data mixed up.
    flags.FLAGS.random_mixup_batch = True
    pre_context = 0
    post_context = 0
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           'train')
    self.assertFalse(all_in_order(input_data))
    self.assertFalse(all_in_order(output))
    matches = np.sum((input_data[:, 0] + 3000) == output[:, 0])
    print('Found %d matches in %d frames.' % (matches, input_data.shape[0]))
    self.assertLess(matches, input_data.shape[0]/64)
    # pylint: enable=bad-whitespace

  @flagsaver.flagsaver
  def test_linear_batching_both_sides(self):
    """Test the batching (pre and post context) for input *and* output.
    """
    print('********** test_linear_batching_both_sides **************')
    batch_size = 999   # Big batch to minimize chance of random alignment.
    # No shift
    def create_test_dataset(pre_context, post_context,
                            output_pre_context, output_post_context, mode):
      test_dataset = decoding.BrainData('input', 'output',
                                        final_batch_size=batch_size,
                                        pre_context=pre_context,
                                        post_context=post_context,
                                        output_pre_context=output_pre_context,
                                        output_post_context=output_post_context,
                                        repeat_count=10)
      next_iterator, _ = self.create_linear_dataset(test_dataset, mode)
      with tf.compat.v1.Session() as sess:
        (input_data, output) = sess.run(next_iterator)
      input_data = input_data['x']
      print('Input_data.shape', input_data.shape)
      self.assertLen(input_data.shape, 2)
      self.assertGreater(input_data.shape[0], 0)
      self.assertGreater(input_data.shape[1], 0)
      print('test_linear_batch input:', input_data[0:3, :])
      print('test_linear_batch output:', output[0:3, :])
      return input_data, output, test_dataset

    def all_in_order(data):
      return np.all(data[1:, :] > data[0:-1, :])

    # Now test with just two samples of output pre context
    pre_context = 0
    post_context = 0
    output_pre_context = 2
    output_post_context = 0
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           output_pre_context,
                                                           output_post_context,
                                                           mode='test')
    num_input_channels = test_dataset.num_input_channels
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 2)
    # Last row of input and output should match, with extra context before.
    # pylint: disable=bad-whitespace
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0, 1000],
                                     [1, 1001],
                                     [2, 1002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[   0,    0, 3000],
                                     [   0, 3000, 3001],
                                     [3000, 3001, 3002]])))
    self.assertEqual(test_dataset.input_fields_width(),
                     num_input_channels*(pre_context+1+post_context))
    self.assertTrue(all_in_order(input_data))
    self.assertTrue(all_in_order(output[2:, :]))

    # Now test with just two samples of post context.
    pre_context = 0
    post_context = 0
    output_pre_context = 0
    output_post_context = 2
    input_data, output, test_dataset = create_test_dataset(pre_context,
                                                           post_context,
                                                           output_pre_context,
                                                           output_post_context,
                                                           mode='test')
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(input_data.shape[1], 2)
    self.assertEqual(output.shape[1], 3)
    # First row of input and output should match, with extra context before.
    self.assertTrue(np.all(np.equal(input_data[0:3, :],
                                    [[0, 1000],
                                     [1, 1001],
                                     [2, 1002]])))
    self.assertTrue(np.all(np.equal(output[0:3, :],
                                    [[3000, 3001, 3002],
                                     [3001, 3002, 3003],
                                     [3002, 3003, 3004]])))
    self.assertTrue(all_in_order(input_data))
    self.assertTrue(all_in_order(output[:-2, :]))
   # pylint: enable=bad-whitespace

  ################## Simply scaled data ################################
  # Simple data for testing the machine learning.  Output is a simple transform
  # of the input data (sin in this case.) Use this data to test simple
  # regression, and data offsets.

  def simply_scaled_transform(self, input_data):
    """Transform to apply to the data.

    Kept as a separate function for easier testing. In this case, just the sin
    of the first column.

    Args:
      input_data: the input data

    Returns:
      the input_data transformed by this function.
    """
    assert len(input_data.shape) == 2
    return np.sin(input_data[:, 0:1] * 2 * np.pi)

  def create_simply_scaled_dataset(self, dataset, data_offset=0,
                                   num_input_channels=2, mode='test'):
    """A dataset where the output is a simple scalar function of the input."""
    assert isinstance(dataset, decoding.BrainData)

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
    dataset.preserve_test_data(input_data, output_data)
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

    def create_shifted_test(data_offset):
      test_dataset = decoding.BrainData('input', 'output',
                                        repeat_count=10,
                                        final_batch_size=batch_size,
                                        pre_context=pre_context,
                                        post_context=post_context)
      next_iterator, _ = self.create_simply_scaled_dataset(test_dataset,
                                                           data_offset)
      with tf.compat.v1.Session() as sess:
        (input_data, output) = sess.run(next_iterator)
      input_data = input_data['x']

      # Select the right column of the shifted input data to compute the
      # expect output.
      num_channels = test_dataset.num_input_channels
      index = num_channels * (pre_context + data_offset)
      expected_output = self.simply_scaled_transform(
          input_data[:, index:index+num_channels])
      if data_offset < 0:
        # Can't predict what we can't see, so force the first few samples to be
        # equal.
        expected_output[0:-data_offset, :] = output[0:-data_offset, :]
      return input_data, output, expected_output, test_dataset

    input_data, output, expected_output, test_dataset = create_shifted_test(0)
    num_channels = test_dataset.num_input_channels

    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     (pre_context + 1 + post_context)*num_channels)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(output.shape[1], 1)
    self.assertEqual(expected_output.shape[0], batch_size)
    self.assertEqual(expected_output.shape[1], 1)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    input_data, output, expected_output, test_dataset = create_shifted_test(2)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     (pre_context + 1 + post_context)*num_channels)
    self.assertEqual(output.shape[0], batch_size)
    self.assertEqual(output.shape[1], 1)
    self.assertEqual(expected_output.shape[0], batch_size)
    self.assertEqual(expected_output.shape[1], 1)
    self.assertLess(np.amax(np.abs(output - expected_output)), 1e-7)

    input_data, output, expected_output, test_dataset = create_shifted_test(-2)
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
    test_dataset = decoding.BrainData('input', 'output',
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    next_iterator, _ = self.create_linear_dataset(test_dataset, mode='train')
    with tf.compat.v1.Session() as sess:
      batch_count = 0
      try:
        while True:
          (eeg, envelope) = sess.run(next_iterator)
          batch_count += 1
          print('Test dataset returned %d items of size:' % batch_count,
                eeg['x'].shape,
                envelope.shape)
      except tf.errors.OutOfRangeError:
        pass
    print('batch_count is %d.' % batch_count)
    print('Impulse dataset returned %d items of size:' % batch_count,
          eeg['x'].shape,
          envelope.shape)
    self.assertEqual(eeg['x'].shape, (batch_size_request,
                                      test_dataset.num_input_channels))
    self.assertEqual(envelope.shape, (batch_size_request, 1))
    self.assertEqual(batch_count,
                     (repeat_count_request * 1000)//batch_size_request)

  @flagsaver.flagsaver
  def test_regression_tf(self):
    """Test simple (no temporal shift) regression."""
    print('\n\n**********test_regression_tf starting... *******')
    self.clear_model()
    repeat_count_request = 500
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'tf'
    test_dataset = decoding.BrainData('input', 'output',
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    self.create_simply_scaled_dataset(test_dataset, mode='train')
    model, results = decoding.create_train_estimator(
        test_dataset, hidden_units=[40, 20], steps=4000)
    print('test_regression_tf.create_train_estimator results:', results)
    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['average_loss']
    print('Test_regression produced an error of', rms_error)
    self.assertLess(rms_error, 0.2)

  @flagsaver.flagsaver
  def test_regression_fullyconnected(self):
    """Test simple (no temporal shift) regression."""
    print('\n\n**********test_regression_fullyconnected starting... *******')
    self.clear_model()
    repeat_count_request = 300
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    test_dataset = decoding.BrainData('input', 'output',
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    self.create_simply_scaled_dataset(test_dataset)
    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[100, 40, 20], steps=1000)
    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation']
    print('Test_regression produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .8)
    self.assertLess(rms_error, 0.2)

  @flagsaver.flagsaver
  def test_offset_shifts(self):
    """Simple tests to make sure we can shift the output in time."""
    def create_shifted_test(data_offset):
      test_dataset = decoding.BrainData('input', 'output',
                                        repeat_count=1,
                                        pre_context=max(-data_offset, 0),
                                        post_context=max(data_offset, 0))
      # next_iterator, _ = test_dataset.create_dataset('train')
      next_iterator, _ = self.create_simply_scaled_dataset(test_dataset,
                                                           data_offset)
      with tf.compat.v1.Session() as sess:
        (input_data, output) = sess.run(next_iterator)
      input_data = input_data['x']

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

  @flagsaver.flagsaver
  def test_offset_regression_positive(self):
    print('\n\n**********test_offset_regression_positive starting... *******')
    self.clear_model()
    pre_context = 1        # Limit the number of variables to permit convergence
    post_context = 1
    print('test_offset_regression_positive contexts:',
          pre_context, post_context)
    repeat_count_request = 500
    batch_size_request = 128
    data_offset = 1
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    num_channels = 1
    test_dataset = decoding.BrainData('input', 'output',
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request,
                                      pre_context=pre_context,
                                      post_context=post_context)
    self.create_simply_scaled_dataset(test_dataset, data_offset=data_offset,
                                      num_input_channels=num_channels)

    # Check the size of the dataset outputs.
    next_iterator, _ = test_dataset.create_dataset('train')
    with tf.compat.v1.Session() as sess:
      (input_data, output) = sess.run(next_iterator)
    input_data = input_data['x']
    self.assertEqual(input_data.shape[0], batch_size_request)
    self.assertEqual(input_data.shape[1],
                     num_channels*(pre_context + 1 + post_context))
    self.assertEqual(output.shape[0], batch_size_request)
    self.assertEqual(output.shape[1], 1)

    # Check to see if the correct answer (as computed from the input data
    # is in the output at the right spot.
    print('Input:', input_data[0:6, :])
    print('Output:', output[0:6, :])
    index = num_channels*pre_context + data_offset
    expected_output = self.simply_scaled_transform(
        input_data[:, index:index + 1])
    print('Expected Output:', expected_output[0:6, :])

    difference = output != expected_output
    self.assertEqual(difference.shape[1], 1)
    # Some frames are bad because they are at the beginning or end of the batch.
    # We look for 0.0 since the frames are shuffled, and we have no other way
    # of finding them.
    good_frames = np.nonzero(expected_output[:, 0] != 0.0)

    np.testing.assert_equal(output[good_frames, 0],
                            expected_output[good_frames, 0])

    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[50, 20, 20], steps=2000)
    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation']
    print('Test_offset_regression_positive produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .70)

  @flagsaver.flagsaver
  def test_offset_regression_negative(self):
    print('\n\n**********test_offset_regression_positive starting... *******')
    self.clear_model()
    pre_context = 1        # Limit the number of variables to permit convergence
    post_context = 1
    print('test_offset_regression_negative contexts:',
          pre_context, post_context)
    repeat_count_request = 500
    batch_size_request = 128
    data_offset = -1
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    num_channels = 1
    test_dataset = decoding.BrainData('input', 'output',
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request,
                                      pre_context=pre_context,
                                      post_context=post_context)
    self.create_simply_scaled_dataset(test_dataset, data_offset=data_offset,
                                      num_input_channels=num_channels)

    # Check the size of the dataset inputs and outputs.
    next_iterator, _ = test_dataset.create_dataset('train')
    with tf.compat.v1.Session() as sess:
      (input_data, output) = sess.run(next_iterator)
    input_data = input_data['x']
    self.assertEqual(input_data.shape[0], batch_size_request)
    self.assertEqual(input_data.shape[1],
                     num_channels*(pre_context + 1 + post_context))
    self.assertEqual(output.shape[0], batch_size_request)
    self.assertEqual(output.shape[1], 1)

    # Check to see if the correct answer (as computed from the input data
    # is in the output at the right spot.
    print('Input:', input_data[0:6, :])
    print('Output:', output[0:6, :])
    index = num_channels*pre_context + data_offset
    expected_output = self.simply_scaled_transform(
        input_data[:, index:index + 1])
    print('Expected Output:', expected_output[0:6, :])

    difference = output != expected_output
    self.assertEqual(difference.shape[1], 1)
    # Some frames are bad because they are at the beginning or end of the batch.
    # We look for 0.0 since the frames are shuffled, and we have no other way
    # of finding them.
    good_frames = np.nonzero(expected_output[:, 0] != 0.0)

    np.testing.assert_equal(output[good_frames, 0],
                            expected_output[good_frames, 0])

    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[50, 20, 20], steps=2000)
    metrics = decoding.evaluate_performance(test_dataset, model)
    print('The returned metrics are:', metrics)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation']
    print('Test_offset_regression_negative produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .70)

  ########################## Simple IIR Dataset ############################
  def create_simple_iir_dataset(self, dataset, num_input_channels=1,
                                mode='test'):
    """Given a pre-created generic dataset, load it with IIR data."""
    num_samples = 1000000
    input_data = np.random.randn(num_samples + 1,
                                 num_input_channels).astype(np.float32)
    output_data = 0.4 * input_data[0:-1,] + 0.6 * input_data[1:, :]
    dataset.preserve_test_data(input_data[1:num_samples + 1, :], output_data)
    return dataset.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_simple_iir_regression32(self):
    """Test simple (two time impulse response) regression."""
    print('\n\n**********test_simple_regression32 starting... *******')
    self.clear_model()
    repeat_count_request = 300
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    test_dataset = decoding.BrainData('input', 'output',
                                      pre_context=32,
                                      post_context=0,
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    self.create_simple_iir_dataset(test_dataset, num_input_channels=1)
    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[100, 40, 20], steps=400)

    print('test_simple_regression32: evaluating the regressor.')
    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation']
    print('Test_regression produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .90)
    self.assertLess(rms_error, 0.025)

  @flagsaver.flagsaver
  def test_simple_iir_regression0(self):
    """Test simple (two time impulse response) regression.

    Shouldn't do as well as the regression32 above, as there is not enough
    context.
    """
    print('\n\n**********test_simple_regression0 starting... *******')
    self.clear_model()
    repeat_count_request = 300
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    test_dataset = decoding.BrainData('input', 'output',
                                      pre_context=0,
                                      post_context=0,
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    self.create_simple_iir_dataset(test_dataset, num_input_channels=1)
    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[100, 40, 20], steps=400)
    print('test_simple_regression0: finished training the regressor.')

    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation']
    print('Test_regression produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .80)

  ################## Simulated EEG Data ################################
  # Simulated EEG data, similar to the one in the Telluride Decoding Toolbox.

  # Single random speech envelope, and then a simulated EEG
  # response different for each channel, is convolved with the speech envelope.
  # No noise so should be perfect reconstruction.

  def create_simulated_impulse_responses(self, num_input_channels,
                                         simulated_unattended_gain=0.10):
    """Create the basic impulse responses for this dataset.

    Generate random responses, one for the attended signal and the other for
    the unattended signal. Give them a realistic overall shape over  250ms.
    Do this once for a dataset (so we can generate multiple examples later
    using the same impulse responses).

    Args:
      num_input_channels: How many input channels to synthesize
      simulated_unattended_gain: How much lower is the unattended channel?

    Creates:
      attended_impulse_response, unattended_impulse_response:
        Two 0.25s x num_input_channel arrays representing the impulse response
        from the input sound (i.e. speech) to the system (i.e. EEG/meg)
        response.
    """
    self.impulse_length = .25  # Length of the TRF
    self.impulse_times = np.arange(self.impulse_length*self.fs) / self.fs
    self.envelope = 30*self.impulse_times * np.exp(-self.impulse_times*30,
                                                   dtype=np.float32)
    self.envelope_all_chans = numpy.matlib.repmat(np.reshape(self.envelope,
                                                             (-1, 1)),
                                                  1, num_input_channels)
    self.attended_impulse_response = np.random.randn(
        self.impulse_times.shape[0],
        num_input_channels) * self.envelope_all_chans
    self.unattended_impulse_response = np.random.randn(
        self.impulse_times.shape[0],
        num_input_channels) * self.envelope_all_chans
    # Cut the magnitude of the unattended inpulse response so as noise it will
    #  have a smaller or bigger effect.
    print('SimulatedData:initialize_dataset: Setting unattended gain to',
          simulated_unattended_gain)
    self.unattended_impulse_response *= simulated_unattended_gain

  def create_simulated_speech_signals(self):
    """Create the source signals for our experiment, attended and unattended.

    Now we create the "speech" signal, which will convolved with the EEG
    responses to create the data for this trial. Create speaker 1 and 2 signals.

    Creates:
      audio_signals: the N x 2 array of attended and unattended "speech"
        signals.
    """
    if self.use_sinusoids:
      self.audio_subj_1 = np.reshape(np.sin(self.recording_times*2*np.pi*5),
                                     (-1, 1))
      self.audio_subj_2 = np.reshape(np.sin(self.recording_times*2*np.pi*7),
                                     (-1, 1))
      self.audio_signals = np.concatenate((self.audio_subj_1,
                                           self.audio_subj_2),
                                          axis=1)
    else:
      # Create the signal with noise, and then linearly upsample so the result
      # is more low pass.
      num_output_channels = 2
      self.audio_low_freq = np.random.randn(math.ceil(len(self.recording_times)/
                                                      10.0),
                                            num_output_channels)
      self.audio_signals = scipy.signal.resample(self.audio_low_freq,
                                                 len(self.recording_times))
    print('SimulatedData: audio signals shape:', self.audio_signals.shape)

  def create_simulated_eeg_data(self, dataset, num_input_channels,
                                mode, simulated_noise_level=0.3):
    """Create one trial's worth of data.

    We have precomputed impulse responses
    so generate a "speech" signal, and convolve the EEG impulse responses with
    it.

    Args:
      dataset: The BrainData dataset for which to create data
      num_input_channels: How many input channels to generate
      mode: Mostly ignored unless you ask for "demo" when then generates a
        signal which oscillates between the two speakers.
      simulated_noise_level: How much noise to add for simulation.

    Returns:
      A TF dataset with the input and output data.
    """
    assert isinstance(dataset, decoding.BrainData)
    assert num_input_channels > 0
    self.signal_length = 1000  # Seconds
    self.fs = 100  # Audio and EEG sample rate in Hz
    self.use_sinusoids = 1  # Random or sinusoidal driving signal
    self.create_simulated_impulse_responses(num_input_channels)

    self.recording_times = np.arange(self.signal_length*self.fs)/float(self.fs)
    self.create_simulated_speech_signals()

    if mode.startswith('demo'):
      # Create the attention signal, switching every attention_duration seconds
      self.attention_duration = 25  # Time in seconds listening to each signal
      self.attention_signal = np.mod(
          np.floor(self.recording_times / self.attention_duration), 2)
      # This signal flips between 0 and 1, every attentionuration seconds
      a = np.reshape(self.attention_signal, (-1, 1))
    else:
      # Attention signal is constant, appropriate for testing.
      a = np.ones((self.recording_times.shape[0], 1), dtype=np.float32)
    self.attention_matrix = np.concatenate((1-a, a), axis=1)

    # Create the signals that correspond to the attended and unattended audio,
    # under control of the deterministic attention_signal created above.
    self.attended_audio = np.sum(self.attention_matrix * self.audio_signals,
                                 axis=1).astype(np.float32)
    self.unattended_audio = np.sum((1 - self.attention_matrix) *
                                   self.audio_signals,
                                   axis=1).astype(np.float32)

    # Now convolve the attended and unattended audio with the two different
    #  impulse responses to create the simulated EEG signals.
    print('Creating an example of the SimulatedData....')
    response = np.zeros((self.attended_audio.shape[0] +
                         self.attended_impulse_response.shape[0] - 1,
                         num_input_channels), dtype=np.float32)
    for c in range(num_input_channels):
      attended_response = np.convolve(self.attended_audio,
                                      self.attended_impulse_response[:, c],
                                      mode='full')
      unattended_response = np.convolve(self.unattended_audio,
                                        self.unattended_impulse_response[:, c],
                                        mode='full')
      # Sum up the attended and unattended response, and then add noise.
      response[:, c] = (
          attended_response + unattended_response +
          simulated_noise_level * np.random.randn(attended_response.shape[0]))
    print('Attended shapes:', self.attended_audio.shape,
          self.unattended_audio.shape)
    both_audio_channels = np.concatenate((np.reshape(self.attended_audio,
                                                     (-1, 1)),
                                          np.reshape(self.unattended_audio,
                                                     (-1, 1))),
                                         axis=1)
    print('Both_audio_channels shape:', both_audio_channels.shape)
    dataset.preserve_test_data(response[0:self.attended_audio.shape[0],
                                        0:num_input_channels],
                               both_audio_channels)
    return dataset.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_simulated_regression(self):
    """Test simulated eeg regression.

    This test starts with both attended and unattended audio, so we test the
    reconstruction accuracy of both signals.
    """
    print('\n\n**********test_simulated_regression starting... *******')
    self.clear_model()
    repeat_count_request = 500
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    num_input_channels = 32
    test_dataset = decoding.BrainData('input', 'output',
                                      pre_context=32,
                                      post_context=0,
                                      repeat_count=repeat_count_request,
                                      final_batch_size=batch_size_request)
    self.create_simulated_eeg_data(test_dataset, num_input_channels,
                                   mode='train')
    model, _ = decoding.create_train_estimator(
        test_dataset, hidden_units=[40, 20], steps=4000)
    metrics = decoding.evaluate_performance(test_dataset, model)
    rms_error = metrics['test/mse']
    r = metrics['test/pearson_correlation_matrix']
    print('Test_simualted_regression produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r[0, 2], 0.90)      # Attended reconstruction
    self.assertGreater(r[1, 3], 0.00)      # Unattended reconstruction
    self.assertGreater(r[0, 2], r[1, 3])   # Attended better than unattended

  @flagsaver.flagsaver
  def test_generic_read(self):
    """Test to see if we can parse a specific TFRecord data file."""
    self.longMessage = True     # For the asserts  pylint: disable=invalid-name
    all_files = []
    for (path, _, files) in tf.io.gfile.walk(self._test_data):
      all_files += [path + '/' + f for f in files if f.endswith('.tfrecords')]
    self.assertNotEmpty(all_files)

    mandatory_feature_sizes = {'envelope': (1, tf.float32),
                               'phonetic_features': (19, tf.float32),
                               'phonemes': (38, tf.float32),
                               'meg': (148, tf.float32),
                               'mel_spectrogram': (64, tf.float32),
                              }

    features = decoding.discover_feature_shapes(all_files[0])
    for k in mandatory_feature_sizes:
      expected_size, expected_type = mandatory_feature_sizes[k]
      self.assertIn(k, features, 'looking for feature ' + k)
      self.assertEqual(expected_size, features[k].shape[0],
                       'testing size of feature ' + k)
      self.assertEqual(expected_type, features[k].dtype,
                       'testing type of feature ' + k)

  @flagsaver.flagsaver
  def test_generic_processing(self):
    """Test reading the Generic TFRecord data.  Duplicates Rotman test.
    """
    print('********** test_generic_processing **************')
    batch_size = 128
    pre_context = 0
    post_context = 0

    flags.FLAGS.tfexample_dir = self._test_data

    def get_one_element(input_feature, output_feature):
      test_dataset = decoding.TFExampleData(input_feature, output_feature,
                                            final_batch_size=batch_size,
                                            pre_context=pre_context,
                                            post_context=post_context,
                                            repeat_count=10)
      next_iterator, _ = test_dataset.create_dataset('test')  # No shuffling
      with tf.compat.v1.Session() as sess:
        (input_data, output_data) = sess.run(next_iterator)
      input_data = input_data['x']
      return test_dataset, input_data, output_data

    input_feature = 'mel_spectrogram'
    output_feature = 'envelope'
    test_dataset, input_data, output_data = get_one_element(input_feature,
                                                            output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     test_dataset.features[input_feature].shape[0])
    self.assertTrue(np.all(input_data >= 0.0))

    self.assertEqual(output_data.shape[0], batch_size)
    self.assertEqual(output_data.shape[1],
                     test_dataset.features[output_feature].shape[0])
    self.assertTrue(np.all(output_data >= 0.0))

    input_feature = 'phonemes'
    output_feature = 'phonetic_features'
    test_dataset, input_data, output_data = get_one_element(input_feature,
                                                            output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     test_dataset.features[input_feature].shape[0])
    # Make sure input data (phonemes) is binary
    self.assertTrue(np.all(np.logical_or(input_data == 0, input_data == 1)))

    self.assertEqual(output_data.shape[0], batch_size)
    self.assertEqual(output_data.shape[1],
                     test_dataset.features[output_feature].shape[0])
    # Make sure output data (phonetic features) is binary
    self.assertTrue(np.all(np.logical_or(output_data == 0, output_data == 1)))

    input_feature = 'meg'
    output_feature = 'envelope'
    test_dataset, input_data, output_data = get_one_element(input_feature,
                                                            output_feature)
    self.assertEqual(input_data.shape[0], batch_size)
    self.assertEqual(input_data.shape[1],
                     test_dataset.features[input_feature].shape[0])
    # Already checked envelope, so we don't need to do it again.

  @flagsaver.flagsaver
  def test_linear_regression(self):
    # First test the basic linear regressor parameters using fixed weight and
    # bias vectors.
    desired_sample_count = 1000
    w_true = [[1, 3], [2, 4]]
    b_true = [[5, 6]]
    x_train = np.random.rand(desired_sample_count, 2).astype(np.float32)
    y_train = np.matmul(x_train, np.array(w_true)) + np.array(b_true)
    dataset = tf.data.Dataset.from_tensor_slices(({'x': x_train},
                                                  y_train))
    dataset = dataset.batch(100).repeat(count=1)
    (w_estimate,
     b_estimate,
     _, _) = decoding.calculate_regressor_parameters_from_dataset(
         dataset, lamb=0.0, use_offset=True)
    print('Estimated w:', w_estimate)
    print('Estimated b:', b_estimate)
    self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(b_true, b_estimate, atol=1e-05))

    (average_error,
     _,
     _,
     num_samples) = decoding.evaluate_regressor_from_dataset(
         w_estimate, b_estimate, dataset)
    self.assertLess(average_error, 1e-10)
    self.assertEqual(num_samples, desired_sample_count)

    # Now redo the test without the offset, and make sure the error is large.
    (w_estimate,
     b_estimate,
     _, _) = decoding.calculate_regressor_parameters_from_dataset(
         dataset, lamb=0.0, use_offset=False)
    print('Estimated w:', w_estimate)
    print('Estimated b:', b_estimate)
    # w will be different because the model no longer fits.
    # self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(0.0, b_estimate, atol=1e-05))

    (average_error,
     _,
     _,
     num_samples) = decoding.evaluate_regressor_from_dataset(w_estimate,
                                                             b_estimate,
                                                             dataset)
    self.assertGreater(average_error, 0.5)

    # Create a TF estimator, initialized with the test dataset above, and make
    # sure it is properly initialized.
    estimator = decoding.create_linear_estimator(dataset, lamb=0.0)

    # Need to run the model at least one step in order to initialize it, and
    # get checkpoints, etc.  But this leads to a warning:
    #    The graph of the iterator is different from the graph the dataset was
    #    created in.
    def my_input_fn(dataset):
      iterator = dataset.make_one_shot_iterator()
      batch_features, batch_labels = iterator.get_next()
      return batch_features, batch_labels
    estimator.train(input_fn=lambda: my_input_fn(dataset), steps=1)

    w_estimate = estimator.get_variable_value('linear_regressor/w')
    b_estimate = estimator.get_variable_value('linear_regressor/b')
    self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(b_true, b_estimate, atol=1e-05))

    # Now test with a new evaluation dataset
    x_eval = np.random.rand(1000, 2).astype(np.float32)
    y_eval = np.matmul(x_eval, np.array(w_true)) + np.array(b_true)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {'x': x_eval}, y_eval, batch_size=4, num_epochs=1, shuffle=False)

    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    self.assertLess(eval_metrics['loss'], 1e-5)

  @flagsaver.flagsaver
  def test_save_model(self):
    """Test simple (small FIR with temporal shift) regression."""
    def get_iir_predictions(brain_data):
      """Query the brain_data class to get a batch of data (input and output).

      Args:
        brain_data: The class which describes the model.

      Returns:
        x (input) and y (output) data for training/testing the IIR model
          (both arrays are np.ndarrays).
      """
      self.assertIsInstance(brain_data, decoding.BrainData)
      # Don't add temporal context here because it's done by the
      # save_estimator_model's _serving_input_receiver_fn
      dataset_next_element, _ = brain_data.create_dataset(
          'train', temporal_context=False)
      with tf.compat.v1.Session() as sess:
        iir_x, iir_y = sess.run(dataset_next_element)
      self.assertEqual(iir_x['x'].shape[0], batch_size_request)
      return iir_x['x'], iir_y

    def get_latest_model(model_dir):
      """Search the model directory for the latest checkpoint."""
      model_dir = model_dir or flags.FLAGS.saved_model_dir
      assert model_dir and isinstance(model_dir, str)
      if not model_dir.endswith('/'):
        model_dir += '/'
      all_dirs = tf.io.gfile.glob(model_dir + '[0-9]*')
      all_model_numbers = sorted([int(p.replace(model_dir, ''))
                                  for p in all_dirs])
      print('test_save_model: Sorted dir names from %s are:' %
            flags.FLAGS.saved_model_dir,
            sorted(all_model_numbers))
      return model_dir + str(all_model_numbers[-1])

    def create_test_dataset(test_data):
      """Given an np array, turn it into a TF.dataset for analysis."""
      assert isinstance(test_data, np.ndarray)
      test_brain_data = decoding.BrainData('input', 'output',
                                           pre_context=pre_context,
                                           post_context=post_context,
                                           repeat_count=repeat_count_request,
                                           final_batch_size=batch_size_request)
      test_brain_data.preserve_test_data(test_data, 0*test_data)
      _, dataset = test_brain_data.create_dataset(mode='test')
      return dataset

    self.clear_model()
    pre_context = 32
    post_context = 0
    repeat_count_request = 300
    batch_size_request = 128
    flags.FLAGS.dnn_regressor = 'fullyconnected'
    flags.FLAGS.saved_model_dir = '/tmp/tf/saved_model'

    # Create the dataset and train the model.
    num_input_channels = 1
    test_brain_data = decoding.BrainData('input', 'output',
                                         pre_context=pre_context,
                                         post_context=post_context,
                                         repeat_count=repeat_count_request,
                                         shuffle_buffer_size=0,
                                         final_batch_size=batch_size_request)
    self.create_simple_iir_dataset(test_brain_data, mode='train',
                                   num_input_channels=num_input_channels)
    iir_model, train_results = decoding.create_train_estimator(
        test_brain_data, hidden_units=[20], steps=400)
    print('test_save_model training results:', train_results)

    # Evaluate performance using the Estimator's method
    full_iir_metrics = decoding.evaluate_performance(test_brain_data, iir_model)
    print('test_save_model full_iir metrics keys:',
          list(full_iir_metrics.keys()))
    rms_error = full_iir_metrics['test/mse']
    r = full_iir_metrics['test/pearson_correlation']
    print('test_save_model: full_iir training produced an error of', rms_error,
          'and a correlation of', r)
    self.assertGreater(r, .95)
    self.assertLess(rms_error, 0.025)

    # Evaluate the performance using our own data.
    # Test with some real data, so we can evaluate the performance ourself.
    iir_x, iir_y = get_iir_predictions(test_brain_data)
    my_iir_test_output = iir_model.predict(lambda: create_test_dataset(iir_x),
                                           yield_single_examples=False)
    # Hack!  Not sure why predict() finalized the graph!  Undo it.
    tf.get_default_graph()._unsafe_unfinalize()
    print('After creating prediction fcn, finalize is',
          tf.get_default_graph().finalized)
    my_iir_test_results = next(my_iir_test_output)
    print('After calling predict method, finalize is',
          tf.get_default_graph().finalized)
    cor = np.corrcoef(iir_y.T, my_iir_test_results['predictions'].T)
    print('my_iir_test_results correlation is:', cor)
    self.assertGreater(cor[0, 1], 0.98)

    # Save the model on disk
    decoding.save_estimator_model(iir_model, test_brain_data,
                                  flags.FLAGS.saved_model_dir)

    # Now restore the model.
    latest_model_loc = get_latest_model(flags.FLAGS.saved_model_dir)
    print('Saved_model_dir latest model:', latest_model_loc)
    all_files = tf.io.gfile.listdir(latest_model_loc)
    print('Saved_model_dir model_dir contains:', all_files)

    predict_fn = tf.contrib.predictor.from_saved_model(
        latest_model_loc, signature_def_key=None)

    # Test with some real data with the predictor we get by restoring the model
    restored_predictions = predict_fn({'input': iir_x})

    self.assertEqual(restored_predictions['predictions'].shape[0],
                     batch_size_request)

    cor = np.corrcoef(iir_y.T, restored_predictions['predictions'].T)
    print('test_save_model test results:', cor)
    self.assertGreater(cor[0, 1], 0.98)


if __name__ == '__main__':
  absltest.main()
