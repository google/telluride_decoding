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

import io
import os
import sys

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import matplotlib
# pylint: disable=g-import-not-at-top
matplotlib.use('Agg')    # Needed for plotting to a file, before the next import
import matplotlib.pyplot as plt
import mock
import numpy as np

from telluride_decoding import brain_data
from telluride_decoding import infer_decoder
from telluride_decoding import ingest

import tensorflow as tf

flags.DEFINE_string(
    'tmp_dir', os.environ.get('TMPDIR') or '/tmp',
    'Temporary directory location.')

FLAGS = flags.FLAGS


@tf.function
def _linear_model(input_dict):
  """The simplest possible linear model for testing.

  Args:
    input_dict: A TF dataset, only one field needed (input_1) containing the
      EEG data from which we predict intensity.

  Returns:
    The predicted intensity
  """
  eeg = input_dict['input_1']
  return _eeg_to_intensity(eeg)


@tf.function
def _cca_model(input_dict, cca_dims=2):
  """The simplest possible CCA model for testing.

  Args:
    input_dict: A TF dataset with two fields that are rotated via CCA.
    cca_dims: How many CCA dimensions to compute.
  Returns:
    A concatenated pair of arrays with the best correlation.
  """
  return tf.concat((input_dict['input_1'][:, 0:cca_dims],   # EEG data
                    input_dict['input_2'][:, 0:cca_dims]),  # Intensity data
                   axis=1)


def _eeg_to_intensity(eeg):
  """Intensity is uniform random between [0, 1], eeg is [-1, 1]."""
  return eeg/2.0 + 0.5


def _intensity_to_eeg(intensity):
  """Intensity is uniform random between [0, 1], eeg is [-1, 1]."""
  return (intensity - 0.5)*2.0

_NUM_TEST_POINTS = 1000   # Arbitrary for testing.


class InferDecoderTest(parameterized.TestCase):

  def setUp(self):
    """Stores and prepares tf.dataset tests with three kinds of test data.

    These data are:
      Plain training data,
      More training data, but with input and output mixed up for null test,
      Test data which switches attention periodically.
    """
    super(InferDecoderTest, self).setUp()
    params = self.get_default_params()
    attended_speaker = 'intensity1'
    self._train_filename = self.create_sample_data_file(with_noise=False,
                                                        with_switches=False)
    self._train_data = infer_decoder.create_dataset(self._train_filename,
                                                    params,
                                                    attended_speaker)
    self._mixed_data = infer_decoder.create_dataset(self._train_filename,
                                                    params,
                                                    attended_speaker,
                                                    mixup_batch=True)

    self._test_filename = self.create_sample_data_file(with_noise=False,
                                                       with_switches=True)
    self._test_data = infer_decoder.create_dataset(self._test_filename, params,
                                                   attended_speaker)

  def create_sample_data_file(self, test_name='test',
                              num_dimensions=4,
                              with_switches=False, with_noise=False):
    """Create a TFRecord data file with two intensity profiles and EEG data."""
    intensity1 = np.random.rand(_NUM_TEST_POINTS, num_dimensions)
    intensity2 = np.random.rand(_NUM_TEST_POINTS, num_dimensions)
    speaker_flag = np.zeros((_NUM_TEST_POINTS), dtype=np.int32)
    if with_switches:
      # Switch to speaker 2 for second half
      speaker_flag[_NUM_TEST_POINTS//2:] = 1

    eeg = np.zeros((_NUM_TEST_POINTS, num_dimensions))
    eeg[speaker_flag == 0, :] = _intensity_to_eeg(
        intensity1[speaker_flag == 0, :])
    eeg[speaker_flag == 1, :] = _intensity_to_eeg(
        intensity2[speaker_flag == 1, :])

    if with_noise:
      for i in range(num_dimensions):
        frac = i/float(num_dimensions)
        eeg[:, i] = (1-frac)*eeg[:, i] + frac*np.random.rand(_NUM_TEST_POINTS,)

    data_dict = {'intensity1': intensity1,
                 'intensity2': intensity2,
                 'attended_speaker': speaker_flag.astype(np.float32),
                 'eeg': eeg,
                }

    brain_trial = ingest.BrainTrial(test_name)
    brain_trial.model_features = data_dict
    data_dir = self.create_tempdir().full_path
    brain_trial.write_data_as_tfrecords(data_dir)
    return os.path.join(data_dir, test_name + '.tfrecords')

  def get_default_params(self):
    return {'input_field': ['eeg'],
            'pre_context': 0,
            'post_context': 0,
            'input2_pre_context': 0,
            'input2_post_context': 0,
           }

  def test_sample_data_file(self):
    """Basic test to make sure we can create the data file and it has data."""
    num_dimensions = 4

    features = brain_data.discover_feature_shapes(self._train_filename)
    print('sample_data_file features are:', features)
    self.assertEqual(features['eeg'].shape, [num_dimensions])
    self.assertEqual(features['intensity1'].shape, [num_dimensions])
    self.assertEqual(features['intensity2'].shape, [num_dimensions])

    count, error = brain_data.count_tfrecords(self._train_filename)
    self.assertEqual(count, _NUM_TEST_POINTS)
    self.assertFalse(error)

  def test_conversions(self):
    """Makes sure that the model mapping is invertable."""
    data = np.random.rand(1000)
    converted = _eeg_to_intensity(_intensity_to_eeg(data))
    np.testing.assert_allclose(data, converted, rtol=1e-5)

  def test_create_dataset(self):
    """Test to see if we can create the right data file for testing a model."""

    num_batches = 0
    for input_data, output_data in self._test_data.take(1):
      predicted_intensity = _eeg_to_intensity(input_data['input_1'].numpy())
      print('Types:', predicted_intensity.dtype, output_data.numpy().dtype)
      print('Shapes:', predicted_intensity.shape, output_data.numpy().shape)
      np.testing.assert_allclose(predicted_intensity,
                                 output_data.numpy(), atol=1e-7, rtol=1e-4)
      num_batches += 1
    self.assertGreater(num_batches, 0)

  def test_correlation_calculation(self):
    num_batches = 50  # Arbitrary
    batch_size = 3400  # Arbitrary
    total_points = num_batches * batch_size
    x = np.random.randn(total_points, 3) + 1.2
    y = x*3 + 3.1
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model)
    for i in range(num_batches):
      s = i*batch_size
      e = s + batch_size
      decoder.add_data_correlator(x[s:e, :], y[s:e, :])
    r = decoder.compute_correlation(x, y)
    np.testing.assert_allclose(np.mean(r), 1, rtol=1e-5)

  def test_correlation_save_model(self):
    num_batches = 50  # Arbitrary
    batch_size = 340  # Arbitrary
    total_points = num_batches * batch_size
    x = np.random.randn(total_points, 3) + 1.2
    y = x*3 + 3.1
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model)
    decoder.add_data_correlator(x, y)
    x_new = np.random.randn(total_points, 3) + 1.2
    y_new = x_new*3 + 3.1
    r = decoder.compute_correlation(x_new, y_new)
    tmp_dir = flags.FLAGS.test_tmpdir or '/tmp'
    corr_save_dir = os.path.join(tmp_dir, 'corr_params.json')
    decoder.save_parameters(corr_save_dir)

    decoder_loaded = infer_decoder.LinearRegressionDecoder(_linear_model)
    decoder_loaded.restore_parameters(corr_save_dir)

    r_loaded = decoder_loaded.compute_correlation(x_new, y_new)
    np.testing.assert_equal(r_loaded, r)

  def test_linear_model(self):
    """Makes sure our sample TF model performs as expected."""
    intensity = np.arange(10) - 5.1  # Arbitrary set of non-positive, non-ints
    eeg = _intensity_to_eeg(intensity)
    prediction = _linear_model({'input_1': eeg})
    np.testing.assert_allclose(intensity, prediction)

  def test_cca_data(self):
    """Checks the data is being loaded into the input_dict correctly for CCA."""
    def pearson_correlation(x, y):
      """Computes the Pearson correlation coefficient between tensors of data.

      This routine computes a vector correlation (ala cosine distance).

      Args:
        x: one of two input arrays.
        y: second of two input arrays.

      Returns:
        scalar correlation coefficient.
      """
      # From: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
      x_m = x - tf.math.reduce_mean(x, axis=0)
      y_m = y - tf.math.reduce_mean(y, axis=0)
      return tf.divide(
          tf.math.reduce_sum(tf.multiply(x_m, y_m), axis=0),
          tf.multiply(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x_m),
                                                      axis=0)),
                      tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_m),
                                                      axis=0))))

    for input_dict, _ in self._test_data.take(1):
      self.assertGreater(np.mean(np.abs(input_dict['input_1'] -
                                        input_dict['input_2'])), 0.1)
      r = pearson_correlation(input_dict['input_1'], input_dict['input_2'])
      np.testing.assert_allclose(r, 1.0, rtol=1e-5)

  @parameterized.named_parameters(
      ('lda', 'lda'),
      ('first', 'first'),
      ('mean', 'mean'),
      ('mean-squared', 'mean-squared'),
      )
  def test_inference(self, reduction):
    """Tests the training and inference stages with a linear model."""

    # Create the basic decoder class, with a simple TF model.
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model,
                                                    reduction=reduction)
    decoder.train(self._mixed_data, self._train_data)

    speaker, labels = decoder.test_all(self._test_data)

    plt.clf()
    plt.plot(labels)
    plt.plot(speaker)
    plt.savefig(os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             'inference_%s.png' % reduction))

    print('test_inference_%s:' % reduction, speaker.shape, labels.shape)
    self.assertGreater(np.mean(speaker[labels == 0]), 0.5)
    self.assertLess(np.mean(speaker[labels == 1]), 0.5)

  @parameterized.named_parameters(
      ('lda', 'lda', 0.85),
      ('first', 'first', 0.6),
      ('mean', 'mean', 0.85),
      ('mean-squared', 'mean-squared', 0.85),
      )
  def test_windowed_inference(self, reduction, expected_mean):
    """Tests the training and inference stages with a linear model."""

    # Create the basic decoder class, with a simple TF model.
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model,
                                                    reduction=reduction)
    decoder.train(self._mixed_data, self._train_data)
    speaker, _ = decoder.test_all(self._test_data)

    window_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    windowed_means = np.zeros(len(window_sizes))
    windowed_stds = np.zeros(len(window_sizes))
    for i, window_size in enumerate(window_sizes):
      results = []
      # Evaluate performance on first half of the training data
      for window_start in range(0, _NUM_TEST_POINTS//2,
                                window_size):
        window_end = window_start + window_size
        results.append(np.mean(speaker[window_start:window_end] > 0.5))
      windowed_means[i] = np.mean(results)
      windowed_stds[i] = np.std(results)

    plt.clf()
    plt.errorbar(window_sizes, windowed_means, windowed_stds)
    plt.gca().set_xscale('log')
    plt.title('Test_windowed_inference with %s' % reduction)
    plt.savefig(os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             'windowed_inference_%s.png' % reduction))
    plt.xlabel('Window size (frames)')

    self.assertAlmostEqual(np.mean(windowed_means), expected_mean, delta=0.05)

  def test_one_window(self):
    """Tests the training and inference stages with a linear model."""
    # Create the basic decoder class, with a simple TF model.
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model)
    decoder.train(self._mixed_data, self._train_data)
    batch_size = 101
    for speaker, label in decoder.test_by_window(self._test_data, batch_size):
      self.assertEqual(speaker.shape, (batch_size, 1))
      self.assertEqual(label.shape, (batch_size, 1))

  def test_train_no_switches(self):
    """Tests the training and inference stages with a linear model."""

    # Create the basic decoder class, with a simple TF model.
    decoder = infer_decoder.LinearRegressionDecoder(_linear_model)
    empty_dataset = tf.data.Dataset.from_tensor_slices(({'input_1': [],
                                                         'input_2': []},
                                                        []))
    with self.assertRaisesRegex(ValueError, 'No data for class 0'):
      decoder.train(empty_dataset, self._mixed_data)
    with self.assertRaisesRegex(ValueError, 'No data for class 1'):
      decoder.train(self._mixed_data, empty_dataset)

  def test_windowing(self):
    data = np.reshape(np.arange(12), (6, 2))
    ave = infer_decoder.average_data(data, window_size=3)
    expected = [[2, 3], [8, 9]]
    np.testing.assert_array_equal(ave, expected)

  @parameterized.named_parameters(
      ('linear_first', 'linear', 'first', 0.1, 1),
      ('linear_lda', 'linear', 'lda', 0.1, 1),
      ('linear_mean_squared', 'linear', 'mean-squared', 0.1, 1),
      ('CCA_first', 'CCA', 'first', 0.1, 1),
      ('CCA_lda', 'CCA', 'lda', 0.16, 1),
      ('CCA_mean_squared', 'CCA', 'mean-squared', 0.1, 1),

      ('linear_first-100', 'linear', 'first', 0.15, 100),
      ('linear_lda-100', 'linear', 'lda', 0.1, 100),
      ('linear_mean_squared-100', 'linear', 'mean-squared', 0.1, 100),
      ('CCA_first-100', 'CCA', 'first', 0.1, 100),
      ('CCA_lda-100', 'CCA', 'lda', 0.16, 100),
      ('CCA_mean_squared-100', 'CCA', 'mean-squared', 0.1, 100),
      )
  def test_training_and_inference(self, regressor_name, reduction,
                                  tolerance=0.1,
                                  window_size=1):
    """Tests the training and inference stages with a linear model."""
    print('Training the %s regressor.' % regressor_name)

    # Create the desired decoder class.
    if regressor_name == 'linear':
      decoder = infer_decoder.LinearRegressionDecoder(_linear_model,
                                                      reduction=reduction)
    elif regressor_name == 'CCA':
      decoder = infer_decoder.CCADecoder(_cca_model, reduction=reduction)
    else:
      raise ValueError('Unknown decoder name: %s' % regressor_name)

    dprime = decoder.train(self._mixed_data, self._train_data,
                           window_size=window_size)
    logging.info('Infer training of %s data via %s gave a dprime of %g.',
                 regressor_name, reduction, dprime)
    speaker, _ = decoder.test_all(self._test_data)

    plt.clf()
    plt.plot(speaker)
    plt.savefig(os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             'inference_train_%s_%s.png' % (regressor_name,
                                                            reduction)))

    self.assertGreater(np.mean(speaker[:(_NUM_TEST_POINTS//2)]),
                       1.0 - tolerance)
    self.assertLess(np.mean(speaker[(_NUM_TEST_POINTS//2):]),
                    tolerance)

    # Make sure we can retrieve and save parameters (without errors)
    decoder.decoding_model_params = decoder.decoding_model_params

  def test_two_dimensional_data(self):
    """A copy of the easiest test from scaled_lda_test. Just to verify function.
    """
    num_dims = 2
    mean_vectors = np.array([[-2, 12], [2, -1]])

    d1 = np.matmul(np.random.randn(_NUM_TEST_POINTS, num_dims),
                   [[2, 0], [0, 0.5]]) + mean_vectors[0, :]
    d2 = np.matmul(np.random.randn(_NUM_TEST_POINTS, num_dims),
                   [[2, 0], [0, 0.5]]) + mean_vectors[1, :]
    # Plot the original data.
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(d1[:, 0], d1[:, 1], 'rx')
    plt.plot(d2[:, 0], d2[:, 1], 'bo')
    plt.title('Original Data')

    x = np.concatenate((d1, d2), axis=0)
    labels = [42, -12]
    y = np.concatenate((np.ones(d1.shape[0])*labels[0],
                        np.ones(d2.shape[0])*labels[1]))

    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    dprime = decoder.compute_lda_model(d1, d2)
    logging.info('test_two_dimensional_data dprime is: %g', dprime)
    self.assertAlmostEqual(dprime, 26.3253, delta=2.0)

    x_lda = decoder.reduce_with_lda(x)
    # Plot the transformed data.
    plt.subplot(2, 1, 2)
    plt.plot(x_lda[y == labels[0], 0], x_lda[y == labels[0], 1], 'rx')
    plt.plot(x_lda[y == labels[1], 0], x_lda[y == labels[1], 1], 'bo')
    plt.title('Transfomed Data')

    plt.savefig(os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             'scaled_lda.png'))

    # Make sure the transformed centers are symmetric on the first (x) axis.
    centers = decoder.reduce_with_lda(mean_vectors)
    logging.info('Transformed centers are: %s', (centers,))
    self.assertAlmostEqual(centers[0, 0], 0., delta=0.1)
    self.assertAlmostEqual(centers[1, 0], 1., delta=0.1)

  def generate_dprime_data(self):
    dims = 10

    # Create two datasets, with coupled dimensions (decreasing with dim. index)
    d1 = np.random.randn(_NUM_TEST_POINTS, dims)
    d2 = np.random.randn(_NUM_TEST_POINTS, dims)
    for i in range(dims):
      p = 2**(-i)
      d2[:, i] = p*d1[:, i] + (1-p)*d2[:, i]
    d2 += np.ones(d2.shape)
    return d1, d2

  def test_lda(self):
    d1, d2 = self.generate_dprime_data()

    # Build and transform the sample data.
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing

    with self.assertRaisesRegex(
        ValueError, 'Must compute the LDA model before reducing data.'):
      decoder.reduce_with_lda(24)

    dprime = decoder.compute_lda_model(d1, d2)
    self.assertAlmostEqual(dprime, 3.31, delta=.1)
    all_data = np.concatenate((d1, d2), axis=0)

    with self.assertRaisesRegex(
        TypeError, 'Input data must be an numpy array, not'):
      decoder.reduce_with_lda(24)

    transformed_data = decoder.reduce_with_lda(all_data)
    self.assertEqual(transformed_data.shape, (2*_NUM_TEST_POINTS,
                                              2))

    dprime = infer_decoder.calculate_dprime(decoder.reduce_with_lda(d1)[:, 0],
                                            decoder.reduce_with_lda(d2)[:, 0])
    self.assertAlmostEqual(dprime, 3.28, delta=.1)

  def test_lda_save_model(self):
    d1, d2 = self.generate_dprime_data()
    # Build and transform the sample data.
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    _ = decoder.compute_lda_model(d1, d2)
    all_data = np.concatenate((d1, d2), axis=0)
    transformed_data = decoder.reduce_with_lda(all_data)
    dprime = infer_decoder.calculate_dprime(decoder.reduce_with_lda(d1)[:, 0],
                                            decoder.reduce_with_lda(d2)[:, 0])
    print(decoder.model_params)
    tmp_dir = flags.FLAGS.test_tmpdir or '/tmp'
    save_lda_dir = os.path.join(tmp_dir, 'lda_params.json')
    decoder.save_parameters(save_lda_dir)

    decoder_loaded = infer_decoder.Decoder(lambda x: x)
    decoder_loaded.restore_parameters(save_lda_dir)
    transformed_data_loaded = decoder_loaded.reduce_with_lda(all_data)
    dprime_loaded = infer_decoder.calculate_dprime(
        decoder_loaded.reduce_with_lda(d1)[:, 0],
        decoder_loaded.reduce_with_lda(d2)[:, 0])
    np.testing.assert_array_equal(transformed_data, transformed_data_loaded)
    np.testing.assert_array_equal(dprime, dprime_loaded)

  def test_dprime(self):
    """Makes sure our d' calculation is correct."""
    num = 1000
    np.random.seed(0)
    d1 = np.random.randn(num)
    d2 = np.random.randn(num) + 1
    dprime = infer_decoder.calculate_dprime(d1, d2)
    self.assertAlmostEqual(dprime, 1.0, delta=0.1)

  def test_correlator(self):
    x = np.random.randn(_NUM_TEST_POINTS)
    y = 2*x + 1

    # Test with simple known correlation (r=1)
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    decoder.add_data_correlator(x, y)
    r = np.mean(decoder.compute_correlation(x, y))
    print('test_correlation r is', r)

    self.assertAlmostEqual(r, 1, delta=1e-5)  # Add noise

    # Test with inhomogenous data,
    x = np.random.randn(_NUM_TEST_POINTS)
    half_num_points = _NUM_TEST_POINTS//2
    y[:half_num_points] = 2*x[:half_num_points] + 1
    y[half_num_points:] = 3*x[half_num_points:] + 4
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    decoder.add_data_correlator(x, y)
    r_whole = np.mean(decoder.compute_correlation(x, y))
    print('test_correlation r_whole is', r_whole)

    # Make sure we get the same answer when we add parts of the data at
    # different times (two differet blocks).
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    decoder.add_data_correlator(x[:half_num_points], y[:half_num_points])
    decoder.add_data_correlator(x[half_num_points:], y[half_num_points:])
    r_parts = np.mean(decoder.compute_correlation(x, y))
    print('test_correlation r_parts is', r_parts)
    self.assertAlmostEqual(r_parts, r_whole, delta=1e-5)

    # Now see if we get the same answer is we compute the output in blocks and
    # average.
    decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
    decoder.add_data_correlator(x, y)
    r_whole = np.mean(decoder.compute_correlation(x, y))
    print('test_correlation r_whole is', r_whole)

    r_part1 = decoder.compute_correlation(x[:half_num_points],
                                          y[:half_num_points])
    r_part1 = np.mean(r_part1)
    self.assertGreater(r_part1, -1)  # Make sure not a nan
    r_part2 = decoder.compute_correlation(x[half_num_points:],
                                          y[half_num_points:])
    r_part2 = np.mean(r_part2)
    self.assertGreater(r_part2, -1)  # Make sure not a nan
    print('test_correlation r_parts are:', r_part1, r_part2)
    self.assertAlmostEqual(r_whole, (r_part1+r_part2)/2, delta=1e-5)

  def test_attended_signal(self):

    num_speaker_1 = 0
    num_speaker_2 = 0
    for input_data, _ in self._test_data:
      num_speaker_1 += np.sum(input_data['attended_speaker'] == 0)
      num_speaker_2 += np.sum(input_data['attended_speaker'] == 1)
    print('test_attended_signal num_points:', num_speaker_1, num_speaker_2)
    half_num_points = _NUM_TEST_POINTS//2
    self.assertEqual(num_speaker_1, half_num_points)
    self.assertEqual(num_speaker_2, half_num_points)

  def test_load_model(self):
    """Tests to make sure we can load a model and get back the right parameters.
    """
    def dummy(x, _):
      """Needed to match functions in saved linear model."""
      return x

    # Not sure why I need to add _main to the Bazel path.
    test_model_dir = os.path.join(flags.FLAGS.test_srcdir, '_main',
                                  'test_data/linear_model')
    # Make sure these files are where they are supposed to be.
    self.assertTrue(os.path.exists(test_model_dir))
    self.assertTrue(os.path.exists(os.path.join(
        test_model_dir, 'saved_model.pb')))
    self.assertTrue(os.path.exists(os.path.join(
        test_model_dir, 'variables/variables.data-00000-of-00001')))
    self.assertTrue(os.path.exists(os.path.join(
        test_model_dir, 'variables/variables.index')))

    object_dict = {'BrainCcaLayer': dummy,
                   'pearson_correlation': dummy,
                   'cca_pearson_correlation_first': dummy,
                  }

    decoder = infer_decoder.Decoder()
    with self.assertRaisesRegex(
        ValueError, 'Model has not been initialized yet. Use load_model'):
      decoder.check_model_and_data(None)

    decoder.load_decoding_model(test_model_dir, object_dict)
    self.assertTrue(callable(decoder.decoding_model))
    self.assertIsInstance(decoder.decoding_model_params, dict)
    self.assertIsInstance(decoder.model_inputs, dict)

    input_1_size = [100, 1364]
    input_2_size = [100, 44]
    output_size = [100, 1]
    self.assertDictEqual({'input_1': input_1_size, 'input_2': input_2_size},
                         decoder.model_inputs)
    self.assertEqual(decoder.model_output, output_size)

    # Now look for database issues.
    with self.assertRaisesRegex(TypeError,
                                'Actual_dataset is not a dataset, but a'):
      decoder.check_model_and_data(42)

    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_1': np.ones(input_1_size),
         'input_2': np.ones(input_2_size)},
        np.ones(output_size))).batch(input_1_size[0])
    decoder.check_model_and_data(dataset)  # No errors.

    with self.assertRaisesRegex(TypeError,
                                'Data for input_1 has the wrong shape'):
      dataset = tf.data.Dataset.from_tensor_slices((
          {'input_1': np.ones(input_2_size),  # Woops
           'input_2': np.ones(input_2_size)},
          np.ones(output_size))).batch(input_1_size[0])
      decoder.check_model_and_data(dataset)  # Shape error

    with self.assertRaisesRegex(TypeError,
                                'Can\'t find needed key input_1 in input_data'):
      dataset = tf.data.Dataset.from_tensor_slices((
          {'input_2': np.ones(input_2_size)},
          np.ones(output_size))).batch(input_1_size[0])
      decoder.check_model_and_data(dataset)  # Missing key error

    with self.assertRaisesRegex(TypeError,
                                'Output data has the wrong shape, expected'):
      dataset = tf.data.Dataset.from_tensor_slices((
          {'input_1': np.ones(input_1_size),
           'input_2': np.ones(input_2_size)},
          np.ones((output_size[0], 3)))).batch(input_1_size[0])  # Woops
      decoder.check_model_and_data(dataset)  # Shape error

  def test_errors(self):
    with self.assertRaisesRegex(
        TypeError, 'Must supply a callable model when initializing'):
      decoder = infer_decoder.Decoder(42)

    with self.assertRaisesRegex(
        TypeError, r'Must provide a file name \(string\) to load-model.'):
      decoder = infer_decoder.Decoder()
      decoder.load_decoding_model(42, 42)
    with self.assertRaisesRegex(
        TypeError, 'If providing an object dictionary, it must be a dict.'):
      decoder = infer_decoder.Decoder()
      decoder.load_decoding_model('foo', 42)

    with self.assertRaisesRegex(
        TypeError, 'Averaging data must be two dimensional.'):
      infer_decoder.average_data(np.ones((42, 42, 42)), 42)

    with self.assertRaisesRegex(
        ValueError, 'Window size .* must be greater-than or equal to zero.'):
      infer_decoder.average_data(np.ones((42, 42)), -2)

    with self.assertRaisesRegex(
        TypeError, 'Must feed training routine data0 with a tf.data.Dataset'):
      decoder = infer_decoder.Decoder()
      decoder.train('not a dictionary', 'also not a dictionary')

    with self.assertRaisesRegex(
        TypeError, 'Must feed training routine data1 with a tf.data.Dataset'):
      decoder = infer_decoder.Decoder()
      decoder.train(self._train_data, 'also not a dictionary')

    with self.assertRaisesRegex(
        NotImplementedError, 'Must be implemented by a subclass.'):
      decoder = infer_decoder.Decoder()
      decoder.decode_one(42, 42)

    with self.assertRaisesRegex(
        TypeError, 'Input d1 must be an numpy array, not'):
      decoder = infer_decoder.Decoder(lambda x: x)  # Dummy model for testing
      decoder.compute_lda_model(42, np.ones((42, 42)))

    with self.assertRaisesRegex(
        ValueError, 'Unknown reduction technique'):
      decoder = infer_decoder.Decoder(lambda x: x, 'foo')

    with self.assertRaisesRegex(
        TypeError, 'Input d2 must be an numpy array, not'):
      decoder.compute_lda_model(np.ones((42, 42)), 42)

  def test_create_decoders(self):
    with self.assertRaisesRegex(
        ValueError, 'Couldn\'t determine model type for'):
      infer_decoder.create_decoder('foo')

    # TODO: Switch to contextlib.redirect_stdout when we move
    # completely to PY3.
    mock_stdout = io.StringIO()
    with mock.patch.object(sys, 'stdout', mock_stdout):
      infer_decoder.create_decoder('This will be a cca decoder')
    self.assertEqual(mock_stdout.getvalue().strip(),
                     u'Creating a CCA decoding model....')

    mock_stdout = io.StringIO()
    with mock.patch.object(sys, 'stdout', mock_stdout):
      infer_decoder.create_decoder('linear')
    self.assertEqual(mock_stdout.getvalue().strip(),
                     u'Creating a linear decoding model....')


if __name__ == '__main__':
  absltest.main()
