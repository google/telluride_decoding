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
import json
import math
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
import numpy.matlib
import scipy

from telluride_decoding import brain_model
from telluride_decoding import cca
from telluride_decoding.brain_data import TestBrainData

import tensorflow as tf

flags.DEFINE_string('telluride_test', 'just for testing',
                    'Just a dummy flag so we can test model saving.')


def evaluate_regressor_from_dataset(a, b, dataset,
                                    testing_data_file=None):
  """Evaluate a pretrained regressor using data from a dataset.

  Using a linear regressor computed by create_regressor_from_dataset, evaluate
  its performance.

  Args:
    a: A matrix in Ax + b = y
    b: b matrix in Ax + b = y
    dataset: TF dataset to be used to evaluate quality
    testing_data_file: For debugging: a file where the input data can be dumped

  Returns:
    average_error, average_power, pearson, num_samples

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset object must be a tf.data.Dataset object.')
  a = np.asarray(a)
  b = np.asarray(b)
  total_error = 0.0
  num_samples = 0
  e_x = 0.0
  e_y = 0.0
  e_xy = 0.0
  e_x2 = 0.0
  e_y2 = 0.0
  if testing_data_file:
    fp_x = open(testing_data_file+'_x.txt', 'w')
    fp_y = open(testing_data_file+'_y.txt', 'w')
  else:
    fp_x = None
    fp_y = None
  for (x_dict, y) in dataset:
    x = x_dict['input_1']
    if fp_x:
      for i in range(x.shape[0]):
        fp_x.write(' '.join([str(f) for f in x[i, :].tolist()]) + '\n')
    if fp_y:
      for i in range(y.shape[0]):
        fp_y.write(' '.join([str(f) for f in y[i, :].tolist()]) + '\n')
    y_est = np.matmul(x, a) + b
    err = y - y_est
    total_error += np.sum(err*err, axis=None)
    num_samples += err.shape[0]
    e_x += np.sum(y_est)
    e_x2 += np.sum(np.square(y_est))
    e_y += np.sum(y)
    e_y2 += np.sum(np.square(y))
    e_xy += np.sum(y * y_est)
  if fp_x:
    logging.info('Wrote testing data to file: %s and %s',
                 testing_data_file+'_x.txt', testing_data_file+'_y.txt')
    fp_x.close()
  if fp_y:
    fp_y.close()
  # From: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  #         #For_a_population
  pearson = (((e_xy/num_samples) - ((e_x/num_samples)*(e_y/num_samples))) /
             np.sqrt(e_x2/num_samples - (e_x/num_samples)**2) /
             np.sqrt(e_y2/num_samples - (e_y/num_samples)**2))
  average_error = total_error/num_samples
  average_power = e_y2/num_samples
  return average_error, average_power, pearson, num_samples


class BrainModelTest(absltest.TestCase):

  def setUp(self):
    super(BrainModelTest, self).setUp()
    self._test_data_dir = os.path.join(flags.FLAGS.test_srcdir, 'test_data')

  def clear_model(self):
    model_dir = '/tmp/tf'
    try:
      # Clear out the model directory, if it exists, because the TF save model
      # code wants an empty directory.
      tf.io.gfile.rmtree(model_dir)
    except tf.errors.NotFoundError:
      pass

  ################## Simply scaled data ################################
  # Simple dataset where the output is a simple matrix rotation of the input.
  def create_linear_algebra_dataset(self, random_noise_level=0.0):
    # First test the basic linear regressor parameters using fixed weight and
    # bias vectors.
    desired_sample_count = 10000
    w_true = [[1, 3], [2, 4]]
    b_true = [[5, 6]]
    x_train = np.random.rand(desired_sample_count, 2).astype(np.float32)
    y_train = np.matmul(x_train, np.array(w_true)) + np.array(b_true)
    y_train = y_train.astype(np.float32)
    if random_noise_level > 0.0:
      x_train += random_noise_level*np.random.standard_normal(x_train.shape)
    linear_dataset = tf.data.Dataset.from_tensor_slices(
        ({'input_1': x_train, 'input_2': x_train[:, :1]}, y_train))
    return linear_dataset.batch(100).repeat(count=1), w_true, b_true

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
    self.assertNotEmpty(input_data.shape)
    return np.sin(input_data[:, 0:1] * 2 * np.pi)

  def create_simply_scaled_dataset(self, dataset, data_offset=0,
                                   num_input_channels=2, mode='program_test'):
    """A dataset where the output is a simple scalar function of the input."""
    self.assertIsInstance(dataset, TestBrainData)

    num_samples = 10000
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

  def test_linear_regressor_training(self):
    linear_dataset, w_true, b_true = self.create_linear_algebra_dataset()
    (w_estimate, b_estimate, _, _,
     _) = brain_model.calculate_linear_regressor_parameters_from_dataset(
         linear_dataset, lamb=0.0)

    logging.info('W: %s vs %s', w_true, w_estimate)
    logging.info('b: %s vs %s', b_true, b_estimate)

    np.testing.assert_allclose(w_true, w_estimate, atol=1e-04)
    np.testing.assert_allclose(b_true, b_estimate, atol=1e-04)

  def test_linear_regressor_regularization(self):
    """Just a test to explore performance as a functiomn of regularization.
    """
    reg_values = list(np.power(10, np.arange(-14.0, 4, 2)))
    scale_values = [1e-3, 1, 1e3]
    results = np.zeros((len(reg_values), len(scale_values)), np.float32)
    linear_dataset, w_true, _ = self.create_linear_algebra_dataset(0.25)
    for i, reg in enumerate(reg_values):
      for j, scale in enumerate(scale_values):

        scaled_dataset = linear_dataset.map(
            lambda x, y: ({'input_1': scale*x['input_1']}, scale*y))
        (w_estimate, _, _, _,
         _) = brain_model.calculate_linear_regressor_parameters_from_dataset(
             scaled_dataset, lamb=reg)
        mse = np.mean((w_true - w_estimate)**2)
        results[i, j] = mse

    logging.info('test_linear_regressor_training lambdas: %s', reg_values)
    logging.info('test_linear_regressor_training scales: %s', scale_values)
    logging.info('test_linear_regressor_training MSE:\n%s', results)

  @flagsaver.flagsaver
  def test_linear_regression_model(self):
    """Test simulated eeg regression.
    """
    logging.info('\n\n**********test_simulated_linear_regression starting... '
                 '*******')
    self.clear_model()
    test_dataset, _, _ = self.create_linear_algebra_dataset()

    logging.info('Creating the model....')
    bmlr = brain_model.BrainModelLinearRegression(test_dataset)
    logging.info('Training the model....')
    bmlr.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                 loss=['mse'],
                 metrics=[brain_model.pearson_correlation_first]
                )
    bmlr.fit(test_dataset)
    logging.info('Estimated W is: %s', bmlr.w_estimate)
    logging.info('Estimated b is: %s', bmlr.b_estimate)
    logging.info('Evaluating the model...')
    metrics = bmlr.evaluate(test_dataset)

    logging.info('test_simulated_linear_regression errors: %s', metrics)
    logging.info('test_linear_regression_model returned these metrics: %s',
                 metrics)
    r = metrics['pearson_correlation_first']
    self.assertGreater(r, 0.99)      # Attended reconstruction

    # Now test the model predictions.

    for x, y in test_dataset.take(1):
      predictions = bmlr.predict(x)
      np.testing.assert_allclose(y, predictions, rtol=1e-5)

  @flagsaver.flagsaver
  def test_linear_regression_model_two(self):
    """Test simulated eeg regression.  Do it twice to look for variable error.
    """
    logging.info('\n\n******* test_simulated_linear_regression_two starting... '
                 '*******')
    self.clear_model()
    test_dataset, _, _ = self.create_linear_algebra_dataset()

    logging.info('Creating the model....')
    bmlr = brain_model.BrainModelLinearRegression(test_dataset)
    logging.info('Training the model....')
    bmlr.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                 loss=['mse'],
                 metrics=[brain_model.pearson_correlation_first]
                )
    bmlr.fit(test_dataset)
    logging.info('Estimated W is: %s', bmlr.w_estimate)
    logging.info('Estimated b is: %s', bmlr.b_estimate)
    logging.info('Evaluating the model...')
    metrics = bmlr.evaluate(test_dataset)

    logging.info('test_simulated_linear_regression errors: %s', metrics)
    logging.info('test_linear_regression_model returned these metrics: %s',
                 metrics)
    r = metrics['pearson_correlation_first']
    self.assertGreater(r, 0.99)      # Attended reconstruction

    # Now test the model predictions.

    for x, y in test_dataset.take(1):
      predictions = bmlr.predict(x)
      np.testing.assert_allclose(y, predictions, rtol=1e-5)

    # Do it again!
    logging.info('Creating the model....')
    bmlr = brain_model.BrainModelLinearRegression(test_dataset)
    logging.info('Training the model....')
    bmlr.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                 loss=['mse'],
                 metrics=[brain_model.pearson_correlation_first]
                )
    bmlr.fit(test_dataset)
    logging.info('Estimated W is: %s', bmlr.w_estimate)
    logging.info('Estimated b is: %s', bmlr.b_estimate)
    logging.info('Evaluating the model...')
    metrics = bmlr.evaluate(test_dataset)

    logging.info('test_simulated_linear_regression errors: %s', metrics)
    logging.info('test_linear_regression_model returned these metrics: %s',
                 metrics)
    r = metrics['pearson_correlation_first']
    self.assertGreater(r, 0.99)      # Attended reconstruction

    # Now test the model predictions.

    for x, y in test_dataset.take(1):
      predictions = bmlr.predict(x)
      np.testing.assert_allclose(y, predictions, rtol=1e-5)

  @flagsaver.flagsaver
  def test_regression_tf(self):
    """Test simple (no temporal shift) regression."""
    logging.info('\n\n**********test_regression_tf starting... *******')
    self.clear_model()
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate)
    test_dataset = self.create_simply_scaled_dataset(test_brain_data,
                                                     mode='train')

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=100)
    logging.info('Evaluating the model....')
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_regression_tf metrics: %s', metrics)
    self.assertLess(metrics['loss'], 0.3)
    self.assertGreater(metrics['pearson_correlation_first'], 0.80)

  @flagsaver.flagsaver
  def test_regression_fullyconnected(self):
    """Test simple (no temporal shift) regression."""
    logging.info('\n\n**********test_regression_fullyconnected starting... '
                 '*******')
    self.clear_model()
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate)
    test_dataset = self.create_simply_scaled_dataset(test_brain_data)

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=100)
    logging.info('Evaluating the model....')
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_regression_fullyconnected metrics: %s', metrics)
    self.assertLess(metrics['loss'], 0.35)
    self.assertGreater(metrics['pearson_correlation_first'], 0.80)

  @flagsaver.flagsaver
  def test_offset_regression_positive(self):
    logging.info('\n\n**********test_offset_regression_positive starting... '
                 '*******')
    self.clear_model()
    pre_context = 1        # Limit the number of variables to permit convergence
    post_context = 1
    logging.info('test_offset_regression_positive contexts: %d and %d',
                 pre_context, post_context)
    batch_size_request = 128
    data_offset = 1
    num_channels = 1
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    final_batch_size=batch_size_request,
                                    pre_context=pre_context,
                                    post_context=post_context)
    test_dataset = self.create_simply_scaled_dataset(
        test_brain_data, data_offset=data_offset,
        num_input_channels=num_channels)

    # Check the size of the dataset outputs.
    for next_element in test_dataset.take(1):
      (input_data, output) = next_element
    logging.info('test_offset_regression_positive dataset is: %s', input_data)
    input_data = input_data['input_1'].numpy()
    output = output.numpy()
    self.assertEqual(input_data.shape[0], batch_size_request)
    self.assertEqual(input_data.shape[1],
                     num_channels*(pre_context + 1 + post_context))
    self.assertEqual(output.shape[0], batch_size_request)
    self.assertEqual(output.shape[1], 1)

    # Check to see if the correct answer (as computed from the input data
    # is in the output at the right spot.
    logging.info('Input: %s', input_data[0:6, :])
    logging.info('Output: %s', output[0:6, :])
    index = num_channels*pre_context + data_offset
    expected_output = self.simply_scaled_transform(
        input_data[:, index:index + 1])
    logging.info('Expected Output: %s', expected_output[0:6, :])

    difference = output != expected_output
    self.assertEqual(difference.shape[1], 1)
    # Some frames are bad because they are at the beginning or end of the batch.
    # We look for 0.0 since the frames are shuffled, and we have no other way
    # of finding them.
    good_frames = np.nonzero(expected_output[:, 0] != 0.0)

    np.testing.assert_equal(output[good_frames, 0],
                            expected_output[good_frames, 0])

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=100)
    logging.info('Evaluating the model....')
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_offset_regression_positive metrics: %s', metrics)
    self.assertLess(metrics['loss'], 0.4)
    self.assertGreater(metrics['pearson_correlation_first'], 0.9)

  @flagsaver.flagsaver
  def test_offset_regression_negative(self):
    logging.info('\n\n**********test_offset_regression_positive starting... '
                 '*******')
    self.clear_model()
    pre_context = 1        # Limit the number of variables to permit convergence
    post_context = 1
    logging.info('test_offset_regression_negative contexts: %s and %s',
                 pre_context, post_context)
    batch_size_request = 128
    data_offset = -1
    num_channels = 1
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    final_batch_size=batch_size_request,
                                    pre_context=pre_context,
                                    post_context=post_context)
    test_dataset = self.create_simply_scaled_dataset(
        test_brain_data, data_offset=data_offset,
        num_input_channels=num_channels)
    # test_dataset = test_dataset.map(lambda x, y: ({'input_1': x['input_1']},
    #                                               y))

    # Check the size of the dataset inputs and outputs.
    for next_element in test_dataset.take(1):
      (input_data, output) = next_element
    input_data = input_data['input_1'].numpy()
    output = output.numpy()
    self.assertEqual(input_data.shape[0], batch_size_request)
    self.assertEqual(input_data.shape[1],
                     num_channels*(pre_context + 1 + post_context))
    self.assertEqual(output.shape[0], batch_size_request)
    self.assertEqual(output.shape[1], 1)

    # Check to see if the correct answer (as computed from the input data
    # is in the output at the right spot.
    logging.info('Input: %s', input_data[0:6, :])
    logging.info('Output: %s', output[0:6, :])
    index = num_channels*pre_context + data_offset
    expected_output = self.simply_scaled_transform(
        input_data[:, index:index + 1])
    logging.info('Expected Output: %s', expected_output[0:6, :])

    difference = output != expected_output
    self.assertEqual(difference.shape[1], 1)
    # Some frames are bad because they are at the beginning or end of the batch.
    # We look for 0.0 since the frames are shuffled, and we have no other way
    # of finding them.
    good_frames = np.nonzero(expected_output[:, 0] != 0.0)

    np.testing.assert_equal(output[good_frames, 0],
                            expected_output[good_frames, 0])

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset,
                                      num_hidden_list=hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=100)
    logging.info('Evaluating the %s model....', bmdnn)
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_offset_regression_negative metrics: %s', metrics)
    self.assertLess(metrics['loss'], 0.4)
    self.assertGreater(metrics['pearson_correlation_first'], 0.88)

  ########################## Simple IIR Dataset ############################
  def load_simple_iir_dataset(self, dataset, num_input_channels=1,
                              mode='program_test'):
    """Given a pre-created generic dataset, load it with IIR data."""
    num_samples = 10000
    input_data = np.random.randn(num_samples + 1,
                                 num_input_channels).astype(np.float32)
    output_data = 0.4 * input_data[0:-1,] + 0.6 * input_data[1:, :]
    dataset.preserve_test_data(input_data[1:num_samples + 1, :], output_data,
                               None)
    return dataset.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_simple_iir_regression32(self):
    """Test simple (two time impulse response) regression."""
    logging.info('\n\n**********test_simple_iir_regression32 starting... '
                 '*******')
    self.clear_model()
    batch_size_request = 128
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    pre_context=32,
                                    post_context=0,
                                    final_batch_size=batch_size_request)
    test_dataset = self.load_simple_iir_dataset(test_brain_data,
                                                num_input_channels=1)

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=10)
    logging.info('Evaluating the model....')
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_simple_iir_regression32 metrics: %s', metrics)
    self.assertLess(metrics['loss'], 0.025)
    self.assertGreater(metrics['pearson_correlation_first'], 0.95)

  @flagsaver.flagsaver
  def test_simple_iir_regression0(self):
    """Test simple (two time impulse response) regression.

    Shouldn't do as well as the regression32 above, as there is not enough
    context.
    """
    logging.info('\n\n**********test_simple_iir_regression0 starting... '
                 '*******')
    self.clear_model()
    batch_size_request = 128
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    pre_context=0,
                                    post_context=0,
                                    final_batch_size=batch_size_request)
    test_dataset = self.load_simple_iir_dataset(test_brain_data,
                                                num_input_channels=1)

    hidden_units = [40, 20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units)
    logging.info('Training the model....')
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    bmdnn.fit(test_dataset, epochs=10)
    logging.info('Evaluating the model....')
    metrics = bmdnn.evaluate(test_dataset)
    logging.info('test_simple_irr_regression0 metrics: %s', metrics)
    self.assertGreater(metrics['loss'], 0.025)  # Bigger than before
    self.assertGreater(metrics['pearson_correlation_first'], 0.80)
    return

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
    logging.info('SimulatedData:initialize_dataset: Setting unattended gain '
                 'to %s', simulated_unattended_gain)
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
    logging.info('SimulatedData: audio signals shape: %s',
                 self.audio_signals.shape)

  def create_simulated_eeg_data(self, brain_data, num_input_channels,
                                mode, simulated_noise_level=0.3,
                                num_output_channels=2):
    """Create one trial's worth of data.

    We create random EEG impulse responses and a random "speech" signal.
    We convolve the EEG impulse responses with the speech data to get a
    simulated EEG signal..

    Args:
      brain_data: The BrainData dataset for which to create data
      num_input_channels: How many input channels to generate
      mode: Mostly ignored unless you ask for "demo" when then generates a
        signal which oscillates between the two speakers.
      simulated_noise_level: How much noise to add for simulation.
      num_output_channels: How many output channels to generate. For 1 just the
        attended speech signal.  For 2, both the attended and unattended speech
        signals.

    Returns:
      A TF dataset with the input and output data.
    """
    self.assertIsInstance(brain_data, TestBrainData)
    self.assertGreater(num_input_channels, 0)
    self.signal_length = 100  # Seconds
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
    logging.info('Creating an example of the SimulatedData....')
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
    logging.info('Attended shapes: %s and %s', self.attended_audio.shape,
                 self.unattended_audio.shape)
    if num_output_channels == 1:
      output_channels = np.reshape(self.attended_audio, (-1, 1))
    else:
      output_channels = np.concatenate((np.reshape(self.attended_audio,
                                                   (-1, 1)),
                                        np.reshape(self.unattended_audio,
                                                   (-1, 1))),
                                       axis=1)
    logging.info('Output_channels shape: %s', output_channels.shape)
    brain_data.preserve_test_data(response[0:self.attended_audio.shape[0],
                                           0:num_input_channels],
                                  output_channels, None)
    return brain_data.create_dataset(mode=mode)

  @flagsaver.flagsaver
  def test_simulated_linear_regression(self):
    """Test simulated eeg regression.

    This test starts with only attended, so we test the reconstruction accuracy
    of only this signal.
    """
    logging.info('\n\n**********test_simulated_linear_regression starting... '
                 '*******')
    self.clear_model()
    batch_size_request = 128
    num_input_channels = 32
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    pre_context=32, post_context=0,
                                    final_batch_size=batch_size_request)
    self.create_simulated_eeg_data(test_brain_data, num_input_channels,
                                   mode='train', num_output_channels=1)
    test_dataset = test_brain_data.create_dataset('train')
    model = brain_model.BrainModelLinearRegression(test_dataset)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    model.fit(test_dataset)
    metrics = model.evaluate(test_dataset)
    logging.info('Test simulated linear regression metrics are: %s', metrics)
    logging.info('Test_simulated_linear_regression produced a MSE of %s',
                 metrics['loss'])
    # Check attended reconstruction
    self.assertGreater(metrics['pearson_correlation_first'], 0.99)

    # Test inference with this model. Note, this depends on the dataset defined
    # by this library to add the proper context to the data.
    test_dataset = test_brain_data.create_dataset('program_test')
    for next_element in test_dataset.take(1):
      (input_dict, output) = next_element
    for k in input_dict:
      logging.info('  %s: %s', k, input_dict[k].shape)
    logging.info('Output size is: %s', output.shape)
    predictions = model.predict(tf.data.Dataset.from_tensors(input_dict))

    edge_count = 10
    signal_power = np.sum(output[edge_count:-edge_count]**2)
    error = output - predictions
    error_power = np.sum(error[edge_count:-edge_count]**2)
    snr = 10*np.log10(signal_power/error_power)
    logging.info('Inference SNR is %s', snr)
    self.assertGreater(snr, 15.0)

  @flagsaver.flagsaver
  def test_simulated_dnn_regression(self):
    """Test simulated eeg regression.

    This test starts with both attended and unattended audio, so we test the
    reconstruction accuracy of both signals.
    """
    logging.info('\n\n**********test_simulated_dnn_regression starting... '
                 '*******')
    self.clear_model()
    batch_size_request = 128
    num_input_channels = 32
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    pre_context=32,
                                    post_context=0,
                                    final_batch_size=batch_size_request)
    test_dataset = self.create_simulated_eeg_data(test_brain_data,
                                                  num_input_channels,
                                                  mode='train')
    model = brain_model.BrainModelLinearRegression(test_dataset)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    model.fit(test_dataset)
    metrics = model.evaluate(test_dataset)
    logging.info('Test simulated linear regression metrics are: %s', metrics)
    logging.info('Test_simulated_linear_regression produced a MSE of %s',
                 metrics['loss'])
    # Check attended reconstruction
    self.assertGreater(metrics['pearson_correlation_first'], 0.95)
    # TODO Can we check unattended reconstruction too?

  @flagsaver.flagsaver
  def test_dnn_classifier(self):
    num_samples = 1000
    num_dim = 3
    input1 = np.random.randn(num_samples, num_dim).astype(np.float32)
    output = (np.random.randn(num_samples, 1,) > 0.5).astype(np.float32)

    input2 = np.random.randn(num_samples, num_dim-1).astype(np.float32)
    input2 = output*2*input1[:, :-1] + (1-output)*input2

    self.clear_model()
    batch_size_request = 128
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    final_batch_size=batch_size_request)
    test_brain_data.preserve_test_data(input_data=input1,
                                       input2_data=input2,
                                       output_data=output)
    test_dataset = test_brain_data.create_dataset('train')
    print('test_dataset is:', test_dataset)
    ib, ob = list(test_dataset.take(1))[0]
    print('test_dataset input is:', ib)
    print('test_dataset output is:', ob)

    model = brain_model.BrainModelClassifier(test_dataset,
                                             num_hidden_list=[20])

    output = model(ib)
    print('Model output is:', output)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                 )
    model.fit(test_dataset, epochs=100)
    metrics = model.evaluate(test_dataset)
    print(metrics)
    self.assertGreater(metrics['accuracy'], 0.90)

  @flagsaver.flagsaver
  def test_linear_regression(self):
    # First test the basic linear regressor parameters using fixed weight and
    # bias vectors.
    desired_sample_count = 1000
    w_true = [[1, 3], [2, 4]]
    b_true = [[5, 6]]
    x_train = np.random.rand(desired_sample_count, 2).astype(np.float32)
    y_train = np.matmul(x_train, np.array(w_true)) + np.array(b_true)
    y_train = tf.cast(y_train, tf.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices(({'input_1': x_train},
                                                       y_train))
    test_dataset = test_dataset.batch(100).repeat(count=1)
    (w_estimate, b_estimate, _, _,
     _) = brain_model.calculate_linear_regressor_parameters_from_dataset(
         test_dataset, lamb=0.0, use_offset=True)
    logging.info('Estimated w: %s', w_estimate)
    logging.info('Estimated b: %s', b_estimate)
    self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(b_true, b_estimate, atol=1e-05))

    (average_error, _, _,
     num_samples) = evaluate_regressor_from_dataset(
         w_estimate, b_estimate, test_dataset)
    self.assertLess(average_error, 1e-10)
    self.assertEqual(num_samples, desired_sample_count)

    # Now redo the test without the offset, and make sure the error is large.
    (w_estimate, b_estimate, _, _,
     _) = brain_model.calculate_linear_regressor_parameters_from_dataset(
         test_dataset, lamb=0.0, use_offset=False)
    logging.info('Estimated w: %s', w_estimate)
    logging.info('Estimated b: %s', b_estimate)
    # w will be different because the model no longer fits.
    # self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(0.0, b_estimate, atol=1e-05))

    (average_error, _, _,
     num_samples) = evaluate_regressor_from_dataset(w_estimate, b_estimate,
                                                    test_dataset)
    self.assertGreater(average_error, 0.5)

    # Create a TF model, initialized with the test dataset above, and make
    # sure it is properly initialized.
    model = brain_model.BrainModelLinearRegression(test_dataset)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                  metrics=[brain_model.pearson_correlation_first]
                 )
    model.fit(test_dataset)

    w_estimate, b_estimate = model.weight_matrices
    self.assertTrue(np.allclose(w_true, w_estimate, atol=1e-05))
    self.assertTrue(np.allclose(b_true, b_estimate, atol=1e-05))

    # Now test with a new evaluation dataset
    x_eval = np.random.rand(1000, 2).astype(np.float32)
    y_eval = np.matmul(x_eval, np.array(w_true)) + np.array(b_true)
    eval_dataset = tf.data.Dataset.from_tensors(({'input_1': x_eval}, y_eval))
    eval_metrics = model.evaluate(eval_dataset)
    self.assertLess(eval_metrics['loss'], 1e-5)

  def test_cca_vector(self):
    """Make sure we get a vector of outputs."""
    num_points = 1000000
    data1 = np.random.rand(num_points, 3)
    data2 = (np.random.rand(num_points, 3) + data1)/2.0
    data2[:, -1] = np.random.rand(num_points)
    a = tf.constant(data1)
    b = tf.constant(data2)

    # bm = brain_model.BrainModel()
    results = np.square(brain_model.pearson_correlation(a, b).numpy())
    tf.debugging.assert_near(results[0:2], 0.5, atol=0.01)
    tf.debugging.assert_near(results[2:], 0.0, atol=0.01)

  @flagsaver.flagsaver
  def test_save_model(self):
    """Test simple (small FIR with temporal shift) regression."""
    self.clear_model()
    pre_context = 16
    post_context = 0
    repeat_count_request = 1
    batch_size_request = 128
    frame_rate = 100.0
    saved_model_dir = '/tmp/tf_saved_model.tf'

    # Create the dataset and train the model.
    num_input_channels = 1
    test_brain_data = TestBrainData('input', 'output', frame_rate,
                                    pre_context=pre_context,
                                    post_context=post_context,
                                    repeat_count=repeat_count_request,
                                    final_batch_size=batch_size_request)
    self.load_simple_iir_dataset(test_brain_data,
                                 num_input_channels=num_input_channels)
    test_dataset = test_brain_data.create_dataset()
    bmlr = brain_model.BrainModelLinearRegression(test_dataset)
    bmlr.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                 loss=['mse'],
                 metrics=[brain_model.pearson_correlation_first]
                )
    bmlr.fit(test_dataset)

    metrics = bmlr.evaluate(test_dataset)
    logging.info('test_save_model evaluate metrics are: %s', metrics)
    self.assertLess(metrics['loss'], 0.001)
    self.assertGreater(metrics['pearson_correlation_first'], .99)

    # Save some predictions for later.
    for inputs, _ in test_dataset.take(1):
      save_x = inputs
      predict_y = bmlr.predict(save_x)

    # Save the model on disk
    bmlr.add_metadata(flags.FLAGS.flag_values_dict(), test_dataset)
    bmlr.save(saved_model_dir, save_format='tf')

    # Now restore the model.
    # Done this way because of (and fixed in TF2.2?)
    with tf.keras.utils.CustomObjectScope(
        {'pearson_correlation_first': brain_model.pearson_correlation_first,
         'BrainCcaLayer': cca.BrainCcaLayer,
        }):
      new_model = tf.keras.models.load_model(saved_model_dir)
    new_predictions = new_model.predict(save_x)
    np.testing.assert_allclose(predict_y, new_predictions, rtol=1e-6, atol=1e-6)

    metadata = json.loads(new_model.telluride_metadata.numpy())
    logging.info('New_model.telluride_metadata: %s', metadata)
    self.assertIsInstance(metadata, dict)
    self.assertEqual(metadata['telluride_test'], 'just for testing')

    inputs = json.loads(new_model.telluride_inputs.numpy())
    self.assertEqual(inputs['input_1'], list(save_x['input_1'].shape))
    self.assertEqual(inputs['input_2'], list(save_x['input_2'].shape))

    output = json.loads(new_model.telluride_output.numpy())
    self.assertEqual(output, list(predict_y.shape))

  def find_event_files(self, search_dir, pattern='events.out.tfevents'):
    """Finds event files in a directory tree.

    Args:
      search_dir: The root of the tree to search
      pattern: A string at the start of the desired file names.

    Returns:
      A list of all files in the directory tree that start with the pattern
      string.
    """
    all_files = []
    for (root, _, files) in os.walk(search_dir):
      all_files.extend([os.path.join(root, f) for f in files
                        if f.startswith(pattern)])
    return all_files

  @flagsaver.flagsaver
  def test_tensorboard(self):
    """Make sure we can put data out for Tensorboard to read."""
    logging.info('\n\n**********test_tensorboard starting... *******')
    self.clear_model()
    frame_rate = 100.0
    test_brain_data = TestBrainData('input', 'output', frame_rate)
    test_dataset = self.create_simply_scaled_dataset(test_brain_data,
                                                     mode='train')

    tensorboard_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                                   'tensorboard')
    logging.info('Writing tensorboard data to %s', tensorboard_dir)

    tf.io.gfile.makedirs(tensorboard_dir)
    hidden_units = [20, 10]
    bmdnn = brain_model.BrainModelDNN(test_dataset, hidden_units,
                                      tensorboard_dir=tensorboard_dir)
    bmdnn.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=['mse'],
                 )
    bmdnn.fit(test_dataset, epochs=10)
    event_files = self.find_event_files(bmdnn.tensorboard_dir)
    logging.info('Tensorboard train directory contains: %s', event_files)
    self.assertNotEmpty(event_files)

    bmdnn.evaluate(test_dataset)
    summary_path = os.path.join(bmdnn.tensorboard_dir, 'results')
    self.assertTrue(tf.io.gfile.exists(summary_path))

    bmdnn.add_tensorboard_summary('foo', 42, 'special')
    summary_path = os.path.join(bmdnn.tensorboard_dir, 'special')
    self.assertTrue(tf.io.gfile.exists(summary_path))


class PearsonTest(absltest.TestCase):
  # Example data from: http://www.statisticshowto.com/
  #       probability-and-statistics/correlation-coefficient-formula/
  correlation_data = np.array([
      [1, 43, 99],
      [2, 21, 65],
      [3, 25, 79],
      [4, 42, 75],
      [5, 57, 87],
      [6, 59, 81]], dtype=np.float32)

  def test_pearson(self):
    """Test the Pearson correlation graph using sample data above.

    Use simulated data and make sure we get the result we expect.
    """
    test_x = PearsonTest.correlation_data[:, 1:2]  # Make sure it's a 2d array
    test_y = PearsonTest.correlation_data[:, 2:3]
    correlation_tensor = brain_model.pearson_correlation(test_x, test_y)

    correlation_r = correlation_tensor.numpy()
    logging.info('Test correlation produced: %s', correlation_r)
    self.assertAlmostEqual(correlation_r, 0.5298, delta=0.0001)

  def test_pearson2(self):
    """Test to make sure TF and np.corrcoef give same answers with random data.

    This test makes sure it works with multi-dimensional features.
    """
    fsize = 2  # feature size
    dsize = 6  # sample count
    x = np.random.random((dsize, fsize))
    y = np.random.random((dsize, fsize))

    cor = brain_model.pearson_correlation(x, y)
    npcor = np.diag(np.corrcoef(x, y, rowvar=False)[:fsize, fsize:])
    logging.info('test_pearson2: cor.eval(): %s', cor.numpy())
    logging.info('test_pearson2: np.corrcoef: %s', npcor)
    logging.info('test_pearson2: Difference: %s', cor.numpy() - npcor)
    self.assertTrue(np.allclose(npcor, cor.numpy(), atol=2e-7))

  def test_pearson_loss(self):
    pcl = brain_model.PearsonCorrelationLoss()
    test_x = PearsonTest.correlation_data[:, 1:2]  # Need rank 2 data
    test_y = PearsonTest.correlation_data[:, 2:3]
    correlation_r = pcl.call(tf.convert_to_tensor(test_x),
                             tf.convert_to_tensor(test_y))
    self.assertAlmostEqual(np.sum(correlation_r.numpy()), -0.5298, delta=0.0001)


if __name__ == '__main__':
  absltest.main()
