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

"""Code to setup and test inference with an EEG decoding model.

This file implements a class (Decoder) that handles the post processing needed
when building an auditory attention decoding (AAD) system.  It implements a
number of classes that make it easy to load and test different decoding models.

The basic use is create the necessary decoder class, pass it a dataset for
training the decoder-specific variables (like the output mean), and then
call one of the infer classes to actually compute a *scalar* representing the
"probability" of the attended speaker.

The sequence of events (data) is as follows:
  Input data (EEG, etc) in the form of a minibatch stored in a input_dict
  -> the decode_one() method which calls the class-specific
    (self._decoding_model) to do the real decoding work. This function converts
    model-specific data into two vectors that are (should) be correlated when
    attention is directed that way. In the linear case it is two scalar vectors
    (real and estimated intensity).  In the CCA case it is two n-dimensional
    vectors that should be correlated. This data is sampled at the EEG/audio
    frame rate.
  -> The infer_one() method converts the pair of (hopefully) correlated vector
    signals into a scalar. It does this by calculating the frame-by-frame
    correlation of each signal dimension, and then summing (integrating) over
    the mini batch to get a multi-dimensional correlation signal.

    This is then reduced to a scalar by one of a number of methods. It does this
    by computing the (multi-dimensional) correlation over a mini batch.  And
    then reducing it to a scalar that indicates whether the data corresponds to
    attention.  The reduction can be done in a number of ways such as first-
    dimension, mean, mean-signed-squared, and LDA.

Both correlation and LDA need to be trained.  Correlation depends on the mean
of the signal, and this is approximated by looking at a large section of the
data. We are assuming that the global mean is better here than the local
(mini batch) mean since the signal is time varying.  In the training phase we
estimate the means and variance of each dimension of the signals.

Finally, when converting from multi-dimensional correlation signals the best
way is to use LDA which finds the optimum projection. This also involves a
training step, along with the basic correlation statistics, is calculated by
the train method.
"""

import collections
import json
import os
from absl import logging

import numpy as np

from telluride_decoding import brain_data
from telluride_decoding import result_store
from telluride_decoding import scaled_lda

import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


class NumpyEncoder(json.JSONEncoder):
  """Encoder for allowing serialization of numpy arrays."""

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      comp_list = obj.tolist()
      if np.iscomplexobj(obj):
        comp_list_re = np.real(comp_list).tolist()
        comp_list_im = np.imag(comp_list).tolist()
        comp_list = [comp_list_re, comp_list_im]
      return comp_list
    return json.JSONEncoder.default(self, obj)

CorrelationParamsTuple = collections.namedtuple('CorrelationParamsTuple', [
    'count', 'sum_x', 'sum_y', 'sum_x2', 'sum_y2', 'mean_x', 'mean_y', 'power'
])
ModelParamsTuple = collections.namedtuple('ModelParamsTuple',
                                          ['correlation_params', 'lda_params'])


class Decoder(object):
  """A generic class for decoding speech from EEG (and related problems.)

  This class defines an abstract interface for accessing brain-decoding models,
  training them, and doing inference.  It is agnostic about the type of the
  model.

  Note: This is the sequence of events in the pipeline:
  1) Run the decoder, getting ground truth and predictions.
  2) Compute the correlation, sample by sample, using the global mean stats.
  3) Run the multi-dimensional signal through LDA.

  In the training phase, we estimate the correlation means and powers so we
  can estimate a new correlation in an online fashion.  (Using the
  global/trained stats as an estimate of the true means and power.)  We also
  estimate the LDA model that converts the multi-dimensional data into a
  projection direction.

  Subclasses should define the train, infer_one (for real-time use) and perhaps
  infer methods.

  Attributes:
    decoding_model: The TF model that decodes the input data. Provided here for
      testing.  Usually this parameter will be None, and it will be loaded via
      the load_model method.
    decoding_model_params: A dictionary that describes the parameters used to
      build the model.
    model_inputs: A dictionary keyed by input_name with data sizes (as a list).
    model_output: A list indicating the output shape.
    model_params: A list of model parameters to be passed to real-time code.
    correlation_params: A dictionary of parameters for correlation computation.
    lda_params: A dictionary of parameters for LDA computation.
  """
  # TODO: Change all params using dictionary to use namedtuple

  def __init__(self, decoding_model=None, reduction='mean-squared'):
    if decoding_model is not None and not callable(decoding_model):
      raise TypeError('Must supply a callable model when initializing a '
                      'Decoder, not a %s.' % type(decoding_model))
    if reduction not in ('mean-squared', 'first', 'second', 'lda',
                         'all', 'mean'):
      raise ValueError('Unknown reduction technique: %s' % reduction)
    self._decoding_model = decoding_model
    self._decoding_model_params = {}
    self._model_inputs = {}
    self._model_output = []
    self._reduction = reduction
    self._lda = None
    self.reset_correlation_statistics()

  @property
  def decoding_model(self):
    return self._decoding_model

  @property
  def decoding_model_params(self):
    """The parameters of the models, in a named tuple."""
    return self._decoding_model_params

  @decoding_model_params.setter
  def decoding_model_params(self, values):
    """Sets the model parameters based on saved values.

    Args:
      values: A named tuple of values returned by the model_params property.
    """
    self._decoding_model_params = values

  @property
  def correlation_params(self):
    return CorrelationParamsTuple(self._count, self._sum_x, self._sum_y,
                                  self._sum_x2, self._sum_y2, self._mean_x,
                                  self._mean_y, self._power)

  def _set_correlation_params(self, values):
    values = CorrelationParamsTuple(*values)
    self._count = values.count
    self._sum_x = values.sum_x
    self._sum_y = values.sum_y
    self._sum_x2 = values.sum_x2
    self._sum_y2 = values.sum_y2
    self._mean_x = values.mean_x
    self._mean_y = values.mean_y
    self._power = values.power

  @property
  def lda_params(self):
    if self._lda is None:
      self._lda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    return self._lda.model_parameters

  def _set_lda_params(self, values):
    if self._lda is None:
      self._lda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    self._lda.model_parameters = values

  @property
  def model_params(self):
    """Returns the model parameters needed to implement this model.

    Generally specialized for each type of decoder.

    Returns:
      A dict of values based on _model_params
    """
    model_params = ModelParamsTuple(self.correlation_params, self.lda_params)
    return model_params

  @model_params.setter
  def model_params(self, values):
    """Sets the model parameters based on saved values.

    Args:
      values: A namedtuple of values returned by the model_params property.
    """
    self._set_parameters(values)

  def _set_parameters(self, values):
    self._set_correlation_params(values.correlation_params)
    self._set_lda_params(values.lda_params)

  @property
  def model_inputs(self):
    """A dictionary of input keys, with a size (list) for each input."""
    return self._model_inputs

  @property
  def model_output(self):
    """A list indicating the output shape."""
    return self._model_output

  def reset_correlation_statistics(self):
    self._count = 0
    self._sum_x = 0.0
    self._sum_y = 0.0
    self._sum_x2 = 0.0
    self._sum_y2 = 0.0
    self._mean_x = 0.0
    self._mean_y = 0.0
    self._power = 1.0

  def save_parameters(self, param_filename):
    params = self.model_params
    with tf.io.gfile.GFile(param_filename, 'w') as f:
      json.dump(params._asdict(), f, cls=NumpyEncoder)

  def restore_parameters(self, param_filename):
    with tf.io.gfile.GFile(param_filename, 'r') as f:
      loaded = json.load(f)
    self.model_params = ModelParamsTuple(**loaded)

  def load_decoding_model(self,
                          saved_model_file,
                          object_dict):
    """Loads a saved Keras mode, linking in our metrics and loss definitions.

    Use the object_dict to tell the new model which code implements the custom
    functions defined in the model.

    Args:
      saved_model_file: A model saved by the decoding software.
      object_dict: Mapping between function names in the model and the code that
        implements that here.

    Raises:
      TypeError: For incorrect input arguments.
    """
    if not saved_model_file or not isinstance(saved_model_file, str):
      raise TypeError('Must provide a file name (string) to load-model, not '
                      'a %s.' % type(saved_model_file))
    if object_dict and not isinstance(object_dict, dict):
      raise TypeError('If providing an object dictionary, it must be a dict '
                      'not a %s.' % type(object_dict))

    logging.info('Loading decoder model data from %s.', saved_model_file)
    with tf.keras.utils.CustomObjectScope(object_dict):
      self._decoding_model = tf.keras.models.load_model(saved_model_file)

    self._decoding_model_params = json.loads(
        self._decoding_model.telluride_metadata.numpy())
    self._model_inputs = json.loads(
        self._decoding_model.telluride_inputs.numpy())
    self._model_output = json.loads(
        self._decoding_model.telluride_output.numpy())

  def add_data_correlator(self, x, y):
    """Adds a batch of data to the correlation calculation.

    Updates the online correlation calculation with another batch of data.  Goes
    ahead and computes the summary statistcs after each batch in preparation for
    the compute_correlation computation.

    Args:
      x: The first input data, num_frames x num_dimensions.
      y: The second input data, num_frames x num_dimensions.
    """
    self._count += x.shape[0]
    self._sum_x += np.sum(x, axis=0)
    self._sum_y += np.sum(y, axis=0)
    self._sum_x2 += np.sum(x**2, axis=0)
    self._sum_y2 += np.sum(y**2, axis=0)

    # Update the means and power so they are ready for use.
    self._mean_x = self._sum_x / self._count
    self._mean_y = self._sum_y / self._count
    self._power = (np.sqrt((self._sum_x2 - self._sum_x**2/self._count) *
                           (self._sum_y2 - self._sum_y**2/self._count)) /
                   self._count)

  def compute_correlation(self, x, y):
    """Computes multidimensional correlation and scaling without the final sum.

    This just a normalized cross-product.  Add a sum over time (first dimension)
    to convert the actual correlation. The mean and power normalizations must
    be computed earlier (or loaded).

    Args:
      x: The first tensor (num_frames x num_features).
      y: The second tensor (num_frames x num_features).

    Returns:
      The normalized cross product (num_frames x num_features).
    """
    # From: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    return ((x - np.broadcast_to(self._mean_x, x.shape)) *
            (y - np.broadcast_to(self._mean_y, y.shape))/ self._power)

  def train(self, data0, data1, window_size=0):
    """Trains the classification aspect of the decoder (means and LDA).

    This routine processes all the data once (to get the decoded data, in
    whatever form it is computed by the model) to get the mean and average
    power. Then it processes the data again in order to calculate the
    correlation signals that become the input to the LDA calculation.

    Note: when using LDA data that fits class 0 will have a label of 0 while
    data1 will have an output label of 1.

    Args:
      data0: A tf.data.Dataset object used to train the decoder for class 0,
        generally this is the unattended, or mixed-up data.
      data1: A tf.data.Dataset object used to train the decoder for class 1,
        generally this is the attended, or normal data.
      window_size: What size window to use when averaging the correlation data
        before computing the LDA.

    Returns:
      d' from the LDA analysis.

    Raises:
      TypeError: for incorrect data types.
    """
    if not isinstance(data0, tf.data.Dataset):
      raise TypeError('Must feed training routine data0 with a tf.data.Dataset'
                      ' not a %s.' % type(data0))
    if not isinstance(data1, tf.data.Dataset):
      raise TypeError('Must feed training routine data1 with a tf.data.Dataset'
                      ' not a %s.' % type(data1))

    # Decode all the data so we can compute the processed means and variances.
    for input_dict, output in data0:
      result1, result2 = self.decode_one(input_dict, output)
      self.add_data_correlator(result1, result2)
    for input_dict, output in data1:
      result1, result2 = self.decode_one(input_dict, output)
      self.add_data_correlator(result1, result2)

    # Now process the data again, calculating the correlations that are used as
    # input to the LDA calculation.
    correlations_data_0 = result_store.NumpyStore()
    correlations_data_1 = result_store.NumpyStore()
    for input_dict, output in data0:
      result1, result2 = self.decode_one(input_dict, output)
      correlations = self.compute_correlation(result1, result2)
      correlations_data_0.add_data(correlations)
    for input_dict, output in data1:
      result1, result2 = self.decode_one(input_dict, output)
      correlations = self.compute_correlation(result1, result2)
      correlations_data_1.add_data(correlations)

    # Make sure the mismatched (class 0) and matched (class 1, attended) have
    # the same width and then compute the LDA parameters.
    all_data_0 = correlations_data_0.all_data
    all_data_1 = correlations_data_1.all_data
    if all_data_0 is None or all_data_0.shape[0] == 0:
      raise ValueError('No data for class 0')
    if all_data_1 is None or all_data_1.shape[0] == 0:
      raise ValueError('No data for class 1')
    logging.info('Found %d frames for class 0 and %d frames for class 1',
                 all_data_0.shape[0], all_data_1.shape[0])

    # The arguments are in this order because we want no speaker to have a
    # label of 0, while speaker present to be 1.
    return self.compute_lda_model(average_data(all_data_0, window_size),
                                  average_data(all_data_1, window_size))

  def decode_one(self, input_dict, ground_truth):
    """Makes one prediction from Tensors using the linear decoding_model.

    Args:
      input_dict: Input dictionary containing Tensors for input_1, and input_2.
      ground_truth: Tensor with the expected answer (if the user is attending
        to this speaker).
    """
    del input_dict
    del ground_truth
    raise NotImplementedError('Must be implemented by a subclass.')

  def infer_one(self, input_dict, output):
    """Infers the speaker given the current model for one minibatch of data.

    This is the entire inference pipeline for online or offline inference.

    Args:
      input_dict: The data (from a dataset derived from the BrainData class.)
        Dictionary containing what is needed by the decoder, input_1 and perhaps
        input_2.
      output: The output half of the dataset, ignored in inference)

    Returns:
      Generally a scalar signal indicating whether attention is directed towards
      this speaker.  0 for unattended, 1 for attended.  Size is num_frames x 1.

      Except when 'all' is requested for the reduction, then all the correlation
      dimensions are returned (for testing.)

      This only reduces across feature/dimensions. The time axis remains. The
      output is (except for 'all') num_frames x 1.
    """
    result1, result2 = self.decode_one(input_dict, output)
    correlations = self.compute_correlation(result1, result2)
    if self._reduction == 'first':
      return correlations[:, 0]
    if self._reduction == 'second':
      return correlations[:, 1]
    elif self._reduction == 'mean':
      return np.mean(correlations, axis=1)
    elif self._reduction == 'mean-squared':
      return np.mean(np.sign(correlations)*correlations**2, axis=1)
    elif self._reduction == 'lda':
      return self.reduce_with_lda(correlations)[:, 0]
    elif self._reduction == 'all':
      # Return all the correlation signals for testing.
      return correlations
    else:
      raise ValueError('Unknown reduction technique: %s.' % self._reduction)

  def test_all(self, exp_data):
    """Runs a model on both speakers on all the data and returns labels.

    For all the data in the exp_data dataset, infers the speaker from the input
    data (decoding, calculating any necessary correlations, performing the
    dimensionality reduction).  Save the results and the ground-truth speaker
    labels and returns two numpy arrays with all the results.

    Args:
      exp_data: tf.dataset with data for the attended speaker.

    Returns:
      Two numpy arrays:
      1) A 1d vector with the "likelihood" of being the desired speaker (based
        on how the dataset was set up.)
      2) A 1d vector with the ground truth speaker label.
    """
    predictions = result_store.NumpyStore(name='test_all predictions')
    labels = result_store.NumpyStore(name='test_all labels')
    for input_dict, output in exp_data:
      infer_results = self.infer_one(input_dict, output)
      predictions.add_data(infer_results)

      # Save the ground truth
      labels.add_data(input_dict['attended_speaker'])
    return predictions.all_data, labels.all_data

  def test_by_window(self, dataset, window_size):
    """Processes a dataset, returning chunked pairs of the correlated signals.

    Args:
      dataset: A tf.data.dataset object from which to pull eeg and audio data.
      window_size: Size of the analysis window, after decoding

    Yields:
      Two numpy arrays, each with window_size frames, after decoding.
    """
    storage = result_store.TwoResultStore(window_width=window_size,
                                          window_step=window_size//2)
    for input_dict, output in dataset:
      infer_results = self.infer_one(input_dict, output)
      storage.add_data(infer_results, input_dict['attended_speaker'])
      for r1, r2 in storage.next_window():
        yield r1, r2

  def compute_lda_model(self, d1, d2):
    """Computes linear-discriminant analysis (LDA) model to separate 2 datasets.

    This routine finds the best rotation via LDA to separate two classes of data
    along a line. The best model is stored internally, but can be used by the
    reduce_with_lda routine to transform multi-dimensional data into a scalar
    axis.

    Args:
      d1: A num_frames x num_dimensions set of data for class 0
      d2: A num_frames x num_dimensions set of data for class 1

    Returns:
      The estimated d', indicating the quality of the separation between the
      two datasets.
    """
    if not isinstance(d1, np.ndarray):
      raise TypeError('Input d1 must be an numpy array, not %s.' % type(d1))
    if not isinstance(d2, np.ndarray):
      raise TypeError('Input d2 must be an numpy array, not %s.' % type(d2))
    data = np.concatenate((d1, d2), axis=0)
    labels = np.concatenate((1*np.ones(d1.shape[0],),
                             2*np.ones(d2.shape[0],)))
    self._lda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    predictions = self._lda.fit_transform(data, labels)
    d_prime = calculate_dprime(predictions[labels == 1, 0],
                               predictions[labels == 2, 0])
    return d_prime

  def reduce_with_lda(self, d1):
    """Rotates the data so as to get the maximal separation in dimension 0.

    Args:
      d1: multi-dimensional ndarray with input data

    Returns:
      Rotated multi-dimensional matrix. num_frames x num_dims, where the
      number of dimensions is the minimum of the original input feature count
      and the number of labels provided to LDA.
    """
    if self._lda is None:
      raise ValueError('Must compute the LDA model before reducing data.')
    if not isinstance(d1, np.ndarray):
      raise TypeError('Input data must be an numpy array, not %s.' % type(d1))
    return self._lda.transform(d1)

  def check_model_and_data(self, actual_dataset):
    """Checks to see if a dataset is compatible with the current decoding model.

    Checks the datasets input and output sizes, and compares them to the model
    that we have loaded.

    Args:
      actual_dataset: A dataset produced by BrainData with appropriate context
        added.
    """
    if not self.model_inputs or not self.model_output:
      raise ValueError('Model has not been initialized yet. Use load_model '
                       'first')
    if not isinstance(actual_dataset, tf.data.Dataset):
      raise TypeError('Actual_dataset is not a dataset, but a %s.' %
                      (type(actual_dataset)))
    for actual_input_dict, actual_output in actual_dataset.take(1):
      for expected_key, expected_input_spec in self.model_inputs.items():
        if expected_key not in actual_input_dict:
          raise TypeError('Can\'t find needed key %s in input_data (%s)' %
                          (expected_key, actual_input_dict.keys()))
        if actual_input_dict[expected_key].shape[1] != expected_input_spec[1]:
          raise TypeError('Data for %s has the wrong shape, expected %s, got %s'
                          % (expected_key, expected_input_spec,
                             actual_input_dict[expected_key].shape))

      if actual_output.shape[1] != self.model_output[1]:
        raise TypeError('Output data has the wrong shape, expected %s, got %s'
                        % (self.model_output, actual_output.shape))


class LinearRegressionDecoder(Decoder):
  """A decoder that connects input to output via linear regression."""

  def decode_one(self, input_dict, ground_truth):
    """Makes one prediction from Tensors using the linear decoding_model.

    Args:
      input_dict: Input dictionary containing Tensors for input_1, and input_2.
      ground_truth: Tensor with the expected answer (if the user is attending
        to this speaker).

    Returns:
      Correlations for this window of data for speaker 1 and speaker 2.
    """
    # TODO Remove this if really not necessary.
    if 'XXattended_speaker' in input_dict:
      input_dict = input_dict.copy()
      del input_dict['attended_speaker']
    predictions = self._decoding_model(input_dict)
    return ground_truth.numpy(), predictions.numpy()


class CCADecoder(Decoder):
  """A decoder using CCA to rotate two sets of data to be maximally correlated.
  """

  def decode_one(self, input_dict, ground_truth):
    """Makes one prediction using the CCA decoding_model.

    Args:
      input_dict: Input dictionary containing Tensors for input_1, and input_2.
      ground_truth: Ignored because we just want to maximize the
        cross-correlation between the two input tensors.

    Returns:
      Correlations for this window of data for speaker 1 and speaker 2.
    """
    del ground_truth
    # TODO Remove this if really not necessary.
    if 'XXattended_speaker' in input_dict:
      input_dict = input_dict.copy()
      del input_dict['attended_speaker']
    predictions = self._decoding_model(input_dict)
    num_cca_dims = predictions.shape[1]//2
    return (predictions[:, :num_cca_dims].numpy(),
            predictions[:, num_cca_dims:].numpy())


def create_decoder(model_tag, reduction='lda', model=None):
  """Creates the appropriate type of Decoder model given a model_tag.

  Hack alert: This looks for an indication of the model type in the directory
  name and creates the right kind of Decoder model. A better approach, someday,
  will be able to infer this from the contents of the model directory.

  Args:
    model_tag: A string indicating where to find the TF model description. This
      might be an explicit tag, or the model path from which we can infer the
      model type.
    reduction: How should the correlated signals be reduced to a scalar?  One of
      the options to a Decoder object.
    model: A pre-computed model with which to initialize the decoding object.

  Returns:
    One of the Decoder classes above.
  """
  if 'linear' in model_tag.lower() or 'fullyconnected' in model_tag.lower():
    logging.info('Creating a %s decoding model....', model_tag)
    print(u'Creating a %s decoding model....' % model_tag)
    # TODO Change LinearRegressionDecoder -> RegressionDecoder
    model_object = LinearRegressionDecoder(model, reduction=reduction)
  elif 'cca' in model_tag.lower():
    logging.info('Creating a CCA decoding model....')
    print(u'Creating a CCA decoding model....')
    model_object = CCADecoder(model, reduction=reduction)
  else:
    raise ValueError('Couldn\'t determine model type for %s.' % model_tag)
  return model_object


def create_dataset(tfrecord_file, params, audio_label, frame_rate=100,
                   mode='test', mixup_batch=False):
  """Creates a tensorflow dataset with the test data and a dictionary of params.

  Args:
    tfrecord_file: The source of the data, brain_data tfrecords.
    params: A dictionary with context parameters needed for preprocessing.
    audio_label: Which feature from the tfrecord file contains the audio
      data for this experiment. Typically loudness1 or loudness2.
    frame_rate: The sampling rate for the frames of EEG and intensity data (Hz).
    mode: train (randomized) or testing mode for the resulting tf.data.dataset.
    mixup_batch: Whether the dataset should be randomized so input and output
      no longer correspond. This is important for a null test.

  Returns:
    A tensorflow dataset ready for feeding into a decoding_model.
  """
  tf_dir, tf_file = os.path.split(tfrecord_file)
  logging.info('Reading data from %s with label %s.',
               tfrecord_file, audio_label)

  exp_brain_data = brain_data.TFExampleData(
      params['input_field'],    # Network input
      audio_label,              # Network output
      frame_rate,               # Frame rate
      pre_context=params['pre_context'],
      post_context=params['post_context'],
      in2_fields=audio_label,
      in2_pre_context=params['input2_pre_context'],
      in2_post_context=params['input2_post_context'],
      attended_field='attended_speaker',
      final_batch_size=200,
      repeat_count=1,
      shuffle_buffer_size=0,
      data_dir=tf_dir,
      data_pattern=tf_file,
      train_file_pattern='',
      validate_file_pattern='',
      test_file_pattern='',
      )
  return exp_brain_data.create_dataset(mode, mixup_batch=mixup_batch)


# From: https://en.m.wikipedia.org/wiki/Sensitivity_index
def calculate_dprime(d1, d2):
  """Calculates d' for two sets of data.

  It provides the separation between the means of the signal and the noise
  distributions, compared against the standard deviation of the signal or
  noise distribution.

  Args:
    d1: An np array of data for distribution 1.
    d2: An np array of data for distribution 2.

  Returns:
    A scalar indicating the separation of the two distributions.

  Raises:
    TypeError: If either input is not a 1-dimensional vector
  """
  if d1.ndim > 2 or (d1.ndim == 2 and d1.shape[1] > 1):
    raise TypeError('d1 array must be a vector, not size %s.' % str(d1.shape))
  if d2.ndim > 2 or (d2.ndim == 2 and d2.shape[1] > 1):
    raise TypeError('d2 array must be a vector, not size %s.' % str(d2.shape))

  d1 = np.asarray(d1)
  d2 = np.asarray(d2)
  m1 = np.mean(d1)
  m2 = np.mean(d2)
  v1 = np.var(d1)
  v2 = np.var(d2)
  return (m2-m1)/np.sqrt((v1+v2)/2.0)


def average_data(data, window_size):
  """Averages data over frames of size window_size to smooth results.

  Chunk the data by frames of size window_size (in time) and compute the
  mean of each time (over time, across dimensions.) This is useful when
  we are training LDA as the variance of the data depends on the length of
  the correlation window (longer gives smaller variance).  Using the same
  window_size in training and testing means we have a more accurate LDA model.

  Args:
    data: np.array of data, num_frames (time) x num_features.
    window_size: Number of (temporal) frames to average.  A size of 0 or 1
      means no averaging

  Returns:
    np.array of results: num_windows x num_features.

  Raises:
    TypeError: For incorrect data types.
  """
  if not isinstance(data, np.ndarray):
    raise TypeError('Data to be averaged must be a numpy array, not %s.' %
                    type(data))
  if data.ndim != 2:
    raise TypeError('Averaging data must be two dimensional, not %s.' %
                    data.ndim)
  if not window_size >= 0:
    raise ValueError('Window size (%s) must be greater-than or equal to '
                     'zero.' % window_size)
  if window_size <= 1:
    return data
  num_frames = data.shape[0]//window_size
  short_array = data[0:num_frames*window_size, :].T
  new_array = np.reshape(short_array, (-1, num_frames, window_size))
  sum_array = np.mean(new_array, axis=2)
  return sum_array.T
