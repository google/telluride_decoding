# Copyright 2019-2020 Google Inc.
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

"""TF code to decode an MEG/EEG signal.

TF models and code to predict MEG/EEG signals from their input audio features,
or vice versa.

"""

import datetime
import json
import os

from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


def pearson_correlation(x, y):
  """Compute the Pearson correlation coefficient between two tensors of data.

  This routine computes a vector correlation using Tensorflow ops.

  Calculation from:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  Args:
    x: one of two input arrays.
    y: second of two input arrays.

  Returns:
    Tensor vector of correlation coefficients, one correlation per column of
    data.

  Note: When used as a Keras metric, the mean of this multidimensional output
  is used, which is probablematic since most of the lower dimensions are close
  to zero, and the mean heads towards zero.  The _first and _second routines are
  probably a better choice.

  Note #2: Do not use this directly to evaluate the output of a CCA model. That
  model concatentates both outputs into the y array, so you need to compute the
  error *within* just y.
  """
  # TODO should this be a tf.function?
  assert x.shape[-1] == y.shape[-1], ('x (%s) and y (%s) do not have the same '
                                      'final dimensionality' % (x.shape,
                                                                y.shape))
  x_m = x - tf.math.reduce_mean(x, axis=0)
  y_m = y - tf.math.reduce_mean(y, axis=0)
  x_p = tf.math.reduce_sum(tf.math.square(x_m), axis=0)
  y_p = tf.math.reduce_sum(tf.math.square(y_m), axis=0)

  def positive_fcn():
    res = tf.divide(tf.math.reduce_sum(tf.multiply(x_m, y_m), axis=0),
                    tf.multiply(tf.math.sqrt(x_p), tf.math.sqrt(y_p)))
    return res

  def negative_fcn():
    return 0*x_m  # Just to get the right size

  zero_cond = tf.math.logical_or(tf.math.reduce_prod(x_p) <= 0,
                                 tf.math.reduce_prod(y_p) <= 0)

  return tf.cond(zero_cond,
                 negative_fcn, positive_fcn)


def pearson_correlation_first(x, y):
  """Return the correlation of the first CCA dimension."""
  result = pearson_correlation(x, y)
  return result[0]


def pearson_correlation_second(x, y):
  """Return the correlation of the second CCA dimension."""
  result = pearson_correlation(x, y)
  return result[1]


class PearsonCorrelationLoss(tf.keras.losses.Loss):
  """Implements the Pearson correlation calculation as a Keras loss.

  Note, this since this is a Keras loss, this function returns the instantenous
  correlation for each data point.  Sum all of these values to get the full
  batch correlation.

  Testing: Not sure if this will train a network to converge.
  """

  def call(self, x, y):
    """The actual correlation calculation.  See class notes above.

    Args:
      x: first data array of size num_frames x num_features.
      y: second data array of the same size as x

    Returns:
      A vector with num_frame individual *negative* correlations. The negative
      of the correlations are returned so that this can be used as a loss.
    """
    if x.shape != y.shape:
      raise ValueError('Two correlation arrays must have the same size, not '
                       ' %s vs %s.' % ((x.shape, y.shape)))
    x_m = x - tf.math.reduce_mean(x, axis=0)
    y_m = y - tf.math.reduce_mean(y, axis=0)

    x_std = tf.math.reduce_sum(tf.math.square(x_m), axis=0)
    y_std = tf.math.reduce_sum(tf.math.square(y_m), axis=0)

    power = tf.sqrt(tf.multiply(x_std, y_std))
    return -tf.math.reduce_sum(tf.divide(tf.multiply(x_m, y_m), power),
                               axis=-1)


class BrainModel(tf.keras.models.Model):
  """Light wrapper around Keras Model with better error checking.

  This class defines several different kinds of networks (e.g. linear, CCA, and
  DNN). This allows us to specialize the training process (in the case of CCA
  and linear, which have deterministic algorithms.)

  In addition this class makes three additions to the standard Keras Model
  class.
  1) The class __init__ method takes a tensorboard_dir argument, which is then
  automatically added as a callback to the fit and evaluate methods.

  2) The fit and evaluate methods both do additional type checking, to make
  sure the dataset are compatible with the call() method, before calling the
  system routines.

  3) The evaluate function returns a dictionary of results (instead of a bare
  list)

  """

  def __init__(self, tensorboard_dir=None, **kwargs):
    """Create a BrainModel object.

    Note, this class essentially serves as a shim, adding a couple of features
    to the standard API.  See above.

    Args:
      tensorboard_dir: location to dump tensorboard logs.  Added to evaluate
        and fit methods.
      **kwargs: Any arguments supported or needed by tf.keras.models.Model
    """

    if tensorboard_dir:
      self._tensorboard_dir = os.path.join(
          tensorboard_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
      logging.info('Writing tensorboard data to %s', self._tensorboard_dir)
    else:
      self._tensorboard_dir = None

    super(BrainModel, self).__init__(**kwargs)

  @property
  def tensorboard_dir(self):
    return self._tensorboard_dir

  def fit(self, x=None, y=None, **kwargs):
    """Train the generic model. Add the tensorboard callback if requested.

    Args:
      x: Data to use for training model
      y: Needed for compatibility with the super class, but should always be
        None, since the dataset supplies inputs and outputs.
      **kwargs: Extra arguments to pass to the main fit method

    Returns:
      A dictionary of training statistics, created by zipping the original
      results and the metric names into a single dictionary.
    """
    if not isinstance(x, tf.data.Dataset) and 'input_1' not in x:
      raise TypeError('BrainModel.train must be called with tf.data.Dataset '
                      'object, not %s' % x)
    if y is not None:
      raise ValueError('Y value not needed, should be part of dataset.')

    if self._tensorboard_dir:
      if 'callbacks' in kwargs:
        kwargs['callbacks'].append(
            tf.keras.callbacks.TensorBoard(log_dir=self._tensorboard_dir))
      else:
        kwargs['callbacks'] = [
            tf.keras.callbacks.TensorBoard(log_dir=self._tensorboard_dir),]
    history = super(BrainModel, self).fit(x, **kwargs)
    logging.info('Training a %s model returned these metrics: %s',
                 self, history)
    return history

  def evaluate(self, x=None, y=None, epoch_count=1, **kwargs):
    """Evaluate a model using a sample dataset.

    The dataset should provide a tuple:
      just the input and output tensors (no dictionary)
    This method is a shim, automatically adding the tensorboard callback and
    reformatting the results into a dictionary (instead of an unlabeled list
    of losses.)

    Args:
      x: data to use for evaluation (a tf.data.Dataset)
      y: An empty argument, to retain compatibility with the superclass
      epoch_count: How many epochs have we trained on so far, for reporting.
      **kwargs: Extra arguments to pass to the super class.

    Returns:
      A dictionary of results.
    """
    logging.info('Evaluating with the %s dataset.', x)
    if not isinstance(x, tf.data.Dataset):
      raise TypeError('BrainModel.evaluate must be called with tf.data.Dataset'
                      ' object.')

    if self._tensorboard_dir:
      if 'callbacks' in kwargs:
        kwargs['callbacks'].append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_dir+'/test'))
      else:
        kwargs['callbacks'] = [tf.keras.callbacks.TensorBoard(
            log_dir=self._tensorboard_dir+'/test'),]

    results = super(BrainModel, self).evaluate(x, **kwargs)
    logging.info('Evaluate names are: %s', self.metrics_names)
    logging.info('Evaluate results are: %s', results)
    if not isinstance(results, list):
      results = [results,]
    metrics = dict(zip(self.metrics_names, results))

    if self._tensorboard_dir:
      # Add our own summary statistics so we can see them in Tensorboard.
      logdir = os.path.join(self._tensorboard_dir, 'results')
      writer = tf.summary.create_file_writer(logdir=logdir)
      with writer.as_default():
        for name, val in metrics.items():
          tf.summary.scalar(name, val, step=epoch_count)

    return metrics

  def add_metadata(self, flags, dataset=None):
    """Add data to the model so it will be saved with the model.

    Data can be in any format, but it probably makes the most sense for it to
    be a dictionary.

    Args:
      flags: parameters to add, probably a dictionary of flag values.
      dataset: Optional dataset from which to infer the input & output sizes
    """
    # Must use variable here, not constant, in order for the value to be
    # saved in the model.
    self.telluride_metadata = tf.Variable(json.dumps(flags))

    if not dataset:
      return
    if not isinstance(dataset, tf.data.Dataset):
      raise TypeError('dataset parameter must be tf.data.Dataset type.')

    for data in dataset.take(1):
      inputs, output = data
    for k in inputs:
      inputs[k] = list(inputs[k].shape)
    output = list(output.shape)
    self.telluride_inputs = tf.Variable(json.dumps(inputs))
    self.telluride_output = tf.Variable(json.dumps(output))

  def add_tensorboard_summary(self, name, data, subdir='train', step=0):
    """Adds a scalar event to the tensorboard data.

    Args:
      name: Name of the variable or event, a string.
      data: A scalar value associated with the variable.
      subdir: A string indicating the sub directory to store the event.
      step: Which time step to associate with the result.
    """
    if not isinstance(name, str):
      raise TypeError('Tensorboard name must be a string, not a %s.' %
                      type(name))
    if not isinstance(subdir, str):
      raise TypeError('Tensorboard subdir must be a string, not a %s.' %
                      type(subdir))
    if self._tensorboard_dir:
      logdir = os.path.join(self._tensorboard_dir, subdir)
      writer = tf.summary.create_file_writer(logdir=logdir)
      with writer.as_default():
        tf.summary.text(name, str(data), step=step)

######################### Create Linear Regressor ###########################


class BrainModelLinearRegression(BrainModel):
  """A linear regression class, computed deterministically from the input data.

  Implemented as a single dense layer.  Use regularization when computing the
  optimum weights.
  """

  def __init__(self, input_dataset, regularization_lambda=0.0,
               tensorboard_dir=None, **kwargs):
    """Create a LinearRegression model.

    Args:
      input_dataset: data used to figure out the network sizes
      regularization_lambda: amount of regularization to perform when computing
        the best model.
      tensorboard_dir: Where to store data for Tensorboard
      **kwargs: Arguments that are ignore and simply passed to the super class.
    """
    super(BrainModelLinearRegression, self).__init__(
        tensorboard_dir=tensorboard_dir, **kwargs)
    if not isinstance(input_dataset, tf.data.Dataset):
      raise ValueError('Dataset must be a tf.data.datasert, not a %s' %
                       type(input_dataset))
    self._input_width = input_dataset.element_spec[0]['input_1'].shape[-1]
    self._output_width = input_dataset.element_spec[1].shape[-1]
    self._regularization_lambda = regularization_lambda

    self._input_1 = tf.keras.Input(shape=self._input_width,
                                   name='input_1')
    self._layer = tf.keras.layers.Dense(self._output_width, activation=None,
                                        use_bias=True, trainable=True,
                                        kernel_initializer=None,
                                        bias_initializer=None)

  def call(self, input_dataset):
    return self._layer(input_dataset['input_1'])

  def compile(self, optimizer=tf.keras.optimizers.RMSprop,
              loss='mse',
              metrics=pearson_correlation_first,
              learning_rate=1e-3, **kwargs):
    """Compile this model, applying the usual defaults for this classifier.

    Args:
      optimizer: Which Keras optimizer to use when training.
      loss: How do we normally define the performance (loss) of the network.
      metrics: Which metric(s) do we report after training.
      learning_rate: A learning rate to pass to the optimizer
      **kwargs: Arguments that are simply passed to the super class' compile().
    """
    if callable(optimizer):
      optimizer = optimizer(learning_rate=learning_rate)
    super(BrainModelLinearRegression, self).compile(
        optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

  def fit(self, input_dataset, **kwargs):
    """Do our own training since optimal parameters can be calculated."""
    del kwargs  # Not needed here since we only need to read the data once.
    if not isinstance(input_dataset, tf.data.Dataset):
      raise TypeError('BrainModelLinearRegression.train must be called with '
                      'tf.data.Dataset, not %s.' % type(input_dataset))

    (self.w_estimate, self.b_estimate, _, _,
     _) = calculate_linear_regressor_parameters_from_dataset(
         input_dataset, lamb=self._regularization_lambda)
    self.b_estimate = np.reshape(self.b_estimate, (-1,))

    # Need to call the function once to build the network.
    for input_data, _ in input_dataset:
      self.call(input_data)
    self._layer.set_weights([self.w_estimate, self.b_estimate])
    return {}  # No training history

  @property
  def weight_matrices(self):
    return self._layer.get_weights()


def calculate_linear_regressor_parameters_from_dataset(dataset, lamb=0.1,
                                                       use_offset=True,
                                                       use_ridge=True):
  """Estimate the parameters for a linear regressor from a dataset.

  Finds A to solve the equation:
    Ax = y
  This routine reads the dataset, calculating the necessary covariances, and
  then returns the solution A to the equation above. Use these values to preload
  a linear regressor estimator.

  Regression calculation defined here:
  https://stackoverflow.com/questions/45959112/get-coefficients-of-a-linear-regression-in-tensorflow

  Note, lambda in this routine corresponds to the shrinkage parameter gamma
  in Blankertz et al. NeuroImage 58 (2011) 814-825, specificially used in Eq 13.

  Args:
    dataset: The tf.dataset from which to read data (dictionary item 'input_1'
      and labels). Dataset is read once (so be sure repeat=1)
    lamb: Regularization parameters for the least squares estimates.
    use_offset: Whether to include the additive bias offset
    use_ridge: Use ridge regression, instead of shrinkage for regularization.

  Returns:
    The estimated A and b matrices. As well as the two covariance matrices for
    debugging, and the optimal shrinkage parameter.

  Raises:
    ValueError and/or TypeError for bad parameter values.

  Automatic regularization (when lamb == -1) based on this paper:
      http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset input to '
                    'calculate_linear_regressor_parameters_from_database must '
                    'be a tf.data.Dataset object')
  sum_x = 0.0
  sum_xtx = 0.0
  sum_x2tx2 = 0  # Accumulate sum of x2^T x2 for all minibatches
  sum_xty = 0  # Accumulate sum of x^T y for all minibatches
  num_mini_batches = 0
  num_samples = 0  # Total number of samples

  for x, y in dataset:
    x = x['input_1'].numpy()
    y = y.numpy()
    num_rows = x.shape[0]
    num_samples += num_rows
    if use_offset:
      # Append a column of 1's so we can compute an offset.
      x = np.hstack((x, np.ones((num_rows, 1), dtype=x.dtype)))
    sum_xtx += np.matmul(x.T, x)
    sum_x += np.sum(x, axis=0, keepdims=True)
    sum_xty += np.matmul(x.T, y)
    if lamb == -1:
      xc = x - sum_x/num_samples
      x2 = xc ** 2
      sum_x2tx2 += np.matmul(x2.T, x2)
    num_mini_batches += 1
  logging.info('Calculate_linear_regressor_parameters_from_dataset: Processed '
               '%d minibatches of size %d', num_mini_batches, num_rows)
  cov_x = sum_xtx / num_samples
  cov_xy = sum_xty / num_samples
  mean_x = sum_x/num_samples
  cov_x_zc = sum_xtx - np.matmul(mean_x.T, mean_x)    # pytype: disable=attribute-error
  n_col = cov_x.shape[0]  # pytype: disable=attribute-error
  mu = np.trace(cov_x_zc) / n_col
  if use_ridge:
    cov_x += lamb * np.identity(n_col)
    shrinkage = lamb
  else:
    if lamb == -1:
      cov_x2 = sum_x2tx2 / num_samples
      delta_ = cov_x_zc.copy()
      delta_.flat[::n_col + 1] -= mu
      delta = (delta_**2).sum() / n_col
      beta_ = 1. / (n_col * num_samples) \
          * np.sum(cov_x2 - (cov_x_zc ** 2))
      beta = min(beta_, delta)
      shrinkage = beta / delta
    elif lamb > 1 or lamb < 0:
      # Shrinkage values are weird outside this range.
      raise ValueError('Regularization lambda must be between 0 and 1, not %g.'%
                       lamb)
    else:
      shrinkage = lamb
    logging.info('Shrinkage scaling is %g, %g, %g', shrinkage, lamb,
                 np.mean(np.trace(cov_x) / n_col))

    # Equation 12 of Blankertz.  Shink eigenvalues toward the mean.
    cov_x = (1 - shrinkage) * cov_x + shrinkage * mu * np.identity(n_col)
  solution = np.linalg.solve(cov_x, cov_xy)
  if use_offset:
    return solution[0:-1, :], solution[-1:, :], cov_x, cov_xy, shrinkage
  else:
    return solution, np.zeros((1,)), cov_x, cov_xy, shrinkage

######################### Create DNN Regressor SubModel ########################


class BrainModelDNN(BrainModel):
  """A deep neural network regressor class.
  """

  def __init__(self, input_dataset, num_hidden_list=None, **kwargs):
    """Creates model based on DNNs, a shim around basic Keras Model class.

    Args:
      input_dataset: Data used to figure out the network sizes.
      num_hidden_list: A list of the number of hidden units in each layer.
      **kwargs: Arguments that are ignore and simply passed to the super class.
    """
    if not isinstance(input_dataset, tf.data.Dataset):
      raise ValueError('Dataset must be a tf.data.datasert, not a %s' %
                       type(input_dataset))
    if num_hidden_list is None:
      num_hidden_list = []
    if not isinstance(num_hidden_list, list):
      raise TypeError('Num_hidden_list must be an list, not a %s.' %
                      type(num_hidden_list))

    super(BrainModelDNN, self).__init__(**kwargs)
    self._input_width = input_dataset.element_spec[0]['input_1'].shape[-1]
    self._output_width = input_dataset.element_spec[1].shape[-1]

    self._input_1 = tf.keras.Input(shape=self._input_width, name='input_1')

    self.num_hidden_list = num_hidden_list
    logging.info('Creating a BrainModelDNN with these hidden unit counts: %s',
                 num_hidden_list)

    self._layer_list = []
    for hidden_units in num_hidden_list:
      self._layer_list.append(tf.keras.layers.Dense(hidden_units,
                                                    activation='relu'))
    self._final_layer = tf.keras.layers.Dense(self._output_width,
                                              activation=None)

  def call(self, input_dataset):
    layer_input = input_dataset['input_1']
    for a_layer in self._layer_list:
      layer_input = a_layer(layer_input)
    return self._final_layer(layer_input)

  def compile(self, optimizer=tf.keras.optimizers.RMSprop,
              loss='mse',
              metrics=(pearson_correlation_first, 'mse'),
              learning_rate=1e-3, **kwargs):
    """Compile this model, applying the usual defaults for this classifier.

    Args:
      optimizer: Which Keras optimizer to use when training.
      loss: How do we normally define the performance (loss) of the network.
      metrics: Which metric(s) do we report after training.
      learning_rate: A learning rate to pass to the optimizer
      **kwargs: Arguments that are simply passed to the super class' compile().
    """
    if callable(optimizer):
      optimizer = optimizer(learning_rate=learning_rate)
    super(BrainModelDNN, self).compile(
        optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

  def fit(self, input_dataset, **kwargs):
    return super(BrainModelDNN, self).fit(input_dataset, None, **kwargs)

####################### Create DNN Classifier SubModel ########################


class BrainModelClassifier(BrainModel):
  """A deep neural network classifier class.

  Skip the model-building stage, and directly classify whether the audio and the
  EEG are coincident.
  """

  def __init__(self, input_dataset, num_hidden_list=None, **kwargs):
    """Creates model based classfifier, a shim around basic Keras Model class.

    This takes two inputs, concatenates them, and optimizes a model that
    predicts the output. This is generally used as a classifier, and more
    specifically a match-mistmatch classifier.

    Args:
      input_dataset: tf.data.datset used to figure out the network sizes.
      num_hidden_list: A list of the number of hidden units in each layer.
      **kwargs: Arguments that are simply passed to the super class.
    """
    if not isinstance(input_dataset, tf.data.Dataset):
      raise TypeError('Dataset must be a tf.data.datasert, not a %s' %
                      type(input_dataset))
    if num_hidden_list is None:
      num_hidden_list = []
    if not isinstance(num_hidden_list, list):
      raise TypeError('Num_hidden_list must be an list, not a %s.' %
                      type(num_hidden_list))

    super(BrainModelClassifier, self).__init__(**kwargs)
    self._output_width = input_dataset.element_spec[1].shape[-1]

    self.num_hidden_list = num_hidden_list
    logging.info('Creating a BrainModelDNN with these hidden unit counts: %s',
                 num_hidden_list)

    self._layer_list = [tf.keras.layers.Dense(hidden_units, activation='relu')
                        for hidden_units in num_hidden_list]
    self._final_layer = tf.keras.layers.Dense(self._output_width,
                                              activation='sigmoid')

  def compile(self, optimizer=tf.keras.optimizers.Adam,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics='accuracy', learning_rate=1e-3, **kwargs):
    """Compile this model, applying the usual defaults for this classifier.

    Args:
      optimizer: Which Keras optimizer to use when training.
      loss: How do we normally define the performance (loss) of the network.
      metrics: Which metric(s) do we report after training.
      learning_rate: learning rate passed to the optimizer.
      **kwargs: Arguments that are simply passed to the super class' compile().
    """
    if callable(optimizer):
      optimizer = optimizer(learning_rate=learning_rate)
    super(BrainModelClassifier, self).compile(
        optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

  def call(self, input_dataset):
    layer_input = tf.concat((input_dataset['input_1'],
                             input_dataset['input_2']),
                            axis=1)
    for a_layer in self._layer_list:
      layer_input = a_layer(layer_input)
    return self._final_layer(layer_input)

  def fit(self, input_dataset, **kwargs):
    return super(BrainModelClassifier, self).fit(input_dataset, None, **kwargs)
