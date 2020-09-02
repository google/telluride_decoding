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

"""Code to implement canonical correlation analysis.

Code, originally from Wieran Wang, that implements canonical correlation
analysis using both numpy (to precompute the best solution) and TF (to optimize
either the linear or non-linear version of) CCA.
"""

from absl import logging

import numpy as np
from telluride_decoding import brain_model
import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


def rmss(x):
  """Compute the root - mean - sign-squared of a vector."""
  logging.info('Using rmss for correlation dimensions.')
  ss = tf.math.sign(x) * tf.math.square(x)
  mss = tf.math.reduce_mean(ss)
  return tf.math.sqrt(tf.math.abs(mss)) * tf.math.sign(mss)


def cca_pearson_correlation(x, y):
  """Calculate the number of correlated dimensions from a CCA output vector.

  The CCA model calculates two rotated vectors, each of width cca_dims.
  These two vectors are concatenated, to give a num_examples x 2*cca_dims
  matrix. This function separates the two sets of data, and computes the
  sum of the individual column correlations.

  Note: Use this function (and the two derivatives below) when computing the
  correlation from the output of a BrainModelCCA because it concatenates the
  two rotated matrices into a single y matrix.  Thus to compute performance
  you only need the y matrix and split it in two vertically to get x and y.

  Args:
    x: The target data, which is ignored by this routine.
    y: The two datasets that are concatenated

  Returns:
    The correlation of just the first dimension, as a placeholder. A
    better result uses LDA to properly reduce the dimensionality.
  """
  del x   # Not used
  cca_width = y.shape[-1] // 2
  if 2*cca_width != y.shape[-1]:
    raise ValueError('CCA y matrix does not have even # dims (%d)' %
                     y.shape[-1])
  rdata1 = y[:, :cca_width]
  rdata2 = y[:, cca_width:]
  correlations = brain_model.pearson_correlation(rdata1, rdata2)
  return correlations


def cca_pearson_correlation_first(x, y):
  correlations = cca_pearson_correlation(x, y)
  return correlations[0]


def cca_pearson_correlation_second(x, y):
  correlations = cca_pearson_correlation(x, y)
  return correlations[1]


######################### Create CCA Regressor ###########################


class BrainCcaLayer(tf.keras.layers.Layer):
  """This layer takes two input batches, and rotates them in an optimal way.

  The CCA layer is designed to rotate the two datasets so they have the maximum
  correlation.  We can find these two rotation matrices using an SVD of the
  covariances. They are computed by running through the data once using a model
  training pass.
  """
  # TODO do we need to define a layer? Can we do this with
  #  conventional layers, and just subclass the model to do the new matrix
  #  calculation. This might be a straightforward example of a functional model.

  def __init__(self, requested_cca_dims, **kwargs):
    self.output_dims = requested_cca_dims
    self._mean1_init = 'uniform'
    self._mean2_init = 'uniform'
    self._rot1_init = 'uniform'
    self._rot2_init = 'uniform'
    super(BrainCcaLayer, self).__init__(**kwargs)

  def set_initial_weights(self, mean1, mean2, rot1, rot2):
    if not isinstance(mean1, np.ndarray) and mean1.shape[0] == 1:    # pytype: disable=attribute-error
      raise TypeError('mean1 matrix has the wrong size (%s)' % mean2.shape)    # pytype: disable=attribute-error
    if not isinstance(mean2, np.ndarray) and mean2.shape[0] == 1:    # pytype: disable=attribute-error
      raise TypeError('mean2 matrix has the wrong size (%s)' % mean2.shape)    # pytype: disable=attribute-error
    if not isinstance(rot1, np.ndarray) and rot1.shape[1] == self.output_dims:
      raise TypeError('rot1 matrix has the wrong size (%s not %s)' %
                      (rot1.shape, self.output_dims))
    if not isinstance(rot2, np.ndarray) and rot2.shape[1] == self.output_dims:
      raise TypeError('rot2 matrix has the wrong size (%s)' % rot2.shape)

    self._mean1_init = tf.constant_initializer(mean1)
    self._mean2_init = tf.constant_initializer(mean2)
    self._rot1_init = tf.constant_initializer(rot1)
    self._rot2_init = tf.constant_initializer(rot2)

    self.set_weights([mean1, mean2, rot1, rot2])

  def build(self, input_shapes):
    logging.info('BrainCcaLayer.build got shapes: %s', input_shapes)

    # Create a trainable weight variable for this layer.
    self.input1_dim = input_shapes[0][-1]
    self.input2_dim = input_shapes[1][-1]
    real_output_dims = min(self.input1_dim, self.input2_dim, self.output_dims)

    self.mean1 = self.add_weight(name='mean1',
                                 shape=(1, self.input1_dim),
                                 initializer=self._mean1_init,
                                 trainable=False)
    self.mean2 = self.add_weight(name='mean2',
                                 shape=(1, self.input2_dim),
                                 initializer=self._mean2_init,
                                 trainable=False)
    self.rot1 = self.add_weight(name='rot1',
                                shape=(self.input1_dim, real_output_dims),
                                initializer=self._rot1_init,
                                trainable=False)
    self.rot2 = self.add_weight(name='rot2',
                                shape=(self.input2_dim, real_output_dims),
                                initializer=self._rot2_init,
                                trainable=False)
    logging.info('BrainCcaLayer.build mean1=%s, mean2=%s, rot1=%s, rot2=%s',
                 self.mean1.shape, self.mean2.shape,
                 self.rot1.shape, self.rot2.shape)

  def call(self, inputs):
    input1 = inputs[0]
    input2 = inputs[1]

    input1 = tf.reshape(input1, (-1, self.input1_dim))
    input2 = tf.reshape(input2, (-1, self.input2_dim))

    rotated_input1 = tf.keras.backend.dot(input1 - self.mean1, self.rot1)
    rotated_input2 = tf.keras.backend.dot(input2 - self.mean2, self.rot2)

    return tf.concat((rotated_input1, rotated_input2),
                     axis=1, name='ConcatenateCcaResults')

  def get_config(self):
    return {'requested_cca_dims': self.output_dims,
           }


class BrainModelCCA(brain_model.BrainModel):
  """A BrainModel class that implements canonical correlation."""

  def __init__(self, input_dataset, cca_dims=5, regularization_lambda=0.0,
               **kwargs):
    super(BrainModelCCA, self).__init__(**kwargs)

    self._cca_dims = cca_dims
    self._regularization_lambda = regularization_lambda
    self._cca_layer = None
    self._input1_width = input_dataset.element_spec[0]['input_1'].shape[-1]
    self._input2_width = input_dataset.element_spec[0]['input_2'].shape[-1]

    if self._input1_width <= 1:
      raise ValueError('Input 1 feature width (%d) should not be <= 1.' %
                       self._input1_width)
    if self._input2_width <= 1:
      raise ValueError('Input 2 feature width (%d) should not be <= 1.' %
                       self._input2_width)

    self._var_1 = tf.keras.Input(shape=self._input1_width, name='input_1')
    self._var_2 = tf.keras.Input(shape=self._input2_width, name='input_2')
    self._cca_layer = BrainCcaLayer(self._cca_dims)

  def call(self, input_data):
    return self._cca_layer([input_data['input_1'], input_data['input_2']])

  def compile(self, optimizer=tf.keras.optimizers.RMSprop,
              loss=cca_pearson_correlation_first,
              metrics=cca_pearson_correlation_first,
              learning_rate=1e-3, **kwargs):
    """Compiles this model, applying the usual defaults for this classifier.

    Args:
      optimizer: Which Keras optimizer class to use when training.
      loss: How do we normally define the performance (loss) of the network.
      metrics: Which metric(s) do we report after training.
      learning_rate: A learning rate to pass to the optimizer
      **kwargs: Arguments that are simply passed to the super class' compile().
    """
    if callable(optimizer):
      optimizer = optimizer(learning_rate=learning_rate)
    super(BrainModelCCA, self).compile(
        optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

  def fit(self, dataset, epochs=1):
    """Train the CCA model, using the given dataset.

    Dataset must return a twople: the input dictionary with two keys [input_1
    and input_2], and the output variable, which is ignored but could be used
    later for match/mis-match indication.

    Args:
      dataset: a tf.data.dataset object from which to draw training data
      epochs: How many epochs to run (should be 1 since this is deterministic)

    Returns:
      A dictionary of metrics after evaluation on the training data
    """
    del epochs  # Not needed here since we just read the data once.
    if not isinstance(dataset, tf.data.Dataset):
      raise TypeError('BrainModelLinearRegression.train must be called with '
                      'tf.data.Dataset.')

    # Be sure to call once to create the network.
    for inputs, _ in dataset.take(1):
      self.call(inputs)

    (self.rot_x, self.rot_y, self.mean_x, self.mean_y,
     e) = calculate_cca_parameters_from_dataset(
         dataset, self._cca_dims, regularization=self._regularization_lambda,
         mini_batch_count=0)
    logging.info('CCA eigenvalues are: %s', e)
    self._cca_layer.set_initial_weights(self.mean_x, self.mean_y,
                                        self.rot_x, self.rot_y)
    return {}  # No training results


# Function to estimate CCA rotations from the raw data in a dataset.
#===============================================================================
#
# (C) 2019 by Weiran Wang (weiranwang@ttic.edu) and
# Qingming Tang (qmtang@ttic.edu)
#
# ==============================================================================
#
# This package contains python+tensorflow code for the Deep Canonical
# Correlation Analysis algorithm. (An earlier MATLAB implementation can be
# found at https://ttic.uchicago.edu/~wwang5/dccae.html)
#
# If you use this code, please cite the following papers.
#
# @inproceedings{Wang15, title={On deep multi-view representation learning},
# author={Wang, Weiran and Arora, Raman and Livescu, Karen and Bilmes, Jeff},
# booktitle={International Conference on Machine Learning}, pages={1083--1092},
# year={2015} }
#
# @inproceedings{Andrew13, title={Deep canonical correlation analysis},
# author={Andrew, Galen and Arora, Raman and Bilmes, Jeff and Livescu, Karen},
# booktitle={International conference on machine learning}, pages={1247--1255},
# year={2013} } https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf


def calculate_cca_parameters_from_dataset(dataset, dim, regularization=0.1,
                                          mini_batch_count=1000, eps_eig=1e-12):
  """Estimate the parameters for CCA rotations from a dataset.

  Args:
    dataset: The tf.dataset from which to read data (dictionary items 'input_1'
      and 'input_2'). Dataset is read once (so be sure repeat=1)
    dim: Number of dimensions to return
    regularization: Regularization parameters for the least squares estimates.
    mini_batch_count: How many mini-batches to ingest when calculating dataset
      statistics.
    eps_eig: Ignore eigenvalues (and dimensions) below this value.

  Returns:
    The estimated rot_x and rot_y matrices by which you rotate the x and y data.
    As well as the two means (so they can be subtracted before rotation), and
    the first "dim" eigenvalues.

  Raises:
    ValueError and/or TypeError for bad parameter values.
  """
  if not isinstance(dataset,
                    tf.data.Dataset) and not isinstance(dataset,
                                                        tf.data.DatasetV2):
    raise TypeError('dataset input to calculate_regressor_from_database must be'
                    ' a tf.data.Dataset object, not %s' % type(dataset))
  if regularization < 0.0:
    raise ValueError('regularization lambda must be >= 0')
  logging.info('Calculating CCA parameters from a dataset with %s.', dataset)
  logging.info(' Looking for %d output dimensions with regularization = %g.',
               dim, regularization)

  cov_xx = 0  # Accumulate sum of x^T x for all minibatches
  cov_yy = 0  # Accumulate sum of y^T y for all minibatches
  cov_xy = 0  # Accumulate sum of x^T y for all minibatches
  sum_x = 0
  sum_y = 0
  num_mini_batches = 0
  total_frames = 0
  for dataset_item in dataset.take(mini_batch_count or -1):
    (x_dict, y) = dataset_item
    if not isinstance(x_dict, dict):
      raise TypeError('X_dict is a %s, not a dict.'  % type(x_dict))
    x = x_dict['input_1'].numpy()
    y = x_dict['input_2'].numpy()
    if x.shape[1] == 0:
      raise ValueError('First input to CCA estimator must have more '
                       'than 0 columns.')
    if y.shape[1] == 0:
      raise ValueError('Second input to CCA estimator must have more '
                       'than 0 columns.')
    n_row = x.shape[0]
    total_frames += x.shape[0]
    cov_xx += np.matmul(x.T, x)
    cov_yy += np.matmul(y.T, y)
    cov_xy += np.matmul(x.T, y)
    sum_x += np.sum(x, axis=0, keepdims=True)
    sum_y += np.sum(y, axis=0, keepdims=True)
    num_mini_batches += 1
    if mini_batch_count and num_mini_batches >= mini_batch_count:
      break
  logging.info('Calculating the CCA parameters from %d minibatches',
               num_mini_batches)
  if not num_mini_batches:
    raise ValueError('No minibatches in dataset, can\'t compute CCA model.')
  mean_x = sum_x/total_frames
  mean_y = sum_y/total_frames
  cov_xx = cov_xx/(num_mini_batches*n_row-1) - np.matmul(mean_x.T, mean_x)    # pytype: disable=attribute-error
  cov_xx += regularization*np.eye(x.shape[1])
  cov_yy = cov_yy/(num_mini_batches*n_row-1) - np.matmul(mean_y.T, mean_y)    # pytype: disable=attribute-error
  cov_yy += regularization*np.eye(y.shape[1])
  cov_xy = cov_xy/(num_mini_batches*n_row-1) - np.matmul(mean_x.T, mean_y)    # pytype: disable=attribute-error

  x_vals, x_vecs = np.linalg.eig(cov_xx)   # E1, x_vecs
  y_vals, y_vecs = np.linalg.eig(cov_yy)   # y_vals, y_vecs

  # For numerical stability.
  idx1 = np.where(x_vals > eps_eig)[0]
  x_vals = x_vals[idx1]
  x_vecs = x_vecs[:, idx1]

  idx2 = np.where(y_vals > eps_eig)[0]
  y_vals = y_vals[idx2]
  y_vecs = y_vecs[:, idx2]

  k11 = np.matmul(np.matmul(x_vecs, np.diag(np.reciprocal(np.sqrt(x_vals)))),
                  x_vecs.transpose())
  k22 = np.matmul(np.matmul(y_vecs, np.diag(np.reciprocal(np.sqrt(y_vals)))),
                  y_vecs.transpose())
  t = np.matmul(np.matmul(k11, cov_xy), k22)
  u, e, v = np.linalg.svd(t, full_matrices=False)
  v = v.transpose()

  rot_x = np.matmul(k11, u[:, 0:dim])
  rot_y = np.matmul(k22, v[:, 0:dim])
  e = e[0:dim]

  return rot_x, rot_y, mean_x, mean_y, e


@tf.function
def cca_loss(x, y, dim, rcov1, rcov2, eps_eig=1e-12):
  """Create a TF graph to compute the joint dimensionality via CCA.

  This function computes the number of "dimensions" that two datasets (x and y)
  share. It creates a TF graph that connects the two TF nodes (x and y) to the
  eigenvalues computed while finding the two optimum rotations to line up the
  two datasets. We want to maximize this measure (- the loss).  This function
  computes the dimensions for two unrotated data.  Use the regular Pearson
  correlation after the CCA model is built.

  See:
  https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

  Args:
    x: The first TF data of size n_frames x n_dims_x
    y: The second TF data of size n_frames x n_dims_y
    dim: The desired number of output dimensions
    rcov1: Amount to regularize the x covariance estimate
    rcov2: Amount to regularize the x covariance estimate
    eps_eig: Ignore eigenvalues (and dimensions) below this value.

  Returns:
    TF node that calculates the sum of the eigenvalues.  (Note, this gets
    larger as you get more dimensions in common, so you probably want to negate
    this to turn it into a real loss.)
  """
  logging.info('In cca_loss, data at start are: %s %s %d-D %g %g %g',
               x, y, dim, rcov1, rcov2, eps_eig)
  # Remove the means.
  m1 = tf.math.reduce_mean(x, axis=0, keepdims=True)
  x = tf.subtract(x, m1)

  m2 = tf.math.reduce_mean(y, axis=0, keepdims=True)
  y = tf.subtract(y, m2)

  batch_norm = tf.cast(tf.shape(x)[0], tf.float32) - 1.0
  d1 = tf.compat.dimension_value(tf.shape(x)[1])  # Get dynamic tensor widths
  d2 = tf.compat.dimension_value(tf.shape(y)[1])
  eye1 = tf.eye(d1, dtype=tf.float32)
  cov_xx = tf.matmul(tf.transpose(x), x) / batch_norm + rcov1*eye1
  eye2 = tf.eye(d2, dtype=tf.float32)
  cov_yy = tf.matmul(tf.transpose(y), y) / batch_norm + rcov2*eye2
  cov_xy = tf.matmul(tf.transpose(x), y) / batch_norm

  x_vals, x_vecs = tf.linalg.eigh(cov_xx)
  y_vals, y_vecs = tf.linalg.eigh(cov_yy)

  # For numerical stability.
  idx1 = tf.where(x_vals > eps_eig)[:, 0]
  x_vals = tf.gather(x_vals, idx1)
  x_vecs = tf.gather(x_vecs, idx1, axis=1)

  idx2 = tf.where(y_vals > eps_eig)[:, 0]
  y_vals = tf.gather(y_vals, idx2)
  y_vecs = tf.gather(y_vecs, idx2, axis=1)

  k11 = tf.matmul(tf.matmul(x_vecs,
                            tf.linalg.tensor_diag(tf.math.reciprocal(
                                tf.sqrt(x_vals)))),
                  tf.transpose(x_vecs))
  k22 = tf.matmul(tf.matmul(y_vecs,
                            tf.linalg.tensor_diag(tf.math.reciprocal(
                                tf.sqrt(y_vals)))),
                  tf.transpose(y_vecs))
  t = tf.matmul(tf.matmul(k11, cov_xy), k22)

  # Eigenvalues are sorted in increasing order.
  vals, _ = tf.linalg.eigh(tf.matmul(t, tf.transpose(t)))
  # Make sure none of the (small) eigenvalues are negative
  estimated_cca_dims = tf.reduce_sum(tf.sqrt(tf.math.maximum(0.0, vals[-dim:])))
  return estimated_cca_dims
