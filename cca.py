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

"""Code to implement canonical correlation analysis.

Code, originally from Wieran Wang, that implements canonical correlation
analysis using both numpy (to precompute the best solution) and TF (to optimize
either the linear or non-linear version of) CCA.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

from telluride_decoding import utils

import numpy as np
from six.moves import range
import tensorflow as tf

FLAGS = flags.FLAGS


# Function to estimate CCA rotations from a dataset.
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


def calculate_cca_from_dataset(dataset, dim, regularization=0.1,
                               mini_batch_count=10, eps_eig=1e-12):
  """Estimate the parameters for CCA rotations from a dataset.

  Args:
    dataset: The tf.dataset from which to read data (dictionary item 'x' and
      labels). Dataset is read once (so be sure repeat=1)
    dim: Number of dimensions to return
    regularization: Regularization parameters for the least squares estimates.
    mini_batch_count: How many mini-batches to ingest when calculating dataset
      statistics.
    eps_eig: Ignore eigenvalues (and dimensions) below this value.

  Returns:
    The estimated rot_x and rot_y matrices by which you rotate the x and y data.
    As well as the two means (so they can be subtracted before rotation), and
    the total of the eigenvalues.

  Raises:
    ValueError and/or TypeError for bad parameter values.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset input to calculate_regressor_from_database must be'
                    ' a tf.data.Dataset object, not %s' % type(dataset))
  if regularization < 0.0:
    raise ValueError('regularization lambda must be >= 0')
  cov_xx = 0  # Accumulate sum of x^T x for all minibatches
  cov_yy = 0  # Accumulate sum of y^T y for all minibatches
  cov_xy = 0  # Accumulate sum of x^T y for all minibatches
  sum_x = 0
  sum_y = 0
  data_iter = dataset.make_one_shot_iterator()
  data_element = data_iter.get_next()
  num_mini_batches = 0
  with tf.compat.v1.Session() as sess:
    while num_mini_batches < mini_batch_count:
      try:
        (x_dict, y) = sess.run(data_element)
        x = x_dict['x']
        n_row = x.shape[0]
        cov_xx += np.matmul(x.T, x)
        cov_yy += np.matmul(y.T, y)
        cov_xy += np.matmul(x.T, y)
        sum_x += np.sum(x, axis=0, keepdims=True)
        sum_y += np.sum(y, axis=0, keepdims=True)
        num_mini_batches += 1
      except tf.errors.OutOfRangeError:
        logging.info('Done training linear regressor -- epoch limit reached '
                     'after %d mini batches', num_mini_batches)
        break
  total_frames = float(num_mini_batches*n_row)
  mean_x = sum_x/total_frames
  mean_y = sum_y/total_frames
  cov_xx = cov_xx/(num_mini_batches*n_row-1) - np.matmul(mean_x.T, mean_x)
  cov_xx += regularization*np.eye(x.shape[1])
  cov_yy = cov_yy/(num_mini_batches*n_row-1) - np.matmul(mean_y.T, mean_y)
  cov_yy += regularization*np.eye(y.shape[1])
  cov_xy = cov_xy/(num_mini_batches*n_row-1) - np.matmul(mean_x.T, mean_y)

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


def cca_loss(x, y, dim, rcov1, rcov2, eps_eig=1e-12):
  """Create a TF graph to compute the joint dimensionality via CCA.

  This function computes the number of "dimensions" that two datasets (x and y)
  share.  It creates a TF graph that connects the two TF nodes (x and y) to the
  eigenvalues computed while finding the two optimum rotations to line up the
  two datasets.

  Args:
    x: The first TF data of size n_frames x n_dims_x
    y: The second TF data of size n_frames x n_dims_x
    dim: The desired number of output dimensions
    rcov1: Amount to regularize the x covariance estimate
    rcov2: Amount to regularize the x covariance estimate
    eps_eig: Ignore eigenvalues (and dimensions) below this value.

  Returns:
    TF node that calculates the sum of the eigenvalues.  (Note, this gets
    larger as you get more dimensions in common, so you probably want to negate
    this to turn it into a real loss.)
  """
  # Remove mean.
  m1 = tf.reduce_mean(x, axis=0, keep_dims=True)
  x = tf.subtract(x, m1)

  m2 = tf.reduce_mean(y, axis=0, keep_dims=True)
  y = tf.subtract(y, m2)

  batch_norm = tf.cast(tf.shape(x)[0], tf.float32) - 1.0
  d1 = tf.compat.dimension_value(x.get_shape()[1])  # Get tensor widths
  d2 = tf.compat.dimension_value(y.get_shape()[1])
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
  return tf.reduce_sum(tf.sqrt(vals[-dim:]))


def create_cca_model_fn(x, y, mode, rot_x=None, rot_y=None,
                        mean_x=None, mean_y=None, regularization=1e-2,
                        dimensions=6):
  """This function creates a Canonical Correlation Analysis (CCA) TF network.

  Uses precomputed rotation and mean arrays.  Outputs the concatenation of the
  two input datasets.

  Args:
    x: A dictionary from tf.data.Dataset, with an 'x' field which
      contains one of the two datasets to rotate
    y: A tensor with the second of the two datasets to rotate
    mode: One of training, eval, infer
    rot_x: The initial value for the x rotation matrix
    rot_y: The initial value for the x rotation matrix
    mean_x: The precalculated mean to remove before rotation
    mean_y: The precalculated mean to remove before rotation
    regularization: Amount to add to diagonal to regularize the SVDs
    dimensions: How many canonical correlates (dimensions) to compute.

  Returns:
    A tf estimator spec used by the estimator model.

  Raises:
    ValueError and/or TypeError for bad parameter values.
  """
  if not isinstance(x, dict):
    raise TypeError('Features input to create_cca_model_fn must be a dict.')
  if not isinstance(x['x'], tf.Tensor):
    raise TypeError('Features[x] to create_cca_model_fn must be a tensor.')
  x = x['x']
  if not (isinstance(y, tf.Tensor) or y is None):
    raise ValueError('Y matrix for create_linear_model_fn must be a tensor or '
                     'None')
  logging.info('Building CCA model for %s with %s and %s:', mode, x, y)
  # Build a linear CCA model and predict values
  if isinstance(rot_x, np.ndarray) or isinstance(rot_x, list):
    rot_x = tf.constant(np.array(rot_x, dtype=np.float32))
  if isinstance(rot_y, np.ndarray) or isinstance(rot_y, list):
    rot_y = tf.constant(np.array(rot_y, dtype=np.float32))
  if isinstance(mean_x, np.ndarray) or isinstance(mean_x, list):
    mean_x = tf.constant(np.array(mean_x, dtype=np.float32))
  if isinstance(mean_y, np.ndarray) or isinstance(mean_y, list):
    mean_y = tf.constant(np.array(mean_y, dtype=np.float32))
  logging.info('create_cca_model_fn initializers: rot_x=%s rot_y=%s',
               rot_x, rot_y)
  logging.info('create_cca_model_fn initializers: mean_x=%s mean_y=%s',
               mean_x, mean_y)
  logging.info('create_cca_model_fn looking for %d dimensions.', dimensions)
  with tf.compat.v1.variable_scope('cca'):
    a = tf.compat.v1.get_variable('a', dtype=tf.float32, initializer=rot_x)
    b = tf.compat.v1.get_variable('b', dtype=tf.float32, initializer=rot_y)
    mean_x_t = tf.compat.v1.get_variable('mean_x', dtype=tf.float32,
                                         initializer=mean_x)
    mean_y_t = tf.compat.v1.get_variable('mean_y', dtype=tf.float32,
                                         initializer=mean_y)
    y1 = tf.matmul((x - mean_x_t), a)
    y2 = tf.matmul((y - mean_y_t), b)
  if mode == 'train' or mode == 'eval':
    # Loss sub-graph
    loss = -cca_loss(y1, y2, dimensions, regularization, regularization)
    # Training sub-graph
    global_step = tf.compat.v1.train.get_global_step()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-15)
    train = tf.group(optimizer.minimize(loss),
                     tf.compat.v1.assign_add(global_step, 1))
    # Loss
    pearson_r = utils.pearson_correlation_graph(tf.transpose(y1),
                                                tf.transpose(y2))
    tf.compat.v1.summary.scalar('pearson_correlation', pearson_r[0, 1])
    # Select just one quadrant of correlation matrix
    dims = tf.cast(tf.shape(y1)[1], tf.int32)
    cca_cors = tf.slice(pearson_r, [0, dims], [dims, dims])
    metrics = {
        'test/pearson_correlation_matrix':
            tf.compat.v1.metrics.mean_tensor(pearson_r),
        'test/pearson_correlation':
            tf.compat.v1.metrics.mean(tf.linalg.trace(cca_cors)),
        'test/correlated_dimensions':
            tf.compat.v1.metrics.mean(tf.linalg.trace(cca_cors)),
    }
    for i in range(dimensions):
      t = tf.compat.v1.metrics.mean(cca_cors[i, i])
      metrics['test/pearson_correlation%02d' % i] = tf.compat.v1.metrics.mean(t)
    logging.info('Metric keys are: %s', str(metrics.keys()))
  else:
    train = None
    loss = None
    metrics = None
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=tf.concat((y1, y2), axis=1),
      loss=loss,
      train_op=train,
      eval_metric_ops=metrics)


def create_cca_estimator(dataset, dimensions=5, regularization=0.1):
  """Create a cca estimator, initializing parameters with data from dataset.

  Use the provided dataset (and regularization parameter) to initialize the
  regressor, creating weight and bias matrices that can be plugged into a graph
  as needed by the Estimator.

  Args:
    dataset: A tf.dataset that provides the data needed to estimate the linear
      regressor.  All the data is read once (be sure to set repeat=1) to find
      the optimal parameters.
    dimensions: Number of dimensions to retain in the CCA analysis.
    regularization: The regularization parameter

  Returns:
    A tf.Estimator instance, initialized to implement the optimal linear
    regressor for the provided data.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset input must be a tf.data.Dataset object.')

  rot_x, rot_y, mean_x, mean_y, e = calculate_cca_from_dataset(
      dataset, dimensions, regularization=regularization)
  logging.info('CCA computed a and b matrices of size: %s and %s',
               rot_x.shape, rot_y.shape)
  logging.info('CCA found %g joint dimensions.', float(np.sum(e)))

  def my_linear_model(features, labels, mode):
    return create_cca_model_fn(features, labels, mode, rot_x=rot_x, rot_y=rot_y,
                               mean_x=mean_x, mean_y=mean_y,
                               regularization=regularization,
                               dimensions=dimensions)
  estimator = tf.estimator.Estimator(model_fn=my_linear_model,
                                     model_dir=FLAGS.decoder_model_dir)
  return estimator
