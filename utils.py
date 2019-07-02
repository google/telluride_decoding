"""Utility routines when decoding brain signals.

More to come.. the CCA functions need to get the Pearson correlation too.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO(malcolmslaney) Check to see if we can use (might need to get into core)
# contrib/streaming_pearson_correlation


def pearson_correlation_graph(x, y):
  """Create a graph to compute the Pearson correlation.

  This works between two (multi-dimensional) feature arrays.  Like np.corrcoef,
  x and y are 1-D or 2-D arrays containing multiple variables and observations.
  Each row of x represents a variable, and each column a single observation of
  all those variables.
  From:
  https://stackoverflow.com/questions/43834830/tensorflow-equivalent-of-np-
  corrcoef-on-a-specific-axis

  Args:
    x: First multidimensional array, n_vars x n_observations
    y: Second multidimensional array, n_vars x n_observations

  Returns:
    Tensorflow operation that computes the correlation matrix.
  """
  def t(x):
    return tf.transpose(x)
  with tf.compat.v1.variable_scope('pearson'):
    if len(x.shape) == 1:
      x = tf.reshape(x, (1, -1), name='pearson_x_reshape')
    if len(y.shape) == 1:
      y = tf.reshape(y, (1, -1), name='pearson_y_reshape')
    xy_t = tf.concat([tf.cast(x, tf.float32),
                      tf.cast(y, tf.float32)], axis=0)
    mean_t = tf.reduce_mean(xy_t, axis=1, keepdims=True)
    dsize = tf.cast(tf.shape(x)[1], tf.float32)
    cov_t = tf.matmul(xy_t - mean_t,
                      t(xy_t - mean_t) / (dsize - 1))
    cov2_t = tf.linalg.tensor_diag(1/tf.sqrt(tf.linalg.tensor_diag_part(cov_t)))
    cor = tf.matmul(cov2_t, tf.matmul(cov_t, cov2_t))
  return cor
