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

"""Utility routines when decoding brain signals.

More to come.. the CCA functions need to get the Pearson correlation too.
"""

import tensorflow.compat.v2 as tf

# TODO Check to see if we can use (might need to get into core)
# contrib/streaming_pearson_correlation


def pearson_correlation_graph(x, y):
  """Use TF to compute the Pearson correlation between two sets of data.

  This works between two (multi-dimensional) feature arrays.  Like np.corrcoef,
  x and y are 1-D or 2-D arrays containing multiple variables and observations.
  Each column of x represents a variable, and each row a single observation of
  all those variables.
  From:
  https://stackoverflow.com/questions/43834830/tensorflow-equivalent-of-np-corrcoef-on-a-specific-axis

  Args:
    x: First multidimensional array, n_observations x n_vars
    y: Second multidimensional array, n_observations x n_vars

  Returns:
    n_vars dimensional vector with the correlation of each pair of columns.
  """
  def t(x):
    return tf.transpose(x)
  with tf.compat.v1.variable_scope('pearson'):
    if len(x.shape) == 1:
      x = tf.reshape(x, (-1, 1), name='pearson_x_reshape')
    if len(y.shape) == 1:
      y = tf.reshape(y, (-1, 1), name='pearson_y_reshape')
    xy_t = tf.concat([tf.cast(x, tf.float32),
                      tf.cast(y, tf.float32)], axis=1)
    mean_t = tf.reduce_mean(xy_t, axis=0, keepdims=True)
    dsize = tf.cast(tf.shape(x)[0], tf.float32)
    cov_t = tf.matmul(t(xy_t - mean_t),
                      (xy_t - mean_t) / (dsize - 1))
    cov2_t = tf.linalg.tensor_diag(1/tf.sqrt(tf.linalg.tensor_diag_part(cov_t)))
    cor = tf.matmul(cov2_t, tf.matmul(cov_t, cov2_t))
  return cor
