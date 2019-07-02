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

"""Tests for decoding the CCA code.

Use two sets (x1 and x2) of random data, but copy one column from x1 to x2 so
there is a hard dependency, and another column of x2 is partially dependent on
x1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from telluride_decoding import cca
from telluride_decoding import decoding

import numpy as np
import tensorflow as tf


class CcaTest(absltest.TestCase):

  def create_test_data(self, num_frames=5000, num_c1=3, num_c2=5, frac=0.5):
    x1 = np.random.randn(num_frames, num_c1).astype(np.float32)
    x2 = np.random.randn(num_frames, num_c2).astype(np.float32)
    # Fourth channel of H2 is equal to first channel of H1
    x2[:, 4] = x1[:, 0]
    # 3rd channel of H2 is a noisy version of 2nd channel H1
    x2[:, 2] = frac*x2[:, 2] + (1-frac)*x1[:, 1]
    return x1, x2

  def test_cca_calculation(self):
    x1, x2 = self.create_test_data()
    dim = 4

    brain_dataset = decoding.BrainData('input', 'output',
                                       repeat_count=1,
                                       final_batch_size=1024)
    brain_dataset.preserve_test_data(x1, x2)
    _, test_dataset = brain_dataset.create_dataset(mode='test',
                                                   temporal_context=False)
    (a, b, mean_x, mean_y, e) = cca.calculate_cca_from_dataset(test_dataset,
                                                               dim)
    print('CCA A Results:', a)
    print('CCA B Results:', b)
    print('CCA B mean x:', mean_x)
    print('CCA B mean y:', mean_y)
    print('CCA Eigenvalues:', e)
    expected_a = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    # Skip 3rd column of b since it's all noise (eigenvalue is very small.)
    expected_b = np.array([[0, 0],
                           [0, 0],
                           [0, 1],
                           [0, 0],
                           [1, 0]])
    np.testing.assert_array_less(0.9, np.abs(a[np.where(expected_a)]))
    np.testing.assert_array_less(
        np.abs(a[np.where(np.logical_not(expected_a))]), 0.1)

    b = b[:, 0:2]
    np.testing.assert_array_less(0.9, np.abs(b[np.where(expected_b)]))
    np.testing.assert_array_less(
        np.abs(b[np.where(np.logical_not(expected_b))]), 0.1)

    self.assertGreater(e[0], 0.90)
    self.assertGreater(e[1], 0.55)
    self.assertLess(e[2], 0.05)

  def test_cca_loss(self):
    dim = 4

    # Test with two completely shared dimensions
    x, y = self.create_test_data(frac=0.0)
    with tf.Session() as sess:
      tf_x = tf.Variable(initial_value=x, name='x_variable')
      tf_y = tf.Variable(initial_value=y, name='y_variable')
      sess.run(tf.global_variables_initializer())
      cca_loss_node = cca.cca_loss(tf_x, tf_y, dim, 1e-4, 1e-2)
      loss = sess.run(cca_loss_node)
      print('test_cca_loss is:', loss)
      self.assertAlmostEqual(loss, 2.05, delta=0.1)

    # Test with 1.5 shared dimensions (one full, one half dimension)
    x, y = self.create_test_data(frac=0.5)
    with tf.Session() as sess:
      tf_x = tf.Variable(initial_value=x, name='x_variable')
      tf_y = tf.Variable(initial_value=y, name='y_variable')
      sess.run(tf.global_variables_initializer())
      cca_loss_node = cca.cca_loss(tf_x, tf_y, dim, 1e-4, 1e-2)
      loss = sess.run(cca_loss_node)
      print('test_cca_loss is:', loss)
      self.assertAlmostEqual(loss, 1.72, delta=0.1)

    # Test with just one shared dimension
    x, y = self.create_test_data(frac=1.0)
    with tf.Session() as sess:
      tf_x = tf.Variable(initial_value=x, name='x_variable')
      tf_y = tf.Variable(initial_value=y, name='y_variable')
      sess.run(tf.global_variables_initializer())
      cca_loss_node = cca.cca_loss(tf_x, tf_y, dim, 1e-4, 1e-2)
      loss = sess.run(cca_loss_node)
      print('test_cca_loss is:', loss)
      self.assertAlmostEqual(loss, 1.0, delta=0.1)

  def test_cca_estimator(self):
    # First test the basic linear regressor parameters using fixed weight and
    # bias vectors.
    x, y = self.create_test_data()
    dims = 3
    dataset = tf.data.Dataset.from_tensor_slices(({'x': x}, y))
    dataset = dataset.batch(100).repeat(count=1)
    rot_x, rot_y, _, _, e = cca.calculate_cca_from_dataset(dataset, dims)
    print('Estimated rot_x:', rot_x)
    print('Estimated rot_y:', rot_y)
    print('Estimated CCA dimensions:', e)
    des_rot_x = [[1, 0], [0, 1], [0, 0]]
    np.testing.assert_allclose(des_rot_x, np.abs(rot_x[:, 0:2]), atol=0.15)
    des_rot_y = [[0, 0], [0, 0], [0, 1.3], [0, 0], [1, 0]]
    np.testing.assert_allclose(des_rot_y, np.abs(rot_y[:, 0:2]), atol=0.15)

    # Create a TF estimator, initialized with the test dataset above, and make
    # sure it is properly initialized.
    estimator = cca.create_cca_estimator(dataset, dimensions=dims)
    # Need to run the model at least one step in order to initialize it, and
    # get checkpoints, etc.  But this leads to a warning:
    #    The graph of the iterator is different from the graph the dataset was
    #    created in.
    def my_input_fn(dataset):
      iterator = dataset.make_one_shot_iterator()
      batch_features, batch_labels = iterator.get_next()
      return batch_features, batch_labels
    estimator.train(input_fn=lambda: my_input_fn(dataset), steps=1)

    saved_rot_x = estimator.get_variable_value('cca/a')
    saved_rot_y = estimator.get_variable_value('cca/a')
    self.assertTrue(np.allclose(saved_rot_x, rot_x, atol=1e-05))
    self.assertTrue(np.allclose(saved_rot_y, rot_x, atol=1e-05))

    # Now test with a new evaluation dataset
    x, y = self.create_test_data()
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {'x': x}, y, batch_size=256, num_epochs=1, shuffle=False)

    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print('CCA Estimator test returned these metrics:', eval_metrics)
    self.assertLess(eval_metrics['loss'], 1e-5)

if __name__ == '__main__':
  absltest.main()
