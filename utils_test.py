"""Tests for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from telluride_decoding import utils

import numpy as np
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_pearson(self):
    """Test the Pearson correlation graph.

    Use simulated data and make sure we get the result we expect.
    """
    # Example data from: http://www.statisticshowto.com/
    # .      probability-and-statistics/correlation-coefficient-formula/
    correlation_data = np.array([
        [1, 43, 99],
        [2, 21, 65],
        [3, 25, 79],
        [4, 42, 75],
        [5, 57, 87],
        [6, 59, 81]])
    # Test the correlation graph using the sample data above
    test_x = tf.constant(correlation_data[:, 1].T, dtype=tf.float32)
    test_y = tf.constant(correlation_data[:, 2].T, dtype=tf.float32)
    correlation_tensor = utils.pearson_correlation_graph(test_x, test_y)
    with tf.compat.v1.Session() as _:
      correlation_r = correlation_tensor.eval()
      print('Test correlation produced:', correlation_r)
      self.assertAlmostEqual(correlation_r[0, 0], 1.0, delta=0.0001)
      self.assertAlmostEqual(correlation_r[0, 1], 0.5298, delta=0.0001)
      self.assertAlmostEqual(correlation_r[1, 0], 0.5298, delta=0.0001)
      self.assertAlmostEqual(correlation_r[1, 1], 1.0, delta=0.0001)

  def test_pearson2(self):
    """Test to make sure TF and np.corrcoef give same answers with random data.

    This test makes sure it works with multi-dimensional features.
    """
    fsize = 2  # feature size
    dsize = 6  # sample count
    x = np.random.random((fsize, dsize))
    y = np.random.random((fsize, dsize))

    tf.InteractiveSession()

    cor = utils.pearson_correlation_graph(x, y)
    print('test_pearson2: np.corrcoef:', np.corrcoef(x, y))
    print('test_pearson2: cor.eval():', cor.eval())
    print('test_pearson2: Difference:', np.corrcoef(x, y) - cor.eval())
    self.assertTrue(np.allclose(np.corrcoef(x, y), cor.eval(), atol=2e-7))


if __name__ == '__main__':
  tf.test.main()
