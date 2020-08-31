# Lint as: python2 python3
"""Tests for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from telluride_decoding import utils

import tensorflow.compat.v2 as tf


class UtilsTest(tf.test.TestCase):

  def test_pearson(self):
    """Test the Pearson correlation graph.

    Use simulated data and make sure we get the result we expect.
    """
    # Example data from: http://www.statisticshowto.com/
    #       probability-and-statistics/correlation-coefficient-formula/
    correlation_data = np.array([
        [1, 43, 99],
        [2, 21, 65],
        [3, 25, 79],
        [4, 42, 75],
        [5, 57, 87],
        [6, 59, 81]], dtype=np.float32)
    # Test the correlation graph using the sample data above
    test_x = correlation_data[:, 1]
    test_y = correlation_data[:, 2]
    correlation_tensor = utils.pearson_correlation_graph(test_x, test_y)

    correlation_r = correlation_tensor.numpy()
    print('Test_pearson produced:', correlation_r)
    np.testing.assert_allclose(np.asarray([[1, .5298],
                                           [.5298, 1]]), correlation_r,
                               atol=0.0001)

  def test_pearson2(self):
    """Test to make sure TF and np.corrcoef give same answers with random data.

    This test makes sure it works with multi-dimensional features.
    """
    fsize = 2  # feature size
    dsize = 6  # sample count
    x = np.random.random((dsize, fsize))
    y = np.random.random((dsize, fsize))

    cor = utils.pearson_correlation_graph(x, y)
    print('test_pearson2: np.corrcoef:', np.corrcoef(x, y, rowvar=False))
    print('test_pearson2: cor.eval():', cor.numpy())
    print('test_pearson2: Difference:',
          np.corrcoef(x, y, rowvar=False) - cor.numpy())
    np.testing.assert_allclose(np.corrcoef(x, y, rowvar=False), cor.numpy(),
                               atol=2e-7)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
