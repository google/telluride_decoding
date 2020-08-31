# Lint as: python2 python3
"""Test for telluride_decoding.scaled_lda."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest

import matplotlib.pyplot as plt
import numpy as np

from telluride_decoding import scaled_lda


class ScaledLdaTest(absltest.TestCase):

  def test_one_dimensional_data(self):
    num_points = 1000
    d1 = np.random.randn(num_points,) - 5
    d2 = np.random.randn(num_points,) + 5

    lda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    lda.fit_two_classes(d1, d2)

    d1_transformed = lda.transform(d1)
    self.assertAlmostEqual(np.mean(d1_transformed), 0)

    d2_transformed = lda.transform(d2)
    self.assertAlmostEqual(np.mean(d2_transformed), 1)

  def test_two_dimensional_data(self):
    num_points = 1000
    num_dims = 2
    d1 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [-2, 1]
    d2 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [2, -1]

    # Plot the original data.
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(d1[:, 0], d1[:, 1], 'rx')
    plt.plot(d2[:, 0], d2[:, 1], 'bo')
    plt.title('Original Data')

    x = np.concatenate((d1, d2), axis=0)
    y = np.concatenate((np.ones(d1.shape[0])*42,
                        np.ones(d2.shape[0])*-12))

    lda = scaled_lda.LinearDiscriminantAnalysis()

    with self.assertRaisesRegex(
        ValueError, 'Must fit the model before transforming.'):
      lda.transform(d1)

    with self.assertRaisesRegex(
        ValueError, 'Must fit the model before transforming.'):
      lda.explained_variance_ratio()

    x_lda = lda.fit_transform(x, y)

    labels = lda.labels
    self.assertLen(labels, 2)

    # Plot the transformed data.
    plt.subplot(2, 1, 2)
    plt.plot(x_lda[y == labels[0], 0], x_lda[y == labels[0], 1], 'rx')
    plt.plot(x_lda[y == labels[1], 0], x_lda[y == labels[1], 1], 'bo')
    plt.title('Transfomed Data')

    # Make sure the transformed centers are symmetric on the first (x) axis.
    mean_vectors = [np.reshape(v, (1, -1)) for v in lda.mean_vectors]
    centers = lda.transform(np.concatenate(mean_vectors, axis=0))
    print('Transformed centers are:', centers)
    self.assertAlmostEqual(centers[0, 0], -centers[1, 0], delta=0.1)
    np.testing.assert_allclose(centers[:, 1], [0., 0.], atol=0.1)

    plt.savefig(os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             'scaled_lda.png'))

    with self.assertRaisesRegex(
        TypeError, 'Inconsistent training and transform sizes'):
      lda.transform(d1[:, 0:1])

    # Now test model from saved parameters
    nlda = scaled_lda.LinearDiscriminantAnalysis()
    nlda.model_parameters = lda.model_parameters   # Get/set parameters test
    centers = nlda.transform(np.concatenate(mean_vectors, axis=0))
    self.assertAlmostEqual(centers[0, 0], -centers[1, 0], delta=0.1)
    np.testing.assert_allclose(centers[:, 1], [0., 0.], atol=0.1)

  def test_fitted_data(self):
    """Makes sure we can generate a fitted model with .from_fitted_data.
    """
    num_points = 1000
    num_dims = 2
    d1 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [-2, 1]
    d2 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [2, -1]
    x = np.concatenate((d1, d2), axis=0)
    y = np.concatenate((np.ones(d1.shape[0])*42,
                        np.ones(d2.shape[0])*-12))

    lda = scaled_lda.LinearDiscriminantAnalysis.from_fitted_data(x, y)

    explained = lda.explained_variance_ratio()
    np.testing.assert_allclose(explained, [1., 0.], atol=1e-8)

  def test_three_class_data(self):
    num_points = 1000
    num_dims = 2
    d1 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [-2, 1]
    d2 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]]) + [2, -1]
    d3 = np.matmul(np.random.randn(num_points, num_dims),
                   [[2, 0], [0, 0.5]])

    x = np.concatenate((d1, d2, d3), axis=0)
    y = np.concatenate((np.ones(d1.shape[0])*42,
                        np.ones(d2.shape[0])*-12,
                        np.ones(d3.shape[0])))

    lda = scaled_lda.LinearDiscriminantAnalysis()

    x_lda = lda.fit_transform(x, y)
    self.assertEqual(x_lda.shape[0], 3*num_points)
    self.assertEqual(x_lda.shape[1], 2)  # Only two dimensional data.

    labels = lda.labels
    self.assertLen(labels, 3)

  def test_four_dimensional_data(self):
    num_points = 1000
    num_dims = 4
    center = np.array([-2, 1, 3, 2])  # Arbitrary
    m1 = np.random.randn(num_points, num_dims) + center
    m2 = np.random.randn(num_points, num_dims) + -center

    x = np.concatenate((m1, m2), axis=0)
    y = np.concatenate((np.ones(m1.shape[0])*0,
                        np.ones(m2.shape[0])*1.0))

    slda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    slda.fit_two_classes(m1, m2)
    m_lda = slda.transform(x)
    self.assertEqual(m_lda.shape, (2*num_points, 2))

    self.assertEqual(slda.coef_array.shape[0], num_dims)
    self.assertLen(slda.labels, slda.coef_array.shape[1])

    mean_vectors = [np.reshape(v, (1, -1)) for v in slda.mean_vectors]
    centers = slda.transform(np.concatenate(mean_vectors, axis=0))[:, 0]
    np.testing.assert_allclose(centers, [0., 1.0], atol=1e-8)

    explained = slda.explained_variance_ratio()
    np.testing.assert_allclose(explained, [1., 0., 0., 0.], atol=1e-8)

    # Now test save and restoring parameters.
    param_dict = slda.model_parameters
    nlda = scaled_lda.ScaledLinearDiscriminantAnalysis()
    nlda.model_parameters = param_dict

    mean_vectors = [np.reshape(v, (1, -1)) for v in nlda.mean_vectors]
    centers = nlda.transform(np.concatenate(mean_vectors, axis=0))[:, 0]
    np.testing.assert_allclose(centers, [0., 1.0], atol=1e-8)

    # Make sure we fail with more than two classes.
    with self.assertRaisesRegex(
        ValueError, 'Scaled LDA can only be done on two-class data.'):
      y[0:2] = 42
      slda.fit_transform(x, y)

if __name__ == '__main__':
  absltest.main()
