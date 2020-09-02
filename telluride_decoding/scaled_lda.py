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

"""Linear discriminant analysis (LDA), for reducing a vector into a scalar.

Perform LDA, linear discriminant analysis, to find the projection that best
maps the critical distinction between classes of data.  Furthermore, the
Scaled LDA maps data from the two classes to 0 and 1, along the best dimension.
"""

import collections
from absl import logging
import numpy as np


LdaParamsTuple = collections.namedtuple(
    'LdaParamsTuple',
    ['w_real', 'w_imag', 'labels', 'mean_vectors', 'slope', 'intercept'])
# Slope and intercept are relevant only for the scaled subclass.


class LinearDiscriminantAnalysis(object):
  """Finds the best linear mapping to discriminate classes of vector data.

  General algorithm based on this article:
    https://sebastianraschka.com/Articles/2014_python_lda.html

  Attributes:
    mean_vectors: The mean vectors of each class, one per row.
    coef_array: The rotation matrix which premultiplies the data to create the
      optimal mapping.
    labels: A list of distinct class labels.
    model_parameters: A dictionary containing the model parameters.
  """

  def __init__(self):
    self._eigen_pairs = []
    self._labels = []
    self._mean_vectors = []
    self._w = None

  @property
  def mean_vectors(self):
    return self._mean_vectors

  @property
  def coef_array(self):
    return self._w

  @property
  def labels(self):
    return self._labels

  @property
  def model_parameters(self):
    """Returns the model parameters needed to implement this model.

    Returns:
      A dictionary of the parameters.
    """
    return LdaParamsTuple(
        np.real(self._w), np.imag(self._w), self._labels, self._mean_vectors,
        None, None)

  @model_parameters.setter
  def model_parameters(self, values):
    """Sets the model parameters based on saved values.

    Args:
      values: A named tuple of values returned by the model_parameters property.
    """
    self._set_parameters(values)

  # https://stackoverflow.com/questions/42763283/access-superclass-property-setter-in-subclass
  def _set_parameters(self, values):
    if values.w_real is not None:
      self._w = np.array(values.w_real) + 1j * np.array(values.w_imag)
    else:
      self._w = None
    self._labels = np.array(values.labels)
    self._mean_vectors = np.array(values.mean_vectors)

  @classmethod
  def from_fitted_data(cls, x, y):
    """Creates an object and fits it to some data.

    The calculated rotation matrix (w) is stored for later use.

    Args:
      x: The input data, a two-dimensional (num_frames x num_dims) np array.
      y: The corresponding class labels (num_frames).

    Returns:
      Class object that has been trained to fit the data.
    """
    obj = cls()
    obj.fit(x, y)
    return obj

  def expand_dims(self, data):
    """Make sure the data is two dimensional.

    Args:
      data: An np array, potentially with only one dimension.

    Returns:
      An array with two dimensions.
    """
    if data.ndim == 1:
      data = np.reshape(data, (-1, 1))
    return data

  def _compute_in_class_scatter_matrix(self, x, y):
    """Computes the in-class scatter matrix for the x data, given labels in y.

    Args:
      x: The input data, a two-dimensional (num_frames x num_dims) np array.
      y: The corresponding class labels (num_frames).

    Returns:
      An np.ndarray of size num_dims x num_dims with the covariances.
    """
    num_dims = x.shape[1]
    scatter_within = 0
    for class_index, mean_vector in enumerate(self._mean_vectors):
      for row in x[y == self._labels[class_index]]:
        row = row.reshape(num_dims, 1)
        mean_vector = mean_vector.reshape(num_dims, 1)
        scatter_within += (row - mean_vector).dot((row - mean_vector).T)
    return scatter_within

  def _compute_between_class_scatter_matrix(self, x, y):
    """Computes the between class scatter matrix for the x data.

    Args:
      x: The input data, a two-dimensional (num_frames x num_dims) np array.
      y: The corresponding class labels (num_frames).

    Returns:
      An np.ndarray of size num_dims x num_dims with the covariances between
      classes.
    """
    num_dims = x.shape[1]
    # Compute the between-class scatter matrix.
    overall_mean = np.mean(x, axis=0)

    scatter_between = 0
    for i, mean_vector in enumerate(self._mean_vectors):
      n = x[y == self._labels[i], :].shape[0]
      mean_vector = mean_vector.reshape(num_dims, 1)   # make column vector
      overall_mean = overall_mean.reshape(num_dims, 1)  # make column vector
      scatter_between += n * (mean_vector - overall_mean).dot(
          (mean_vector - overall_mean).T)
    return scatter_between

  def fit(self, x, y):
    """Creates a model that best fits the data.

    The calculated rotation matrix (w) is stored for later use.

    Args:
      x: The input data, a two-dimensional (num_frames x num_dims) np array.
      y: The corresponding class labels (num_frames).
    """
    x = self.expand_dims(x)

    self._labels = sorted(set(y))
    # Create a list by class of mean_vectors.
    self._mean_vectors = [np.mean(x[y == label], axis=0)
                          for label in self._labels]

    scatter_within = self._compute_in_class_scatter_matrix(x, y)
    scatter_between = self._compute_between_class_scatter_matrix(x, y)

    # Compute the transform.
    eigen_vals, eigen_vecs = np.linalg.eig(
        np.linalg.inv(scatter_within).dot(scatter_between))

    # Make a list of (eigenvalue, eigenvector) tuples. Eigenvalues are from a
    # list, and eigenvectors are rows from an array.
    self._eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                         for i in range(len(eigen_vals))]
    print('in Fit: eigen_pairs are:', self._eigen_pairs)
    # Sort the (eigenvalue, eigenvector) tuples from high to low.
    self._eigen_pairs = sorted(self._eigen_pairs, key=lambda k: k[0],
                               reverse=True)
    num_dims = x.shape[1]
    if len(self._eigen_pairs) > 1:
      self._w = np.hstack((self._eigen_pairs[0][1].reshape(num_dims, 1),
                           self._eigen_pairs[1][1].reshape(num_dims, 1)))
    else:
      self._w = np.array([[1,],])

  def transform(self, x):
    """Transforms some data based on the learned LDA model.

    Args:
      x:  The input data, a two-dimensional (num_frames x num_dims) np array.

    Returns:
      The scaled and rotated data which maximizes the between class covariance.
      The output is num_frames x num_eigen_dimensions.
    """
    if self._w is None:
      raise ValueError('Must fit the model before transforming.')
    x = self.expand_dims(x)
    if np.ndim(x) != 2 or self._w.shape[0] != x.shape[1]:
      raise TypeError('Inconsistent training and transform sizes. %s vs %s' %
                      (x.shape, self._w.shape))
    return np.real(x.dot(self._w))

  def fit_transform(self, x, y):
    self.fit(x, y)
    return self.transform(x)

  def explained_variance_ratio(self):
    """Calculates a vector describing the explained variance for each dimension.

    Returns:
      An array of variances, one per transformed dimension.
    """
    if self._w is None:
      raise ValueError('Must fit the model before transforming.')

    eigen_vals = np.array([val for val, _ in self._eigen_pairs])
    return eigen_vals / np.sum(eigen_vals)


class ScaledLinearDiscriminantAnalysis(LinearDiscriminantAnalysis):
  """A refinement of LDA with scaling so we can use the output as a label.

  The result of this scaling means that data associated with the small class
  label (based on sort) returns a value to 0 and the data with the larger
  class label returns the value of 1.
  """

  def __init__(self):
    self._slope = 1
    self._intercept = 0
    super(ScaledLinearDiscriminantAnalysis, self).__init__()

  @property
  def model_parameters(self):
    """Returns the model parameters needed to implement this model.

    Returns:
      A dictionary of the parameters.
    """
    values = LdaParamsTuple(
        np.real(self._w), np.imag(self._w), self._labels, self._mean_vectors,
        self._slope, self._intercept)
    return values

  @model_parameters.setter
  def model_parameters(self, values):
    """Sets the model parameters based on saved values.

    Args:
      values: A dictionary of values returned by the model_parameters property.
    """
    self._set_parameters(values)

  def _set_parameters(self, values):
    print('Values:{}'.format(values))
    values = LdaParamsTuple(*values)

    super(ScaledLinearDiscriminantAnalysis, self)._set_parameters(values)
    self._slope = values.slope
    self._intercept = values.intercept

  def fit(self, x, y, y0=0, y1=1):
    """Fits the transformed data to the given labels.

    Args:
      x: The input data, a two-dimensional (num_frames x num_dims) np array.
      y: The corresponding class labels (num_frames).
      y0: Desired output label for the first class, with the smaller label.
      y1: Desired output label for the second class, with the larger label.
    """
    x = self.expand_dims(x)
    super(ScaledLinearDiscriminantAnalysis, self).fit(x, y)
    if len(self.labels) != 2:
      raise ValueError('Scaled LDA can only be done on two-class data.')

    # Map the LDA axis into a zero-one scale so we can use it for ML.
    # Fit the **first** axis with these equations:
    #   y0 = m x0 + b
    #   y1 = m x1 + b
    # which is solved by subtracting the two equations and solving for m,
    # then b.
    x0 = self.transform(np.reshape(self.mean_vectors[0], (1, -1)))[0, 0]
    x1 = self.transform(np.reshape(self.mean_vectors[1], (1, -1)))[0, 0]
    if x0 == x1:
      # Ooops, probably a programming error.  Two classes have the same mean.
      raise ValueError('X0 and X1 are identical (%g and %g)' % (x0, x1))
    self._slope = (y0 - y1) / (x0 - x1)
    self._intercept = y0 - self._slope*x0
    logging.info('Scaled LDA slope and intercept are: %g, %g',
                 self._slope, self._intercept)

  def fit_two_classes(self, class0, class1):
    """Fit the LDA model with two classes of data.

    Args:
      class0: np.array data that after LDA should result in an output of 0.
      class1: np.array data that after LDA should result in an output of 1.
    """
    class0 = np.asarray(class0)
    class1 = np.asarray(class1)
    if class0.ndim*class1.ndim != 1 and class0.shape[1] != class1.shape[1]:
      raise ValueError('Class 0 and Class1 must have the same number of '
                       'dimensions (%s vs %s).' % (class0.shape, class1.shape))

    x = np.concatenate((class0, class1), axis=0)
    y = np.concatenate((np.ones(class0.shape[0])*0,
                        np.ones(class0.shape[0])*1))
    self.fit(x, y)

  def transform(self, x):
    """Transforms the input data, applying the LDA transform and the scaling.

    Args:
      x: a num_frames x num_dimensions array of data to transform and scale.

    Returns:
      Num_frames of scalar predictions, with the mean of the first class
      going to 0 and the second class going to 1.
    """
    x_lda = super(ScaledLinearDiscriminantAnalysis, self).transform(x)
    return np.real(self._slope*x_lda + self._intercept)
