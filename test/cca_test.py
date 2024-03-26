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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from telluride_decoding import brain_data
from telluride_decoding import cca
import tensorflow as tf


flags.DEFINE_bool('random_mixup_batch',
                  False,
                  'Mixup the data, so labels are random, for testing.')
flags.DEFINE_string('summary_dir', '/tmp/tf',
                    'Location of event logs and checkpoints.')


class CcaTest(parameterized.TestCase):

  def create_test_data(self, num_frames=5000, num_c1=3, num_c2=5, frac=0.5):
    """Create two arrays that can be used for testing CCA algorithms."""
    x1 = np.random.randn(num_frames, num_c1).astype(np.float32)
    x2 = np.random.randn(num_frames, num_c2).astype(np.float32)
    # Fourth channel of H2 is equal to first channel of H1
    x2[:, 4] = x1[:, 0]
    # 3rd channel of H2 is a noisy version of 2nd channel H1
    x2[:, 2] = frac*x2[:, 2] + (1-frac)*x1[:, 1]
    return x1, x2

  def test_brain_data_segmentation(self):
    """Make sure we can retrieve the two sets of parallel data."""
    total_frames = 5000
    final_batch_size = 1024
    frame_rate = 100.0  # Arbitrary
    x1 = np.cumsum(np.ones((total_frames, 1), dtype=np.int32), axis=0)
    x2 = np.cumsum(np.ones((total_frames, 1), dtype=np.int32), axis=0)
    print('Raw x1:', x1[0:10, :])
    print('Raw x2:', x2[0:10, :])

    bd = brain_data.TestBrainData('input_1', 'input_2', frame_rate,
                                  repeat_count=1,
                                  final_batch_size=final_batch_size)
    bd.preserve_test_data(x1,
                          np.ones((x1.shape[0], 1)),
                          input2_data=x2)
    test_dataset = bd.create_dataset(mode='program_test',
                                     temporal_context=False)
    batch_num = 0
    for (x_dict, y) in test_dataset:
      x = x_dict['input_1'].numpy()
      y = x_dict['input_2'].numpy()
      np.testing.assert_array_equal(x, x1[batch_num*final_batch_size:
                                          (batch_num+1)*final_batch_size, :])
      np.testing.assert_array_equal(y, x2[batch_num*final_batch_size:
                                          (batch_num+1)*final_batch_size, :])
      batch_num += 1

  def test_cca_calculation(self):
    np.random.seed(42)
    x1, x2 = self.create_test_data()
    dim = 4
    frame_rate = 100.0  # Arbitrary

    bd = brain_data.TestBrainData('input_1', 'input_2', frame_rate,
                                  repeat_count=1,
                                  final_batch_size=1024)
    bd.preserve_test_data(x1,
                          np.ones((x1.shape[0], 1)),
                          input2_data=x2)
    test_dataset = bd.create_dataset(mode='program_test',
                                     temporal_context=False)
    (a, b, mean_x, mean_y, e) = cca.calculate_cca_parameters_from_dataset(
        test_dataset, dim, mini_batch_count=1000)
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
    # Check diagonal of expected_a to make sure a has big values
    np.testing.assert_array_less(0.9, np.abs(a[np.where(expected_a)]))
    # Check off-diagonal of expected_a to make sure a has small values
    np.testing.assert_array_less(
        np.abs(a[np.where(np.logical_not(expected_a))]), 0.05)

    b = b[:, 0:2]
    np.testing.assert_array_less(0.9, np.abs(b[np.where(expected_b)]))
    np.testing.assert_array_less(
        np.abs(b[np.where(np.logical_not(expected_b))]), 0.1)

    self.assertGreater(e[0], 0.90)
    self.assertGreater(e[1], 0.60)
    self.assertLess(e[2], 0.02)

  def test_cca_loss(self):
    dim = 4

    # Test with two completely shared dimensions
    x, y = self.create_test_data(frac=0.0)
    loss = cca.cca_loss(x, y, dim, 1e-4, 1e-2)
    print('New test_cca_loss is:', loss)
    self.assertAlmostEqual(loss, 2.05, delta=0.1)

    # Test with 1.5 shared dimensions (one full, one half dimension)
    x, y = self.create_test_data(frac=0.5)
    loss = cca.cca_loss(x, y, dim, 1e-4, 1e-2)
    print('New test_cca_loss is:', loss)
    self.assertAlmostEqual(loss, 1.72, delta=0.1)

    # Test with just one shared dimension
    x, y = self.create_test_data(frac=1.0)
    loss = cca.cca_loss(x, y, dim, 1e-4, 1e-2)
    print('New test_cca_loss is:', loss)
    self.assertAlmostEqual(loss, 1.0, delta=0.1)

  def test_cca_brain_model(self):
    """Test simple (no temporal shift) CCA."""
    print('\n\n**********test_cca_brain_model starting... *******')
    x1, x2 = self.create_test_data()
    cca_dims = 3
    batch_size = 1024
    frame_rate = 100.0  # Arbitrary

    brain_dataset = brain_data.TestBrainData('input_1', 'output_1', frame_rate,
                                             in2_fields='input_2',
                                             repeat_count=1,
                                             final_batch_size=batch_size)
    brain_dataset.preserve_test_data(x1, np.ones((x1.shape[0], 1),
                                                 dtype=np.float32),
                                     input2_data=x2)
    test_dataset = brain_dataset.create_dataset(mode='test',
                                                temporal_context=False)
    print('*** Now create the model ***')
    bm_cca = cca.BrainModelCCA(test_dataset, cca_dims=cca_dims)
    for x, _ in test_dataset.take(1):
      cca_input = x

    bm_cca.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                   metrics=[cca.cca_pearson_correlation_first,],
                   loss=['mse'])

    result = bm_cca(cca_input)   # Be sure to call it once so network gets built
    self.assertEqual(result.shape[0], batch_size)
    self.assertEqual(result.shape[1], 2*cca_dims)

    print('*** Now train the model ***')
    bm_cca.fit(test_dataset)
    bm_cca.summary()

    # Drop the labels when calling the model to infer output.
    just_input_dataset = test_dataset.map(lambda x, y: x)

    print('*** Run some data through the model ***')
    # for input_data in just_input_dataset.take(1):
    for input_data in just_input_dataset.take(1):
      results = bm_cca(input_data)
      self.assertEqual(results.shape[0], batch_size)
      self.assertEqual(results.shape[1], 2*cca_dims)

    print('***Now get ready to evaluate the model ***')
    print('In test_cca_brain_model the test dataset is:', test_dataset)
    metrics = bm_cca.evaluate(test_dataset)
    print('Evaluate produced the final metrics:', metrics)
    self.assertAlmostEqual(metrics['cca_pearson_correlation_first'], 1.,
                           delta=.01)

  # TODO: Add shifts for temporal shifts.

  @parameterized.named_parameters(('dim1', 1, 4),
                                  ('dim2', 4, 1))
  def test_cca_brain_model_low_dim_error(self, dim1, dim2):
    """Test features with too small dimensions."""
    num_points = 5000
    x1 = np.random.randn(num_points, dim1)
    x2 = np.random.randn(num_points, dim2)
    cca_dims = 3
    batch_size = 1024
    frame_rate = 100.0  # Arbitrary

    brain_dataset = brain_data.TestBrainData('input_1', 'output_1', frame_rate,
                                             in2_fields='input_2',
                                             repeat_count=1,
                                             final_batch_size=batch_size)
    brain_dataset.preserve_test_data(x1, np.ones((x1.shape[0], 1),
                                                 dtype=np.float32),
                                     input2_data=x2)
    test_dataset = brain_dataset.create_dataset(mode='test',
                                                temporal_context=False)
    with self.assertRaises(ValueError):
      cca.BrainModelCCA(test_dataset, cca_dims=cca_dims)

  def test_save_model(self):
    """Test save and restore CCA model."""
    x1, x2 = self.create_test_data()
    cca_dims = 3
    batch_size = 1024
    frame_rate = 100.0  # Arbitrary

    brain_dataset = brain_data.TestBrainData('input_1', 'output_1', frame_rate,
                                             in2_fields='input_2',
                                             repeat_count=1,
                                             final_batch_size=batch_size)
    brain_dataset.preserve_test_data(x1, np.ones((x1.shape[0], 1),
                                                 dtype=np.float32),
                                     input2_data=x2)
    test_dataset = brain_dataset.create_dataset(mode='test',
                                                temporal_context=False)

    print('*** Now create the model ***')
    bm_cca = cca.BrainModelCCA(test_dataset, cca_dims=cca_dims)
    bm_cca.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                   metrics=[cca.cca_pearson_correlation_first,],
                   loss=['mse'])

    for x, _ in test_dataset.take(1):
      cca_input = x

    result = bm_cca(cca_input)   # Be sure call it once so network gets built
    self.assertEqual(result.shape[0], batch_size)
    self.assertEqual(result.shape[1], 2*cca_dims)

    print('*** Now train the model ***')
    bm_cca.fit(test_dataset)
    bm_cca.summary()

    print('*** Save some predictions for later ***')
    for x, _ in test_dataset.take(1):
      true_x = x
      predict_y = bm_cca.predict(true_x)

    print('*** Now save the model ***')
    saved_model_file = '/tmp/tf_cca_saved_model.tf'
    bm_cca.save(saved_model_file, save_format='tf')
    self.assertTrue(tf.io.gfile.exists(saved_model_file))

    print('*** Now load and test the model ***')
    # Custom_objects parameter in the load_model call doesn't seem to work, but
    # with ....CustomObjectScope does.
    with tf.keras.utils.CustomObjectScope(
        {'BrainCcaLayer': cca.BrainCcaLayer,
         'cca_pearson_correlation_first': cca.cca_pearson_correlation_first,
        }):
      new_model = tf.keras.models.load_model(saved_model_file)
    new_predictions = new_model.predict(true_x)
    np.testing.assert_allclose(predict_y, new_predictions, rtol=1e-6, atol=1e-6)

    total_cor = cca.cca_pearson_correlation_first(None, new_predictions)
    print('test_save_model -- Final correlations are:', total_cor)
    self.assertAlmostEqual(total_cor.numpy(), 1., delta=0.01)


if __name__ == '__main__':
  absltest.main()
