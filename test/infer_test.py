# Lint as: python2 python3
"""Test for telluride_decoding.infer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from absl.testing import absltest

import matplotlib.pyplot as plt
import mock
import numpy as np

from telluride_decoding import brain_data
from telluride_decoding import brain_model
from telluride_decoding import decoding
from telluride_decoding import infer

import tensorflow.compat.v2 as tf
# End user should call tf.compat.v1.enable_v2_behavior()


class InferTest(absltest.TestCase):

  def setUp(self):
    super(InferTest, self).setUp()
    self._test_data_dir = os.path.join(
        flags.FLAGS.test_srcdir, '__main__',
        'test_data/',
        'meg')

  def test_calculate_time_axis(self):
    centers = infer.calculate_time_axis(5, 1, 2, 1)*60   # Convert to seconds
    centers = infer.calculate_time_axis(np.arange(5), 1, 2, 1)*60
    np.testing.assert_array_almost_equal(centers, [1, 2, 3, 4, 5])

    with self.assertRaisesRegex(TypeError,
                                'Unknown type passed as input argument.'):
      infer.calculate_time_axis('hello', 1, 2, 1)

  def test_first_segement(self):
    pattern = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    self.assertEqual(infer.find_first_segment(pattern), 5)
    self.assertEqual(infer.find_first_segment(np.logical_not(pattern)), 5)
    self.assertEqual(infer.find_first_segment(pattern[0:3]), 0)

    with self.assertRaisesRegex(TypeError, 'Labels input must be an ndarray'):
      infer.find_first_segment(True)

    with self.assertRaisesRegex(TypeError,
                                'Labels input must be one-dimensional'):
      infer.find_first_segment(np.array(((1, 2), (3, 4))))

  def test_get_brain_data(self):
    train_files = 'subj01'
    test_files = 'subj02'
    tf_dir = self._test_data_dir
    params = {'input_field': 'meg',
              'pre_context': 0,
              'post_context': 0,
              'input2_pre_context': 0,
              'input2_post_context': 0,
             }
    audio_label = 'envelope'
    bd = infer.create_brain_data(tf_dir, train_files, test_files,
                                 params, audio_label)
    self.assertIsInstance(bd, brain_data.BrainData)

  def create_model(self):
    train_files = 'subj01'
    test_files = 'subj02'
    tf_dir = self._test_data_dir
    params = {'input_field': 'meg',
              'output_field': 'envelope',
              'pre_context': 0,
              'post_context': 0,
              'input2_pre_context': 0,
              'input2_post_context': 0,
              'attended_field': None,
             }
    audio_label = 'envelope'
    bd = infer.create_brain_data(tf_dir, train_files, test_files,
                                 params, audio_label)
    bd_train = bd.create_dataset('train')

    test_model = brain_model.BrainModelLinearRegression(bd_train)
    learning_rate = 1e-3
    my_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    test_model.compile(optimizer=my_optimizer, metrics=['mse'], loss=['mse'])
    test_model.fit(bd_train)

    # results = test_model.evaluate(bd_train)
    # print('Test results are:', results)
    test_model.add_metadata(params, dataset=bd_train)

    dprime, final_decoder = decoding.train_lda_model(bd, test_model, params)
    print('dprime after training is %g.' % dprime)

    saved_model_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                                   'linear_saved_model')
    logging.info('Writing saved model to %s', saved_model_dir)
    test_model.save(saved_model_dir)
    final_decoder.save_parameters(os.path.join(saved_model_dir,
                                               'decoder_model.json'))
    return saved_model_dir, train_files, test_files

  @mock.patch.object(plt, 'savefig')
  def test_run_reduction_test(self, mock_savefig):
    saved_model_dir, train_files, test_files = self.create_model()

    reduction = 'first'
    decoder_type = 'wta'
    plot_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp', 'plots')
    window_results = infer.run_reduction_test(
        saved_model_dir, self._test_data_dir, train_files, test_files,
        reduction, decoder_type, 'envelope', 'envelope', plot_dir=plot_dir)
    print('Reduction test results:', window_results)
    self.assertLess(window_results[10], 0.95)
    self.assertGreater(window_results[100], 0.95)
    self.assertGreater(window_results[200], 0.95)
    self.assertGreater(window_results[400], 0.95)
    self.assertGreater(window_results[700], 0.95)
    self.assertGreater(window_results[1000], 0.95)

    # For reasons I don't understand the savefig call is failing in the test so
    # let's just mock it out.
    mock_savefig.assert_called()

  @mock.patch.object(plt, 'savefig')
  def test_run_comparison(self, mock_savefig):
    model_dir, train_files, test_files = self.create_model()

    plot_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp', 'plots')
    all_results = infer.run_comparison_test(model_dir, self._test_data_dir,
                                            train_files, test_files,
                                            'envelope', 'envelope',
                                            plot_dir, reduction_list=['lda',])
    print('all_results are:', all_results)
    for reduction in ['lda',]:
      # Not doing SSD here as there is not enough data to train the model.
      for decoder in ['wta', 'stepped',]:
        window_results = all_results[reduction, decoder]
        self.assertLess(window_results[10], 0.95)
        self.assertGreater(window_results[100], 0.95)
        self.assertGreater(window_results[200], 0.95)
        self.assertGreater(window_results[400], 0.95)
        self.assertGreater(window_results[700], 0.95)
        self.assertGreater(window_results[1000], 0.95)

    # For reasons I don't understand the savefig call is failing in the test so
    # let's just mock it out.
    mock_savefig.assert_called()

    # Remove the decoder model file, which we can only check once we have a
    # otherwise completed model directory.
    tf.io.gfile.rmtree(os.path.join(model_dir, 'decoder_model.json'))
    with self.assertRaisesRegex(IOError,
                                'Can not load decoder model parameters'):
      infer.load_model(model_dir, 'lda')

  def test_for_errors(self):
    with self.assertRaisesRegex(ValueError,
                                'Couldn\'t determine model type'):
      infer.load_model('/foo/bar', 'none')

    with self.assertRaisesRegex(ValueError,
                                'Unknown reduction technique'):
      infer.load_model('/foo/cca', 'none')

    with self.assertRaisesRegex(IOError,
                                'SavedModel file does not exist'):
      infer.load_model('/foo/cca', 'lda')

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
