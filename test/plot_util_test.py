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

from absl.testing import absltest
import mock
from telluride_decoding import plot_util
import tensorflow.compat.v2 as tf


class PlotUtilTest(absltest.TestCase):

  @mock.patch.object(plot_util, 'matplotlib_pyplot')
  @mock.patch.object(tf.io.gfile, 'GFile')
  def test_plot_mean_std_save_png(self, mock_gfile, mock_pyplot):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6, 0.7]
    run_std = [0.1, 0.2, 0.3]

    plot_util.plot_mean_std(
        test_name,
        regularization_list,
        run_mean,
        run_std,
        png_file_name='test.png')

    mock_plt = mock_pyplot.return_value
    mock_plt.figure.assert_called_once_with()
    mock_plt.errorbar.assert_called_once_with(
        regularization_list, run_mean, run_std, color='blue', label='actual')
    mock_plt.xscale.assert_called_once_with('log')
    mock_plt.xlabel.assert_called_once_with('Regularization lambda (log10)')
    mock_plt.ylabel.assert_called_once_with('Mean correlation')
    mock_plt.title.assert_called_once_with('test experiment correlation')
    mock_gfile.assert_called_once_with('test.png', 'wb')
    mock_plt.savefig.assert_called_once_with(
        mock_gfile.return_value.__enter__.return_value, format='png')

  @mock.patch.object(plot_util, 'matplotlib_pyplot')
  @mock.patch.object(tf.io.gfile, 'GFile')
  def test_plot_mean_std_save_png_with_golden_mean_std(self, mock_gfile,
                                                       mock_pyplot):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6, 0.7]
    run_std = [0.1, 0.2, 0.3]
    golden_mean_std_dict = {1e-6: (1.2, 2.3, 0.1, 0.1),
                            1e-3: (2.4, 3.5, 0.2, 0.2),
                            1: (3.7, 4.8, 0.3, 0.3)}

    plot_util.plot_mean_std(
        test_name,
        regularization_list,
        run_mean,
        run_std,
        golden_mean_std_dict=golden_mean_std_dict,
        png_file_name='test.png')

    mock_plt = mock_pyplot.return_value
    mock_plt.figure.assert_called_once_with()
    calls = mock_plt.errorbar.call_args_list
    args, kwargs = calls[0]
    self.assertEqual(args, ([1e-6, 1e-3, 1], [1.2, 2.4, 3.7], [2.3, 3.5, 4.8]))
    self.assertEqual(kwargs, {
        'color': 'orange',
        'label': 'golden',
        'lolims': True,
        'uplims': True
    })
    args, kwargs = calls[1]
    self.assertEqual(args, ([1e-6, 1e-3, 1], [0.5, 0.6, 0.7], [0.1, 0.2, 0.3]))
    self.assertEqual(kwargs, {
        'color': 'blue',
        'label': 'actual',
    })
    mock_plt.xscale.assert_called_once_with('log')
    mock_plt.xlabel.assert_called_once_with('Regularization lambda (log10)')
    mock_plt.ylabel.assert_called_once_with('Mean correlation')
    mock_plt.title.assert_called_once_with('test experiment correlation')
    mock_gfile.assert_called_once_with('test.png', 'wb')
    mock_plt.savefig.assert_called_once_with(
        mock_gfile.return_value.__enter__.return_value, format='png')

  @mock.patch.object(plot_util, 'matplotlib_pyplot')
  def test_plot_mean_std_show_plot(self, mock_pyplot):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6, 0.7]
    run_std = [0.1, 0.2, 0.3]

    plot_util.plot_mean_std(
        test_name, regularization_list, run_mean, run_std, show_plot=True)

    mock_plt = mock_pyplot.return_value
    mock_plt.figure.assert_called_once_with()
    mock_plt.errorbar.assert_called_once_with(
        regularization_list, run_mean, run_std, color='blue', label='actual')
    mock_plt.xscale.assert_called_once_with('log')
    mock_plt.xlabel.assert_called_once_with('Regularization lambda (log10)')
    mock_plt.ylabel.assert_called_once_with('Mean correlation')
    mock_plt.title.assert_called_once_with('test experiment correlation')
    mock_plt.save_fig.assert_not_called()
    mock_plt.show.assert_called_once_with()

  def test_plot_mean_std_no_png_and_show_plot(self):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6, 0.7]
    run_std = [0.1, 0.2, 0.3]

    with self.assertRaises(TypeError):
      plot_util.plot_mean_std(test_name, regularization_list, run_mean, run_std)

  def test_plot_mean_std_length_mean_mismatch(self):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6]
    run_std = [0.1, 0.2, 0.3]

    with self.assertRaises(TypeError):
      plot_util.plot_mean_std(
          test_name, regularization_list, run_mean, run_std, show_plot=True)

  def test_plot_mean_std_length_std_mismatch(self):
    test_name = 'test'
    regularization_list = [1e-6, 1e-3, 1]
    run_mean = [0.5, 0.6, 0.7]
    run_std = [0.1, 0.2]

    with self.assertRaises(TypeError):
      plot_util.plot_mean_std(
          test_name, regularization_list, run_mean, run_std, show_plot=True)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
