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

# Lint as: python2 python3

import collections
import os
from absl import flags
from absl.testing import absltest
import mock
import numpy as np

from telluride_decoding import csv_util
from telluride_decoding import plot_util


class CsvUtilTest(absltest.TestCase):

  def setUp(self):
    super(CsvUtilTest, self).setUp()
    self._test_data_dir = os.path.join(
        flags.FLAGS.test_srcdir, '__main__',
        'test_data/')

  def test_write_results(self):
    temp_dir = self.create_tempdir().full_path
    csv_file_name = os.path.join(temp_dir, 'x/y', 'test.csv')
    regularization_list = [1e-6, 1e-3, 1]
    all_results = np.array([[1.1, 1.2, 2.3, 2.4],
                            [3.5, 3.6, 4.7, 4.8],
                            [5.9, 5.1, 6.2, 6.3]])

    csv_util.write_results(csv_file_name, regularization_list, all_results)

    expected_content = """\
1e-06,1.1,1.2,2.3,2.4
0.001,3.5,3.6,4.7,4.8
1,5.9,5.1,6.2,6.3
"""
    with open(csv_file_name, 'r') as csv_file:
      content = csv_file.read()
      self.assertEqual(content, expected_content)

  def test_write_results_mismatch_size(self):
    temp_dir = self.create_tempdir().full_path
    csv_file_name = os.path.join(temp_dir, 'x/y', 'test.csv')
    regularization_list = [1e-6, 1e-3]
    all_results = np.array([[[1.1, 1.2], [2.3, 2.4]], [[3.5, 3.6], [4.7, 4.8]],
                            [[5.9, 5.1], [6.2, 6.3]]])

    with self.assertRaises(ValueError):
      csv_util.write_results(csv_file_name, regularization_list, all_results)

  def test_read_results_from_directory(self):
    dir_name = os.path.join(self._test_data_dir, 'csv_results')

    results = csv_util.read_all_results_from_directory(dir_name)

    self.assertDictEqual(
        results, {
            1e-6: [1.1, 1.2, 2.3, 2.4, 4.2, 5.3],
            0.001: [3.5, 3.6, 4.7, 4.8, 6.7, 8.2],
            1.0: [5.9, 5.1, 6.2, 6.3, 9.9, 7.1],
        })

  def test_read_results_from_directory_mismatch(self):
    dir_name = os.path.join(self._test_data_dir, 'mismatch_csv_results')

    with self.assertRaises(ValueError):
      csv_util.read_all_results_from_directory(dir_name)

  @mock.patch.object(plot_util, 'plot_mean_std')
  def test_save_results_plot(self, mock_plot_mean_std):
    dir_name = os.path.join(self._test_data_dir, 'csv_results')

    results = csv_util.read_all_results_from_directory(dir_name)
    csv_util.plot_csv_results(
        'test', results, png_file_name='/tmp/test.png', show_plot=True)

    args, kwargs = mock_plot_mean_std.call_args_list[0]
    self.assertEqual(args[0], 'test')
    self.assertEqual(args[1], [1e-6, 0.001, 1.0])
    self.assertEqual(args[2], [2.75, 5.25, 6.75])
    self.assertEqual(args[3],
                     [1.5305227865013968, 1.68794747153656, 1.5272524349301266])
    self.assertEqual(kwargs['png_file_name'], '/tmp/test.png')
    self.assertTrue(kwargs['show_plot'])

  @mock.patch.object(plot_util, 'plot_mean_std')
  def test_save_results_plot_with_golden_results(self, mock_plot_mean_std):
    dir_name = os.path.join(self._test_data_dir, 'csv_results')

    results = csv_util.read_all_results_from_directory(dir_name)
    golden_mean_std_dict = collections.OrderedDict([
        (1e-06, (2.75, 1.53)),
        (0.001, (5.65, 1.79)),
        (1.0, (6.35, 1.41)),
    ])

    csv_util.plot_csv_results(
        'test',
        results,
        golden_mean_std_dict=golden_mean_std_dict,
        png_file_name='/tmp/test.png',
        show_plot=True)

    args, kwargs = mock_plot_mean_std.call_args_list[0]
    self.assertEqual(args[0], 'test')
    self.assertEqual(args[1], [1e-6, 0.001, 1.0])
    self.assertEqual(args[2], [2.75, 5.25, 6.75])
    self.assertEqual(args[3],
                     [1.5305227865013968, 1.68794747153656, 1.5272524349301266])
    self.assertEqual(kwargs['golden_mean_std_dict'], {
        1e-6: (2.75, 1.53),
        0.001: (5.65, 1.79),
        1.0: (6.35, 1.41)
    })
    self.assertEqual(kwargs['png_file_name'], '/tmp/test.png')
    self.assertTrue(kwargs['show_plot'])


if __name__ == '__main__':
  absltest.main()
