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

"""Utilities to save results to a CSV file.

Each row in the CSV file represents results for one regularization value.
The first column in each row is the regularization value, the rest of the
columns are correlation numbers for the experiments.
"""

import collections
import csv
import os
import numpy as np

from telluride_decoding import plot_util


def write_results(file_name, regularization_list, all_results):
  """"Writes results to a CSV file.

  Args:
    file_name: The name of the CSV file to write the results.
    regularization_list: A list of the regularization values.
    all_results: The correlation results as a 2D array. This results is
      generated by regression.py. The first dimension is for each
      regularization value, the second dimension is for each tf record file used
      for testing.
  """
  if len(regularization_list) != len(all_results):
    raise ValueError('Length of regularization list and results do no match.')
  base_dir = os.path.split(file_name)[0]
  if base_dir and not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  with tf.io.gfile.GFile(file_name, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for i, regularization in enumerate(regularization_list):
      row = [str(regularization),]
      row.extend([str(value) for value in all_results[i]])
      csv_writer.writerow(row)


def _read_results(file_name, skip_header=False):
  """"Reads results from a CSV file.

  Args:
    file_name: The name of the CSV file to read the results.
    skip_header: Skip the first line when it is a header.

  Returns:
    An ordered dictionary with regularization values as the keys and the
      correlation results as the values.
  """
  results = collections.OrderedDict()
  with tf.io.gfile.GFile(file_name, 'r') as csv_file:
    content = list(csv.reader(csv_file))
    if skip_header:
      del content[0]
    for row in content:
      if len(row) < 2:
        raise ValueError('Row %s does not have enough columns.' % row)
      regularization_value = row[0]
      correlations = row[1:]
      results[float(regularization_value)] = [float(c) for c in correlations]
  return results


def read_all_results_from_directory(dir_name, skip_header=False, pattern=''):
  """Reads results from all the CSV files in a directory.

  Args:
    dir_name: A name of the directory with all the CSV files.
    skip_header: Skip the first line when it is a header.
    pattern: Substring that must be in the files to read.

  Returns:
    An ordered dictionary with regularization values as the keys and the
      correlation results as the values.
  """
  all_results = collections.OrderedDict()
  file_names = tf.io.gfile.listdir(dir_name)
  for name in file_names:
    if not name.endswith('csv') or pattern not in name:
      continue
    curr_name = os.path.join(dir_name, name)
    curr_results = _read_results(curr_name, skip_header)
    if not all_results:
      all_results = curr_results
      continue
    if all_results.keys() != curr_results.keys():
      raise ValueError(
          'Files do not have the same regularization values %s vs %s' %
          (all_results.keys(), curr_results.keys()))
    for regularization_value, correlations in curr_results.items():
      all_results[regularization_value].extend(correlations)
  return all_results


def plot_csv_results(test_name,
                     results,
                     golden_mean_std_dict=None,
                     png_file_name=None,
                     show_plot=False):
  """Calculates the mean and standard deviation from the results and plot them.

  Args:
    test_name: The name of the test that will show in the title of the plot.
    results: An ordered dictionary with regularization values as the keys and
      the correlation results as the values.
    golden_mean_std_dict: The golden results as an ordered dictionary with the
      regularization values as the keys and tuples with mean value and standard
      deviations as as the values.
    png_file_name: If file name is not empty, save the plot to the PNG file.
    show_plot: If true, show the plot in a window.
  """
  regularization_list = []
  mean_list = []
  std_list = []
  for regularization_value in results.keys():
    regularization_list.append(regularization_value)
    correlations = results[regularization_value]
    mean_list.append(np.mean(correlations))
    std_list.append(np.std(correlations))
  plot_util.plot_mean_std(
      test_name,
      regularization_list,
      mean_list,
      std_list,
      golden_mean_std_dict=golden_mean_std_dict,
      png_file_name=png_file_name,
      show_plot=show_plot)
