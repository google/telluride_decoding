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
"""Utilities to plot results."""

# To prevent tkinter errors as per: https://stackoverflow.com/a/37605654
import os
import matplotlib
matplotlib.use('Agg')
import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top
# User should call tf.compat.v1.enable_v2_behavior()


def matplotlib_pyplot():
  """Imports matplotlib pyplot."""
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  return plt


def plot_mean_std(test_name,
                  regularization_list,
                  run_mean,
                  run_std,
                  golden_mean_std_dict=None,
                  png_file_name=None,
                  show_plot=False):
  """Plots the mean and standard deviation as an error bar figure.

  Args:
    test_name: The name of the test for use in plot title.
    regularization_list: A list of regularization values.
    run_mean: the mean from each regularization value, over tests
    run_std: the standard deviation from each regularization value.
    golden_mean_std_dict: The golden results as an ordered dictionary with the
      regularization values as the keys and tuples with these stats of the mean
      and standard deviation estimates:
         mean(m), means(s), std(m), std(s)
    png_file_name: If file name is not empty, save the plot to the PNG file.
    show_plot: If true, show the plot in a window.

  Raises:
    TypeError: If the input parameters are not correct.
  """
  if not png_file_name and not show_plot:
    raise TypeError('PNG file name is empty and show_plot is false.')
  if len(regularization_list) != len(run_mean):
    raise TypeError('Lengths of regularizations (%d) and means (%d) are not '
                    'equal.' % (len(regularization_list), len(run_mean)))
  if len(regularization_list) != len(run_std):
    raise TypeError('Lengths of regularizations (%d) and stds (%d) are not '
                    'equal.' % (len(regularization_list), len(run_std)))

  plt = matplotlib_pyplot()
  plt.figure()
  if golden_mean_std_dict:
    golden_regularization_list = []
    golden_mean_list = []
    golden_std_list = []
    for regularization_value, (mean_m, mean_s,
                               _, _) in golden_mean_std_dict.items():
      golden_regularization_list.append(regularization_value)
      golden_mean_list.append(mean_m)
      golden_std_list.append(mean_s)
    plt.errorbar(
        golden_regularization_list,
        golden_mean_list,
        golden_std_list,
        color='orange',
        uplims=True,
        lolims=True,
        label='golden')
  plt.errorbar(
      regularization_list, run_mean, run_std, color='blue', label='actual')
  plt.xscale('log')
  plt.xlabel('Regularization lambda (log10)')
  plt.ylabel('Mean correlation')
  plt.title(test_name + ' experiment correlation')
  plt.legend(loc='lower right')
  if png_file_name:
    base_dir = os.path.split(png_file_name)[0]
    if base_dir and not tf.io.gfile.exists(base_dir):
      tf.io.gfile.makedirs(base_dir)
    with tf.io.gfile.GFile(png_file_name, 'wb') as png_file:
      plt.savefig(png_file, format='png')
  if show_plot:
    plt.show()
