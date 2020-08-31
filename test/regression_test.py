# Lint as: python2 python3
"""Script to run the regression test for selected options.

The tests here just test to see if the plumbing works, and only check to see if
event and results files are created, suggesting that training has at least
proceeded this far.
"""

import os
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
from telluride_decoding import decoding
from telluride_decoding import regression
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


def compute_count_bounds_exceed(jk_mean, jk_std, run_mean):
  # mean -4*std value - likely to fail once every 43 years
  jk_lower_bounds = jk_mean - 4 * jk_std
  jk_upper_bounds = jk_mean + 4 * jk_std
  bound_exceed_count = sum(
      np.logical_or((run_mean < jk_lower_bounds),
                    (run_mean > jk_upper_bounds)))
  return bound_exceed_count


class RegressionTest(absltest.TestCase):

  def set_common_flags(self):
    """Flags common to all tests."""
    self.my_flags = decoding.DecodingOptions().set_flags()
    self.my_flags.frame_rate = 100.0
    self.my_flags.max_test_count = 2
    self.my_flags.logtostderr = True
    self.my_flags.summary_base_dir = self.tmp_dir('summary')
    self.my_flags.model_base_dir = self.tmp_dir('model')
    self.my_flags.tensorboard_dir = self.tmp_dir('tensorboard')

    self.my_flags.input_field = 'eeg'
    self.my_flags.output_field = 'loudness'
    self.my_flags.post_context = 3
    self.my_flags.input2_post_context = 3
    self.my_flags.input2_pre_context = 2

  def tmp_dir(self, final_part='regression_test'):
    full_path = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                             final_part)
    tf.io.gfile.makedirs(full_path)
    return full_path

  def find_event_files(self, search_dir, pattern='events.out.tfevents'):
    """Finds event files in a directory tree.

    Args:
      search_dir: The root of the tree to search
      pattern: A string at the start of the desired file names.

    Returns:
      A list of all files in the directory tree that start with the pattern
      string.
    """
    all_files = []
    for (root, _, files) in os.walk(search_dir):
      all_files.extend([os.path.join(root, f) for f in files
                        if f.startswith(pattern)])
    return all_files


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
