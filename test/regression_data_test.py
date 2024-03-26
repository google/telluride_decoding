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

"""Test for telluride_decoding.regression_data."""

import os
import subprocess

from absl import flags
from absl.testing import absltest

from telluride_decoding import brain_data
from telluride_decoding import regression_data

import tensorflow as tf

# Note these tests do NOT test the data download cdoe.  These are hard to test,
# only run occasionally, and are obvious when they don't work in real use.


class TellurideDataTest(absltest.TestCase):

  def setUp(self):
    super(TellurideDataTest, self).setUp()
    self._test_data_dir = os.path.join(flags.FLAGS.test_srcdir, '_main',
                                       'test_data')
    if not os.path.exists(self._test_data_dir):
      # Debugging: If not here, where.
      subprocess.run(['ls', flags.FLAGS.test_srcdir])
      subprocess.run(['ls', os.path.join(flags.FLAGS.test_srcdir, '_main')])
      self.assertTrue(os.path.exists(self._test_dir),
                      f'Test data dir does not exist: {self._test_dir}')

  def test_data_ingestion(self):
    cache_dir = os.path.join(self._test_data_dir, 'telluride4')
    tmp_dir = self.create_tempdir().full_path
    tf_dir = os.path.join(tmp_dir, 'telluride4_tf')

    # Create the data object and make sure we have the downloaded archive file.
    rd = regression_data.RegressionDataTelluride4()
    if not rd.is_data_local(cache_dir):
      url = 'https://drive.google.com/uc?id=0ByZjGXodIlspWmpBcUhvenVQa1k'
      rd.download_data(url, cache_dir, debug=True)
    self.assertTrue(rd.is_data_local(cache_dir))

    # Now ingest the data, making sure it's not present at start, then present.
    self.assertFalse(rd.is_data_ingested(tmp_dir))
    rd.ingest_data(cache_dir, tf_dir, 128)
    self.assertTrue(rd.is_data_ingested(tf_dir))

    # Check the data files.
    test_file = os.path.join(tf_dir, 'trial_01.tfrecords')
    features = brain_data.discover_feature_shapes(test_file)
    print('Telluride features:', features)
    self.assertIn('eeg', features)
    self.assertEqual(features['eeg'].shape, [63])
    self.assertIn('intensity', features)
    self.assertEqual(features['intensity'].shape, [1])

    self.assertEqual(brain_data.count_tfrecords(test_file), (8297, False))


class JensMemoryDataTest(absltest.TestCase):

  def setUp(self):
    super(JensMemoryDataTest, self).setUp()
    self._test_data_dir = os.path.join(flags.FLAGS.test_srcdir, '_main', 
                                       'test_data')

  def test_data_ingestion(self):
    cache_dir = os.path.join(self._test_data_dir, 'jens_memory')
    tmp_dir = self.create_tempdir().full_path
    tf_dir = os.path.join(tmp_dir, 'jens_memory')
    num_subjects = 1   # Only 1 of 22 subjects loaded for test.
    num_trials = 5   # That one subject has been shortened to 5/40 trials.

    # Create the data object and make sure we have the downloaded archive file.
    rd = regression_data.RegressionDataJensMemory()
    subprocess.run(['ls', flags.FLAGS.test_srcdir])
    subprocess.run(['ls', self._test_data_dir])
    subprocess.run(['ls', cache_dir])
    self.assertTrue(rd.is_data_local(cache_dir, num_subjects))

    # Now ingest the data, making sure it's not present at start, then present.
    self.assertFalse(rd.is_data_ingested(tmp_dir, num_subjects))
    rd.ingest_data(cache_dir, tf_dir, 128)
    self.assertTrue(rd.is_data_ingested(tf_dir, num_subjects, num_trials))

    # Check the data files.
    test_file = os.path.join(tf_dir, 'subject_01', 'trial_01.tfrecords')
    features = brain_data.discover_feature_shapes(test_file)
    self.assertIn('eeg', features)
    self.assertEqual(features['eeg'].shape, [69])
    self.assertIn('intensity', features)
    self.assertEqual(features['intensity'].shape, [1])

    self.assertEqual(brain_data.count_tfrecords(test_file), (7442, False))


if __name__ == '__main__':
  absltest.main()
