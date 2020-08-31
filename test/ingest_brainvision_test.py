# Lint as: python2 python3
"""Tests for ingest_brain_vision."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import absltest
from telluride_decoding import ingest
from telluride_decoding import ingest_brainvision


class IngestBrainVisionTest(absltest.TestCase):

  def setUp(self):
    super(IngestBrainVisionTest, self).setUp()
    self._test_data = os.path.join(
        flags.FLAGS.test_srcdir, '__main__',
        'test_data/')

  def test_read_bv_file(self):
    header_filename = os.path.join(self._test_data, 'brainvision_test.vhdr')
    self.assertTrue(os.path.exists(header_filename),
                    'Can not find %s' % header_filename)
    header, data = ingest_brainvision.read_bv_file(header_filename)
    self.assertContainsSubset(['Common Infos', 'Binary Infos', 'Channel Infos'],
                              list(header.keys()))
    self.assertEqual(header['Common Infos']['SamplingInterval'], 2000)
    self.assertEqual(header['Common Infos']['NumberOfChannels'], 65)
    self.assertEqual(data.shape[0], 5)
    self.assertEqual(data.shape[1], header['Common Infos']['NumberOfChannels'])

  def test_brainvision_data_file(self):
    bvdf = ingest_brainvision.BvBrainDataFile('brainvision_test.vhdr')
    bvdf.load_all_data(self._test_data)
    print('Channel names:', bvdf.signal_names)
    expected_channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1',
                              'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
                              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6',
                              'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
                              'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5',
                              'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                              'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6',
                              'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
                              'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz',
                              'TRIG', 'EOG']
    self.assertCountEqual(expected_channel_names, bvdf.signal_names)
    self.assertEqual(bvdf.signal_fs('foo'), 500)
    self.assertEqual(bvdf.find_channel_index(), 63)
    self.assertEqual(bvdf.find_channel_index('TRIG'), 63)
    self.assertEqual(bvdf.find_channel_index('C3'), 7)
    channel_data = bvdf.signal_values('CH1')
    self.assertFalse(channel_data)
    channel_data = bvdf.signal_values('TRIG')
    self.assertIsNotNone(channel_data)
    print('Channel_data shape is', channel_data.shape)
    self.assertLen(channel_data.shape, 1)
    self.assertEqual(channel_data.shape[0], 5)

  def test_brain_experiment(self):
    df = ingest_brainvision.BvBrainDataFile('brainvision_test.vhdr')
    sound_filename = 'subj01_1ksamples.wav'
    trial_name = ingest.BrainExperiment.delete_suffix(sound_filename, '.wav')
    trial_dict = {trial_name: [sound_filename, df]}
    experiment = ingest.BrainExperiment(trial_dict,
                                        self._test_data, self._test_data+'/meg')
    experiment.load_all_data(self._test_data+'/meg', self._test_data)
    summary = experiment.summary()
    self.assertIn('Found 1 trials', summary)
    self.assertIn('Trial subj01_1ksamples: 65 EEG channels with 0.01s of '
                  'eeg data', summary)
    experiment.z_score_all_data()

if __name__ == '__main__':
  absltest.main()
