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

"""Tests for the decoding part of the library.

"""
import io
import math
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
import attr
import mock
import numpy as np
import numpy.matlib
import scipy

from telluride_decoding import decoding
from telluride_decoding import infer_decoder
from telluride_decoding.brain_data import TestBrainData

import tensorflow as tf


class DecodingTest(absltest.TestCase):

  def setUp(self):
    super(DecodingTest, self).setUp()
    self.model_flags = decoding.DecodingOptions().set_flags()
    self.fs = 100  # Audio and EEG sample rate in Hz
    self._test_data_dir = os.path.join(
        flags.FLAGS.test_srcdir, 'test_data', 'meg')

  def clear_model(self, model_dir='/tmp/tf'):
    try:
      # Clear out the model directory, if it exists, because the TF save model
      # code wants an empty directory.
      try:
        tf.io.gfile.rmtree(model_dir)
      except tf.errors.NotFoundError:
        pass
    except OSError:
      pass

  # Single random speech envelope, and then a simulated EEG
  # response different for each channel, is convolved with the speech envelope.
  # No noise so should be perfect reconstruction.

  def create_simulated_impulse_responses(self, num_input_channels,
                                         simulated_unattended_gain=0.10):
    """Create the basic impulse responses for this dataset.

    Generate random responses, one for the attended signal and the other for
    the unattended signal. Give them a realistic overall shape over  250ms.
    Do this once for a dataset (so we can generate multiple examples later
    using the same impulse responses).

    Args:
      num_input_channels: How many input channels to synthesize
      simulated_unattended_gain: How much lower is the unattended channel?

    Creates:
      attended_impulse_response, unattended_impulse_response:
        Two 0.25s x num_input_channel arrays representing the impulse response
        from the input sound (i.e. speech) to the system (i.e. EEG/meg)
        response.
    """
    self.impulse_length = .25  # Length of the TRF
    self.impulse_times = np.arange(self.impulse_length*self.fs) / self.fs
    self.envelope = 30*self.impulse_times * np.exp(-self.impulse_times*30,
                                                   dtype=np.float32)
    self.envelope_all_chans = numpy.matlib.repmat(np.reshape(self.envelope,
                                                             (-1, 1)),
                                                  1, num_input_channels)
    self.attended_impulse_response = np.random.randn(
        self.impulse_times.shape[0],
        num_input_channels) * self.envelope_all_chans
    self.unattended_impulse_response = np.random.randn(
        self.impulse_times.shape[0],
        num_input_channels) * self.envelope_all_chans
    # Cut the magnitude of the unattended inpulse response so as noise it will
    #  have a smaller or bigger effect.
    logging.info('SimulatedData:initialize_dataset: Setting unattended ' +
                 'gain to %s', simulated_unattended_gain)
    self.unattended_impulse_response *= simulated_unattended_gain

  def create_simulated_speech_signals(self):
    """Create the source signals for our experiment, attended and unattended.

    Now we create the "speech" signal, which will convolved with the EEG
    responses to create the data for this trial. Create speaker 1 and 2 signals.

    Creates:
      audio_signals: the N x 2 array of attended and unattended "speech"
        signals.
    """
    if self.use_sinusoids:
      self.audio_subj_1 = np.reshape(np.sin(self.recording_times*2*np.pi*5),
                                     (-1, 1))
      self.audio_subj_2 = np.reshape(np.sin(self.recording_times*2*np.pi*7),
                                     (-1, 1))
      self.audio_signals = np.concatenate((self.audio_subj_1,
                                           self.audio_subj_2),
                                          axis=1)
    else:
      # Create the signal with noise, and then linearly upsample so the result
      # is more low pass.
      num_output_channels = 2
      self.audio_low_freq = np.random.randn(math.ceil(len(self.recording_times)/
                                                      10.0),
                                            num_output_channels)
      self.audio_signals = scipy.signal.resample(self.audio_low_freq,
                                                 len(self.recording_times))
    logging.info('SimulatedData: audio signals shape: %s',
                 self.audio_signals.shape)

  def create_simulated_eeg_data(self, num_input_channels,
                                mode, simulated_noise_level=0.3,
                                num_output_channels=2,
                                simulated_unattended_gain=0.10):
    """Create one trial's worth of data.

    We create random EEG impulse responses and a random "speech" signal.
    We convolve the EEG impulse responses with the speech data to get a
    simulated EEG signal..

    Args:
      num_input_channels: How many input channels to generate
      mode: Mostly ignored unless you ask for "demo" when then generates a
        signal which oscillates between the two speakers.
      simulated_noise_level: How much noise to add for simulation.
      num_output_channels: How many output channels to generate. For 1 just the
        attended speech signal.  For 2, both the attended and unattended speech
        signals.
      simulated_unattended_gain: Fraction of attended signal level for
        unattended data.

    Returns:
      The EEG response, and the simulated audio intensity, both numpy arrays.
    """
    self.assertGreater(num_input_channels, 0)
    self.signal_length = 1000  # Seconds
    self.use_sinusoids = 1  # Random or sinusoidal driving signal
    self.create_simulated_impulse_responses(
        num_input_channels, simulated_unattended_gain=simulated_unattended_gain)

    self.recording_times = np.arange(self.signal_length*self.fs)/float(self.fs)
    self.create_simulated_speech_signals()

    if mode.startswith('demo'):
      # Create the attention signal, switching every attention_duration seconds
      self.attention_duration = 25  # Time in seconds listening to each signal
      self.attention_signal = np.mod(
          np.floor(self.recording_times / self.attention_duration), 2)
      # This signal flips between 0 and 1, every attentionuration seconds
      a = np.reshape(self.attention_signal, (-1, 1))
    else:
      # Attention signal is constant, appropriate for testing.
      a = np.ones((self.recording_times.shape[0], 1), dtype=np.float32)
    self.attention_matrix = np.concatenate((1-a, a), axis=1)

    # Create the signals that correspond to the attended and unattended audio,
    # under control of the deterministic attention_signal created above.
    self.attended_audio = np.sum(self.attention_matrix * self.audio_signals,
                                 axis=1).astype(np.float32)
    self.unattended_audio = np.sum((1 - self.attention_matrix) *
                                   self.audio_signals,
                                   axis=1).astype(np.float32)

    # Now convolve the attended and unattended audio with the two different
    #  impulse responses to create the simulated EEG signals.
    logging.info('Creating an example of the SimulatedData....')
    response = np.zeros((self.attended_audio.shape[0] +
                         self.attended_impulse_response.shape[0] - 1,
                         num_input_channels), dtype=np.float32)
    for c in range(num_input_channels):
      attended_response = np.convolve(self.attended_audio,
                                      self.attended_impulse_response[:, c],
                                      mode='full')
      unattended_response = np.convolve(self.unattended_audio,
                                        self.unattended_impulse_response[:, c],
                                        mode='full')
      # Sum up the attended and unattended response, and then add noise.
      response[:, c] = (
          attended_response + unattended_response +
          simulated_noise_level * np.random.randn(attended_response.shape[0]))
    logging.info('Attended shapes: %s and %s', self.attended_audio.shape,
                 self.unattended_audio.shape)
    if num_output_channels == 1:
      output_channels = np.reshape(self.attended_audio, (-1, 1))
    else:
      output_channels = np.concatenate((np.reshape(self.attended_audio,
                                                   (-1, 1)),
                                        np.reshape(self.unattended_audio,
                                                   (-1, 1))),
                                       axis=1)
    logging.info('Output_channels shape: %s', output_channels.shape)
    return (response[0:self.attended_audio.shape[0], 0:num_input_channels],
            output_channels)

  @flagsaver.flagsaver
  def test_train_and_test_linear(self):
    self.clear_model()
    test_brain_data = TestBrainData(
        'input', 'output', self.model_flags.frame_rate,
        final_batch_size=self.model_flags.batch_size,
        pre_context=self.model_flags.pre_context,
        post_context=self.model_flags.post_context,
        repeat_count=1)
    num_input_channels = 32
    (response, speech) = self.create_simulated_eeg_data(
        num_input_channels, mode='train', num_output_channels=1)
    test_brain_data.preserve_test_data(response, speech)
    test_dataset = test_brain_data.create_dataset('train')

    self.model_flags.dnn_regressor = 'linear'
    self.model_flags.regularization_lambda = 0.0
    linear_model = decoding.create_brain_model(self.model_flags, test_dataset)
    (train_results, test_results) = decoding.train_and_test(
        self.model_flags, test_brain_data, linear_model)
    logging.info('test_train_and_test_linear training results: %s',
                 train_results)
    logging.info('test_train_and_test_linear testing results: %s',
                 test_results)
    self.assertGreater(test_results['pearson_correlation_first'], 0.97)

  @flagsaver.flagsaver
  def test_train_and_test_dnn(self):
    self.clear_model()
    self.model_flags.dnn_regressor = 'fullyconnected'
    num_input_channels = 32
    test_brain_data = TestBrainData(
        'input', 'output', self.model_flags.frame_rate,
        final_batch_size=self.model_flags.batch_size,
        pre_context=self.model_flags.pre_context,
        post_context=self.model_flags.post_context,
        repeat_count=1)
    (response, speech) = self.create_simulated_eeg_data(
        num_input_channels, mode='train', num_output_channels=1)
    test_brain_data.preserve_test_data(response, speech)
    train_dataset = test_brain_data.create_dataset('train')

    dnn_model = decoding.create_brain_model(self.model_flags, train_dataset)
    (train_results, test_results) = decoding.train_and_test(
        self.model_flags, test_brain_data, dnn_model, epochs=10)
    logging.info('test_train_and_test_dnn training results: %s', train_results)
    logging.info('test_train_and_test_dnn testing results: %s', test_results)
    self.assertGreater(test_results['pearson_correlation_first'], 0.97)

  @flagsaver.flagsaver
  def test_train_and_test_cca(self):
    self.clear_model()
    self.model_flags.dnn_regressor = 'cca'
    self.model_flags.pre_context = 2
    self.model_flags.post_context = 3
    self.model_flags.input2_field = 'speech'
    test_brain_data = TestBrainData(
        'eeg', 'none', self.model_flags.frame_rate,
        final_batch_size=self.model_flags.batch_size,
        pre_context=self.model_flags.pre_context,
        post_context=self.model_flags.post_context,
        in2_fields=self.model_flags.input2_field,
        in2_pre_context=self.model_flags.pre_context,
        in2_post_context=self.model_flags.post_context,
        repeat_count=1)
    num_input_channels = 32
    (response, speech) = self.create_simulated_eeg_data(
        num_input_channels, mode='train', num_output_channels=1,
        simulated_noise_level=0.0,
        simulated_unattended_gain=0.00
        )
    test_brain_data.preserve_test_data(response, 0*response[:, 0:1], speech)
    train_dataset = test_brain_data.create_dataset('train')

    self.model_flags.cca_dimensions = 4
    cca_model = decoding.create_brain_model(self.model_flags, train_dataset)
    (train_results, test_results) = decoding.train_and_test(
        self.model_flags, test_brain_data, cca_model)
    logging.info('test_train_and_test_cca training results: %s', train_results)
    logging.info('test_train_and_test_cca testing results: %s', test_results)
    # This test was for 3.8 when CCA was summarized with sum, and now it is 0.75
    # because the test was flakey.
    self.assertGreater(abs(test_results['cca_pearson_correlation_first']), 0.75)

    # Now test the correlation and LDA training code. The piece below is a
    # separate test, but depends on the setup above, so it is added here.
    self.model_flags.dnn_regressor = 'cca'
    dprime, decoder = decoding.train_lda_model(test_brain_data, cca_model,
                                               self.model_flags)
    print('train_lda_model got a dprime of', dprime)
    self.assertGreater(dprime, 0.7)  # Conservative, just testing plumbing.
    self.assertIsInstance(decoder, infer_decoder.Decoder)

  def test_create_brain_model(self):
    test_brain_data = TestBrainData('eeg', 'none', self.fs,
                                    in2_post_context=3,  # Needed for CCA
                                   )
    (response, speech) = self.create_simulated_eeg_data(
        2, mode='train', num_output_channels=1)
    test_brain_data.preserve_test_data(response, speech)
    test_dataset = test_brain_data.create_dataset('train')

    with self.assertRaisesRegex(TypeError,
                                'Model_flags must be a DecodingOptions'):
      decoding.create_brain_model(42, test_dataset)
    with self.assertRaisesRegex(TypeError, 'input_dataset must be a tf.data'):
      decoding.create_brain_model(self.model_flags, 42)

    with self.subTest(name='fullyconnected'):
      fully_flags = attr.evolve(self.model_flags)
      fully_flags.dnn_regressor = 'fullyconnected'
      self.assertTrue(decoding.create_brain_model(fully_flags, test_dataset))
    with self.subTest(name='linear'):
      linear_flags = attr.evolve(self.model_flags)
      linear_flags.dnn_regressor = 'linear'
      self.assertTrue(decoding.create_brain_model(linear_flags, test_dataset))
    with self.subTest(name='cca'):
      cca_flags = attr.evolve(self.model_flags)
      cca_flags.dnn_regressor = 'cca'
      self.assertTrue(decoding.create_brain_model(cca_flags, test_dataset))

  @flagsaver.flagsaver
  def test_main_check_files(self):
    """Just test the short path that checks the data is good.

    Make sure we find all the files, and it is all good.
    """
    self.model_flags.tfexample_dir = os.path.join(
        flags.FLAGS.test_srcdir, 
        'test_data')
    self.model_flags.check_file_pattern = True

    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      decoding.run_decoding_experiment(self.model_flags)
    self.assertIn('Found 3 files for TFExample data analysis.',
                  mock_stdout.getvalue())

  @flagsaver.flagsaver
  def test_main(self):
    """Just test the short path that checks the data is good.

    Make sure the code runs without exceptions, as other tests do the parts.
    """
    self.model_flags.tfexample_dir = os.path.join(
        flags.FLAGS.test_srcdir, 'test_data/')
    tensorboard_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                                   'tensorboard')
    self.model_flags.tensorboard_dir = tensorboard_dir
    summary_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp', 'summary')
    self.model_flags.summary_dir = summary_dir
    saved_model_dir = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                                   'saved_model')
    self.model_flags.saved_model_dir = saved_model_dir

    self.clear_model(tensorboard_dir)
    self.clear_model(summary_dir)
    decoding.run_decoding_experiment(self.model_flags)

    def all_files(root_dir):
      """Gets all files within a directory.

      Needed since tensorboard events are put into a subdirectory that main()
      doesn't know about.

      Args:
        root_dir: The root of the tree.

      Yields:
        A sequence of full path names.
      """
      for base_dir, _, files in tf.io.gfile.walk(root_dir):
        for f in files:
          yield os.path.join(base_dir, f)

    self.assertTrue(tf.io.gfile.exists(os.path.join(summary_dir,
                                                    'results.txt')))
    train_files = [f for f in all_files(tensorboard_dir)
                   if 'train/events' in f]
    self.assertNotEmpty(train_files)

    dprime_files = [f for f in all_files(tensorboard_dir)
                    if 'dprime/events' in f]
    self.assertNotEmpty(dprime_files)

    self.assertTrue(tf.io.gfile.exists(os.path.join(saved_model_dir,
                                                    'decoder_model.json')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(saved_model_dir,
                                                    'saved_model.pb')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(saved_model_dir,
                                                    'variables',
                                                    'variables.index')))


if __name__ == '__main__':
  absltest.main()
