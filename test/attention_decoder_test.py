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

"""Tests for telluride_decoding.attention_decoder."""

from absl.testing import absltest
from absl.testing import parameterized

import matplotlib.axes as axes
import matplotlib.pyplot as plt

import mock
import numpy as np

from telluride_decoding import attention_decoder


class AttentionDecoderPlotTest(absltest.TestCase):

  @mock.patch.object(axes.Axes, 'add_patch')
  @mock.patch.object(plt, 'fill_between')
  @mock.patch.object(plt, 'title')
  @mock.patch.object(plt, 'ylabel')
  @mock.patch.object(plt, 'xlabel')
  @mock.patch.object(plt, 'plot', wraps=plt.plot)
  def test_basic(self, mock_plot, mock_xlabel, mock_ylabel, mock_title,
                 mock_fill_between, mock_add_patch):
    ones = np.ones(10)
    decision = ones*0.5
    xlabel = 'Mock Time (frames)'
    ylabel = 'Mock Prob of Speaker 1'
    title = 'Mock AAD Decoding Result'
    attention_decoder.plot_aad_results(decision, attention_flag=None,
                                       decision_upper=None, decision_lower=None,
                                       t=None, xlabel=xlabel,
                                       ylabel=ylabel,
                                       title=title)
    mock_plot.assert_called_once_with(mock.ANY, mock.ANY, 'blue')
    mock_xlabel.assert_called_once_with(xlabel)
    mock_ylabel.assert_called_once_with(ylabel)
    mock_title.assert_called_once_with(title)
    mock_fill_between.assert_not_called()

    attention_flag = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    attention_decoder.plot_aad_results(decision,
                                       attention_flag=attention_flag,
                                       decision_upper=ones*.4,
                                       decision_lower=ones*.6,
                                       t=None, xlabel=xlabel,
                                       ylabel=ylabel,
                                       title=title)
    mock_fill_between.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY,
                                              color='lightblue')
    # There are two separate segments where the attention_flag variable above
    # where attention_flag=1, so we expect just two gray patches in the plot.
    self.assertEqual(mock_add_patch.call_count, 2)


class AttentionDecoderPlotErrorTest(parameterized.TestCase):

  ones = np.ones(10)

  @parameterized.named_parameters(
      ('bad_decision',  True, ones, ones, ones, ones, 'decision'),  # pylint: disable=bad-whitespace
      ('bad_attention', ones, True, ones, ones, ones, 'attention_flag'),  # pylint: disable=bad-whitespace
      ('bad_upper',     ones, ones, True, ones, ones, 'decision_upper'),  # pylint: disable=bad-whitespace
      ('bad_lower',     ones, ones, ones, True, ones, 'decision_lower'),  # pylint: disable=bad-whitespace
      ('bad_t',         ones, ones, ones, ones, True, 't'),  # pylint: disable=bad-whitespace
      )
  def test_bad_param(self, decision, attention, upper, lower, t, var):
    with self.assertRaisesRegex(TypeError,
                                'Argument %s must be an np array, not' % var):
      attention_decoder.plot_aad_results(decision=decision,
                                         attention_flag=attention,
                                         decision_upper=upper,
                                         decision_lower=lower,
                                         t=t)

  part = ones[:5]

  @parameterized.named_parameters(
      ('bad_attention', ones, part, ones, ones, ones, 'attention_flag'),  # pylint: disable=bad-whitespace
      ('bad_upper',     ones, ones, part, ones, ones, 'decision_upper'),  # pylint: disable=bad-whitespace
      ('bad_lower',     ones, ones, ones, part, ones, 'decision_lower'),  # pylint: disable=bad-whitespace
      ('bad_t',         ones, ones, ones, ones, part, 't'),  # pylint: disable=bad-whitespace
      )
  def test_short_param(self, decision, attention, upper, lower, t, var):
    with self.assertRaisesRegex(TypeError,
                                'Input %s must match length of decision' % var):
      attention_decoder.plot_aad_results(decision=decision,
                                         attention_flag=attention,
                                         decision_upper=upper,
                                         decision_lower=lower,
                                         t=t)


class AttentionDecoder(absltest.TestCase):

  def test_basics(self):
    ad = attention_decoder.AttentionDecoder()

    self.assertTrue(ad.attention(0.6, 0.4)[0])
    self.assertFalse(ad.attention(0.4, 0.6)[0])

    self.assertTrue(ad.attention(0.6*np.ones(5), 0.4*np.ones(5))[0])
    self.assertFalse(ad.attention(0.4*np.ones(5), 0.6*np.ones(5))[0])

    cor1 = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
    cor2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    desi = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    ad.tune(cor1, cor2)
    results = [ad.attention(r1, r2)[0] for (r1, r2) in zip(cor1, cor2)]

    np.testing.assert_array_equal(desi, results)


class StepAttentionDecoder(absltest.TestCase):

  def test_basics(self):
    cor1 = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
    cor2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    desi = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    ad = attention_decoder.StepAttentionDecoder()
    results = [ad.attention(r1, r2)[0] for (r1, r2) in zip(cor1, cor2)]

    np.testing.assert_array_equal(desi, results)

  def test_short_switch(self):
    cor1 = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cor2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    desi = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ad = attention_decoder.StepAttentionDecoder()
    results = [ad.attention(r1, r2)[0] for (r1, r2) in zip(cor1, cor2)]

    np.testing.assert_array_equal(desi, results)


class StateAttentionDecoderTest(absltest.TestCase):

  def get_synth_corr(self, seconds, fs, state_switch, corr_scale=0.3):
    """Synth data to approximate attentional switch every state_switch seconds.

    Args:
      seconds: number of seconds total for generated sample.
      fs: fictitious sample rate.
      state_switch: the interval in seconds at which the attentional state
        switches, ex. 10 implies an attentional state switches between s1 and s2
        every ten seconds.
      corr_scale: scale the correlation measures for more realistic behaviour.

    Returns:
      true_state: the true underlying attentional state.
      correlations: the simulated correlation values.
    """
    max_t = fs*seconds
    noise_std = 0.5
    t = np.arange(max_t)
    true_state = np.zeros((max_t,))
    true_state += (np.floor(t/(state_switch*fs))%2)+1

    correlations = np.zeros((max_t, 2))
    correlations[:, 0] = np.random.randn(max_t)*noise_std + (2-true_state)
    correlations[:, 1] = np.random.randn(max_t)*noise_std + (true_state-1)
    correlations *= corr_scale
    correlations = np.minimum(1, np.maximum(-1, correlations))

    return true_state, correlations

  def test_attention_decoder(self):
    """Test StateSpaceAttentionDecoder can decode the simple synthetic data."""
    print('\n\n**********test_attention_decoder starting... *******')
    # fictitious data comprising 1000 seconds with attentional switch every 300s
    # starting with speaker 1.
    fs_corr = 1  # Hertz
    len_data = 1000  # seconds
    switch_interval = 300  # seconds
    init_interval = 100  # seconds
    init_frames = int(fs_corr*init_interval)  # frames
    true_state, correlations = self.get_synth_corr(len_data, fs_corr,
                                                   switch_interval)
    # Create attention decoder object
    outer_iter = 20
    inner_iter = 1
    newton_iter = 10
    ad = attention_decoder.StateSpaceAttentionDecoder(outer_iter,
                                                      inner_iter,
                                                      newton_iter, fs_corr)
    # Tune log normal priors using first 100s of data
    ad.tune_log_normal_priors(correlations[:init_frames, 0],
                              correlations[:init_frames, 1])

    # Decode attention based on each set of correlations
    results = np.zeros((correlations.shape[0]-ad.k_w-init_frames, 3))
    for i in range(correlations[init_frames:, :].shape[0]):
      res = ad.attention(correlations[i+init_frames, 0],
                         correlations[i+init_frames, 1])
      if res is not None:
        results[i-ad.k_w] = res
      print('ADD result: ', res)

    s1 = results[:, 0] > 0.5  # classified as speaker 1

    # true_state is a vector of 1,2 for speaker 1,2 respectively
    # s1 is true if classified as attending to speaker 1

    # Here, we test to make sure the error rate from a naive decision, classify
    # as attending to speaker 1 if the point estimator results[:,0] is above
    # 0.5. We use the first 100 observations to tune the hyperparameters of the
    # model using tuneLogNormalPriors above, so we drop this from the subsequent
    # analysis. The ad.k_w is the window size of the number of correlation
    # values that are used to make a single attentional state decision so the
    # first decoded attentional state corresponds to the 100+ad.k_w true_state
    # (this is complicated a bit when using forward lags because the first
    # decoded attentional state would correspond to the 100+ad.k_w-ad.k_f th
    # true_state).

    # The true_state vector is a vector of 1s and 2s corresponding to the labels
    # speaker 1 and speaker 2, so we convert to a boolean where True indicates
    # attending to speaker 1 and False to attending to speaker 2.
    error = np.mean(np.abs(s1 != (true_state[init_frames+ad.k_w:] < 2)))
    self.assertLess(error, 0.15)

  def test_log_normal_initialization(self):
    # Pick two different sets of parameters and make sure the initialization
    # code recovers the correct Gaussian parameters

    # Number of observations
    num_data = 1000000

    # Generate attended speaker data (log-normal)
    mu_a = 0.2
    var_a = 0.1
    data_a = np.exp(np.random.randn(num_data)*var_a + mu_a)

    # Generate unattended speaker data (log-normal)
    mu_b = 0.0
    var_b = 0.1
    data_b = np.exp(np.random.randn(num_data)*var_b + mu_b)

    # Create attention decoder object
    ad = attention_decoder.StateSpaceAttentionDecoder(20, 1, 10, 1, 1)

    # Tune log-normal priors
    ad.tune(data_a, data_b)

    # Log-transform and normalize between [0,1]
    log_data_a = np.log(np.absolute(data_a))
    log_data_b = np.log(np.absolute(data_b))

    # Compute mean and precision
    mu_a = np.mean(log_data_a)
    mu_b = np.mean(log_data_b)
    rho_a = np.var(log_data_a)
    rho_b = np.var(log_data_b)

    self.assertAlmostEqual(ad.mu_d[0], mu_a, delta=0.0001)
    self.assertAlmostEqual(ad.mu_d[1], mu_b, delta=0.0001)
    self.assertAlmostEqual(ad.rho_d[0], 1.0/rho_a, delta=5)
    self.assertAlmostEqual(ad.rho_d[1], 1.0/rho_b, delta=5)

    # Create new attention decoder object
    ad = attention_decoder.StateSpaceAttentionDecoder(20, 1, 10, 1, 1,
                                                      offset=1.0)

    # Tune log-normal priors
    ad.tune(data_a, data_b)

    # Make sure the mu value gets bigger with a positive offset.
    self.assertGreater(ad.mu_d[0], mu_a + 0.01)


class CreateTest(absltest.TestCase):

  def test_all(self):
    ad = attention_decoder.create_attention_decoder('wta')
    self.assertIsInstance(ad, attention_decoder.AttentionDecoder)

    ad = attention_decoder.create_attention_decoder('stepped')
    self.assertIsInstance(ad, attention_decoder.StepAttentionDecoder)

    ad = attention_decoder.create_attention_decoder('ssd')
    self.assertIsInstance(ad, attention_decoder.StateSpaceAttentionDecoder)

    with self.assertRaisesRegex(ValueError, 'Unknown type'):
      ad = attention_decoder.create_attention_decoder('bad type name')

if __name__ == '__main__':
  absltest.main()
