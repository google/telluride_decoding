"""AttentionDecoderCorr takes an array of correlations and outputs attention.


This code is based on Matlab code published here:
https://github.com/sinamiran/Real-Time-Tracking-of-Selective-Auditory-Attention
Based on the work:
S. Miran, S. Akram, A. Sheikhattar, J. Z. Simon, T. Zhang, and B. Babadi,
Real-Time Tracking of Selective Auditory Attention from M/EEG: A Bayesian
Filtering Approach, Frontiers in Neuroscience, Vol. 12, pp. 262, May 2018

and

S. Akram, J. Z. Simon, S. A. Shamma, and B. Babadi,
A State-Space Model for Decoding Auditory Attentional Modulation from MEG in a
Competing-Speaker Environment, 2014 Neural Information Processing Systems,
Dec 2014, Montreal, QC, Canada.
"""

import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_aad_results(decision, attention_flag=None,
                     decision_upper=None, decision_lower=None,
                     t=None, xlabel='Time (frames)',
                     ylabel='Prob of Speaker 1',
                     title='AAD Decoding Result',
                     linecolor='blue'):
  """Plots the results of an attention decoding experiment.

  Show decision variable, along with upper and lower probability bounds.
  Note: the graph is not cleared, so multiple results can be plotted on the same
  axis.

  Args:
    decision: estimate for attention direction to speaker 1
    attention_flag: ground truth indication of attended speaker: 0 or 1
    decision_upper: positive delta /probability of SSD estimate
    decision_lower: minus delta probability of SSD estimate
    t: optional time axis, otherwise x-axis is in terms of frames
    xlabel: Label for the x-axis of the graph
    ylabel: Label for the y-axis of the graph
    title: Title of the graph
    linecolor: Color of the line, and then lightColor for the confidence bounds.
  """
  if not isinstance(decision, np.ndarray):
    raise TypeError('Argument decision must be an np array, not %s' %
                    type(decision))
  if attention_flag is not None:
    if not isinstance(attention_flag, np.ndarray):
      raise TypeError('Argument attention_flag must be an np array, not %s' %
                      type(attention_flag))
    elif len(decision) != len(attention_flag):
      raise TypeError('Input attention_flag must match length of decision,'
                      ' not %d and %d' % (len(decision), len(attention_flag)))
  if decision_upper is not None:
    if not isinstance(decision_upper, np.ndarray):
      raise TypeError('Argument decision_upper must be an np array, not %s' %
                      type(decision_upper))
    elif len(decision) != len(decision_upper):
      raise TypeError('Input decision_upper must match length of decision,'
                      ' not %d and %d' % (len(decision), len(decision_upper)))
  if decision_lower is not None:
    if not isinstance(decision_lower, np.ndarray):
      raise TypeError('Argument decision_lower must be an np array, not %s' %
                      type(decision_lower))
    elif len(decision) != len(decision_lower):
      raise TypeError('Input decision_lower must match length of decision,'
                      ' not %d and %d' % (len(decision), len(decision_lower)))
  if t is not None:
    if not isinstance(t, np.ndarray):
      raise TypeError('Argument t must be an np array, not %s' %
                      type(t))
    elif len(decision) != len(t):
      raise TypeError('Input t must match length of decision,'
                      ' not %d and %d' % (len(decision), len(t)))
  else:  # Default is sample based time axis.
    t = np.arange(len(decision))

  plt.plot(t, decision, linecolor)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)

  if decision_upper is not None and decision_lower is not None:
    # Add blue band for the confidence interval.
    plt.fill_between(t, decision_upper, decision_lower, color='light'+linecolor)

  # Check to see if the attention signal has two values, indicating that
  # there is at least one attention switch.
  if attention_flag is not None and (np.sum(attention_flag == 0) and
                                     np.sum(attention_flag != 0)):
    # Add gray boxes from top to bottom for every other attention section.
    axis_limits = plt.axis()
    start_index = 0
    attention_flag = list(attention_flag)
    for attention_value, values in itertools.groupby(attention_flag):
      duration = len(list(values))
      if attention_value:  # Draw a gray box where attention value is != 0
        rect = patches.Rectangle((t[start_index], axis_limits[2]),
                                 t[start_index + duration - 1] - t[start_index],
                                 axis_limits[3]-axis_limits[2],
                                 facecolor='lightgray',
                                 alpha=0.5)
        plt.gca().add_patch(rect)
      start_index += duration


class AttentionDecoder(object):
  """Base attention decoder.  Winner takes all.
  """

  def attention(self, r1, r2):
    return np.mean(r1) > np.mean(r2), 0, 0

  def tune(self, r1, r2):
    """An optional training step for tuning parameters."""
    del r1, r2


class StepAttentionDecoder(AttentionDecoder):
  """Decodes using a bit of hysteresis.  Steps between 0.1 and 0.9.

  Decode final decision by maintaining a state variable.  It starts at 0.5
  and increases by 0.1 everytime correlation 1 is bigger than correlation 2. It
  decreases by 0.1 in the opposite case. The state variable is limited to be
  between 0.1 and 0.9.
  """

  def __init__(self):
    self.state = 0.5

  def attention(self, r1, r2):
    """Calculate the attention decision using simple comparison at this time.

    Args:
      r1: Scalar signal indicating likelihood of speaker 1.
      r2: Scalar signal indicating likelihood of speaker 2.

    Returns:
      Boolean decision, plus empty confidence bounds (delta is zero for both
      upper and lower intervals.)
    """
    if np.mean(r1) > np.mean(r2):
      self.state = min(0.9, self.state + 0.1)
    else:
      self.state = max(0.1, self.state - 0.1)
    return self.state > 0.5, 0, 0


class StateSpaceAttentionDecoder(AttentionDecoder):
  """Object to contain the attention state decoding.

  Will expect an incoming stream of real-time correlations computed using
  some sort of decoding algorithm.
  """

  def __init__(self, outer_iter, inner_iter, newton_iter, fs_corr,
               forward_lag=0, backward_lag=13, offset=0.0):
    """Initializer.

    Args:
      outer_iter: number of iterations in the outer EM loop.
      inner_iter: number of iterations in the inner EM loop.
      newton_iter: number of iteration in the application of Newton's algo.
      fs_corr: sample rate of the correlations (not EEG data).
      forward_lag: amount of data used in frames after the prediction t to
        make the predictions.
      backward_lag: amount of data used in frames prior to the
        prediction t to make the predictions.
      offset: Temporary hack to move data away from negative #s, use positive
        1 or 2 to move things away from zero.
    """
    self._offset = offset

    self.outer_iter = outer_iter  # number of iterations in outer EM loop
    self.inner_iter = inner_iter  # number of iterations in inner EM loop
    self.newton_iter = newton_iter  # number of iterations in Newton Alg loop

    # parameters of the fixed-lag sliding window
    self.fs_corr = fs_corr
    self.forward_lag = forward_lag
    self.backward_lag = backward_lag
    self.k_f = self.forward_lag
    self.k_b = self.backward_lag
    self.k_w = self.k_f + self.k_b + 1  # sliding window size of the decoding
    print('StateSpaceAttentionDecoder init: k_f is %d, k_b is %d' %
          (self.k_f, self.k_b))

    # 95% confidence intervals (changed from 90%)
    self.c0 = 1.96

    # parameters of the inverse-gamma prior on the state-space variances
    self.mean_p = 0.2
    self.var_p = 5
    self.a_0 = 2 + self.mean_p**2 / self.var_p
    self.b_0 = self.mean_p * (self.a_0 - 1)

    # Keep track of the number of times the attention function is called to
    # easily track when to update the correlation vectors and state estimates
    self.calls = 0

    # track the computed correlations
    self.r1 = []
    self.r2 = []

    # track the estimated parameters of the state space model, initialised as
    # below.
    self.z_smoothed = []
    self.eta_smoothed = []
    self.z_dyn = []
    self.eta_dyn = []
    for _ in range(self.k_w):
      self.z_smoothed.append(0.0)
      self.z_dyn.append(0.0)
      self.eta_smoothed.append(0.3)
      self.eta_dyn.append(0.0)

    # set the degree of smoothness of attention over the window
    self.lambda_state = 1.0

    # initialise the Kalman filtering variables used for smoothing
    self.z_k_k = np.zeros((self.k_w+1,))
    self.sig_k_k = np.zeros((self.k_w+1,))
    self.z_k_k_1 = np.zeros((self.k_w+1,))
    self.sig_k_k_1 = np.zeros((self.k_w+1))

    self.z_k_k_cap = np.zeros((self.k_w+1,))
    self.sig_k_k_cap = np.zeros((self.k_w+1,))

    self.sm = np.zeros((self.k_w,))

    # Default tuned prior hyperparameters of the attended and unattended
    # Log-Normal distributions
    self.alpha_0 = [6.4113e+02, 4.0434e+03]
    self.beta_0 = [3.7581e+02, 6.2791e+03]
    self.mu_0 = [-0.3994, -1.5103]

    self.rho_d = [1.7060, 0.64395]
    self.mu_d = [-0.3994, -1.5103]

  def tune(self, r1, r2):
    """A more user friendly name, to mirror super class' name."""
    return self.tune_log_normal_priors(r1, r2)

  def tune_log_normal_priors(self, r1, r2):
    """Tune the prior distributions' parameters on some initial data.

    Find the MLE estimate of the log normal parameters for the attended and
    unattended log-normal distributions. Source1 MUST be the attended
    speaker in the training phase

    Args:
        r1: an initial 1d vector of correlations for the attended speaker.
        r2: an initial 1d vector of correlations for the unattended speaker.

    """

    # Normalize correlations between 0 and 1.
    abs_r1 = np.absolute(np.asarray(r1) + self._offset)
    abs_r2 = np.absolute(np.asarray(r2) + self._offset)

    n = abs_r1.shape[0]

    # Compute the mean and precision of the attended speaker r-values
    u_a = np.sum(abs_r1)/n
    v_a = np.sum((abs_r1-u_a)**2)/n
    rho_a = 1/np.log(v_a/u_a**2 + 1)
    mu_a = np.log(u_a) - 0.5/rho_a

    # Compute the mean and precision of the unattended speaker r-values
    # These equations implement the parameter estimation in the Wikipedia page
    #   https://en.wikipedia.org/wiki/Log-normal_distribution#Estimation_of_parameters
    # using the sample mean and variances.
    u_u = np.sum(abs_r2)/n
    v_u = np.sum((abs_r2-u_u)**2)/n
    rho_u = 1/np.log(v_u/u_u**2 + 1)  # This is actually 1.0/variance
    mu_u = np.log(u_u) - 0.5/rho_u

    # Initialised attended and unattended Log-Normal distribution parameters
    # From Behtash: mu is the mean of the log-normal and rho is the "precision
    # parameter", which is the inverse of the variance. It is notationally more
    # convenient to parameterize the log-normal density with the inverse of the
    # variance.
    #
    self.rho_d = [rho_a, rho_u]
    self.mu_d = [mu_a, mu_u]

    self.mu_0 = [mu_a, mu_u]

    # tuned prior hyperparameters of the attended and unattended Log-Normal
    # distributions, these were computed by the UMD researchers on their data by
    # cross-validation. They appear hardcoded in the original MATLAB code and
    # seem to produce fine results.
    self.alpha_0 = [6.4113e+02, 4.0434e+03]
    self.beta_0 = [3.7581e+02, 6.2791e+03]

  def attention(self, r1, r2):
    """Compute the attentional state after receiving two new correlations.

    Returns the mean and error bounded prediction of the attentional state
    given the history of the correlation values for the two speakers and the
    previous attention, which are stored in AttentionDecoderCorr, along with the
    two new correlation values r1, r2, corresponding to speaker 1 and 2.

    Args:
        r1: a new correlation value for speaker 1.
        r2: a new correlation value for speaker 2.

    Returns:
        Tuple: (mean, lowerbound, upperbound)
    """

    self.calls += 1
    self.r1.append(np.abs(r1 + self._offset))
    self.r2.append(np.abs(r2 + self._offset))

    if self.calls >= self.k_w:
      r1 = np.array(self.r1[-self.k_w:])
      r2 = np.array(self.r2[-self.k_w:])

      z = np.array(self.z_smoothed[-self.k_w:])
      eta = np.array(self.eta_smoothed[-self.k_w:])

      # Begin Outer EM loop
      for _ in range(self.outer_iter):
        # Calculating epsilon_k's in the current iteration (E-Step)
        p_11 = (1.0/r1)*np.sqrt(self.rho_d[0])*np.exp(
            -0.5*self.rho_d[0]*(np.log(r1)-self.mu_d[0])**2)

        p_12 = (1.0/r1)*np.sqrt(self.rho_d[1])*np.exp(
            -0.5*self.rho_d[1]*(np.log(r1)-self.mu_d[1])**2)

        p_21 = (1.0/r2)*np.sqrt(self.rho_d[1])*np.exp(
            -0.5*self.rho_d[1]*(np.log(r2)-self.mu_d[1])**2)

        p_22 = (1.0/r2)*np.sqrt(self.rho_d[0])*np.exp(
            -0.5*self.rho_d[0]*(np.log(r2)-self.mu_d[0])**2)

        p = 1.0/(1.0+np.exp(-z))

        ep = (p*p_11*p_21)/(p*p_11*p_21+(1.0-p)*p_12*p_22)

        # distribution parameters update (M-step)
        self.mu_d[0] = (np.sum(ep*np.log(r1)+(1.0-ep)*np.log(r2)) +
                        self.k_w*self.mu_0[0])/(2.0*self.k_w)

        self.mu_d[1] = (np.sum(ep*np.log(r2)+(1.0-ep)*np.log(r1)) +
                        self.k_w*self.mu_0[1])/(2.0*self.k_w)

        self.rho_d[0] = (2.0*self.k_w*self.alpha_0[0])/ \
            (np.sum(ep*((np.log(r1)-self.mu_d[0])**2)+
                    (1.0-ep)*((np.log(r2)-self.mu_d[0])**2))+
             self.k_w*(2.0*self.beta_0[0]+(self.mu_d[0]-self.mu_0[0])**2))

        self.rho_d[1] = (2.0*self.k_w*self.alpha_0[1])/ \
            (np.sum(ep*((np.log(r2)-self.mu_d[1])**2)+
                    (1.0-ep)*((np.log(r1)-self.mu_d[1])**2))+
             self.k_w*(2.0*self.beta_0[1]+(self.mu_d[1]-self.mu_0[1])**2))

        # begin inner EM loop
        for _ in range(self.inner_iter):
          # Filtering
          for k in range(1, self.k_w+1):
            self.z_k_k_1[k] = self.lambda_state*self.z_k_k[k-1]
            self.sig_k_k_1[k] = self.lambda_state**2*self.sig_k_k[k-1]+eta[k-1]

            # Newton's Algorithm
            for _ in range(self.newton_iter):
              self.z_k_k[k] = self.z_k_k[k]- \
                  (self.z_k_k[k] - self.z_k_k_1[k] -
                   self.sig_k_k_1[k]*(ep[k-1] -
                                      np.exp(self.z_k_k[k])/
                                      (1+np.exp(self.z_k_k[k]))))/ \
                  (1 + self.sig_k_k_1[k]*np.exp(self.z_k_k[k])/
                   ((1+np.exp(self.z_k_k[k]))**2))

            self.sig_k_k[k] = 1.0/ (1.0/self.sig_k_k_1[k] +
                                    np.exp(self.z_k_k[k])/
                                    ((1+np.exp(self.z_k_k[k]))**2))

          # Smoothing
          self.z_k_k_cap[self.k_w] = self.z_k_k[self.k_w]
          self.sig_k_k_cap[self.k_w] = self.sig_k_k[self.k_w]
          # sig_k_k_cap vs sig_k_k?

          for k in range(self.k_w):
            self.sm[k] = self.sig_k_k[k]*self.lambda_state/self.sig_k_k_1[k+1]

            self.z_k_k_cap[k] = self.z_k_k[k] + self.sm[k]*(self.z_k_k_cap[k+1]-
                                                            self.z_k_k_1[k+1])

            self.sig_k_k_cap[k] = self.sig_k_k[k] + self.sm[k]**2* \
            (self.sig_k_k_cap[k+1]-self.sig_k_k_1[k+1])

          self.z_k_k[0] = self.z_k_k_cap[0]
          self.sig_k_k[0] = self.sig_k_k_cap[0]

          eta = ((self.z_k_k_cap[1:]-self.z_k_k_cap[:-1])**2+
                 self.sig_k_k_cap[1:]+self.sig_k_k_cap[:-1]-
                 2.0*self.sig_k_k_cap[1:]*self.sm+2*self.b_0)/(1+2*(self.a_0+1))

        z = self.z_k_k_cap[1:]

      # Updated the z's and eta's
      self.z_smoothed += list(self.z_k_k_cap[1:])
      self.eta_smoothed += list(eta)
      self.z_k_k[0] = self.z_k_k_cap[1]
      self.z_dyn.append(self.z_smoothed[-1 - self.k_f])
      self.eta_dyn.append(self.eta_smoothed[-1 - self.k_f])

      return (1.0/(1+np.exp(-self.z_dyn[-1])),
              1.0/(1+np.exp(-self.z_dyn[-1]-self.c0*np.sqrt(self.eta_dyn[-1]))),
              1.0/(1+np.exp(-self.z_dyn[-1]+self.c0*np.sqrt(self.eta_dyn[-1]))))
    return (0.5, 0.5, 0.5)  # No information, so return undecided.


def create_attention_decoder(type_name, window_step=100, frame_rate=100.0,
                             ssd_offset=0.0):
  """Creates any of the attention decoders, based on a name string.

  Args:
    type_name: One of wta, stepped or ssd to indicate the desired decoder type.
    window_step:  How many frames between the start of each window
    frame_rate: The sampling rate of frames in frames/second.
    ssd_offset: How much to offset the values of the input correlation, in order
      to prevent correlations going negative, which doesn't fit the log-normal
      model.

  Returns:
    The desired type of Attention Decoder.
  """
  if type_name == 'wta':
    return AttentionDecoder()
  elif type_name == 'stepped' or type_name == 'step':
    return StepAttentionDecoder()
  elif type_name == 'ssd':
    outer_iter = 20
    inner_iter = 1
    newton_iter = 10
    fs_corr = window_step * float(frame_rate) / 2.0

    return StateSpaceAttentionDecoder(outer_iter, inner_iter, newton_iter,
                                      fs_corr, offset=ssd_offset)
  raise ValueError('Unknown type (%s) requested from create_attention_decoder' %
                   type_name)
