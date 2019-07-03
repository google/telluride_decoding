# Lint as: python2, python3
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

"""TF code to decode an MEG/EEG signal.

TF models and code to predict MEG/EEG signals from their input audio features,
or vice versa.

ToDo(malcolmslaney): Add automatic normalization from Falak.
ToDo(malcolmslaney): Split classes into separate files.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import re
import sys

from absl import app
from absl import flags
from absl import logging

from telluride_decoding import cca
from telluride_decoding import utils

import numpy as np
import six
from six.moves import range
import tensorflow as tf

FLAGS = flags.FLAGS

# Data parameters
flags.DEFINE_enum('data', 'tfrecords',
                  ['simulated', 'simple', 'test', 'tfrecords'],
                  'Dataset to use for this experiment.')
flags.DEFINE_integer('pre_context', 0,
                     'Number of frames of context before prediction')
flags.DEFINE_integer('post_context', 0,
                     'Number of frames of context before prediction')
flags.DEFINE_integer('output_pre_context', 0,
                     'Number of frames of pre context for output features')
flags.DEFINE_integer('output_post_context', 0,
                     'Number of frames of post context for output features')
flags.DEFINE_integer('min_context', 0,
                     'Minimum number of frames of context for '
                     'prediction')
flags.DEFINE_string('input_field', 'mel_spectrogram',
                    'Input field to use for predictions.')
flags.DEFINE_string('output_field', 'envelope',
                    'Output field to predict.')
flags.DEFINE_string('train_file_pattern', '',
                    'A regular expression for picking training files.')
flags.DEFINE_string('test_file_pattern', '',
                    'A regular expression for picking testing files.')
flags.DEFINE_string('validate_file_pattern', '',
                    'A regular expression for picking validation files.')
flags.DEFINE_string('check_file_pattern', '',
                    'A regular expression to check file integrity.')
flags.DEFINE_string('tfexample_dir',
                    None,
                    'location of generic TFRecord data')
flags.DEFINE_bool('random_mixup_batch',
                  False,
                  'Mixup the data, so labels are random, for testing.')
flags.DEFINE_float('input_gain', 1.0,
                   'Multiply the input by this gain factor.')

# Network parameters
flags.DEFINE_enum('dnn_regressor', 'fullyconnected',
                  ['fullyconnected', 'tf', 'linear', 'linear_with_bias', 'cca'],
                  'DNN regressor code to use for this experiment.')
flags.DEFINE_string('hidden_units', '20-20',
                    'Number of hidden layers in regressor')
flags.DEFINE_float('dropout', 0.0,
                   'The dropout rate, between 0 and 1. E.g. "rate=0.1" '
                   'would drop 10% of input units.')
flags.DEFINE_float('regularization_lambda', 0.1,
                   'Regularization parameter for the parameter estimates'
                   ' needed for linear regression.')
flags.DEFINE_float('learning_rate', 0.05,
                   'The initial learning rate for the ADAM optimizer.')
flags.DEFINE_enum('loss', 'mse', ['mse', 'pearson'],
                  'The type of loss to use in the training step.')
flags.DEFINE_enum('context_method', 'new', ['old', 'new'],
                  'Switch to control temporal window approach.')
flags.DEFINE_bool('batch_norm', False,
                  'Switch to enable batch normalization in the network.')

# Basic experiment parameters
flags.DEFINE_integer('steps', 40000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 512,
                     'Number of frames (with context) per minibatch')
flags.DEFINE_string('decoder_model_dir', '/tmp/tf',
                    'Location of event logs and checkpoints.')
flags.DEFINE_integer('prefetch_buffer_size', 100,
                     'Number of elements to prefretch')
flags.DEFINE_integer('shuffle_buffer_size', 100000,
                     'Number of elements to shuffle')
flags.DEFINE_integer('run', 0,
                     'Just for parallel testing... which run # is this.')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory in which to save the model.')

# Test experiment parameters
flags.DEFINE_bool('debug', False, 'Turn on informational debug print stmts.')
flags.DEFINE_integer('num_input_channels', 1,
                     'Number of input channels in test simulations.')
######################### Brain Data Classes ##############################
# A generic class for reading brain decoding data. This class reads in the
# data, adds temporal context and prepares the data for a TF dataset.


class BrainData(object):
  """Basic object describing the data we read and use for regression."""

  def __init__(self, in_fields, out_field,
               pre_context=0,
               post_context=0,
               output_pre_context=0,
               output_post_context=0,
               initial_batch_size=1000,
               final_batch_size=1000,
               repeat_count=-1,
               shuffle_buffer_size=1000):
    """Describe the type of data we are using in this experiment.

    This class encapsulates everything we know about the dataset, so we can
    later generate training, eval and testing subsets.

    Args:
      in_fields: A list of fields from data used as input to regression.
      out_field: A single field name to predict
      pre_context: Number of input samples before the current time in regression
      post_context: Number of input samples after the current time in regression
      output_pre_context: Number of output samples before the current time
        in regression
      output_post_context: Number of output samples after the current time
        in regression
      initial_batch_size: Number of samples to use before adding context
      final_batch_size: Size of minibatch passed to estimator
      repeat_count: Number of times to repeat the data when streaming it out
      shuffle_buffer_size: Number of samples to accumulate before shuffling
    Raises:
      ValueError for bad parameter values.
    """
    if not in_fields:
      raise ValueError('Must specify at least one input field.')
    if not out_field:
      raise ValueError('Must specify an output field.')
    if pre_context < 0:
      raise ValueError('pre_context must be >= 0')

    self.in_fields = in_fields
    self.out_field = out_field
    self.pre_context = pre_context
    self.post_context = post_context
    self.output_pre_context = output_pre_context
    self.output_post_context = output_post_context
    self.initial_batch_size = initial_batch_size
    self.final_batch_size = final_batch_size
    self.repeat_count = repeat_count
    self.shuffle_buffer_size = shuffle_buffer_size
    self.all_files = None   # Initialize a cache for this list if needed.
    self.use_saved_data = False
    self.initialize_dataset()

  def initialize_dataset(self):
    pass

  def get_data_file_names(self, filter1=None, filter2=None):
    """Get the data pathnames for this dataset.

    Just a dummy list of names by default for classes that synthesize data.
    Real datasets will need to specialize this function to return the real file
    names.

    Args:
      filter1: Dummy filter parameters for subclasses
      filter2: Dummy filter parameters for subclasses

    Returns:
      A list of file pathnames.
    """
    del filter1
    del filter2
    return ['Dummy',]  # Return a placeholder so derived classes work

  def filter_filenames(self, mode):
    """Filter all available files based on the experiment mode (train, test...)

    Depending on the training/testing mode, filter the available files into
    a list that we use for this stage.

    Args:
      mode: Arbitrary, but currently one of {train, valid, test}.  This mode
        determines which flag is used to provide the file_pattern

    Implied FLAGS:
      train_file_pattern, validate_file_pattern, test_file_pattern:
        These are regular expressions which filter the names.

    Returns:
      A list of filenames to be used in this phase of the program.

    Raises:
      ValueError for bad parameter values.
    """
    if mode not in set(['test', 'valid', 'train']):
      raise ValueError('mode must be one of test, valid or train')
    filename_list = self.get_data_file_names(mode)
    if mode.startswith('test'):
      pattern_re = re.compile(FLAGS.test_file_pattern)
    elif mode.startswith('valid'):
      pattern_re = re.compile(FLAGS.validate_file_pattern)
    elif mode.startswith('train'):
      if FLAGS.train_file_pattern == 'allbut':
        pattern_re = re.compile('')
      else:
        pattern_re = re.compile(FLAGS.train_file_pattern)
    else:
      pattern_re = re.compile('')

    if mode == 'train' and FLAGS.train_file_pattern == 'allbut':
      # Must specify some pattern for test and validate if using allbut.
      if not (FLAGS.test_file_pattern and FLAGS.validate_file_pattern):
        raise ValueError('both test and validate must be specified if using'
                         'allbut pattern')
      test_re = re.compile(FLAGS.test_file_pattern)
      validate_re = re.compile(FLAGS.validate_file_pattern)
      filename_list = [f for f in filename_list if not (test_re.search(f) or
                                                        validate_re.search(f))]
    else:
      filename_list = [f for f in filename_list if pattern_re.search(f)]
    logging.info('Using %d files for %s.', len(filename_list), mode)
    logging.info(' Files for %s are: %s', mode, filename_list)
    return filename_list

  def create_dataset(self, mode='train', temporal_context=True):
    """Create the full TF dataset, ready to feed an estimator.

    Args:
      mode: One of {train, eval, test} to determine how to set up the
        full stream.
      temporal_context: Flag that controls whether we add temporal context to
        the data. Normally true, but set to false to extract the original data
        without context (for debugging and prediction.)

    Returns:
      Two items in a tuple: the dataset iterator, and the actual tf.data.dataset
      object.

    Raises:
      ValueError for bad parameter values.
    """
    if self.use_saved_data:
      saved_dataset = tf.data.Dataset.from_tensor_slices(
          (self.saved_input_data, self.saved_output_data))
      additional_context = (self.pre_context + self.post_context +
                            self.output_pre_context + self.output_post_context)
      if temporal_context and additional_context > 0:
        saved_dataset = self.add_temporal_context(saved_dataset)
      return self.finalize_dataset(mode, saved_dataset)

    filename_list = self.filter_filenames(mode)
    if not filename_list:
      raise ValueError('No files to process in mode %s' % mode)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filename_list)

    # Map over all the filename (strings) using interleave so we get some extra
    # randomness.  And each prepare_data call only applies to one file, so we
    # don't extend the temporal context across files.
    interleaved_dataset = filename_dataset.interleave(
        lambda x: self.prepare_data(x, mode, temporal_context=temporal_context),
        len(filename_list))
    return self.finalize_dataset(mode, interleaved_dataset)

  def preserve_test_data(self, input_data, output_data):
    """Put some data into a dataset for testing.

    Args:
      input_data: data used as the input feature
      output_data: data used as the output data to be predicted

    Raises:
      TypeError for bad parameter values.
    """
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    self.use_saved_data = True
    self.saved_input_data = input_data
    self.saved_output_data = output_data
    self.num_input_channels = input_data.shape[1]
    self.num_output_channels = output_data.shape[1]
    self.features = {
        'input': tf.io.FixedLenFeature([input_data.shape[1],], tf.float32),
        'output': tf.io.FixedLenFeature([output_data.shape[1],], tf.float32),
    }

  def finalize_dataset(self, mode, input_dataset):
    """Do all the work we need to do prepare the dataset for serving.

    Args:
      mode: train or testing mode, determines whether data is shuffled)
      input_dataset: The actual dataset to prepare.

    Returns:
      A tuple consisting of:
        The TF element from which to pull data
        The final dataset object.
    """

    # First shuffle the data (and repeat it) for better SGD performance.
    if mode == 'train' and self.shuffle_buffer_size > 0:
      repeated_dataset = input_dataset.repeat(self.repeat_count)
      shuffled_dataset = repeated_dataset.shuffle(self.shuffle_buffer_size)
    else:
      shuffled_dataset = input_dataset

    # Then batch the data into minibatches.
    # Drop the remainder so testing is easier (no odd sized batches). Losing a
    # few samples at the end shouldn't matter for real (big) datasets.
    batched_dataset = shuffled_dataset.batch(self.final_batch_size,
                                             drop_remainder=True)

    if FLAGS.random_mixup_batch:
      logging.warning('finalize_dataset: Mixing up the batches of data for '
                      'testing!!!!')
      def mixup_batch(x, y):
        """Mixup the order of the labels so data is mismatched. For baseline."""
        return x, tf.random.shuffle(y)
      batched_dataset = batched_dataset.map(mixup_batch)

    # tf.Estimator API needs the first element (input) of the dataset to be a
    # dictionary with the input labeled as 'x'.  Add it here.
    estimator_dataset = batched_dataset.map(lambda x, y: ({'x': x}, y),
                                            num_parallel_calls=32)
    logging.info('Create_dataset: the %s estimator_dataset is: %s',
                 mode, estimator_dataset)

    # Do I still need to do these things for an estimator???
    dataset_iterator = estimator_dataset.make_one_shot_iterator()
    dataset_next_element = dataset_iterator.get_next()
    return dataset_next_element, estimator_dataset

  def prepare_data(self, filenames, mode, temporal_context=True):
    """Prepare a specific example of data for this dataset.

    Dataset creation function that takes filename(s) and outputs the proper
    fields from the dataset (no context yet).  This base method is only useful
    when reading/parsing TFRecord data.  Otherwise, specialize.

    Args:
      filenames: a tensor containing filename from which to read the data.
      mode: the training/eval/test mode for this dataset, if needed.
      temporal_context: Should we add the temporal context to the input data?

    Returns:
      A two-stream dataset, one for input and the other the labels. Batch size
      of 1 at this point.

    Raises:
      TypeError for bad parameter values.
    """
    if not isinstance(filenames, tf.Tensor):
      raise TypeError('filenames must be a tensor')
    del mode  # Unused by the generic dataset type

    filename_dataset = tf.data.Dataset.from_tensors(filenames)
    raw_proto_dataset = tf.data.TFRecordDataset(filename_dataset,
                                                num_parallel_reads=32)
    parsed_data = raw_proto_dataset.map(self.parse_tfrecord,
                                        num_parallel_calls=32)
    if temporal_context and (self.pre_context + self.post_context > 0):
      parsed_data = self.add_temporal_context(parsed_data)
    return parsed_data

  def parse_tfrecord(self, raw_proto):
    """Dataset map function that parses a TFRecord example."""
    if isinstance(self.in_fields, six.string_types):
      self.in_fields = [self.in_fields,]

    # https://stackoverflow.com/questions/41951433/tensorflow-valueerror-shape-must-be-rank-1-but-is-rank-0-for-parseexample-pa
    parsed_features = tf.io.parse_example([raw_proto], self.features)

    in_data = tf.concat([parsed_features[k] for k in self.in_fields], axis=1)
    out_data = parsed_features[self.out_field]

    in_data = tf.reshape(in_data, (-1,), name='input_reshape')
    out_data = tf.reshape(out_data, (-1,), name='output_reshape')
    return in_data, out_data

  def add_temporal_context(self, dataset_without_context):
    """Add context to a datstream.

    Create a dataset stream from files of TFRecords, containing input and
    output data. This dataset is unique because we add temporal context to the
    input data, so the output depends on the input over a time window.  We do
    this using the dataset so we can create the context on the fly (and not
    precompute it and save it in a much larger file.)

    Args:
      dataset_without_context: dataset to which we will add temporal context.
        This dataset consists of a two (unnamed) streams.

    External args:
      self.pre_context - Number of frames to prepend to the input data.
      self.post_context - Number of frames to append after the current frame.

    Returns:
      The new dataset with the desired temporal context.

    Raises:
      TypeError for bad parameter values.
    """
    def window_one_stream(x, pre_context, post_context):
      """Create extra temporal context for one stream of data."""
      total_context = pre_context + 1 + post_context
      channels = x.shape[1]
      padded_x = tf.concat((tf.zeros((pre_context, channels), dtype=x.dtype),
                            x,
                            tf.zeros((post_context, channels),
                                     dtype=x.dtype)),
                           axis=0)
      new_x = (tf.data.Dataset.from_tensors(padded_x)
               .apply(tf.data.experimental.unbatch())
               .window(size=total_context, shift=1, drop_remainder=True)
               .flat_map(lambda x: x.batch(total_context))
               .map(lambda x: tf.reshape(x, (-1,), name='wos_reshape_old'),
                    num_parallel_calls=32))
      return new_x

    def window_one_stream_new(x, pre_context, post_context):
      """Create extra temporal context for one stream of data."""
      total_context = pre_context + 1 + post_context
      channels = x.shape[1]
      padded_x = tf.concat((tf.zeros((pre_context, channels), dtype=x.dtype),
                            x,
                            tf.zeros((post_context, channels),
                                     dtype=x.dtype)),
                           axis=0)
      new_data = tf.contrib.signal.frame(padded_x, total_context,
                                         frame_step=1, axis=0)
      flat_data = tf.reshape(new_data, (-1, total_context*channels),
                             name='wos_reshape_new')
      new_x = tf.data.Dataset.from_tensor_slices(flat_data)
      return new_x

    def window_data(x, y, pre_context=1, post_context=2,
                    output_pre_context=0, output_post_context=0):
      """Create extra temporal context for both input and output streams."""
      if FLAGS.context_method == 'old':
        x_with_context = window_one_stream(x, pre_context, post_context)
        y_with_context = window_one_stream(y, output_pre_context,
                                           output_post_context)
      else:
        x_with_context = window_one_stream_new(x, pre_context, post_context)
        y_with_context = window_one_stream_new(y, output_pre_context,
                                               output_post_context)
      return tf.data.Dataset.zip((x_with_context, y_with_context))

    if not isinstance(dataset_without_context, tf.data.Dataset):
      raise TypeError('dataset for window_data must be a tf.data.Dataset')
    additional_context = (self.pre_context + self.post_context +
                          self.output_pre_context + self.output_post_context)
    if additional_context > 0:
      batched_dataset = dataset_without_context.batch(self.initial_batch_size)
      new_dataset = batched_dataset.flat_map(
          lambda x, y: window_data(   # pylint: disable=g-long-lambda
              x, y,
              pre_context=self.pre_context,
              post_context=self.post_context,
              output_pre_context=self.output_pre_context,
              output_post_context=self.output_post_context))
    else:
      new_dataset = dataset_without_context
    return new_dataset

  def input_fields_width(self):
    """Compute the width of the input.

    Sum up the width of all the fields to pass this to the estimator ---
    *after* adding the temporal context.

    Returns:
      An integer that counts how wide the input feature is (in float32s)

    Raises:
      TypeError for bad parameter values.
    """
    logging.info('input_fields_width type(in_fields): %s', type(self.in_fields))
    if isinstance(self.in_fields, six.string_types):
      self.in_fields = [self.in_fields,]
    for k in self.in_fields:
      if k not in list(self.features.keys()):
        raise TypeError('Can\'t find %s in valid features: %s' %
                        (k, [','.join(list(self.features.keys()))]))
    widths = [self.features[k].shape[0] for k in self.in_fields]
    return sum(widths)*(self.pre_context+1+self.post_context)

  def output_field_width(self):
    assert self.out_field in list(self.features.keys()), (
        'Could not find output_field %s in %s' % (self.out_field,
                                                  self.features.keys()))
    return self.features[self.out_field].shape[0]


def discover_feature_shapes(tfrecord_file_name):
  """Read a TFRecord file, parse one TFExample, and return the structure.

  Args:
    tfrecord_file_name: Where to read the data (just one needed)

  Returns:
    A dictionary of names and tf.io.FixedLenFeatures suitable for
    tf.io.parse_example.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(tfrecord_file_name, six.string_types):
    raise TypeError('discover_feature_shapes: input must be a string filename.')

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(tfrecord_file_name)
    itr = dataset.make_one_shot_iterator()

    with tf.compat.v1.Session() as sess:
      a_record = sess.run(itr.get_next())

  an_example = tf.train.Example.FromString(a_record)
  assert isinstance(an_example, tf.train.Example)

  feature_keys = list(an_example.features.feature.keys())

  shapes = {}
  for k in feature_keys:
    feature_list = an_example.features.feature[k]
    if feature_list.float_list.value:
      dimensionality = len(feature_list.float_list.value)
      feature_type = tf.float32
    elif feature_list.int64_list.value:
      dimensionality = len(feature_list.int64_list.value)
      feature_type = tf.int64
    elif feature_list.bytes_list.value:
      dimensionality = len(feature_list.byte_list.value)
      feature_type = tf.str
    shapes[k] = tf.io.FixedLenFeature([dimensionality,], feature_type)
  return shapes


class TFExampleData(BrainData):
  """Generic dataset consisting of TFExamples in multiple files."""

  def initialize_dataset(self):
    self.get_data_file_names(None)
    self.features = discover_feature_shapes(self.all_files[0])

  def get_data_file_names(self, mode):
    """Get some files with TFRecord BrainData.

    Args:
      mode: training or testing.. ignored for now.

    Returns:
      A list of path names to the desired data.
    """
    del mode   # Not needed here.
    if self.all_files:    # Check to see if we have already Walked the tree...
      return self.all_files
    logging.info('Reading TFExample data from %s', FLAGS.tfexample_dir)
    self.all_files = []
    exp_data_dir = FLAGS.tfexample_dir
    for (path, _, files) in tf.io.gfile.walk(exp_data_dir):
      self.all_files += [
          path + '/' + f
          for f in files
          if f.endswith('.tfrecords') and '-bad-' not in f
      ]
    logging.info('Found %d files for TFExample data analysis.',
                 len(self.all_files))
    assert self.all_files    # Should be not empty

    return self.all_files


######################### Create Linear Regressor ###########################


def create_linear_model_fn(features, labels, mode, init_w=None, init_b=None):
  """This function creates a linear-regressor TF network for an Estimator.

  Use the init_w and init_b parameters to specify precomputed values for the
  linear equation
    y = wx + b

  Args:
    features: A dictionary from tf.data.Dataset, with an 'x' field which
      contains the feature data.
    labels: A tensor with labels (values to predict)
    mode: One of training, eval, infer
    init_w: The initial value for the weight matrix in the regressor
    init_b: The initial value for the bias vector in the regressor

  Returns:
    A tf estimator spec used by the estimator model.

  Raises:
    ValueError and/or TypeError for bad parameter values.
  """
  if not isinstance(features, dict):
    raise TypeError('Features input to create_linear_model_fn must be a dict.')
  if not isinstance(features['x'], tf.Tensor):
    raise TypeError('Features[x] to create_linear_model_fn must be a tensor.')
  if not (isinstance(labels, tf.Tensor) or labels is None):
    raise ValueError('Labels for create_linear_model_fn must be a tensor or '
                     'None')
  logging.info('Building model for %s with features: %s', mode, features)
  # Build a linear model and predict values
  if isinstance(init_w, np.ndarray) or isinstance(init_w, list):
    init_w = tf.constant(np.array(init_w, dtype=np.float32))
  if isinstance(init_b, np.ndarray) or isinstance(init_b, list):
    init_b = tf.constant(np.array(init_b, dtype=np.float32))
  logging.info('create_linear_model_fn initializers: %s %s', init_w, init_b)
  with tf.compat.v1.variable_scope('linear_regressor'):
    w = tf.compat.v1.get_variable('w', dtype=tf.float32, initializer=init_w)
    b = tf.compat.v1.get_variable('b', dtype=tf.float32, initializer=init_b)
    y = tf.matmul(features['x'], w) + b
  if mode == 'train' or mode == 'eval':
    # Loss sub-graph
    loss, _, _, metrics = compute_and_summarize_losses(labels, y)
    # Training sub-graph
    global_step = tf.compat.v1.train.get_global_step()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-15)
    train = tf.group(optimizer.minimize(loss),
                     tf.compat.v1.assign_add(global_step, 1))
  else:
    train = None
    loss = None
    metrics = None
  # EstimatorSpec connects subgraphs we built to the appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train,
      eval_metric_ops=metrics)


# Function to estimate linear regressor from a dataset.
# Regression calculation defined here:
# https://stackoverflow.com/questions/45959112/get-coefficients-of-a-linear-regression-in-tensorflow
def calculate_regressor_parameters_from_dataset(dataset, lamb=0.1,
                                                use_offset=True,
                                                max_mini_batches=10000):
  """Estimate the parameters for a linear regressor from a dataset.

  Finds A to solve the equation:
    Ax = y
  This routine reads the dataset, calculating the necessary covariances, and
  then returns the solution A to the equation above. Use these values to preload
  a linear regressor estimator.

  Note, lambda in this routine corresponds to the shrinkage parameter gamma
  in Blankertz et al. NeuroImage 58 (2011) 814-825, specificially used in Eq 13.

  Args:
    dataset: The tf.dataset from which to read data (dictionary item 'x' and
      labels). Dataset is read once (so be sure repeat=1)
    lamb: Regularization parameters for the least squares estimates.
    use_offset: Whether to include the additive bias offset
    max_mini_batches: How many minibatches to pull from the dataset when
      estimating the data's covariances.

  Returns:
    The estimated A and b matrices. As well as the two covariance matrices for
    debugging, and the optimal shrinkage parameter.

  Raises:
    ValueError and/or TypeError for bad parameter values.

  ToDo(malcolmslaney): Implement equation 13 of Blankertz, based on the
    derivation from this paper:
      http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset input to calculate_regressor_parameters_from_'
                    'database must be a tf.data.Dataset object')
  if lamb < 0.0:
    raise ValueError('regularization lambda must be >= 0')
  cov_xx = 0  # Accumulate sum of x^T x for all minibatches
  cov_xy = 0  # Accumulate sum of x^T y for all minibatches
  data_iter = dataset.make_one_shot_iterator()
  data_element = data_iter.get_next()
  num_mini_batches = 0
  xmax = 0
  sum_xx = 0.0
  sum_x = 0.0
  with tf.compat.v1.Session() as sess:
    while num_mini_batches < max_mini_batches:
      try:
        (x_dict, y) = sess.run(data_element)
        x = x_dict['x']
        n_row = x.shape[0]
        if use_offset:
          # Append a column of 1s so we can compute an offset.
          x = np.hstack((x, np.ones((n_row, 1), dtype=x.dtype)))
        cov_xx += x.T.dot(x)
        cov_xy += x.T.dot(y)
        num_mini_batches += 1
        xmax = max(np.amax(abs(x)), xmax)
        sum_x += np.sum(x)
        sum_xx += np.sum(np.square(x))
      except tf.errors.OutOfRangeError:
        logging.info('Done training linear regressor -- epoch limit reached '
                     'after %d mini batches', num_mini_batches)
        break
  n_col = cov_xx.shape[0]
  shrink_scale = lamb * np.mean(np.linalg.eigvals(cov_xx))
  logging.debug('Shrinkage scaling is %g, %g, %g', shrink_scale, lamb,
                np.mean(np.trace(cov_xx)/n_col))
  # Equation 12 of Blankertz.  Shink eigenvalues toward the mean.
  cov_xx = (1-lamb)*cov_xx + shrink_scale * np.identity(n_col)
  solution = np.linalg.solve(cov_xx, cov_xy)
  if use_offset:
    return solution[0:-1, :], solution[-1:, :], cov_xx, cov_xy
  else:
    return solution, np.zeros((1,), dtype=solution.dtype), cov_xx, cov_xy


def evaluate_regressor_from_dataset(a, b, dataset,
                                    testing_data_file=None):
  """Evaluate a pretrained regressor using data from a dataset.

  Using a linear regressor computed by create_regressor_from_dataset, evaluate
  its performance.

  Args:
    a: A matrix in Ax + b = y
    b: b matrix in Ax + b = y
    dataset: TF dataset to be used to evaluate quality
    testing_data_file: For debugging: a file where the input data can be dumped

  Returns:
    average_error, average_power, pearson, num_samples

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset object must be a tf.data.Dataset object.')
  a = np.asarray(a)
  b = np.asarray(b)
  data_iter = dataset.make_one_shot_iterator()
  data_element = data_iter.get_next()
  total_error = 0.0
  num_samples = 0
  e_x = 0.0
  e_y = 0.0
  e_xy = 0.0
  e_x2 = 0.0
  e_y2 = 0.0
  if testing_data_file:
    fp_x = open(testing_data_file+'_x.txt', 'w')
    fp_y = open(testing_data_file+'_y.txt', 'w')
  else:
    fp_x = None
    fp_y = None
  with tf.compat.v1.Session() as sess:
    while True:
      try:
        (x_dict, y) = sess.run(data_element)
        x = x_dict['x']
        if fp_x:
          for i in range(x.shape[0]):
            fp_x.write(' '.join([str(f) for f in x[i, :].tolist()]) + '\n')
        if fp_y:
          for i in range(y.shape[0]):
            fp_y.write(' '.join([str(f) for f in y[i, :].tolist()]) + '\n')
        y_est = np.matmul(x, a) + b
        err = y - y_est
        total_error += np.sum(err*err, axis=None)
        num_samples += err.shape[0]
        e_x += np.sum(y_est)
        e_x2 += np.sum(np.square(y_est))
        e_y += np.sum(y)
        e_y2 += np.sum(np.square(y))
        e_xy += np.sum(y * y_est)
      except tf.errors.OutOfRangeError:
        if fp_x:
          logging.info('Wrote testing data to file: %s and %s',
                       testing_data_file+'_x.txt', testing_data_file+'_y.txt')
          fp_x.close()
        if fp_y:
          fp_y.close()
        break
  # From: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  #         #For_a_population
  pearson = (((e_xy/num_samples) - ((e_x/num_samples)*(e_y/num_samples))) /
             np.sqrt(e_x2/num_samples - (e_x/num_samples)**2) /
             np.sqrt(e_y2/num_samples - (e_y/num_samples)**2))
  average_error = total_error/num_samples
  average_power = e_y2/num_samples
  return average_error, average_power, pearson, num_samples


def create_linear_estimator(dataset, lamb=0.1, use_offset=True):
  """Create a linear estimator, initializing parameters with data from dataset.

  Use the provided dataset (and regularization parameter) to initialize the
  regressor, creating weight and bias matrices that can be plugged into a graph
  as needed by the Estimator.

  Args:
    dataset: A tf.dataset that provides the data needed to estimate the linear
      regressor.  All the data is read once (be sure to set repeat=1) to find
      the optimal parameters.
    lamb: The regularization parameter (to add to the diagonal)
    use_offset: Add a constant offset (b) to the regression: ax + b = y

  Returns:
    A tf.Estimator instance, initialized to implement the optimal linear
    regressor for the provided data.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError('dataset input must be a tf.data.Dataset object.')
  w, b, _, _ = calculate_regressor_parameters_from_dataset(
      dataset, lamb, use_offset=use_offset)
  logging.info('Linear regression W has shape: %s and standard deviation of %g',
               w.shape, np.std(w))
  (average_error, average_power,
   pearson, num_samples) = evaluate_regressor_from_dataset(w, b, dataset)
  logging.info('Linear regressor got an error of %g per sample or an SNR of %g',
               average_error, 10*math.log10(average_power/average_error))
  logging.info('Pearson correlation is %g from %d samples.',
               pearson, num_samples)

  def my_linear_model(features, labels, mode):
    return create_linear_model_fn(features, labels, mode, init_w=w, init_b=b)
  estimator = tf.estimator.Estimator(model_fn=my_linear_model,
                                     model_dir=FLAGS.decoder_model_dir)

  return estimator


######################### Define the loss functions ###########################


def compute_and_summarize_losses(labels, predictions):
  """Define the TF network pieces to calculate the losses we care about.

  Args:
    labels: The true values
    predictions: The values we have predicted.

  Returns:
    loss: The loss (determined by FLAGS.loss) used for optimization
    mse: The mean squared error loss.
    correlation: Pearson's correlation matrix between labels and predictions
    metrics: a dictionary of loss statistics
  """
  with tf.compat.v1.variable_scope('losses'):
    mse = tf.compat.v1.losses.mean_squared_error(
        labels, predictions,
        loss_collection=None,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    pearson_r = utils.pearson_correlation_graph(tf.transpose(predictions),
                                                tf.transpose(labels))
    if FLAGS.loss == 'mse':
      loss = mse
    else:
      loss = 1 - pearson_r[0, 1]

  with tf.compat.v1.variable_scope('train'):
    tf.compat.v1.summary.scalar('mse', mse)
    tf.compat.v1.summary.tensor_summary('pearson_correlation_matrix', pearson_r)
    # This only works for a scalar prediction output.
    tf.compat.v1.summary.scalar('pearson_correlation', pearson_r[0, 1])
    tf.compat.v1.summary.scalar('train_loss', loss)
    params = experiment_parameters()
    tf.compat.v1.summary.text('params', tf.convert_to_tensor(params))

  # Assemble evaluation metrics. These get evaluated and saved only during eval.
  # These show up in the eval job in Tensorboard.
  # batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
  metrics = {
      'test/mse': tf.compat.v1.metrics.mean(mse),
      'test/pearson_correlation_matrix':
          tf.compat.v1.metrics.mean_tensor(pearson_r),
      'test/pearson_correlation': tf.compat.v1.metrics.mean(pearson_r[0, 1]),
      'test/loss': tf.compat.v1.metrics.mean(loss),
  }

  return loss, mse, pearson_r, metrics

######################### Define the DNN Regressor ############################


# From: https://www.tensorflow.org/guide/custom_estimators
def create_dnn_regressor(features, labels, mode, params):
  """Define a TF DNN with N hidden layers in the default graph.

  Args:
    features: A dictionary of TF.fixed_len_features describing the input data.
    labels: What we want to predict
    mode: Is this the training or test phase?
    params: A dictionary of parameters describing the network.

  Returns:
    A tf.estimator.EstimatorSpec
  """
  # Create N fully connected layers
  net = tf.compat.v1.feature_column.input_layer(features,
                                                params['feature_columns'])
  net = net * FLAGS.input_gain
  with tf.compat.v1.variable_scope('dnn_regressor'):
    for units in params['hidden_units']:
      logging.info('Adding a layer with %d units.', units)
      net = tf.layers.dense(
          net,
          units=units,
          kernel_initializer=tf.glorot_uniform_initializer(),
          activation=None)
      if FLAGS.batch_norm:
        net = tf.contrib.layers.batch_norm(
            net, center=True, scale=True,
            is_training=(mode == tf.estimator.ModeKeys.TRAIN))
      net = tf.nn.relu(net)
      if FLAGS.dropout > 0.0:
        net = tf.layers.dropout(inputs=net, rate=FLAGS.dropout,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Final fully connected layer for final regression computation.
    logging.info('Adding a final layer with %d units.', params['output_units'])
    prediction = tf.layers.dense(
        net,
        units=params['output_units'],
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=None,
        name='prediction')

  # Compute predictions.
  if mode == tf.estimator.ModeKeys.PREDICT:
    prediction_dict = {'predictions': prediction}
    return tf.estimator.EstimatorSpec(mode, predictions=prediction_dict)

  # Compute loss.
  loss, _, pearson_r, metrics = compute_and_summarize_losses(
      FLAGS.input_gain*labels, prediction)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  with tf.compat.v1.variable_scope('train'):
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # Use learning rate from tensorflow/python/estimator/canned/dnn.py
      optimizer = tf.compat.v1.train.AdagradOptimizer(
          learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(
          loss, global_step=tf.compat.v1.train.get_global_step())

  # From: https://stackoverflow.com/questions/45353389/printing-extra-training-
  #       metrics-with-tensorflow-estimator
  logging_hook = tf.estimator.LoggingTensorHook(
      {
          'tf_hook_loss': loss,
          'tf_hook_pearson_correlation': pearson_r
      },
      every_n_iter=500)

  # TODO(malcolmslaney): Does eval_metrics matter at this point?
  return tf.estimator.EstimatorSpec(
      mode, loss=loss, train_op=train_op, training_hooks=[logging_hook],
      eval_metric_ops=metrics)


def create_train_estimator(telluride_data, hidden_units=(20, 20), steps=10000):
  """Create and train the model.

  Create and train an estimator that predicts an output (i.e. intensity) from
  the input data (i.e. EEG). The exact type of this regressor is based on the
  value of FLAGS.dnn_regressor, which can have these values:

    tf: Use the normal tf.estimator.DNNRegressor, which uses a MSE error.

    linear: Calculate an optimal linear regressor, create a network to implement
    it, and initialize the weights with the optimal parameters

    fullyconnected: Create a network of fully connected layers.

  Args:
    telluride_data: A dataset from which to pull training data
    hidden_units: a list of integers, specifying the number of hidden units at
      the output of each fully-connected layer.
    steps: The number of iterations to make when training the model

  Returns:
    model: a TF.Estimator with a pretrained model
    eval_results: How well this model does on the evaluation data.

  Raises:
    ValueError and/or TypeError for bad parameter values.
  """
  if not isinstance(telluride_data, BrainData):
    raise TypeError('telluride_data input must be a BrainData object')
  if steps <= 0:
    raise ValueError('steps must be >= 0')
  output_units = telluride_data.output_field_width()
  def create_dataset_iterator(ds, mode):
    my_iterator, _ = ds.create_dataset(mode=mode)
    return my_iterator
  train_dataset_iterator = lambda: create_dataset_iterator(telluride_data,  # pylint: disable=g-long-lambda
                                                           mode='train')
  test_dataset_iterator = lambda: create_dataset_iterator(telluride_data,  # pylint: disable=g-long-lambda
                                                          mode='test')

  feature_columns = [
      tf.compat.v1.feature_column.numeric_column(
          key='x', shape=[
              telluride_data.input_fields_width(),
          ]),
  ]

  # Build a DNNRegressor, with the specified hidden layers, with the feature
  # columns defined above as input.
  # Config change not necessary, but reduces the amount of logging output
  log_steps = min(steps/100, 10000)
  logging.info('Requesting logging every %d steps.', log_steps)
  my_config = tf.estimator.RunConfig().replace(log_step_count_steps=log_steps)
  my_config = tf.estimator.RunConfig().replace(save_summary_steps=log_steps)

  # Define the model configuration
  assert output_units == 1 or FLAGS.dnn_regressor == 'fullyconnected'
  if FLAGS.dnn_regressor == 'tf':
    # Do it with the default DNNRegressor defined by TF.estimator
    model = tf.estimator.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=feature_columns,
        config=my_config,
        model_dir=FLAGS.decoder_model_dir,
        # MEAN seems to perform worse than SUM?!?
        loss_reduction=tf.compat.v1.losses.Reduction.SUM
        )
  elif FLAGS.dnn_regressor.startswith('linear'):
    telluride_data.repeat_count = 1
    telluride_data.shuffle_buffer_size = 0
    _, training_dataset = telluride_data.create_dataset(mode='train')
    if FLAGS.dnn_regressor == 'linear':
      use_offset = False
    else:
      use_offset = True
    model = create_linear_estimator(training_dataset,
                                    FLAGS.regularization_lambda,
                                    use_offset=use_offset)
    steps = 1
    # TODO(malcolmslaney) return dataset directly, and remove lambda using it
    def create_test_dataset(telluride_data, mode):
      dataset_iterator, _ = telluride_data.create_dataset(mode=mode)
      return dataset_iterator
    # The call to model.evaluate creates the /eval runs for Tensorboard
    #  with the metrics returned and passed to the estimator.
    num_batches_to_evaluate = 100
    eval_results = model.evaluate(lambda: create_test_dataset(telluride_data,   # pylint: disable=g-long-lambda
                                                              mode='test'),
                                  steps=num_batches_to_evaluate)
  elif FLAGS.dnn_regressor.startswith('cca'):
    telluride_data.repeat_count = 1
    telluride_data.shuffle_buffer_size = 0
    _, training_dataset = telluride_data.create_dataset(mode='train')
    model = cca.create_cca_estimator(training_dataset,
                                     regularization=FLAGS.regularization_lambda)
    steps = 1
    # TODO(malcolmslaney) return dataset directly and remove the lambda using it
    def create_test_dataset(telluride_data, mode):
      dataset_iterator, _ = telluride_data.create_dataset(mode=mode)
      return dataset_iterator
    # The call to model.evaluate creates the /eval runs for Tensorboard
    #  with the metrics returned and passed to the estimator.
    num_batches_to_evaluate = 100
    eval_results = model.evaluate(lambda: create_test_dataset(telluride_data,   # pylint: disable=g-long-lambda
                                                              mode='test'),
                                  steps=num_batches_to_evaluate)
  else:
    # Fully connected DNN estimator code with custom losses.
    model = tf.estimator.Estimator(
        model_fn=create_dnn_regressor,
        params={
            'feature_columns': feature_columns,
            'hidden_units': hidden_units,
            'output_units': output_units,
        },
        config=my_config,
        model_dir=FLAGS.decoder_model_dir)
  if steps > 0:
    # See if we have training to do. (Minimal steps for cca and linear which are
    # precomputed.)
    train_spec = tf.estimator.TrainSpec(train_dataset_iterator, max_steps=steps)
    eval_spec = tf.estimator.EvalSpec(test_dataset_iterator, steps=2*log_steps,
                                      name='eval_spec',
                                      start_delay_secs=0,
                                      throttle_secs=0)
    logging.info('Training the estimator for %d steps....', steps)
    (eval_results, _) = tf.estimator.train_and_evaluate(model,
                                                        train_spec, eval_spec)
    logging.info('....Finished training the model %s', model)
  return model, eval_results


def evaluate_performance(telluride_data, model, dataset_mode='test'):
  """Evaluate performance of a model.

  Get some test data (same as training data so far) and the model
  predictions. This is more difficult since we need to draw the data, store it,
  and then make predictions.  (The estimator API doesn't make this easy, so
  we have to grab the data outselves.)
  Args:
    telluride_data: A BrainData object that specifies the data
    model: The TF Estimator model object
    dataset_mode: Which mode of the dataset to create.

  Returns:
    RMS error, and the (Pearson) correlation between the predictions and the
    labels.decoder_

    The call to model.evaluate creates the /eval runs for Tensorboard
    with the metrics returned and passed to the estimator.

   Raises:
     TypeError for bad parameter values.
  """
  if not isinstance(telluride_data, BrainData):
    raise TypeError('input dataset must be a BrainData object.')
  if not isinstance(model, tf.estimator.Estimator):
    raise TypeError('model input must be a tf.Estimator object.')
  def create_test_dataset(telluride_data, mode):
    dataset_iterator, _ = telluride_data.create_dataset(mode=mode)
    return dataset_iterator
  num_batches_to_evaluate = None  # evaluate on entire test set
  results = model.evaluate(lambda: create_test_dataset(telluride_data,   # pylint: disable=g-long-lambda
                                                       mode=dataset_mode),
                           steps=num_batches_to_evaluate,
                           name='final_test')
  logging.info('evaluate_performance results: %s', results)
  return results


def save_estimator_model(estimator_model, dataset, saved_model_dir):
  """Save the estimator model for later prediction.

  Args:
    estimator_model: A tf.estimator.Estimator model to be saved
    dataset: The dataset with which the estimator model was trained. (This is
      needed so we know how to replicate the temporal context that is needed for
      this model.)
    saved_model_dir: Where to store the model code, a directory

  Raises:
    TypeError for bad parameter values.
  """
  # TODO(malcolmslaney) add output context so saving CCA models works.
  if not isinstance(estimator_model, tf.estimator.Estimator):
    raise TypeError('estimator_model must be a tf.Estimator object.')
  if not isinstance(dataset, BrainData):
    raise TypeError('dataset must be a BrainData object.')

  def _serving_input_receiver_fn(dataset):
    """Create the function that adds context to the input."""
    total_context = dataset.pre_context + 1 + dataset.post_context
    width = dataset.input_fields_width()//total_context
    input_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=[None, width],
                                       name='input_example_tensor')
    input_dict = {'input': input_placeholder}
    padded_x = tf.concat((tf.zeros((dataset.pre_context, width),
                                   dtype=input_placeholder.dtype),
                          input_placeholder,
                          tf.zeros((dataset.post_context, width),
                                   dtype=input_placeholder.dtype)),
                         axis=0)
    new_data = tf.contrib.signal.frame(padded_x, total_context,
                                       frame_step=1, axis=0)
    flat_data = tf.reshape(new_data, (-1, total_context*width),
                           name='wos_reshape_new')
    feature_dict = {'x': flat_data}
    recv = tf.estimator.export.ServingInputReceiver(feature_dict,
                                                    input_dict)
    return recv

  estimator_model.export_saved_model(
      saved_model_dir, lambda: _serving_input_receiver_fn(dataset))
  logging.info('saving estimator model to: %s', saved_model_dir)

######################### Main Program ##############################


def run_experiment(input_dataset, hidden_units=(20, 20), steps=40000):
  """Run one whole experiment using the given dataset and model params.

  Args:
    input_dataset: The source of the data for this experiment
    hidden_units: a list of integers specifying the desired number of hidden
      units after each layer.
    steps:  The number of optimization steps to compute.

  Returns:
    All summary data as a dictionary of results.
  """
  (model, train_results) = create_train_estimator(input_dataset, steps=steps,
                                                  hidden_units=hidden_units)

  results = evaluate_performance(input_dataset, model)

  params = experiment_parameters()
  results_file = FLAGS.decoder_model_dir + '/results.txt'
  with tf.io.gfile.GFile(results_file, 'w') as fp:
    for k in results:
      if isinstance(results[k], np.ndarray):
        fp.write('%s: Final Testing/%s is %s\n' %
                 (params, k, ' '.join([str(f) for f in np.reshape(results[k],
                                                                  (-1))])))
      else:
        fp.write('%s: Final Testing/%s is %g\n' % (params, k, results[k]))
    if train_results:
      for k in train_results:
        if isinstance(train_results[k], np.ndarray):
          fp.write('%s: Final Validation/%s is %s\n' %
                   (params, k,
                    ' '.join([str(f) for f in np.reshape(train_results[k],
                                                         (-1))])))
        else:
          fp.write('%s: Final Validation/%s is %g\n' % (params, k,
                                                        train_results[k]))

    logging.info('Wrote summary results to %s', results_file)
    if FLAGS.saved_model_dir:
      save_estimator_model(model, input_dataset, FLAGS.saved_model_dir)
  return train_results, model


def add_command_line_summary(args, summary_name='CommandLineArgs: '):
  """Not working (yet) attempt to save parameters so tensorboard gets them."""
  arg_tensor = tf.convert_to_tensor(summary_name + ' '.join(args))
  tf.summary.text('CommandLine', arg_tensor)


def experiment_parameters():
  """Turn the list of parameters into a readable string for summaries."""
  s = ''
  s += 'data=%s' % FLAGS.data
  s += ',pre_context=%d' % FLAGS.pre_context
  s += ',post_context=%d' % FLAGS.post_context
  s += ',steps=%d' % FLAGS.steps
  s += ',batch_size=%d' % FLAGS.batch_size
  s += ',hidden_units=%s' % FLAGS.hidden_units
  s += ',dropout=%g' % FLAGS.dropout
  s += ',batch_norm=%s' % FLAGS.batch_norm
  s += ',regularization_lambda=%g' % FLAGS.regularization_lambda
  s += ',learning_rate=%g' % FLAGS.learning_rate
  s += ',loss=%s' % FLAGS.loss
  s += ',input_field=%s' % FLAGS.input_field
  s += ',output_field=%s' % FLAGS.output_field
  s += ',train_file_pattern=%s' % FLAGS.train_file_pattern
  s += ',test_file_pattern=%s' % FLAGS.test_file_pattern
  s += ',validate_file_pattern=%s' % FLAGS.validate_file_pattern
  s += ',dnn_regressor=%s' % FLAGS.dnn_regressor
  s += ',random_mixup_batch=%s' % FLAGS.random_mixup_batch
  if FLAGS.data == 'simulated':
    s += ',simulated_unattended_gain=%g' % FLAGS.simulated_unattended_gain
    s += ',simulated_noise_level=%g' % FLAGS.simulated_noise_level
  return s


def count_tfrecords(tfrecord_file_name):
  """Count and validate TFRecords in an input file.

  Args:
    tfrecord_file_name: file to check

  Returns:
    Tuple consisting of valid records and whether an exception was found.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(tfrecord_file_name, six.string_types):
    raise TypeError('tfrecord_file_name must be a string.')

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(tfrecord_file_name)
    itr = dataset.make_one_shot_iterator()

    with tf.compat.v1.Session() as sess:
      record_count = 0
      next_element = itr.get_next()
      while True:
        try:
          a_record = sess.run(next_element)
          an_example = tf.train.Example.FromString(a_record)
          assert isinstance(an_example, tf.train.Example)
          record_count += 1
        except tf.errors.OutOfRangeError:
          break
        except:    # pylint: disable=bare-except
          return record_count, True
  return record_count, False


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.debug:
    # Will need to add --logtostderr when running standalone inside Google
    logging.set_verbosity(logging.DEBUG)

  if FLAGS.pre_context + 1 + FLAGS.post_context < FLAGS.min_context:
    FLAGS.post_context = FLAGS.min_context - (FLAGS.pre_context + 1)

  if not FLAGS.decoder_model_dir.endswith('/'):
    FLAGS.decoder_model_dir = FLAGS.decoder_model_dir + '/'
  tf.io.gfile.makedirs(FLAGS.decoder_model_dir)
  params = experiment_parameters()
  logging.info('Params string is: %s', params)
  logging.info('TFRecord data from: %s', FLAGS.tfexample_dir)
  logging.info('TF GPU available: %s', tf.test.is_gpu_available())

  # Convert the string for hidden units into a list of numbers.
  if not FLAGS.hidden_units:
    logging.info('Setting the number of hidden units to []')
    hidden_units = []
  else:
    hidden_units = [int(x) for x in FLAGS.hidden_units.split('-')]
  logging.info('Defining a network with these hidden units: %s', hidden_units)

  if FLAGS.check_file_pattern:
    exp_data_dir = FLAGS.tfexample_dir
    all_files = []
    for (path, _, files) in tf.io.tf.io.gfile.walk(exp_data_dir):
      all_files += [
          os.path.join(path, f)
          for f in files
          if f.endswith('.tfrecords')
      ]
    logging.info('Found %d files for TFExample data analysis.', len(all_files))
    for f in all_files:
      logging.info('%s: %d', f, count_tfrecords(f))
    return
  # Create the dataset we'll use for this experiment.
  if FLAGS.data == 'tfrecords':
    dataset = TFExampleData(FLAGS.input_field, FLAGS.output_field,
                            pre_context=FLAGS.pre_context,
                            post_context=FLAGS.post_context,
                            output_pre_context=FLAGS.output_pre_context,
                            output_post_context=FLAGS.output_post_context,
                            final_batch_size=FLAGS.batch_size,
                            shuffle_buffer_size=FLAGS.shuffle_buffer_size)
  else:
    logging.info('Error: Unknown dataset for experiment: %s', FLAGS.data)
    sys.exit(1)

  add_command_line_summary(experiment_parameters())

  run_experiment(dataset, steps=FLAGS.steps, hidden_units=hidden_units)

if __name__ == '__main__':
  app.run(main)
