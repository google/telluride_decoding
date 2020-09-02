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
"""

import os
import random
import re
import sys

from absl import logging

import numpy as np

from telluride_decoding import preprocess
import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


brain_data_print = sys.stdout  # Feel free to redirect this elsewhere.

# pylint: disable=g-long-lambda

######################### Brain Data Classes ##############################


class BrainData(object):
  """Basic object describing the data we read and use for regression.

  A generic class for reading brain decoding data. This class reads in the
  data, adds temporal context and prepares the data as a TF dataset for
  processing by the rest of the decoding system. Use the create_dataset method
  to get a tf.data.Dataset object given a BrainData object.

  The resulting dataset object represents a stream of two-ples, consisting of:
    input_dictionary, output_data
  where the input dictionary has three keys
    input_1, input_2 (might be empty), and attended_speaker (might be empty).
  """

  def __init__(self, in_fields, out_field,
               frame_rate,
               pre_context=0,
               post_context=0,
               in2_fields=None,
               in2_pre_context=0,
               in2_post_context=0,
               attended_field=None,
               initial_batch_size=1000,
               final_batch_size=1000,
               repeat_count=1,
               shuffle_buffer_size=1000,
               data_dir=None,
               data_pattern='',
               train_file_pattern='',
               validate_file_pattern='',
               test_file_pattern=''):
    """Describes the type of data we are using in this experiment.

    This class encapsulates everything we know about the dataset, so we can
    later generate training, eval and testing subsets.

    Args:
      in_fields: A list of fields from a dataset used as input to regression.
      out_field: A single field name to predict (also dataset field).
      frame_rate: Sample rate of the data, needed for preprocessing filters.
      pre_context: Number of input samples before the current time in
        regression.
      post_context: Number of input samples after the current time in
        regression.
      in2_fields: A second list of fields for methods that take two inputs.
      in2_pre_context: Number of samples of input 2 before the current time.
        in regression.
      in2_post_context: Number of samples of input 2 after the current time
        in regression.
      attended_field: TFRecord feature name that says which speaker is being
        attended.
      initial_batch_size: Number of samples to use before adding context.
        Longer is better because you have fewer edge effects.
      final_batch_size: Size of minibatch passed to estimator.
      repeat_count: Number of times to repeat the data when streaming it out.
      shuffle_buffer_size: Number of samples to accumulate before shuffling.
      data_dir: Where file-based data classes look for data.
      data_pattern: String that must be in all filenames used here.
      train_file_pattern: A regular expression that selects the training files.
        Note, the special string "allbut" specifies that all files not selected
        for validation or testing should be used for testing.  Furthermore,
        specifying allbut_NN specifies that a random NN of the files be used
        for training (so we can test amount of data vs performance.)
      validate_file_pattern: A regular expression that selects the validation
        files.
      test_file_pattern: A regular expression that selects the testing files.

    Raises:
      ValueError for bad parameter values.
    """
    logging.info('BrainData initialization: %s, %s @ %gHz -> %s',
                 in_fields, in2_fields, frame_rate, out_field)
    if not in_fields:
      raise ValueError('Must specify at least one input field.')
    if not out_field:
      raise ValueError('Must specify an output field.')
    if frame_rate < 0:
      raise ValueError('frame_rate must be >= 0')
    if pre_context < 0:
      raise ValueError('pre_context must be >= 0')
    if post_context < 0:
      raise ValueError('post_context must be >= 0')

    if isinstance(in_fields, str):
      in_fields = [in_fields,]
    self.in1_fields = in_fields
    if isinstance(in2_fields, str) and in2_fields:
      in2_fields = [in2_fields,]
    self.in2_fields = in2_fields
    self.out_field = out_field
    self.frame_rate = frame_rate
    self.in1_pre_context = pre_context
    self.in1_post_context = post_context
    self.in2_pre_context = in2_pre_context
    self.in2_post_context = in2_post_context
    self.attended_field = attended_field
    self.initial_batch_size = initial_batch_size
    self.final_batch_size = final_batch_size
    self.repeat_count = repeat_count
    self.shuffle_buffer_size = shuffle_buffer_size
    self.data_dir = data_dir
    self.data_pattern = data_pattern
    self.train_file_pattern = train_file_pattern
    self.validate_file_pattern = validate_file_pattern
    self.test_file_pattern = test_file_pattern
    # Internal state
    self.use_saved_data = False
    self._cached_file_names = []  # Initialize cache for this list if needed.
    self.all_files()   # preload data files so we know what the data looks like.

  def all_files(self, max_count=0):
    """Returns a list of files available for this class."""
    if not self._cached_file_names:
      # Load the potential files if we haven't already.
      self._get_data_file_names()
      if self._cached_file_names:
        random.shuffle(self._cached_file_names)  # Shuffle them once.
    if max_count > 0 and len(self._cached_file_names) > max_count:
      return self._cached_file_names[:max_count]
    return self._cached_file_names

  def set_file_patterns(self, train, validate, test):
    logging.info('brain_data setting file patterns to %s, %s, %s',
                 train, validate, test)
    self.train_file_pattern = train
    self.validate_file_pattern = validate
    self.test_file_pattern = test

  def create_dataset(self, mode='train', temporal_context=True):
    """Creates the full TF dataset, ready to feed an estimator.

    This class should be specialized.  The basic flow is:
      Select the file names for this mode (if needed).
      Read in each file (interleaved) and do the following operations:
        Parse the data.
        Add temporal context.
      Finalize the dataset by:
        Shuffle the data.
        Assemble into mini batches.

    Args:
      mode: One of {train, eval, test} to determine how to set up the
        full stream.
      temporal_context: Flag that controls whether we add temporal context to
        the data. Normally true, but set to false to extract the original data
        without context (for debugging and prediction.)
    """
    raise NotImplementedError

  def _get_data_file_names(self):
    """Get the data pathnames for this dataset.

    Just a dummy list of names by default for classes that synthesize data.
    Real datasets will need to specialize this function to return the real file
    names.

    Caches a list of file pathnames. In this generic case an empty list.
    """
    self._cached_file_names = []  # No files by default

  def filter_file_names(self, mode):
    """Filters all available files based on the experiment mode (train, test...)

    Depending on the training/testing mode, filter the available files into
    a list that we use for this stage.

    Args:
      mode: Arbitrary, but currently one of {train, validate, test}.  This mode
        determines which flag is used to provide the file_pattern.

    Implied flags from the class object:
      train_file_pattern, validate_file_pattern, test_file_pattern:
        These are regular expressions that filter the returned names.

    Returns:
      A list of filenames to be used in this phase of the program.

    Raises:
      ValueError for bad parameter values.
    """
    if mode == 'program_test':
      mode = 'test'
    if mode not in set(['test', 'validate', 'train']):
      raise ValueError('mode must be one of test, validate or train')
    filename_list = self.all_files()
    if not isinstance(filename_list, list):
      raise TypeError('Filename_list is a %s, not a list.' %
                      type(filename_list))
    logging.info('Filter_file_names: filename_list: %s', filename_list)
    logging.info('Filter_file_names: train_file_pattern: %s',
                 self.train_file_pattern)
    logging.info('Filter_file_names: validate_file_pattern: %s',
                 self.validate_file_pattern)
    logging.info('Filter_file_names: test_file_pattern: %s',
                 self.test_file_pattern)
    if mode.startswith('test'):
      pattern_re = re.compile(self.test_file_pattern)
    elif mode.startswith('validate'):
      pattern_re = re.compile(self.validate_file_pattern)
    else:   # Only valid option left is 'train'
      if self.train_file_pattern == 'allbut':
        pattern_re = re.compile('')
      else:
        pattern_re = re.compile(self.train_file_pattern)

    if mode == 'train' and self.train_file_pattern.startswith('allbut'):
      # Must specify some pattern for test and validate if using allbut.
      if not (self.test_file_pattern and self.validate_file_pattern):
        raise ValueError('Both test and validate must be specified if using '
                         'allbut pattern')
      test_re = re.compile(self.test_file_pattern)
      validate_re = re.compile(self.validate_file_pattern)
      filename_list = [f for f in filename_list if not (test_re.search(f) or
                                                        validate_re.search(f))]
      if self.train_file_pattern.startswith('allbut_'):
        allbut = self.train_file_pattern.replace('allbut_', '', 1)
        if allbut.isdigit():
          count = int(allbut)
        else:
          raise ValueError('allbut_ spec must be an integer, not %s.' % allbut)
        if count < len(filename_list):
          logging.info('Reducing list of %d files to %d.',
                       len(filename_list), count)
          filename_list = filename_list[:count]
      logging.info('filter_file_names: post filename_list %s', filename_list)
    else:
      filename_list = [f for f in filename_list if pattern_re.search(f)]
    logging.info('Using %d files for %s.', len(filename_list), mode)
    logging.info(' Files for %s are: %s', mode, filename_list)
    return filename_list

  def final_shuffle_and_batch(self, mode, input_dataset, mixup_batch=False):
    """Does all the work we need to do prepare the dataset for serving.

    Args:
      mode: Train or testing mode, which determines whether data is shuffled.
      input_dataset: The actual TF dataset to prepare.
      mixup_batch: Whether the inputs and outputs are randomized with respect
        to each other.

    Returns:
      The final dataset object.  This dataset has two components: a dictionary
      of model inputs (input_1, input_2 and attended_speaker), and a single
      output field.
    """

    # TODO Look into tf.data cache or snapshot
    # First shuffle the data (and repeat it) for better SGD performance.
    logging.info('final_shuffle_and_batch mode %s and batching %s',
                 mode, mixup_batch)
    if mode == 'train':
      repeated_dataset = input_dataset.repeat(self.repeat_count)

      if self.shuffle_buffer_size > 0:
        shuffled_dataset = repeated_dataset.shuffle(self.shuffle_buffer_size)
      else:
        shuffled_dataset = repeated_dataset
    elif mode == 'program_test':
      shuffled_dataset = input_dataset
    else:
      # Shuffle the data in test or eval mode too so we get better stats.
      if self.shuffle_buffer_size > 0:
        shuffled_dataset = input_dataset.shuffle(self.shuffle_buffer_size)
      else:
        shuffled_dataset = input_dataset

    # Then batch the data into minibatches.
    # Drop the remainder so testing is easier (no odd sized batches). Losing a
    # few samples at the end shouldn't matter for real (big) datasets.
    batched_dataset = shuffled_dataset.batch(self.final_batch_size,
                                             drop_remainder=True)

    if mixup_batch:
      print('final_shuffle_and_batch: Mixing up the batches of data '
            'for testing!!!!', file=brain_data_print)
      logging.warning('final_shuffle_and_batch: Mixing up the batches of data '
                      'for testing!!!!')
      def mixup_batch_function(x, x2, y, a):
        """Mixup the order of the labels so data is mismatched. For baseline."""
        return x, tf.random.shuffle(x2), tf.random.shuffle(y), a
      batched_dataset = batched_dataset.map(mixup_batch_function)

    # Convert the four-tuple to a two-tuple: a dictionary for the inputs, and
    # the output.
    final_dataset = batched_dataset.map(
        lambda x, x2, y, a: ({'input_1': x,
                              'input_2': x2,
                              'attended_speaker': a}, y),
        num_parallel_calls=32)
    logging.info('Create_dataset: the %s final_dataset is: %s',
                 mode, final_dataset)
    return final_dataset

  def add_temporal_context(self, dataset_without_context):
    """Adds context to a datstream.

    Create a dataset stream from files of TFRecords, containing input and
    output data. This dataset is unique because we add temporal context to the
    input data, so the output depends on the input over a time window.  We do
    this using the dataset so we can create the context on the fly (and not
    precompute it and save it in a much larger file.)

    Args:
      dataset_without_context: dataset to which we will add temporal context.
        This dataset consists of a four (unnamed) streams (input_1, input_2,
        output, and attention).

    External args:
      self.in1_pre_context - Number of frames to prepend to the input data.
      self.in1_post_context - Number of frames to append after the current
        frame.
      self.in2_pre_context - Number of frames to prepend to the second input.
      self.in2_post_context - Number of frames to append after the second input.

    Returns:
      The new dataset with the desired temporal context.

    Raises:
      TypeError for bad parameter values.
    """

    def window_one_stream_new(x, pre_context, post_context):
      """Create extra temporal context for one stream of data."""
      logging.info(' Window_one_stream: adding %d and %d frames of context '
                   'to stream.', pre_context, post_context)
      total_context = pre_context + 1 + post_context
      channels = x.shape[1]
      logging.info(' Window_one_stream: %s channels.', channels)
      padded_x = tf.concat((tf.zeros((pre_context, channels), dtype=x.dtype),
                            x,
                            tf.zeros((post_context, channels),
                                     dtype=x.dtype)),
                           axis=0)
      new_data = tf.signal.frame(padded_x, total_context, frame_step=1, axis=0)
      flat_data = tf.reshape(new_data, (-1, total_context*channels),
                             name='window_one_stream_reshape_new')
      new_x = tf.data.Dataset.from_tensor_slices(flat_data)
      return new_x

    def window_data(x, x2, y, a, pre_context=0, post_context=0,
                    in2_pre_context=0, in2_post_context=0):
      """Creates extra temporal context for both input streams."""
      x_with_context = window_one_stream_new(x, pre_context, post_context)
      x2_with_context = window_one_stream_new(x2, in2_pre_context,
                                              in2_post_context)
      y_with_context = window_one_stream_new(y, 0, 0)
      a = tf.data.Dataset.from_tensor_slices(a)
      return tf.data.Dataset.zip((x_with_context, x2_with_context,
                                  y_with_context, a))

    if not isinstance(dataset_without_context, tf.data.Dataset):
      raise TypeError('dataset for window_data must be a tf.data.Dataset')
    additional_context = (self.in1_pre_context or self.in1_post_context or
                          self.in2_pre_context + self.in2_post_context)
    if additional_context:
      batched_dataset = dataset_without_context.batch(self.initial_batch_size)
      new_dataset = batched_dataset.flat_map(
          lambda x, x2, y, a: window_data(
              x, x2, y, a,
              pre_context=self.in1_pre_context,
              post_context=self.in1_post_context,
              in2_pre_context=self.in2_pre_context,
              in2_post_context=self.in2_post_context))
    else:
      new_dataset = dataset_without_context
    return new_dataset

  def input_fields_width(self, input_number=1):
    """Computes the width of the input.

    Sum up the width of all the fields to pass this to the estimator ---
    *after* adding the temporal context.

    Args:
      input_number: Set to either 1 or 2, which determines whether this function
        is calculating the feature weight for the first or second feature data.
    Returns:
      An integer that counts how wide the input feature is (in float32s).

    Raises:
      TypeError for bad parameter values.
    """
    if input_number != 1 and input_number != 2:
      raise ValueError('Only 1st or 2nd input is supported here.')
    if input_number == 1:
      fields = self.in1_fields
    else:
      fields = self.in2_fields
    logging.info('input_fields_width (%d) type(in_fields) is %s with value %s',
                 input_number, type(fields), fields)
    if isinstance(fields, str) and fields:
      fields = [fields,]
    if fields:
      for k in fields:
        if k not in list(self.features.keys()):
          raise TypeError('Can\'t find **%s** in valid features: %s' %
                          (k, [','.join(list(self.features.keys()))]))
      widths = [self.features[k].shape[0] for k in fields]
    else:
      widths = [1]
    if input_number == 1:
      return sum(widths)*(self.in1_pre_context+1+self.in1_post_context)
    else:
      return sum(widths)*(self.in2_pre_context+1+self.in2_post_context)

  def output_field_width(self):
    if self.out_field not in list(self.features.keys()):
      raise ValueError('Could not find output_field **%s** in %s' %
                       (self.out_field, self.features.keys()))
    return self.features[self.out_field].shape[0]


class TestBrainData(BrainData):
  """Dataset which produces fixed (saved) values, useful for testing."""

  def create_dataset(self, mode='train', temporal_context=True,
                     mixup_batch=False):
    """Creates the full TF dataset, ready to feed an estimator.

    This is the default entry into this class, creating a dataset for training,
    testing, or validation, depending on the mode.

    Args:
      mode: One of {train, eval, test} to determine how to set up the
        full stream.
      temporal_context: Flag that controls whether we add temporal context to
        the data. Normally true, but set to false to extract the original data
        without context (for debugging and prediction.)
      mixup_batch: Boolean that specifies whether inputs and outputs are
        shuffled with respect to each other to create baseline.

    Returns:
      The requested tf.data.Dataset object.

    Raises:
      ValueError for bad parameter values.
    """
    if not hasattr(self, 'saved_input_data'):
      raise ValueError('Must call preserve_test_data before create_dataset.')

    saved_dataset = tf.data.Dataset.from_tensor_slices(
        (self.saved_input_data, self.saved_input2_data,
         self.saved_output_data, self.saved_attention_data))
    if temporal_context and (self.in1_pre_context or self.in1_post_context or
                             self.in2_pre_context or self.in2_post_context):
      saved_dataset = self.add_temporal_context(saved_dataset)
    return self.final_shuffle_and_batch(mode, saved_dataset,
                                        mixup_batch=mixup_batch)

  def preserve_test_data(self, input_data, output_data,
                         input2_data=None, attention_data=None):
    """Puts some data into a dataset for testing.

    Args:
      input_data: data used as the input feature. (time x channel).
      output_data: data used as the output data to be predicted.
      input2_data: Optional second input array.
      attention_data: Optional array for attention target signal.

    Raises:
      TypeError for bad parameter values.
    """
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    if input_data.shape[0] != output_data.shape[0]:
      raise ValueError('input shape (%s) and output shape (%s) are not equal.'
                       % (input_data.shape, output_data.shape))
    self.saved_input_data = input_data
    self.saved_output_data = output_data
    self.num_input_channels = input_data.shape[1]
    self.num_output_channels = output_data.shape[1]
    self.features = {
        'input_1': tf.io.FixedLenFeature([input_data.shape[1],], tf.float32),
        'output': tf.io.FixedLenFeature([output_data.shape[1],], tf.float32),
    }

    # Add the optional input_2.
    if input2_data is None:
      input2_data = np.zeros((input_data.shape[0], 1),
                             dtype=input_data.dtype)
    input2_data = np.asarray(input2_data)
    if input_data.shape[0] != input2_data.shape[0]:
      raise ValueError('input shape (%s) and input2 shape (%s) are not equal.'
                       % (input_data.shape, input2_data.shape))
    self.saved_input2_data = input2_data
    self.features['input_2'] = tf.io.FixedLenFeature([input2_data.shape[1],],
                                                     tf.float32)

    # Add the optional attention signal.
    if attention_data is None:
      attention_data = np.zeros((input_data.shape[0], 1),
                                dtype=input_data.dtype)
    attention_data = np.asarray(attention_data)
    if input_data.shape[0] != attention_data.shape[0]:
      raise ValueError('input shape (%s) and attention shape (%s) '
                       'are not equal.'
                       % (input_data.shape, attention_data.shape))
    self.saved_attention_data = attention_data
    self.features['attention'] = tf.io.FixedLenFeature(
        [attention_data.shape[1],], tf.float32)


class TFExampleData(BrainData):
  """Generic dataset consisting of TFExamples in multiple files."""

  def _get_data_file_names(self):
    """Gets the files in data_dir ending with .tfrecords and have data_pattern.

    Walk the directory tree, grabbing all the files that and in ".tfrecords" and
    contain the string indicated by self.data_pattern. We'll filter them into
    training, validation and testing sets later.

    Returns:
      A list of path names to the desired data.
    """
    if not self.data_dir:
      raise ValueError('Missing data_dir in TFExampleData initialization. '
                       'Must specify the source of the data (FLAGS.tfrecords).')

    logging.info('Reading TFExample data from %s, filtering for **%s**',
                 self.data_dir, self.data_pattern)
    if not isinstance(self.data_dir, str):
      raise TypeError('data_dir must be a string, not a %s (**%s**)' %
                      (type(self.data_dir), self.data_dir))
    self._cached_file_names = []
    exp_data_dir = self.data_dir
    for (path, _, files) in tf.io.gfile.walk(exp_data_dir):
      # pylint: disable=g-complex-comprehension
      self._cached_file_names += [
          os.path.join(path, f)
          for f in files
          if (f.endswith('.tfrecords') and
              '-bad-' not in f and
              self.data_pattern in f)
      ]
    logging.info('Found %d files for TFExample data analysis.',
                 len(self._cached_file_names))
    if not self._cached_file_names:
      raise ValueError('Should not have an empty list of data files from %s.' %
                       exp_data_dir)
    self.features = discover_feature_shapes(self._cached_file_names[0])
    logging.info('Discover_feature_shapes found: %s', self.features)

  def create_dataset(self, mode='train', temporal_context=True,
                     mixup_batch=False):
    """Create the full TF dataset, ready to feed an estimator.

    This is the default entry into this class, creating a dataset for training,
    testing, or validation, depending on the mode.

    Args:
      mode: One of {train, eval, test} to determine how to set up the
        full stream.
      temporal_context: Flag that controls whether we add temporal context to
        the data. Normally true, but set to false to extract the original data
        without context (for debugging and prediction.)
      mixup_batch: Whether the inputs and outputs are randomized with respect
        to each other.

    Returns:
      The requested tf.data.dataset object.

    Raises:
      ValueError for bad parameter values.
    """
    filename_list = self.filter_file_names(mode)
    if not filename_list:
      raise ValueError('No files to process in mode %s from %s' %
                       (mode, self.data_dir))
    filename_dataset = tf.data.Dataset.from_tensor_slices(filename_list)

    # Map over all the filename (strings) using interleave so we get some extra
    # randomness.  And each read_data_into_dataset call only applies to one
    # file, so we don't extend the temporal context across files.
    interleaved_dataset = filename_dataset.interleave(
        lambda x: self.read_data_into_dataset(
            x, temporal_context=temporal_context),
        len(filename_list))
    return self.final_shuffle_and_batch(mode, interleaved_dataset,
                                        mixup_batch=mixup_batch)

  def read_data_into_dataset(self, filenames, temporal_context=True):
    """Prepares a specific example of data for this dataset.

    Dataset creation function that takes filename(s) and outputs the proper
    fields from the dataset (no context yet).  This base method is only useful
    when reading/parsing TFRecord data.  Otherwise, specialize.

    Args:
      filenames: a tensor containing one (usual case due to interleave) or more
      filenames from which to read the data.
      temporal_context: Should we add the temporal context to the input data?

    Returns:
      A two-stream dataset, one for input and the other the labels. Batch size
      of 1 at this point.

    Raises:
      TypeError for bad parameter values.
    """
    if not isinstance(filenames, tf.Tensor):
      raise TypeError('filenames must be a tensor')

    filename_dataset = tf.data.Dataset.from_tensors(filenames)
    raw_proto_dataset = tf.data.TFRecordDataset(filename_dataset,
                                                num_parallel_reads=32)
    parsed_data = raw_proto_dataset.map(self.parse_and_select_from_tfrecord,
                                        num_parallel_calls=32)
    if temporal_context and (self.in1_pre_context or self.in1_post_context or
                             self.in2_pre_context or self.in2_post_context):
      parsed_data = self.add_temporal_context(parsed_data)
    return parsed_data

  def preprocess_list(self, name_params_list, frame_rate):
    if not name_params_list:
      return []
    pp_list = []
    for name_param in name_params_list:
      pp_list.append(preprocess.Preprocessor(name_param, frame_rate,
                                             frame_rate))
    return pp_list

  def parse_and_select_from_tfrecord(self, raw_proto):
    """Dataset map function that parses a TFRecord example and select fields.

    Note, this routine has a special hack to create a field called "ones" which
    is always one, and used for cases like CCA which have no output, just two
    inputs.

    Args:
      raw_proto: An example of a TFRecord, in proto format.

    Returns:
      A 4-ple consisting of parsed input, input2, output data, and attended
      direction (if supplied) in Tensors.

      Each tensor consists of one sample point (shape[0]) but the width of each
      data depends on the user's input data request (in1_fields, in2_fields,
      out_field, and attended_field)
    """
    # https://stackoverflow.com/questions/41951433/tensorflow-valueerror-shape-must-be-rank-1-but-is-rank-0-for-parseexample-pa
    parsed_features = tf.io.parse_example([raw_proto], self.features)

    if set(self.in1_fields) - set(parsed_features.keys()):
      raise ValueError('Could not find all desired features (%s) in data (%s)' %
                       (self.in1_fields, parsed_features.keys()))
    in_data = tf.concat([parsed_features[k] for k in self.in1_fields], axis=1)
    in_data = tf.reshape(in_data, (-1,), name='input_reshape')

    if self.out_field == 'ones':
      logging.info('Selecting ones from %s', in_data)
      out_data = in_data[0:1]*0.0 + 1
    else:
      out_data = parsed_features[self.out_field]
    out_data = tf.reshape(out_data, (-1,), name='output_reshape')

    if self.in2_fields:
      for k in self.in2_fields:
        if k not in parsed_features:
          raise ValueError('Could not find %s in parsed_features[%s]' %
                           (k, parsed_features.keys()))
      in2_data = tf.concat([parsed_features[k] for k in self.in2_fields],
                           axis=1)
      in2_data = tf.reshape(in2_data, (-1,), name='input2_reshape')
    else:
      # Fill in dummy data so the dataset maps to come don't get upset.
      # Only need first data element, replicated across batches later.
      # This will need to be done by hand when feeding saved models.
      logging.info('Did not find %s field for input2, so synthesizing one.',
                   self.in2_fields)
      in2_data = in_data[0:1]

    if self.attended_field:
      attended_data = parsed_features[self.attended_field]
      attended_data = tf.reshape(attended_data, (-1,), name='attended_reshape')
    else:
      logging.info('Did not find %s field for attention, so synthesizing one.',
                   self.attended_field)
      # Placeholder.  Just get some 0/1 data into this field. Keep it as a float
      # since the original attend field is a float.
      attended_data = tf.cast(in_data[0:1] > 0, tf.float32)

    return in_data, in2_data, out_data, attended_data

  # TODO Switch to this new parse function so we can do pre-
  # processing on the fly.  Right now it doesn't work yet.
  def parse_and_select_from_tfrecord2(self, raw_proto):
    """Dataset map function that parses a TFRecord example and select fields."""
    # https://stackoverflow.com/questions/41951433/tensorflow-valueerror-shape-must-be-rank-1-but-is-rank-0-for-parseexample-pa
    parsed_features = tf.io.parse_example([raw_proto], self.features)

    self._in1_preprocessors = self.preprocess_list(self.in1_fields,
                                                   self.frame_rate)
    # pylint: disable=g-complex-comprehension
    in_data = tf.concat([tf.py_function(pp.process,
                                        inp=[parsed_features[pp.name]],
                                        Tout=tf.float32)
                         for pp in self._in1_preprocessors], axis=1)
    in_data = tf.reshape(in_data, (-1,), name='input1_reshape')

    if self.in2_fields:
      self._in2_preprocessors = self.preprocess_list(self.in2_fields,
                                                     self.frame_rate)
      # pylint: disable=g-complex-comprehension
      in2_data = tf.concat([tf.py_function(pp.process,
                                           inp=[parsed_features[pp.name]],
                                           Tout=tf.float32)
                            for pp in self._in2_preprocessors], axis=1)
      in2_data = tf.reshape(in2_data, (-1,), name='input2_reshape')
    else:
      in2_data = in_data[0:1]

    self._out_preprocessors = self.preprocess_list([self.out_field],
                                                   self.frame_rate)
    # pylint: disable=g-complex-comprehension
    out_data = tf.concat([tf.py_function(pp.process,
                                         inp=[parsed_features[pp.name]],
                                         Tout=tf.float32)
                          for pp in self._out_preprocessors], axis=1)
    out_data = tf.reshape(out_data, (-1,), name='output_reshape')
    if self.attended_direction:
      attended_data = parsed_features[self.attended_direction]
      attended_data = tf.reshape(attended_data, (-1), name='attended_reshape')
    else:
      attended_data = None
    return in_data, in2_data, out_data, attended_data


def discover_feature_shapes(tfrecord_file_name):
  """Reads a TFRecord file, parse one TFExample, and return the structure.

  Args:
    tfrecord_file_name: Where to read the data (just one needed).

  Returns:
    A dictionary of names and tf.io.FixedLenFeatures suitable for
    tf.io.parse_example.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(tfrecord_file_name, str):
    raise TypeError('discover_feature_shapes: input must be a string filename.')

  dataset = tf.data.TFRecordDataset(tfrecord_file_name)

  for a_record in dataset:
    an_example = tf.train.Example.FromString(a_record.numpy())
    break
  if not isinstance(an_example, tf.train.Example):
    raise TypeError('record from %s should be a tf.train.Example, not %s.' %
                    (tfrecord_file_name, type(an_example)))

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


def count_tfrecords(tfrecord_file_name):
  """Counts and validates the number of TFRecords in an input file.

  Args:
    tfrecord_file_name: File to check.

  Returns:
    Tuple consisting of valid records and whether an exception was found.

  Raises:
    TypeError for bad parameter values.
  """
  if not isinstance(tfrecord_file_name, str):
    raise TypeError('tfrecord_file_name must be a string.')

  dataset = tf.data.TFRecordDataset(tfrecord_file_name)
  record_count = 0
  for a_record in dataset:
    try:
      an_example = tf.train.Example.FromString(a_record.numpy())
      if not isinstance(an_example, tf.train.Example):
        raise TypeError('record from %s should be a tf.train.Example, not %s.' %
                        (tfrecord_file_name, type(an_example)))
      record_count += 1
    except:    # pylint: disable=bare-except
      return record_count, True
  return record_count, False


def create_brain_dataset(data_type, in_fields, out_field, frame_rate,
                         pre_context=0,
                         post_context=0,
                         in2_fields=None,
                         in2_pre_context=0,
                         in2_post_context=0,
                         attended_field=None,
                         initial_batch_size=1000,
                         final_batch_size=1000,
                         repeat_count=1,
                         shuffle_buffer_size=1000,
                         data_dir=None,
                         data_pattern='',
                         train_file_pattern=None,
                         validate_file_pattern=None,
                         test_file_pattern=None):
  """Creates any of the brain datasets that we know about.

  Args:
    data_type: Desired type of dataset.
    in_fields: A list of fields from a dataset used as input to regression.
    out_field: A single field name to predict (also dataset field).
    frame_rate: Sample rate of the data, needed for preprocessing filters.
    pre_context: Number of input samples before the current time in regression.
    post_context: Number of input samples after the current time in regression.
    in2_fields: A second list of fields for methods that take two inputs.
    in2_pre_context: Number of samples of input 2 before the current time
      in regression.
    in2_post_context: Number of samples of input 2 after the current time
      in regression.
    attended_field: Where is the subject attending? This signal is passed
      through the pipeline and is not used until verifying the performance.
    initial_batch_size: Number of samples to use before adding context.
    final_batch_size: Size of minibatch passed to estimator.
    repeat_count: Number of times to repeat the data when streaming it out.
    shuffle_buffer_size: Number of samples to accumulate before shuffling.
    data_dir: Where file-based data classes look for data.
    data_pattern: String that must be in the filename.
    train_file_pattern: A regular expression that selects the training files.
    validate_file_pattern: A regular expression that selects the validation
      files.
    test_file_pattern: A regular expression that selects the testing files.

  Returns:
    The desired type of BrainData
  """
  if not isinstance(data_type, str):
    raise TypeError('create_brain_dataset type must be a string.')
  if frame_rate <= 0:
    raise ValueError('frame_rate must be greater than 0.')
  if (data_type == 'tfrecord' or data_type == 'tfrecords' or
      data_type == 'tfexample'):
    return TFExampleData(in_fields, out_field, frame_rate,
                         pre_context=pre_context,
                         post_context=post_context,
                         in2_fields=in2_fields,
                         in2_pre_context=in2_pre_context,
                         in2_post_context=in2_post_context,
                         attended_field=attended_field,
                         initial_batch_size=initial_batch_size,
                         final_batch_size=final_batch_size,
                         repeat_count=repeat_count,
                         shuffle_buffer_size=shuffle_buffer_size,
                         data_dir=data_dir,
                         data_pattern=data_pattern,
                         train_file_pattern=train_file_pattern,
                         validate_file_pattern=validate_file_pattern,
                         test_file_pattern=test_file_pattern)
  if data_type == 'test':
    return TestBrainData(in_fields, out_field, frame_rate,
                         pre_context=pre_context,
                         post_context=post_context,
                         in2_fields=in2_fields,
                         in2_pre_context=in2_pre_context,
                         in2_post_context=in2_post_context,
                         initial_batch_size=initial_batch_size,
                         final_batch_size=final_batch_size,
                         repeat_count=repeat_count,
                         shuffle_buffer_size=shuffle_buffer_size,
                         data_dir=data_dir,
                         data_pattern=data_pattern,
                         train_file_pattern=train_file_pattern,
                         validate_file_pattern=validate_file_pattern,
                         test_file_pattern=test_file_pattern)
  raise TypeError('create_brain_dataset unknown data type %s' % data_type)

