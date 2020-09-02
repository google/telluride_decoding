# Copyright 2019-2020 Google Inc.
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

"""TF code to train and test an MEG/EEG decoder.

TF models and code to predict MEG/EEG signals from their input audio features,
or vice versa.

This programn does the following:
1) Creates the desired kind of brain model (linear, cca, dnn, etc.)
2) Trains and tests the model, returning training and testing performance.
3) If desired, writes a summary file containing the parameters and train/test
   results
4) Saves the TF model for later inference.

"""

import os
import typing

from absl import app
from absl import flags
from absl import logging

import attr
import numpy as np
from telluride_decoding import brain_data
from telluride_decoding import brain_model
from telluride_decoding import cca
from telluride_decoding import infer_decoder

import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()



# Use attr to not preclude PY2 usage for now.
@attr.s
class DecodingOptions(object):
  """A class encapsulating all the parameters for a decoding experiment."""
  attended_field = attr.ib(init=False, type=typing.Text, default='attend')
  batch_norm = attr.ib(init=False, type=bool, default=False)
  batch_size = attr.ib(init=False, type=int, default=512)
  cca_dimensions = attr.ib(init=False, type=int, default=10)
  check_file_pattern = attr.ib(init=False, type=typing.Text, default='')
  correlation_frames = attr.ib(init=False, type=int, default=100)
  correlation_reducer = attr.ib(init=False, type=typing.Text, default='lda')
  data = attr.ib(init=False, type=typing.Text, default='tfrecords')
  debug = attr.ib(init=False, type=bool, default=False)
  dnn_regressor = attr.ib(init=False, type=typing.Text,
                          default='fullyconnected')
  dropout = attr.ib(init=False, type=float, default=0.0)
  epoch_count = attr.ib(init=False, type=int, default=100)
  frame_rate = attr.ib(init=False, type=float, default=100.0)
  hidden_units = attr.ib(init=False, type=typing.Text, default='20-20')
  input2_field = attr.ib(init=False, type=typing.Text, default='')
  input2_post_context = attr.ib(init=False, type=int, default=0)
  input2_pre_context = attr.ib(init=False, type=int, default=0)
  input_field = attr.ib(init=False, type=typing.Text, default='mel_spectrogram')
  learning_rate = attr.ib(init=False, type=float, default=0.05)
  loss = attr.ib(init=False, type=typing.Text, default='mse')
  min_context = attr.ib(init=False, type=int, default=0)
  output_field = attr.ib(init=False, type=typing.Text, default='envelope')
  post_context = attr.ib(init=False, type=int, default=0)
  pre_context = attr.ib(init=False, type=int, default=0)
  random_mixup_batch = attr.ib(init=False, type=bool, default=False)
  regularization_lambda = attr.ib(init=False, type=float, default=0.1)
  saved_model_dir = attr.ib(init=False, type=typing.Text, default=None)
  shuffle_buffer_size = attr.ib(init=False, type=int, default=100000)
  summary_dir = attr.ib(init=False, type=typing.Text, default='/tmp/tf')
  tensorboard_dir = attr.ib(init=False, type=typing.Text, default=None)
  test_file_pattern = attr.ib(init=False, type=typing.Text, default='')
  test_metric = attr.ib(init=False, type=typing.Text,
                        default='pearson_correlation_first')
  tfexample_dir = attr.ib(init=False, type=typing.Text, default=None)
  tfexample_pattern = attr.ib(init=False, type=typing.Text, default='')
  train_file_pattern = attr.ib(init=False, type=typing.Text, default='')
  validate_file_pattern = attr.ib(init=False, type=typing.Text, default='')

  def set_flags(self, all_flags=flags.FLAGS):
    """Sets all the parameters of a model based on global FLAGS variable."""
    self.attended_field = all_flags.attended_field
    self.batch_norm = all_flags.batch_norm
    self.batch_size = all_flags.batch_size
    self.cca_dimensions = all_flags.cca_dimensions
    self.check_file_pattern = all_flags.check_file_pattern
    self.correlation_frames = all_flags.correlation_frames
    self.correlation_reducer = all_flags.correlation_reducer
    self.data = all_flags.data
    self.debug = all_flags.debug
    self.dnn_regressor = all_flags.dnn_regressor
    self.dropout = all_flags.dropout
    self.epoch_count = all_flags.epoch_count
    self.frame_rate = all_flags.frame_rate
    self.hidden_units = all_flags.hidden_units
    self.input2_field = all_flags.input2_field
    self.input2_post_context = all_flags.input2_post_context
    self.input2_pre_context = all_flags.input2_pre_context
    self.input_field = all_flags.input_field
    self.learning_rate = all_flags.learning_rate
    self.loss = all_flags.loss
    self.min_context = all_flags.min_context
    self.output_field = all_flags.output_field
    self.post_context = all_flags.post_context
    self.pre_context = all_flags.pre_context
    self.random_mixup_batch = all_flags.random_mixup_batch
    self.regularization_lambda = all_flags.regularization_lambda
    self.saved_model_dir = all_flags.saved_model_dir
    self.shuffle_buffer_size = all_flags.shuffle_buffer_size
    self.summary_dir = all_flags.summary_dir
    self.tensorboard_dir = all_flags.tensorboard_dir
    self.test_file_pattern = all_flags.test_file_pattern
    self.test_metric = all_flags.test_metric
    self.tfexample_dir = all_flags.tfexample_dir
    self.tfexample_pattern = all_flags.tfexample_pattern
    self.train_file_pattern = all_flags.train_file_pattern
    self.validate_file_pattern = all_flags.validate_file_pattern
    return self

  def experiment_parameters(self, delimiter=','):
    """Turns the list of parameters into a readable string for summaries.

    Args:
      delimiter: character to separate the parameters from each other. Use None
      to get the raw list of argument strings.

    Returns:
      A string of parameter names and values, or a list of strings when None is
      the delimiter.
    """
    new_params_as_dict = attr.asdict(self)
    keys_and_values = ['%s=%s' % (k, new_params_as_dict[k])
                       for k in sorted(new_params_as_dict)]
    if delimiter:
      return delimiter.join(keys_and_values)
    return keys_and_values

  def set_from_dict(self, new_values):
    for k, v in new_values.items():
      setattr(self, k, v)
    return self


defaults = DecodingOptions()  # Just so we can get the defaults for each flag.
FLAGS = flags.FLAGS

# Basic experiment parameters
flags.DEFINE_string('attended_field', '',
                    'Which data field indicates the attended feature (used for'
                    'decoding tests.')
flags.DEFINE_bool('batch_norm', defaults.batch_norm,
                  'Switch to enable batch normalization in the network.')
flags.DEFINE_integer('batch_size', defaults.batch_size,
                     'Number of frames (with context) per minibatch')
flags.DEFINE_integer('cca_dimensions', defaults.cca_dimensions,
                     'Number of dimensions in the CCA analysis')
flags.DEFINE_string('check_file_pattern', defaults.check_file_pattern,
                    'A regular expression that enables a file integrity check.')
flags.DEFINE_integer('correlation_frames', defaults.correlation_frames,
                     'How many frames to combine when estimating correlation')
flags.DEFINE_enum('correlation_reducer', defaults.correlation_reducer,
                  ['lda', 'first', 'second', 'mean', 'mean-squared'],
                  'How to reduce the correlation vector to a scalar.')
flags.DEFINE_enum('data', defaults.data,
                  ['tfrecords'],
                  'Dataset to use for this experiment.')
flags.DEFINE_bool('debug', defaults.debug,
                  'Turn on informational debug print stmts.')
flags.DEFINE_enum('dnn_regressor', defaults.dnn_regressor,
                  ['fullyconnected', 'tf', 'linear', 'linear_with_bias', 'cca',
                   'classifier'],
                  'DNN regressor code to use for this experiment.')
flags.DEFINE_float('dropout', defaults.dropout,
                   'The dropout rate, between 0 and 1. E.g. "rate=0.1" '
                   'would drop 10% of input units.')
flags.DEFINE_integer('epoch_count', defaults.epoch_count,
                     'Number of epochs to use when estimating stats.')
flags.DEFINE_float('frame_rate', defaults.frame_rate,
                   'Number of frames per second in TFRecord data')
flags.DEFINE_string('hidden_units', defaults.hidden_units,
                    'Number of hidden layers in regressor')
flags.DEFINE_string('input_field', defaults.input_field,
                    'Input field to use for predictions.')
flags.DEFINE_string('input2_field', defaults.input2_field,
                    'Second input field for methods that need two inputs.')
flags.DEFINE_integer('input2_pre_context', defaults.input2_pre_context,
                     'Number of frames of pre context for output features')
flags.DEFINE_integer('input2_post_context', defaults.input2_post_context,
                     'Number of frames of post context for output features')
flags.DEFINE_float('learning_rate', defaults.learning_rate,
                   'The initial learning rate for the ADAM optimizer.')
flags.DEFINE_enum('loss', defaults.loss, ['mse', 'pearson'],
                  'The type of loss to use in the training step.')
flags.DEFINE_integer('min_context', defaults.min_context,
                     'Minimum number of frames of context for '
                     'prediction')
flags.DEFINE_string('output_field', defaults.output_field,
                    'Output field to predict.')
flags.DEFINE_integer('pre_context', defaults.pre_context,
                     'Number of frames of context before prediction')
flags.DEFINE_integer('post_context', defaults.post_context,
                     'Number of frames of context before prediction')
flags.DEFINE_float('regularization_lambda', defaults.regularization_lambda,
                   'Regularization parameter for the parameter estimates'
                   ' needed for linear regression.')
flags.DEFINE_bool('random_mixup_batch', defaults.random_mixup_batch,
                  'Mixup the data, so labels are random, for testing.')
flags.DEFINE_string('saved_model_dir', defaults.saved_model_dir,
                    'Directory in which to save the model.')
flags.DEFINE_integer('shuffle_buffer_size', defaults.shuffle_buffer_size,
                     'Number of elements to shuffle')
flags.DEFINE_string('summary_dir', defaults.summary_dir,
                    'Location of summary files.')
flags.DEFINE_string('tensorboard_dir', defaults.tensorboard_dir,
                    'Location of tensorboard files.')  # No logging by default
flags.DEFINE_string('test_file_pattern', defaults.test_file_pattern,
                    'A regular expression for picking testing files.')
flags.DEFINE_string('test_metric', defaults.test_metric,
                    'What metric from the training job should be summarized.'
                    '  This is usually set by the specific Regression model')
flags.DEFINE_string('tfexample_dir', defaults.tfexample_dir,
                    'location of generic TFRecord data')
flags.DEFINE_string('tfexample_pattern', defaults.tfexample_pattern,
                    'training/testing files must contain this string.')
flags.DEFINE_string('train_file_pattern', defaults.train_file_pattern,
                    'A regular expression for picking training files.')
flags.DEFINE_string('validate_file_pattern', defaults.validate_file_pattern,
                    'A regular expression for picking validation files.')


# Flags that are not stored in the DecodingOptions object
flags.DEFINE_enum('context_method', 'new', ('new', 'old'),
                  'Switch to control temporal window approach.')
flags.DEFINE_integer('num_input_channels', 1,
                     'Number of input channels in test simulations.')
flags.DEFINE_integer('prefetch_buffer_size', 100,
                     'Number of elements to prefretch')
flags.DEFINE_integer('run', 0,
                     'Just for parallel testing... which run # is this.')
######################### Main Program ##############################


def create_brain_model(model_flags, input_dataset):
  """Creates the right kind of brain model.

  Args:
    model_flags: A DecodingOptions structure giving the desired model pararms.
    input_dataset: Some models infer the data size from this dataset.

  Returns:
    The desired BrainModel object.

  TODO Make models use input_dataset, so output_dim goes away.
  """
  if not isinstance(model_flags, DecodingOptions):
    raise TypeError('Model_flags must be a DecodingOptions, not a %s' %
                    type(model_flags))
  if not isinstance(input_dataset, tf.data.Dataset):
    raise TypeError('input_dataset must be a tf.data.Dataset, not %s' %
                    type(input_dataset))
  if model_flags.dnn_regressor == 'fullyconnected':
    logging.info('Create_brain_model creating a fullyconnected model.')

    # Convert the string for hidden units into a list of numbers.
    if not model_flags.hidden_units:
      logging.info('Setting the number of hidden units to []')
      hidden_units = []
    else:
      hidden_units = [int(x) for x in model_flags.hidden_units.split('-')]

    bm = brain_model.BrainModelDNN(input_dataset, hidden_units,
                                   tensorboard_dir=model_flags.tensorboard_dir)
  elif model_flags.dnn_regressor == 'classifier':
    logging.info('Create_brain_model creating a fullyconnected model.')
    bm = brain_model.BrainModelClassifier(
        input_dataset, model_flags.hidden_units,
        tensorboard_dir=model_flags.tensorboard_dir)
  elif model_flags.dnn_regressor == 'linear':
    logging.info('Create_brain_model creating a linear model.')
    bm = brain_model.BrainModelLinearRegression(
        input_dataset, model_flags.regularization_lambda,
        tensorboard_dir=model_flags.tensorboard_dir)
  elif model_flags.dnn_regressor == 'cca':
    logging.info('Create_brain_model creating a CCA model.')
    bm = cca.BrainModelCCA(
        input_dataset, cca_dims=model_flags.cca_dimensions,
        regularization_lambda=model_flags.regularization_lambda,
        tensorboard_dir=model_flags.tensorboard_dir)
  else:
    raise TypeError('Unknown model type %s in create_brain_model.' % type)

  bm.compile(learning_rate=model_flags.learning_rate)
  return bm


def train_and_test(my_flags, test_brain_data, test_brain_model, epochs=1):
  """Trains and tests using the given dataset and model params.

  Args:
    my_flags: Decoding flags specifying model parameters.
    test_brain_data: The source of the BrainData for this experiment.
    test_brain_model: The BrainModel object to train and test.
    epochs: Number of epochs of the data to use when training.

  Returns:
    A two-ple consisting of:
    1) the train_results, which is an an empty dictionary for linear methods.
    2) the test_results
  """
  if not isinstance(test_brain_data, brain_data.BrainData):
    raise TypeError('test_brain_data must be a BrainData object, not a %s' %
                    test_brain_data)
  if not isinstance(test_brain_model, brain_model.BrainModel):
    raise TypeError('Model in train_and_test must be a BrainModel object, '
                    'not %s' % test_brain_model)
  if not isinstance(my_flags, DecodingOptions):
    raise TypeError('Train_and_test needs a DecodingOptions object, not %s.' %
                    type(my_flags))

  logging.info('train_and_test: %s', my_flags.experiment_parameters())
  train_dataset = test_brain_data.create_dataset('train')
  logging.info('Fitting model %s with %s data, and %d epochs',
               test_brain_model, train_dataset, epochs)
  train_results = test_brain_model.fit(train_dataset, epochs=epochs)

  test_dataset = test_brain_data.create_dataset('test')
  test_results = test_brain_model.evaluate(test_dataset)
  return train_results, test_results


def write_experiment_summary(my_flags, train_results, test_results,
                             dprime=None):
  """Writes a summary of the experiment to the output directory.

  Note: The token PARAMS is replaced with the current experiment summary. This
  is so that a parallel job can use the same summary flag, and yet results will
  all be placed in separate directories.

  Args:
    my_flags: Decoding parameters for model experiment.
    train_results: A dictionary of training results
    test_results: A dictionary of test results
    dprime: Classification performance as measured by d'
  """
  if not isinstance(my_flags, DecodingOptions):
    raise TypeError('Write_experiment_summary needs a DecodingOptions object,' +
                    ' not %s.' % type(my_flags))

  summary_dir = my_flags.summary_dir
  if summary_dir:
    if 'PARAMS' in summary_dir:
      summary_dir = summary_dir.replace('PARAMS',
                                        my_flags.experiment_parameters(','))
    results_file = os.path.join(summary_dir, 'results.txt')
    tf.io.gfile.makedirs(summary_dir)
    with tf.io.gfile.GFile(results_file, 'w') as fp:
      params = my_flags.experiment_parameters(';')
      fp.write('Parameters: %s\n' % params)
      for k in test_results:
        if isinstance(test_results[k], np.ndarray):
          fp.write('Final_Test/%s: %s\n' %
                   (k, ' '.join([str(f) for f in np.reshape(test_results[k],
                                                            (-1))])))
        else:
          fp.write('Final_Testing/%s: %g\n' % (k, test_results[k]))
      if dprime is not None:
        fp.write('Final_Testing/dprime: %g\n' % dprime)

      if train_results:
        pass
      # TODO What results do we get back from actual training
      # with Keras? # Original code is commented out below.
      #   for k in train_results.history.keys():
      #     if isinstance(train_results.history[k], np.ndarray):
      #       fp.write('Final_Training/%s: %s\n' %
      #                (k, ' '.join([str(f) for f in
      #                              np.reshape(train_results.history[k],
      #                                         -1)])))
      #     if isinstance(train_results.history[k], list):
      #       fp.write('Final_Training/%s: %s\n' %
      #                (k, ' '.join([str(f) for
      #                              f in train_results.history[k]])))
      #     else:
      #       fp.write('Final_Training/%s: %g\n' % (k,
      #                                             train_results.history[k]))
    logging.info('Wrote summary results to %s', results_file)


def check_files(exp_data_dir, tfexample_pattern='.tfrecords'):
  """Checks all the input files to make sure they contain valid TFExample data.

  Logs the number of records in each file.

  Args:
    exp_data_dir: Where to find the input files
    tfexample_pattern: A string used to limit the files to check.  All files
      checked must contain this string.
  """
  all_files = []
  for (path, _, files) in tf.io.gfile.walk(exp_data_dir):
    all_files += [
        os.path.join(path, f)
        for f in files
        if (f.endswith('.tfrecords') and tfexample_pattern in f)
    ]
  logging.info('Found %d files for TFExample data analysis.', len(all_files))
  print('Found %d files for TFExample data analysis.' % len(all_files))
  for f in all_files:
    logging.info('%s: %d', f, brain_data.count_tfrecords(f)[0])


def train_lda_model(brain_dataset, trained_model, my_flags):
  """Train the LDA dimension reducer on the output of a regressor model.

  This routine takes a dataset, runs it through a pretrained model that computes
  a pair of correlated signals, and then uses LDA to compute the optimum
  projection.

  Args:
    brain_dataset: A brain_data object, from which training and testing data
      can be extracted
    trained_model: A TF (keras) model which implements the regression.
    my_flags: The DecodingOptions object that describes this experiment.

  Returns:
    dprime: the normalized separation between unattended and attended speech
      signals.
    decoder: a Decoder object, which encapsulates all that is needed for
      inference: regression via TF, correlation calculation, and LDA for
      dimensionality reduction.
  """
  if not isinstance(brain_dataset, brain_data.BrainData):
    raise TypeError('Train_lda_model needs BrainData, not %s.' %
                    type(brain_dataset))
  if not callable(trained_model):
    raise TypeError('Trained_model parameter is not a callable function, '
                    'but a %s.' % type(trained_model))
  if isinstance(my_flags, dict):
    # Small hack, until we can add DecodingOptions to the infer tools.
    my_flags = DecodingOptions().set_from_dict(my_flags)
  elif not isinstance(my_flags, DecodingOptions):
    raise TypeError('Train_lda_model needs a DecodingOptions object, not %s.' %
                    type(my_flags))

  attended_data = brain_dataset.create_dataset('test', mixup_batch=False)

  unattended_data = brain_dataset.create_dataset('test', mixup_batch=True)

  decoder = infer_decoder.create_decoder(my_flags.dnn_regressor,
                                         reduction=my_flags.correlation_reducer,
                                         model=trained_model)
  dprime = decoder.train(unattended_data, attended_data,
                         window_size=my_flags.correlation_frames)

  return dprime, decoder


def run_decoding_experiment(my_flags):
  """Runs one decoding experiment: assemble data, train, evaluate.

  Args:
    my_flags: A decoding flags object telling how to do the decoding.
  """
  if my_flags.debug:
    logging.set_verbosity(logging.DEBUG)

  if my_flags.pre_context + 1 + my_flags.post_context < my_flags.min_context:
    my_flags.post_context = my_flags.min_context - (my_flags.pre_context + 1)

  if not my_flags.summary_dir.endswith('/'):
    my_flags.summary_dir = my_flags.summary_dir + '/'
  params = my_flags.experiment_parameters()
  logging.info('Params string is: %s', params)
  logging.info('TFRecord data from: %s with %s', my_flags.tfexample_dir,
               my_flags.tfexample_pattern)

  if my_flags.check_file_pattern:
    check_files(my_flags.tfexample_dir, my_flags.tfexample_pattern)
    return

  # Create the dataset we'll use for this experiment.
  test_brain_data = brain_data.create_brain_dataset(
      my_flags.data, my_flags.input_field, my_flags.output_field,
      frame_rate=my_flags.frame_rate,
      pre_context=my_flags.pre_context, post_context=my_flags.post_context,
      in2_fields=my_flags.input2_field,
      in2_pre_context=my_flags.input2_pre_context,
      in2_post_context=my_flags.input2_post_context,
      final_batch_size=my_flags.batch_size,
      shuffle_buffer_size=my_flags.shuffle_buffer_size,
      data_dir=my_flags.tfexample_dir,
      data_pattern=my_flags.tfexample_pattern,
      train_file_pattern=my_flags.train_file_pattern,
      validate_file_pattern=my_flags.validate_file_pattern,
      test_file_pattern=my_flags.test_file_pattern)

  some_dataset = test_brain_data.create_dataset('train')

  tensorboard_dir = my_flags.tensorboard_dir
  if tensorboard_dir and 'PARAMS' in tensorboard_dir:
    tensorboard_dir = tensorboard_dir.replace(
        'PARAMS', my_flags.experiment_parameters(','))
  test_model = create_brain_model(my_flags, some_dataset)
  test_model.add_tensorboard_summary('Parameters',
                                     my_flags.experiment_parameters(' '))

  train_results, test_results = train_and_test(my_flags,
                                               test_brain_data, test_model,
                                               epochs=my_flags.epoch_count)
  print()   # Needed to finish off line after TF training status updates.
  test_model.summary()
  test_model.add_metadata(attr.asdict(my_flags), dataset=some_dataset)

  dprime, final_decoder = train_lda_model(test_brain_data, test_model, my_flags)

  logging.info('train_and_test got these results: %s and test %s',
               train_results, test_results)
  print('train_and_test got these results: %s and test %s' %
        (train_results, test_results))
  logging.info('Calculated dprime is %g.', dprime)
  print('Calculated dprime is %g.' % dprime)

  if my_flags.summary_dir:
    logging.info('Writing train/test results to %s', my_flags.summary_dir)
    write_experiment_summary(my_flags, train_results, test_results,
                             dprime)
    print('Wrote train/test results to %s.' % my_flags.summary_dir)

  if my_flags.tensorboard_dir:
    # Note: Keras' evaluate method doesn't write out results. So
    # instead, we have to write out our own version of the final results.
    logdir = os.path.join(test_model.tensorboard_dir, 'dprime')
    writer = tf.summary.create_file_writer(logdir=logdir)
    with writer.as_default():
      tf.summary.scalar('dprime', dprime, step=my_flags.epoch_count)

  if my_flags.saved_model_dir:
    logging.info('Writing saved model to %s', my_flags.saved_model_dir)
    test_model.save(my_flags.saved_model_dir)
    final_decoder.save_parameters(
        os.path.join(my_flags.saved_model_dir, 'decoder_model.json'))
    print('Wrote saved model to %s.' % my_flags.saved_model_dir)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments: %s.' % argv)

  my_flags = DecodingOptions().set_flags(FLAGS)

  run_decoding_experiment(my_flags)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
