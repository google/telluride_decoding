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

"""Code to read a model, and test attention switching.

This code reads a precomputed model file, and tests the AAD/SSD performance
using a test file. The code creates summary plots and calculates final accuracy.
"""

import collections
import numbers
import os

from absl import app
from absl import flags
from absl import logging

import matplotlib
# The next change breaks colab, so add "%matplotlib inline" after importing
# this file.
# pylint: disable=g-import-not-at-top
matplotlib.use('Agg')    # Needed for plotting to a file, before the next import
import matplotlib.pyplot as plt

import numpy as np

from telluride_decoding import attention_decoder
from telluride_decoding import brain_data
from telluride_decoding import brain_model
from telluride_decoding import cca
from telluride_decoding import infer_decoder

import tensorflow.compat.v2 as tf
# End user should call tf.compat.v1.enable_v2_behavior()

base_tf_dir = 'test_data/aad'
default_tf_dir = os.path.join(base_tf_dir, '001_2019-09-19/tfrecords/all')

base_summary_dir = 'test_data/tf2'
default_model_dir = os.path.join(base_summary_dir,
                                 '001_2019-09-19_all',
                                 '001_2019-09-19_all_CCA', 'model')
default_plot_dir = os.path.join(base_summary_dir,
                                '001_2019-09-19_all',
                                '001_2019-09-19_all_CCA')

default_test_file = os.path.join(base_tf_dir,
                                 '001_2019-09-19', 'tfrecords', 'all',
                                 'test', 'Trial 05.tfrecords')

flags.DEFINE_string('tf_dir', default_tf_dir,
                    'Location of the training data for evaluation.')
flags.DEFINE_string('model_dir', default_model_dir,
                    'Location of the saved BrainModel Keras model')
flags.DEFINE_string('plot_dir', default_plot_dir,
                    'Where to store result plots')
flags.DEFINE_string('save_results_csv', None,
                    'Path to results csv file')

flags.DEFINE_multi_string('training_file', [],
                          'Training files to calculate parameters for the final'
                          'decoding test.')
flags.DEFINE_string('test_file', default_test_file,
                    'Testing file to evaluate with which to verify '
                    'performance.')

flags.DEFINE_integer('window_width', 1000,
                     'Number of frames of data to use when estimating '
                     'correlation.')
flags.DEFINE_integer('window_step', 500,
                     'Number of frames to step the window when estimating the '
                     'correlation.')
flags.DEFINE_float('window_overlap', 0.5,
                   'Factor of window width for overlapping test windows.')
if 'frame_rate' not in flags.FLAGS:
  flags.DEFINE_float('frame_rate', 100,
                     'EEG and audio frame rates in Hz, after decoding.')
flags.DEFINE_enum('reduction', 'lda',
                  ['first', 'second', 'lda', 'mean', 'mean-squared', 'all'],
                  'How to reduce decoder dimensionality to a scalar.')

# Use separate variable so we can iterate over the valid options in
# run_comparison_test()
allowable_decoder_types = ['wta', 'stepped', 'ssd']

flags.DEFINE_enum('decoder', 'wta', allowable_decoder_types,
                  'How to summarize multiple correlation windows.')
flags.DEFINE_bool('window_test', False,
                  'Run a test with different window sizes')
flags.DEFINE_bool('comparison_test', False,
                  'Run a test with all decoders and infers')
flags.DEFINE_string('audio_label', 'loudness',
                    'TFRecord field containing the audio signal')

FLAGS = flags.FLAGS


def create_brain_data(tf_dir, train_files, test_files, params, audio_label):
  """Creates a braindata object from which we can get train and test datasets.

  Args:
    tf_dir: Where to find the training and testing files.
    train_files: A list of files to be used for training (within tf_dir).
    test_files: A list of files to be used for testing (within tf_dir).
    params: A dictionary with context parameters needed for preprocessing.
    audio_label: Which feature from the tfrecord file contains the audio
      data for this experiment. Typically loudness or loudness2.

  Returns:
    A BrainData object, from which we can then select train or testing files.
  """
  if isinstance(train_files, str):
    train_files = [train_files,]
  train_file_re = '|'.join(['%s' % s for s in train_files])

  if isinstance(test_files, str):
    test_files = [test_files,]
  test_file_re = '|'.join(['%s' % s for s in test_files])

  print('Training data from %s -> %s with label %s.' %
        (tf_dir, train_file_re, audio_label))
  print('Testing data from %s -> %s with label %s.' %
        (tf_dir, test_file_re, audio_label))

  if 'attended_field' in params:
    attended = params['attended_field']
  else:
    # This is the default name for the attention signal during ingestion.
    attended = 'attend'

  exp_brain_data = brain_data.TFExampleData(
      params['input_field'],    # Network input
      audio_label,              # Network output
      100,                      # Frame rate
      pre_context=params['pre_context'],
      post_context=params['post_context'],
      in2_fields=audio_label,
      in2_pre_context=params['input2_pre_context'],
      in2_post_context=params['input2_post_context'],
      attended_field=attended,
      final_batch_size=200,
      repeat_count=1,
      # Shuffle_buffer_size must be 0 so we don't shuffle in test mode.
      # Otherwise the inference stage won't see data in the right order!!!
      shuffle_buffer_size=0,
      data_dir=tf_dir,
      data_pattern='',
      train_file_pattern=train_file_re,
      validate_file_pattern='',
      test_file_pattern=test_file_re,
      )
  return exp_brain_data


def calculate_time_axis(data, window_step, window_width, frame_rate):
  """Calculates the time axis (in minutes) for a windowed signal.

  Data originates by windowing data with the given window width and step.
  The returned time corresponds to the *center* of the analysis window.

  Args:
    data: Sample data frame count. Can also be an array for sizing.
    window_step: Number of samples by which the window is advanced.
    window_width: Analysis window width.
    frame_rate: The underlying sample rate, before windowing, in samples/second.

  Returns:
    A one-dimensional numpy array, with the true time (in minutes) of the start
    of each analysis window.
  """
  if isinstance(data, numbers.Number):
    num_points = int(data)
  elif isinstance(data, list):
    num_points = len(data)
  elif isinstance(data, np.ndarray):
    num_points = data.shape[0]
  else:
    raise TypeError('Unknown type passed as input argument.')
  return (np.arange(num_points)*window_step+window_width/2.0)/frame_rate/60.0


def get_data_for_model(tf_dir, train_file, test_file, model_object,
                       audio_label_1, audio_label_2):
  """Gets all the brain_data objects needed for trainin and testing.

  Args:
    tf_dir: where to find all the .tfrecords files.
    train_file: a regular expression matching the training files in tf_dir.
    test_file: a regular expression matching the testing files in tf_dir.
    model_object: a brain_model object which we use to make sure the sizes
      of the train and test data are compatible.
    audio_label_1: Field name for the audio from speaker 1
    audio_label_2: Field name for the audio from speaker 2

  Returns:
    Four brain_data objects:
      training file for speaker 1 (loudness feature)
      test file for speaker 1 (loudness feature)
      training file for speaker 2 (loudness2 feature)
      test file for speaker 2 (loudness2 feature)
  """
  brain_data_1 = create_brain_data(tf_dir, train_file, test_file,
                                   model_object.decoding_model_params,
                                   audio_label_1)  # Speaker 1
  brain_data_2 = create_brain_data(tf_dir, train_file, test_file,
                                   model_object.decoding_model_params,
                                   audio_label_2)  # Speaker 2

  bd1_train = brain_data_1.create_dataset(mode='train')
  bd1_test = brain_data_1.create_dataset(mode='test')
  bd2_train = brain_data_2.create_dataset(mode='train')
  bd2_test = brain_data_2.create_dataset(mode='test')

  model_object.check_model_and_data(bd1_train)
  model_object.check_model_and_data(bd1_test)
  model_object.check_model_and_data(bd2_train)
  model_object.check_model_and_data(bd2_test)
  print('Finished the model checking.')
  return bd1_train, bd1_test, bd2_train, bd2_test


def regress_and_correlate(model_object, test_data, window_size):
  full_results = []
  labels = []
  for results, label in model_object.test_by_window(test_data, window_size):
    full_results.append(np.mean(results))
    labels.append(np.mean(label))
  return full_results, labels


def load_model(model_dir, reducer):
  """Loads a model from disk.

  Args:
    model_dir: Where to find the TF model on disk.
    reducer: What type of correlation reducer to use, a string.

  Returns:
    An infer_decoder.AttentionStateDecoder object.

  Raises:
    IOError: Raises error if we can't load the model parameters for decoding.
  """
  model_object = infer_decoder.create_decoder(model_dir.lower(),
                                              reduction=reducer)

  brain_object_dict = {'pearson_correlation': brain_model.pearson_correlation,
                       'BrainCcaLayer': cca.BrainCcaLayer,
                       'cca_pearson_correlation_first':
                           cca.cca_pearson_correlation_first,
                      }
  model_object.load_decoding_model(model_dir, brain_object_dict)
  decoder_param_filename = os.path.join(model_dir, 'decoder_model.json')
  if tf.io.gfile.exists(decoder_param_filename):
    model_object.restore_parameters(decoder_param_filename)
  else:
    raise IOError('Can not load decoder model parameters from %s' %
                  decoder_param_filename)
  return model_object


def find_first_segment(labels):
  """Finds the first section of data based on the ground truth.

  This is used to train the SSD's probability models, by looking for the first
  segment, all with the subject attending to the same speaker.

  Args:
    labels: An np array of segment labels, generally 0 for speaker 1 and 1 for
      speaker 2.

  Returns:
    The frame number at the end of the first segment with all the same label.
  """
  if isinstance(labels, list):
    labels = np.asarray(labels)
  if not isinstance(labels, np.ndarray):
    raise TypeError('Labels input must be an ndarray, not %s' % type(labels))
  if labels.ndim != 1:
    raise TypeError('Labels input must be one-dimensional, not %s' %
                    str(labels.shape))
  end_section = np.nonzero(np.logical_xor(labels, labels[0]))
  if end_section[0].shape[0]:
    return end_section[0][0]
  return 0


def run_reduction_test(model_dir, tf_dir, train_file, test_file, reduction,
                       decoder_type, audio_label_1, audio_label_2,
                       plot_dir=None):
  """Runs a complete test for a given reduction and decoder type.

  Args:
    model_dir: Where to find the model used to initialize the data. If there is
      no model, then the train_file pattern is used to find training data.
    tf_dir: A directory containing .tfrecord files for training and/or testing.
    train_file: Which files to use for training, a regular expression in tf_dir.
    test_file: Which files to use for testing, a regular expression in tf_dir.
    reduction: One of the valid CCA reduction techniques
    decoder_type: One of the valid AttentionDecoder type names.
    audio_label_1: Field name for the audio from speaker 1
    audio_label_2: Field name for the audio from speaker 2
    plot_dir: Where to put a summary plot.

  Returns:
    A dictionary keyed by window length and giving fraction of correctly
    infered frames.
  """
  print('Running regression test with %s.' % reduction)
  model_object = load_model(model_dir, reduction)

  bd1_train, bd1_test, bd2_train, bd2_test = get_data_for_model(
      tf_dir, train_file, test_file, model_object,
      audio_label_1, audio_label_2)

  if model_object.decoding_model_params:
    print('Found saved mode, no need to train the decoding model.')
  else:
    model_object.train(bd1_train, bd2_train)
    print('Finished the model training.')

  window_results = []
  window_list = [10, 100, 200, 400, 700, 1000]
  print('Infer Classification window_size are:', window_list)
  for window_size in window_list:
    window_step = window_size // 2

    d1_results, _ = regress_and_correlate(model_object, bd1_test,
                                          window_size)
    # The attention labels are embedded in the training data.
    d2_results, labels = regress_and_correlate(model_object, bd2_test,
                                               window_size)

    decoder = attention_decoder.create_attention_decoder(
        decoder_type, window_step=window_step, frame_rate=FLAGS.frame_rate)

    end_first_section = find_first_segment(labels)
    if end_first_section:
      # Grab just the decision score and drop the upper/lower confidence bounds.
      decoder.tune(d1_results[:end_first_section],
                   d2_results[:end_first_section])
    else:
      logging.info('Could not find both true and false values in the '
                   'attention signal: %s. Not tuning decoder', labels)

    # Calculate the decoded attention results.
    attention = np.array([decoder.attention(c1, c2) for c1, c2
                          in zip(d1_results, d2_results)])
    labels = np.asarray(labels)

    correct = np.logical_xor(attention[:, 0] >= 0.5, labels)
    frac_correct = np.sum(correct)/float(len(correct))
    window_results.append(frac_correct)

    if plot_dir:
      title = 'AAD Correlation on %s with %gs windows %g%% accuracy.' % (
          os.path.split(test_file)[1], window_size/100.0, frac_correct*100.0)
      t = calculate_time_axis(np.asarray(d1_results), window_step,
                              window_size, FLAGS.frame_rate)
      plt.clf()
      attention_decoder.plot_aad_results(np.asarray(d1_results), t=t,
                                         linecolor='blue')
      attention_decoder.plot_aad_results(np.asarray(d2_results), t=t,
                                         linecolor='red')
      attention_decoder.plot_aad_results(attention[:, 0], t=t,
                                         attention_flag=np.asarray(labels),
                                         decision_upper=attention[:, 1],
                                         decision_lower=attention[:, 2],
                                         linecolor='green',
                                         title=title)
      plt.legend(('Speaker 1', 'Speaker 2', 'Decision'))

      plot_file = os.path.join(plot_dir, 'test_results_%s_%s_%05d.png' %
                               (reduction, decoder_type, window_size))
      with tf.io.gfile.GFile(plot_file, 'wb') as fp:
        assert fp, 'Can not open %s for plotting AAD' % plot_file
        plt.savefig(fp)
      print('Saved final test attention switch result plot to', plot_file)

  print('Infer classification result for %s with %s and %s: %s' %
        (os.path.join(tf_dir, test_file), reduction,
         decoder_type, window_results))
  if FLAGS.save_results_csv is not None:
    print('Saving results to {}'.format(FLAGS.save_results_csv))
    with open(FLAGS.save_results_csv, 'w') as f:
      f.write('Window size,Accuracy\n')
      for (wl, wr) in zip(window_list, window_results):
        f.write('{},{}\n'.format(wl, wr))
  if plot_dir:
    plt.clf()
    plt.semilogx(window_list, window_results)
    plt.xlabel('Window Size (frames)')
    plt.ylabel('Fraction correct')
    plt.title('Testing with %s, reducing with %s, decoding with %s' %
              (os.path.split(test_file)[1], reduction, decoder_type))
    plot_file = os.path.join(plot_dir, 'test_results_%s_%s.png' %
                             (reduction, FLAGS.decoder))
    with tf.io.gfile.GFile(plot_file, 'wb') as fp:
      assert fp, 'Can not open %s for plotting AAD' % plot_file
      plt.savefig(fp)
    print('Saved final test classification result plot to', plot_file)
  return dict(zip(window_list, window_results))


def run_comparison_test(model_dir, tf_dir, training_file, test_file,
                        audio_label, audio_label_2, plot_dir,
                        reduction_list, decoder_list=None):
  """Runs a test comparing all the different reducers and decoders.

  Args:
    model_dir: Directory containing the pretrained model.
    tf_dir: Directory containing the TFRecord data for training and testing.
    training_file: Regular expression that picks the training data.
    test_file: Regular expression that picks the testing data.
    audio_label: The feature name for the attended audio loudness data.
    audio_label_2: The feature name for the unattended audio loudness data.
    plot_dir: Where to store the output plots.
    reduction_list: A list of valid reduction types to test.
    decoder_list: A list of valid decoder types to test.

  Returns:
    A dictionary of test results, keyed by the reduction and decoder types.
  """
  all_results = collections.OrderedDict()
  for reduction in reduction_list:
    for decoder in decoder_list or allowable_decoder_types:
      print('Running the regression test with %s and %s.' %
            (reduction, decoder))
      results = run_reduction_test(model_dir, tf_dir,
                                   training_file, test_file,
                                   reduction, decoder,
                                   audio_label, audio_label_2,
                                   plot_dir)
      label = (reduction, decoder)
      all_results[label] = results

  plt.clf()
  for reduction_decoder, results in all_results.items():
    if reduction_decoder[0] != 'lda':
      style = '--'
    else:
      style = '-'
    sizes = sorted(results.keys())
    acc = [results[s] for s in sizes]
    plt.semilogx(sizes, acc, style, label='%s %s' % reduction_decoder)

  if plot_dir:
    plt.xlabel('Window Size (frames)')
    plt.ylabel('Fraction correct')
    plt.title('Testing with %s' % os.path.split(FLAGS.test_file)[1])
    plt.legend()

    plot_file = os.path.join(FLAGS.plot_dir, 'test_results-comparison.png')
    with tf.io.gfile.GFile(plot_file, 'wb') as fp:
      assert fp, 'Can not open %s for plotting AAD' % plot_file
      plt.savefig(fp)
    print('Saved find test result plot to', plot_file)
  return all_results


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments: %s.' % argv)

  if FLAGS.tf_dir and not tf.io.gfile.exists(FLAGS.tf_dir):
    raise app.UsageError('Can not find tf_dir: %s' % FLAGS.tf_dir)
  if not tf.io.gfile.exists(FLAGS.model_dir):
    raise app.UsageError('Can not find model_dir: %s' % FLAGS.model_dir)

  print('FLAGS.reduction is', FLAGS.reduction)
  if FLAGS.comparison_test:
    run_comparison_test(FLAGS.model_dir, FLAGS.tf_dir,
                        FLAGS.training_file, FLAGS.test_file,
                        FLAGS.audio_label, FLAGS.audio_label + '2',
                        FLAGS.plot_dir,
                        reduction_list=['first', 'lda'])
  else:
    run_reduction_test(FLAGS.model_dir, FLAGS.tf_dir, FLAGS.training_file,
                       FLAGS.test_file, FLAGS.reduction, FLAGS.decoder,
                       FLAGS.audio_label, FLAGS.audio_label + '2',
                       FLAGS.plot_dir)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
