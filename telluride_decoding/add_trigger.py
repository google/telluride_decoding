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

r"""Code to add trigger events to second channel of an audio file.

This code generates a sequence of random event times, and then adds a pulse
at these times to the second channel of an audio file, so that we can trigger
the Natus event box.

To run:
  add_trigger \
    --input_filename test_data/tapestry.wav \
    --output_filename /tmp/tapestry_events.wav --verbose True
    --number_of_events 5

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import scipy.io.wavfile
import six
from six.moves import range

from tensorflow.io.gfile import GFile

FLAGS = flags.FLAGS

flags.DEFINE_string('input_filename', None, 'Input audio filename')
flags.DEFINE_string('output_filename', None, 'Output audio filename')
flags.DEFINE_integer('number_of_events', -1,
                     'Number of events to add (-X for 1 per X seconds.)')
flags.DEFINE_boolean('verbose', False, 'Show log messages.')
flags.DEFINE_float('pulse_length', 0.1, 'Length of the pulse (seconds)')
flags.DEFINE_float('pulse_freq', 0, 'Frequency of the pulse (Hz)')


def random_times(duration, number, minimum_interval=0.5, include_zero=True):
  """Return a list of random times with at least the minimum interval between.

  Args:
    duration: Maximum time for an event (seconds)
    number: Desired number of events
    minimum_interval: Minimum time between events
    include_zero: Whether the first point should be at 0.0

  Returns:
    A sorted list of event times or None if I can't find a list that satisfies
    the constraints..
  """
  if (number - 1)*minimum_interval > duration:
    raise ValueError('Not enough time for %d events with %gs between them in '
                     '%gs.' % (number, minimum_interval, duration))
  number = int(number)  # So array sizes are integers
  for _ in range(1000):  # Try a bunch of times to get a good set of times.
    bucket_of_times = np.random.uniform(low=0, high=duration, size=8*number)
    if include_zero:
      bucket_of_times[0] = 0.0
    while len(bucket_of_times) >= number:
      sorted_times = np.sort(bucket_of_times[:number])
      intervals = sorted_times[1:number] - sorted_times[0:(number-1)]
      # Look for intervals that are too close together.
      too_short_indices = np.nonzero(intervals < minimum_interval)
      too_short_indices = too_short_indices[0]
      if too_short_indices.shape[0]:
        # For each time at the end of a too-short interval...
        for t in sorted_times[too_short_indices+1]:
          # Find the time in the unsorted list
          item_index = np.nonzero(np.abs(bucket_of_times - t) <
                                  minimum_interval/10.0)[0]
          # Don't remove the event at time zero (if called for).
          if include_zero and item_index.shape[0] > 0 and item_index[0] == 0:
            item_index = item_index[1:]
          # Delete the bad time.
          bucket_of_times = np.delete(bucket_of_times, item_index)
          if include_zero:
            assert bucket_of_times[0] == 0.0, ('at %g, item_index=%s, %s' %
                                               (t, item_index, random_times))
      else:
        return sorted_times
  return None


def add_events_to_audio(audio_signal, event_times, fs=16000,
                        pulse_length=0.1, pulse_freq=0):
  """Add pulses to an audio channel to indicate the event times.

  Given a list of event times, add a second channel to the audio_signal that
  pulses at the right time. By default the pulses are full-scale positive DC
  pulses, but a frequency can be specified to turn them into full-scale tone-
  blips.

  Args:
    audio_signal: a 1D np.ndarray with the audio data
    event_times: A list or np.ndarray of event times (largest must be less
      than the length of the audio signal.)
    fs: sampling frequency of the audio signal.
    pulse_length: length of the pulse (or tone blip) in seconds
    pulse_freq: if non-zero, the frequency of the tone blip indicating an event.

  Returns:
    A stereo audio signal, with the original audio in channel (column) 0, and
    the tone blips in the second channel. Final size is num_times x 2.
  """
  if not isinstance(audio_signal, np.ndarray):
    raise TypeError('audio signal must be an np.ndarray')
  audio_signal = audio_signal.astype(np.int16)
  audio_signal = audio_signal.squeeze()
  if len(audio_signal.shape) > 1:
    channels = tuple(range(1, len(audio_signal.shape)))
    audio_signal = np.mean(audio_signal, axis=channels)
  if len(audio_signal.shape) != 1:
    raise TypeError('audio signal (after squeezing) must be 1-dimensional.')
  if fs < 8000.0:   # Make sure it's an audio frequency
    raise ValueError('Sampling rate is generally > 8000Hz.')
  if not (isinstance(event_times, list) or
          isinstance(event_times, np.ndarray)) or len(event_times) < 3:
    raise ValueError('event_times must be a list of at least 3 elements.')
  audio_length = audio_signal.shape[0]
  new_channel = np.zeros((audio_length, 1), dtype=np.int16)
  for t in event_times*fs:
    t = int(t)
    new_channel[t:t+int(fs*pulse_length)] = 32767   # Largest int16
  if pulse_freq > 0:   # Convert the pulse into a tone.
    new_channel = np.multiply(new_channel,
                              np.sin(np.reshape(np.arange(audio_length),
                                                (-1, 1))/
                                     float(fs)*2*np.pi*pulse_freq))
  stereo_signal = np.concatenate((np.reshape(audio_signal, (-1, 1)),
                                  np.reshape(new_channel, (-1, 1))),
                                 axis=1).astype(np.int16)
  return stereo_signal


def read_audio_wave_file(audio_filename):
  if not isinstance(audio_filename, six.string_types):
    raise TypeError('audio_filename must be a string.')

  [fs, audio_signal] = scipy.io.wavfile.read(audio_filename)
  logging.info('Read_audio_file: Read %s samples from %s at %gHz.',
               audio_signal.shape, audio_filename, fs)
  assert audio_signal.dtype == np.int16
  return fs, audio_signal


def write_audio_wave_file(audio_filename, audio_signal, fs):
  if not isinstance(audio_filename, six.string_types):
    raise TypeError('audio_filename must be a string.')
  if not isinstance(audio_signal, np.ndarray):
    raise TypeError('audio_signal must be an np.ndarray')

  scipy.io.wavfile.write(audio_filename, fs, audio_signal)
  logging.info('Write_audio_file: wrote %s samples to %s at %gHz.',
               audio_signal.shape, audio_filename, fs)



def main(_):
  if FLAGS.verbose:
    logging.set_verbosity(logging.INFO)
  if FLAGS.pulse_length <= 0.0:
    raise ValueError('Pulse length (%g) must be greater than 0.' %
                     FLAGS.pulse_length)

  [audio_fs, audio_signal] = read_audio_wave_file(FLAGS.input_filename)

  audio_seconds = audio_signal.shape[0]/float(audio_fs)
  if FLAGS.number_of_events < 0:
    number = int(audio_seconds)//(-FLAGS.number_of_events)
  elif FLAGS.number_of_events == 0:
    raise ValueError('Can not add 0 events.')
  else:
    number = FLAGS.number_of_events
  event_times = random_times(audio_seconds - 2*FLAGS.pulse_length,
                             number=number,
                             minimum_interval=0.5,
                             include_zero=True)
  logging.info('Adding events at times: %s',
               ','.join(str(e) for e in event_times))
  stereo_signal = add_events_to_audio(audio_signal, event_times, audio_fs,
                                      pulse_length=FLAGS.pulse_length,
                                      pulse_freq=FLAGS.pulse_freq)
  write_audio_wave_file(FLAGS.output_filename, stereo_signal, audio_fs)


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_filename', 'output_filename'])
  app.run(main)
