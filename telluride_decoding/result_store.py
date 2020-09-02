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

"""Code to save a signal stored as a numpy array, and return it (in parts).

This file implements three classes that make it easy to store and retrieve
signals stored as numpy arrays.  This is necessary because some signals are
available in pieces, and then must be retrieved with different segment sizes.

For example, data arrives in minibatches with num_frames frames at a time, but
needs to be retrieved with segments of size window_size.

The three clases do the following (briefly for context):
NumpyStore: Basic storage of one signal
WindowedDataStore: Above, plus retrieve pieces (windows) of the data
TwoResultStore: Two of the WindowedDataStore, for two signals.
"""

from absl import logging
import numpy as np


class NumpyStore(object):
  """A generic class that stores pieces of numpy data, and grows as needed.

  This class is needed because we want to grab all the minibatches so we can
  access them as a long set of data for training and analysis.  Do it
  efficiently by doubling the size as needed.

  Data is assumed to be num_frames x num_channels, so final result concatenates
  all the data along the first axis.

  Attributes:
    all_data: Return all the valid data in the buffer as a numpy array. Returns
      None before any data is added.
      Note: this is a *link* to the original/internal data to this class. It
      does not modify/ empty the original buffer when returning the contents.
    count: How many elements are in the buffer (mostly for debugging).
  """

  def __init__(self, init_frame_count=10000, name='Generic'):
    """Creates the class, and store the initial frame count.

    Args:
      init_frame_count: How many frames of data to allocate on first use.
      name: Optional name of this storage object, useful for debugging prints.
    """
    self._count = 0
    self._data_store = None
    if init_frame_count > 0:
      self._init_frame_count = init_frame_count
    else:
      raise ValueError('Initial frame count must be greater than 0, not %s' %
                       (init_frame_count))
    self._name = name

  @property
  def count(self):
    return self._count

  @property
  def all_data(self):
    """Returns all the data accumulated so far."""
    if self._data_store is None:
      return None
    return self._data_store[:self._count, :]

  def create_storage(self, data):
    """Creates the storage needed for the signals, increasing size as needed.

    This routine allocates the initial storage (an np array) when first called
    (so it knows how wide the data is), and then doubles the size as necessary,
    copying the old data into the new array.

    Args:
      data: A prototype of the data, needed to get the width of the storage.

    Raises:
      TypeError for data that is not an np array of rank 2.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
      raise TypeError('data must be a 2D numpy array, not %s' % type(data))

    if self._data_store is None:
      frame_count = max(self._init_frame_count, 2*data.shape[0])
      self._data_store = np.zeros((frame_count, data.shape[1]))
      self._count = 0
      logging.info('ResultStorage %s: Created storage of size %dx%d.',
                   self._name,
                   self._data_store.shape[0], self._data_store.shape[1])
    elif self._data_store.shape[0] < self._count + data.shape[0]:
      self._grow_storage(data)
    if data.shape[1] != self._data_store.shape[1]:
      raise ValueError(
          'Data\'s shape has changed, and this is not allowed.(%d to %d).' %
          (self._data_store.shape[1], data.shape[1]))

  def _grow_storage(self, data):
    current_size = self._data_store.shape[0]
    new_size = max(current_size*2, current_size + 2*data.shape[0])
    new_data_store = np.zeros((new_size, data.shape[1]))
    new_data_store[:self._count, :] = self._data_store[:self._count, :]
    self._data_store = new_data_store
    logging.info('ResultStorage %s: Increased storage to size %dx%d.',
                 self._name,
                 self._data_store.shape[0], self._data_store.shape[1])

  def add_data(self, data):
    """Adds some data to the cache.

    Data should be num_frames x num_channels in size.

    Args:
      data: Signal data to store.

    Raises:
      TypeError for inconsistent data.
      ValueError when the data is too big for the buffer.
    """
    data = np.asarray(data)
    if data.ndim < 2:
      data = np.reshape(data, (-1, 1))

    self.create_storage(data)

    start = self._count
    finish = start + data.shape[0]
    self._data_store[start:finish, :] = data
    self._count += data.shape[0]

  def next_window(self, window_size):
    """Fetch fixed window size data from beginning (earliest time) of the cache.

    Args:
      window_size: Number of samples to fetch.

    Yields:
      p1: Chunk of data of window size
    """
    if self._count < window_size:
      yield None
    else:
      # Preserve what we want to return by copying it.
      p1 = np.copy(self._data_store[:window_size, :])
      # Remove the data we don't need any more from the front of the buffer.
      frames_to_keep = self._count - window_size
      self._data_store[:frames_to_keep,
                       :] = self._data_store[window_size:self._count, :]
      self._count -= window_size
      yield p1


class WindowedDataStore(NumpyStore):
  """Class for storing a signal and pulling fixed-sized windows out to process.

  Add any number of frames at a time (but don't overfill the buffer). Then it
  can be pulled out as fixed-sized windows.  Each window of data has size
    (2*half_window_width + 1) x num_features
  The windows are separated by window_step samples.

  Note: this class uses a fixed-length array, based on the max_frames
  variable defined below.  There is no mechanism for expanding the storage.

  To use:
    Create object
    for data in input_stream
      object.add_data(data)  # data can be any size
      for win in object.next_window():
        Process data, which has size (2*half_window_width+1) x num_features
  """

  def __init__(self, window_step=100, window_width=None, pre_context=0,
               initial_frame_count=100):
    """Creates the storage object.

    Note: this class only handles integer step sizes, so downsampling by a
    non-integer rate (i.e. 22050 to 100hz) doesn't work.

    Args:
      window_step: How many frames to advance each time we grab some data.
      window_width: Width of the analysis window. If not set, then
        the default is a window_width of int(3*window_step) so the total
        window is approximate 3x the window_step.
      pre_context: How much data to prepend before the first window. This has
        the effect of shifting the center point of the window. The default value
        is zero, so the returned window starts at time 0. But you can set it to
        a bigger value so the window is centered elsewhere.  Use window_size//2
        so each analysis window is *centered* on the sample time.
      initial_frame_count: How many frames worth of data to buffer (before they
        are pulled out.)
    """
    super(WindowedDataStore, self).__init__()
    if int(window_step) != window_step:
      raise ValueError('Must be an integer window_step for now, not %g.' %
                       window_step)

    if window_width is None:
      window_width = int(3 * window_step)
    logging.info('Initializing AudioDataStore with step of %d and width of %d.',
                 window_step, window_width)

    # TODO allow for fractional steps
    if window_step > window_width:
      raise ValueError('window_step (%d) must be less than or equal to '
                       'window_width (%d)' % (window_step, window_width))
    self._window_width = int(window_width)
    self._pre_context = int(pre_context)
    self._window_step = int(window_step)
    self._max_frames = int(initial_frame_count * max(window_step,
                                                     window_width))
    self._data_store = None    # Where we store the data till it is used.
    self._count = 0

  def create_storage(self, data):
    """Create the storage needed for the signals.

    Args:
      data: a prototype data, needed to get the width of the storage
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
      raise TypeError('data must be a 2D numpy array, not %s' % type(data))
    if self._data_store is None:
      self._data_store = np.zeros((max(self._max_frames, data.shape[0]),
                                   data.shape[1]))
      self._count = 0
      # Prime the store with half-window's worth of data, so when we return
      # data the window of data is centered on the current sample.
      if self._pre_context > 0:
        self.add_data(self._data_store[:self._pre_context, :])
      logging.info('Created storage of %dx%d.',
                   max(self._max_frames, self._data_store.shape[0]),
                   self._data_store.shape[1])
    elif self._data_store.shape[0] < self._count + data.shape[0]:
      self._grow_storage(data)
    if data.shape[1] != self._data_store.shape[1]:
      raise ValueError('Data\'s shape has changed, and this is not allowed.')

  def next_window(self):
    """Iterator that returns the next window of data.

    Also removes the no-longer needed data by shifting the data in the internal
    array.

    Yields:
      An array of data, centered on the current time.
    """
    while self._count >= self._window_width:
      # Preserve what we want to return by copying it.
      p1 = np.copy(self._data_store[:self._window_width, :])

      # Remove the data we don't need any more from the front of the buffer.
      frames_to_keep = self._count - self._window_step
      self._data_store[:frames_to_keep,
                       :] = self._data_store[self._window_step:self._count, :]
      self._count -= self._window_step
      yield p1


class TwoResultStore(object):
  """Class for storing a signal and pulling fixed-sized windows out to process.

  Note: this class uses the AudioDataStore class as its underlying storage,
  which by default returns windows of window_width starting from frame 0.

  Attributes:
    all_data: The valid data from both streams as two numpy arrays.
      Will return (None, None) before data is first added to the array.
      Note: this is a *link* to the original/internal data to this class. Do not
      change the values of this array, or if you do, do not expect this call to
      return the original data.
  """

  def __init__(self, window_width=100, window_step=100, pre_context=0,
               initial_frame_count=100):
    """Creates the storage object.

    Args:
      window_width: How many frames are needed for each output window.
        The actual window size returned is window_width.
      window_step: How many frames to advance each time we grab some data.
      pre_context: Amount of precontext if we want to center the frame. Set it
                   to (window_width)//2 for centering over the window
      initial_frame_count: How much space to allocate on first call.
    """
    self._store1 = WindowedDataStore(
        window_step, window_width=window_width, pre_context=pre_context,
        initial_frame_count=initial_frame_count)
    self._store2 = WindowedDataStore(
        window_step, window_width=window_width, pre_context=0,
        initial_frame_count=initial_frame_count)

  @property
  def all_data(self):
    return self._store1.all_data, self._store2.all_data

  def add_data(self, s1, s2):
    """Adds some data to the cache, two parallel signals in this case.

    Args:
      s1: signal 1 to store.
      s2: signal 2 to store.
    """
    if s1.shape[0] != s2.shape[0]:
      raise ValueError('Both data must have the same # frames, not %d vs. %d' %
                       (s1.shape[0], s2.shape[0]))
    self._store1.add_data(s1)
    self._store2.add_data(s2)

  def next_window(self):
    """Iterator that returns the next window of data.

    Also removes the no-longer needed data from the internal cache.

    Yields:
      Two sets of data, each representing window_width frames.
    """
    for p1 in self._store1.next_window():
      for p2 in self._store2.next_window():
        yield p1, p2
        break   # Break this loop because we only want one p2 per p1.
