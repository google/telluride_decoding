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

"""Ingest the native BrainVision data format.

This consists of three file types:
  .vhdr: the header, a text file
  .vmrk: the event marker file, not parsed by this code.
  .eeg: the binary file with the actual eeg data.

See the detailed file specification at:
  https://www.brainproducts.com/filedownload.php?path=products/more/BrainVisionCoreDataFormat_1-0.pdf
"""

import collections
import os
import re

import numpy as np
from telluride_decoding import ingest
import tensorflow.compat.v2 as tf
# User should call tf.compat.v1.enable_v2_behavior()


def parse_bv_keywords(section):
  """Grab the keywords in a BrainVision section, and create a dictionary.

  Args:
    section: a section starts with [xxx], and contains lines of key=value

  Returns:
    A dictionary of keys and values from this section.
  """
  section = section.split(']', 1)[1]
  # Use OrderedDict so the channel list stays in order
  section_dict = collections.OrderedDict(())
  for key_value in section.split('\n'):
    if not key_value or key_value[0] == ';':
      continue
    if '=' in key_value:
      key, value = key_value.split('=', 1)
      key = key.strip()
      value = value.strip()
      try:
        value = int(value) if value.isdigit() else float(value)
      except ValueError:
        # Leave value as a str.
        pass
      section_dict[key] = value
  return section_dict


def parse_bv_header(hdr):
  """Parse the header file for BrainVision data.

  The header consists of sections, each of which starts with [___ Infos] and
  then contains a list of keys and values.

  Args:
    hdr: A header string, the contents of the .vhdr file.

  Returns:
    A dictionary (of the different info types) where each value is a section
    dictionary.
  """
  section_list = re.split(r'^\[', hdr, flags=re.MULTILINE)
  sections = {}
  for section in section_list:
    if section.startswith('Common Infos'):
      sections['Common Infos'] = parse_bv_keywords(section)
    elif section.startswith('Binary Infos'):
      sections['Binary Infos'] = parse_bv_keywords(section)
    elif section.startswith('Channel Infos'):
      channel_dict = parse_bv_keywords(section)
      for key, vals in channel_dict.items():
        if isinstance(vals, str):
          vals = vals.split(',')
          channel_name, reference_channel_name, resolution, unit = vals
          channel_dict[key] = {'channel_name': channel_name,
                               'reference_channel_name': reference_channel_name,
                               'resolution': float(resolution),
                               'unit': unit}
        else:
          raise TypeError('Expected a string of key-vals, not a %s.' %
                          type(vals))
      sections['Channel Infos'] = channel_dict
    elif section.startswith('Comment'):
      sections['Comment'] = section.split(']', 1)[1].split('\n')
  return sections


def read_bv_file(header_filename):
  """Read the BrainVision header and data files.

  Args:
    header_filename: where to read the header information (this file also points
    to the eeg data file.

  Returns:
    A tuple consisting of
    1) A dictionary of dictionaries with the header information
    2) a numpy array with the EEG data (num_frames x num_channels)
  """
  if not header_filename.endswith('.vhdr'):
    header_filename += '.vhdr'
  with tf.io.gfile.GFile(header_filename.encode('utf-8'), 'r') as fp:
    header_data = fp.read()
    header = parse_bv_header(header_data)
  data_filename = header['Common Infos']['DataFile']
  if '$b' in data_filename:
    basename = header_filename.rsplit('.', 1)[0]
    data_filename = data_filename.replace('$b', basename)
  # Kind of tricky since we don't really know the full path in the data file.
  if '/' in header_filename and '/' not in data_filename:
    dirname = os.path.dirname(header_filename)
    data_filename = os.path.join(dirname, data_filename)
  if header['Binary Infos']['BinaryFormat'] != 'IEEE_FLOAT_32':
    raise ValueError('Can\'t read BrainVision data that has format %s' %
                     header['Binary Infos']['BinaryFormat'])
  # Not defined by the standard, so assume little-endian...
  with tf.io.gfile.GFile(data_filename.encode('utf-8'), 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.float32)
  num_channels = header['Common Infos']['NumberOfChannels']
  data = np.reshape(data, (-1, num_channels))
  return header, data


class BvBrainDataFile(ingest.BrainDataFile):
  """Code to read the BV data file format."""

  def __init__(self, filename, data_type=None, **kwds):
    self._header = {}
    super(BvBrainDataFile, self).__init__(filename, data_type=data_type, **kwds)

  def load_all_data(self, data_dir):
    if not tf.io.gfile.exists(data_dir):
      raise IOError('Data_dir does not exist:', data_dir)
    data_filename = os.path.join(data_dir, self._data_filename)
    self._header, self._data = read_bv_file(data_filename)

  @property
  def signal_names(self):
    channel_keys = self._header['Channel Infos'].keys()
    return [self._header['Channel Infos'][k]['channel_name']
            for k in channel_keys]

  def signal_values(self, name):
    if not isinstance(name, str):
      raise ValueError('Must search for values with a string name.')
    channel_index = self.find_channel_index(name)
    channel_resolution = self.find_channel_resolution(name)
    if channel_index is not None:
      return self._data[:, channel_index] * channel_resolution
    return None

  def signal_fs(self, name):
    del name
    return 1e6/float(self._header['Common Infos']['SamplingInterval'])

  def find_channel_index(self, desired_label='TRIG'):
    """Look through the BV channel list for the desired channel index.

    Args:
      desired_label: what is the name of the channel we want to find.

    Returns:
      Return the index number in the data array.
    """
    assert 'Channel Infos' in self._header  # To keep type checker happy.
    for index, label in enumerate(self._header['Channel Infos'].keys()):
      if self._header['Channel Infos'][label]['channel_name'] == desired_label:
        return index
    return None

  def find_channel_resolution(self, desired_label='TRIG'):
    """Look through the BV channel list for the desired channel resolution.

    Args:
      desired_label: what is the name of the channel we want to find.

    Returns:
      Return the resolution of the desired channel.
    """
    assert 'Channel Infos' in self._header  # To keep type checker happy.
    for name in self._header['Channel Infos'].keys():
      if self._header['Channel Infos'][name]['channel_name'] == desired_label:
        return self._header['Channel Infos'][name]['resolution']
    return None
