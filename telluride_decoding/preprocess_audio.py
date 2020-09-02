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

"""Code to compute the audio intensity for preprocessing.

Code that stores incoming arbitrary audio data, and then yields fixed window
sizes for processing (like computing the intensity.)

After initializing the object, add a block of data to the object, and then
pull fixed sized blocks of data with a given half_window_width, and separated
by window_step samples. Data is always X x num_features, where X can change from
add_data call to call, but num_features must not change. Do not reuse the object
because it has internal state from previous calls.

"""

import numpy as np

from telluride_decoding import result_store


class AudioIntensityStore(result_store.WindowedDataStore):
  """Process a window of data, calculating the mean-squared value.
  """

  def next_window(self):
    for win in super(AudioIntensityStore, self).next_window():
      yield np.mean(np.square(win))


class AudioLoudnessMick(result_store.WindowedDataStore):
  """Process a window of data, using Mick's loudness approximation.
  """

  def next_window(self):
    for audio_data in super(AudioLoudnessMick, self).next_window():
      yield np.mean(np.abs(audio_data) ** np.log10(2))
