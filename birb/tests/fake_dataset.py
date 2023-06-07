# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""A fake dataset for testing."""

from chirp.data import bird_taxonomy
import numpy as np


class FakeDataset(bird_taxonomy.BirdTaxonomy):
  """Fake dataset."""

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(100),
        'test': self._generate_examples(20),
    }

  def _generate_examples(self, num_examples):
    for i in range(num_examples):
      yield i, {
          'audio': np.random.uniform(
              -1.0, 0.99, self.info.features['audio'].shape
          ).astype(np.float32),
          'recording_id': i,
          'segment_id': -1 + i,
          'segment_start': 17,
          'segment_end': 17 + self.info.features['audio'].shape[0],
          'label': np.random.choice(
              self.info.features['label'].names, size=[3]
          ),
          'bg_labels': np.random.choice(
              self.info.features['bg_labels'].names, size=[2]
          ),
          'filename': 'placeholder',
          'quality_score': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F']),
          'license': '//creativecommons.org/licenses/by-nc-sa/4.0/',
          'country': np.random.choice(['South Africa', 'Colombia', 'Namibia']),
          'altitude': str(np.random.uniform(0, 3000)),
          'length': np.random.choice(['1:10', '0:01']),
          'bird_seen': np.random.choice(['yes', 'no']),
          'latitude': str(np.random.uniform(0, 90)),
          'longitude': str(np.random.uniform(0, 90)),
          'playback_used': 'yes',
          'recordist': 'N/A',
          'remarks': 'N/A',
          'sound_type': np.random.choice(['call', 'song']),
      }
