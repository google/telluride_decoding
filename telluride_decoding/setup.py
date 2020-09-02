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

"""Create the attention_decoder package files.
"""

import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

# Fix up reference to the image on the decscription page.  Point to GitHub
long_description = long_description.replace(
    'doc/AuditoryAttentionDecoding.jpg',
    'https://raw.githubusercontent.com/google/telluride_decoding/master/doc/'
    'AuditoryAttentionDecoding.jpg')

long_description = long_description.replace(
    'doc/decoding.md',
    'https://github.com/google/telluride_decoding/blob/master/doc/decoding.md')

setuptools.setup(
    name='telluride_decoding',
    version='2.1.1',   # TF2 and Python 3 (no Py2)
    author='Malcolm Slaney',
    author_email='telluride-decoding-maintainers@googlegroups.com',
    description='Telluride Decoding Toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/telluride_decoding',
    packages=['telluride_decoding'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'attrs',
        'matplotlib',  # For plotting
        'mock',  # For some of the tests
        'pandas',  # For regression_data to read in results
        'pyedflib',  # For ingesting EEG data in EDF format
        'tensorflow>=2',
    ],
    include_package_data=True,  # Using the files specified in MANIFEST.in
)
