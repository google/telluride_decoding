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
    version='2.0.0',
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
