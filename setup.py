"""#############################################################################
Copyright 2022 Reem A. Suwaileh, rs081123@qu.edu.qa
This LMR model is adopting the NER example from HuggingFace library.
#############################################################################"""

#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_file = f.read()

setup(
    name='TLLMR4CM',
    version='1.1',
    description='Location Mention Recognition (LMR) in social media posts during disasters',
    long_description=readme,
    author='Reem A. Suwaileh',
    author_email='rs081123@qu.edu.qa',
    url='https://github.com/rsuwaileh/TLLMR4CM',
    packages=['TLLMR4CM'],
    package_data={'TLLMR4CM': ['data/hd-tb/*.txt']},
    install_requires=[
          'numpy==1.18.1',
          'tqdm==4.43.0',
          'torch==1.4.0',
          'transformers',
          'pytorch-pretrained-bert==0.4.0',
          'seqeval==0.0.12'
      ]
)
#packages=find_packages(),
