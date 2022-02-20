#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name='brainscore',
      version='1.0',
      description='Compute brain score of deep nets activations',
      author='Charlotte Caucheteux',
      author_email='ccaucheteux@fb.com',
      url='',
      # packages=['brainscore', 'src'],
      # package_dir={'brainscore': 'src'},
      packages=find_packages(include=['brainscore', 'brainscore.*'])
      )
