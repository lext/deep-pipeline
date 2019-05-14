#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = ('numpy', 'opencv-python', 'torch>=1.0.0', 'solt>=0.1.5')

setup_requirements = ()

test_requirements = ('pytest',)

description = """A collection of building blocks for deep learning pipelines"""

setup(
    author="Aleksei Tiulpin",
    author_email='aleksei.tiulpin@oulu.fi',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    description="Building blocks for Deep Learning",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='deeep learning, image segmentation, image-classification',
    name='deep-pipeline',
    packages=find_packages(include=['deeppipeline']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lext/deep-pipeline',
    version='0.0.1',
    zip_safe=False,
)
