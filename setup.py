#!/usr/bin/env python

# -*- coding: utf-8 -*-

#
# Copyright (c) 2023-2024, Cyrille Favreau (cyrille.favreau@gmail.com)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""setup.py"""
import os
import pathlib
import pkg_resources
from setuptools import find_packages, setup

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    install_reqs = list()
    with pathlib.Path(reqs_file).open() as requirements_txt:
        install_reqs = [str(requirement)
                        for requirement
                        in pkg_resources.parse_requirements(requirements_txt)]
    return install_reqs


REQS = parse_reqs(os.path.join(BASEDIR, "requirements.txt"))

# read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    packages=find_packages(),
    install_requires=REQS,
    name='Nuon Model Visualizer',
    description='Python visualizer for the Nuon Model collisions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/favreau/NuonModelVisualizer.git',
    author='Cyrille Favreau',
    license='Apache License, Version 2.0',
    project_urls={
            "Documentation": "https://favreau.github.io/NuonModelVisualizer/",
            "Source": "https://github.com/favreau/NuonModelVisualizer",
    }
)
