# Copyright (c) 2021 Peptone.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
from setuptools import setup

from adopt import __version__

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteonemo",
    version=__version__,
    packages=setuptools.find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/PeptoneInc/ProteoNeMo",
    license="Apache License, Version 2.0",
    author="Peptone Ltd.",
    author_email="carlo@peptone.io",
    description=" Ongoing research training transformer models on proteome at scale",
    data_files=[(".", ["LICENSE", "README.md", "CHANGELOG.md", "CITATION.cff"])],
    zip_safe=True,
)
