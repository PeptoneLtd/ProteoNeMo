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
import importlib.util

spec = importlib.util.spec_from_file_location('package_info', 'proteonemo/package_info.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__description__ = package_info.__description__
__download_url__ = package_info.__download_url__
__homepage__ = package_info.__homepage__
__keywords__ = package_info.__keywords__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__

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
