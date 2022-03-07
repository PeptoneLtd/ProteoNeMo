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

import bz2
import os
import urllib.request
import subprocess
import sys
import subprocess

class UniParcDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/uniparc'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.download_url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/uniparc_active.fasta.gz'
        self.output_file = 'uniparc_active.fasta.gz'


    def download(self):
        url = self.download_url
        filename = self.output_file

        print('Downloading:', url)
        if os.path.isfile(self.save_path + '/' + filename):
            print('** Download file already exists, skipping download')
        else:
            cmd = ['wget', url, '--output-document={}'.format(self.save_path + '/' + filename)]
            print('Running:', cmd)
            status = subprocess.run(cmd)
            if status.returncode != 0:
                raise RuntimeError('UniParc download not successful')

        # Always unzipping since this is relatively fast and will overwrite
        print('Unzipping:', filename)
        subprocess.run('gunzip ' + self.save_path + '/' + filename, shell=True, check=True)

