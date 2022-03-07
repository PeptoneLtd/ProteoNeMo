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

class UniProtKBDownloader:
    def __init__(self, annotation, save_path):
        self.save_path = save_path + '/uniprotkb_' + annotation

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.annotation = annotation
        self.download_urls = {
            'swissprot' : 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',
            'trembl' : 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',
            'isoformseqs' : 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot_varsplic.fasta.gz'
        }

        self.output_files = {
            'swissprot' : 'uniprot_sprot.fasta.gz',
            'trembl' : 'uniprot_trembl.fasta.gz',
            'isoformseqs' : 'uniprot_sprot_varsplic.fasta.gz'
        }


    def download(self):
        if self.annotation in self.download_urls:
            url = self.download_urls[self.annotation]
            filename = self.output_files[self.annotation]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + filename):
                print('** Download file already exists, skipping download')
            else:
                cmd = ['wget', url, '--output-document={}'.format(self.save_path + '/' + filename)]
                print('Running:', cmd)
                status = subprocess.run(cmd)
                if status.returncode != 0:
                    raise RuntimeError('UniProtKB download not successful')

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.annotation])
            subprocess.run('gunzip ' + self.save_path + '/' + filename, shell=True, check=True)

        else:
            assert False, 'UniProtKBDownloader not implemented for this cluster yet.'
