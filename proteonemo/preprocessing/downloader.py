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

from proteonemo.preprocessing.uniref_downloader import UniRefDownloader
from proteonemo.preprocessing.uniprotkb_downloader import UniProtKBDownloader
from proteonemo.preprocessing.UniParcDownloader import UniParcDownloader


class Downloader:

    def __init__(self, dataset_name, save_path):
        self.dataset_name = dataset_name
        self.save_path = save_path

    def download(self):
        if self.dataset_name == 'uniref_50':
            self.download_uniref('50')

        elif self.dataset_name == 'uniref_90':
            self.download_uniref('90')

        elif self.dataset_name == 'uniref_100':
            self.download_uniref('100')

        elif self.dataset_name == 'uniprotkb_swissprot':
            self.download_uniprotkb("swissprot")

        elif self.dataset_name == 'uniprotkb_trembl':
            self.download_uniprotkb("trembl")

        elif self.dataset_name == 'uniprotkb_isoformseqs':
            self.download_uniprotkb("isoformseqs")

        elif self.dataset_name == 'uniparc':
            self.download_uniparc()

        elif self.dataset_name == 'all':
            self.download_uniref('50')
            self.download_uniref('90')
            self.download_uniref('100')
            self.download_uniprotkb("swissprot")
            self.download_uniprotkb("trembl")
            self.download_uniprotkb("isoformseqs")
            self.download_uniparc()

        else:
            print(self.dataset_name)
            assert False, 'Unknown dataset_name provided to downloader'

    def download_uniref(self, clusters):
        downloader = UniRefDownloader(clusters, self.save_path)
        downloader.download()

    def download_uniprotkb(self, annotation):
        downloader = UniProtKBDownloader(annotation, self.save_path)
        downloader.download()

    def download_uniparc(self):
        downloader = UniParcDownloader(self.save_path)
        downloader.download()
