# ProteoNeMo
This repository containes the code for pre-training and inference procedures of **protein language models** with [Nvidia NeMo toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html) from Peptone Ltd.

[![GitHub Super-Linter](https://github.com/peptoneinc/ProteoNeMo/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

ProteoNeMo can be used to extract residue level representations of proteins and to train related downstream tasks.

ProteoNeMo can be pre-trained on:
* [UniRef](https://www.uniprot.org/uniref/)
  * UniRef 50
  * UniRef 90
  * UniRef 100
* [UniParc](https://www.uniprot.org/uniparc/)  
* [UniProtKB](https://www.uniprot.org/uniprot/)
  * UniProtKB Swiss-Prot
  * UniProtKB TrEMBL
  * [UniProtKB isoform sequences](https://www.uniprot.org/help/canonical_and_isoforms)

## Table of Contents

- [ProteoNeMo](#proteonemo)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
    - [Quick start](#quick-start)
    - [Download and preprocess datasets](#download-and-preprocess-datasets)
  - [Licence](#licence)

## Usage

### Quick-start

As a prerequisite, you must have [NeMo 1.7](https://github.com/NVIDIA/NeMo) or later installed to use this repository.

Install the **proteonemo** package:

Clone the ProteoNeMo repository, go to the ProteoNeMo directory and run

```bash
python setup.py install
```

### Download and preprocess datasets

ProteoNeMo can be pre-trained on the datasets pointed-out above. You can choose your preferred one or make use of two or more of them at the same time.

Each dataset will be:
* **Downloaded** from [UniProt](https://www.uniprot.org/) and decopressed as a [.fasta](https://en.wikipedia.org/wiki/FASTA_format) file
* **Sharded** into several smaller `.txt` sub-files containing a random set of the related `.fasta` file, already splitted into **training** and **evaluation** samples
* **Tokenized** into several [.hdf5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, one for each `.txt` sharded file, where the **masking** procedure has been already applied

In the ProteoNeMo directory run:
```bash
BERT_PREP_WORKING_DIR = <your_dir>
cd scripts
bash create_datasets_from_start.sh <to_download> 
```

Where:

- `BERT_PREP_WORKING_DIR` defines the directory where the data will be downloaded and preprocessed
- `<to_download>` defines the datasets we want to download and preprocess where `uniref_50_only` is the default. 

The outputs are the *download*, *sharded* and *hdf5* directories under the `BERT_PREP_WORKING_DIR` parent directory, containing the related files.

| To Download | Datasets |
|-------------|----------|
| `uniref_50_only` | **UniRef 50** |
| `uniref_all`| **UniRef 50, 90** and **100** |
| `uniparc`| **UniParc** |
| `uniprotkb_all`| **UniProtKB Swiss-Prot, TrEMBL** and  **isoform sequences** |

## Licence

This source code is licensed under the Apache 2.0 license found in the `LICENSE` file in the root directory of this source tree.
