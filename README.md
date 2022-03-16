# ProteoNeMo
This repository containes the code for pre-training and inference procedures of **protein language models** with [Nvidia NeMo toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html) from Peptone Ltd.

[![GitHub Super-Linter](https://github.com/peptoneinc/ProteoNeMo/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

ProteoNeMo can be used to extract residue level representations of proteins and to train related downstream tasks.

## Table of Contents

- [ProteoNeMo](#proteonemo)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
    - [Quick start](#quick-start)
    - [Datasets](#datasets)
    - [Download and preprocess datasets](#download-and-preprocess-datasets)
    - [ProteoNeMo pre-training](#proteonemo-pre-training)
    - [Residue level representations extraction](#residue-level-representations-extraction)
  - [Licence](#licence)

## Usage

### Quick-start

As a prerequisite, you must have [NeMo 1.7](https://github.com/NVIDIA/NeMo) or later installed to use this repository.

Install the **proteonemo** package:

Clone the ProteoNeMo repository, go to the ProteoNeMo directory and run

```bash
python setup.py install
```

### Datasets

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

### Download and preprocess datasets

ProteoNeMo can be pre-trained on the datasets pointed-out above. You can choose your preferred one or make use of two or more of them at the same time.

Each dataset will be:
* **Downloaded** from [UniProt](https://www.uniprot.org/) and decopressed as a [.fasta](https://en.wikipedia.org/wiki/FASTA_format) file
* **Sharded** into several smaller `.txt` sub-files containing a random set of the related `.fasta` file, already splitted into **training** and **evaluation** samples
* **Tokenized** into several [.hdf5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, one for each `.txt` sharded file, where the **masking** procedure has been already applied

In the ProteoNeMo directory run:
```bash
export BERT_PREP_WORKING_DIR=<your_dir>
cd scripts
bash create_datasets_from_start.sh <to_download> 
```

Where:

- `BERT_PREP_WORKING_DIR` defines the directory where the data will be downloaded and preprocessed
- `<to_download>` defines the datasets we want to download and preprocess where `uniref_50_only` is the default. 

The outputs are the `download`, `sharded` and `hdf5` directories under the `$BERT_PREP_WORKING_DIR` parent directory, containing the related files.

| To Download | Datasets |
|-------------|----------|
| `uniref_50_only` | **UniRef 50** |
| `uniref_all`| **UniRef 50, 90** and **100** |
| `uniparc`| **UniParc** |
| `uniprotkb_all`| **UniProtKB Swiss-Prot, TrEMBL** and  **isoform sequences** |

### ProteoNeMo pre-training

Once the download and preprocessing procedure is completed you're ready to pre-train ProteoNeMo.

The pre-training procedure exploits NeMo to solve the **Masked Language Modeling** (Masked LM) task. One training instance of Masked LM is a single modified protein sequence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with a random token and the remaining 10% the token is retained. The task is then to predict the original token.

We have currently integrated [BERT](https://arxiv.org/abs/1810.04805)-like *uncased* models from [HuggingFace](https://huggingface.co/).

The first thing you need to do is creating a `model_config.yaml` file in the [conf](conf) directory, specifying the relative pre-training and model options. You can use [this](conf/bert_pretrained_from_preprocessed_config.yaml) config as template. 

Take a look to [these](https://github.com/NVIDIA/NeMo/tree/main/tutorials) NeMo tutorials to get familiar with such options.

Secondly, you have to modify the `config_name` argument of the `@hydra_runner` decorator in [bert_pretraining.py](scripts/bert_pretraining.py)

Lastly, in the ProteoNeMo directory run:
```bash
cd scripts
python bert_pretraining.py 
```

The pre-training will start and a progress bar will appear  ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/50)

#### Tensorboard monitoring

Once the pre-training procedure has started a `nemo_experiments` directory will be automatically created under the [scripts](scripts) directory. 

Based on the `name: <PretrainingModelName>` parameter in the `.yaml` configuration file, a `<PretrainingModelName>` sub-directory containing all the related pre-training experiment logs will be created under `nemo_experiments`.

In the ProteoNeMo directory run: 
```bash
tensorboard --logdir=scripts/nemo_experiments/<PretrainingModelName> 
```

The Tensorboard UI will be available on port 6006

### Residue level representations extraction

Once a ProteoNeMo model will be pre-trained you'll get a `.nemo` file, placed in the `nemo_path` you've utilised in the `.yaml` configuration file.

You're now ready to extract the residue level representations of each protein a `.fasta` file.

In the ProteoNeMo directory run:
```bash
cd scripts
python bert_eval.py --input_file <fasta_input_file> \
                    --vocab_file ../static/vocab.txt \
                    --output_dir <reprs_output_dir> \
                    --model_file <nemo_pretrained_model>
```

Where:

- `--input_file` defines the `.fasta` file containing the proteins for which you want to extract the residue level representations
- `--vocab_file` defines the `.txt` file containing the vacabulary you want to use during the inference phase. We suggets you use the [standard](static/vocab.txt) one
- `--output_dir` defines the output directory where the residue level representations will be written. You'll get a `.pt` file for each protein sequence in the `--input_file` 
- `--model_file` defines the `.nemo` file used to get the pre-trained weights needed to get the residue level representations

## Licence

This source code is licensed under the Apache 2.0 license found in the `LICENSE` file in the root directory of this source tree.
