
## 📑 Table of Contents

- [Introduction](#-introduction)
- [Repository Structure](#repository-structure)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#-usage)
  - [Dataset Preparation](#dataset-preparation)
    - [Dataset Download Instructions](#dataset-download-instructions)
    - [List of Datasets](#list-of-datasets)
    - [Dataset Path](#dataset-path)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Framework Components](#framework-components)
- [Citation](#-citation)
- [License](#-license)




## 🧠 Introduction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch 2.2.0](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/get-started/previous-versions/#v220)
[![PyG 2.6.1](https://img.shields.io/badge/PyG-2.6.1-orange)](https://pytorch-geometric.readthedocs.io/en/2.6.1/)

While both papers aim to accelerate the IC design process by predicting parasitic effects before layout, they differ in scope and methodology.
Official implementation of the following papers:

| Title                                                                                    | Link                                                                                                                 |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Design           | [OpenReview](https://openreview.net/forum?id=xkljKdGe4E)<br>[ACM DL](https://dl.acm.org/doi/10.1145/3649476.3658754) |
| ParaGraph: Layout Parasitics and Device Parameter Prediction using Graph Neural Networks | [arXiv](https://arxiv.org/abs/2502.09263)<br>[IEEE Xplore](https://ieeexplore.ieee.org/document/9218515)             |


This repository provides the code and dataset for reproducing the results from the above papers. The approach utilizes Graph Neural Networks (GNNs) to predict parasitic capacitance and device parameters before physical layout, accelerating the design process for SRAM and other circuits.


## 💻 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/username/dlpl.git
cd dlpl

# Create and activate a conda environment
conda create -n dlpl python=3.10
conda activate dlpl

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Dataset Preparation

#### Dataset Download Instructions

The datasets used for training and testing CircuitGCL are available for download via the following links. You can use `curl` to directly download these files from the provided URLs.

##### List of Datasets

| Dataset Name    | Description                          | Download Link                                                                              |
| --------------- | ------------------------------------ | ------------------------------------------------------------------------------------------ |
| SSRAM           | Static Random Access Memory dataset  | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/ssram.pt)           |
| DIGITAL_CLK_GEN | Digital Clock Generator dataset      | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/digtime.pt)         |
| TIMING_CTRL     | Timing Control dataset               | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/timing_ctrl.pt)     |
| ARRAY_128_32    | Array with dimensions 128x32 dataset | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/array_128_32_8t.pt) |
| ULTRA8T         | Ultra 8 Transistor dataset           | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/ultra8t.pt)         |
| SANDWICH-RAM    | Sandwich RAM dataset                 | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/sandwich.pt)        |
| SP8192W         | Specialized 8192 Width dataset       | [Download](https://circuitgcl-sram.s3.ap-southeast-2.amazonaws.com/raw/sp8192w.pt)         |

### Dataset path
After downloading the above dataset, please add its path into the dataset_dir='' field in the main.py function.
```bash

dataset_dir=''
```

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{shen2024prelayout,
  title     = {Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs},
  author    = {Shan Shen and Dingcheng Yang and Yuyang Xie and Chunyan Pei and Wenjian Yu and Bei Yu},
  booktitle = {Proceedings of the Great Lakes Symposium on VLSI (GLSVLSI)},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3649476.3658754},
  isbn      = {979-8-4007-0605-9/24/06}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
