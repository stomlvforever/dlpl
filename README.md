## 🧠While both papers aim to accelerate the IC design process by predicting parasitic effects before layout, they differ in scope and methodology.

Official implementation of the following papers:

- **Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs**
- **ParaGraph: Layout Parasitics and Device Parameter Prediction using Graph Neural Networks**

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

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{shen2024prelayout,
  title     = {Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs},
  author    = {Shan Shen and Dingcheng Yang and Yuyang Xie and Chunyan Pei and Wenjian Yu and Bei Yu},
  booktitle = {Proceedings of the Great Lakes Symposium on VLSI (GLSVLSI)},
  year      = {2024},
  publisher = {ACM},
  address   = {Clearwater, FL, USA},
  doi       = {10.1145/3649476.3658754},
  isbn      = {979-8-4007-0605-9/24/06}
}


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
