# dlpl
Official implementation of the paper "Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs".
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
