> **Anonymous Submission**
> This repository contains the official implementation for the paper submission. For double-blind review purposes, all author identities, affiliations, and absolute paths have been rigorously anonymized.

---

##  Environment Setup

We recommend using Anaconda or Miniconda to manage the environment.

```bash
# Create and activate environment
conda create -n vlm_attack python=3.10 -y
conda activate vlm_attack

# Install PyTorch (Modify according to your CUDA version)
pip install torch torchvision torchaudio --index-url 

# Install required packages
pip install -r requirements.txt