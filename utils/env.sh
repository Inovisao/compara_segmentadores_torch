#!/bin/bash

# Instala as dependências
conda create -n seg_torch
conda activate seg_torch
pip install torch numpy opencv-python torchvision tqdm albumentations matplotlib