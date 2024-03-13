#!/bin/bash

# Instala as dependências
conda create -y -n seg_comp python=3.11
conda activate seg_comp
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia pandas seaborn transformers
pip install pycocotools albumentations timm einops 