#!/bin/bash

# Disable Flash Attention for Blackwell GPU compatibility
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_FLASH_ATTN=1

# Run training
python train.py --config conf/config_libero_object.yaml "$@"
