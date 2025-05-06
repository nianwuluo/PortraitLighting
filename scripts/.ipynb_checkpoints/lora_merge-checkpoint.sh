#!/bin/bash

python MakeAnything-main/networks/flux_merge_lora.py \
  --flux_model "autodl-tmp/flux1-dev.safetensors" \
  --save_to "autodl-tmp/merged_model.safetensors" \
  --models "autodl-tmp/flux-lora-name.safetensors" \
  --loading_device 'cpu' --working_device 'cuda' --mem_eff_load_save\
  --ratios 1
