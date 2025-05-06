#!/bin/bash

BASE_FLUX_CHECKPOINT="autodl-tmp/merged_model.safetensors"
LORA_WEIGHTS_PATH="result/output_name-step00000500.safetensors"
OUTPUT_DIR="result"

CLIP_L_PATH="autodl-tmp/clip/clip_l.safetensors"
T5XXL_PATH="autodl-tmp/clip/t5xxl_fp16.safetensors"
AE_PATH="autodl-tmp/ae.safetensors"

#SAMPLE_IMAGES_FILE="sample/sample.txt"
#SAMPLE_PROMPTS_FILE="sample/sample_prompts/1.txt"
SAMPLE_IMAGES_FILE="sample-ch/sample.txt"
SAMPLE_PROMPTS_FILE="sample-ch/sample_prompts/1.txt"
frame_num=4     # 4 or 9



python MakeAnything-main/flux_inference_recraft.py \
    --base_flux_checkpoint "$BASE_FLUX_CHECKPOINT" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --clip_l_path "$CLIP_L_PATH" \
    --t5xxl_path "$T5XXL_PATH" \
    --ae_path "$AE_PATH" \
    --sample_images_file "$SAMPLE_IMAGES_FILE" \
    --sample_prompts_file "$SAMPLE_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --frame_num $frame_num