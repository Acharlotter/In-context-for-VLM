#!/bin/bash
# Evaluation script for VQAv2 using Qwen2.5-VL with ICL

DEVICE=0  # GPU number
RANDOM_ID="VQAv2_Qwen_Result"
RESULTS_FILE="results_${RANDOM_ID}.json"

# Set paths to Qwen2.5-VL model
# Download from: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
QWEN_MODEL_PATH="path/to/Qwen2.5-VL-7B-Instruct"  # Change this to your model path
QWEN_TOKENIZER_PATH="path/to/Qwen2.5-VL-7B-Instruct"  # Usually same as model path

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python eval/evaluate_vqa.py \
    --model qwen2_5vl \
    --retrieval_name $RANDOM_ID \
    --qwen_model_path $QWEN_MODEL_PATH \
    --qwen_tokenizer_path $QWEN_TOKENIZER_PATH \
    --precision fp16 \
    --device $DEVICE \
    --eval_vqav2 \
    --num_shots 2 \
    --shots 0 2 4 \
    --num_trials 1 \
    --trial_seeds 0 \
    --results_file $RESULTS_FILE
