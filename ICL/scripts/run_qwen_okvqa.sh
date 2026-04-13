#!/bin/bash
# Evaluation script for OK-VQA using Qwen2.5-VL with ICL

DEVICE=0  # GPU number
RANDOM_ID="OKVQA_Qwen_Result"
RESULTS_FILE="results_${RANDOM_ID}.json"

# Set paths to Qwen2.5-VL model
QWEN_MODEL_PATH="path/to/Qwen2.5-VL-7B-Instruct"
QWEN_TOKENIZER_PATH="path/to/Qwen2.5-VL-7B-Instruct"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python eval/evaluate_vqa.py \
    --model qwen2_5vl \
    --retrieval_name $RANDOM_ID \
    --qwen_model_path $QWEN_MODEL_PATH \
    --qwen_tokenizer_path $QWEN_TOKENIZER_PATH \
    --precision fp16 \
    --device $DEVICE \
    --eval_ok_vqa \
    --num_shots 2 \
    --shots 0 2 4 \
    --num_trials 1 \
    --trial_seeds 0 \
    --results_file $RESULTS_FILE
