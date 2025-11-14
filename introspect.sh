#!/bin/bash

# Get input parameters
read -p "Enter endpoint URL (e.g., http://localhost:3000): " endpoint
read -p "Enter model name (llama3-{1b,3b,8b} or qwen-{1b,3b,7b}): " model_name

# Retry configuration
# Define flag combinations
flag_combos=(
    ""
    "e-persona-you"
    "e-persona-human"
    "e-implcit-target"
    "e-implcit-target e-persona-you"
    "e-implcit-target e-persona-human"
)

datasets=(
    "IMDB"
    "RTE"
)

# Run analysis for each flag combination with unlimited retries and no delay
for dataset in "${datasets[@]}"; do
    for flags in "${flag_combos[@]}"; do
        echo "Running analysis with flags: $flags"
        attempt=1
        while true; do
            python llm-introspection-main/experiments/analysis.py \
                --persistent-dir "$PWD/introspections" \
                --endpoint "$endpoint" \
                --task counterfactual \
                --task-config "$flags" \
                --model-name "$model_name" \
                --dataset "$dataset" \
                --split test \
                --seed 0 \
                --max-workers 1 \
                --client VLLM
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "Run succeeded."
                break
            fi
            echo "Run failed (exit code $rc). Retrying immediately... (attempt $attempt)"
            attempt=$((attempt + 1))
        done
    done
done