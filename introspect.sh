#!/bin/bash

# Get input parameters
read -p "Enter endpoint URL of VLLM server: " endpoint
echo "Choose a model:"
select model_name in llama3-1b llama3-3b llama3-8b qwen-1b qwen-3b qwen-7b; do
    [[ -n "$model_name" ]] && break
    echo "Invalid selection."
done

echo "You picked: $model_name"

# Retry configuration
# Define flag combinations
flag_combos=(
    ""
    "e-persona-you"
    "e-persona-human"
    "e-implcit-target"
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
    while true; do
        python llm-introspection-main/experiments/analysis.py \
            --persistent-dir "$PWD/introspections" \
            --endpoint "$endpoint" \
            --task counterfactual \
            --task-config "e-implcit-target" "e-persona-you" \
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
    while true; do
        python llm-introspection-main/experiments/analysis.py \
            --persistent-dir "$PWD/introspections" \
            --endpoint "$endpoint" \
            --task counterfactual \
            --task-config "e-implcit-target" "e-persona-human" \
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