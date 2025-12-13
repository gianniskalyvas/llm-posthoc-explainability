#!/bin/bash

# Configuration for external inference providers
echo "External Inference Provider Setup for Llama3-70B"
echo "================================================"

echo "Choose an inference provider:"
select provider in "Together AI" "Hugging Face" "Anyscale" "OpenAI" "Featherless AI" "Custom OpenAI-compatible"; do
    [[ -n "$provider" ]] && break
    echo "Invalid selection."
done

echo "You selected: $provider"

# Set default configurations based on provider
case $provider in
    "Together AI")
        default_endpoint="https://api.together.xyz/v1"
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        echo "Together AI selected."
        echo "You'll need to set your TOGETHER_API_KEY environment variable."
        echo "Visit: https://api.together.xyz/ to get your API key"
        api_key_var="TOGETHER_API_KEY"
        ;;
    "Hugging Face")
        default_endpoint="https://api-inference.huggingface.co/v1"
        default_model="meta-llama/Meta-Llama-3-70B-Instruct"
        echo "Hugging Face Inference API selected."
        echo "You'll need to set your HF_TOKEN environment variable."
        echo "Visit: https://huggingface.co/settings/tokens to get your token"
        api_key_var="HF_TOKEN"
        ;;
    "Anyscale")
        default_endpoint="https://api.endpoints.anyscale.com/v1"
        default_model="meta-llama/Meta-Llama-3-70B-Instruct"
        echo "Anyscale selected."
        echo "You'll need to set your ANYSCALE_API_KEY environment variable."
        echo "Visit: https://console.anyscale.com/ to get your API key"
        api_key_var="ANYSCALE_API_KEY"
        ;;
    "OpenAI")
        default_endpoint="https://api.openai.com/v1"
        default_model="gpt-4"
        echo "OpenAI selected (Note: No Llama3-70B available, using GPT-4)"
        echo "You'll need to set your OPENAI_API_KEY environment variable."
        api_key_var="OPENAI_API_KEY"
        ;;
    "Featherless AI")
        default_endpoint="https://api.featherless.ai/v1"
        default_model="Qwen/Qwen2.5-32B-Instruct"
        echo "Featherless AI selected."
        echo "You'll need to set your FEATHERLESS_API_KEY environment variable."
        echo "Visit: https://featherless.ai/ to get your API key"
        api_key_var="FEATHERLESS_API_KEY"
        ;;
    "Custom OpenAI-compatible")
        read -p "Enter the API endpoint URL: " default_endpoint
        read -p "Enter the model name: " default_model
        read -p "Enter the environment variable name for API key: " api_key_var
        ;;
esac

# Get user inputs with defaults
read -p "API Endpoint [$default_endpoint]: " endpoint
endpoint=${endpoint:-$default_endpoint}

read -p "Model name [$default_model]: " model_name
model_name=${model_name:-$default_model}

echo "API Key Environment Variable: $api_key_var"

# Check if API key is set
if [ -z "${!api_key_var}" ]; then
    echo "WARNING: $api_key_var is not set!"
    echo "Please set it with: export $api_key_var=your_api_key"
    echo "Or set it for this session:"
    read -p "Enter your API key: " -s api_key
    echo
    export "$api_key_var=$api_key"
fi

echo "Configuration:"
echo "  Provider: $provider"
echo "  Endpoint: $endpoint"
echo "  Model: $model_name"
echo "  API Key: $(echo ${!api_key_var} | cut -c1-8)..."

# Updated model selection for your script
echo "Choose model identifier for your experiment:"
select exp_model in llama3-70b llama3-1b llama3-3b llama3-8b qwen-1b qwen-3b qwen-7b qwen-14b qwen-32b qwen-72b; do
    [[ -n "$exp_model" ]] && break
    echo "Invalid selection."
done

echo "Experiment model identifier: $exp_model"

# Retry configuration
flags=(
    ""
    "e-persona-you"
    "e-persona-human"
    "e-implcit-target"
)

datasets=(
     "IMDB"
     "RTE"
)

# Export the API key for the Python script
export API_KEY="${!api_key_var}"
export MODEL_NAME="$model_name"

# Run analysis for each flag combination with unlimited retries and no delay
for dataset in "${datasets[@]}"; do
    for flag in "${flags[@]}"; do
        echo "Running analysis with flags: $flag"
        attempt=1
        while true; do
            python llm-introspection-main/experiments/analysis.py \
                --persistent-dir "$PWD/introspections" \
                --endpoint "$endpoint" \
                --task counterfactual \
                --task-config "$flag" \
                --model-name "$exp_model" \
                --dataset "$dataset" \
                --split test \
                --seed 0 \
                --max-workers 1 \
                --client OpenAI 
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "Run succeeded."
                break
            fi
            echo "Run failed (exit code $rc). Retrying immediately... (attempt $attempt)"
            attempt=$((attempt + 1))
        done
    done
    
    # Combined configurations
    attempt=1
    while true; do
        python llm-introspection-main/experiments/analysis.py \
            --persistent-dir "$PWD/introspections" \
            --endpoint "$endpoint" \
            --task counterfactual \
            --task-config "e-implcit-target" "e-persona-you" \
            --model-name "$exp_model" \
            --dataset "$dataset" \
            --split test \
            --seed 0 \
            --max-workers 1 \
            --client OpenAI 
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "Run succeeded."
            break
        fi
        echo "Run failed (exit code $rc). Retrying immediately... (attempt $attempt)"
        attempt=$((attempt + 1))
    done
    
    attempt=1
    while true; do
        python llm-introspection-main/experiments/analysis.py \
            --persistent-dir "$PWD/introspections" \
            --endpoint "$endpoint" \
            --task counterfactual \
            --task-config "e-implcit-target" "e-persona-human" \
            --model-name "$exp_model" \
            --dataset "$dataset" \
            --split test \
            --seed 0 \
            --max-workers 1 \
            --client OpenAI 
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "Run succeeded."
            break
        fi
        echo "Run failed (exit code $rc). Retrying immediately... (attempt $attempt)"
        attempt=$((attempt + 1))
    done
done

echo "Experiment completed!"