# External Inference Providers for Llama3-70B

This guide explains how to run your experiments using external inference providers instead of hosting your own vLLM server. This is much more cost-effective for large models like Llama3-70B.

## Quick Start

1. **Use the new script**: `./introspect_external.sh`
2. **Choose a provider** from the menu
3. **Set your API key** as prompted
4. **Run your experiment** as usual

## Supported Providers

### 1. Together AI (Recommended)
- **Endpoint**: `https://api.together.xyz/v1`
- **Model**: `meta-llama/Meta-Llama-3-70B-Instruct`
- **Cost**: ~$0.90 per 1M tokens
- **Setup**: Get API key from https://api.together.xyz/

```bash
export TOGETHER_API_KEY=your_api_key_here
./introspect_external.sh
```

### 2. Hugging Face Inference API
- **Endpoint**: `https://api-inference.huggingface.co/v1`
- **Model**: `meta-llama/Meta-Llama-3-70B-Instruct`
- **Cost**: Pay-per-use
- **Setup**: Get token from https://huggingface.co/settings/tokens

```bash
export HF_TOKEN=your_token_here
./introspect_external.sh
```

### 3. Anyscale
- **Endpoint**: `https://api.endpoints.anyscale.com/v1`
- **Model**: `meta-llama/Meta-Llama-3-70B-Instruct`
- **Cost**: Competitive pricing
- **Setup**: Get API key from https://console.anyscale.com/

```bash
export ANYSCALE_API_KEY=your_api_key_here
./introspect_external.sh
```

## Manual Setup

If you prefer to set things up manually:

```bash
# Set your API credentials
export API_KEY=your_api_key
export MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct

# Run your experiment
python llm-introspection-main/experiments/analysis.py \
    --persistent-dir "$PWD/introspections" \
    --endpoint "https://api.together.xyz/v1" \
    --task counterfactual \
    --model-name llama3-70b \
    --dataset IMDB \
    --split test \
    --client OpenAI
```

## Cost Comparison

| Provider | Cost per 1M tokens | Notes |
|----------|-------------------|-------|
| Together AI | ~$0.90 | Fast, reliable |
| Hugging Face | Variable | May have rate limits |
| Anyscale | ~$1.00 | Good performance |
| Self-hosted 70B | $50-100/hour | Very expensive |

## Testing Your Setup

Use the test script to verify your connection:

```bash
export API_KEY=your_api_key
export MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
python test_openai_client.py
```

## Troubleshooting

### "API key not found"
Make sure you've exported the correct environment variable for your provider.

### "Model not found" 
Check that you're using the correct model name for your provider. Model names may vary between providers.

### Rate limiting
Some providers have rate limits. The script automatically retries failed requests.

### Timeout errors
Increase the timeout in the client if you're getting timeouts:
- Large models may take longer to respond
- Network issues can cause delays

## Provider-Specific Notes

### Together AI
- Fast inference
- Good rate limits
- Supports many Llama variants

### Hugging Face
- Free tier available
- May have slower cold starts
- Good for experimentation

### Anyscale
- Enterprise-focused
- Good reliability
- Competitive pricing