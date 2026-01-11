import os
import re
from typing import TypedDict, Required, NotRequired
from timeit import default_timer as timer

import aiohttp
import asyncio
import json

from ..types import GenerateResponse, GenerateConfig, GenerateError
from ._abstract_client import AbstractClient

class OpenAIInfo(TypedDict):
    model: str
    provider: str

class OpenAIGenerateConfig(GenerateConfig):
    # Standard OpenAI API parameters
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop: NotRequired[list[str]]

class OpenAIGeneratePayload(TypedDict):
    model: Required[str]
    messages: NotRequired[list[dict[str, str]]]
    prompt: NotRequired[str]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop: NotRequired[list[str]]
    stream: NotRequired[bool]

class OpenAIError(Exception):
    pass

class OpenAICompatibleClient(AbstractClient[OpenAIInfo]):
    """This client connects to OpenAI-compatible API endpoints
    
    This works with many inference providers including:
    - Hugging Face Inference API
    - Together AI
    - Anyscale
    - OpenAI
    - And many others that implement OpenAI-compatible APIs
    
    Usage:
    - Set base_url to the provider's endpoint (e.g., "https://api.together.xyz/v1")
    - Set api_key in environment variable or pass it
    - Set model_name to the specific model (e.g., "meta-llama/Llama-3-70b-chat-hf")
    """
    
    def __init__(self, base_url: str, cache=None, connect_timeout_sec=60*60, 
                 max_reconnects=5, record=False):
        """
        Args:
            base_url: The API endpoint URL (e.g., "https://api.together.xyz/v1")
            
        API key and model name are read from environment variables:
        - API_KEY: The API key for authentication
        - MODEL_NAME: The model identifier to use
        
        The client automatically detects whether to use chat completions or completions
        based on the model name pattern.
        """
        super().__init__(base_url, cache, connect_timeout_sec, max_reconnects, record)
        
        # Get API key and model from environment variables
        self._api_key = os.environ.get('API_KEY')
        self._model_name = os.environ.get('MODEL_NAME', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
        
        # Determine if this is a chat model based on model name patterns
        chat_model_patterns = ['Qwen', 'gpt-', 'claude-', 'Chat']
        self._use_chat_completions = any(pattern in self._model_name for pattern in chat_model_patterns)
        
        # Common headers for OpenAI-compatible APIs
        self._headers = {
            'Content-Type': 'application/json',
        }
        
        if self._api_key:
            self._headers['Authorization'] = f'Bearer {self._api_key}'
        else:
            print("Warning: No API_KEY found in environment variables")

    async def _try_connect(self) -> bool:
        """Test connection with a simple request"""
        if self._use_chat_completions:
            payload: OpenAIGeneratePayload = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'Hello'}],
                'max_tokens': 1,
                'temperature': 0
            }
            endpoint = f'{self._base_url}/chat/completions'
        else:
            payload: OpenAIGeneratePayload = {
                'model': self._model_name,
                'prompt': 'Hello',
                'max_tokens': 1,
                'temperature': 0
            }
            endpoint = f'{self._base_url}/completions'

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            try:
                async with session.post(endpoint, json=payload, headers=self._headers) as response:
                    return response.status == 200
            except (aiohttp.ClientOSError, asyncio.TimeoutError):
                return False

    def _parse_qwen_prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """Parse a Qwen-formatted prompt back to messages array"""
        messages = []
        
        # Find all message blocks in order
        message_pattern = r'<\|im_start\|>(system|user|assistant)(?:\n(.*?))?(?:\n<\|im_end\|>|$)'
        matches = re.finditer(message_pattern, prompt, re.DOTALL)
        
        for match in matches:
            role = match.group(1)
            content = match.group(2)
            
            # Only add messages with actual content (except for assistant messages at the end which might be empty)
            if content and content.strip():
                messages.append({'role': role, 'content': content.strip()})
            elif role == 'assistant' and not content:
                # This is likely the final assistant turn waiting for completion - don't add it
                continue
        
        return messages

    async def _info(self) -> OpenAIInfo:
        return {
            'model': self._model_name,
            'provider': self._base_url
        }

    async def _generate(self, prompt: str, config: GenerateConfig) -> GenerateResponse:
        """Generate response using OpenAI-compatible API"""
        
        request_start_time = timer()
        
        if self._use_chat_completions:
            # Use chat completions endpoint with messages format
            # Check if this is a Qwen-formatted prompt and parse it
            if '<|im_start|>' in prompt:
                messages = self._parse_qwen_prompt_to_messages(prompt)
            else:
                messages = [{'role': 'user', 'content': prompt}]
                
            payload = {
                'model': self._model_name,
                'messages': messages,
                'max_tokens': config.get('max_new_tokens'),
                'temperature': config.get('temperature'),
                'top_p': config.get('top_p'),
                'stream': False
            }
            endpoint = f'{self._base_url}/chat/completions'
        else:
            # Use completions endpoint with prompt format
            payload = {
                'model': self._model_name,
                'prompt': prompt,
                'max_tokens': config.get('max_new_tokens'),
                'temperature': config.get('temperature'),
                'top_p': config.get('top_p'),
                'stream': False
            }
            endpoint = f'{self._base_url}/completions'
            
        # Add stop sequences if provided
        if config.get('stop'):
            payload['stop'] = config['stop']
            
        # Map repetition_penalty to frequency_penalty (approximate)
        if config.get('repetition_penalty') and config['repetition_penalty'] != 1.0:
            payload['frequency_penalty'] = (config['repetition_penalty'] - 1.0) * 2.0

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self._connect_timeout_sec,
                    connect=60,  # 60s for connection
                    sock_read=300  # 5min for reading response
                ),
                connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            ) as session:
                async with session.post(endpoint, json=payload, headers=self._headers) as response:
                    
                    if response.status == 429:  # Rate limited
                        error_text = await response.text()
                        raise GenerateError(f'Rate limited (429): {error_text}. Consider adding delays between requests.')
                    elif response.status == 503:  # Service unavailable  
                        error_text = await response.text()
                        raise GenerateError(f'Service unavailable (503): {error_text}. Provider may be overloaded.')
                    elif response.status != 200:
                        error_text = await response.text()
                        raise OpenAIError(f'API request failed with status {response.status}: {error_text}')
                    
                    answer = await response.json()
                    
                    # Extract the generated text from response
                    if 'choices' not in answer or not answer['choices']:
                        raise OpenAIError('No choices in API response')
                    
                    if self._use_chat_completions:
                        # Chat completions format
                        response_text = answer['choices'][0]['message']['content']
                    else:
                        # Completions format
                        response_text = answer['choices'][0]['text']
                        
                    duration = timer() - request_start_time
                    
                    return {
                        'response': response_text,
                        'duration': duration
                    }

        except (OpenAIError, asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as err:
            if 'rate' in str(err).lower() or 'limit' in str(err).lower() or '429' in str(err):
                raise GenerateError(f'Rate limiting detected: {str(err)}. Consider adding delays between requests.') from err
            elif 'timeout' in str(err).lower() or 'connect' in str(err).lower():
                raise GenerateError(f'Connection/timeout error: {str(err)}. Provider may be overloaded.') from err
            else:
                raise GenerateError(f'LLM generate failed: {str(err)}') from err
