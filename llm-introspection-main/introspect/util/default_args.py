
from argparse import Namespace

def default_model_type(args: Namespace) -> str:
    if args.model_type is not None:
        return args.model_type

    if args.model_id in ['meta-llama/Llama-2-70b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-7b-chat-hf']:
        return 'Llama2'
    elif args.model_id in ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-70B-Instruct']:
        return 'Llama3'
    elif args.model_id in ['tiiuae/falcon-40b-instruct', 'tiiuae/falcon-7b-instruct', 'tiiuae/Falcon3-1B-Instruct','tiiuae/Falcon3-3B-Instruct','tiiuae/Falcon3-7B-Instruct', 'tiiuae/Falcon3-10B-Instruct']:
        return 'Falcon'
    elif args.model_id in ['mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.1']:
        return 'Mistral'
    elif args.model_id in ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B-Instruct']:
        return 'Qwen'
    else:
        raise ValueError(f'unknown model-id {args.model_id}')

def default_model_id(args: Namespace) -> str:
    if args.model_id is not None:
        return args.model_id

    if args.model_name == 'llama2-70b':
        return 'meta-llama/Llama-2-70b-chat-hf'
    elif args.model_name == 'llama2-13b':
        return 'meta-llama/Llama-2-13b-chat-hf'
    elif args.model_name == 'llama2-7b':
        return  'meta-llama/Llama-2-7b-chat-hf'
    elif args.model_name == 'llama3-1b':
        return  'meta-llama/Llama-3.2-1B-Instruct'
    elif args.model_name == 'llama3-3b':
        return  'meta-llama/Llama-3.2-3B-Instruct'
    elif args.model_name == 'llama3-8b':
        return  'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model_name == 'llama3-70b':
        return  'meta-llama/Llama-3.1-70B-Instruct'
    elif args.model_name == 'qwen-1b':
        return 'Qwen/Qwen2.5-1.5B-Instruct'
    elif args.model_name == 'qwen-3b':
        return 'Qwen/Qwen2.5-3B-Instruct'
    elif args.model_name == 'qwen-7b':
        return 'Qwen/Qwen2.5-7B-Instruct'
    elif args.model_name == 'qwen-14b':
        return 'Qwen/Qwen2.5-14B-Instruct'
    elif args.model_name == 'qwen-32b':
        return 'Qwen/Qwen2.5-32B-Instruct'
    elif args.model_name == 'qwen-72b': 
        return 'Qwen/Qwen2.5-72B-Instruct'
    elif args.model_name == 'mistral-v1-7b':
        return 'mistralai/Mistral-7B-Instruct-v0.1'
    elif args.model_name == 'mistral-v2-7b':
        return 'mistralai/Mistral-7B-Instruct-v0.2'
    else:
        raise ValueError(f'unknown model-name {args.model_name}')

def default_system_message(args: Namespace) -> str:
    if args.system_message is not None:
        return args.system_message

    if args.model_type == 'Llama2':
        return 'none'
    if args.model_type == 'Llama3':
        return 'none'
    if args.model_type == 'Gemma':
        return 'none'
    if args.model_type == 'Falcon':
        return 'default'
    if args.model_type == 'Mistral':
        return 'none'
    if args.model_type == 'Qwen':
        return 'none'
    else:
        raise ValueError(f'unknown model-type {args.model_type}')
