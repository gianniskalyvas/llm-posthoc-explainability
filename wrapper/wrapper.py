from textattack.models.wrappers import ModelWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
import numpy as np

import nltk
# ensure nltk resource is present; use canonical resource name and avoid repeated downloads in CI
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

import sys, os
# make introspect/tasks path relative to this repository instead of using an absolute path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
introspect_tasks_path = os.path.join(project_root, 'introspect', 'tasks')
if os.path.isdir(introspect_tasks_path):
    sys.path.insert(0, introspect_tasks_path)
else:
    # fallback: attempt one level up (useful if workspace layout differs)
    alt_path = os.path.abspath(os.path.join(project_root, '..', 'llm-introspection-main', 'introspect', 'tasks'))
    if os.path.isdir(alt_path):
        sys.path.insert(0, alt_path)
    else:
        # keep previous behavior minimal: do not insert a hardcoded absolute path
        pass

from _common_match import match_contains, match_pair_match, match_startwith

class TextAttackWrapper(ModelWrapper):
    def __init__(self, model_family, model, tokenizer, task, max_new_tokens=1024, batch_size=500):
        self.model = model.eval().to('cuda')
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.task = task
        self.model_family = model_family
        self.batch_size = batch_size

    def make_prompt(self, texts) -> str:
        if self.task == "sentiment":
            paragraph = texts[0]
            system_msg = f'You are a sentiment classifier. Answer only "positive" or "negative". Do not explain the answer. What is the sentiment of the user\'s paragraph?'
        elif self.task == "entailment":
            hypothesis = texts[0].split('.')[0]
            paragraph = texts[0].split('.')[1]
            system_msg =  f'You are an entailment classifier. Does the statement "{hypothesis}" entail from the following paragraph? Answer either "yes" for entailment or "no" for no entailment. Do not explain the answer.'
        else:
            raise ValueError("Task must be either sentiment or entailment!")

        if self.model_family == 'llama':
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nParagraph: {paragraph}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif self.model_family == 'qwen':
            prompt = f"<|im_start|>system\n{system_msg}\n<|im_end|><|im_start|>user\nParagraph: {paragraph}\n<|im_end|><|im_start|>assistant"
        else:
            raise ValueError("Model-Family must be either llama or _!")
        return prompt

    @staticmethod
    def _extract_sentiment(source: str):
        source = source.lower()
        pair_match_prefixes = ('could be','are multiple','to express','might be','are some','be some','it contains','paragraph contains','paragraph has','tool detects','be considered','classified as','there are','seems','seems to be','seems to be mostly','appears to be','is:','is')

        if match_startwith(('positive', 'sentiment: positive'))(source) \
        or match_pair_match(pair_match_prefixes, ('positive', '"positive"'))(source):
            sentiment = 'positive'
        elif match_startwith(('negative', 'sentiment: negative'))(source) \
        or match_pair_match(pair_match_prefixes, ('negative', '"negative"'))(source):
            sentiment = 'negative'
        elif match_startwith(('mixed', 'neutral'))(source) \
        or match_pair_match(pair_match_prefixes, ('neutral', '"neutral"', 'mixed', '"mixed"'))(source):
            sentiment = 'neutral'
        elif match_startwith(('unknown', 'i am sorry', 'sorry'))(source) \
        or match_pair_match(pair_match_prefixes, ('unknown', '"unknown"'))(source) \
        or match_contains(('both positive and negative','difficult to determine','no explicit sentiments','no clear sentiment','cannot provide','unable to determine','cannot determine','cannot be determined','cannot be accurately determined'))(source):
            sentiment = 'unknown'
        else:
            sentiment = None

        return sentiment

    @staticmethod
    def _extract_entailment(source: str):
        source = source.lower()

        if match_contains((
            'unknown',
            'cannot provide',
            'cannot determine',
            'insufficient context',
            'unable to determine',
            'impossible for me to determine'
        ))(source):
            return 'unknown'
        elif match_contains(('yes',))(source):
            return 'yes'
        elif match_contains(('no',))(source):
            return 'no'

        return None


    def __call__(self, text_list):
        device = next(self.model.parameters()).device
        all_responses, confidences = [], []

        for i in range(0, len(text_list), self.batch_size):
            batch_texts = text_list[i : i + self.batch_size]
            formatted_texts = [self.make_prompt([t]) for t in batch_texts]
            inputs = self.tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to(device)

            seed = 0
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    repetition_penalty=1.0,
                    do_sample=False,
                    top_p=1,
                    top_k=0,
                    temperature=0,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            logits = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(logits, dim=-1)
            entropy = torch.distributions.Categorical(probs=probs).entropy()
            mean_entropy = entropy.mean(dim=1).cpu().numpy()  # move to CPU early
            confidences.extend(0.5 / (1.0 + mean_entropy))

            for input_ids, gen_ids in zip(inputs["input_ids"], outputs.sequences):
                response = self.tokenizer.decode(gen_ids[len(input_ids):], skip_special_tokens=True).strip()
                all_responses.append(response)

            del inputs, outputs, logits, probs, entropy
            torch.cuda.synchronize()

        results = []
        for response, conf in zip(all_responses, confidences):

            if self.task == "sentiment":
                sentiment = self._extract_sentiment(response)
                conf = conf.item() if isinstance(conf, torch.Tensor) else float(conf)
                if sentiment == "positive":
                    preds = [0.5 - conf, 0.5 + conf]
                elif sentiment == "negative":
                    preds = [0.5 + conf, 0.5 - conf]
                else:
                    preds = [0.5, 0.5]
                results.append(preds)

            elif self.task == "entailment":
                entailment = self._extract_entailment(response)

                conf = conf.item() if isinstance(conf, torch.Tensor) else float(conf)
                if entailment == "yes":
                    preds = [0.5 - conf, 0.5 + conf]
                elif entailment == "no":
                    preds = [0.5 + conf, 0.5 - conf]
                else:
                    preds = [0.5, 0.5]
                results.append(preds)

        return torch.tensor(results, dtype=torch.float32)