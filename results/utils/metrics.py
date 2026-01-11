import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast, T5Tokenizer
from difflib import SequenceMatcher
import re
import editdistance
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load mpnet
_mpnet = None
def get_mpnet():
    global _mpnet
    if _mpnet is None:
        _mpnet = SentenceTransformer('all-mpnet-base-v2').to(device)
    return _mpnet

def semantic_similarity(text1, text2):
    model = get_mpnet()
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return cosine_similarity(
        embeddings[0].cpu().numpy().reshape(1, -1),
        embeddings[1].cpu().numpy().reshape(1, -1)
    )[0][0]

# load MNLI
_roberta_mnli = None
_tokenizer = None
def get_roberta_mnli():
    global _roberta_mnli, _tokenizer
    if _roberta_mnli is None:
        _tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        _roberta_mnli = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    return _tokenizer, _roberta_mnli

# load GPT-2 large for fluency
_gpt2_model = None
_gpt2_tokenizer = None
def get_gpt2_large():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
        _gpt2_tokenizer.pad_token = _gpt2_tokenizer.eos_token
    return _gpt2_tokenizer, _gpt2_model

# load T5-small tokenizer for truncation
_t5_tokenizer = None
def get_t5_small_tokenizer():
    global _t5_tokenizer
    if _t5_tokenizer is None:
        _t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return _t5_tokenizer

def calculate_fluency(text):
    """Calculate fluency using GPT-2 large perplexity"""
    tokenizer, model = get_gpt2_large()
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
        
    return perplexity

def calculate_self_bleu(text):
    """Calculate self-BLEU for diversity measurement"""
    sentences = text.split('.')
    sentences = [s.strip().split() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    smoothie = SmoothingFunction().method4
    total_score = 0
    count = 0
    
    for i in range(len(sentences)):
        hypothesis = sentences[i]
        references = [sentences[j] for j in range(len(sentences)) if i != j]
        
        if hypothesis and references:
            score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0.0

def detect_contradiction(premise, hypothesis):
    tokenizer, model = get_roberta_mnli()
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs[0].item()


def calculate_closeness(original_text, perturbed_text):
    cleanText = original_text.lower()                           # lowercase
    cleanText = re.sub(r"[^\w\s]", "", cleanText)               # remove punctuation

    cleanCf = re.sub(r"</?new>", "", perturbed_text)            # remove <new> tags if extraction failed
    cleanCf = cleanCf.lower()                             
    cleanCf = re.sub(r"[^\w\s]", "", cleanCf)            

    # Calculate Levenshtein distance (word level)    # Calculate edit distance (character level)
    char_edit_distance = editdistance.eval(cleanText, cleanCf)
    
    # Calculate normalized edit distance
    max_len = max(len(cleanText), len(cleanCf))
    normalized_edit_distance = char_edit_distance / max_len if max_len > 0 else 0
    closeness = 1 - normalized_edit_distance

    return closeness



def evidence_evaluation(original_text, important_list, perturbed_text):

    orig_tokens = original_text.split()[:len(important_list)]
    pert_tokens = perturbed_text.split()[:len(important_list)]  

    matcher = SequenceMatcher(None, orig_tokens, pert_tokens)

    TP = FP = 0
    touched_evidence = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for idx in range(i1, i2):
                if idx < len(important_list) and important_list[idx] == 1:
                    TP += 1
                    touched_evidence.add(idx)
                else:
                    FP += 1
        elif tag == "insert":
            # Check if insert is adjacent to important tokens (before or at position)
            is_adjacent = False
            if i1 > 0 and i1-1 < len(important_list) and important_list[i1-1] == 1:
                is_adjacent = True
            if i1 < len(important_list) and important_list[i1] == 1:
                is_adjacent = True
                
            if is_adjacent:
                TP += 1
                if i1 < len(important_list):
                    touched_evidence.add(i1)
            else:
                FP += (j2 - j1)

    FN = sum(important_list) - len(touched_evidence)
    total_tokens = len(orig_tokens)
    TN = total_tokens - TP - FP - FN

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / total_tokens if total_tokens > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }



def evaluate_cf(text, evidence_ranges, crest_rationales, cf):
   
    closeness = calculate_closeness(text, cf)
    semantic = semantic_similarity(cf, text)
    contradiction = detect_contradiction(cf, text)
    #fluency = calculate_fluency(cf)
    #diversity = calculate_self_bleu(cf)

    evidence_list = [0] * len(text.split())
    for start, end in evidence_ranges:
        for i in range(start, end):
            evidence_list[i] = 1


    evidence_metrics = evidence_evaluation(text, evidence_list, cf)
    crest_metrics = evidence_evaluation(text, crest_rationales, cf)

    return {
        "closeness": closeness,
        "similarity_metrics": semantic,
        "contradiction": contradiction,
        #"fluency": fluency,
        #"diversity": diversity,
        "evidence_accuracy": evidence_metrics['accuracy'],
        "evidence_precision": evidence_metrics['precision'],
        "evidence_recall": evidence_metrics['recall'],
        "evidence_f1": evidence_metrics['f1'],
        "crest_accuracy": crest_metrics['accuracy'],
        "crest_precision": crest_metrics['precision'],
        "crest_recall": crest_metrics['recall'],
        "crest_f1": crest_metrics['f1']
    }
