import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difflib import SequenceMatcher
import re

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

def detect_contradiction(premise, hypothesis):
    tokenizer, model = get_roberta_mnli()
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs[0].item()

def calculate_metrics(original_text, evidence_ranges, perturbed_text):

    cleanText = original_text.lower()                           # lowercase
    cleanText = re.sub(r"[^\w\s]", "", cleanText)               # remove punctuation

    cleanCf = re.sub(r"</?new>", "", perturbed_text)            # remove <new> tags if extraction failed
    cleanCf = cleanCf.lower()                             
    cleanCf = re.sub(r"[^\w\s]", "", cleanCf)            

    matcher = SequenceMatcher(None, cleanText.split(), cleanCf.split())
    distance = 1 - matcher.ratio()


    orig_tokens = original_text.split()
    pert_tokens = perturbed_text.split()

    # use dataset-provided ranges directly
    evidence_tokens = {i for start, end in evidence_ranges for i in range(start, end)}
        
    # Precompute adjacency mask for O(1) evidence support check
    is_evidence_adjacent = [False] * (len(orig_tokens) + 1)
    for idx in evidence_tokens:
        is_evidence_adjacent[idx] = True
        if idx + 1 < len(is_evidence_adjacent):
            is_evidence_adjacent[idx+1] = True

    matcher = SequenceMatcher(None, orig_tokens, pert_tokens)

    TP = FP = 0
    touched_evidence = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for idx in range(i1, i2):
                if idx in evidence_tokens:
                    TP += 1
                    touched_evidence.add(idx)
                else:
                    FP += 1
        elif tag == "insert":
            if is_evidence_adjacent[i1]:
                # count as TP once for touching evidence
                touched_evidence.add(i1)
                TP += 1
            else:
                FP += (j2 - j1)  # each inserted token = FP

    FN = len(evidence_tokens - touched_evidence)
    total_tokens = len(orig_tokens)
    TN = total_tokens - TP - FP - FN

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / total_tokens if total_tokens > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "distance": distance,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }
    
def evaluate_cf(text, evidence_ranges, cf):
   
    matcher_metrics = calculate_metrics(text, evidence_ranges, cf)
    semantic = semantic_similarity(cf, text)
    contradiction = detect_contradiction(cf, text)

    return {
        "distance": matcher_metrics['distance'],
        "evidence_accuracy": matcher_metrics['accuracy'],
        "evidence_precision": matcher_metrics['precision'],
        "evidence_recall": matcher_metrics['recall'],
        "evidence_f1": matcher_metrics['f1'],
        "similarity_metrics": semantic,
        "contradiction": contradiction
    }
