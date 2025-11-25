# LongT5 Integration for Custom Datasets

This document describes the modifications made to use `google/long-t5-tglobal-base` instead of `t5-small` to handle longer sequences (4096 tokens vs 512 tokens).

## Problem

The original CREST implementation uses T5-small with a hard 512-token limit. 85% of movie reviews in the custom dataset exceed this limit, requiring truncation that was unacceptable for the use case.

## Solution

Modified the codebase to use `google/long-t5-tglobal-base`:
- **Model size**: 250M parameters (vs 60M for t5-small)
- **Max sequence length**: 4096 tokens (vs 512 for t5-small)
- **Memory footprint**: ~3-4GB per model (vs ~2GB for t5-small)
- **Total VRAM needed**: ~12GB for full CREST system (3 models)

## Modified Files

### 1. Core Script Changes

#### `/crest/scripts/get_edits.py`
- **Line 225**: Changed `max_seq_len: 512` → `max_seq_len: 4096`
- **Line 245**: Changed `max_length: 512` → `max_length: 4096`

### 2. Config Files Updated

#### `/crest/configs/masker/imdb_sparsemap_50p.yaml`
- Changed all model references: `t5-small` → `google/long-t5-tglobal-base`
- Updated `max_seq_len: 512` → `max_seq_len: 4096`

#### `/crest/configs/editor/imdb_sparsemap_50p.yaml`
- Changed all model references: `t5-small` → `google/long-t5-tglobal-base`
- Updated `max_seq_len: 512` → `max_seq_len: 4096`
- Updated `max_length: 512` → `max_length: 4096` (in cf_generate_kwargs)

### 3. New Config Files Created

#### Masker Configs (Train Factual Rationalizer)
- `/crest/configs/masker/my_movies_longt5.yaml`
  - Uses `dm: 'contrast_imdb_cf'` for training data
  - `batch_size: 2` (reduced for LongT5 memory requirements)
  - `max_seq_len: 4096`
  - All models: `google/long-t5-tglobal-base`
  
- `/crest/configs/masker/my_esnli_longt5.yaml`
  - Uses `dm: 'snli'` for training data
  - `batch_size: 4`
  - `max_seq_len: 4096`
  - All models: `google/long-t5-tglobal-base`

#### Editor Configs (Train Counterfactual Generator)
- `/crest/configs/editor/my_movies_longt5.yaml`
  - Uses `dm: 'contrast_imdb_cf'` for training data
  - `batch_size: 2`
  - `max_seq_len: 4096`, `max_length: 4096`
  - All models: `google/long-t5-tglobal-base`
  - `factual_ckpt`: Points to trained masker checkpoint (UPDATE after training)
  
- `/crest/configs/editor/my_esnli_longt5.yaml`
  - Uses `dm: 'snli'` for training data
  - `batch_size: 4`
  - `max_seq_len: 4096`, `max_length: 4096`
  - All models: `google/long-t5-tglobal-base`
  - `factual_ckpt`: Points to trained masker checkpoint (UPDATE after training)
  - `ignore_neutrals: True`

### 4. Shell Scripts Updated

#### `/crest/scripts/get_edits_my_movies.sh`
- Updated checkpoint paths to use `my_movies_longt5` experiments
- Reduced `BATCH_SIZE=2` for LongT5
- Points to correct masker and editor checkpoints

#### `/crest/scripts/get_edits_my_esnli.sh`
- Updated checkpoint paths to use `my_esnli_longt5` experiments
- Reduced `BATCH_SIZE=4` for LongT5
- Includes `--ignore-neutrals` flag
- Points to correct masker and editor checkpoints

## Training Pipeline

### Step 1: Train Masker (Factual Rationalizer)

#### For Movies Dataset:
```bash
cd /home/user/Desktop/diploma/crest
python train.py --config configs/masker/my_movies_longt5.yaml
```

**Expected output**: Checkpoint saved to `experiments/masker_my_movies_longt5/version_0/checkpoints/best.ckpt`

**Training details**:
- Uses `contrast_imdb_cf` dataset for training
- Trains on sentiment classification task
- Learns to identify important tokens (rationales)
- **Estimated time**: Days to weeks depending on GPU

#### For e-SNLI Dataset:
```bash
cd /home/user/Desktop/diploma/crest
python train.py --config configs/masker/my_esnli_longt5.yaml
```

**Expected output**: Checkpoint saved to `experiments/masker_my_esnli_longt5/version_0/checkpoints/best.ckpt`

**Training details**:
- Uses `snli` dataset for training
- Trains on NLI task (entailment/contradiction)
- Learns to identify important tokens in hypothesis
- **Estimated time**: Days to weeks depending on GPU

### Step 2: Update Editor Config

After masker training completes, update the `factual_ckpt` path in the editor configs:

#### For Movies:
Edit `/crest/configs/editor/my_movies_longt5.yaml` line 10:
```yaml
factual_ckpt: experiments/masker_my_movies_longt5/version_0/checkpoints/best.ckpt
```

#### For e-SNLI:
Edit `/crest/configs/editor/my_esnli_longt5.yaml` line 10:
```yaml
factual_ckpt: experiments/masker_my_esnli_longt5/version_0/checkpoints/best.ckpt
```

### Step 3: Train Editor (Counterfactual Generator)

#### For Movies Dataset:
```bash
cd /home/user/Desktop/diploma/crest
python train.py --config configs/editor/my_movies_longt5.yaml
```

**Expected output**: Checkpoint saved to `experiments/editor_my_movies_longt5/version_0/checkpoints/best.ckpt`

**Training details**:
- Uses trained masker checkpoint
- Trains on contrast_imdb_cf dataset
- Learns to generate counterfactual edits
- **Estimated time**: Days to weeks depending on GPU

#### For e-SNLI Dataset:
```bash
cd /home/user/Desktop/diploma/crest
python train.py --config configs/editor/my_esnli_longt5.yaml
```

**Expected output**: Checkpoint saved to `experiments/editor_my_esnli_longt5/version_0/checkpoints/best.ckpt`

**Training details**:
- Uses trained masker checkpoint
- Trains on snli dataset
- Learns to generate counterfactual edits for NLI
- **Estimated time**: Days to weeks depending on GPU

### Step 4: Extract Counterfactuals from Custom Datasets

After both masker and editor are trained, update shell scripts with correct checkpoint paths and run:

#### For Movies Dataset:
```bash
cd /home/user/Desktop/diploma/crest/scripts

# Update get_edits_my_movies.sh with correct version numbers, then:
bash get_edits_my_movies.sh
```

**Output**: Counterfactual edits saved to `experiments/editor_my_movies_longt5/version_0/test_edits.txt`

#### For e-SNLI Dataset:
```bash
cd /home/user/Desktop/diploma/crest/scripts

# Update get_edits_my_esnli.sh with correct version numbers, then:
bash get_edits_my_esnli.sh
```

**Output**: Counterfactual edits saved to `experiments/editor_my_esnli_longt5/version_0/test_edits.txt`

## Key Architecture Notes

### CREST System Components
1. **Factual Rationalizer (Masker)**: Identifies important tokens in input
2. **Editor**: Generates counterfactual edits based on rationales
3. Both use the same transformer backbone (LongT5 in this case)

### Memory Considerations
- Full CREST system loads 3 LongT5 models simultaneously:
  - Generator (masker)
  - Predictor (masker)
  - CF Generator (editor)
- **Total VRAM**: ~12GB (fits in 16GB GPU with batch_size=2-4)
- Batch sizes reduced compared to t5-small to fit in memory

### Why Complete Retraining is Necessary
- Existing t5-small checkpoints cannot be used with LongT5
- Weight dimensions differ (t5-small: 512-dim, long-t5: 768-dim)
- Architecture differs (LongT5 uses TGlobal attention mechanism)
- Loading t5-small weights into LongT5 would fail with dimension mismatch

## Data Module Notes

### Training vs Inference
- **Masker training**: Uses `contrast_imdb_cf` or `snli` (has both positive and negative examples)
- **Editor training**: Uses same training datasets
- **Inference**: Uses `my_movies` or `my_esnli` (test-only custom datasets)

### Dataset Characteristics
- **my_movies**: 199 test samples, 85% exceed 512 tokens
- **my_esnli**: 6598 test samples, binary labels (0/1), no neutrals

## Verification Steps

After all changes, verify setup:

```bash
cd /home/user/Desktop/diploma/crest/scripts

# Check data modules load correctly
python -c "
from rationalizers.data_modules import available_data_modules
print('my_movies' in available_data_modules)
print('my_esnli' in available_data_modules)
"

# Test LongT5 model loading
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/long-t5-tglobal-base')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Model params: {model.num_parameters() / 1e6:.1f}M')
"
```

## Summary of Changes

| Component | Original | Modified |
|-----------|----------|----------|
| Model | t5-small (60M params) | google/long-t5-tglobal-base (250M params) |
| Max tokens | 512 | 4096 |
| VRAM per model | ~2GB | ~3-4GB |
| Total VRAM | ~6GB | ~12GB |
| Batch size (movies) | 8 | 2 |
| Batch size (e-SNLI) | 32 | 4 |
| Training time | Hours to days | Days to weeks |

## Important Notes

1. **No backward compatibility**: Cannot use existing t5-small checkpoints
2. **Longer training time**: LongT5 is 4x larger, sequences 8x longer
3. **Higher memory usage**: Need 16GB GPU for full pipeline
4. **No truncation needed**: All movie reviews now fit in 4096 tokens
5. **Test-only inference**: Custom datasets only have test splits

## References

- LongT5 Paper: https://arxiv.org/abs/2112.07916
- HuggingFace Model: https://huggingface.co/google/long-t5-tglobal-base
- CREST Paper: https://arxiv.org/abs/2305.17075
