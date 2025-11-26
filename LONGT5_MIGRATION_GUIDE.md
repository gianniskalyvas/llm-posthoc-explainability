# LongT5 Integration for Custom Datasets

This document describes the modifications made to use `google/long-t5-tglobal-base` instead of `t5-small` to handle longer sequences (4096 tokens vs 512 tokens).

## Problem

The original CREST implementation uses T5-small with a hard 512-token limit. 85% of movie reviews in the custom dataset exceed this limit, requiring truncation that was unacceptable for the use case.

## Solution

Modified the codebase to use `google/long-t5-tglobal-base`:
- **Model size**: 250M parameters (vs 60M for t5-small)
- **Max sequence length**: 6144 tokens (vs 512 for t5-small)
- **Memory footprint**: ~3-4GB per model (vs ~2GB for t5-small)
- **Total VRAM needed**: ~12GB for full CREST system (3 models)

## Modified Files

### 1. Core Script Changes

#### `/crest/scripts/get_edits.py`
- **Line 226**: Changed `max_seq_len: 512` → `max_seq_len: 6144`
- **Line 246**: Changed `max_length: 512` → `max_length: 6144`

### 2. Config Files Updated

#### `/crest/configs/masker/imdb_sparsemap_50p.yaml`
- Changed all model references: `t5-small` → `google/long-t5-tglobal-base`
- Updated `max_seq_len: 512` → `max_seq_len: 6144`

#### `/crest/configs/editor/imdb_sparsemap_50p.yaml`
- Changed all model references: `t5-small` → `google/long-t5-tglobal-base`
- Updated `max_seq_len: 512` → `max_seq_len: 6144`
- Updated `max_length: 512` → `max_length: 6144` (in cf_generate_kwargs)

### 3. New Config Files Created

#### Masker Configs (Train Factual Rationalizer)
- `/crest/configs/masker/my_movies_longt5.yaml`
  - Uses `dm: 'contrast_imdb_cf'` for training data
  - `batch_size: 2` (reduced for LongT5 memory requirements)
  - `max_seq_len: 6144`
  - All models: `google/long-t5-tglobal-base`
  
- `/crest/configs/masker/my_esnli_baset5.yaml`
  - Uses `dm: 'snli'` for training data
  - `batch_size: 8` (standard t5-base batch size)
  - `max_seq_len: 512` (sufficient for NLI premise-hypothesis pairs)
  - All models: `t5-base` (not LongT5 - sequences are short enough)

#### Editor Configs (Train Counterfactual Generator)
- `/crest/configs/editor/my_movies_longt5.yaml`
  - Uses `dm: 'contrast_imdb_cf'` for training data
  - `batch_size: 2`
  - `max_seq_len: 6144`, `max_length: 6144`
  - All models: `google/long-t5-tglobal-base`
  - `factual_ckpt`: Points to trained masker checkpoint (UPDATE after training)
  
- `/crest/configs/editor/my_esnli_baset5.yaml`
  - Uses `dm: 'snli'` for training data
  - `batch_size: 8`
  - `max_seq_len: 512`, `max_length: 512`
  - All models: `t5-base` (not LongT5 - sequences are short enough)
  - `factual_ckpt`: Points to trained masker checkpoint (UPDATE after training)
  - `ignore_neutrals: True`
  - `cf_task_name: 'nli_no_neutrals'`

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
cd crest
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
cd crest
python train.py --config configs/masker/my_esnli_baset5.yaml
```

**Expected output**: Checkpoint saved to `experiments/masker_my_esnli_t5base/version_0/checkpoints/best.ckpt`

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
Edit `/crest/configs/editor/my_esnli_baset5.yaml` line 10:
```yaml
factual_ckpt: experiments/masker_my_esnli_t5base/version_0/checkpoints/best.ckpt
```

### Step 3: Train Editor (Counterfactual Generator)

#### For Movies Dataset:
```bash
cd crest
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
cd crest
python train.py --config configs/editor/my_esnli_baset5.yaml
```

**Expected output**: Checkpoint saved to `experiments/editor_my_esnli_t5base/version_0/checkpoints/best.ckpt`

**Training details**:
- Uses trained masker checkpoint
- Trains on snli dataset
- Learns to generate counterfactual edits for NLI
- **Estimated time**: Days to weeks depending on GPU

### Step 4: Extract Counterfactuals from Custom Datasets

After both masker and editor are trained, update shell scripts with correct checkpoint paths and run:

#### For Movies Dataset:
```bash
cd crest/scripts

# Update get_edits_my_movies.sh with correct version numbers, then:
bash get_edits_my_movies.sh
```

**Output**: Counterfactual edits saved to `experiments/editor_my_movies_longt5/version_0/test_edits.txt`

#### For e-SNLI Dataset:
```bash
cd crest/scripts

# Update get_edits_my_esnli.sh with correct version numbers, then:
bash get_edits_my_esnli.sh
```

**Output**: Counterfactual edits saved to `experiments/editor_my_esnli_t5base/version_0/test_edits.txt`

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
- **my_movies**: 199 test samples, 85% exceed 512 tokens, longest is ~3040 tokens
- **my_esnli**: 6598 test samples, binary labels (0/1), no neutrals

## Verification Steps

After all changes, verify setup:

```bash
cd crest/scripts

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
| Max tokens | 512 | 6144 |
| VRAM per model | ~2GB | ~3-4GB |
| Total VRAM | ~6GB | ~12GB |
| Batch size (movies) | 8 | 2 |
| Batch size (e-SNLI) | 32 | 4 |
| Training time | Hours to days | Days to weeks |

## Important Notes

1. **No backward compatibility**: Cannot use existing t5-small checkpoints
2. **Longer training time**: LongT5 is 4x larger, sequences 12x longer (for movies only)
3. **Higher memory usage**: Need 16GB GPU for movies pipeline, 8GB sufficient for e-SNLI
4. **No truncation needed**: All movie reviews now fit in 6144 tokens (2x longest review)
5. **Test-only inference**: Custom datasets only have test splits
6. **e-SNLI uses standard t5-base**: NLI sequences are short enough (512 tokens), no need for LongT5

## References

- LongT5 Paper: https://arxiv.org/abs/2112.07916
- HuggingFace Model: https://huggingface.co/google/long-t5-tglobal-base
- CREST Paper: https://arxiv.org/abs/2305.17075
