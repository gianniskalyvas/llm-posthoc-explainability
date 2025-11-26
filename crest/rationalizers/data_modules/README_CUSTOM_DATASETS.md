# Custom Dataset Modules for CREST

This document explains how to use the custom data modules for extracting counterfactual edits from your datasets.

## Available Data Modules

### 1. MyMoviesDataModule (`my_movies`)
- **Location**: `eraserbenchmark-master/movies_dataset_builder/my_dataset/`
- **Task**: Sentiment classification (binary: negative/positive)
- **Features**: Single text field
- **Usage**: `--dm-name my_movies`

### 2. MyESNLIDataModule (`my_esnli`)
- **Location**: `eraserbenchmark-master/esnli_dataset_builder/my_dataset/`
- **Task**: Natural Language Inference (binary: not_entailment/entailment)
- **Features**: Two text fields (sentence1, sentence2)
- **Usage**: `--dm-name my_esnli`

## Important Notes

### Test Split Only
Both data modules are configured to **use only the test split** for generating counterfactuals. This means:
- When you run `get_edits.py` with `--dm-dataloader test`, it will process your test data
- The train and validation splits are also mapped to the test data internally
- This ensures counterfactuals are generated only from your test instances

### Binary Classification
- **Movies**: Labels are 0 (negative) and 1 (positive)
- **e-SNLI**: Labels are 0 (not_entailment) and 1 (entailment)
- No neutral instances exist in these datasets

## Usage Examples

### Extracting Counterfactuals from Movies Dataset

```bash
cd crest/scripts

python get_edits.py \
    --ckpt-name "my_movies_editor" \
    --ckpt-path "/path/to/your/editor/checkpoint.ckpt" \
    --ckpt-path-factual "/path/to/your/factual/checkpoint.ckpt" \
    --dm-name "my_movies" \
    --dm-dataloader "test" \
    --batch-size 16 \
    --num-beams 15
```

### Extracting Counterfactuals from e-SNLI Dataset

```bash
cd crest/scripts

python get_edits.py \
    --ckpt-name "my_esnli_editor" \
    --ckpt-path "/path/to/your/editor/checkpoint.ckpt" \
    --ckpt-path-factual "/path/to/your/factual/checkpoint.ckpt" \
    --dm-name "my_esnli" \
    --dm-dataloader "test" \
    --batch-size 16 \
    --num-beams 15
```

### Using the Provided Shell Scripts

Convenience scripts have been created:

```bash
# For Movies dataset
cd crest/scripts
# Edit the script to add your checkpoint paths
nano get_edits_my_movies.sh
# Then run it
bash get_edits_my_movies.sh

# For e-SNLI dataset
cd crest/scripts
# Edit the script to add your checkpoint paths
nano get_edits_my_esnli.sh
# Then run it
bash get_edits_my_esnli.sh
```

## Output

The counterfactual edits will be saved to:
```
crest/data/edits/{dm_name}_{dm_dataloader}_{sample_mode}_{num_beams}_{ckpt_name}.tsv
```

For example:
- `data/edits/my_movies_test_beam_15_my_movies_editor.tsv`
- `data/edits/my_esnli_test_beam_15_my_esnli_editor.tsv`

The TSV file contains:
- `orig_texts`: Original input texts
- `orig_labels`: Original labels
- `orig_predictions`: Model predictions on originals
- `orig_z`: Rationales for originals
- `edits_texts`: Counterfactual edits
- `edits_labels`: Labels for edits (should be flipped)
- `edits_predictions`: Model predictions on edits
- `edits_z_pre`: Rationales before editing
- `edits_z_pos`: Rationales after editing

## Requirements

Before running, ensure you have:
1. Trained editor model checkpoint
2. (Optional) Trained factual rationalizer checkpoint
3. Sufficient GPU memory for beam search
4. CREST environment properly set up

## Parameters

### Common Parameters
- `--ckpt-name`: Name for saving output files
- `--ckpt-path`: Path to editor checkpoint (required)
- `--ckpt-path-factual`: Path to factual rationalizer (optional)
- `--dm-name`: Data module name (`my_movies` or `my_esnli`)
- `--dm-dataloader`: Which split to use (use `test`)
- `--batch-size`: Batch size for generation (default: 16)
- `--num-beams`: Number of beams for beam search (default: 15)

### Optional Parameters
- `--do-sample`: Use sampling instead of beam search
- `--sparsemap-budget`: Budget for sparsemap (if using sparsemap)
- `--max-dataset-size`: Limit number of samples (for testing)

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (e.g., to 8 or 4)
- Reduce `--num-beams` (e.g., to 10 or 5)
- Use `--max-dataset-size` to process fewer samples

### Missing Checkpoint
- Ensure the checkpoint path is absolute and correct
- Check that the checkpoint file exists and is not corrupted

### Wrong Labels
- Verify your CSV files have the correct format
- Check that labels are 0 and 1 (binary)
- Ensure no header row issues in CSV files
