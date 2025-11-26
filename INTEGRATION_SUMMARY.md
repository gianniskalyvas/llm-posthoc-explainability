# Summary: Custom Dataset Integration for CREST Counterfactual Generation

## What Was Done

I've integrated your custom datasets from `eraserbenchmark-master/` into the CREST framework so you can extract counterfactual edits using the existing `get_edits.py` script.

## Files Created/Modified

### New Data Modules
1. **`crest/rationalizers/data_modules/my_movies.py`**
   - Data module for your Movies dataset
   - Loads from: `/eraserbenchmark-master/movies_dataset_builder/my_dataset/`
   - Binary sentiment classification (0=negative, 1=positive)
   - Uses **only test split** for counterfactual generation

2. **`crest/rationalizers/data_modules/my_esnli.py`**
   - Data module for your e-SNLI dataset
   - Loads from: `/eraserbenchmark-master/esnli_dataset_builder/my_dataset/`
   - Binary NLI classification (0=not_entailment, 1=entailment)
   - Uses **only test split** for counterfactual generation

### Updated Files
3. **`crest/rationalizers/data_modules/__init__.py`**
   - Registered both new data modules
   - Available as `my_movies` and `my_esnli`

### Helper Scripts
4. **`crest/scripts/get_edits_my_movies.sh`**
   - Example script to extract counterfactuals from Movies dataset

5. **`crest/scripts/get_edits_my_esnli.sh`**
   - Example script to extract counterfactuals from e-SNLI dataset

### Documentation
6. **`crest/rationalizers/data_modules/README_CUSTOM_DATASETS.md`**
   - Complete documentation on usage, parameters, and troubleshooting

## Key Features

### ✅ Test Split Only
Both data modules are configured to use **only the test split** for all operations. When you run the script with any dataloader option (`train`, `val`, or `test`), it will always process your test data.

### ✅ No Neutral Filtering
Your e-SNLI dataset already has binary labels (0 and 1), so no neutral filtering is needed or applied.

### ✅ Drop-in Replacement
The data modules follow the same interface as existing CREST modules, so they work seamlessly with `get_edits.py`.

## How to Use

### Basic Usage

```bash
cd crest/scripts

# For Movies dataset
python get_edits.py \
    --ckpt-name "my_movies_cf" \
    --ckpt-path "/path/to/editor.ckpt" \
    --dm-name "my_movies" \
    --dm-dataloader "test" \
    --batch-size 16 \
    --num-beams 15

# For e-SNLI dataset
python get_edits.py \
    --ckpt-name "my_esnli_cf" \
    --ckpt-path "/path/to/editor.ckpt" \
    --dm-name "my_esnli" \
    --dm-dataloader "test" \
    --batch-size 16 \
    --num-beams 15
```

### Using Shell Scripts

1. Edit the shell script to add your checkpoint paths:
   ```bash
   nano get_edits_my_movies.sh  # or get_edits_my_esnli.sh
   ```

2. Update these lines:
   ```bash
   CKPT_PATH="/path/to/your/editor/checkpoint.ckpt"
   CKPT_PATH_FACTUAL="/path/to/your/factual/rationalizer.ckpt"
   ```

3. Run the script:
   ```bash
   bash get_edits_my_movies.sh
   ```

## Output Location

Counterfactual edits will be saved to:
```
crest/data/edits/{dataset}_{split}_{mode}_{beams}_{name}.tsv
```

Example:
- `data/edits/my_movies_test_beam_15_my_movies_cf.tsv`
- `data/edits/my_esnli_test_beam_15_my_esnli_cf.tsv`

## What You Need

Before running:
1. ✅ Trained editor model checkpoint
2. ✅ (Optional) Factual rationalizer checkpoint
3. ✅ Your datasets are already in place at:
   - `/eraserbenchmark-master/movies_dataset_builder/my_dataset/`
   - `/eraserbenchmark-master/esnli_dataset_builder/my_dataset/`

## Next Steps

1. **Update the shell scripts** with your actual checkpoint paths
2. **Run the scripts** to generate counterfactuals
3. **Analyze the output** TSV files containing:
   - Original texts and labels
   - Generated counterfactual edits
   - Model predictions
   - Rationale selections

## Questions?

See the detailed documentation in:
`crest/rationalizers/data_modules/README_CUSTOM_DATASETS.md`
