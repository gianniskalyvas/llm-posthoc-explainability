#!/bin/bash

# Script to extract counterfactual edits from my_movies dataset using LongT5
# First train masker with: configs/masker/my_movies_longt5.yaml (uses contrast_imdb_cf for training)
# Then train editor with: configs/editor/my_movies_longt5.yaml
# Finally run this script with the trained editor checkpoint

# Variables - UPDATE THESE PATHS TO YOUR ACTUAL MODEL CHECKPOINTS
CKPT_NAME="my_movies_longt5_editor"
CKPT_PATH="../experiments/editor_my_movies_longt5/version_0/checkpoints/best.ckpt"  # UPDATE THIS
CKPT_PATH_FACTUAL="../experiments/masker_my_movies_longt5/version_0/checkpoints/best.ckpt"  # UPDATE THIS
DM_NAME="my_movies"
DM_DATALOADER="test"  # only test split for counterfactuals
BATCH_SIZE=2  # reduced for LongT5
NUM_BEAMS=15

# Run the script
python get_edits.py \
    --ckpt-name "$CKPT_NAME" \
    --ckpt-path "$CKPT_PATH" \
    --ckpt-path-factual "$CKPT_PATH_FACTUAL" \
    --dm-name "$DM_NAME" \
    --dm-dataloader "$DM_DATALOADER" \
    --batch-size $BATCH_SIZE \
    --num-beams $NUM_BEAMS

# Alternative: without factual model
# python get_edits.py \
#     --ckpt-name "$CKPT_NAME" \
#     --ckpt-path "$CKPT_PATH" \
#     --dm-name "$DM_NAME" \
#     --dm-dataloader "$DM_DATALOADER" \
#     --batch-size $BATCH_SIZE \
#     --num-beams $NUM_BEAMS

# Alternative: using sampling instead of beam search
# python get_edits.py \
#     --ckpt-name "$CKPT_NAME" \
#     --ckpt-path "$CKPT_PATH" \
#     --dm-name "$DM_NAME" \
#     --dm-dataloader "$DM_DATALOADER" \
#     --batch-size $BATCH_SIZE \
#     --do-sample \
#     --num-beams 1
