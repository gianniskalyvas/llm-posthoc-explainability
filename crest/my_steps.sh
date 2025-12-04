python3 rationalizers train --config configs/masker/e_movies_sparsemap_50p.yaml --seed 0

python3 rationalizers train --config configs/editor/e_movies_sparsemap_50p.yaml --seed 0




python3 scripts/get_rationales.py \
    --ckpt-name "sparsemap_50p" \
    --ckpt-path "experiments/masker_e_movies_sparsemap_50p/versionNone/checkpoints/epoch=4.ckpt" \
    --dm-name "e_movies" \
    --dm-dataloader "test" \
    --sparsemap-budget 50


python3 scripts/get_edits.py \
  --ckpt-name "sparsemap_50p" \
  --ckpt-path "experiments/editor_e_movies_sparsemap_50p/versionNone/checkpoints/epoch=19.ckpt" \
  --dm-name "e_movies" \
  --dm-dataloader "test" \
  --num-beams 15




python3 rationalizers train --config configs/masker/my_esnli_sparsemap_30p.yaml --seed 0

python3 rationalizers train --config configs/editor/my_esnli_sparsemap_30p.yaml --seed 0



python3 scripts/get_rationales.py \
    --ckpt-name "sparsemap_30p" \
    --ckpt-path "experiments/editor_my_esnli_sparsemap_30p_baseT5/versionNone/checkpoints/epoch=19.ckpt" \
    --dm-name "my_esnli" \
    --dm-dataloader "test" \
    --sparsemap-budget 30

# 2.0) Before proceeding, we need to extract counterfactuals for all training examples (this may take a while)
python3 scripts/get_edits.py \
  --ckpt-name "sparsemap_30p" \
  --ckpt-path "experiments/editor_my_esnli_sparsemap_30p_baseT5/versionNone/checkpoints/epoch=19.ckpt" \
  --dm-name "my_esnli" \
  --dm-dataloader "test" \
  --num-beams 15
