# 1.1) Train Masker
python3 rationalizers train --config configs/masker/snli_sparsemap_30p.yaml --seed 0
# >>> experiments/masker_imdb_sparsemap_30p/versionjkexuek0/checkpoints/epoch=1.ckpt

# 1.2) Train Editor
python3 rationalizers train --config configs/editor/snli_sparsemap_30p.yaml --seed 0
