# 1.1) Train Masker
python3 rationalizers train --config configs/editor/imdb_sparsemap_50p.yaml --seed 0
# >>> experiments/masker_imdb_sparsemap_50p/versionNone/checkpoints/epoch=5.ckpt

# 1.2) Train Editor
python3 rationalizers train --config configs/masker/snli_sparsemap_30p.yaml --seed 0
python3 rationalizers train --config configs/editor/imdb_sparsemap_50p.yaml --seed 0

# >>> experiments/masker_snli_sparsemap_30p/versionNone/checkpoints/epoch=10.ckpt