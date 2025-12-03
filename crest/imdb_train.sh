# 1.1) Train Masker
python3 rationalizers train --config configs/masker/my_movies_sparsemap_50p_longT5.yaml --seed 0
# >>> experiments/masker_imdb_sparsemap_30p/versionjkexuek0/checkpoints/epoch=1.ckpt

# 1.2) Train Editor
python3 rationalizers train --config configs/editor/my_movies_sparsemap_50p_longT5.yaml --seed 0