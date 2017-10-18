#!/usr/bin/env sh
addtime() {
  while IFS= read -r line; do
    echo "$(date +"%T-%m-%d") $line"
  done
}

th train.lua \
-input_h5 ./data/combined_captions_th5.h5 \
-input_json ./data/combined_captions_th5.json \
-id combined_170526 \
-val_images_use 300 \
-max_iters 15000 \
-save_checkpoint_every 10000 \
-input_encoding_size 768 \
-finetune_cnn_after -1  \
-language_eval 0 \
-seq_per_img 5  \
-learning_rate 6.25e-6 \
-checkpoint_path checkpoint/ | addtime 2>&1 | tee log/log_combined_170526.txt
