#!/usr/bin/env sh
addtime() {
  while IFS= read -r line; do
    echo "$(date +"%T-%m-%d") $line"
  done
}


#th eval.lua -model '/media/share33/User/henrylu/guru_t7/undo/model_id_combined_singleLSTM_th3_161113_150000.t7' \
#-coco_json '/home/iis/Downloads/aspecttalk/script/prepro_combined/val_list_label.json' \
#-num_images 300 -image_folder '/home/iis/Downloads/aspecttalk/vis/val_img'

th eval.lua \
-dec_model './model/FusionNet.t7' \
-num_images 20000 -image_folder './vis/val_img' \
-index_json './vis/val_list_label.json' \
| addtime 2>&1 | tee log/log_eval.txt
