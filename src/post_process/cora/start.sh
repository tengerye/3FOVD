#!/bin/bash

pip install -r requirements_cora.txt

if [ ! -d "./out" ]; then
  # 如果输出文件夹不存在，创建它
  echo "Directory './out' does not exist. Creating it..."
  mkdir ./out
fi

python cora_ablation_study.py --anno_path $1 --img_path $2  --csv_path $3 --data_type $4 --exp_type $5 --out ./out --backbone clip_RN50 --resume $6 --region_prompt_path $7 --text_len 15 --ovd --save_every_epoch 50 --dim_feedforward 1024 --use_nms --num_queries 1000 --anchor_pre_matching --remove_misclassified --condition_on_text --enc_layers 3 --text_dim 1024 --condition_bottleneck 128 --split_class_p 0.2 --model_ema --model_ema_decay 0.99996 --save_best --label_version RN50base --disable_init --target_class_factor 1.0 --batch_size 1 --epochs 35 --lr_drop 35