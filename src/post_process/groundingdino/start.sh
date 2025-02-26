#!/bin/bash

pip install -r requirements_dino.txt

if [ ! -d "./out" ]; then
  # 如果输出文件夹不存在，创建它
  echo "Directory './out' does not exist. Creating it..."
  mkdir ./out
fi

python dino_ablation_study.py --anno_path $1 --img_dir $2
      --csv_path $3  --data_type $4 --exp_type $5  --out ./out



















