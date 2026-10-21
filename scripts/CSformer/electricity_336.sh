#!/usr/bin/env bash
# Modified for CSformer: CSformer entry, complete paper horizons and robust execution.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p "logs/LongForecasting/electricity"

seq_len=336
model_name=CSformer

root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=Electricity_lr_top3
data_name=custom
top_k=3
random_seed=2024
for pred_len in 96 192 336 720
do
    "${PYTHON:-python3}" -u run.py \
      --random_seed $random_seed \
      --val_protocol legacy \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "${model_id_name}_${seq_len}_${pred_len}" \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1\
      --n_heads 1\
      --d_model 128 \
      --d_ff 256\
      --dropout 0.4\
      --fc_dropout 0\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --top_k $top_k \
      --period_list 336 168 84 42 21  \
      --des 'Exp' \
      --n_f_layers 1\
      --train_epochs 100\
      --patience 10\
      --itr 1 --batch_size 64 --learning_rate 0.005 >logs/LongForecasting/electricity/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done



