#!/usr/bin/env bash
# Modified for CSformer: CSformer entry, complete paper horizons and robust execution.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p "logs/LongForecasting/ettm2"

seq_len=336
model_name=CSformer

root_path_name=./dataset/ETT/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
top_k=2

random_seed=2024
for pred_len in  96 192 336 720
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
      --enc_in 7 \
      --e_layers 1\
      --n_heads 1\
      --d_model 32\
      --d_ff  128\
      --top_k $top_k \
      --period_list 336 168 84 42 21\
      --dropout 0.9\
      --fc_dropout 0\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 1 --batch_size 256 --learning_rate 0.001 >logs/LongForecasting/ettm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$top_k.log 
done