export CUDA_VISIBLE_DEVICES=0

#cd ..

for model in FEDformer
do

# ETT m1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path New_Final_nasdaq_gold_btc.csv \
  --task_id Final \
  --model $model \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --lradj 'type3' \
  --target LABEL \
  --dropout 0.05 \
  --train_epochs 10 \
  --patience 8 \
  --learning_rate 0.0001 \

done
