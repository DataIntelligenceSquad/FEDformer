export CUDA_VISIBLE_DEVICES=0

#cd ..

for model in FEDformer
do

# ETT m1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path LABEL_BTC_0_with_BTC_new.csv \
  --task_id Final \
  --model $model \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --lradj 'type1' \
  --target LABEL \
  --dropout 0.05 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0001 \

done
