# BASELINE RoPE
python train.py --cuda --data /data/wt103 --work_dir /data/weights/wkt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim sgd --lr 0.75 --warmup_step 20 --max_step 100 --eval-interval 10 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 \
    --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 1 --tele-high 1.0 --tele-low 1.0 --tele-cons 1  --tele-layer all 

# TELEPORT RoPE
python train.py --cuda --data /data/wt103 --work_dir /data/weights/wkt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim sgd --lr 0.75 --warmup_step 20 --max_step 100 --eval-interval 10 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 \
    --n-teleport 8 --tele-epoch 2 --tele-batch 128 --tele-start 500 --tele-limit 1000 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16  --tele-layer all 

# BASELINE RoPE Muon
python train.py --cuda --data /data/wt103 --work_dir /data/weights/wkt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim muon --lr 0.75 --warmup_step 20 --max_step 100 --eval-interval 10 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 \
    --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 1 --tele-high 1.0 --tele-low 1.0 --tele-cons 1  --tele-layer all 

# TELEPORT RoPE Muon
python train.py --cuda --data /data/wt103 --work_dir /data/weights/wkt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim muon --lr 0.75 --warmup_step 20 --max_step 100 --eval-interval 10 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 \
    --n-teleport 8 --tele-epoch 2 --tele-batch 128 --tele-start 500 --tele-limit 1000 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16  --tele-layer all 

