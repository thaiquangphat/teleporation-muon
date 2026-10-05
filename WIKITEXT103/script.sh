# attn_type == 2: Attention use Sinusoidal
# attn_type == 123 and no_pos: Attention use RoPE
# BASELINE Sinusoidal PE
CUDA_VISIBLE_DEVICES=0  python train.py --cuda --data ~/data/wt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 \
    --optim sgd --lr 0.75 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 --use_wandb\
    --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 1 --tele-high 1.0 --tele-low 1.0 --tele-cons 1
#TELEPORT Sinusoidal PE
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --data ~/data/wt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 \
    --optim sgd --lr 0.75 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111  --use_wandb\
    --n-teleport 8 --tele-epoch 2 --tele-batch 128 --tele-start 500 --tele-limit 1000 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16  --tele-layer all 
# BASELINE RoPE
CUDA_VISIBLE_DEVICES=0  python train.py --cuda --data ~/data/wt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim sgd --lr 0.75 --warmup_step 2000 --max_step 500000 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111 --use_wandb\
    --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 1 --tele-high 1.0 --tele-low 1.0 --tele-cons 1  --tele-layer all 
#TELEPORT RoPE
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --data ~/data/wt103 --dataset wt103 \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --no_pos\
    --optim sgd --lr 0.75 --warmup_step 2000 --max_step 500000 --attn_type 123 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --seed 1111  --use_wandb\
    --n-teleport 8 --tele-epoch 2 --tele-batch 128 --tele-start 500 --tele-limit 1000 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16  --tele-layer all 

