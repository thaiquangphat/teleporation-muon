# # No Teleport RoPE
# echo "- Running no teleport rope"
# python train.py --data-path /data/imagenet\
#     --hidden-size 192 --num-hidden-layer 12 --intermediate-size 768 --num-attention-heads 3 --patch-size 16 --batch-size 256 --epochs 1 \
#     --opt sgd --lr 0.05 --warmup-lr 1e-7 --eta-min 1e-5 --position-embedding rope --save-dir /data/weights/imagenet \
#     --seed 1  --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 0 --tele-high 1.0 --tele-low 0.0 --tele-cons 1 

# No Teleport RoPE Muon
echo "- Running no teleport rope muon"
python train.py --data-path /data/imagenet\
    --hidden-size 192 --num-hidden-layer 12 --intermediate-size 768 --num-attention-heads 3 --patch-size 16 --batch-size 256 --epochs 1 \
    --opt muon --lr 0.05 --warmup-lr 1e-7 --eta-min 1e-5 --position-embedding rope --save-dir /data/weights/imagenet \
    --seed 1  --n-teleport 0 --tele-epoch 0 --tele-batch 0 --tele-start 0 --tele-limit 0 --tele-att 0 --tele-mlp 0 --tele-opt 0 --tele-high 1.0 --tele-low 0.0 --tele-cons 1 

# # Teleport RoPE
# python train.py --data-path /data/imagenet\
#     --hidden-size 192 --num-hidden-layer 12 --intermediate-size 768 --num-attention-heads 3 --patch-size 16 --batch-size 256 --epochs 1 \
#     --opt sgd --lr 0.05 --warmup-lr 1e-7 --eta-min 1e-5 --position-embedding rope --save-dir /data/weights/imagenet \
#     --seed 1  --n-teleport 8 --tele-epoch 1 --tele-batch 64 --tele-start 2 --tele-limit 2 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16 

# # Teleport RoPE Muon
# python train.py --data-path /data/imagenet\
#     --hidden-size 192 --num-hidden-layer 12 --intermediate-size 768 --num-attention-heads 3 --patch-size 16 --batch-size 256 --epochs 1 \
#     --opt muon --lr 0.05 --warmup-lr 1e-7 --eta-min 1e-5 --position-embedding rope --save-dir /data/weights/imagenet \
#     --seed 1  --n-teleport 8 --tele-epoch 1 --tele-batch 64 --tele-start 2 --tele-limit 2 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-high 1.2 --tele-low 0.8 --tele-cons 16 
