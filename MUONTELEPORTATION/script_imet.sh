# ImageNet-1K data should contain train/ and val/ class subdirectories.
IMAGENET_DATA_DIR="${IMAGENET_DATA_DIR:-./data/imagenet}"

# ImageNet-1K No teleport Muon
run_for_seeds --dataset "IMAGENET" --data-dir "$IMAGENET_DATA_DIR" --img-size 224 --patch-size 16 \
  --num_channels 3 --num-classes 1000 --batch-size 256 --epochs 50 \
  --d-model 192 --num-hidden-layer 12 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.02 --momentum 0.9 --weight_decay 1e-5 --save_dir ./checkpoints/imagenet_plain

# ImageNet-1K Naive Teleport Muon
run_for_seeds --dataset "IMAGENET" --data-dir "$IMAGENET_DATA_DIR" --img-size 224 --patch-size 16 \
  --num_channels 3 --num-classes 1000 --batch-size 256 --epochs 50 \
  --d-model 192 --num-hidden-layer 12 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.02 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1,5 \
  --save_dir ./checkpoints/imagenet_naive

# ImageNet-1K Muon-alignment Teleport
run_for_seeds --dataset "IMAGENET" --data-dir "$IMAGENET_DATA_DIR" --img-size 224 --patch-size 16 \
  --num_channels 3 --num-classes 1000 --batch-size 256 --epochs 50 \
  --d-model 192 --num-hidden-layer 12 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.02 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method muon_alignment --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --tele-candidates 6 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1,5 \
  --tele-accept-tol 1e-6 --save_dir ./checkpoints/imagenet_muon_alignment