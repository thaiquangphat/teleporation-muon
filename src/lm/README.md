# LM architecture

This folder contains the GPT-2 language-model training code for Muon teleportation experiments.

## Prepare WikiText-103

Local file mode is the default:

```bash
mkdir -p "$HOME/data/wt103"
bash data_wkt103.sh
```

This creates files like:

```text
$HOME/data/wt103/
├── train.txt
├── valid.txt
└── test.txt
```

## Run a single LM experiment

```bash
python src/lm/train.py \
  --data-dir "$HOME/data/wt103" \
  --dataset-source local \
  --max-steps 1000 \
  --warmup-steps 50 \
  --opt muon \
  --lr 0.002 \
  --adam-lr 5e-5 \
  --momentum 0.9 \
  --tele-method csd_l2_nesterov \
  --nesterov \
  --tele-candidates 6 \
  --tele-att 1 \
  --tele-mlp 0 \
  --devices cuda:0 \
  --save-dir ./checkpoints-lm
```

## Optional: Hugging Face dataset source

```bash
python src/lm/train.py \
  --dataset-source huggingface \
  --model-name gpt2 \
  --max-steps 1000 \
  --opt muon
```

## Teleport methods supported

- `none`
- `naive`
- `csd_l2`
- `csd_l2_nesterov`
- `csd_l2_nesterov_frobenius`
- `csd_l2_nesterov_post_orth`
- `csd_l2_nesterov_post_orth_frobenius`

For Nesterov-based methods, add `--nesterov` so the optimizer update matches the scoring direction.

Tuned hyperparameters are displayed in `../../scripts_lm.sh`.
