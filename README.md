# Muon Teleportation

This repo compares Muon optimization with symmetry-based teleportation in two architectures:

- ViT: image classification with CIFAR-10, CIFAR-100, MNIST, and ImageNet-1K
- LM: GPT-2 language modeling on WikiText-103

For setup and run instructions, use the architecture-specific guides in:

- `src/vit/README.md`
- `src/lm/README.md`

## Quick start

```bash
git clone https://github.com/thaiquangphat/teleporation-muon.git
cd teleporation-muon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp wandb.yaml.example wandb.yaml
```

Edit `wandb.yaml` with your W&B settings if you want experiment tracking.

## Repository layout

```text
.
├── src/
│   ├── vit/                 # ViT training and teleportation
│   └── lm/                  # GPT-2 training and teleportation
├── scripts_vit.sh           # ViT experiment launcher
├── scripts_lm.sh            # LM experiment launcher
├── data_imet.sh             # ImageNet preparation
├── data_wkt103.sh           # WikiText preparation
├── wandb.yaml.example       # Example W&B config
├── wandb_config.sh          # Local W&B config loader
├── requirements.txt
└── README.md
```

See each folder README for the exact launch commands and dataset setup.
