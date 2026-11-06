#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${IMAGENET_DATA_DIR:-$SCRIPT_DIR/data/imagenet}}"
mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

TRAIN_ARCHIVE="ILSVRC2012_img_train.tar"
VAL_ARCHIVE="ILSVRC2012_img_val.tar"
VALPREP_URL="https://raw.githubusercontent.com/PatrickHua/EasyImageNet/main/valprep.sh"

cd "$DATA_DIR"
echo "Preparing ImageNet-1K at $DATA_DIR"

echo "Downloading or resuming $TRAIN_ARCHIVE"
wget -c --no-check-certificate "https://image-net.org/data/ILSVRC/2012/$TRAIN_ARCHIVE"
echo "Downloading or resuming $VAL_ARCHIVE"
wget -c --no-check-certificate "https://image-net.org/data/ILSVRC/2012/$VAL_ARCHIVE"

mkdir -p train
train_class_count="$(find train -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$train_class_count" -ne 1000 ]]; then
    echo "Extracting ImageNet training classes"
    tar -xf "$TRAIN_ARCHIVE" -C train
fi

while IFS= read -r -d '' class_archive; do
    class_dir="${class_archive%.tar}"
    mkdir -p "$class_dir"
    tar -xf "$class_archive" -C "$class_dir"
    rm -f "$class_archive"
done < <(find train -mindepth 1 -maxdepth 1 -type f -name "*.tar" -print0)

train_class_count="$(find train -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$train_class_count" -ne 1000 ]]; then
    echo "Expected 1000 training class folders; found $train_class_count." >&2
    exit 1
fi

mkdir -p val
val_class_count="$(find val -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$val_class_count" -ne 1000 ]]; then
    echo "Extracting and organizing ImageNet validation classes"
    tar -xf "$VAL_ARCHIVE" -C val
    if [[ ! -s val/valprep.sh ]]; then
        wget -O val/valprep.sh "$VALPREP_URL"
    fi
    (cd val && sh valprep.sh)
fi

val_class_count="$(find val -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$val_class_count" -ne 1000 ]]; then
    echo "Expected 1000 validation class folders; found $val_class_count." >&2
    exit 1
fi

train_classes="$(find train -mindepth 1 -maxdepth 1 -type d -print | sed 's#^.*/##' | sort)"
val_classes="$(find val -mindepth 1 -maxdepth 1 -type d -print | sed 's#^.*/##' | sort)"
if [[ "$train_classes" != "$val_classes" ]]; then
    echo "Training and validation class folders do not match." >&2
    exit 1
fi

echo "ImageNet-1K is ready for the MUONTELEPORTATION ImageFolder loader."
