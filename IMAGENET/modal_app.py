import modal
import subprocess


app = modal.App("imagenet-muon")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .add_local_file(
        "requirements.txt",
        "/root/requirements.txt",
        copy=True,
    )
    .run_commands(
        "pip install uv",
        "uv pip install --system -r /root/requirements.txt",
    )
    .add_local_dir(
        ".",
        "/root/IMAGENET",
        ignore=[".git", "__pycache__", "*.pyc"],
    )
)

imagenet_volume = modal.Volume.from_name("imagenet")
weights_volume = modal.Volume.from_name("imagenet-weights")

@app.function(
    image=image,
    gpu="T4",
    cpu=12,
    memory=32 * 1024,
    volumes={
        "/data/imagenet": imagenet_volume,
        "/data/weights": weights_volume,
    },
    timeout=60 * 60 * 12,
)
def run_test():

    subprocess.run(
        [
            "bash",
            "/root/IMAGENET/script_test.sh",
        ],
        cwd="/root/IMAGENET",
        check=True,
    )

    weights_volume.commit()

@app.local_entrypoint()
def main():
    run_test.remote()