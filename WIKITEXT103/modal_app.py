import modal
import subprocess


app = modal.App("wikitext103-muon")

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
        "/root/WIKITEXT103",
        ignore=[".git", "__pycache__", "*.pyc"],
    )
)

wkt103_volume = modal.Volume.from_name("wt103-data")
wkt103_weights = modal.Volume.from_name("wt103-weights")

@app.function(
    image=image,
    gpu="A100",
    cpu=12,
    memory=32 * 1024,
    volumes={
        "/data/wt103": wkt103_volume,
        "/data/weights": wkt103_weights,
    },
    timeout=60 * 60 * 12,
)
def run_test():

    subprocess.run(
        [
            "bash",
            "/root/WIKITEXT103/script_test.sh",
        ],
        cwd="/root/WIKITEXT103",
        check=True,
    )

@app.local_entrypoint()
def main():
    run_test.remote()