import modal
import os
import shutil

app = modal.App("fix-wt103-path")
volume = modal.Volume.from_name("wt103-data")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/data/wt103": volume},
)
def fix_path():
    src = "/data/wt103/wt103"
    dst = "/data/wt103"

    for name in ["train.txt", "valid.txt", "test.txt"]:
        src_file = os.path.join(src, name)
        dst_file = os.path.join(dst, name)

        print(f"Moving {src_file} -> {dst_file}")
        shutil.move(src_file, dst_file)

    # Remove empty nested directory
    os.rmdir(src)

    volume.commit()

    print("\nFinal structure:")
    for root, dirs, files in os.walk("/data/wt103"):
        print(root)
        for f in files:
            print("  ", f)


@app.local_entrypoint()
def main():
    fix_path.remote()