import modal


# ============================================================
# Modal App
# ============================================================

app = modal.App("wt103-training")


# ============================================================
# Persistent Volume
# ============================================================

volume = modal.Volume.from_name(
    "wt103-data",
    create_if_missing=True,
)


# ============================================================
# Kaggle Secret
# ============================================================

kaggle_secret = modal.Secret.from_name("kaggle")


# ============================================================
# Container Image
# ============================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install("kaggle")
)


# ============================================================
# Download and Prepare WT103
# ============================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[kaggle_secret],
    timeout=60 * 60,
)
def download_wt103():

    import os
    import json
    import shutil
    import subprocess
    from pathlib import Path

    # --------------------------------------------------------
    # 1. Configure Kaggle credentials
    # --------------------------------------------------------

    kaggle_dir = "/root/.config/kaggle"
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json = os.path.join(
        kaggle_dir,
        "kaggle.json",
    )

    with open(kaggle_json, "w") as f:
        json.dump(
            {
                "username": os.environ["username"],
                "key": os.environ["key"],
            },
            f,
        )

    os.chmod(kaggle_json, 0o600)

    print("Kaggle credentials configured.")


    # --------------------------------------------------------
    # 2. Directories
    # --------------------------------------------------------

    download_dir = "/tmp/wt103-download"
    extract_dir = "/tmp/wt103-extracted"
    output_dir = "/data/wt103"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    # --------------------------------------------------------
    # 3. Check whether dataset already exists
    # --------------------------------------------------------

    required_files = [
        "train.txt",
        "valid.txt",
        "test.txt",
    ]

    if all(
        os.path.exists(
            os.path.join(output_dir, f)
        )
        for f in required_files
    ):

        print("\nWT103 already exists in the Volume.")

        for filename in required_files:

            path = os.path.join(
                output_dir,
                filename,
            )

            print(
                f"{filename}: "
                f"{os.path.getsize(path) / 1024**2:.2f} MB"
            )

        return


    # --------------------------------------------------------
    # 4. Download WT103 from Kaggle
    # --------------------------------------------------------

    print("\nDownloading Wikitext-103...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "vadimkurochkin/wikitext-103",
            "-p",
            download_dir,
        ],
        check=True,
    )


    # --------------------------------------------------------
    # 5. Locate ZIP file
    # --------------------------------------------------------

    zip_file = os.path.join(
        download_dir,
        "wikitext-103.zip",
    )

    if not os.path.exists(zip_file):

        # Fallback in case Kaggle uses a different filename
        zip_files = list(
            Path(download_dir).glob("*.zip")
        )

        if not zip_files:
            raise FileNotFoundError(
                "Could not find downloaded Wikitext-103 ZIP file."
            )

        zip_file = str(zip_files[0])

    print(f"\nZIP file: {zip_file}")


    # --------------------------------------------------------
    # 6. Extract ZIP
    # --------------------------------------------------------

    print("\nExtracting dataset...")

    subprocess.run(
        [
            "unzip",
            "-q",
            zip_file,
            "-d",
            extract_dir,
        ],
        check=True,
    )


    # --------------------------------------------------------
    # 7. Find WT103 files
    # --------------------------------------------------------

    print("\nSearching for WT103 files...")

    try:

        train_src = next(
            Path(extract_dir).rglob(
                "wiki.train.tokens"
            )
        )

        valid_src = next(
            Path(extract_dir).rglob(
                "wiki.valid.tokens"
            )
        )

        test_src = next(
            Path(extract_dir).rglob(
                "wiki.test.tokens"
            )
        )

    except StopIteration:

        print("\nCould not find expected WT103 files.")

        print("\nExtracted files:")

        for path in Path(extract_dir).rglob("*"):

            if path.is_file():
                print(path)

        raise


    print("\nFound:")

    print("  train:", train_src)
    print("  valid:", valid_src)
    print("  test :", test_src)


    # --------------------------------------------------------
    # 8. Copy files into expected format
    # --------------------------------------------------------

    print("\nCopying dataset files...")

    shutil.copy2(
        train_src,
        os.path.join(
            output_dir,
            "train.txt",
        ),
    )

    shutil.copy2(
        valid_src,
        os.path.join(
            output_dir,
            "valid.txt",
        ),
    )

    shutil.copy2(
        test_src,
        os.path.join(
            output_dir,
            "test.txt",
        ),
    )


    # --------------------------------------------------------
    # 9. Show final dataset
    # --------------------------------------------------------

    print("\nDataset prepared:")

    for filename in required_files:

        path = os.path.join(
            output_dir,
            filename,
        )

        print(
            f"  {filename}: "
            f"{os.path.getsize(path) / 1024**2:.2f} MB"
        )


    # --------------------------------------------------------
    # 10. Persist to Modal Volume
    # --------------------------------------------------------

    print("\nCommitting Volume...")

    volume.commit()

    print("\n========================================")
    print("WT103 successfully prepared!")
    print("Location: /data/wt103")
    print("========================================")


# ============================================================
# Local Entrypoint
# ============================================================

@app.local_entrypoint()
def main():

    download_wt103.remote()