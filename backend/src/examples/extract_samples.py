"""
One-time script: extracts 10 cancer + 10 no-cancer 96x96 PNG patches
from the PatchCamelyon TFDS test split.

Run from repo root:
    python backend/src/examples/extract_samples.py
"""
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

N_PER_CLASS = 10
OUT_DIR = Path(__file__).parent
CANCER_DIR = OUT_DIR / "cancer"
NO_CANCER_DIR = OUT_DIR / "no_cancer"


def save_png(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, format="PNG")


def main():
    CANCER_DIR.mkdir(parents=True, exist_ok=True)
    NO_CANCER_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PatchCamelyon test split...")
    ds = tfds.load("patch_camelyon", split="test", shuffle_files=True)
    cancer_n = no_cancer_n = 0

    for sample in ds:
        img = sample["image"].numpy()
        label = int(sample["label"].numpy())

        if label == 1 and cancer_n < N_PER_CLASS:
            cancer_n += 1
            save_png(img, CANCER_DIR / f"cancer_{cancer_n:02d}.png")
            print(f"  cancer_{cancer_n:02d}.png saved")
        elif label == 0 and no_cancer_n < N_PER_CLASS:
            no_cancer_n += 1
            save_png(img, NO_CANCER_DIR / f"no_cancer_{no_cancer_n:02d}.png")
            print(f"  no_cancer_{no_cancer_n:02d}.png saved")

        if cancer_n >= N_PER_CLASS and no_cancer_n >= N_PER_CLASS:
            break

    print(f"\nDone: {cancer_n} cancer + {no_cancer_n} no-cancer images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
