"""Download the KaoKore v1.3 dataset (images + official labels/split) into data/kaokore.

Images are hosted externally and referenced by dataset_v1.3/urls.txt in the official
repo (https://github.com/rois-codh/kaokore); this script mirrors their download logic
but writes directly into this project's data/ folder.
"""
import argparse
import io
import os
from multiprocessing import Pool
from urllib.request import urlopen

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_BASE = "https://raw.githubusercontent.com/rois-codh/kaokore/master/dataset_v1.3"


def _download_one(args):
    index, url, out_dir = args
    save_path = os.path.join(out_dir, f"{index:08d}.jpg")
    if os.path.isfile(save_path):
        return
    try:
        data = urlopen(url, timeout=20).read()
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            arr = np.asarray(img)
            arr = np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr
            Image.fromarray(arr, mode="RGB").save(save_path)
        else:
            with open(save_path, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"failed image {index}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/kaokore")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    images_dir = os.path.join(args.out, "images_256")
    os.makedirs(images_dir, exist_ok=True)

    for fname in ["labels.csv", "urls.txt"]:
        dest = os.path.join(args.out, fname)
        if not os.path.isfile(dest):
            with urlopen(f"{RAW_BASE}/{fname}") as r, open(dest, "wb") as f:
                f.write(r.read())

    with open(os.path.join(args.out, "urls.txt")) as f:
        urls = [line.strip() for line in f]
    jobs = [(i, url, images_dir) for i, url in enumerate(urls) if url]

    with Pool(args.threads) as pool:
        list(tqdm(pool.imap_unordered(_download_one, jobs), total=len(jobs)))


if __name__ == "__main__":
    main()
