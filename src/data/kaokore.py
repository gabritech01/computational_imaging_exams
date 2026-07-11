"""KaoKore dataset loader, built around the official labels.csv split ('set' column:
train/dev/test) instead of a custom split, see NOTE_ORALE.md for the reasoning."""
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor

GENDER_MAP = {0: "male", 1: "female"}
STATUS_MAP = {0: "noble", 1: "warrior", 2: "incarnation", 3: "commoner"}


class KaokoreDataset(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be one of train, dev, test")
        self.root = root
        self.image_size = image_size

        images_dir = os.path.join(root, "images_256")
        with open(os.path.join(root, "labels.csv"), newline="") as f:
            rows = list(csv.DictReader(f))

        self.entries = [
            row for row in rows
            if row["set"] == split and os.path.isfile(os.path.join(images_dir, row["image"]))
        ]
        self.images_dir = images_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        img = Image.open(os.path.join(self.images_dir, row["image"])).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = resize(img, [self.image_size, self.image_size])
        x = to_tensor(img)  # float32 in [0, 1], shape (3, H, W)
        label = {"gender": int(row["gender"]), "status": int(row["status"])}
        return x, label
