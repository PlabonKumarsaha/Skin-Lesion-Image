"""HAM10000 splits, datasets and transforms."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/processed/ham10000_400"
SPLITS = ROOT / "splits"
CLASSES_7 = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASSES_4 = ["nv", "mel", "bkl", "bcc"]
SITES = ["rosendahl", "vidir_modern"]


def load_task(task):
    """Return (classes, {set_name: DataFrame}).

    task = "random"                    -> train/val/test from splits/random.csv
    task = "site_<A>" (A in SITES)     -> train/val on site A; test sets test_<A> (in-site) and test_<B> (cross-site)
    """
    if task == "random":
        df = pd.read_csv(SPLITS / "random.csv")
        return CLASSES_7, {s: df[df.split == s] for s in ("train", "val", "test")}
    site = task.removeprefix("site_")
    assert site in SITES, task
    df = pd.read_csv(SPLITS / "site_shift.csv")
    sets = {s: df[(df.dataset == site) & (df.split == s)] for s in ("train", "val")}
    sets |= {f"test_{s}": df[(df.dataset == s) & (df.split == "test")] for s in SITES}
    return CLASSES_4, sets


class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, df, classes, transform):
        self.ids = df.image_id.tolist()
        self.labels = [classes.index(d) for d in df.dx]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(IMAGES / f"{self.ids[i]}.jpg") as im:
            return self.transform(im.convert("RGB")), self.labels[i]


def make_transforms(size, mean, std):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02), *norm])
    test = T.Compose([T.Resize(size), T.CenterCrop(size), *norm])
    return train, test
