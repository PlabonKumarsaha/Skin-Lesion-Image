"""Build lesion-level splits for HAM10000.

Outputs (in splits/):
  random.csv       7-class, all images, stratified 70/15/15 by lesion diagnosis.
  site_shift.csv   E1: rosendahl vs vidir_modern, classes nv/mel/bkl/bcc, each site split 70/15/15.
                   Train on site A's train, then test on A's test (in-site) and B's test (cross-site),
                   so both directions share the same two test sets.

Every image of a lesion lands in the same split. Lesions seen at more than one site are excluded from site_shift.

Usage: .venv/bin/python -I scripts/make_splits.py data/raw/ham10000/HAM10000_metadata.csv
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SITE_SHIFT_SITES = ["rosendahl", "vidir_modern"]
SITE_SHIFT_CLASSES = ["nv", "mel", "bkl", "bcc"]
OUT = Path("splits")


def split_lesions(df, seed=SEED):
    """Return {lesion_id: split} with a 70/15/15 split stratified by lesion diagnosis."""
    lesions = df.groupby("lesion_id")["dx"].first()
    train, rest = train_test_split(lesions.index, test_size=0.30, stratify=lesions, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=lesions[rest], random_state=seed)
    return {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val"), **dict.fromkeys(test, "test")}


def check(df, name):
    leaked = df.groupby("lesion_id")["split"].nunique().gt(1).sum()
    assert leaked == 0, f"{name}: {leaked} lesions in more than one split"
    print(f"\n{name}: {len(df)} images, {df.lesion_id.nunique()} lesions, leakage check passed")


def main(meta_path):
    meta = pd.read_csv(meta_path)
    assert meta.groupby("lesion_id")["dx"].nunique().max() == 1, "a lesion has more than one diagnosis"
    OUT.mkdir(exist_ok=True)
    cols = ["image_id", "lesion_id", "dx", "dataset", "age", "sex", "localization", "split"]

    rand = meta.assign(split=meta.lesion_id.map(split_lesions(meta)))
    check(rand, "random")
    print(pd.crosstab(rand.dx, rand.split).loc[:, ["train", "val", "test"]])
    rand[cols].to_csv(OUT / "random.csv", index=False)

    multi_site = meta.groupby("lesion_id")["dataset"].nunique().loc[lambda s: s > 1].index
    site = meta[meta.dataset.isin(SITE_SHIFT_SITES) & meta.dx.isin(SITE_SHIFT_CLASSES)
                & ~meta.lesion_id.isin(multi_site)]
    parts = [g.assign(split=g.lesion_id.map(split_lesions(g))) for _, g in site.groupby("dataset")]
    site = pd.concat(parts)
    check(site, "site_shift")
    print(f"excluded {len(multi_site)} multi-site lesions")
    print(pd.crosstab([site.dataset, site.split], site.dx)[SITE_SHIFT_CLASSES])
    site[cols].to_csv(OUT / "site_shift.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
