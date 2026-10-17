"""Prepare the external test sets (never used for training): BCN20000 labelled split and PAD-UFES-20.

Outputs:
  splits/external.csv  dataset, key, img (cached path), dx (HAM 7-class name or NaN), mel, malignant, group, age, sex, site, fitzpatrick
  data/processed/{bcn,pad}_400/<key>.jpg   long side 400 px

Label mapping to HAM10000 classes (classes with no HAM equivalent are kept with dx = NaN, but still count for melanoma
and malignant metrics):
  BCN20000:   NV->nv MEL->mel BCC->bcc BKL->bkl AK->akiec DF->df VASC->vasc SCC->(none, malignant)
  PAD-UFES-20: NEV->nv MEL->mel BCC->bcc SEK->bkl ACK->akiec SCC->(none, malignant)

Usage: .venv/bin/python -I scripts/prepare_external.py
"""
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
BCN_MAP = {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "AK": "akiec", "DF": "df", "VASC": "vasc", "SCC": None}
PAD_MAP = {"NEV": "nv", "MEL": "mel", "BCC": "bcc", "SEK": "bkl", "ACK": "akiec", "SCC": None}
MALIGNANT = {"MEL", "BCC", "SCC"}


def resize(job):
    zpath, member, dst = job
    if dst.exists():
        return True
    with zipfile.ZipFile(zpath) as z, z.open(member) as f, Image.open(f) as im:
        im = im.convert("RGB")
        im.thumbnail((400, 400), Image.BICUBIC)
        im.save(dst, quality=95)
    return True


def members(zips):
    """{basename: (zip, member)} for all image files in the zips (refuses absolute or parent-relative paths)."""
    out = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                assert not m.startswith("/") and ".." not in Path(m).parts, m
                if m.lower().endswith((".jpg", ".jpeg", ".png")):
                    out[Path(m).name] = (zp, m)
    return out


def main():
    bcn = pd.read_csv(RAW / "bcn20000/bcn_20k_train.csv")
    bz = members([RAW / "bcn20000/BCN_20k_train.zip"])
    bcn = bcn[bcn.bcn_filename.isin(bz)]
    b = pd.DataFrame({"dataset": "bcn20000", "key": bcn.bcn_filename.str.replace(".jpg", "", regex=False),
                      "dx": bcn.diagnosis.map(BCN_MAP), "orig_dx": bcn.diagnosis,
                      "group": bcn.lesion_id, "age": bcn.age_approx, "sex": bcn.sex, "site": bcn.anatom_site_general,
                      "fitzpatrick": None})
    b["img"] = [f"bcn_400/{k}.jpg" for k in b.key]
    jobs = [(*bz[f], OUT / "bcn_400" / f) for f in bcn.bcn_filename]

    pad = pd.read_csv(RAW / "pad_ufes20/metadata.csv")
    pz = members(sorted((RAW / "pad_ufes20").glob("imgs_part_*.zip")))
    pad = pad[pad.img_id.isin(pz)]
    p = pd.DataFrame({"dataset": "pad_ufes20", "key": pad.img_id.str.rsplit(".", n=1).str[0],
                      "dx": pad.diagnostic.map(PAD_MAP), "orig_dx": pad.diagnostic,
                      "group": pad.patient_id, "age": pad.age, "sex": pad.gender, "site": pad.region,
                      "fitzpatrick": pad.fitspatrick})
    p["img"] = [f"pad_400/{k}.jpg" for k in p.key]
    jobs += [(*pz[f], OUT / "pad_400" / f"{Path(f).stem}.jpg") for f in pad.img_id]

    ext = pd.concat([b, p], ignore_index=True)
    ext["mel"] = ext.orig_dx.eq("MEL")
    ext["malignant"] = ext.orig_dx.isin(MALIGNANT)
    for d in ("bcn_400", "pad_400"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(resize, jobs, chunksize=32))
    ext.to_csv(ROOT / "splits/external.csv", index=False)
    print(ext.groupby(["dataset", "orig_dx"]).size().to_string())
    print(ext.groupby("dataset")[["mel", "malignant"]].sum())


if __name__ == "__main__":
    main()
