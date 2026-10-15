"""Build analysis-ready tables and resized image caches for Derm7pt and DDI.

Outputs:
  splits/derm7pt.csv  one row per case: diagnosis, mel flag, HAM-style dx, the 7 concepts (+ binary flags), official split, image paths
  splits/ddi.csv      one row per image: disease, malignant, Fitzpatrick group, mel flag, image path
  data/processed/derm7pt_400/{derm,clinic}/<case_num>.jpg and data/processed/ddi_400/<DDI_ID>.jpg  (long side 400 px)

Usage: .venv/bin/python -I scripts/prepare_concept_sets.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D7 = ROOT / "data/raw/derm7pt/release_v0"
DDI = ROOT / "data/raw/ddi"
OUT = ROOT / "data/processed"
LONG_SIDE = 400

# Derm7pt diagnosis -> nearest HAM10000 class (None = no HAM equivalent)
D7_TO_HAM = {"basal cell carcinoma": "bcc", "seborrheic keratosis": "bkl", "lentigo": "bkl",
             "dermatofibroma": "df", "vascular lesion": "vasc", "miscellaneous": None, "melanosis": None}
CONCEPTS = ["pigment_network", "streaks", "pigmentation", "regression_structures",
            "dots_and_globules", "blue_whitish_veil", "vascular_structures"]
# 7-point checklist "major/minor" criterion presence (Argenziano), as binary flags
CHECKLIST = {
    "atypical_pigment_network": ("pigment_network", {"atypical"}),
    "blue_whitish_veil_present": ("blue_whitish_veil", {"present"}),
    "atypical_vascular": ("vascular_structures", {"dotted", "linear irregular", "within regression"}),
    "irregular_streaks": ("streaks", {"irregular"}),
    "irregular_pigmentation": ("pigmentation", {"diffuse irregular", "localized irregular"}),
    "irregular_dots_globules": ("dots_and_globules", {"irregular"}),
    "regression_present": ("regression_structures", {"blue areas", "white areas", "combinations"}),
}


def resize(job):
    src, dst = job
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((LONG_SIDE, LONG_SIDE), Image.BICUBIC)
        im.save(dst, quality=95)


def derm7pt():
    m = pd.read_csv(D7 / "meta/meta.csv")
    split = {}
    for name, f in [("train", "train_indexes.csv"), ("val", "valid_indexes.csv"), ("test", "test_indexes.csv")]:
        for i in pd.read_csv(D7 / "meta" / f)["indexes"]:
            split[i] = name
    m["split"] = [split.get(i) for i in range(len(m))]
    m["mel"] = m.diagnosis.str.contains("melanoma")
    m["ham_dx"] = [("mel" if mel else D7_TO_HAM.get(d, "nv")) for d, mel in zip(m.diagnosis, m.mel)]
    for flag, (col, vals) in CHECKLIST.items():
        m[flag] = m[col].isin(vals)
    jobs = []
    for kind in ("derm", "clinic"):
        (OUT / f"derm7pt_400/{kind}").mkdir(parents=True, exist_ok=True)
        m[f"{kind}_img"] = [f"derm7pt_400/{kind}/{c}.jpg" for c in m.case_num]
        jobs += [(D7 / "images" / p, OUT / f"derm7pt_400/{kind}/{c}.jpg") for p, c in zip(m[kind], m.case_num)]
    cols = (["case_num", "diagnosis", "mel", "ham_dx", "seven_point_score", *CONCEPTS, *CHECKLIST,
             "level_of_diagnostic_difficulty", "sex", "location", "split", "derm_img", "clinic_img"])
    m[cols].to_csv(ROOT / "splits/derm7pt.csv", index=False)
    return m, jobs


def ddi():
    d = pd.read_csv(DDI / "ddi_metadata.csv", index_col=0)
    d["fst_group"] = d.skin_tone.map({12: "I-II", 34: "III-IV", 56: "V-VI"})
    d["mel"] = d.disease.str.contains("melanoma")
    d["img"] = [f"ddi_400/{i}.jpg" for i in d.DDI_ID]
    (OUT / "ddi_400").mkdir(parents=True, exist_ok=True)
    d[["DDI_ID", "disease", "malignant", "mel", "skin_tone", "fst_group", "img"]].to_csv(ROOT / "splits/ddi.csv", index=False)
    return d, [(DDI / f, OUT / f"ddi_400/{i}.jpg") for f, i in zip(d.DDI_file, d.DDI_ID)]


def main():
    m, j1 = derm7pt()
    d, j2 = ddi()
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(resize, j1 + j2, chunksize=32))
    print(f"derm7pt: {len(m)} cases, split {m.split.value_counts().to_dict()}, mel {int(m.mel.sum())}")
    print("derm7pt ham_dx:", m.ham_dx.value_counts().to_dict())
    print("derm7pt checklist positives:", {k: int(m[k].sum()) for k in CHECKLIST})
    print(f"ddi: {len(d)} images, fst {d.fst_group.value_counts().to_dict()}, malignant {int(d.malignant.sum())}")
    print("cached images:", sum(1 for _ in (OUT / 'derm7pt_400').rglob('*.jpg')), "+", sum(1 for _ in (OUT / 'ddi_400').glob('*.jpg')))


if __name__ == "__main__":
    main()
