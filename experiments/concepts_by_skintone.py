"""E4 part 2: zero-shot detection of SkinCon clinical concepts on DDI, by Fitzpatrick group.

For each image-text model: score(concept) = cos(image, "with <concept>") - cos(image, "without <concept>"), using the
cached DDI image features from foundation_eval.py; AUROC vs dermatologist SkinCon labels per group (I-II, III-IV, V-VI).
Only the 8 concepts with >= 30 positives on DDI are used. Aggregates only.

Usage: .venv/bin/python -I experiments/concepts_by_skintone.py --model dermlip_vitb16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinbench.foundation import Foundation  # noqa: E402

CONCEPTS = {  # SkinCon column -> (present phrase, absent phrase)
    "Papule": ("a papule, a small raised bump", "a flat lesion with no papules"),
    "Brown(Hyperpigmentation)": ("brown hyperpigmentation", "no hyperpigmentation"),
    "Erythema": ("erythema, redness of the skin", "no redness of the skin"),
    "Plaque": ("a plaque, a raised flat-topped area", "no plaque"),
    "Scale": ("scale, flaky skin", "smooth skin without scale"),
    "Dome-shaped": ("a dome-shaped lesion", "a flat lesion"),
    "Crust": ("crust, dried exudate on the surface", "no crust"),
    "Nodule": ("a nodule, a large firm raised lesion", "no nodule"),
}
TEMPLATES = ["a clinical photo of a skin lesion with {}.", "a photograph showing {}.", "skin with {}."]
GROUPS = ["I-II", "III-IV", "V-VI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")  # row order of the cached features
    X = np.load(ROOT / "results/features" / a.model / "ddi__all.npy")
    assert len(X) == len(ddi)
    con = pd.read_csv(ROOT / "splits/ddi_concepts.csv")[["DDI_ID", *CONCEPTS]]
    d = ddi.reset_index().merge(con, on="DDI_ID")  # 636 usable images; 'index' = feature row
    fm = Foundation(a.model, torch.device("cpu"))
    vec = lambda ph: torch.nn.functional.normalize(fm.encode_text([t.format(ph) for t in TEMPLATES]).mean(0), dim=-1).numpy()
    out = {}
    for c, (pos, neg) in CONCEPTS.items():
        s = X[d["index"].values] @ (vec(pos) - vec(neg))
        out[c] = {"all": roc_auc_score(d[c], s)}
        for g in GROUPS:
            m = (d.fst_group == g).values
            out[c][g] = roc_auc_score(d[c][m], s[m]) if 0 < d[c][m].sum() < m.sum() else None
    out["macro"] = {k: float(np.mean([v[k] for v in out.values() if v.get(k) is not None])) for k in ["all", *GROUPS]}
    res = ROOT / "results/concepts_by_skintone.json"
    allres = json.load(open(res)) if res.exists() else {}
    allres[a.model] = out
    json.dump(allres, open(res, "w"), indent=2)
    print(a.model, "macro", {k: round(v, 3) for k, v in out["macro"].items()},
          "| Erythema", {k: (round(v, 3) if v else None) for k, v in out["Erythema"].items()})


if __name__ == "__main__":
    main()
