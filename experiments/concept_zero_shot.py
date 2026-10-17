"""E2 non-generative baseline: zero-shot 7-point-criterion scores from image-text models on Derm7pt test.

score(criterion) = cos(image, "present" prompts) - cos(image, "absent" prompts), from cached image features
(results/features/<model>/derm7pt__test.npy, written by foundation_eval.py). AUROC vs expert labels is directly
comparable with a VLM's yes/no citation, whose AUROC equals its balanced accuracy.

Usage: .venv/bin/python -I experiments/concept_zero_shot.py --model monet
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

TEMPLATES = ["a dermoscopic image showing {}.", "dermoscopy of a skin lesion with {}.", "{} seen on dermoscopy."]
PROMPTS = {  # criterion -> (present, absent) phrases
    "atypical_pigment_network": ("an atypical, irregular pigment network", "a regular, typical pigment network"),
    "blue_whitish_veil_present": ("a blue-whitish veil", "no blue-whitish veil"),
    "atypical_vascular": ("atypical dotted or irregular linear vessels", "no visible blood vessels"),
    "irregular_streaks": ("irregular streaks and pseudopods at the periphery", "no streaks"),
    "irregular_pigmentation": ("irregular blotches of pigmentation", "uniform, regular pigmentation"),
    "irregular_dots_globules": ("irregular dots and globules", "regular, evenly distributed dots and globules"),
    "regression_present": ("regression structures with white scar-like areas and blue peppering", "no regression structures"),
}


def text_vec(fm, phrase):
    v = fm.encode_text([t.format(phrase) for t in TEMPLATES]).mean(0)
    return torch.nn.functional.normalize(v, dim=-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    d7 = pd.read_csv(ROOT / "splits/derm7pt.csv")
    d7 = d7[(d7.split == "test") & d7.ham_dx.notna()]  # same rows and order as foundation_eval's derm7pt/test
    X = np.load(ROOT / "results/features" / a.model / "derm7pt__test.npy")
    assert len(X) == len(d7), (len(X), len(d7))
    fm = Foundation(a.model, torch.device("cpu"))
    out = {}
    for crit, (pos, neg) in PROMPTS.items():
        s = X @ text_vec(fm, pos) - X @ text_vec(fm, neg)
        out[crit] = roc_auc_score(d7[crit].values.astype(bool), s)
    out["macro_auroc"] = float(np.mean(list(out.values())))
    res = ROOT / "results/concept_zero_shot.json"
    allres = json.load(open(res)) if res.exists() else {}
    allres[a.model] = out
    json.dump(allres, open(res, "w"), indent=2)
    print(a.model, {k: round(v, 3) for k, v in out.items()})


if __name__ == "__main__":
    main()
