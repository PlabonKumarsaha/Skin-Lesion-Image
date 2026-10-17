"""E2 image-dependence control: compare VLM answers on original vs blurred Derm7pt images (identical prompts).

Blurring (Gaussian, radius 25 px) removes every dermoscopic structure. A model whose explanation comes from the image
should change its cited features and lose discrimination; an unchanged answer means the explanation is not image-driven.
Aggregates only.

Usage: .venv/bin/python -I experiments/blur_control.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / "results/vlm"
FEATURES = ["pigment_network", "streaks", "pigmentation", "regression_structures",
            "dots_and_globules", "blue_whitish_veil", "vascular_structures"]


def load(f):
    d = pd.DataFrame([json.loads(line) for line in open(f)]).drop_duplicates("key", keep="last")
    return d.set_index(d.key.astype(int))


def feats(d):
    p = d.parsed.apply(lambda x: x if isinstance(x, dict) else {})
    f = p.apply(lambda x: x.get("features", {}) if isinstance(x.get("features"), dict) else {})
    out = pd.DataFrame({c: f.apply(lambda z: str(z.get(c, "")).lower()) for c in FEATURES}, index=d.index)
    out["dx"] = p.apply(lambda x: str(x.get("diagnosis", "")).lower())
    out["prob"] = pd.to_numeric(p.apply(lambda x: x.get("melanoma_probability")), errors="coerce").fillna(0.5)
    return out


def main():
    gt = pd.read_csv(ROOT / "splits/derm7pt.csv").set_index("case_num")
    rows = []
    for m in sorted(p.name for p in VLM.iterdir() if p.is_dir()):
        o, b = VLM / m / "derm7pt_test__shuffled.jsonl", VLM / m / "derm7pt_test__shuffled__blur.jsonl"
        if o.exists() and b.exists():
            fo, fb = feats(load(o)), feats(load(b))
            k = fo.index.intersection(fb.index)
            fo, fb, y = fo.loc[k], fb.loc[k], gt.loc[k, "mel"].values.astype(bool)
            r = {"model": m, "prompt": "JSON (shuffled)", "n": len(k),
                 "feature_agreement": float((fo[FEATURES].values == fb[FEATURES].values).mean()),
                 "all7_identical": float((fo[FEATURES].values == fb[FEATURES].values).all(1).mean()),
                 "diagnosis_agreement": float((fo.dx == fb.dx).mean()),
                 "mel_auroc_original": roc_auc_score(y, fo.prob), "mel_auroc_blurred": roc_auc_score(y, fb.prob),
                 "said_melanoma_original": float(fo.dx.str.contains("melanoma").mean()),
                 "said_melanoma_blurred": float(fb.dx.str.contains("melanoma").mean())}
            for c in FEATURES:
                r[f"agree_{c}"] = float((fo[c] == fb[c]).mean())
            rows.append(r)
        o, b = VLM / m / "derm7pt_test__yesno.jsonl", VLM / m / "derm7pt_test__yesno__blur.jsonl"
        if o.exists() and b.exists():
            po, pb = load(o).p_yes, load(b).p_yes
            k = po.index.intersection(pb.index)
            y = gt.loc[k, "mel"].values.astype(bool)
            rows.append({"model": m, "prompt": "yes/no", "n": len(k),
                         "diagnosis_agreement": float(((po.loc[k] > 0.5) == (pb.loc[k] > 0.5)).mean()),
                         "p_yes_correlation": float(np.corrcoef(po.loc[k], pb.loc[k])[0, 1]),
                         "mel_auroc_original": roc_auc_score(y, po.loc[k]), "mel_auroc_blurred": roc_auc_score(y, pb.loc[k])})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/blur_control.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(res.T.to_string())


if __name__ == "__main__":
    main()
