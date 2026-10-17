"""Diagnostic performance of the open VLMs on every set they were run on (aggregates only).

Melanoma score: P(yes) for the yes/no prompt; stated melanoma probability for JSON prompts.
For the HAM site-shift test set the VLMs are zero-shot (no training), so results are split by site to show whether
they simply perform differently on Queensland vs Vienna images.

Usage: .venv/bin/python -I experiments/vlm_summary.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def truth():
    r = pd.read_csv(ROOT / "splits/random.csv").query("split == 'test'")
    s = pd.read_csv(ROOT / "splits/site_shift.csv").query("split == 'test'")
    d7 = pd.read_csv(ROOT / "splits/derm7pt.csv").query("split == 'test'")
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")
    return {
        "ham_random_test": pd.DataFrame({"key": r.image_id, "y": r.dx.eq("mel"), "sub": "all"}),
        "ham_site_test": pd.DataFrame({"key": s.image_id, "y": s.dx.eq("mel"), "sub": s.dataset}),
        "derm7pt_test": pd.DataFrame({"key": d7.case_num.astype(str), "y": d7.mel, "sub": "all"}),
        "ddi": pd.DataFrame({"key": ddi.DDI_ID.astype(str), "y": ddi.malignant, "sub": ddi.fst_group}),
    }


def load(f):
    d = pd.DataFrame([json.loads(line) for line in open(f)]).drop_duplicates("key", keep="last")
    d["key"] = d.key.astype(str)
    if "p_yes" in d:
        d["score"] = d.p_yes
        d["said"] = d.p_yes > 0.5
    else:
        p = d.parsed.apply(lambda x: x if isinstance(x, dict) else {})
        d["score"] = pd.to_numeric(p.apply(lambda x: x.get("melanoma_probability")), errors="coerce").fillna(0.5)
        d["said"] = p.apply(lambda x: "melanoma" in str(x.get("diagnosis", "")).lower())
    return d[["key", "score", "said"]]


def main():
    gt = truth()
    rows = []
    for f in sorted((ROOT / "results/vlm").glob("*/*.jsonl")):
        model, name = f.parent.name, f.stem
        base, _, variant = name.partition("__")
        variant = variant or "schema"
        if base not in gt:
            continue
        d = gt[base].merge(load(f), on="key")
        for sub, x in [("all", d)] + ([(g, x) for g, x in d.groupby("sub")] if d["sub"].nunique() > 1 else []):
            y = x.y.values.astype(bool)
            if not 0 < y.sum() < len(y):
                continue
            rows.append({"model": model, "set": base, "variant": variant, "subset": sub, "n": len(x),
                         "auroc": roc_auc_score(y, x.score), "recall": x.said[y].mean(), "specificity": (~x.said[~y]).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/vlm_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format, "display.max_rows", 200):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
