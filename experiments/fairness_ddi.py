"""E4: malignant-vs-benign performance on DDI by Fitzpatrick group (I-II, III-IV, V-VI). Aggregates only.

Scores per model:
  image-text zero-shot   p_malignant from results/runs_foundation/<model>/zs_ddi__all.csv
  open VLMs, yes/no      P(yes) to "Is this lesion malignant?"
  open VLMs, JSON        stated melanoma probability (a weaker proxy: DDI malignancies are mostly non-melanoma)
For each: AUROC with 95% bootstrap CI per group, and the worst-minus-best group gap.

Usage: .venv/bin/python -I experiments/fairness_ddi.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["I-II", "III-IV", "V-VI"]


def auroc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, s)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def scores():
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")
    out = {}
    for f in sorted((ROOT / "results/runs_foundation").glob("*/zs_ddi__all.csv")):
        if "_limit" in f.parent.name:
            continue
        z = pd.read_csv(f)
        out[f"{f.parent.name} · zero-shot"] = pd.Series(z.p_malignant.values, index=z.group.values)
    for m in ["qwen25vl7b", "medgemma15_4b"]:
        y = ROOT / f"results/vlm/{m}/ddi__yesno.jsonl"
        if y.exists():
            d = pd.DataFrame([json.loads(line) for line in open(y)]).drop_duplicates("key", keep="last")
            out[f"{m} · yes/no"] = pd.Series(d.p_yes.values, index=d.key.astype(int).values)
        j = ROOT / f"results/vlm/{m}/ddi__shuffled.jsonl"
        if j.exists():
            d = pd.DataFrame([json.loads(line) for line in open(j)]).drop_duplicates("key", keep="last")
            p = d.parsed.apply(lambda x: x.get("melanoma_probability") if isinstance(x, dict) else None)
            out[f"{m} · JSON (mel. prob.)"] = pd.Series(pd.to_numeric(p, errors="coerce").fillna(0.5).values,
                                                       index=d.key.astype(int).values)
    return ddi.set_index("DDI_ID"), out


def main():
    ddi, sc = scores()
    rows = []
    for name, s in sc.items():
        d = ddi.join(s.rename("score"), how="inner")
        r = {"model": name, "n": len(d)}
        for g in GROUPS:
            x = d[d.fst_group == g]
            a, lo, hi = auroc_ci(x.malignant.values.astype(int), x.score.values)
            r[g] = a
            r[f"{g}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        a, lo, hi = auroc_ci(d.malignant.values.astype(int), d.score.values)
        r["all"], r["all_ci"] = a, f"[{lo:.3f}, {hi:.3f}]"
        r["gap_best_minus_worst"] = max(r[g] for g in GROUPS) - min(r[g] for g in GROUPS)
        r["worst_group"] = min(GROUPS, key=lambda g: r[g])
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("all", ascending=False)
    res.to_csv(ROOT / "results/fairness_ddi.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(res[["model", "n", "all", "I-II", "III-IV", "V-VI", "gap_best_minus_worst", "worst_group",
                   "V-VI_ci"]].to_string(index=False))


if __name__ == "__main__":
    main()
