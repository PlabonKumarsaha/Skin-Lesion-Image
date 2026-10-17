"""E4 part 3: melanoma AUROC of the fine-tuned models by age band, sex and body site (HAM test, BCN20000).

Averages over models x seeds; reports the per-subgroup mean and the spread across models. Subgroups with fewer than
20 melanomas are skipped. Aggregates only.

Usage: .venv/bin/python -I experiments/subgroups.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MIN_POS = 20


def age_band(a):
    return pd.cut(a, [0, 40, 60, 120], right=False, labels=["<40", "40-59", "60+"])


def frames():
    ham = pd.read_csv(ROOT / "splits/random.csv").query("split == 'test'")
    ext = pd.read_csv(ROOT / "splits/external.csv").query("dataset == 'bcn20000'")
    for f in glob.glob(str(ROOT / "results/runs/random/*/seed*/pred_test.csv")):
        model = Path(f).parts[-3]
        if model == "dinov2_probe":
            continue
        p = pd.read_csv(f).merge(ham, on="image_id")
        yield "ham_test", model, pd.DataFrame({"y": p.dx.eq("mel"), "s": p.p_mel, "age": age_band(p.age),
                                              "sex": p.sex, "site": p.localization})
    for f in glob.glob(str(ROOT / "results/external/pred_*_seed*_bcn20000.csv")):
        model = Path(f).name[5:].rsplit("_seed", 1)[0]
        p = pd.read_csv(f).merge(ext, on="key")
        yield "bcn20000", model, pd.DataFrame({"y": p.mel, "s": p.p_mel, "age": age_band(p.age),
                                              "sex": p.sex, "site": p.site})


def main():
    rows = []
    for ds, model, d in frames():
        for attr in ["age", "sex", "site"]:
            for g, x in d.groupby(attr, observed=True):
                if x.y.sum() >= MIN_POS and (~x.y).sum() >= MIN_POS:
                    rows.append({"dataset": ds, "attr": attr, "group": str(g), "model": model,
                                 "n": len(x), "n_mel": int(x.y.sum()), "auroc": roc_auc_score(x.y, x.s)})
    r = pd.DataFrame(rows)
    agg = r.groupby(["dataset", "attr", "group"]).agg(n=("n", "first"), n_mel=("n_mel", "first"),
                                                       auroc=("auroc", "mean"), sd=("auroc", "std")).reset_index()
    agg["gap_to_best"] = agg.groupby(["dataset", "attr"]).auroc.transform("max") - agg.auroc
    agg.to_csv(ROOT / "results/subgroups.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format, "display.max_rows", 100):
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
