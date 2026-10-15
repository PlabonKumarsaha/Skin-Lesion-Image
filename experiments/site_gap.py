"""Paired site-shift gap (E1): on one test site, compare the in-site model with the cross-site model.

For each architecture, seed and test site T, the gap is
  metric(model trained on T, T's test set) - metric(model trained on the other site, T's test set).
Both use the same images, so lesions are resampled jointly (paired bootstrap). Seeds are averaged per draw.

Usage: .venv/bin/python -I experiments/site_gap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinbench import metrics  # noqa: E402
from skinbench.data import CLASSES_4, SITES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"
N_BOOT = 2000


def load(train_site, model, seed, test_site):
    f = RUNS / f"site_{train_site}" / model / f"seed{seed}" / f"pred_test_{test_site}.csv"
    return pd.read_csv(f) if f.exists() else None


def stats(pred, idx):
    y, p = pred.y.values[idx], pred[[f"p_{c}" for c in CLASSES_4]].values[idx]
    mel = (y == CLASSES_4.index("mel")).astype(int)
    s = p[:, CLASSES_4.index("mel")]
    return np.array([roc_auc_score(mel, s), metrics.sens_at_spec(mel, s), metrics.compute(y, p, CLASSES_4)["balanced_acc"]])


def main():
    names = ["mel_auroc", "mel_sens_at_spec95", "balanced_acc"]
    rows = []
    models = sorted({p.parent.parent.name for p in RUNS.glob("site_*/*/seed*/pred_test_*.csv")})
    for model in models:
        for test in SITES:
            other = [s for s in SITES if s != test][0]
            pairs = [(load(test, model, s, test), load(other, model, s, test)) for s in range(3)]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                continue
            ref = pairs[0][0]
            assert all((a.image_id.values == ref.image_id.values).all() and (b.image_id.values == ref.image_id.values).all()
                       for a, b in pairs)
            uniq, inv = np.unique(ref.lesion_id.values, return_inverse=True)
            members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
            point = np.mean([stats(a, slice(None)) - stats(b, slice(None)) for a, b in pairs], 0)
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(N_BOOT):
                idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
                draws.append(np.mean([stats(a, idx) - stats(b, idx) for a, b in pairs], 0))
            draws = np.array(draws)
            for i, n in enumerate(names):
                lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
                p = 2 * min((draws[:, i] <= 0).mean(), (draws[:, i] >= 0).mean())
                rows.append({"model": model, "test_site": test, "seeds": len(pairs), "metric": n,
                             "gap": point[i], "ci_low": lo, "ci_high": hi, "p_boot": min(1.0, max(p, 1 / N_BOOT))})
    df = pd.DataFrame(rows)
    out = ROOT / "results/site_gap.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
