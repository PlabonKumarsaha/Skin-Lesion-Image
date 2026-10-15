"""Shared evaluation for every model family.

All metrics take integer labels `y` (n,) and class probabilities `p` (n, K).
Confidence intervals come from a lesion-level bootstrap: lesions are resampled, so
multiple images of one lesion never count as independent evidence.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve


def sens_at_spec(y_bin, score, spec=0.95):
    """Sensitivity at the ROC point with specificity >= `spec` (read off the test ROC)."""
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y_bin, score, sens=0.95):
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = tpr >= sens
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def threshold_at_spec(y_bin, score, spec=0.95):
    """Score threshold giving specificity >= `spec`; choose it on validation, apply to test."""
    neg = np.sort(score[y_bin == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)])


def ece(y, p, bins=15):
    """Expected calibration error of the top-label confidence."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs((pred[idx == b] == y[idx == b]).mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def compute(y, p, classes, mel_threshold=None):
    """Point estimates. `mel_threshold` (from validation) adds a deployment-style sensitivity/specificity."""
    k = len(classes)
    pred = p.argmax(1)
    out = {
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(k), zero_division=0),
        "ece": ece(y, p),
        "brier": float(((p - np.eye(k)[y]) ** 2).sum(1).mean()),
    }
    present = np.unique(y)
    if len(present) > 1:
        out["macro_auroc"] = float(np.mean([roc_auc_score(y == c, p[:, c]) for c in present]))
    for c, name in enumerate(classes):
        out[f"recall_{name}"] = recall_score(y == c, pred == c, zero_division=0)
    if "mel" in classes:
        m = classes.index("mel")
        y_mel, s_mel = (y == m).astype(int), p[:, m]
        if 0 < y_mel.sum() < len(y_mel):
            out["mel_auroc"] = roc_auc_score(y_mel, s_mel)
            out["mel_sens_at_spec95"] = sens_at_spec(y_mel, s_mel, 0.95)
            out["mel_spec_at_sens95"] = spec_at_sens(y_mel, s_mel, 0.95)
            if mel_threshold is not None:
                hit = s_mel >= mel_threshold
                out["mel_sens_valthr"] = float(hit[y_mel == 1].mean())
                out["mel_spec_valthr"] = float((~hit)[y_mel == 0].mean())
    return {key: float(v) for key, v in out.items()}


def bootstrap(y, p, groups, classes, mel_threshold=None, n=1000, seed=0):
    """Point estimates plus 95% percentile CIs from resampling lesions (`groups`)."""
    point = compute(y, p, classes, mel_threshold)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draws = {key: [] for key in point}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        res = compute(y[idx], p[idx], classes, mel_threshold)
        for key in point:
            if key in res:
                draws[key].append(res[key])
    return {key: {"value": v, "ci_low": float(np.percentile(draws[key], 2.5)),
                  "ci_high": float(np.percentile(draws[key], 97.5))} for key, v in point.items()}
