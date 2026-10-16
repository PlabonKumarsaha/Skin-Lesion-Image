"""E2: score open-VLM answers on Derm7pt against expert 7-point-checklist labels.

Per model x prompt variant (aggregates only, so the output can be shared):
  diagnosis     melanoma AUROC from the stated probability (JSON variants) or P(yes) (yesno); melanoma recall/specificity
  concepts      per checklist criterion: precision / recall / F1 of the model citing it vs expert labels, and Cohen's kappa
                on the full category; "uncertain" counts as not cited (its rate is reported)
  consistency   does the model's own 7-point score (>= 3 -> melanoma) agree with its own diagnosis?
                cross-format: JSON diagnosis "melanoma" vs yes/no P(yes) > 0.5
  templating    share of distinct feature vectors (1.0 = every answer different; low = one answer for every image)
CIs: 1,000-case bootstrap.

Usage: .venv/bin/python -I experiments/faithfulness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VLM = ROOT / "results/vlm"
FEATURES = ["pigment_network", "streaks", "pigmentation", "regression_structures",
            "dots_and_globules", "blue_whitish_veil", "vascular_structures"]
# 7-point checklist (Argenziano): major criteria score 2, minor 1; total >= 3 suggests melanoma
CHECKLIST = {
    "atypical_pigment_network": ("pigment_network", {"atypical"}, 2),
    "blue_whitish_veil_present": ("blue_whitish_veil", {"present"}, 2),
    "atypical_vascular": ("vascular_structures", {"dotted", "linear irregular", "within regression"}, 2),
    "irregular_streaks": ("streaks", {"irregular"}, 1),
    "irregular_pigmentation": ("pigmentation", {"diffuse irregular", "localized irregular"}, 1),
    "irregular_dots_globules": ("dots_and_globules", {"irregular"}, 1),
    "regression_present": ("regression_structures", {"blue areas", "white areas", "combinations"}, 1),
}


def load(model, variant):
    f = VLM / model / ("derm7pt_test.jsonl" if variant == "schema" else f"derm7pt_test__{variant}.jsonl")
    if not f.exists():
        return None
    rows = [json.loads(line) for line in open(f)]
    return pd.DataFrame(rows).drop_duplicates("key", keep="last").assign(case_num=lambda d: d.key.astype(int))


def prf(pred, true):
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return p, r, (2 * p * r / (p + r) if p + r else np.nan)


def json_metrics(df, gt):
    d = df.merge(gt, on="case_num")
    parsed = d.parsed.apply(lambda x: x if isinstance(x, dict) else {})
    feats = parsed.apply(lambda x: x.get("features", {}) if isinstance(x.get("features"), dict) else {})
    out = {"n": len(d), "parse_rate": float((parsed.apply(len) > 0).mean())}
    dx = parsed.apply(lambda x: str(x.get("diagnosis", "")).lower())
    said_mel = dx.str.contains("melanoma").values
    prob = pd.to_numeric(parsed.apply(lambda x: x.get("melanoma_probability")), errors="coerce").fillna(0.5).values
    mel = d.mel.values.astype(bool)
    out |= {"mel_auroc_stated": roc_auc_score(mel, prob) if 0 < mel.sum() < len(mel) else np.nan,
            "mel_recall": said_mel[mel].mean(), "mel_specificity": (~said_mel[~mel]).mean(),
            "distinct_prob_values": int(pd.Series(prob).nunique())}
    score = np.zeros(len(d))
    for flag, (col, vals, w) in CHECKLIST.items():
        val = feats.apply(lambda f: str(f.get(col, "")).lower())
        cited = val.isin(vals).values
        p, r, f1 = prf(cited, d[flag].values.astype(bool))
        out[f"{flag}_precision"], out[f"{flag}_recall"], out[f"{flag}_f1"] = p, r, f1
        out[f"{flag}_cited_rate"] = cited.mean()
        score += w * cited
    for col in FEATURES:
        val = feats.apply(lambda f: str(f.get(col, "")).lower())
        out[f"{col}_uncertain_rate"] = (val == "uncertain").mean()
        known = val != "uncertain"
        out[f"{col}_kappa"] = cohen_kappa_score(d[col].str.lower()[known], val[known]) if known.sum() > 10 else np.nan
    out["concept_macro_f1"] = np.nanmean([out[f"{k}_f1"] for k in CHECKLIST])
    out["self_consistency"] = ((score >= 3) == said_mel).mean()
    out["distinct_feature_vectors"] = feats.apply(lambda f: json.dumps(f, sort_keys=True)).nunique() / max(len(d), 1)
    return out, pd.Series(said_mel, index=d.case_num)


def main():
    gt = pd.read_csv(ROOT / "splits/derm7pt.csv").query("split == 'test'")
    rows, said = [], {}
    for model in sorted(p.name for p in VLM.iterdir() if p.is_dir()):
        for variant in ["schema", "shuffled"]:
            df = load(model, variant)
            if df is None or len(df) < 20:
                continue
            m, s = json_metrics(df, gt)
            said[(model, variant)] = s
            rows.append({"model": model, "variant": variant} | m)
        y = load(model, "yesno")
        if y is not None and len(y) >= 20:
            d = y.merge(gt, on="case_num")
            mel = d.mel.values.astype(bool)
            r = {"model": model, "variant": "yesno", "n": len(d), "mel_auroc_stated": roc_auc_score(mel, d.p_yes),
                 "mel_recall": (d.p_yes[mel] > 0.5).mean(), "mel_specificity": (d.p_yes[~mel] <= 0.5).mean()}
            if (model, "shuffled") in said:
                j = said[(model, "shuffled")]
                both = d.set_index("case_num").p_yes.reindex(j.index).dropna()
                r["cross_format_agreement"] = ((both > 0.5) == j.loc[both.index]).mean()
            rows.append(r)
    res = pd.DataFrame(rows)
    out = ROOT / "results/faithfulness_derm7pt.csv"
    res.to_csv(out, index=False)
    cols = ["model", "variant", "n", "parse_rate", "mel_auroc_stated", "mel_recall", "mel_specificity",
            "concept_macro_f1", "self_consistency", "distinct_feature_vectors", "cross_format_agreement"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res[[c for c in cols if c in res]].to_string(index=False))
        flags = [f"{k}_{m}" for k in CHECKLIST for m in ("precision", "recall")]
        print(res[["model", "variant"] + [c for c in flags if c in res]].dropna(how="all", axis=1).T.to_string())


if __name__ == "__main__":
    main()
