"""Phase 1b, families B/C: zero-shot + linear probe for open image-text foundation models.

Per model it caches L2-normalised image features, then:
  zero-shot   prompt-ensemble class embeddings -> softmax(logit_scale * cos)  on every test set
  probe       class-balanced logistic regression on frozen features (C picked on val), as for DINOv2
Test sets: HAM random (7-class), both site-shift tasks (4-class), Derm7pt dermoscopic test (HAM classes), DDI (malignant).

Usage: .venv/bin/python -I experiments/foundation_eval.py --model biomedclip [--device cpu] [--limit 64]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinbench import metrics  # noqa: E402
from skinbench.data import CLASSES_7, ROOT, load_task  # noqa: E402
from skinbench.foundation import Foundation  # noqa: E402

PROC = ROOT / "data/processed"
CS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 10.0]
NAMES = {
    "nv": "a melanocytic nevus", "mel": "melanoma",
    "bkl": "a benign keratosis such as seborrheic keratosis or solar lentigo",
    "bcc": "basal cell carcinoma", "akiec": "actinic keratosis or intraepithelial carcinoma",
    "vasc": "a vascular lesion such as an angioma", "df": "dermatofibroma",
    "benign": "a benign skin lesion", "malignant": "a malignant skin lesion, skin cancer",
}
DERMOSCOPIC = ["a dermoscopic image of {}.", "dermoscopy of {}.", "a dermatoscopic photo showing {}.",
               "this is a dermoscopy image of {}."]
CLINICAL = ["a clinical photo of {}.", "a photograph of {} on the skin.", "a close-up clinical image of {}.",
            "this is a clinical photograph of {}."]


def image_sets(limit):
    """{set_name: (paths, labels, groups, classes, templates)} for every evaluation set."""
    sets = {}
    for task in ["random", "site_rosendahl", "site_vidir_modern"]:
        classes, parts = load_task(task)
        for name, df in parts.items():
            sets[f"{task}/{name}"] = ([PROC / "ham10000_400" / f"{i}.jpg" for i in df.image_id],
                                      np.array([classes.index(d) for d in df.dx]), df.lesion_id.values, classes, DERMOSCOPIC)
    d7 = pd.read_csv(ROOT / "splits/derm7pt.csv")
    d7 = d7[(d7.split == "test") & d7.ham_dx.notna()]
    sets["derm7pt/test"] = ([PROC / p for p in d7.derm_img], np.array([CLASSES_7.index(d) for d in d7.ham_dx]),
                            d7.case_num.values, CLASSES_7, DERMOSCOPIC)
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")
    sets["ddi/all"] = ([PROC / p for p in ddi.img], ddi.malignant.astype(int).values, ddi.DDI_ID.values,
                       ["benign", "malignant"], CLINICAL)
    if limit:
        rng = np.random.default_rng(0)
        pick = {k: np.sort(rng.permutation(len(v[1]))[:limit]) for k, v in sets.items()}
        sets = {k: ([p[i] for i in pick[k]], y[pick[k]], g[pick[k]], c, t) for k, (p, y, g, c, t) in sets.items()}
    return sets


def features(fm, paths, cache, batch=64):
    if cache.exists():
        return np.load(cache)
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        out.append(fm.encode_pixels(torch.stack([fm.preprocess(im) for im in ims])).numpy())
    f = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, f)
    return f


def zero_shot(fm, feats, classes, templates):
    W = torch.stack([fm.encode_text([t.format(NAMES[c]) for t in templates]).mean(0) for c in classes])
    W = torch.nn.functional.normalize(W, dim=-1).numpy()
    logits = fm.logit_scale * feats @ W.T
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def proba(clf, X, k):
    """predict_proba padded to all k classes (a class missing from training gets probability 0)."""
    out = np.zeros((len(X), k))
    out[:, clf.classes_] = clf.predict_proba(X)
    return out


def evaluate(y, p, groups, classes, thr=None):
    if "mel" not in classes:  # binary malignant task: report malignant as the positive class via the same metrics
        classes = ["benign", "mel"]
    return metrics.bootstrap(y, p, groups, classes, thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N images per set")
    a = ap.parse_args()
    fm = Foundation(a.model, torch.device(a.device))
    sets = image_sets(a.limit)
    tag = f"{a.model}_limit{a.limit}" if a.limit else a.model
    feats = {k: features(fm, v[0], ROOT / "results/features" / tag / f"{k.replace('/', '__')}.npy") for k, v in sets.items()}
    out = ROOT / "results/runs_foundation" / tag
    out.mkdir(parents=True, exist_ok=True)
    results = {"zero_shot": {}, "probe": {}}

    for k, (paths, y, g, classes, templates) in sets.items():
        if k.endswith(("/train", "/val")) or not fm.has_text:
            continue
        p = zero_shot(fm, feats[k], classes, templates)
        results["zero_shot"][k] = evaluate(y, p, g, classes)
        pd.DataFrame(p, columns=[f"p_{c}" for c in classes]).assign(y=y, group=g).to_csv(out / f"zs_{k.replace('/', '__')}.csv", index=False)

    for task in ["random", "site_rosendahl", "site_vidir_modern"]:
        tr, va = sets[f"{task}/train"], sets[f"{task}/val"]
        classes = tr[3]
        fits = {c: LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(feats[f"{task}/train"], tr[1]) for c in CS}
        val = {c: metrics.compute(va[1], proba(f, feats[f"{task}/val"], len(classes)), classes)["balanced_acc"] for c, f in fits.items()}
        best = max(val, key=val.get)
        clf = fits[best]
        pv = proba(clf, feats[f"{task}/val"], len(classes))
        m = classes.index("mel")
        thr = metrics.threshold_at_spec((va[1] == m).astype(int), pv[:, m])
        tests = [k for k in sets if k.startswith(f"{task}/test")]
        if task == "random":
            tests.append("derm7pt/test")  # cross-dataset: HAM-trained probe on Derm7pt dermoscopy
        for k in tests:
            p = proba(clf, feats[k], len(classes))
            results["probe"][f"{task}->{k}"] = evaluate(sets[k][1], p, sets[k][2], classes, thr) | {"C": {"value": best, "ci_low": best, "ci_high": best}}

    json.dump({"model": a.model, "logit_scale": fm.logit_scale, "limit": a.limit, "results": results},
              open(out / "metrics.json", "w"), indent=2)
    for kind, res in results.items():
        for k, v in res.items():
            print(f"{kind:<9} {k:<45} melAUC={v.get('mel_auroc', {}).get('value', float('nan')):.3f} "
                  f"bacc={v['balanced_acc']['value']:.3f}")


if __name__ == "__main__":
    main()
