"""Linear probe on frozen DINOv2-B features (Phase 1, family A).

Features are the CLS embedding at 224 px with no augmentation. A class-balanced logistic
regression is fit on train with C chosen on validation balanced accuracy. Deterministic, so one run per task.

Usage: .venv/bin/python -I experiments/linear_probe.py --task random
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinbench import metrics  # noqa: E402
from skinbench.data import ROOT, LesionDataset, load_task, make_transforms  # noqa: E402

TIMM_ID = "vit_base_patch14_dinov2.lvd142m"
CS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 10.0]


def features(model, df, classes, tf, device, cache):
    if cache.exists():
        return np.load(cache)
    dl = torch.utils.data.DataLoader(LesionDataset(df, classes, tf), 64, num_workers=6)
    model.eval()
    with torch.no_grad():
        f = torch.cat([model(x.to(device)).cpu() for x, _ in dl]).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, f)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classes, sets = load_task(a.task)
    model = timm.create_model(TIMM_ID, pretrained=True, num_classes=0, img_size=224).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    _, tf = make_transforms(224, cfg["mean"], cfg["std"])

    cache = ROOT / "results/features/dinov2_b" / a.task
    X = {k: features(model, v, classes, tf, device, cache / f"{k}.npy") for k, v in sets.items()}
    y = {k: np.array([classes.index(d) for d in v.dx]) for k, v in sets.items()}
    scaler = StandardScaler().fit(X["train"])
    X = {k: scaler.transform(v) for k, v in X.items()}

    fits = {}
    for c in CS:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(X["train"], y["train"])
        fits[c] = (metrics.compute(y["val"], clf.predict_proba(X["val"]), classes)["balanced_acc"], clf)
    best_c = max(fits, key=lambda c: fits[c][0])
    clf = fits[best_c][1]

    pv = clf.predict_proba(X["val"])
    m = classes.index("mel")
    mel_thr = metrics.threshold_at_spec((y["val"] == m).astype(int), pv[:, m])
    out = ROOT / "results/runs" / a.task / "dinov2_probe" / "seed0"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df in sets.items():
        if name == "train":
            continue
        p = clf.predict_proba(X[name])
        pd.DataFrame(p, columns=[f"p_{c}" for c in classes]).assign(
            image_id=df.image_id.values, lesion_id=df.lesion_id.values, y=y[name]).to_csv(out / f"pred_{name}.csv", index=False)
        if name != "val":
            results[name] = metrics.bootstrap(y[name], p, df.lesion_id.values, classes, mel_thr)
    json.dump({"config": {"model": "dinov2_probe", "task": a.task, "timm_id": TIMM_ID, "classes": classes,
                          "C": best_c, "val_bacc_by_C": {str(c): fits[c][0] for c in CS},
                          "mel_threshold_val_spec95": mel_thr},
               "results": results}, open(out / "metrics.json", "w"), indent=2)
    print("done", out, "C =", best_c, {k: round(v["mel_auroc"]["value"], 3) for k, v in results.items()})


if __name__ == "__main__":
    main()
