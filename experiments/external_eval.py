"""External test: HAM10000-trained fine-tuned models (random split, all seeds) on BCN20000 and PAD-UFES-20.

Metrics per dataset (lesion/patient-level bootstrap CIs):
  melanoma   AUROC, sensitivity at 95% specificity (test ROC), and sensitivity/specificity at the HAM-validation
             threshold (what a deployed model would do)
  malignant  AUROC of p(mel) + p(bcc) + p(akiec)
  balanced accuracy over images whose class exists in HAM10000

Usage: .venv/bin/python -I experiments/external_eval.py [--models convnext_tiny vit_base efficientnet_b3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinbench import metrics  # noqa: E402
from skinbench.data import CLASSES_7, make_transforms  # noqa: E402
from experiments.train_supervised import MODELS  # noqa: E402

PROC = ROOT / "data/processed"


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def evaluate(df, p, thr, n_boot=1000, seed=0):
    """Point estimates + group-bootstrap CIs."""
    mel, mal = df.mel.values.astype(int), df.malignant.values.astype(int)
    pm = p[:, CLASSES_7.index("mel")]
    pmal = p[:, [CLASSES_7.index(c) for c in ("mel", "bcc", "akiec")]].sum(1)
    known = df.dx.notna().values
    yk = np.array([CLASSES_7.index(d) for d in df.dx[known]])

    def stats(i):
        k = known[i]
        return {"mel_auroc": roc_auc_score(mel[i], pm[i]),
                "mel_sens_at_spec95": metrics.sens_at_spec(mel[i], pm[i]),
                "mel_sens_hamthr": float((pm[i][mel[i] == 1] >= thr).mean()),
                "mel_spec_hamthr": float((pm[i][mel[i] == 0] < thr).mean()),
                "malignant_auroc": roc_auc_score(mal[i], pmal[i]),
                "balanced_acc_ham_classes": balanced_accuracy_score(
                    np.array([CLASSES_7.index(d) for d in df.dx.values[i][k]]), p[i][k].argmax(1))}

    point = stats(np.arange(len(df)))
    groups = df.group.values
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = [stats(np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])) for _ in range(n_boot)]
    return {k: {"value": v, "ci_low": float(np.percentile([d[k] for d in draws], 2.5)),
                "ci_high": float(np.percentile([d[k] for d in draws], 97.5))} for k, v in point.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ext = pd.read_csv(ROOT / "splits/external.csv")
    out = ROOT / "results/external"
    out.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out / "finetuned.json")) if (out / "finetuned.json").exists() else {}
    for name in a.models:
        timm_id, _ = MODELS[name]
        for seed in a.seeds:
            tag = f"{name}/seed{seed}"
            if tag in results:
                continue
            run = ROOT / f"results/runs/random/{name}/seed{seed}"
            thr = json.load(open(run / "metrics.json"))["config"]["mel_threshold_val_spec95"]
            model = timm.create_model(timm_id, pretrained=False, num_classes=7)
            model.load_state_dict(torch.load(ROOT / f"results/checkpoints/random/{name}/seed{seed}.pt", map_location="cpu"))
            model.to(device).eval()
            cfg = timm.data.resolve_data_config({}, model=model)
            _, tf = make_transforms(224, cfg["mean"], cfg["std"])
            results[tag] = {}
            for ds, df in ext.groupby("dataset"):
                dl = torch.utils.data.DataLoader(Images([PROC / p for p in df.img], tf), 64, num_workers=6)
                with torch.no_grad():
                    p = torch.cat([torch.softmax(model(x.to(device)).float(), 1).cpu() for x in dl]).numpy()
                pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES_7]).assign(key=df.key.values).to_csv(
                    out / f"pred_{name}_seed{seed}_{ds}.csv", index=False)
                results[tag][ds] = evaluate(df, p, thr)
                print(tag, ds, {k: round(v["value"], 3) for k, v in results[tag][ds].items()}, flush=True)
            json.dump(results, open(out / "finetuned.json", "w"), indent=2)


if __name__ == "__main__":
    main()
