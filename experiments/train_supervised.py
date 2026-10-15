"""Fine-tune a timm image model on one HAM10000 task (Phase 1, family A).

Selects the best epoch on validation balanced accuracy, then writes test predictions
and bootstrap metrics for every test set of the task.

Usage:
  .venv/bin/python -I experiments/train_supervised.py --model convnext_tiny --task random --seed 0
  .venv/bin/python -I experiments/train_supervised.py --model convnext_tiny --task random --epochs 1 --timing
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinbench import metrics  # noqa: E402
from skinbench.data import ROOT, LesionDataset, load_task, make_transforms  # noqa: E402

MODELS = {  # short name -> (timm id, learning rate)
    "efficientnet_b3": ("efficientnet_b3.ra2_in1k", 3e-4),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", 1e-4),
    "vit_base": ("vit_base_patch16_224.augreg2_in21k_ft_in1k", 5e-5),
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--task", required=True, help="random | site_rosendahl | site_vidir_modern")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timing", action="store_true", help="time one epoch, write nothing")
    return ap.parse_args()


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(F.softmax(model(x.to(device)).float(), 1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    timm_id, lr = MODELS[a.model]
    classes, sets = load_task(a.task)

    model = timm.create_model(timm_id, pretrained=True, num_classes=len(classes)).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf_train, tf_test = make_transforms(a.size, cfg["mean"], cfg["std"])

    def loader(df, train):
        ds = LesionDataset(df, classes, tf_train if train else tf_test)
        g = torch.Generator().manual_seed(a.seed)
        return torch.utils.data.DataLoader(ds, a.batch, shuffle=train, num_workers=a.workers, generator=g,
                                           drop_last=train, persistent_workers=a.workers > 0)

    train_dl = loader(sets["train"], True)
    eval_dls = {k: loader(v, False) for k, v in sets.items() if k != "train"}

    counts = np.bincount(train_dl.dataset.labels, minlength=len(classes))
    weight = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = a.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))

    out = ROOT / "results/runs" / a.task / a.model / f"seed{a.seed}"
    ckpt = ROOT / "results/checkpoints" / a.task / a.model / f"seed{a.seed}.pt"
    if not a.timing:
        out.mkdir(parents=True, exist_ok=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)

    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            loss = F.cross_entropy(model(x.to(device)), y.to(device), weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        t_train = time.time() - t0
        yv, pv = predict(model, eval_dls["val"], device)
        val_bacc = metrics.compute(yv, pv, classes)["balanced_acc"]
        history.append({"epoch": epoch, "loss": total / len(train_dl), "val_bacc": val_bacc,
                        "train_sec": t_train, "epoch_sec": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if a.timing:
            return
        if val_bacc > best:
            best = val_bacc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    yv, pv = predict(model, eval_dls["val"], device)
    mel_thr = metrics.threshold_at_spec((yv == classes.index("mel")).astype(int), pv[:, classes.index("mel")])

    results = {}
    for name, dl in eval_dls.items():
        y, p = predict(model, dl, device)
        df = sets[name]
        pd.DataFrame(p, columns=[f"p_{c}" for c in classes]).assign(
            image_id=df.image_id.values, lesion_id=df.lesion_id.values, y=y).to_csv(out / f"pred_{name}.csv", index=False)
        if name != "val":
            results[name] = metrics.bootstrap(y, p, df.lesion_id.values, classes, mel_thr)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    json.dump({"config": vars(a) | {"timm_id": timm_id, "lr": lr, "classes": classes, "device": str(device),
                                    "best_val_bacc": best, "mel_threshold_val_spec95": mel_thr},
               "results": results}, open(out / "metrics.json", "w"), indent=2)
    print("done", out)


if __name__ == "__main__":
    main()
