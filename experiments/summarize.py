"""Collect results/runs/**/metrics.json into results/phase1_summary.md (mean ± SD over seeds; CI shown when one seed).

Usage: .venv/bin/python -I experiments/summarize.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["balanced_acc", "macro_f1", "macro_auroc", "mel_auroc", "mel_sens_at_spec95", "mel_sens_valthr", "ece"]


def cell(vals):
    if len(vals) == 1:
        v = vals[0]
        return f"{v['value']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    xs = [v["value"] for v in vals]
    return f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f} (n={len(xs)})"


def main():
    runs = defaultdict(list)  # (task, test_set, model) -> [results]
    for f in sorted((ROOT / "results/runs").glob("*/*/seed*/metrics.json")):
        task, model = f.parts[-4], f.parts[-3]
        for test_set, res in json.load(open(f))["results"].items():
            runs[(task, test_set, model)].append(res)

    lines = ["# Phase 1 baseline results", "",
             "One seed: value [95% lesion-bootstrap CI]. Several seeds: mean ± SD.", ""]
    for task in sorted({k[0] for k in runs}):
        lines += [f"## {task}", "", "| test set | model | " + " | ".join(METRICS) + " |",
                  "|" + "---|" * (len(METRICS) + 2)]
        for (t, test_set, model), vals in sorted(runs.items()):
            if t == task:
                lines.append(f"| {test_set} | {model} | " + " | ".join(
                    cell([v[m] for v in vals]) if m in vals[0] else "" for m in METRICS) + " |")
        lines.append("")

    gap = ["## Site-shift gap (melanoma AUROC, in-site minus cross-site, same model)", "",
           "| trained on | model | in-site | cross-site | gap |", "|---|---|---|---|---|"]
    for site, other in [("rosendahl", "vidir_modern"), ("vidir_modern", "rosendahl")]:
        for model in sorted({k[2] for k in runs if k[0] == f"site_{site}"}):
            a, b = runs.get((f"site_{site}", f"test_{site}", model)), runs.get((f"site_{site}", f"test_{other}", model))
            if a and b:
                ia, ib = np.mean([v["mel_auroc"]["value"] for v in a]), np.mean([v["mel_auroc"]["value"] for v in b])
                gap.append(f"| {site} | {model} | {ia:.3f} | {ib:.3f} | {ia - ib:+.3f} |")
    out = ROOT / "results/phase1_summary.md"
    out.write_text("\n".join(lines + gap) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    sys.exit(main())
