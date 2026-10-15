"""Run all Phase 1 baselines sequentially. Resumable: jobs whose metrics.json exists are skipped.

Order: DINOv2 probes, then seed 0 of every model x task (a full first picture), then seeds 1 and 2.

Usage (keeps the Mac awake):
  caffeinate -i .venv/bin/python -I experiments/run_phase1.py > results/logs/phase1.log 2>&1
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TASKS = ["site_rosendahl", "site_vidir_modern", "random"]
MODELS = ["convnext_tiny", "efficientnet_b3", "vit_base"]
SEEDS = [0, 1, 2]


def jobs():
    for task in TASKS:
        yield ROOT / f"results/runs/{task}/dinov2_probe/seed0", ["experiments/linear_probe.py", "--task", task]
    for seed in SEEDS:
        for model in MODELS:
            for task in TASKS:
                yield (ROOT / f"results/runs/{task}/{model}/seed{seed}",
                       ["experiments/train_supervised.py", "--model", model, "--task", task, "--seed", str(seed)])


def main():
    logs = ROOT / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for out, cmd in jobs():
        if (out / "metrics.json").exists():
            print("skip", out.relative_to(ROOT), flush=True)
            continue
        name = "_".join(out.relative_to(ROOT / "results/runs").parts)
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "start", name, flush=True)
        with open(logs / f"{name}.log", "w") as log:
            rc = subprocess.run([PY, "-I", *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M:%S"), "end", name, f"rc={rc}", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
