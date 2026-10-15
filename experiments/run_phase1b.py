"""Run all Phase 1b jobs after Phase 1 frees the GPU. Resumable: finished jobs are skipped, VLM runs resume per image.

Order (most important first):
  1. foundation models: zero-shot + linear probe (fast, ~5-15 min each)
  2. VLMs on Derm7pt test (core of E2): yesno, shuffled, schema (order-echo control)
  3. VLMs on HAM random test, DDI, HAM site-shift test: yesno, shuffled

Usage:
  caffeinate -i .venv/bin/python -I experiments/run_phase1b.py > results/logs/phase1b.log 2>&1
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FOUNDATION = ["biomedclip", "monet", "dermlip_vitb16", "dermlip_panderm", "panderm_l"]
VLMS = ["qwen25vl7b", "medgemma15_4b"]


def jobs():
    for m in FOUNDATION:
        yield ROOT / f"results/runs_foundation/{m}/metrics.json", ["experiments/foundation_eval.py", "--model", m]
    for variant in ["yesno", "shuffled", "schema"]:
        for m in VLMS:
            yield None, ["experiments/vlm_eval.py", "--model", m, "--set", "derm7pt_test", "--variant", variant]
    for s in ["ham_random_test", "ddi", "ham_site_test"]:
        for variant in ["yesno", "shuffled"]:
            for m in VLMS:
                yield None, ["experiments/vlm_eval.py", "--model", m, "--set", s, "--variant", variant]


def main():
    while subprocess.run(["pgrep", "-f", "run_phase1.py"], capture_output=True).returncode == 0:
        time.sleep(120)  # Phase 1 still training
    logs = ROOT / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for done, cmd in jobs():
        if done is not None and done.exists():
            print("skip", " ".join(cmd[1:]), flush=True)
            continue
        name = "_".join(c.lstrip("-") for c in cmd[1:] if c not in ("--model", "--set", "--variant"))
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "start", name, flush=True)
        with open(logs / f"p1b_{name}.log", "a") as log:
            rc = subprocess.run([PY, "-I", *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M:%S"), "end", name, f"rc={rc}", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
