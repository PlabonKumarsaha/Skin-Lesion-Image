"""Phase 1b/2, family D: open multimodal LLMs answer a fixed JSON schema per image.

The schema asks for a diagnosis, a melanoma probability and the 7-point checklist features in Derm7pt's
own vocabulary, so faithfulness can later be scored directly against expert concept labels (E2).
Raw responses are cached per image (JSONL), so runs resume; parsing is separate from generation.

Prompt variants (controls for option-order echoing and for a usable melanoma score):
  schema    fixed option order, as written below
  shuffled  every option list shuffled per image (seeded by its key) and "uncertain" added to each feature
  yesno     "Is this lesion melanoma? yes/no": melanoma score = P(yes) from the first answer token

Usage:
  .venv/bin/python -I experiments/vlm_eval.py --model qwen25vl7b --set derm7pt_test --variant shuffled --limit 3
Sets: derm7pt_test, ham_random_test, ham_site_test, ddi
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinbench.data import ROOT  # noqa: E402

PROC = ROOT / "data/processed"
MODELS = {  # all 4-bit MLX builds of open-weight models, run locally
    "qwen25vl7b": ("mlx", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    "medgemma15_4b": ("mlx", "mlx-community/medgemma-1.5-4b-it-4bit"),
}
DX = ["melanoma", "melanocytic nevus", "benign keratosis", "basal cell carcinoma",
      "actinic keratosis / intraepithelial carcinoma", "vascular lesion", "dermatofibroma"]
FEATURES = {
    "pigment_network": ["absent", "typical", "atypical"],
    "streaks": ["absent", "regular", "irregular"],
    "pigmentation": ["absent", "diffuse regular", "diffuse irregular", "localized regular", "localized irregular"],
    "regression_structures": ["absent", "blue areas", "white areas", "combinations"],
    "dots_and_globules": ["absent", "regular", "irregular"],
    "blue_whitish_veil": ["absent", "present"],
    "vascular_structures": ["absent", "arborizing", "comma", "hairpin", "within regression", "wreath", "dotted", "linear irregular"],
}


def schema(rng=None):
    """JSON answer template; with `rng`, option lists are shuffled and "uncertain" added to each feature."""
    dx = list(DX)
    feats = {k: list(v) for k, v in FEATURES.items()}
    if rng:
        rng.shuffle(dx)
        for k in feats:
            feats[k] = feats[k] + ["uncertain"]
            rng.shuffle(feats[k])
    keys = list(feats)
    if rng:
        rng.shuffle(keys)
    return "{\n" + ",\n".join([
        f'  "diagnosis": one of {json.dumps(dx)}',
        '  "melanoma_probability": a number from 0 to 1',
        '  "features": {\n' + ",\n".join(f'    "{k}": one of {json.dumps(feats[k])}' for k in keys) + "\n  }",
        '  "rationale": one or two sentences naming the visible features that support the diagnosis',
    ]) + "\n}"


INTRO = {
    "dermoscopic": ("You are assisting with research on dermoscopy. Examine this dermoscopic image of a skin lesion "
                    "and report what you see using the 7-point checklist. Answer with JSON only, exactly in this format:\n"),
    "clinical": ("You are assisting with research on skin lesions. Examine this clinical photograph of a skin lesion. "
                 "Answer with JSON only, exactly in this format:\n"),
}
YESNO = {
    "dermoscopic": "Look at this dermoscopic image of a skin lesion. Is this lesion a melanoma? Answer with one word: yes or no.",
    "clinical": "Look at this clinical photograph of a skin lesion. Is this lesion malignant (skin cancer)? Answer with one word: yes or no.",
}


def prompt(variant, modality, key):
    if variant == "yesno":
        return YESNO[modality]
    rng = random.Random(f"{key}") if variant == "shuffled" else None
    return INTRO[modality] + schema(rng)


def image_set(name):
    """Return DataFrame with columns key, path, modality."""
    if name == "derm7pt_test":
        d = pd.read_csv(ROOT / "splits/derm7pt.csv").query("split == 'test'")
        return pd.DataFrame({"key": d.case_num.astype(str), "path": [PROC / p for p in d.derm_img], "modality": "dermoscopic"})
    if name == "ham_random_test":
        d = pd.read_csv(ROOT / "splits/random.csv").query("split == 'test'")
    elif name == "ham_site_test":
        d = pd.read_csv(ROOT / "splits/site_shift.csv").query("split == 'test'")
    elif name == "ddi":
        d = pd.read_csv(ROOT / "splits/ddi.csv")
        return pd.DataFrame({"key": d.DDI_ID.astype(str), "path": [PROC / p for p in d.img], "modality": "clinical"})
    else:
        raise ValueError(name)
    return pd.DataFrame({"key": d.image_id, "path": [PROC / "ham10000_400" / f"{i}.jpg" for i in d.image_id], "modality": "dermoscopic"})


class MLXModel:
    def __init__(self, repo):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        self.model, self.processor = load(repo)
        self._generate, self._template = generate, apply_chat_template

    def __call__(self, image, text, max_tokens=500):
        prompt = self._template(self.processor, self.model.config, text, num_images=1)
        out = self._generate(self.model, self.processor, prompt, image=[str(image)], max_tokens=max_tokens,
                             temperature=0.0, verbose=False)
        return getattr(out, "text", out)

    def yes_prob(self, image, text):
        """P(yes) vs P(no) from the first generated token's log-probabilities."""
        import mlx.core as mx
        from mlx_vlm import stream_generate
        prompt = self._template(self.processor, self.model.config, text, num_images=1)
        first = next(iter(stream_generate(self.model, self.processor, prompt, image=[str(image)], max_tokens=1, temperature=0.0)))
        lp = first.logprobs.astype(mx.float32)
        tok = getattr(self.processor, "tokenizer", self.processor)
        ids = {w: {t[0] for v in (w, w.capitalize(), " " + w, " " + w.capitalize()) if len(t := tok.encode(v, add_special_tokens=False)) == 1}
               for w in ("yes", "no")}
        mass = {w: float(mx.logsumexp(lp[mx.array(sorted(i))])) for w, i in ids.items()}
        m = max(mass.values())
        e = {w: 2.718281828459045 ** (v - m) for w, v in mass.items()}
        return e["yes"] / (e["yes"] + e["no"]), first.text


def parse(text):
    """Extract the JSON object from a response; None if it can't be parsed."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--variant", choices=["schema", "shuffled", "yesno"], default="schema")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    backend, repo = MODELS[a.model]
    imgs = image_set(a.set)
    if a.limit:
        imgs = imgs.sample(n=min(a.limit, len(imgs)), random_state=0)
    suffix = "" if a.variant == "schema" else f"__{a.variant}"
    out = ROOT / "results/vlm" / a.model / f"{a.set}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    todo = imgs[~imgs.key.isin(done)]
    print(f"{a.model} on {a.set}: {len(done)} cached, {len(todo)} to run", flush=True)
    if len(todo) == 0:
        return
    vlm = MLXModel(repo)
    with open(out, "a") as f:
        for i, r in enumerate(todo.itertuples(), 1):
            t = time.time()
            rec = {"key": r.key, "model": repo, "set": a.set, "variant": a.variant}
            if a.variant == "yesno":
                p_yes, first = vlm.yes_prob(r.path, prompt("yesno", r.modality, r.key))
                rec |= {"p_yes": p_yes, "first_token": first}
            else:
                text = vlm(r.path, prompt(a.variant, r.modality, r.key))
                rec |= {"response": text, "parsed": parse(text)}
            f.write(json.dumps(rec | {"sec": round(time.time() - t, 2)}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} ({time.time() - t:.1f}s last)", flush=True)


if __name__ == "__main__":
    main()
