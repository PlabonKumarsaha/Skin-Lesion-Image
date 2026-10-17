"""Paper figures (PNG, 300 dpi) and paper/numbers.json: every number the manuscript cites, read from results files.

Colour: one categorical slot per model family, fixed order (reference palette slots 1-3, validated for all-pairs use):
  A fine-tuned / self-supervised vision = blue, B dermatology foundation = orange, C general biomedical = aqua;
  open VLMs = neutral grey (+ direct labels). Ordinal Fitzpatrick groups use one sequential blue ramp.

Usage: .venv/bin/python -I experiments/make_figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
FAM = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = {"I-II": "#9ec5f0", "III-IV": "#2a78d6", "V-VI": "#0b3d7a"}
MODELS = {  # key -> (label, family)
    "convnext_tiny": ("ConvNeXt-T (FT)", "A"), "vit_base": ("ViT-B/16 (FT)", "A"),
    "efficientnet_b3": ("EfficientNet-B3 (FT)", "A"), "dinov2_probe": ("DINOv2-B (LP)", "A"),
    "panderm_l": ("PanDerm-L (LP)", "B"), "monet": ("MONET (LP)", "B"),
    "dermlip_vitb16": ("DermLIP ViT-B/16 (LP)", "B"), "dermlip_panderm": ("DermLIP-PanDerm (LP)", "B"),
    "biomedclip": ("BiomedCLIP (LP)", "C"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
N = {}


def ft_metric(model, key, test="test"):
    v = [json.load(open(f))["results"][test][key]["value"] for f in glob.glob(str(R / f"runs/random/{model}/seed*/metrics.json"))]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def fnd(model):
    return json.load(open(R / f"runs_foundation/{model}/metrics.json"))["results"]


# ---------- numbers: benchmark ----------
bench = {}
for m in ["convnext_tiny", "vit_base", "efficientnet_b3"]:
    bench[m] = {k: ft_metric(m, k) for k in ["mel_auroc", "mel_sens_at_spec95", "balanced_acc", "macro_f1", "ece"]}
d = json.load(open(R / "runs/random/dinov2_probe/seed0/metrics.json"))["results"]["test"]
bench["dinov2_probe"] = {k: (d[k]["value"], 0.0) for k in ["mel_auroc", "mel_sens_at_spec95", "balanced_acc", "macro_f1", "ece"]}
for m in ["panderm_l", "monet", "dermlip_vitb16", "dermlip_panderm", "biomedclip"]:
    p = fnd(m)["probe"]["random->random/test"]
    bench[m] = {k: (p[k]["value"], 0.0) for k in ["mel_auroc", "mel_sens_at_spec95", "balanced_acc", "macro_f1", "ece"]}
N["benchmark"] = bench

# ---------- Fig 1: paired site gap ----------
gap = pd.read_csv(R / "site_gap.csv").query("metric == 'mel_auroc'")
N["site_gap"] = {f"{r.model}|{r.test_site}": [r.gap, r.ci_low, r.ci_high, int(r.seeds)] for r in gap.itertuples()}
order = ["convnext_tiny", "vit_base", "efficientnet_b3", "dinov2_probe", "panderm_l", "monet", "biomedclip",
         "dermlip_panderm", "dermlip_vitb16"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
for ax, site, title in zip(axes, ["rosendahl", "vidir_modern"], ["Queensland test images", "Vienna test images"]):
    for i, m in enumerate(order):
        r = gap[(gap.model == m) & (gap.test_site == site)].iloc[0]
        c = FAM[MODELS[m][1]]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.gap, i, "o", ms=6, color=c, mec="white", mew=1.2)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Melanoma AUROC: in-site − cross-site model")
    ax.set_xlim(-0.12, 0.25)
axes[0].set_yticks(range(len(order)), [MODELS[m][0] for m in order])
axes[0].invert_yaxis()
handles = [plt.Line2D([], [], marker="o", ls="", color=FAM[f], label=l) for f, l in
           [("A", "Fine-tuned / self-supervised vision"), ("B", "Dermatology foundation"), ("C", "General biomedical")]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig1_site_gap.png", dpi=300)
plt.close(fig)

# ---------- Fig 2: HAM -> BCN20000 external drop ----------
ext = json.load(open(R / "external/finetuned.json"))
ext_rows = []
for m in ["convnext_tiny", "vit_base", "efficientnet_b3"]:
    b = [ext[f"{m}/seed{s}"]["bcn20000"]["mel_auroc"]["value"] for s in range(3)]
    sens = [ext[f"{m}/seed{s}"]["bcn20000"]["mel_sens_hamthr"]["value"] for s in range(3)]
    pmal = [ext[f"{m}/seed{s}"]["pad_ufes20"]["malignant_auroc"]["value"] for s in range(3)]
    ext_rows.append((m, bench[m]["mel_auroc"][0], float(np.mean(b)), float(np.mean(sens)), float(np.mean(pmal))))
for m in ["dermlip_vitb16", "dermlip_panderm", "panderm_l", "biomedclip", "monet"]:
    p = fnd(m)["probe"]
    ext_rows.append((m, p["random->random/test"]["mel_auroc"]["value"], p["random->bcn20000/all"]["mel_auroc"]["value"], None, None))
ext_df = pd.DataFrame(ext_rows, columns=["model", "ham", "bcn", "sens_hamthr", "pad_malignant"]).sort_values("bcn")
N["external"] = {r.model: {"ham": r.ham, "bcn": r.bcn, "sens_hamthr": r.sens_hamthr, "pad_malignant": r.pad_malignant}
                 for r in ext_df.itertuples()}
N["zero_shot"] = {m: {k: fnd(m)["zero_shot"][k]["mel_auroc"]["value"] for k in fnd(m)["zero_shot"] if "mel_auroc" in fnd(m)["zero_shot"][k]}
                  for m in ["biomedclip", "monet", "dermlip_vitb16", "dermlip_panderm"]}
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for i, r in enumerate(ext_df.itertuples()):
    c = FAM[MODELS[r.model][1]]
    ax.plot([r.bcn, r.ham], [i, i], color=c, lw=2, alpha=0.5)
    ax.plot(r.ham, i, "o", ms=6, mfc="white", mec=c, mew=1.6)
    ax.plot(r.bcn, i, "o", ms=6, color=c, mec="white", mew=1.2)
    ax.text(r.bcn - 0.006, i, f"{r.bcn:.3f}", ha="right", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(ext_df)), [MODELS[m][0] for m in ext_df.model])
ax.set_xlabel("Melanoma AUROC (open = HAM10000 test, filled = BCN20000)")
ax.set_xlim(0.64, 0.94)
fig.tight_layout()
fig.savefig(OUT / "fig2_external_bcn.png", dpi=300)
plt.close(fig)

# ---------- Fig 3: faithfulness ----------
f = pd.read_csv(R / "faithfulness_derm7pt.csv")
cz = json.load(open(R / "concept_zero_shot.json"))
blur = pd.read_csv(R / "blur_control.csv")
N["faithfulness"] = f.to_dict("records")
N["concept_zero_shot"] = cz
N["blur"] = blur.to_dict("records")
rows = []
for r in f[f.variant.isin(["schema", "shuffled"])].itertuples():
    lo, hi = [float(x) for x in r.concept_macro_auroc_ci.strip("[]").split(",")]
    name = {"qwen25vl7b": "Qwen2.5-VL-7B", "medgemma15_4b": "MedGemma 1.5-4B"}[r.model]
    rows.append((f"{name} · {'JSON' if r.variant == 'schema' else 'JSON shuffled'}", r.concept_macro_auroc, lo, hi, "D"))
for m, lab in [("biomedclip", "BiomedCLIP zero-shot"), ("monet", "MONET zero-shot"), ("dermlip_panderm", "DermLIP-PanDerm zero-shot"),
               ("dermlip_vitb16", "DermLIP ViT-B/16 zero-shot")]:
    rows.append((lab, cz[m]["macro_auroc"], None, None, MODELS[m][1]))
rows.sort(key=lambda x: x[1])
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
for i, (lab, v, lo, hi, fam) in enumerate(rows):
    if lo is not None:
        a.plot([lo, hi], [i, i], color=FAM[fam], lw=2)
    a.plot(v, i, "o", ms=6, color=FAM[fam], mec="white", mew=1.2)
a.axvline(0.5, color=MUTED, lw=0.9, ls="--")
a.text(0.504, len(rows) - 1, "chance", fontsize=7, color=MUTED, va="center")
a.set_yticks(range(len(rows)), [x[0] for x in rows])
a.set_xlabel("7-point criterion detection, macro AUROC")
a.set_title("(a) Feature detection vs expert labels", loc="left", fontsize=9, color=INK)
jb = blur[blur.prompt == "JSON (shuffled)"].set_index("model")
cats = ["Cited features\nunchanged", "Diagnosis\nunchanged"]
x = np.arange(len(cats))
for k, (m, lab, col) in enumerate([("qwen25vl7b", "Qwen2.5-VL-7B", "#52514e"), ("medgemma15_4b", "MedGemma 1.5-4B", "#a3a29c")]):
    vals = [jb.loc[m, "feature_agreement"], jb.loc[m, "diagnosis_agreement"]]
    bars = b.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
    for bx, v in zip(bars, vals):
        b.text(bx.get_x() + bx.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5, color=INK)
b.set_xticks(x, cats)
b.set_ylim(0, 1.32)
b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
b.set_ylabel("Share of answers unchanged after blurring")
b.set_title("(b) Blurred-lesion control", loc="left", fontsize=9, color=INK)
b.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
b.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_faithfulness.png", dpi=300)
plt.close(fig)

# ---------- Fig 4: DDI fairness ----------
fd = pd.read_csv(R / "fairness_ddi.csv")
N["fairness_ddi"] = fd.to_dict("records")
N["concepts_by_skintone"] = json.load(open(R / "concepts_by_skintone.json"))
N["subgroups"] = pd.read_csv(R / "subgroups.csv").to_dict("records")
fd = fd.sort_values("all")
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for i, (_, r) in enumerate(fd.iterrows()):
    vals = [r[g] for g in SEQ]
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, zorder=1)
    for g, v in zip(SEQ, vals):
        ax.plot(v, i, "o", ms=6.5, color=SEQ[g], mec="white", mew=1.2, zorder=2)
ax.axvline(0.5, color=MUTED, lw=0.9, ls="--")
lab = (fd.model.str.replace("dermlip_panderm", "DermLIP-PanDerm").str.replace("dermlip_vitb16", "DermLIP ViT-B/16")
       .str.replace("biomedclip", "BiomedCLIP").str.replace("monet", "MONET").str.replace("qwen25vl7b", "Qwen2.5-VL")
       .str.replace("medgemma15_4b", "MedGemma 1.5").str.replace(" (mel. prob.)", ""))
ax.set_yticks(range(len(fd)), lab)
ax.set_xlabel("Malignant vs benign AUROC on DDI")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=f"Fitzpatrick {g}") for g, c in SEQ.items()],
          frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig4_fairness_ddi.png", dpi=300)
plt.close(fig)

# ---------- remaining numbers ----------
N["vlm_summary"] = pd.read_csv(R / "vlm_summary.csv").to_dict("records")
N["dedup"] = json.load(open(R / "dedup_audit.json"))
N["dedup_external"] = json.load(open(R / "dedup_external_vs_ham.json"))
json.dump(N, open(ROOT / "paper/numbers.json", "w"), indent=1, default=float)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
