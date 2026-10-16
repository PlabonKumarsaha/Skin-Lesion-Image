# Project Summary

**Status (2026-10-16, 15:30):** Phase 1 complete (30/30 runs, site-shift gate passed). Phase 1b running: foundation models now, then open VLMs overnight. Phase 0–1b code on GitHub.

## What we're doing

We're writing **one Q1 journal paper** that benchmarks AI models for dermoscopic skin lesion classification. Your original plan (benchmark CLIP / BLIP-2 / BiomedCLIP on HAM10000) was too incremental for Q1, so we reframed it around three questions:

1. **Faithfulness.** When a VLM explains a diagnosis (e.g. "blue-white veil"), is that feature actually in the image? Scored against expert labels in Derm7pt and SkinCon.
2. **Site shift.** How much does performance drop going from Vienna to Queensland images within HAM10000 (and to external sets)?
3. **Contamination.** Are benchmark scores inflated because foundation models saw ISIC images during pretraining?

The paper also reports triage metrics (sensitivity at 95% specificity, calibration) and subgroup fairness.

**Working title:** *Do Dermoscopic Vision-Language Models See What They Say? Concept Faithfulness, Site-Shift Robustness and Pretraining Contamination in Skin Lesion AI*

## How we're working

Code first. Each phase ends with a gate, and its results feed back into the proposal.

`Phase 0 data → Phase 1 CNN/ViT baselines → 1b one VLM per family → 2 faithfulness pilot → 3 full runs → 4 write-up`

Estimated time: about 2.5–3 months part-time to a submittable draft.

## Key files

| File | What |
|---|---|
| [Proposal doc](https://claude.ai/code/artifact/ab2ba9eb-467b-4f39-9883-253db639b0b8) | Revised proposal (living document) |
| [KNOWLEDGE.md](KNOWLEDGE.md) | Your inputs, decisions, verified facts |
| [TASKS.md](TASKS.md) | Phase-by-phase checklist |
| [scripts/check_ham_sites.py](scripts/check_ham_sites.py) | Site × class analysis of HAM10000 |
| [data/raw/ham10000/](data/raw/ham10000) | Images, metadata, manifest with checksums |

## Progress log

| Date | Done |
|---|---|
| 2026-10-15 | Reviewed the original proposal; rated novelty ~4/10 as written. Wrote the revised proposal with a related-work comparison. |
| 2026-10-15 | Decided to start with CNN/ViT baselines. |
| 2026-10-15 | Downloaded HAM10000 v4.0 (checksums OK, 10,015 images). Found Queensland = 22.6% (not ~50%) and strong site–label confounding. Proposed rosendahl vs vidir_modern as the site-shift pair. |
| 2026-10-15 | Chose Mac (M4) for training; set up `.venv` (torch 2.14.1, MPS OK). Built `random.csv` and `site_shift.csv` splits (leakage-checked). |
| 2026-10-15 | Wrote the eval module, training, probe and runner scripts. Timing: ConvNeXt-T 265 s/epoch. Pilot DINOv2 probe (trained on Queensland): melanoma AUROC 0.79 in-site vs 0.69 cross-site, so a site gap is visible. Launched all Phase 1 runs. Proposal doc updated with the HAM site facts. |
| 2026-10-15 15:13 | First ConvNeXt-T run (trained on Queensland, 18 min): melanoma AUROC 0.869 in-site vs 0.772 on Vienna (gap +0.097). The DINOv2 probe trained on Vienna is *better* on Queensland (0.814 vs 0.778), so the gap is asymmetric; check whether it comes from the different negative-class mix. |
| 2026-10-15 15:54 | ConvNeXt-T site shift is symmetric: Qld→Vienna gap +0.097, Vienna→Qld gap +0.089 (melanoma AUROC). At the val-chosen threshold, Vienna-trained sensitivity drops 0.41 → 0.24 on Queensland. The probe's reverse gap (−0.036) looks probe-specific. TODO: paired bootstrap of the gap on a shared test set. 5/30 jobs done. |
| 2026-10-15 22:04 | Seed 0 almost done (11/30). Site gap of +0.08 to +0.10 melanoma AUROC in 5/6 fine-tuned runs (ConvNeXt, ViT, EffNet; both directions). Cross-site AUROC converges to 0.75–0.78 whatever the architecture. ECE rises cross-site for every model. Exception: EffNet-B3 trained on Queensland is weak in-site (0.775); check seeds 1–2. Random split: EffNet-B3 0.907, ConvNeXt-T 0.903, DINOv2 probe 0.846 melanoma AUROC. |
| 2026-10-15 22:15 | Seed 0 complete (12/30). ViT-B/16 random-split melanoma AUROC 0.912 (best). Paired bootstrap on the same test images: the melanoma AUROC gap is significant for ConvNeXt-T and ViT-B/16 on both sites (≈+0.09, CIs exclude 0) and for EffNet-B3 on Vienna (+0.117). Not significant: EffNet on Queensland, DINOv2 probe on Queensland. |
| 2026-10-16 | DDI received and extracted (656 images; FST groups 208/241/207; 171 malignant, only 21 melanomas, so use malignant-vs-benign for fairness). Derm7pt still pending. |
| 2026-10-16 | Derm7pt received: 1,011 cases (clinical + dermoscopic), 252 melanomas, all 7 checklist concepts labelled for every case. Ground truth for the faithfulness experiment (E2) is ready. |
| 2026-10-16 | SkinCon received (Fitzpatrick17k file from you; DDI file downloaded from the SkinCon site). DDI concepts joined: 636 images, 8 concepts with ≥30 positives. Erythema labelled 105× on FST I–II vs 40× on V–VI, an early H4 signal. All Phase 2 data now in hand. |
| 2026-10-16 03:25 | Phase 1b code: foundation_eval.py (BiomedCLIP, MONET pass smoke tests) and vlm_eval.py (Qwen2.5-VL-7B via MLX, ~15–20 s/image while training runs). First 3 Derm7pt answers from Qwen are near-identical "everything suspicious" templates (melanoma, p=0.8, atypical network, blue areas…) whatever the image: an early H1 signal, but also a prompt-design risk to control for. Phase 0–1 pushed to GitHub. |
| 2026-10-16 08:02 | Seeds 0–1 done. Site gap (paired, 2 seeds, melanoma AUROC): ConvNeXt +0.083 / +0.117, ViT +0.073 / +0.157 (Queensland / Vienna test images, all CIs exclude 0); EffNet +0.032 (n.s.) / +0.110. Random split: ConvNeXt 0.906, ViT 0.905, EffNet 0.894 melanoma AUROC. |
| 2026-10-16 10:30 | Phase 1b code complete: 5 foundation models (BiomedCLIP, MONET, DermLIP ViT-B/16, DermLIP-PanDerm via the authors' fork, PanDerm-L via the authors' code) + 2 open VLMs (Qwen2.5-VL-7B, MedGemma 1.5-4B; 4-bit MLX). VLM controls added. Pilot (6 Derm7pt images): Qwen calls everything melanoma in JSON mode, even with shuffled options; MedGemma's JSON answers separate melanoma from nevi 6/6, but its yes/no answer is "no" for all, including melanomas. Strong format-dependence (self-consistency) signal for E2. Phase 1b runner queued to start after Phase 1. |
| 2026-10-16 15:21 | **Phase 1 complete** (30/30, 0 failures). 3-seed random split (melanoma AUROC): ConvNeXt-T 0.903 ± 0.006, ViT-B/16 0.899 ± 0.012, EffNet-B3 0.887 ± 0.018, DINOv2 probe 0.846. Paired site gap: ConvNeXt +0.090 / +0.111, ViT +0.085 / +0.133, EffNet +0.042 (n.s.) / +0.119. Sensitivity at 95% specificity drops by up to 0.36 cross-site. Gate passed. |
| 2026-10-16 16:45 | Foundation models done (5/5). Linear probes reach 0.83–0.89 melanoma AUROC (MONET 0.888, DermLIP-B/16 0.887, PanDerm-L 0.886). **DermLIP ViT-B/16 shows no site gap** (+0.010 / +0.012, n.s.) vs +0.04–0.13 for fine-tuned CNN/ViTs and +0.161 for PanDerm-L on Vienna test images. Must rule out Derm1M ⊃ HAM10000 contamination (E3). Zero-shot best: DermLIP-PanDerm (0.88 on Derm7pt). Proposal doc updated with a Preliminary results section. |
