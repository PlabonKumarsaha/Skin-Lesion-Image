# Project Summary

**Status (2026-10-15):** Phase 0 nearly done. HAM10000 is verified, the Mac environment is ready (MPS) and the lesion-level splits are built. Next: evaluation module + training script, then the CNN/ViT baselines.

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
