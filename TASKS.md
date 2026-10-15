# Task List

Tick items as they're done (`[~]` = partly done). Updated 2026-10-15 22:27. The full plan is in the proposal doc (link in [SUMMARY.md](SUMMARY.md)).

## Phase 0: Data foundation

- [x] Revise the proposal for a Q1 target (faithfulness + site shift + contamination)
- [x] Download HAM10000 v4.0 from Harvard Dataverse; verify MD5s
- [x] Extract images; check that metadata ↔ images match (10,015 / 10,015)
- [x] Site × class check ([scripts/check_ham_sites.py](scripts/check_ham_sites.py))
- [x] Decide the compute target → **Mac (M4, 24 GB, MPS)**
- [x] Set up the Python environment (`.venv`, Python 3.12, torch 2.14.1 + MPS, timm 1.0.30; `requirements.lock.txt`)
- [x] Build lesion-level splits → `splits/` ([scripts/make_splits.py](scripts/make_splits.py), seed 42)
  - [x] Random stratified 70/15/15 (7-class) → `splits/random.csv`
  - [x] Site shift: rosendahl ↔ vidir_modern (nv/mel/bkl/bcc + mel-vs-rest) → `splits/site_shift.csv`
  - [x] Exclude the 9 lesions that span more than one site from the site-shift splits
- [x] **(You)** DDI downloaded → `data/raw/ddi/` (656 images, verified)
- [x] **(You)** Derm7pt downloaded → `data/raw/derm7pt/` (1,011 cases, verified)
- [x] SkinCon: Fitzpatrick17k + DDI annotations → `data/raw/skincon/`; DDI joined → `splits/ddi_concepts.csv` (636 images, 8 usable concepts)
- [x] Update the proposal doc: Queensland = 22.6%; redesigned site-shift experiment (E1)
- [x] Cache images at 400×300 → `data/processed/ham10000_400` ([scripts/prepare_images.py](scripts/prepare_images.py))

## Phase 1: CNN/ViT baselines

- [x] Shared evaluation module ([skinbench/metrics.py](skinbench/metrics.py)): balanced accuracy, macro-F1, melanoma AUROC, sensitivity at 95% specificity (test ROC + val threshold), ECE, Brier, lesion-level bootstrap CIs
- [x] Training script ([experiments/train_supervised.py](experiments/train_supervised.py)): timm, weighted cross-entropy, AdamW + cosine, 20 epochs, 224 px, best epoch by val balanced accuracy
- [x] DINOv2 linear-probe script ([experiments/linear_probe.py](experiments/linear_probe.py))
- [x] Timing test: ConvNeXt-T 265 s/epoch on the full train set (M4, fp32; AMP gave <20%, not used)
- [ ] **Running:** resumable runner ([experiments/run_phase1.py](experiments/run_phase1.py)), started 2026-10-15 14:45. **12/30 jobs done, 0 failures**; ETA ~noon 2026-10-16. Log: `results/logs/phase1.log`
- [x] DINOv2-B linear probe × 3 splits (deterministic, so one run each)
- [x] Seed 0 of every model × 3 splits (done 22:15)
- [ ] Seed 1 of every model × 3 splits (running)
- [ ] Seed 2 of every model × 3 splits
- [x] Summary script ([experiments/summarize.py](experiments/summarize.py)) → `results/phase1_summary.md`
- [x] Paired bootstrap CI for the site gap ([experiments/site_gap.py](experiments/site_gap.py) → `results/site_gap.csv`)
- [ ] Rerun `summarize.py` + `site_gap.py` on all 3 seeds
- [ ] Check whether EfficientNet-B3 / DINOv2 are weak when trained on Queensland in all seeds (small train set, 1,362 images); if so, report it as a data-size limitation
- [ ] Decide on 5-fold lesion-grouped CV for E1 (only if the 3-seed CIs are still wide)
- [~] **Gate:** is the site gap non-trivial (ΔAUROC > 0.03, CI excludes 0)? **Seed 0: yes**, ≈ +0.09 for ConvNeXt-T and ViT-B/16 on both sites. Confirm with seeds 1–2, then update the proposal.

## Phase 1b: Foundation models + open VLMs (open weights only)

- [x] Install open_clip, transformers, mlx-vlm (core package versions unchanged)
- [x] Check access + licences for every open model (table in KNOWLEDGE.md)
- [x] BiomedCLIP loads (open_clip, tested on CPU)
- [ ] **(You)** Free Hugging Face login + accept terms for DermLIP and MedGemma (gated, auto-approved)
- [ ] Shared CLIP-family script: zero-shot + linear probe on frozen features (BiomedCLIP, MONET, DermLIP ×2)
- [ ] PanDerm-B/L: download the official checkpoints (Google Drive) + authors' model code → linear probe
- [ ] Open VLM script with a fixed JSON output schema (diagnosis + 7-point features): Qwen2.5-VL-7B (MLX), MedGemma-4B, LLaVA-Med v1.5
- [ ] Run all of Phase 1b once the GPU is free (~6–10 h)

## Phase 2: Faithfulness pilot

- [x] Download Derm7pt (2026-10-16)
- [x] Analysis-ready tables + 400 px caches ([scripts/prepare_concept_sets.py](scripts/prepare_concept_sets.py)) → `splits/derm7pt.csv`, `splits/ddi.csv`
- [ ] Concept synonym map (free text → 7-point criteria)
- [ ] Concept precision/recall for the open VLMs (Qwen2.5-VL-7B, MedGemma-4B, LLaVA-Med)
- [ ] **Gate:** is there a clear gap between accuracy and faithfulness?

## Phase 3: Full runs

- [ ] All models, all seeds
- [ ] E3 contamination audit (pHash + DINOv2 nearest-neighbour)
- [ ] E4 subgroup / fairness (DDI, SkinCon, age/sex/site)
- [ ] E5 triage metrics (calibration, deferral, decision curves)
- [ ] External tests: PAD-UFES-20, BCN20000 (deduplicated)

## Phase 4: Write-up

- [ ] Lock results; turn the proposal into the paper's Introduction and Methods
- [ ] Figures 1–5; CLAIM / TRIPOD+AI checklist
- [ ] Release code, splits, prompts
- [ ] Submit (first target: Computers in Biology and Medicine or IEEE JBHI)
