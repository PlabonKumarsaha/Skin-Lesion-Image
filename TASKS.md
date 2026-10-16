# Task List

Tick items as they're done (`[~]` = partly done). Updated 2026-10-16 15:30. The full plan is in the proposal doc (link in [SUMMARY.md](SUMMARY.md)).

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
- [x] Resumable runner ([experiments/run_phase1.py](experiments/run_phase1.py)): **30/30 jobs done, 0 failures** (finished 2026-10-16 15:21) Log: `results/logs/phase1.log`
- [x] DINOv2-B linear probe × 3 splits (deterministic, so one run each)
- [x] Seed 0 of every model × 3 splits (done 22:15)
- [x] Seed 1 of every model × 3 splits (done 07:56)
- [x] Seed 2 of every model × 3 splits
- [x] Summary script ([experiments/summarize.py](experiments/summarize.py)) → `results/phase1_summary.md`
- [x] Paired bootstrap CI for the site gap ([experiments/site_gap.py](experiments/site_gap.py) → `results/site_gap.csv`)
- [x] Rerun `summarize.py` + `site_gap.py` on all 3 seeds → `results/phase1_summary.md`, `results/site_gap.csv`
- [x] EfficientNet-B3 trained on Queensland: no significant gap on Queensland test images in 3 seeds (+0.042, CI −0.028 to 0.106). Report as a small-training-set limitation
- [x] 5-fold CV for E1: not needed (3-seed paired CIs exclude 0 for 4/6 model×site pairs)
- [x] **Gate passed:** 3-seed paired melanoma-AUROC gap: ConvNeXt +0.090 / +0.111, ViT +0.085 / +0.133, EffNet +0.042 (n.s.) / +0.119 (Queensland / Vienna test images)
- [x] Update the proposal doc with the Phase 1 results (new "Preliminary results" section)

## Phase 1b: Foundation models + open VLMs (open weights only)

- [x] Install open_clip, transformers, mlx-vlm (core package versions unchanged)
- [x] Check access + licences for every open model (table in KNOWLEDGE.md)
- [x] BiomedCLIP loads (open_clip, tested on CPU)
- [x] **(You)** Hugging Face login (`pkumarsaha71`) + access to DermLIP ×2 and MedGemma confirmed
- [ ] **(You)** Revoke the HF token pasted in chat (browser login doesn't need it)
- [x] DermLIP ViT-B/16 smoke test OK
- [x] DermLIP-PanDerm loads via the Derm1M fork (2 local patches); smoke test OK
- [x] Shared CLIP-family script ([experiments/foundation_eval.py](experiments/foundation_eval.py)): zero-shot + linear probe; BiomedCLIP, MONET, DermLIP ViT-B/16 smoke-tested
- [x] PanDerm-L: official checkpoint (1.3 GB, loaded weights-only) + authors' model code; linear probe smoke test OK (vision-only, so no zero-shot)
- [x] Open VLM script ([experiments/vlm_eval.py](experiments/vlm_eval.py)): Qwen2.5-VL-7B + MedGemma 1.5-4B (4-bit MLX)
- [ ] LLaVA-Med v1.5: postponed (no MLX build; needs the legacy LLaVA codebase). Add only if time allows
- [x] VLM prompt controls: `shuffled` variant (option order shuffled + "uncertain") and `yesno` variant (melanoma score = P(yes) from token log-probs)
- [x] Phase 1b runner ([experiments/run_phase1b.py](experiments/run_phase1b.py)): waits for Phase 1, then foundation models → VLMs on Derm7pt → HAM/DDI
- [x] Foundation models: 5/5 done (zero-shot + probes; probe predictions saved; paired site gap computed)
- [ ] **Running:** Phase 1b (started 15:22, 2026-10-16; foundation models ~1 h, then ~15 h of VLM inference, Derm7pt first). Log: `results/logs/phase1b.log`

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
