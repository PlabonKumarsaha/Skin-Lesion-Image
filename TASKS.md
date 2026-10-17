# Task List

Tick items as they're done (`[~]` = partly done). Updated 2026-10-16 22:32. The full plan is in the proposal doc (link in [SUMMARY.md](SUMMARY.md)).

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
- [x] Phase 1b VLM runs: 23/23 jobs done, 0 failures (finished 2026-10-17 12:27)
- [x] VLM diagnostic summary across all sets ([experiments/vlm_summary.py](experiments/vlm_summary.py) → `results/vlm_summary.csv`)

## Phase 2: Faithfulness pilot

- [x] Download Derm7pt (2026-10-16)
- [x] Analysis-ready tables + 400 px caches ([scripts/prepare_concept_sets.py](scripts/prepare_concept_sets.py)) → `splits/derm7pt.csv`, `splits/ddi.csv`
- [x] Concept mapping: not needed for the pilot, since the JSON schema uses Derm7pt's own vocabulary ([experiments/faithfulness.py](experiments/faithfulness.py)). Free-text rationales still to be scored later
- [x] Concept precision/recall, self-consistency, cross-format agreement and templating on Derm7pt test (395 cases) for Qwen2.5-VL-7B + MedGemma 1.5 → `results/faithfulness_derm7pt.csv`
- [x] **Gate passed:** self-consistency 0.89–1.00 but concept F1 0.39–0.47 ("consistent but unfaithful"); Qwen is templated (3.8% distinct answers, AUROC 0.51)
- [x] Baselines for E2: "always cite all" (concept F1 0.404, so Qwen is no better) and zero-shot concept AUROC from image-text models ([experiments/concept_zero_shot.py](experiments/concept_zero_shot.py)): DermLIP-B/16 0.708 > MONET 0.676 > BiomedCLIP 0.662 > DermLIP-PanDerm 0.643, vs MedGemma 0.63 / Qwen 0.52–0.55
- [x] Bootstrap CIs for the faithfulness metrics (1,000 case resamples)
- [x] Image-dependence control ([experiments/blur_control.py](experiments/blur_control.py)): with the lesion blurred away, Qwen keeps 79% of features and 93% of diagnoses (still says melanoma 96%), so its explanations are not image-driven; yes/no AUROC barely drops (0.671→0.648). MedGemma's diagnosis is image-dependent (AUROC 0.706→0.517) but 62% of its features are unchanged

## Phase 3: Full runs

- [ ] All models, all seeds
- [~] E3 contamination audit
  - [x] Documentation audit: Derm1M = SCIN + MSKCC + YouTube/PubMed/edu/forums; PanDerm = mostly Australian/NZ clinics + ISIC2024. Neither reports deduplication against evaluation sets
  - [x] Own-dataset near-duplicates ([experiments/dedup_audit.py](experiments/dedup_audit.py)): pHash unreliable for dermoscopy, so DermLIP cosine ≥ 0.98 (visually calibrated). HAM: 29 cross-lesion-ID duplicate pairs, 10 test images (0.7%) leak across splits, ΔAUROC ≤ 0.0004 when removed. DDI: 5 internal duplicate pairs. Derm7pt↔HAM: none
  - [ ] **(You)** Accept the Derm1M dataset terms on HF; then download public/pubmed/edu (~11.6 GB) and search for copies of HAM/Derm7pt test images
  - [ ] Ask a clinician (or you) to confirm the 34 flagged pairs (`results/dedup_pairs.csv`, score ≥ 0.98)
- [~] E4 subgroup / fairness
  - [x] DDI malignant-vs-benign AUROC by Fitzpatrick group ([experiments/fairness_ddi.py](experiments/fairness_ddi.py)): V–VI is the worst group for 6/8 model×prompt combos (0.46–0.58, gaps 0.12–0.23); DermLIP-PanDerm is the exception (0.770 on V–VI)
  - [x] Concept detection by skin tone ([experiments/concepts_by_skintone.py](experiments/concepts_by_skintone.py)): no drop on V–VI (macro AUROC 0.68–0.74 vs 0.60–0.67 on I–II), so H4 is not supported. The dark-skin drop is in diagnosis, not concept detection
  - [x] HAM/BCN subgroups ([experiments/subgroups.py](experiments/subgroups.py)): age <40 worst on both (gaps 0.15 / 0.11); palms/soles worst site on BCN (0.654, gap 0.16); sex gap ≤ 0.04
- [ ] E5 triage metrics (calibration, deferral, decision curves)
- [~] External tests
  - [x] Download + verify BCN20000 (12,413 labelled) and PAD-UFES-20 (2,298); `splits/external.csv` + 400 px caches ([scripts/prepare_external.py](scripts/prepare_external.py))
  - [x] Eval code: [experiments/external_eval.py](experiments/external_eval.py) (9 fine-tuned checkpoints) + external sets in foundation_eval.py
  - [x] Ran all: 9 fine-tuned checkpoints + 5 foundation models (finished 14:00). BCN melanoma AUROC: DermLIP probes 0.82, fine-tuned 0.75–0.79, PanDerm-L 0.757
  - [x] Aligned image sets: excluding SCC changes fine-tuned external AUROC by ≤ 0.003 (`results/external/finetuned_aligned.csv`)
  - [ ] PAD-UFES-20 zero-shot 0.97 (DermLIP-PanDerm) is suspiciously high: include PAD in the Derm1M contamination check
  - [x] Dedup BCN20000 / PAD-UFES-20 vs HAM10000: 0 pairs at cosine ≥ 0.98 (`results/dedup_external_vs_ham.json`)

## Phase 4: Write-up

- [ ] Lock results; turn the proposal into the paper's Introduction and Methods
- [ ] Figures 1–5; CLAIM / TRIPOD+AI checklist
- [ ] Release code, splits, prompts
- [ ] Submit (first target: Computers in Biology and Medicine or IEEE JBHI)
