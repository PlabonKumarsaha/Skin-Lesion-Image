# Task List

Tick items as they're done. The full plan is in the proposal doc (link in [SUMMARY.md](SUMMARY.md)).

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
  - [ ] Consider 5-fold lesion-grouped CV for the site-shift experiment (Queensland test set has only 54 melanoma images)
- [ ] Request access to Derm7pt and DDI (start now; approval takes days)
- [ ] Update the proposal doc: Queensland = 22.6%; redesigned site-shift experiment (E1)

## Phase 1: CNN/ViT baselines

- [ ] Shared evaluation module: balanced accuracy, macro-F1, melanoma AUROC, sensitivity at 95% specificity, ECE, bootstrap CIs
- [ ] Training script (timm, weighted cross-entropy, augmentation, fixed seeds)
- [ ] EfficientNet-B3 × 3 splits × 3 seeds
- [ ] ConvNeXt-T × 3 splits × 3 seeds
- [ ] ViT-B/16 × 3 splits × 3 seeds
- [ ] DINOv2-B linear probe × 3 splits × 3 seeds
- [ ] Baseline results table + site-shift gap
- [ ] **Gate:** is the site gap non-trivial (ΔAUROC > 0.03, CI excludes 0)? Update the proposal accordingly.

## Phase 1b: One model per remaining family

- [ ] PanDerm linear probe
- [ ] BiomedCLIP zero-shot
- [ ] Qwen2.5-VL-7B zero-shot (fixed JSON output schema)

## Phase 2: Faithfulness pilot

- [ ] Download Derm7pt (after approval)
- [ ] Concept synonym map (free text → 7-point criteria)
- [ ] Concept precision/recall for Qwen2.5-VL-7B + one closed model
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
