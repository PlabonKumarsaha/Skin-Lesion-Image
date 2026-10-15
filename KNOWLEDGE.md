# Knowledge Base

What you've told me, and facts we've verified. Add to it as new information comes in.
Last updated: 2026-10-15.

## 1. From you (goals, constraints, preferences)

| Topic | What you said |
|---|---|
| Goal | Publish **one paper in a Q1 journal**. Not a thesis. |
| Workflow | **Code first**, then use the benchmark results to improve the proposal. |
| First experiments | Start with **CNN / ViT baselines**, before any VLM. |
| Dataset version | Use the **latest** HAM10000 release. |
| Author email | pkumarsaha71@gmail.com |

### Original proposal (your draft, summarised)

- **Title:** *Skin Lesion Image Analysis: Benchmarking Vision, Language, and Multimodal AI Models for Melanoma and Skin Cancer Classification*
- **Primary RQ:** How effectively can VLMs classify and interpret skin lesion images and give real-time decision support, compared with traditional deep learning?
- **Secondary RQs:**
  - Generalisation to Australian-sourced imagery.
  - Demographic disparities (age, sex, skin tone).
  - Prompt engineering / RAG without retraining.
  - Reuse of the method for other tasks in the domain.
- **Datasets planned:** ISIC Archive, HAM10000 (main), BCN20000 (cross-dataset); Fitzpatrick17k and DDI optional for skin tone.
- **Models planned:** CLIP, BLIP-2, BiomedCLIP; zero-shot, prompt-engineered, RAG and LoRA settings.
- **Metrics planned:** accuracy, precision, recall, F1, AUROC, BLEU/ROUGE, interpretability.
- **Other components:** fairness and robustness analysis, a chat-based decision-support prototype, clinician validation.
- **Australian framing:** Australia has among the world's highest melanoma incidence (draft cites ~16,878 new cases in one year; **not verified**, check against AIHW).
- **Key references cited:** Esteva et al. CNNs [3]; HAM10000 [4]; BCN20000 [5]; BiomedCLIP [8]; GPT-4V/LLaVA melanoma [9]; PanDerm [10]; spurious cues [13]; skin-tone fairness [12].

## 2. Decisions made together

| Date | Decision | Why |
|---|---|---|
| 2026-10-15 | Reframe the paper around **faithfulness, site shift and contamination** | The original "benchmark CLIP/BLIP-2/BiomedCLIP" framing is too incremental for a Q1 journal |
| 2026-10-15 | Drop BLIP-2, BLEU/ROUGE, the chat interface and the clinician study from this paper | Scope limited to one paper; follow-up work |
| 2026-10-15 | Model families: A supervised CNN/ViT · B dermatology foundation models (PanDerm, DermLIP, MONET) · C legacy (BiomedCLIP, CLIP) · D multimodal LLMs | Each research question needs a contrast between families |
| 2026-10-15 | Run CNN/ViT baselines first | Fewest unknowns; checks the splits and metrics; gives reference numbers |
| 2026-10-15 | Download HAM10000 from Harvard Dataverse, not Kaggle | Only the Dataverse copy has the `dataset` (site) column |
| 2026-10-15 | Split by `lesion_id`, never by image | HAM10000 has several images per lesion; image-level splits leak |
| 2026-10-15 | Train on your **Mac** (M4, MPS) | Your choice; fine for the baselines |
| 2026-10-15 | Site-shift test design: train on site A, test on A-test and B-test | Both directions are scored on the same two test sets, so the gaps are directly comparable |

## 3. Verified facts

### HAM10000 (Harvard Dataverse, doi:10.7910/DVN/DBW86T)

- **Version:** v4.0, released 2023-02-07 (latest). **Licence:** CC BY-NC 4.0. Open download, no login.
- **Size:** 10,015 images, 7,470 lesions, 600×450 JPG.
- **Columns:** `lesion_id, image_id, dx, dx_type, age, sex, localization, dataset`.
- **Confirmation:** histo 5,340 · follow_up 3,704 · consensus 902 · confocal 69.

**Site × class (images):**

| Site | Total | nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|---|---|
| rosendahl (Queensland) | 2,259 | 803 | 342 | 490 | 296 | 295 | 3 | 30 |
| vidir_modern (Vienna) | 3,363 | 1,832 | 680 | 475 | 211 | 32 | 82 | 51 |
| vidir_molemax (Vienna) | 3,954 | 3,720 | 24 | 124 | 2 | 0 | 54 | 30 |
| vienna_dias (Vienna) | 439 | 350 | 67 | 10 | 5 | 0 | 3 | 4 |

**What this means for the paper:**

- Queensland is **22.6%** of HAM10000, not "roughly half" as the draft says. Correct this in the paper.
- 90% of akiec images are from Queensland. molemax is 94% nevi with 0.6% melanoma, so site and label are confounded.
- The cleanest site-shift pair is **rosendahl vs vidir_modern**, on nv/mel/bkl/bcc plus melanoma-vs-rest.
- **9 lesions span more than one site.** Exclude them from the site-shift splits.

### Splits (seed 42, lesion-level, leakage-checked)

- `splits/random.csv`: 7-class. Lesion counts: train 5,229 / val 1,120 / test 1,121 (images 6,981 / 1,532 / 1,502).
- `splits/site_shift.csv`: 5,112 images, 2,928 lesions.

| Site | Split | nv | mel | bkl | bcc |
|---|---|---|---|---|---|
| rosendahl | train / val / test | 569 / 118 / 116 | 237 / 51 / 54 | 342 / 74 / 74 | 214 / 43 / 39 |
| vidir_modern | train / val / test | 1,286 / 264 / 282 | 463 / 99 / 101 | 327 / 68 / 80 | 153 / 26 / 32 |

The Queensland test set has only **54 melanoma images**, so expect wide CIs (roughly ±0.05 AUROC). Cross-validation may be needed.

### Related work found (to read in full before citing)

- Cirone 2024 (JMIR Derm): GPT-4V 85% vs LLaVA 45% on melanoma.
- MONET (Nat Med 2024): 105,550 literature images.
- PanDerm (Nat Med 2025): 2M+ images, 11 institutions, 28 benchmarks.
- Derm1M/DermLIP (ICCV 2025): 1.03M image-text pairs.
- DermBench/DermEval (arXiv 2511.09195), DermEVAL (WACV 2026), DermaBench (arXiv 2601.14084).
- Skin-R1 (arXiv 2511.14900) and ConVLM (engrXiv).
- **Gap:** no study found that combines concept-level faithfulness, controlled site shift and ISIC contamination auditing.

## 4. Environment

- **Machine:** Apple M4 (10-core GPU), 24 GB RAM, ~769 GB free, macOS (Darwin 25.6).
- **Python env:** `.venv` (Python 3.12, via uv): torch 2.14.1 (MPS works), timm 1.0.30, scikit-learn 1.9.1, pandas 3.0.6. Exact versions in `requirements.lock.txt`.
- **Data location:** `data/raw/ham10000/` (images/, metadata CSV, manifest with MD5s).
- **Compute target:** Mac (MPS).

## 5. Open questions

- [ ] Verify the Australian melanoma incidence figure (AIHW).
- [ ] Access requests for Derm7pt and DDI (approval takes days).
- [ ] Budget for closed-model API runs (GPT / Gemini / Claude)?
