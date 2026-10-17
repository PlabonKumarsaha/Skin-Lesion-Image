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
| Git | Commit and push to GitHub after **each stage** (2026-10-17), not only each phase. Repo `PlabonKumarsaha/Skin-Lesion-Image` (public). (2026-10-16) |
| Manuscript privacy | Draft text (`paper/manuscript.md`, `.docx`) kept out of the public repo until submission; scripts, figures and aggregate numbers are committed. (2026-10-17, my default; change if you want it public) |
| Manuscript | **Word (.docx)**, first target **Computers in Biology and Medicine** (Elsevier; ~8,000 words; numbered references). (2026-10-17) |
| Models | **Open, free models only. No commercial/closed APIs** (GPT, Gemini, Claude): this is an academic paper. (2026-10-16) |

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
| 2026-10-15 | Full permission to proceed with the task list; ask only when something is ambiguous | Your instruction |
| 2026-10-16 | Get SkinCon's DDI annotations (not Fitzpatrick17k images) | Your choice: concept labels for the 656 DDI images we already have, for the skin-tone faithfulness test |
| 2026-10-16 | Family D = open multimodal LLMs only; closed APIs dropped | Your instruction; also fully reproducible, no API cost |
| 2026-10-15 | Baseline recipe: 224 px, fp32, AdamW, cosine with 1-epoch warmup, 20 epochs, batch 32, inverse-frequency weighted CE; select by val balanced accuracy. Models: convnext_tiny.fb_in22k_ft_in1k, efficientnet_b3.ra2_in1k, vit_base_patch16_224.augreg2_in21k_ft_in1k, DINOv2-B probe | Standard, comparable across models |
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

### DDI (Stanford AIMI; received 2026-10-16)

- **Files:** 656 clinical photos (PNG, ~230–460 px) + `ddi_metadata.csv` (`DDI_ID, DDI_file, skin_tone, malignant, disease`). Stored in `data/raw/ddi/`; zip MD5 in `MANIFEST.txt`.
- **Skin tone groups:** FST I–II (12): 208 · III–IV (34): 241 · V–VI (56): 207.
- **Labels:** 171 malignant / 485 benign; 78 disease labels.
- **Only 21 melanomas** (7 melanoma, 7 acral lentiginous, 6 in situ, 1 nodular). That's too few for per-skin-tone melanoma metrics, so use **malignant vs benign** per FST group, as the original DDI paper does.
- **Clinical photos, not dermoscopy.** This is a modality shift for HAM-trained models; report it separately.
- **Terms:** non-commercial research; **no redistribution, no derivatives, don't share the download link.** The paper can release code and results only.

### Derm7pt release_v0 (SFU; received 2026-10-16)

- **Files:** 1,011 cases, each with a clinical and a dermoscopic image (2,013 images; none missing). `meta/meta.csv` plus official train/valid/test index files (413 / 203 / 395). Stored in `data/raw/derm7pt/`; MD5 in `MANIFEST.txt`.
- **Diagnoses:** 20 labels; **252 melanomas** (all subtypes). Largest classes: Clark nevus 399, Reed/Spitz 79, SK 45, BCC 42.
- **Concept ground truth for E2 (every case labelled):**
  - pigment_network: absent / typical / atypical
  - streaks: absent / regular / irregular
  - pigmentation: absent / diffuse or localised, regular or irregular
  - regression_structures: absent / blue / white / combinations
  - dots_and_globules: absent / regular / irregular
  - blue_whitish_veil: absent / present (195 present)
  - vascular_structures: 8 types
  - Also: `seven_point_score`, diagnostic difficulty (low/medium/high), sex, location, management.
- **Licence:** CC BY-NC-ND 4.0; no redistribution. Images © Dr. Giuseppe Argenziano. Cite Kawahara et al., *IEEE JBHI* 2019.
- **Prepared (`splits/derm7pt.csv`):** official split kept. Diagnoses mapped to HAM classes (nv 575, mel 252, bkl 69, bcc 42, vasc 29, df 20; 24 miscellaneous/melanosis have no equivalent). 7-point checklist flags: atypical network 230, blue-white veil 195, atypical vascular 117, irregular streaks 251, irregular pigmentation 305, irregular dots/globules 448, regression 253.
- **Contamination note:** the images come from the Argenziano dermoscopy atlas, which may also appear in PubMed/atlas-derived pretraining corpora (e.g. Derm1M, MONET). Include in the E3 audit.

### SkinCon (received 2026-10-16)

- **Fitzpatrick17k annotations:** `data/raw/skincon/skincon_fitzpatrick17k.csv`, copied from `~/Downloads/image.csv` (identical to the official file); MD5 in `MANIFEST.txt`.
- **Contents:** 3,690 Fitzpatrick17k images (MD5-hash filenames), 48 binary *clinical-morphology* concepts (Erythema 2,139, Plaque 1,967, Papule 1,170, Brown 760, Scale 686, …). 460 rows are flagged "Do not consider", leaving 3,230 usable. 26 concepts have fewer than 50 positives.
- **No images:** these labels point to Fitzpatrick17k photos that must be fetched from atlas URLs.
- **DDI annotations** (`skincon_ddi.csv`, downloaded from the SkinCon site 2026-10-16): all 656 DDI images matched; 636 usable after 20 "do not consider" rows. Joined table: `splits/ddi_concepts.csv`.
- **Only 8 concepts have ≥30 positives on DDI** (Papule 410, Brown 363, Erythema 235, Plaque 168, Scale 103, Dome-shaped 67, Crust 53, Nodule 46). Use these 8 for E4.
- **Early signal for H4:** Erythema is labelled far less on FST V–VI (40) than I–II (105). This is a known visibility problem on darker skin and makes erythema a key concept for the faithfulness-by-skin-tone test.
- **Relevance:** these concepts are clinical (papule, plaque), not dermoscopic. Derm7pt stays the core of E2. SkinCon-DDI would let us test whether faithfulness drops on darker skin (H4).

### External test sets (downloaded 2026-10-17, checksums verified)

| Dataset | Source / licence | Images | Melanoma | Malignant | Notes |
|---|---|---|---|---|---|
| BCN20000 (labelled "train" split) | figshare 24140028, CC BY 4.0 | 12,413 dermoscopic | 2,857 | 6,097 | Barcelona; official test split has no labels. Includes 431 SCC (no HAM class) |
| PAD-UFES-20 | Mendeley zr7vgbcyr2, CC BY 4.0 | 2,298 smartphone clinical | **52** | 1,089 | Brazil; modality shift; use malignant AUROC as the primary metric |

Mapping to HAM classes: AK/ACK→akiec, BKL/SEK→bkl, NV/NEV→nv; SCC has no HAM class (counts for malignant only). Table: `splits/external.csv` (shareable, CC BY 4.0).

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

### Open model access (checked 2026-10-16)

| Model | Family | Source | Licence | Access |
|---|---|---|---|---|
| BiomedCLIP | C | HF `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT | Open; loads with open_clip (tested) |
| MONET | B | HF `suinleelab/monet` (safetensors) | see repo | Open |
| PanDerm-L / PanDerm-B | B | Google Drive (official repo links); needs the authors' code | CC BY-NC-ND 4.0 | Open download |
| DermLIP ViT-B/16, DermLIP-PanDerm | B | HF `redlessone/...` | CC BY-NC 4.0 | Gated: **access granted** 2026-10-16 (HF user `pkumarsaha71`). DermLIP-PanDerm needs the Derm1M fork of open_clip |
| Qwen2.5-VL-7B | D | HF `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` (MLX, fast on Mac) | Apache-2.0 | Open |
| LLaVA-Med v1.5 | D | HF `microsoft/llava-med-v1.5-mistral-7b` | Apache-2.0 | Open |
| MedGemma-4B | D | HF `google/medgemma-4b-it` | Health AI Developer Foundations terms | Gated: **access granted** 2026-10-16 |

## 4. Environment

- **Machine:** Apple M4 (10-core GPU), 24 GB RAM, ~769 GB free, macOS (Darwin 25.6).
- **Python env:** `.venv` (Python 3.12, via uv): torch 2.14.1 (MPS works), timm 1.0.30, scikit-learn 1.9.1, pandas 3.0.6. Exact versions in `requirements.lock.txt`.
- **Data location:** `data/raw/ham10000/` (images/, metadata CSV, manifest with MD5s).
- **Compute target:** Mac (MPS).

- **DermLIP-PanDerm loading:** via the Derm1M fork in `third_party/Derm1M/src` (loaded only in its own process). Local patches: (1) skip the separate PanDerm init file, since the HF checkpoint has the full weights; (2) `tokenizer.py:441` uses `tokenizer(...)` because `batch_encode_plus` was removed in transformers 5. Also needs `setuptools<80` (pkg_resources).
- **Contamination flag:** DermLIP-PanDerm zero-shot reaches 0.89 melanoma AUROC on the Derm7pt smoke set. Check whether Derm1M's "two public datasets" include Derm7pt or HAM10000 (E3).
- **VLMs:** 4-bit MLX builds (`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`, `mlx-community/medgemma-1.5-4b-it-4bit`). State the quantisation in the paper. ~15–20 s/image for JSON answers, ~2 s for yes/no (while training shares the GPU). LLaVA-Med postponed (no MLX build).
- **Derm1M training sources (paper, arXiv 2503.14911):** "two public datasets including SCIN and MSKCC", plus YouTube, forums, PubMed (566k articles) and educational material. HAM10000, Fitzpatrick17k and PAD-UFES-20 are used only for evaluation; DDI and BCN20000 are not mentioned. **No deduplication against evaluation sets is reported.** MSKCC is ISIC-hosted, and PubMed figures may reproduce HAM10000/Derm7pt images, so near-duplicate checks are still needed (E3).
- **PanDerm pretraining (arXiv 2410.15038):** 2.15M images: MYM/HOP cohorts (Australian TBP + dermoscopy), MMT (150+ clinics, Australia/NZ), NSSI (Brisbane), Edu1/2, ISIC2024, TCGA-SKCM, UAH89k. Mostly Australian, which fits its zero gap on Queensland but +0.161 on Vienna test images. HAM/BCN/Derm7pt/DDI/PAD not listed; no dedup procedure described. Open question: is Rosendahl's practice among the MMT clinics?
- **Dedup audit (2026-10-17):** pHash fails on dermoscopy (centred-blob layout gives false matches). DermLIP-B/16 cosine ≥ 0.98 finds true duplicates (visual review; ~5 of 34 flagged pairs looked like different lesions). HAM10000 has same-lesion images under different `lesion_id`s, but only 10 test images are affected and the effect is negligible.
- **Secrets:** HF login is stored in `~/.cache/huggingface/` (outside the repo). `.git/hooks/pre-commit` blocks commits containing token-like strings.

## 5. Open questions

- [ ] Verify the Australian melanoma incidence figure (AIHW).
- [x] DDI received (2026-10-16).
- [x] Derm7pt received (2026-10-16).
