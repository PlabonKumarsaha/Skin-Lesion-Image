"""E3 part 1: near-duplicate audit across our own evaluation sets (HAM10000, Derm7pt, DDI).

Calibration (2026-10-17, visual review of contact sheets): pHash is NOT reliable for dermoscopy (distance 0-2 pairs
were mostly different lesions sharing a centred-blob layout), so conclusions use the embedding signal only; pHash pairs
are still written out for transparency. Embedding pairs >= 0.98 were mostly the same lesion; 0.96-0.98 were not.

Two independent signals:
  pHash   64-bit perceptual hash of each cached image; Hamming distance <= HAM_MAX -> near-duplicate (catches copies/resizes)
  embed   cosine similarity of DermLIP ViT-B/16 image features >= COS_MIN (catches crops/recolours; features cached by
          foundation_eval.py)
Also checks HAM10000 against itself: near-duplicates under *different* lesion_ids would leak across our lesion-level splits.

Outputs results/dedup_audit.json (counts) and results/dedup_pairs.csv (pairs with distances, keys only).
Usage: .venv/bin/python -I experiments/dedup_audit.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
HAM_MAX = 6      # pHash Hamming distance (of 64 bits) counted as near-duplicate
COS_MIN = 0.98   # embedding cosine counted as near-duplicate (calibrated by visual review: <0.98 were different lesions)


def phash(path):
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def catalogue():
    ham = pd.read_csv(ROOT / "splits/random.csv")
    d7 = pd.read_csv(ROOT / "splits/derm7pt.csv")
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")
    rows = [("ham", i, l, s, PROC / "ham10000_400" / f"{i}.jpg") for i, l, s in zip(ham.image_id, ham.lesion_id, ham.split)]
    rows += [("derm7pt_derm", str(c), str(c), s, PROC / p) for c, s, p in zip(d7.case_num, d7.split, d7.derm_img)]
    rows += [("derm7pt_clinic", str(c), str(c), s, PROC / p) for c, s, p in zip(d7.case_num, d7.split, d7.clinic_img)]
    rows += [("ddi", str(i), str(i), "all", PROC / p) for i, p in zip(ddi.DDI_ID, ddi.img)]
    return pd.DataFrame(rows, columns=["dataset", "key", "group", "split", "path"])


def hamming_pairs(h, max_d):
    """All index pairs i<j with popcount(h_i xor h_j) <= max_d, chunked to bound memory."""
    out = []
    bits = lambda x: np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
    for a in range(0, len(h), 512):
        d = bits(h[a:a + 512, None] ^ h[None, :])
        i, j = np.nonzero(d <= max_d)
        keep = (i + a) < j
        out += list(zip(i[keep] + a, j[keep], d[i[keep], j[keep]]))
    return out


def main():
    cat = catalogue()
    with ProcessPoolExecutor(max_workers=4) as ex:
        h = np.array(list(ex.map(phash, cat.path, chunksize=64)), dtype=np.uint64)
    pairs = [("phash", i, j, float(d)) for i, j, d in hamming_pairs(h, HAM_MAX)]

    # embeddings: HAM (random split = all 10,015 images), Derm7pt dermoscopic test, DDI
    feat = ROOT / "results/features/dermlip_vitb16"
    emb_parts = []
    ham_ids = pd.read_csv(ROOT / "splits/random.csv")
    for s in ["train", "val", "test"]:
        ids = ham_ids[ham_ids.split == s].image_id.tolist()
        emb_parts.append((np.load(feat / f"random__{s}.npy"), [cat.index[(cat.dataset == "ham") & (cat.key == i)][0] for i in ids]))
    d7 = pd.read_csv(ROOT / "splits/derm7pt.csv")
    d7t = d7[(d7.split == "test") & d7.ham_dx.notna()]
    emb_parts.append((np.load(feat / "derm7pt__test.npy"),
                      [cat.index[(cat.dataset == "derm7pt_derm") & (cat.key == str(c))][0] for c in d7t.case_num]))
    ddi = pd.read_csv(ROOT / "splits/ddi.csv")
    emb_parts.append((np.load(feat / "ddi__all.npy"), [cat.index[(cat.dataset == "ddi") & (cat.key == str(i))][0] for i in ddi.DDI_ID]))
    E = np.concatenate([e for e, _ in emb_parts])
    idx = np.concatenate([i for _, i in emb_parts])
    for a in range(0, len(E), 1024):
        sim = E[a:a + 1024] @ E.T
        i, j = np.nonzero(sim >= COS_MIN)
        keep = (i + a) < j
        pairs += [("embed", idx[x + a], idx[y], float(sim[x, y])) for x, y in zip(i[keep], j[keep])]

    p = pd.DataFrame(pairs, columns=["signal", "i", "j", "score"])
    for side in "ij":
        p[f"{side}_dataset"] = cat.dataset.values[p[side]]
        p[f"{side}_key"] = cat.key.values[p[side]]
        p[f"{side}_group"] = cat.group.values[p[side]]
        p[f"{side}_split"] = cat.split.values[p[side]]
    same_group = (p.i_dataset == p.j_dataset) & (p.i_group == p.j_group)
    p = p[~same_group]  # same lesion / same Derm7pt case (e.g. its own clinical-dermoscopic pair is not a duplicate)
    p = p[~(p.i_dataset.str.startswith("derm7pt") & p.j_dataset.str.startswith("derm7pt") & (p.i_key == p.j_key))]
    p.drop(columns=["i", "j"]).to_csv(ROOT / "results/dedup_pairs.csv", index=False)

    def kind(r):
        a, b = sorted([r.i_dataset.split("_")[0], r.j_dataset.split("_")[0]])
        return f"{a}-{b}"
    p["kind"] = p.apply(kind, axis=1) if len(p) else []
    ham_leak = p[(p.signal == "embed") & (p.i_dataset == "ham") & (p.j_dataset == "ham") & (p.i_split != p.j_split)]
    summary = {
        "thresholds": {"phash_hamming_max": HAM_MAX, "embed_cosine_min": COS_MIN},
        "images": cat.dataset.value_counts().to_dict(),
        "pairs_by_kind_and_signal": p.groupby(["kind", "signal"]).size().rename("n").reset_index().to_dict("records"),
        "ham_cross_lesion_pairs_across_splits_embed": int(len(ham_leak)),
        "ham_cross_lesion_pairs_test_involved": int(((ham_leak.i_split == "test") | (ham_leak.j_split == "test")).sum()),
        "ham_test_images_with_cross_split_duplicate": int(pd.concat([ham_leak[ham_leak.i_split == "test"].i_key,
                                                                    ham_leak[ham_leak.j_split == "test"].j_key]).nunique()),
    }
    json.dump(summary, open(ROOT / "results/dedup_audit.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
