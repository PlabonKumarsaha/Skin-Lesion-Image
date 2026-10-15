"""Phase 0 check: HAM10000 site (`dataset` column) x diagnosis mix, at image and lesion level.

Usage: python3 -I scripts/check_ham_sites.py data/raw/ham10000/HAM10000_metadata.csv
"""
import csv
import sys
from collections import Counter, defaultdict

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
QUEENSLAND = {"rosendahl"}


def main(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if "dataset" not in rows[0]:
        sys.exit("No `dataset` column: site split impossible with this metadata file.")

    images = Counter((r["dataset"], r["dx"]) for r in rows)
    lesions = defaultdict(set)
    for r in rows:
        lesions[(r["dataset"], r["dx"])].add(r["lesion_id"])
    sites = sorted({r["dataset"] for r in rows})

    print(f"{len(rows)} images, {len({r['lesion_id'] for r in rows})} lesions\n")
    print("Images per site x class")
    print(f"{'site':<14}" + "".join(f"{c:>7}" for c in CLASSES) + f"{'total':>8}{'mel %':>8}")
    for s in sites:
        n = [images[(s, c)] for c in CLASSES]
        print(f"{s:<14}" + "".join(f"{x:>7}" for x in n) + f"{sum(n):>8}{100 * n[1] / sum(n):>7.1f}%")

    print("\nLesions per site x class (unit for splitting)")
    for s in sites:
        n = [len(lesions[(s, c)]) for c in CLASSES]
        print(f"{s:<14}" + "".join(f"{x:>7}" for x in n) + f"{sum(n):>8}")

    print("\nGrouped for E1 (Queensland = rosendahl, Vienna = all other sources)")
    for name, keep in [("Queensland", lambda s: s in QUEENSLAND), ("Vienna", lambda s: s not in QUEENSLAND)]:
        n = [sum(images[(s, c)] for s in sites if keep(s)) for c in CLASSES]
        print(f"{name:<14}" + "".join(f"{x:>7}" for x in n) + f"{sum(n):>8}{100 * n[1] / sum(n):>7.1f}%")

    lesion_sites = defaultdict(set)
    for r in rows:
        lesion_sites[r["lesion_id"]].add(r["dataset"])
    print(f"\nLesions spanning >1 site: {sum(len(v) > 1 for v in lesion_sites.values())}")
    print("dx_type:", dict(Counter(r["dx_type"] for r in rows)))


if __name__ == "__main__":
    main(sys.argv[1])
