"""Deterministic, leakage-free calibration/test partition of the FLIR val set.

Why this exists
---------------
The risk-optimal confidence threshold ``theta*`` (Eq. for argmin E[C]) and the
calibration temperature ``T`` MUST be fit on data that is disjoint from the data
used to report headline metrics. Fitting ``theta*`` on the same images it is
evaluated on is a leakage error that reviewers at T-ITS / TNNLS reject. This
module partitions the 1,144-image FLIR thermal-val set once, deterministically,
into:

    * ``calibration`` -- used to fit ``theta*`` and ``T`` (and conformal quantiles);
    * ``test``        -- used for every reported number (mAP, ECE, E[C], ...).

The partition is *stratified* by each image's class-presence signature over
{person, bike, car} so that the rare ``bike`` class is balanced across both
halves, then frozen to JSON for reproducibility.

CLI
---
    python3 src/data/splits.py \
        --root datasets/flir_adas_v2/FLIR_ADAS_v2 \
        --out  configs/flir_val_split.json \
        --calib-frac 0.5 --seed 42

The CLI uses only the standard library so it runs in any interpreter. The
``dataset_subset`` helper (for wiring the split into a torch ``FLIRDataset``)
imports torch lazily and is not needed by the CLI.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Mirrors FLIRDataset.KEEP_CLASSES (person=0, bike=1, car=2). Kept local so this
# module has zero heavy dependencies and can run as a standalone script.
KEEP_CLASS_NAMES: Tuple[str, ...] = ("person", "bike", "car")


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
def _load_val_index(
    root: Path,
) -> Tuple[Dict[int, str], Dict[int, Set[str]], Dict[int, Dict[str, int]]]:
    """Parse images_thermal_val/coco.json.

    Returns:
        id_to_fname:  image_id -> basename of the JPEG file
        present:      image_id -> set of kept class names appearing in the image
        per_counts:   image_id -> {class_name: annotation_count}
    """
    ann_file = root / "images_thermal_val" / "coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"FLIR val annotations not found: {ann_file}")

    coco = json.loads(ann_file.read_text())

    # category_id -> kept class name (look up by name so we are robust to id drift)
    catid_to_name: Dict[int, str] = {
        c["id"]: c["name"] for c in coco["categories"] if c["name"] in KEEP_CLASS_NAMES
    }

    id_to_fname: Dict[int, str] = {
        im["id"]: Path(im["file_name"]).name for im in coco["images"]
    }

    present: Dict[int, Set[str]] = defaultdict(set)
    per_counts: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {k: 0 for k in KEEP_CLASS_NAMES}
    )
    for a in coco["annotations"]:
        name = catid_to_name.get(a["category_id"])
        if name is not None:
            present[a["image_id"]].add(name)
            per_counts[a["image_id"]][name] += 1

    # Ensure every image is represented, even background-only frames.
    for iid in id_to_fname:
        present.setdefault(iid, set())
        if iid not in per_counts:
            per_counts[iid] = {k: 0 for k in KEEP_CLASS_NAMES}

    return id_to_fname, dict(present), dict(per_counts)


def make_split(
    root: str, calib_frac: float = 0.5, seed: int = 42
) -> Tuple[Dict[int, str], Dict[int, Dict[str, int]], List[int], List[int]]:
    """Build the stratified calibration/test partition.

    Stratification key = sorted tuple of class names present in an image. Within
    each stratum the image ids are sorted (determinism), shuffled with a seeded
    RNG, and split by ``calib_frac``. Sorting both the ids and the strata keys
    guarantees the same partition on every machine for a given seed.
    """
    id_to_fname, present, per_counts = _load_val_index(Path(root))

    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for iid in sorted(id_to_fname):
        key = tuple(sorted(present[iid]))
        groups[key].append(iid)

    rng = random.Random(seed)
    calibration: List[int] = []
    test: List[int] = []
    for key in sorted(groups):
        ids = groups[key][:]
        rng.shuffle(ids)
        n_calib = round(len(ids) * calib_frac)
        calibration.extend(ids[:n_calib])
        test.extend(ids[n_calib:])

    calibration.sort()
    test.sort()
    return id_to_fname, per_counts, calibration, test


def _class_totals(per_counts: Dict[int, Dict[str, int]], ids: List[int]) -> Dict[str, int]:
    tot = {k: 0 for k in KEEP_CLASS_NAMES}
    for iid in ids:
        for k, v in per_counts[iid].items():
            tot[k] += v
    return tot


def _n_empty(per_counts: Dict[int, Dict[str, int]], ids: List[int]) -> int:
    return sum(1 for iid in ids if sum(per_counts[iid].values()) == 0)


def build_and_save(
    root: str, out_path: str, calib_frac: float = 0.5, seed: int = 42
) -> Dict:
    """Build the split, write JSON + two filename lists, and return the payload."""
    id_to_fname, per_counts, calibration, test = make_split(root, calib_frac, seed)

    payload = {
        "meta": {
            "seed": seed,
            "calib_frac": calib_frac,
            "n_total": len(id_to_fname),
            "n_calibration": len(calibration),
            "n_test": len(test),
            "stratified_by": "class-presence signature over {person, bike, car}",
            "purpose": (
                "calibration: fit theta* and temperature T (and conformal "
                "quantiles); test: report ALL headline metrics"
            ),
        },
        "calibration": calibration,
        "test": test,
        "filenames": {str(iid): id_to_fname[iid] for iid in id_to_fname},
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    # Plain filename lists make it trivial to build YOLO-format val subsets later.
    (out.parent / "flir_val_calibration_images.txt").write_text(
        "\n".join(id_to_fname[i] for i in calibration) + "\n"
    )
    (out.parent / "flir_val_test_images.txt").write_text(
        "\n".join(id_to_fname[i] for i in test) + "\n"
    )

    return payload


# ----------------------------------------------------------------------------
# Integration helpers (used by training / eval code; torch imported lazily)
# ----------------------------------------------------------------------------
def load_split(split_json: str, which: str) -> Set[int]:
    """Return the set of image_ids belonging to ``which`` in {'calibration','test'}."""
    if which not in ("calibration", "test"):
        raise ValueError("which must be 'calibration' or 'test'")
    data = json.loads(Path(split_json).read_text())
    return set(data[which])


def dataset_subset(dataset, split_json: str, which: str):
    """Wrap a FLIRDataset(split='val') as a torch Subset restricted to ``which``.

    ``dataset.images`` is a list of (image_id, filename); we select the indices
    whose image_id falls in the requested split.
    """
    from torch.utils.data import Subset  # lazy: keep CLI dependency-free

    ids = load_split(split_json, which)
    idxs = [i for i, (iid, _) in enumerate(dataset.images) if iid in ids]
    return Subset(dataset, idxs)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _print_verification(payload: Dict, root: str) -> None:
    _, per_counts, calibration, test = make_split(
        root, payload["meta"]["calib_frac"], payload["meta"]["seed"]
    )
    c_tot = _class_totals(per_counts, calibration)
    t_tot = _class_totals(per_counts, test)

    print("\n=== Split verification (stratified, seed="
          f"{payload['meta']['seed']}) ===")
    print(f"{'':14s}{'images':>9s}{'empty':>8s}"
          + "".join(f"{n:>9s}" for n in KEEP_CLASS_NAMES))
    for name, ids, tot in (("calibration", calibration, c_tot),
                           ("test", test, t_tot)):
        print(f"{name:14s}{len(ids):>9d}{_n_empty(per_counts, ids):>8d}"
              + "".join(f"{tot[n]:>9d}" for n in KEEP_CLASS_NAMES))

    # Per-class balance drift: |calib - test| / total, lower is better.
    print(f"{'drift %':14s}{'':>9s}{'':>8s}", end="")
    for n in KEEP_CLASS_NAMES:
        total = c_tot[n] + t_tot[n]
        drift = 100.0 * abs(c_tot[n] - t_tot[n]) / total if total else 0.0
        print(f"{drift:>9.1f}", end="")
    print("\n(drift = |calib-test| / class total; < ~5% is well balanced)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Path to FLIR_ADAS_v2 dir (contains images_thermal_val/)")
    ap.add_argument("--out", default="configs/flir_val_split.json",
                    help="Output JSON path for the frozen split")
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="Fraction of val assigned to the calibration split")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    args = ap.parse_args()

    payload = build_and_save(args.root, args.out, args.calib_frac, args.seed)
    m = payload["meta"]
    print(f"Wrote {args.out}")
    print(f"  total={m['n_total']}  calibration={m['n_calibration']}  test={m['n_test']}")
    print(f"  filename lists: {Path(args.out).parent}/flir_val_"
          "{calibration,test}_images.txt")
    _print_verification(payload, args.root)


if __name__ == "__main__":
    main()
