"""Deterministic 50/50 calibration/test partition of the LLVIP test set.

LLVIP has no separate val set (only train=12025 and test=3463).  The 3463
test images are split deterministically for the domain-transfer experiments:

    * calibration (50%) — fit theta* and recalibration temperature T on the
      *target* domain (leakage-free: never used for headline metrics)
    * test (50%)        — report mAP, ECE, E[C] for domain-transfer results

Single class (person only), so no class-presence stratification is needed;
images are sorted alphabetically then shuffled with a seeded RNG.

CLI
---
    python3 src/data/llvip_splits.py \
        --root /path/to/LLVIP-YOLO \
        --out  configs/llvip_val_split.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def make_llvip_split(
    llvip_yolo_root: str,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Return (calibration_stems, test_stems) for the LLVIP test split.

    Args:
        llvip_yolo_root: Path to LLVIP-YOLO directory (contains train/ and test/).
        calib_frac:      Fraction of test images assigned to calibration.
        seed:            RNG seed for reproducibility.

    Returns:
        Two sorted lists of image stem strings (no extension, no directory),
        e.g. ["010001", "010002", …].
    """
    img_dir = Path(llvip_yolo_root) / "test" / "lwir" / "images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"LLVIP test IR images not found: {img_dir}\n"
            "  Make sure LLVIP-YOLO is downloaded and symlinks are in place."
        )

    stems = sorted(
        p.stem for p in img_dir.glob("*.jpg") if not p.name.startswith(".")
    )
    if not stems:
        raise RuntimeError(f"No .jpg images found in {img_dir}")

    rng = random.Random(seed)
    stems_shuffled = stems[:]
    rng.shuffle(stems_shuffled)

    n_calib = round(len(stems_shuffled) * calib_frac)
    calibration = sorted(stems_shuffled[:n_calib])
    test = sorted(stems_shuffled[n_calib:])
    return calibration, test


def build_and_save(
    llvip_yolo_root: str,
    out_path: str,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> Dict:
    """Build the split, write JSON, and return the payload dict."""
    calibration, test = make_llvip_split(llvip_yolo_root, calib_frac, seed)

    payload = {
        "meta": {
            "seed": seed,
            "calib_frac": calib_frac,
            "n_total": len(calibration) + len(test),
            "n_calibration": len(calibration),
            "n_test": len(test),
            "domain": "LLVIP",
            "split_of": "LLVIP test set",
            "purpose": (
                "calibration: fit theta* and temperature T on target domain; "
                "test: report ALL headline domain-transfer metrics"
            ),
        },
        "calibration": calibration,
        "test": test,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"  total={payload['meta']['n_total']}  "
          f"calibration={len(calibration)}  test={len(test)}")
    return payload


def load_split(split_json: str, which: str) -> List[str]:
    """Return the list of image stems for 'calibration' or 'test'."""
    if which not in ("calibration", "test"):
        raise ValueError("which must be 'calibration' or 'test'")
    return json.loads(Path(split_json).read_text())[which]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Path to LLVIP-YOLO directory (contains train/ and test/)")
    ap.add_argument("--out", default="configs/llvip_val_split.json")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_and_save(args.root, args.out, args.calib_frac, args.seed)
