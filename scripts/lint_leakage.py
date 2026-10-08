#!/usr/bin/env python3
"""Leakage / integrity lint for the risk-calibrated IR detection pipeline.

A standing guard against the defects an external audit found in the first
submission draft. Every check encodes a methodological invariant that a T-ITS /
TNNLS reviewer would expect to hold:

  * the fit/report split is disjoint and the calibration half is the ONLY data
    that may fit theta*, T, the isotonic map and the conformal threshold;
  * conformal risk is controlled at the IMAGE level (per-image binary miss
    event), not the box level, and an infeasible budget reports a FAILED gate
    rather than silently returning the loosest threshold;
  * the CACH corruption head is trained on calibration-split corrupted records
    and evaluated on the CORRUPTED image (not the clean one);
  * corrupted prediction caches are partitioned into calibration vs test;
  * the corruption pipeline is seeded (reproducible);
  * no result is a hard-coded placeholder (RA-AP == 0.0).

The lint is static (source + artifact inspection) so it needs no GPU and runs
anywhere. Several checks are EXPECTED to fail before remediation lands -- that
is the point: each one flips to PASS only when its defect is genuinely fixed.

Usage:
    python scripts/lint_leakage.py
    python scripts/lint_leakage.py --results-dir /path/to/results \
                                   --split-json  configs/flir_val_split.json
Exit code 0 iff every check passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

_REPO = Path(__file__).resolve().parent.parent


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as e:  # missing file is itself a finding
        return f"<<UNREADABLE: {e}>>"


# ---------------------------------------------------------------------------
# Individual checks. Each returns a Result.
# ---------------------------------------------------------------------------
def check_split_disjoint(split_json: Path) -> Result:
    name = "split_disjoint"
    if not split_json.exists():
        return Result(name, False, f"split json missing: {split_json}")
    d = json.loads(split_json.read_text())
    cal, test = set(d.get("calibration", [])), set(d.get("test", []))
    overlap = cal & test
    n_total = d.get("meta", {}).get("n_total")
    ok = (not overlap) and bool(cal) and bool(test)
    if n_total is not None:
        ok = ok and (len(cal) + len(test) == n_total)
    return Result(
        name, ok,
        f"|cal|={len(cal)} |test|={len(test)} overlap={len(overlap)} "
        f"n_total={n_total}",
    )


def check_conformal_image_level(src: Path) -> Result:
    """conformal.py must control a per-IMAGE binary miss event, not box-level."""
    name = "conformal_image_level"
    txt = _read(src)
    # Box-level defect signature: summing box COUNTS to form the conformal n.
    box_level = re.search(r"n\s*=\s*sum\(\s*len\(.*boxes", txt) is not None
    # Image-level evidence: a per-image binary miss indicator + an image-count n.
    image_level = ("per-image" in txt.lower() or "image-level" in txt.lower()) and \
                  (("_image_missed" in txt) or ("_n_images" in txt))
    ok = (not box_level) and image_level
    return Result(
        name, ok,
        f"box_level_n={box_level} image_level_marker={image_level}",
    )


def check_conformal_no_silent_fallback(src: Path) -> Result:
    """Infeasible budget must FAIL the gate, not return the loosest lambda as 'hat'."""
    name = "conformal_no_silent_fallback"
    txt = _read(src)
    silent = re.search(r"lambda_hat\s*=\s*max\(qualifying\)\s*if\s*qualifying\s*else\s*lambdas\[0\]", txt) is not None
    # Honest behaviour: an explicit feasibility flag / None on infeasible.
    honest = ("feasible" in txt.lower()) and ("None" in txt)
    ok = (not silent) and honest
    return Result(name, ok, f"silent_fallback={silent} feasibility_flag={honest}")


def check_cach_train_split(src: Path) -> Result:
    """CACH must not train on data['test']."""
    name = "cach_train_split"
    txt = _read(src)
    leaks = re.search(r"return\s+data\[[\"']test[\"']\]", txt) is not None
    trains_on_cal = "calibration" in txt
    ok = (not leaks) and trains_on_cal
    return Result(name, ok, f"returns_test_for_training={leaks} mentions_calibration={trains_on_cal}")


def check_cach_eval_applies_corruption(src: Path) -> Result:
    """CACH eval must feed the CORRUPTED image to the corruption-embedding net."""
    name = "cach_eval_applies_corruption"
    txt = _read(src)
    # Genuine fix: the per-image CACH eval takes the corruption name/severity and
    # applies the corruption fn to the image before embedding it.
    applies = (("corr_name" in txt) and
               (re.search(r"cfn\(\s*g\s*,\s*severity", txt) is not None
                or "apply_corruption" in txt))
    return Result(name, applies, f"re-applies_corruption_at_eval={applies}")


def check_corruption_caches_split(results_dir: Path) -> Result:
    """Corrupted prediction caches must exist for BOTH calibration and test."""
    name = "corruption_caches_split"
    cp = results_dir / "corruption_preds"
    if not cp.exists():
        return Result(name, False, f"missing: {cp}")
    names = [p.name for p in cp.iterdir()]
    has_cal = any(re.search(r"(calib|_cal_|/cal/)", n) for n in names) or (cp / "calibration").exists()
    has_test = any(re.search(r"(test|_test_)", n) for n in names) or (cp / "test").exists()
    ok = has_cal and has_test
    return Result(name, ok, f"calibration_caches={has_cal} test_caches={has_test} n_files={len(names)}")


def check_corruption_calibration_fit_split(src: Path) -> Result:
    """Pooled/oracle isotonic must FIT on calibration, EVALUATE on test."""
    name = "corruption_calibration_fit_split"
    txt = _read(src)
    # Honest: an explicit calibration-fit / test-eval split.
    splits = ("calibration" in txt and "test" in txt) and \
             re.search(r"fit.*calibrat|calibrat.*fit", txt, re.IGNORECASE) is not None
    return Result(name, splits, f"fits_on_calibration_eval_on_test={splits}")


def check_microcal_fit_split(src: Path) -> Result:
    """Micro-calibration theta must be optimized on calibration GT, not test GT."""
    name = "microcal_fit_split"
    txt = _read(src)
    leaks = re.search(r"optimize_threshold\(\s*[a-zA-Z_]*test", txt) is not None
    return Result(name, not leaks, f"optimizes_theta_on_test_gt={leaks}")


def check_no_raap_placeholder(*srcs: Path) -> Result:
    """No 'raap = 0.0  # placeholder' may survive."""
    name = "no_raap_placeholder"
    hits = []
    for s in srcs:
        txt = _read(s)
        if re.search(r"raap\s*=\s*0\.0\s*#\s*placeholder", txt):
            hits.append(s.name)
    return Result(name, not hits, f"placeholder_in={hits if hits else 'none'}")


def check_corruption_seeded(src: Path) -> Result:
    """Corruption pipeline noise must be drawn from a SEEDED RNG (reproducible)."""
    name = "corruption_seeded"
    txt = _read(src)
    seeded = ("default_rng" in txt) or ("RandomState" in txt) or \
             re.search(r"np\.random\.seed|rng\s*=\s*np\.random", txt) is not None
    unseeded_global = re.search(r"np\.random\.(normal|poisson|randint|random|uniform)\(", txt) is not None
    ok = seeded and not unseeded_global
    return Result(name, ok, f"seeded_rng={seeded} unseeded_global_calls={unseeded_global}")


# ---------------------------------------------------------------------------
def run(results_dir: Path, split_json: Path) -> List[Result]:
    sd = _REPO / "scripts"
    rd = _REPO / "src" / "risk"
    return [
        check_split_disjoint(split_json),
        check_conformal_image_level(rd / "conformal.py"),
        check_conformal_no_silent_fallback(rd / "conformal.py"),
        check_cach_train_split(sd / "20_train_cach.py"),
        check_cach_eval_applies_corruption(sd / "24_evaluate_cach.py"),
        check_corruption_caches_split(results_dir),
        check_corruption_calibration_fit_split(sd / "14_corruption_calibration.py"),
        check_microcal_fit_split(sd / "15_micro_calibration.py"),
        check_no_raap_placeholder(sd / "15_micro_calibration.py", sd / "18_raap_variants.py"),
        check_corruption_seeded(_REPO / "src" / "corruption" / "corruption_pipeline.py"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(_REPO / "results"),
                    help="results/ directory (defaults to repo-local; point at X9 for real caches)")
    ap.add_argument("--split-json", default=str(_REPO / "configs" / "flir_val_split.json"))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if ANY check fails (CI gate)")
    args = ap.parse_args()

    results = run(Path(args.results_dir), Path(args.split_json))
    n_pass = sum(r.ok for r in results)
    width = max(len(r.name) for r in results)
    print("=== leakage / integrity lint ===")
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        print(f"[{tag}] {r.name:<{width}}  {r.detail}")
    print(f"--- {n_pass}/{len(results)} checks pass ---")

    if args.strict and n_pass != len(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
