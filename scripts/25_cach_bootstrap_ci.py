"""Task A — Bootstrap 95% CIs on per-condition ECE values for YOLOv8m.

Inputs:
  results/cach_eval.csv  (24 yolov8m rows: 6 corruption types x 4 severities)

Outputs:
  results/cach_bootstrap_ci.csv   — mean, ci_lo, ci_hi per protocol
  results/cach_bootstrap_ci.json  — includes CACH-pooled gap CI + gap_ci_excludes_zero

Usage:
  python3 scripts/25_cach_bootstrap_ci.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
N_BOOT = 10_000
SEED = 42


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Percentile bootstrap CI for the mean. Resample conditions (rows)."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return lo, hi


def bootstrap_paired_gap_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Paired bootstrap CI for mean(a) - mean(b), resampling SAME indices."""
    n = len(a)
    assert len(b) == n, "a and b must have same length"
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_gaps = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(boot_gaps, 2.5))
    hi = float(np.percentile(boot_gaps, 97.5))
    return lo, hi


def main():
    csv_path = REPO / "results" / "cach_eval.csv"
    df = pd.read_csv(csv_path)
    yolo = df[df["model"] == "yolov8m"].reset_index(drop=True)
    assert len(yolo) == 24, f"Expected 24 yolov8m rows, got {len(yolo)}"

    protocols = {
        "nocal":  "ece_nocal",
        "clean":  "ece_clean",
        "pooled": "ece_pooled",
        "oracle": "ece_oracle",
        "cach":   "ece_cach",
    }

    rng = np.random.default_rng(SEED)

    print(f"Bootstrap CI (n_boot={N_BOOT}, 95% percentile CI, seed={SEED})")
    print(f"Dataset: {len(yolo)} YOLOv8m conditions (6 corruption types x 4 severities)\n")
    print(f"{'Protocol':<10} {'Mean ECE':>10} {'CI_lo':>10} {'CI_hi':>10}")
    print("-" * 44)

    results = {}
    for proto, col in protocols.items():
        vals = yolo[col].values.astype(np.float64)
        mean = float(vals.mean())
        lo, hi = bootstrap_mean_ci(vals, N_BOOT, rng)
        results[proto] = {"mean": round(mean, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
        print(f"{proto:<10} {mean:>10.6f} {lo:>10.6f} {hi:>10.6f}")

    # Paired gap: CACH minus pooled
    cach_vals   = yolo["ece_cach"].values.astype(np.float64)
    pooled_vals = yolo["ece_pooled"].values.astype(np.float64)
    gap_point   = float(cach_vals.mean() - pooled_vals.mean())
    gap_lo, gap_hi = bootstrap_paired_gap_ci(cach_vals, pooled_vals, N_BOOT, rng)
    # CI excludes 0 when both bounds have same sign
    gap_ci_excludes_zero = bool((gap_lo > 0 and gap_hi > 0) or (gap_lo < 0 and gap_hi < 0))

    print()
    print(f"CACH - Pooled gap:")
    print(f"  Point estimate: {gap_point:+.6f}")
    print(f"  95% CI:         [{gap_lo:+.6f}, {gap_hi:+.6f}]")
    print(f"  CI excludes 0:  {gap_ci_excludes_zero}")
    if gap_ci_excludes_zero:
        direction = "CACH is WORSE than pooled" if gap_point > 0 else "CACH is BETTER than pooled"
        print(f"  Interpretation: {direction} (statistically significant at 95%)")
    else:
        print(f"  Interpretation: Gap is NOT statistically significant at 95%")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = REPO / "results" / "cach_bootstrap_ci.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "mean", "ci_lo", "ci_hi"])
        w.writeheader()
        for proto, vals in results.items():
            w.writerow({"protocol": proto, **vals})
    print(f"\nSaved: {out_csv}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = REPO / "results" / "cach_bootstrap_ci.json"
    payload = {
        "protocols": results,
        "cach_minus_pooled": {
            "point_estimate":    round(gap_point, 6),
            "ci_lo":             round(gap_lo, 6),
            "ci_hi":             round(gap_hi, 6),
            "gap_ci_excludes_zero": gap_ci_excludes_zero,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
