# `formal/`: Machine-checked properties (Lean 4 + mathlib)

This directory contains the formal, kernel-checked proofs of the four structural
propositions stated in the paper

> **Risk-Calibrated Infrared Object Detection under Low Signal-to-Noise Ratio and Distribution Shift.**

Every theorem below is verified by the Lean 4 kernel against `mathlib`. There are
no `sorry`s and no axioms beyond Lean's standard logical foundations.

## What is proved

| Paper | Lean file | Statement |
|-------|-----------|-----------|
| **Prop. 1** (scene-level conformal selection) | `RiskCalibration/Conformal.lean` | For a non-decreasing empirical risk, the feasibility test `(n·R̂(λ)+1)/(n+1) ≤ α` is downward-closed (`feasible_downward_closed`). If some grid threshold is feasible, `λ̂ = max{λ ∈ Λ : feasible}` exists, is feasible, and is the greatest feasible threshold (`conformal_lambda_hat`). If none is feasible, no admissible threshold exists, so the gate fails and no guarantee is issued (`conformal_infeasible`). This is the *structural* half of the conformal rule; the probabilistic coverage bound `E[r(λ̂)] ≤ α` is cited from Angelopoulos et al. (2022). |
| **Prop. 2** (RA-AP) | `RiskCalibration/RAAP.lean` | With `RA-AP = mAP − C̄/(c_FN·ḡ)`: jointly scaling `C̄` and `ḡ` by any `k ≠ 0` leaves RA-AP unchanged (`raAP_density_invariant`); RA-AP lies in `[mAP − 1, mAP]` whenever `C̄ ≤ c_FN·ḡ` (`raAP_le_mAP`, `raAP_ge_mAP_sub_one`); and RA-AP reverses the mAP ranking of two detectors iff their normalised cost gap exceeds their mAP gap (`raAP_reversal`), with the complementary `≤` form (`raAP_preserves_order`). |
| **Prop. 3** (CACH order preservation) | `RiskCalibration/Calibration.lean` | The CACH recalibration map `p ↦ σ(T·logit p + b)` with `T = softplus(·) > 0` is strictly increasing on `(0,1)` (`cach_strictMonoOn`), hence order-preserving (`cach_orderPreserving`): CACH never reorders detections. |
| **Prop. 4** (corruption-metric fairness) | `RiskCalibration/Robustness.lean` | Dividing by a common positive baseline preserves the order of corrupted errors (`ceShared_orderFaithful`), so the shared-baseline corruption error ranks detectors faithfully; a concrete witness shows the self-relative mCE can invert that order (`ceSelf_can_invert`); and a K-fold mean lies between its smallest and largest fold value (`cvMean_mem_Icc`). |

`RiskCalibration/Cost.lean` holds the shared analytic facts (strict monotonicity
of the sigmoid and logit links). `RiskCalibration.lean` is the root that imports
everything, and `Audit.lean` prints the axioms of every theorem above.

## Building

The toolchain version is pinned in `lean-toolchain` and is installed automatically
by [`elan`](https://github.com/leanprover/elan).

```bash
cd formal
lake exe cache get   # download prebuilt mathlib (no local mathlib compile needed)
lake build           # builds RiskCalibration; must exit 0 with no errors
lake env lean Audit.lean
```

`Audit.lean` must report only `propext`, `Classical.choice` and `Quot.sound` (the
standard mathlib foundations) for every theorem, and never `sorryAx`.

## Note on scope

These are *structural and analytic* guarantees about the conformal selection rule,
the RA-AP metric, the calibration map and the corruption metric. They complement,
and do not replace, the empirical evaluation in the paper. The measure-theoretic
conformal coverage bound is out of scope here and is used as cited prior work.
