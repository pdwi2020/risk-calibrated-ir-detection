# `formal/` — Machine-checked properties (Lean 4 + mathlib)

This directory contains the formal, kernel-checked proofs of the three structural
properties stated in the paper

> **Risk-Calibrated Infrared Object Detection under Low-SNR and Distribution Shift.**

Every theorem below is verified by the Lean 4 kernel against `mathlib`; there are
no `sorry`s and no unproved axioms beyond Lean's standard logical foundations.

## What is proved

| Paper | Lean file | Statement |
|-------|-----------|-----------|
| **Prop. 1** (Eq. 7) | `RiskCalibration/Calibration.lean` | The CACH recalibration map `p ↦ σ(T·logit p + b)` with `T = softplus(·) > 0` is strictly increasing on `(0,1)` (`cach_strictMonoOn`), hence order-preserving (`cach_orderPreserving`): CACH never reorders detections. |
| **Prop. 2** (Eq. 6) | `RiskCalibration/RAAP.lean` | *(novel)* RA-AP reverses the mAP ranking iff the normalised expected-cost gap exceeds the mAP gap, `α·(E[C_B]−E[C_A])/N > mAP_B − mAP_A` (`raAP_reversal`); the complementary `≤` form (`raAP_preserves_order`) pins the divergence boundary exactly. |
| **Prop. 3** (Eq. 5) | `RiskCalibration/Conformal.lean` | The conformal threshold `λ̂` is well-defined: feasibility is downward-closed for monotone empirical risk (`feasible_downward_closed`), and when some candidate is feasible, `λ̂ = max{λ∈Λ : feasible}` exists, is feasible, and is the greatest feasible threshold (`conformal_lambda_hat`). This is the *structural* half of Eq. (5); the probabilistic coverage bound `E[r(λ̂)] ≤ α` is cited from Angelopoulos et al. (2022) and is left as future formalization work. |

`RiskCalibration/Cost.lean` holds the shared analytic facts (strict monotonicity
of the sigmoid and logit links). `RiskCalibration.lean` is the root that imports
everything.

## Building

The toolchain version is pinned in `lean-toolchain` and is installed automatically
by [`elan`](https://github.com/leanprover/elan).

```bash
cd formal
lake exe cache get   # download prebuilt mathlib (no local mathlib compile needed)
lake build           # builds RiskCalibration; must exit 0 with no errors
```

To confirm the development is axiomatically clean, the proofs are checked with
`#print axioms` (expected: `propext`, `Classical.choice`, `Quot.sound` only — the
standard mathlib foundations).

## Note on scope

These are *structural / analytic* guarantees about the calibration map, the RA-AP
metric, and the conformal selection rule. They complement — they do not replace —
the empirical evaluation in the paper. The measure-theoretic conformal coverage
bound is out of scope here and is used as cited prior work.
