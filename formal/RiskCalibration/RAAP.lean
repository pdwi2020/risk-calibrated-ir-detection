import Mathlib

/-!
# Proposition 2 (novel) — properties of Risk-Adjusted Average Precision

Risk-Adjusted Average Precision (Eq. 6, redesigned) discounts mAP by the fraction
of the worst-case "detect-nothing" cost incurred at the cost-optimal threshold:

    RA-AP = mAP − ρ,     ρ = C̄(θ*) / (c_FN · ḡ),

where `C̄` is the mean per-image cost, `c_FN` the missed-detection cost, and `ḡ`
the mean number of ground-truth objects per image, so `c_FN · ḡ` is the per-image
cost of missing every object. Unlike the earlier `mAP − α·E[C]` form with an
arbitrary weight `α`, this metric carries no tuning constant. We machine-check the
three properties the paper relies on:

* **density-invariance** (the substantive structural property): jointly scaling the
  per-image cost and the object density by any `k ≠ 0` leaves RA-AP unchanged, so
  RA-AP is comparable across datasets / corruption conditions of different object
  density — it is genuinely a *per-object-normalised* risk, not a raw cost;
* **boundedness**: `ρ ≥ 0` gives `RA-AP ≤ mAP`, and whenever the cost does not exceed
  the worst case (`C̄ ≤ c_FN·ḡ`) we get `RA-AP ≥ mAP − 1`, so `RA-AP ∈ [mAP−1, mAP]`;
* **ranking reversal**: on a shared evaluation set (common `c_FN, ḡ`), RA-AP reverses
  the mAP ranking of two detectors *iff* their normalised cost gap exceeds the mAP
  gap — an exact divergence threshold.
-/

namespace RiskCalibration

/-- Risk-Adjusted Average Precision (Eq. 6):
`RA-AP = mAP − C̄ / (c_FN · ḡ)`, the cost-discounted average precision. -/
noncomputable def raAP (mAP cBar cFN gBar : ℝ) : ℝ := mAP - cBar / (cFN * gBar)

/-- **Proposition 2a (density-invariance).** Scaling the per-image cost `C̄` and the
object density `ḡ` by a common nonzero factor `k` leaves RA-AP unchanged: the metric
depends on cost *per object*, not on absolute object counts. This is what makes
RA-AP comparable across conditions with different densities. -/
theorem raAP_density_invariant {mAP cBar cFN gBar k : ℝ}
    (hk : k ≠ 0) (hcFN : cFN ≠ 0) (hg : gBar ≠ 0) :
    raAP mAP (k * cBar) cFN (k * gBar) = raAP mAP cBar cFN gBar := by
  unfold raAP
  have hscale : k * cBar / (cFN * (k * gBar)) = cBar / (cFN * gBar) := by
    field_simp
  rw [hscale]

/-- **Proposition 2b (discount, upper bound).** With non-negative cost and positive
weights, RA-AP never exceeds mAP — the risk term only ever subtracts. -/
theorem raAP_le_mAP {mAP cBar cFN gBar : ℝ}
    (hc : 0 ≤ cBar) (hcFN : 0 < cFN) (hg : 0 < gBar) :
    raAP mAP cBar cFN gBar ≤ mAP := by
  unfold raAP
  have hden : 0 < cFN * gBar := mul_pos hcFN hg
  have hρ : 0 ≤ cBar / (cFN * gBar) := div_nonneg hc (le_of_lt hden)
  linarith

/-- **Proposition 2c (lower bound).** When the cost does not exceed the worst-case
"detect-nothing" cost `c_FN·ḡ`, the risk term `ρ ≤ 1`, so `RA-AP ≥ mAP − 1`.
Together with `raAP_le_mAP` this gives `RA-AP ∈ [mAP−1, mAP]`. -/
theorem raAP_ge_mAP_sub_one {mAP cBar cFN gBar : ℝ}
    (hcFN : 0 < cFN) (hg : 0 < gBar) (hle : cBar ≤ cFN * gBar) :
    mAP - 1 ≤ raAP mAP cBar cFN gBar := by
  unfold raAP
  have hden : 0 < cFN * gBar := mul_pos hcFN hg
  have hρ : cBar / (cFN * gBar) ≤ 1 := (div_le_one hden).mpr hle
  linarith

/-- **Proposition 2d (RA-AP ranking reversal).** For two detectors evaluated on the
same set (shared `c_FN, ḡ`), RA-AP strictly reverses the mAP order
(`RA-AP_A > RA-AP_B`) exactly when the normalised cost gap exceeds the mAP gap. -/
theorem raAP_reversal {mAPA mAPB cBarA cBarB cFN gBar : ℝ} :
    raAP mAPA cBarA cFN gBar > raAP mAPB cBarB cFN gBar ↔
      (cBarB - cBarA) / (cFN * gBar) > mAPB - mAPA := by
  unfold raAP
  have key : (cBarB - cBarA) / (cFN * gBar)
      = cBarB / (cFN * gBar) - cBarA / (cFN * gBar) := by
    rw [div_sub_div_same]
  simp only [gt_iff_lt, key]
  constructor <;> intro h <;> linarith

/-- **Proposition 2e (order coincidence).** The complementary boundary: RA-AP
preserves the mAP order exactly when the normalised cost gap does not exceed the
mAP gap. With `raAP_reversal` this pins the divergence threshold precisely. -/
theorem raAP_preserves_order {mAPA mAPB cBarA cBarB cFN gBar : ℝ} :
    raAP mAPA cBarA cFN gBar ≤ raAP mAPB cBarB cFN gBar ↔
      (cBarB - cBarA) / (cFN * gBar) ≤ mAPB - mAPA := by
  unfold raAP
  have key : (cBarB - cBarA) / (cFN * gBar)
      = cBarB / (cFN * gBar) - cBarA / (cFN * gBar) := by
    rw [div_sub_div_same]
  rw [key]
  constructor <;> intro h <;> linarith

end RiskCalibration
