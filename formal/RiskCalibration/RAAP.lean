import Mathlib

/-!
# Proposition 2 (novel) — the RA-AP ranking-reversal condition

Risk-Adjusted Average Precision (Eq. 6) is

    RA-AP = mAP − α · ( E[C] / N ).

The paper observes empirically that mAP and RA-AP rankings *coincide on clean data
but diverge under corruption and distribution shift*.  This file turns that
observation into an exact, machine-checked condition: for two detectors `A`, `B`,
RA-AP reverses the mAP ranking **iff** the normalised expected-cost gap exceeds the
mAP gap.  The proof is elementary real algebra — its value is that the divergence
threshold is now *exact* rather than anecdotal.
-/

namespace RiskCalibration

/-- Risk-Adjusted Average Precision: `RA-AP = mAP − α · (E[C] / N)` (Eq. 6). -/
noncomputable def raAP (mAP α EC N : ℝ) : ℝ := mAP - α * (EC / N)

/-- **Proposition 2 (RA-AP ranking reversal).**
For detectors `A`, `B` with dataset size `N > 0` and cost weight `α > 0`, the
RA-AP order strictly *reverses* the mAP order — `RA-AP_A > RA-AP_B` — exactly when
the normalised expected-cost gap exceeds the mAP gap:

    α · (E[C_B] − E[C_A]) / N  >  mAP_B − mAP_A. -/
theorem raAP_reversal
    {mAPA mAPB ECA ECB α N : ℝ} (_hN : 0 < N) (_hα : 0 < α) :
    raAP mAPA α ECA N > raAP mAPB α ECB N ↔
      α * (ECB - ECA) / N > mAPB - mAPA := by
  unfold raAP
  have key : α * (ECB - ECA) / N = α * (ECB / N) - α * (ECA / N) := by ring
  simp only [gt_iff_lt, key]
  constructor <;> intro h <;> linarith

/-- **Proposition 2 (order-coincidence form).**
The RA-AP order *preserves* the mAP order exactly when the normalised cost gap does
not exceed the mAP gap.  Taken with `raAP_reversal` this pins the divergence
boundary precisely: rankings agree iff `α·(E[C_B]−E[C_A])/N ≤ mAP_B − mAP_A`. -/
theorem raAP_preserves_order {mAPA mAPB ECA ECB α N : ℝ} :
    raAP mAPA α ECA N ≤ raAP mAPB α ECB N ↔
      α * (ECB - ECA) / N ≤ mAPB - mAPA := by
  unfold raAP
  have key : α * (ECB - ECA) / N = α * (ECB / N) - α * (ECA / N) := by ring
  rw [key]
  constructor <;> intro h <;> linarith

end RiskCalibration
