import RiskCalibration.Cost

/-!
# Proposition 1 — the CACH recalibration map is order-preserving

The Corruption-Adaptive Calibration Head (CACH, Eq. 7) maps a raw confidence `p`
to a recalibrated score

    p̂ = σ( T(e) · logit p + b(e) )

with a strictly positive, image-conditioned temperature `T(e) = softplus(·) > 0`
and shift `b(e)`.  Here we fix the embedding `e` (hence fixed scalars `T, b`) and
prove the map is **strictly increasing** on `(0,1)`.  Consequence: CACH never
reorders detections, so thresholding the calibrated score is equivalent to
thresholding the raw score at a transformed cut-off — a decision-order-preserving
safety invariant.
-/

namespace RiskCalibration

open Real Set

/-- Softplus `ζ(z) = log(1 + exp z)`, the CACH temperature link; strictly positive. -/
noncomputable def softplus (z : ℝ) : ℝ := Real.log (1 + Real.exp z)

/-- The CACH temperature is strictly positive (so the map below is well-posed). -/
lemma softplus_pos (z : ℝ) : 0 < softplus z := by
  unfold softplus
  apply Real.log_pos
  have := Real.exp_pos z
  linarith

/-- The CACH recalibration map for a fixed embedding: `p ↦ σ(T · logit p + b)`. -/
noncomputable def cach (T b : ℝ) (p : ℝ) : ℝ := sigmoid (T * logit p + b)

/-- **Proposition 1.** For any positive temperature `T` and any shift `b`, the CACH
recalibration map is strictly increasing on `(0,1)`. -/
theorem cach_strictMonoOn {T b : ℝ} (hT : 0 < T) :
    StrictMonoOn (cach T b) (Ioo (0 : ℝ) 1) := by
  intro a ha c hc hac
  unfold cach
  apply strictMono_sigmoid
  have hlogit : logit a < logit c := strictMonoOn_logit ha hc hac
  have hmul : T * logit a < T * logit c := mul_lt_mul_of_pos_left hlogit hT
  linarith

/-- A `softplus` temperature is always admissible for Proposition 1. -/
theorem cach_softplus_strictMonoOn (z b : ℝ) :
    StrictMonoOn (cach (softplus z) b) (Ioo (0 : ℝ) 1) :=
  cach_strictMonoOn (softplus_pos z)

/-- **Order preservation (Prop. 1, corollary).** On `(0,1)` the calibrated scores
compare exactly as the raw confidences do: CACH never reorders two detections. -/
theorem cach_orderPreserving {T b : ℝ} (hT : 0 < T) {p q : ℝ}
    (hp : p ∈ Ioo (0 : ℝ) 1) (hq : q ∈ Ioo (0 : ℝ) 1) :
    cach T b p < cach T b q ↔ p < q :=
  (cach_strictMonoOn hT).lt_iff_lt hp hq

end RiskCalibration
