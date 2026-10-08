import Mathlib

/-!
# Proposition 4 (novel) — corruption-robustness metrics

The paper measures corruption robustness with a *shared-baseline* corruption error

    CE_shared(E_c) = E_c / b,     b = mean clean error across all detectors (b > 0),

instead of the conventional *self-relative* mean corruption error

    mCE_self(E_c, E_clean) = E_c / E_clean.

We machine-check the two facts the corruption-robustness analysis rests on:

* **order-faithfulness of the shared baseline** — with a common positive baseline,
  `CE_shared` ranks two detectors exactly by their absolute corrupted error `E_c`,
  so the comparison is fair and never reorders;
* **self-relative distortion** — the self-relative metric can *invert* that order:
  there are detectors where one has the strictly smaller corrupted error yet the
  strictly larger self-relative mCE, because dividing by each detector's own clean
  error flatters the weaker (higher-clean-error) detector. This is exactly the
  ranking reversal observed on FLIR (RT-DETR vs. RetinaNet).

We also record a stability fact for the `K`-fold cross-validation of Section
(K-fold): the cross-validated estimate is the mean of the per-fold estimates and
therefore lies between the smallest and largest fold value — no single fold
dominates the reported number.
-/

namespace RiskCalibration

open Finset

/-- Shared-baseline corruption error: corrupted error over a common baseline. -/
noncomputable def ceShared (Ec b : ℝ) : ℝ := Ec / b

/-- Self-relative corruption error (conventional mCE): corrupted error over the
detector's *own* clean error. -/
noncomputable def ceSelf (Ec Eclean : ℝ) : ℝ := Ec / Eclean

/-- **Proposition 4a (order-faithfulness).** With a common positive baseline `b`,
the shared-baseline error orders two detectors exactly by their absolute corrupted
error `E_c`: it never reorders them. -/
theorem ceShared_orderFaithful {E₁ E₂ b : ℝ} (hb : 0 < b) :
    ceShared E₁ b ≤ ceShared E₂ b ↔ E₁ ≤ E₂ := by
  unfold ceShared
  constructor
  · intro h
    have h2 := mul_le_mul_of_nonneg_right h hb.le
    rwa [div_mul_cancel₀ E₁ hb.ne', div_mul_cancel₀ E₂ hb.ne'] at h2
  · intro h
    gcongr

/-- **Proposition 4b (self-relative distortion).** The self-relative metric can
invert the true robustness order: there exist positive corrupted errors `E₁ < E₂`
(so detector 1 is strictly more robust in absolute terms) yet
`mCE_self` ranks detector 2 as the more robust one, because detector 2 has the
larger clean error (a weaker clean detector) inflating its denominator. -/
theorem ceSelf_can_invert :
    ∃ E₁ E₂ Eclean₁ Eclean₂ : ℝ,
      0 < E₁ ∧ 0 < E₂ ∧ 0 < Eclean₁ ∧ 0 < Eclean₂ ∧
      E₁ < E₂ ∧ ceSelf E₂ Eclean₂ < ceSelf E₁ Eclean₁ := by
  refine ⟨1, 2, 1, 10, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> norm_num [ceSelf]

/-- `K`-fold cross-validation estimate: the mean of the per-fold estimates. -/
noncomputable def cvMean {k : ℕ} (fold : Fin k → ℝ) : ℝ :=
  (∑ i, fold i) / k

/-- **Proposition 4c (K-fold stability).** The cross-validated estimate lies between
the smallest and largest per-fold estimate — it is a genuine average, so no single
fold dominates the reported number. -/
theorem cvMean_mem_Icc {k : ℕ} (hk : 0 < k) (fold : Fin k → ℝ)
    {lo hi : ℝ} (hlo : ∀ i, lo ≤ fold i) (hhi : ∀ i, fold i ≤ hi) :
    lo ≤ cvMean fold ∧ cvMean fold ≤ hi := by
  have hkR : (0 : ℝ) < (k : ℝ) := by exact_mod_cast hk
  have hcard : (Finset.univ : Finset (Fin k)).card = k := by simp
  constructor
  · rw [cvMean, le_div_iff₀ hkR]
    calc lo * k = ∑ _i : Fin k, lo := by rw [Finset.sum_const, hcard]; ring
      _ ≤ ∑ i, fold i := Finset.sum_le_sum (fun i _ => hlo i)
  · rw [cvMean, div_le_iff₀ hkR]
    calc ∑ i, fold i ≤ ∑ _i : Fin k, hi := Finset.sum_le_sum (fun i _ => hhi i)
      _ = hi * k := by rw [Finset.sum_const, hcard]; ring

end RiskCalibration
