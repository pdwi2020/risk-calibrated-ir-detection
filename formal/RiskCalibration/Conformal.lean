import Mathlib

/-!
# Proposition 3 — the conformal threshold λ̂ is well-defined

The conformal risk-control selection rule (Eq. 5) picks

    λ̂ = max { λ ∈ Λ : (n·R̂(λ) + 1) / (n + 1) ≤ α }

from a finite candidate grid `Λ`, where `R̂(λ)` is the empirical miss-rate.  The
risk is controlled at the level of the *scene*: `R̂(λ)` is the per-image miss rate
(fraction of ground-truth-bearing images with at least one missed object), so the
results below are stated for an abstract monotone `R̂ : ℝ → ℝ` and apply verbatim to
that image-level instantiation.  Three structural facts make the rule well-posed,
*independently* of the probabilistic coverage bound `E[r(λ̂)] ≤ α` (cited from
Angelopoulos et al., 2022):

* feasibility is **downward-closed** — because `R̂` is monotone in `λ` (a more
  aggressive threshold misses at least as much), if `λ` is feasible then so is any
  smaller `μ`;
* when some candidate is feasible, `λ̂` is the **greatest** feasible threshold in
  `Λ` (the maximiser exists and is itself feasible); and
* when **no** candidate is feasible the gate genuinely fails — there is no `λ̂` to
  return, so no guarantee may be issued (the honest infeasibility semantics: the
  loosest threshold must *not* be passed off as `λ̂`).
-/

namespace RiskCalibration

open Finset
open scoped Classical

/-- The conformal feasibility test of Eq. (5): the plug-in upper bound
`(n·R̂(λ) + 1) / (n + 1)` is at most the target risk level `α`. -/
def Feasible (n α : ℝ) (Rhat : ℝ → ℝ) (lam : ℝ) : Prop :=
  (n * Rhat lam + 1) / (n + 1) ≤ α

/-- **Proposition 3a (downward closure).** With non-negative calibration count `n`
and monotone empirical risk `R̂`, feasibility propagates to every smaller
threshold: if `λ` passes the test and `μ ≤ λ`, then `μ` passes it too. -/
theorem feasible_downward_closed
    (Rhat : ℝ → ℝ) (n α : ℝ) (hn : 0 ≤ n) (hmono : Monotone Rhat)
    {lam μ : ℝ} (hμ : μ ≤ lam) (hfeas : Feasible n α Rhat lam) :
    Feasible n α Rhat μ := by
  unfold Feasible at hfeas ⊢
  have hR : Rhat μ ≤ Rhat lam := hmono hμ
  have hden : (0 : ℝ) < n + 1 := by linarith
  have hstep : (n * Rhat μ + 1) / (n + 1) ≤ (n * Rhat lam + 1) / (n + 1) := by
    rw [← sub_nonneg]
    have heq : (n * Rhat lam + 1) / (n + 1) - (n * Rhat μ + 1) / (n + 1)
        = (n * (Rhat lam - Rhat μ)) / (n + 1) := by
      rw [div_sub_div_same]
      congr 1
      ring
    rw [heq]
    apply div_nonneg _ (le_of_lt hden)
    exact mul_nonneg hn (by linarith)
  exact le_trans hstep hfeas

/-- **Proposition 3b (well-posed maximiser).** If the candidate grid `Λ` contains a
feasible threshold, then `λ̂ := max {λ ∈ Λ : Feasible λ}` exists, is itself
feasible, lies in `Λ`, and is the greatest feasible threshold in `Λ`. -/
theorem conformal_lambda_hat
    (Λ : Finset ℝ) (Rhat : ℝ → ℝ) (n α : ℝ)
    (hne : (Λ.filter (fun l => Feasible n α Rhat l)).Nonempty) :
    Feasible n α Rhat ((Λ.filter (fun l => Feasible n α Rhat l)).max' hne)
    ∧ ((Λ.filter (fun l => Feasible n α Rhat l)).max' hne) ∈ Λ
    ∧ (∀ l ∈ Λ, Feasible n α Rhat l →
        l ≤ (Λ.filter (fun l => Feasible n α Rhat l)).max' hne) := by
  have hmem : (Λ.filter (fun l => Feasible n α Rhat l)).max' hne
      ∈ Λ.filter (fun l => Feasible n α Rhat l) := Finset.max'_mem _ hne
  refine ⟨?_, ?_, ?_⟩
  · exact (Finset.mem_filter.mp hmem).2
  · exact (Finset.mem_filter.mp hmem).1
  · intro l hl hlf
    exact Finset.le_max' _ l (Finset.mem_filter.mpr ⟨hl, hlf⟩)

/-- **Proposition 3c (infeasibility ⇒ no guarantee).** If no candidate threshold in
`Λ` passes the feasibility test, then there is genuinely no feasible threshold: the
gate fails and no `λ̂` (hence no risk guarantee) may be issued. This is the honest
counterpart to `conformal_lambda_hat` — returning the loosest threshold as `λ̂` in
this case would advertise a guarantee that was never established. -/
theorem conformal_infeasible
    (Λ : Finset ℝ) (Rhat : ℝ → ℝ) (n α : ℝ)
    (hempty : Λ.filter (fun l => Feasible n α Rhat l) = ∅) :
    ∀ l ∈ Λ, ¬ Feasible n α Rhat l := by
  intro l hl hf
  have hmem : l ∈ Λ.filter (fun l => Feasible n α Rhat l) :=
    Finset.mem_filter.mpr ⟨hl, hf⟩
  rw [hempty] at hmem
  simp at hmem

end RiskCalibration
