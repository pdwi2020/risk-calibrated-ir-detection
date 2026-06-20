import Mathlib

/-!
# Foundations: the sigmoid and logit links, with strict monotonicity

Shared definitions used by the calibration results.  The confidence calibrators
in the paper are built from the logistic `σ` and its inverse `logit`; here we
record that both are strictly monotone on their domains, the single analytic
fact the calibration guarantee (Prop. 1) rests on.

The division steps are written with bedrock lemmas (`sub_pos`, `div_sub_div`,
`div_pos`) rather than the churn-prone `div_lt_div_iff` family.
-/

namespace RiskCalibration

open Real Set

/-- Logistic sigmoid `σ(x) = 1 / (1 + exp (-x))`. -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- Logit (inverse sigmoid) on `(0,1)`: `logit p = log (p / (1 - p))`. -/
noncomputable def logit (p : ℝ) : ℝ := Real.log (p / (1 - p))

/-- The sigmoid denominator is strictly positive. -/
lemma one_add_exp_pos (x : ℝ) : 0 < 1 + Real.exp (-x) := by positivity

/-- The sigmoid is strictly positive (a valid lower probability bound). -/
lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid; positivity

/-- `σ` is strictly increasing on all of `ℝ`. -/
lemma strictMono_sigmoid : StrictMono sigmoid := by
  intro a b hab
  have ha : (0 : ℝ) < 1 + Real.exp (-a) := one_add_exp_pos a
  have hb : (0 : ℝ) < 1 + Real.exp (-b) := one_add_exp_pos b
  have hexp : Real.exp (-b) < Real.exp (-a) := Real.exp_lt_exp.mpr (by linarith)
  unfold sigmoid
  rw [← sub_pos]
  have heq : 1 / (1 + Real.exp (-b)) - 1 / (1 + Real.exp (-a))
      = (Real.exp (-a) - Real.exp (-b)) / ((1 + Real.exp (-b)) * (1 + Real.exp (-a))) := by
    rw [div_sub_div _ _ hb.ne' ha.ne']
    congr 1
    ring
  rw [heq]
  apply div_pos (by linarith) (mul_pos hb ha)

/-- On `(0,1)` the odds map `p ↦ p / (1 - p)` is strictly increasing. -/
lemma strictMonoOn_odds : StrictMonoOn (fun p => p / (1 - p)) (Ioo (0 : ℝ) 1) := by
  intro a ha b hb hab
  show a / (1 - a) < b / (1 - b)
  have h1a : 0 < 1 - a := by have := ha.2; linarith
  have h1b : 0 < 1 - b := by have := hb.2; linarith
  rw [← sub_pos]
  have heq : b / (1 - b) - a / (1 - a) = (b - a) / ((1 - b) * (1 - a)) := by
    rw [div_sub_div _ _ h1b.ne' h1a.ne']
    congr 1
    ring
  rw [heq]
  exact div_pos (by linarith) (mul_pos h1b h1a)

/-- `logit` is strictly increasing on `(0,1)`. -/
lemma strictMonoOn_logit : StrictMonoOn logit (Ioo (0 : ℝ) 1) := by
  intro a ha b hb hab
  unfold logit
  have h1a : 0 < 1 - a := by have := ha.2; linarith
  have hoa : 0 < a / (1 - a) := div_pos ha.1 h1a
  have hodds : a / (1 - a) < b / (1 - b) := strictMonoOn_odds ha hb hab
  exact Real.log_lt_log hoa hodds

end RiskCalibration
