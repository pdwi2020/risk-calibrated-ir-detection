import RiskCalibration
import RiskCalibration.Robustness

open RiskCalibration

/-
  Axiom audit. Each proposition should depend only on Lean/mathlib's standard
  foundational axioms: `propext`, `Classical.choice`, `Quot.sound`.
-/
#print axioms cach_strictMonoOn
#print axioms cach_orderPreserving
#print axioms raAP_density_invariant
#print axioms raAP_le_mAP
#print axioms raAP_ge_mAP_sub_one
#print axioms raAP_reversal
#print axioms raAP_preserves_order
#print axioms feasible_downward_closed
#print axioms conformal_lambda_hat
#print axioms conformal_infeasible
#print axioms ceShared_orderFaithful
#print axioms ceSelf_can_invert
#print axioms cvMean_mem_Icc
