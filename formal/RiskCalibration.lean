/-
  RiskCalibration — machine-checked properties accompanying

    "Risk-Calibrated Infrared Object Detection under Low-SNR and
     Distribution Shift"

  Root module. Re-exports the three formal results stated in the paper:

    * `RiskCalibration.Calibration` — Prop. 1: the CACH recalibration map is
      strictly monotone (order-preserving), so it never reorders detections.
    * `RiskCalibration.RAAP`        — Prop. 2 (novel): exact condition under
      which the Risk-Adjusted Average Precision (RA-AP) ranking reverses the
      mAP ranking.
    * `RiskCalibration.Conformal`   — Prop. 3: the conformal threshold λ̂ from
      Eq. (5) is well-defined — the largest element of a downward-closed
      feasible set (the *structural* guarantee; the probabilistic coverage
      bound E[r(λ̂)] ≤ α is cited from Angelopoulos et al., 2022).

  All proofs are `sorry`-free and kernel-checked. See `README.md`.
-/
import RiskCalibration.Cost
import RiskCalibration.Calibration
import RiskCalibration.RAAP
import RiskCalibration.Conformal
import RiskCalibration.Robustness
