"""Temperature scaling + detection ECE for confidence calibration.

Dependency-free (pure Python) so it runs anywhere on cached (confidence, is_TP)
arrays. Temperature scaling is applied in LOGIT space — the statistically correct
form for binary calibration — so calibrated scores always stay in (0, 1):

    p_cal = sigmoid( logit(conf) / T ),    logit(p) = log(p / (1 - p))

T is fit by minimizing the negative log-likelihood of the true-positive labels on
a held-out CALIBRATION split (never the test split). T > 1 softens overconfident
scores; T < 1 sharpens. Detection ECE bins predictions by confidence and measures
the gap between mean confidence and empirical precision (fraction of TPs).
"""

import math
from typing import List, Sequence


def _sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


class DetectionTemperatureScaling:
    """Post-hoc temperature calibration of per-detection confidence scores."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    # ------------------------------------------------------------------
    def transform(self, confidences: Sequence[float], temperature: float = None) -> List[float]:
        """Apply calibration p_cal = sigmoid(logit(conf) / T)."""
        T = self.temperature if temperature is None else float(temperature)
        T = max(T, 1e-3)
        return [_sigmoid(_logit(float(c)) / T) for c in confidences]

    @staticmethod
    def _nll(confidences: Sequence[float], is_tp: Sequence[int], T: float,
             eps: float = 1e-12) -> float:
        """Mean negative log-likelihood of TP labels under temperature T."""
        T = max(T, 1e-3)
        total = 0.0
        for c, y in zip(confidences, is_tp):
            p = _sigmoid(_logit(float(c)) / T)
            p = min(1.0 - eps, max(eps, p))
            total -= math.log(p) if y else math.log(1.0 - p)
        return total / max(len(confidences), 1)

    def fit(self, val_confidences: Sequence[float], val_is_tp: Sequence[int],
            t_min: float = 0.05, t_max: float = 10.0, iters: int = 100) -> float:
        """Fit T by golden-section minimization of NLL over [t_min, t_max].

        Args:
            val_confidences: raw detection confidences on the calibration split.
            val_is_tp: 1 if the detection is a true positive (IoU-matched), else 0.
        Returns:
            The fitted temperature (also stored on self.temperature).
        """
        conf = [float(c) for c in val_confidences]
        tp = [1 if y else 0 for y in val_is_tp]
        if not conf:
            return self.temperature

        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...
        a, b = t_min, t_max
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc, fd = self._nll(conf, tp, c), self._nll(conf, tp, d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = self._nll(conf, tp, c)
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = self._nll(conf, tp, d)
            if abs(b - a) < 1e-4:
                break
        self.temperature = (a + b) / 2.0
        return self.temperature

    def compute_detection_ece(self, confidences: Sequence[float],
                              is_tp: Sequence[int], n_bins: int = 10) -> float:
        """Expected Calibration Error = Σ_b (n_b/N)·|precision_b − meanconf_b|.

        Equal-width confidence bins over [0, 1]; conf == 1 falls in the last bin.
        """
        conf = [float(c) for c in confidences]
        tp = [1 if y else 0 for y in is_tp]
        n = len(conf)
        if n == 0:
            return 0.0

        bin_conf: List[List[float]] = [[] for _ in range(n_bins)]
        bin_tp: List[List[int]] = [[] for _ in range(n_bins)]
        for c, y in zip(conf, tp):
            b = min(n_bins - 1, max(0, int(c * n_bins)))
            bin_conf[b].append(c)
            bin_tp[b].append(y)

        ece = 0.0
        for b in range(n_bins):
            if not bin_conf[b]:
                continue
            avg_conf = sum(bin_conf[b]) / len(bin_conf[b])
            precision = sum(bin_tp[b]) / len(bin_tp[b])
            ece += (len(bin_conf[b]) / n) * abs(precision - avg_conf)
        return ece


class PerClassTemperatureScaling:
    """Independent temperature per class; calibrate each detection by its class T.

    Detection confidence is often miscalibrated differently per class (e.g. small
    thermal 'person' boxes vs large 'car'), so a per-class temperature can beat a
    single global one while staying a simple, defensible parametric calibrator.
    """

    def __init__(self):
        self.temperatures: dict = {}  # class_id -> fitted T

    def fit(self, confidences, is_tp, labels, **fit_kwargs) -> dict:
        grouped: dict = {}
        for c, y, l in zip(confidences, is_tp, labels):
            cs, ys = grouped.setdefault(int(l), ([], []))
            cs.append(float(c))
            ys.append(1 if y else 0)
        self.temperatures = {}
        for l, (cs, ys) in grouped.items():
            self.temperatures[l] = DetectionTemperatureScaling().fit(cs, ys, **fit_kwargs)
        return self.temperatures

    def transform(self, confidences, labels) -> List[float]:
        out = []
        for c, l in zip(confidences, labels):
            T = self.temperatures.get(int(l), 1.0)
            out.append(_sigmoid(_logit(float(c)) / max(T, 1e-3)))
        return out


# ----------------------------------------------------------------------------
# Unit test: python3 src/calibration/temperature_scaling.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ts = DetectionTemperatureScaling()

    # Overconfident detector: asserts 0.9 confidence but only 50% are true positives.
    conf = [0.9] * 100
    is_tp = [1] * 50 + [0] * 50
    ece_before = ts.compute_detection_ece(conf, is_tp)          # ≈ |0.5 - 0.9| = 0.4
    T = ts.fit(conf, is_tp)                                     # should soften (T > 1)
    ece_after = ts.compute_detection_ece(ts.transform(conf), is_tp)

    assert T > 1.0, T
    assert ece_after < ece_before, (ece_before, ece_after)
    assert all(0.0 < p < 1.0 for p in ts.transform(conf))
    # Already-calibrated data should leave a near-1 temperature and tiny ECE
    cal_conf = [0.5] * 100
    cal_tp = [1] * 50 + [0] * 50
    assert ts.compute_detection_ece(cal_conf, cal_tp) < 0.05

    print("temperature_scaling self-test OK: T=%.2f  ECE %.3f -> %.3f"
          % (T, ece_before, ece_after))

    # Per-class temperature: class 0 overconfident, class 1 well-calibrated.
    pc_conf = [0.9] * 100 + [0.5] * 100
    pc_tp = [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50
    pc_lab = [0] * 100 + [1] * 100
    pct = PerClassTemperatureScaling()
    temps = pct.fit(pc_conf, pc_tp, pc_lab)
    assert temps[0] > 1.0, temps  # class 0 softened
    ece_pc_before = ts.compute_detection_ece(pc_conf, pc_tp)
    ece_pc_after = ts.compute_detection_ece(pct.transform(pc_conf, pc_lab), pc_tp)
    assert ece_pc_after < ece_pc_before, (ece_pc_before, ece_pc_after)
    print("per_class_temp self-test OK: T_by_class=%s  ECE %.3f -> %.3f"
          % ({k: round(v, 2) for k, v in temps.items()}, ece_pc_before, ece_pc_after))
