"""Isotonic-regression calibration of detection confidences (pure Python).

Fits a monotone non-decreasing map  conf -> calibrated probability  on a held-out
calibration split via the Pool-Adjacent-Violators algorithm (PAV) -- the same
estimator scikit-learn's IsotonicRegression uses -- implemented dependency-free.
Unlike a single temperature (one scalar), isotonic regression can correct *any*
monotone miscalibration, so it typically drives detection ECE far below a global
scaling when the confidence->precision relationship is non-linear.

Fit data: per-detection (confidence, is_TP in {0,1}) on the CALIBRATION split.
Predict: piecewise-linear interpolation between the fitted step points, clamped to
the calibration range (out-of-range queries take the nearest endpoint value).
"""
from __future__ import annotations

from typing import List, Sequence


class IsotonicCalibrator:
    """Non-parametric monotone calibration map fit by PAV."""

    def __init__(self):
        self._x: List[float] = []  # increasing confidence nodes
        self._y: List[float] = []  # calibrated values (monotone non-decreasing)

    def fit(self, confidences: Sequence[float], is_tp: Sequence[int]) -> "IsotonicCalibrator":
        pts = sorted(((float(c), 1.0 if y else 0.0)
                      for c, y in zip(confidences, is_tp)), key=lambda t: t[0])
        if not pts:
            return self
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Pool Adjacent Violators: each block = [value, weight, x_start, x_end].
        # Merge adjacent blocks while the fit is decreasing (a monotonicity violation).
        blocks: List[List[float]] = []
        for x, y in zip(xs, ys):
            blocks.append([y, 1.0, x, x])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, s2, e2 = blocks.pop()
                v1, w1, s1, e1 = blocks.pop()
                w = w1 + w2
                blocks.append([(v1 * w1 + v2 * w2) / w, w, s1, e2])

        # Turn the constant-value blocks into interpolation nodes at block edges.
        X: List[float] = []
        Y: List[float] = []
        for v, w, s, e in blocks:
            X.append(s); Y.append(v)
            if e != s:
                X.append(e); Y.append(v)

        # Enforce strictly increasing x (dedupe equal x, keep the larger -> monotone y).
        ux: List[float] = []
        uy: List[float] = []
        for x, y in zip(X, Y):
            if ux and x <= ux[-1]:
                uy[-1] = max(uy[-1], y)
            else:
                ux.append(x); uy.append(y)
        self._x, self._y = ux, uy
        return self

    def _interp(self, x: float) -> float:
        X, Y = self._x, self._y
        if not X:
            return x
        if x <= X[0]:
            return Y[0]
        if x >= X[-1]:
            return Y[-1]
        lo, hi = 0, len(X) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if X[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0, x1, y0, y1 = X[lo], X[hi], Y[lo], Y[hi]
        if x1 == x0:
            return y0
        return y0 + (x - x0) / (x1 - x0) * (y1 - y0)

    def transform(self, confidences: Sequence[float]) -> List[float]:
        """Map raw confidences to calibrated probabilities."""
        return [self._interp(float(c)) for c in confidences]


# ----------------------------------------------------------------------------
# Unit test: python3 src/calibration/isotonic.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    def _ece(conf, tp, nbins=15):
        bc = [[] for _ in range(nbins)]
        bt = [[] for _ in range(nbins)]
        for c, y in zip(conf, tp):
            b = min(nbins - 1, max(0, int(c * nbins)))
            bc[b].append(c)
            bt[b].append(y)
        n = len(conf)
        e = 0.0
        for b in range(nbins):
            if bc[b]:
                e += (len(bc[b]) / n) * abs(sum(bt[b]) / len(bt[b]) - sum(bc[b]) / len(bc[b]))
        return e

    rng = random.Random(0)
    conf, tp = [], []
    for _ in range(8000):
        c = rng.random()
        conf.append(c)
        tp.append(1 if rng.random() < c * c else 0)  # precision = c^2 (nonlinear monotone)
    cal_c, cal_t = conf[:4000], tp[:4000]
    te_c, te_t = conf[4000:], tp[4000:]

    iso = IsotonicCalibrator().fit(cal_c, cal_t)
    e_raw = _ece(te_c, te_t)
    e_iso = _ece(iso.transform(te_c), te_t)

    # isotonic fixes the c^2 miscalibration a single temperature cannot
    assert e_iso < e_raw, (e_raw, e_iso)
    assert e_iso < 0.05, e_iso
    # predictions are monotone non-decreasing and map conf~0.5 -> precision~0.25
    pr = iso.transform([0.1, 0.3, 0.5, 0.7, 0.9])
    assert all(pr[i] <= pr[i + 1] + 1e-9 for i in range(len(pr) - 1)), pr
    assert abs(iso.transform([0.5])[0] - 0.25) < 0.06, iso.transform([0.5])[0]

    print("isotonic self-test OK: ECE %.3f -> %.3f ; map(0.5)=%.3f (expect ~0.25)"
          % (e_raw, e_iso, iso.transform([0.5])[0]))
