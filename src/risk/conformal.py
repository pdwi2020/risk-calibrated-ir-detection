"""Conformal risk control for detection confidence thresholds.

Implements Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster,
2022): choose the confidence threshold ``lambda`` on a held-out CALIBRATION
split so that the expected value of a bounded, monotone risk on unseen TEST
data is provably controlled at level ``alpha``.

Risk = the **per-image missed-detection event** (a safety-critical quantity for
IR pedestrian/vehicle detection). The exchangeable unit of the conformal
guarantee is the *image*, not the bounding box:

    r_i(lambda) = 1[ image i contains >= 1 ground-truth box NOT matched by any
                     prediction with score >= lambda ]
    R(lambda)   = (1 / n) * sum_i r_i(lambda)

evaluated over the ``n`` images that contain at least one ground-truth box
(background-only frames carry no miss event and are excluded from the rate;
they would only dilute it). ``r_i`` is bounded in {0, 1} and non-decreasing in
lambda (a stricter threshold keeps fewer detections, so an image can only go
from "all GT found" to "some GT missed"), so the calibration is a 1-D
threshold search and the finite-sample CRC bound applies with B = 1.

Finite-sample guarantee (risk bounded in [0, 1], B = 1):

    lambda_hat = max { lambda :  (n * Rhat(lambda) + 1) / (n + 1) <= alpha }

i.e. the *strictest* (highest-precision) threshold whose conservative,
finite-sample-corrected calibration risk still respects the budget alpha. By
the CRC theorem, E[R_test(lambda_hat)] <= alpha.

If NO threshold in the grid satisfies the corrected budget, the problem is
**infeasible**: there is no lambda for which the guarantee can be issued. In
that case ``lambda_hat`` is ``None`` and ``feasible`` is ``False`` -- the gate
FAILS. Returning the loosest threshold and calling it ``lambda_hat`` would
falsely advertise a guarantee that was never established, so we refuse to.

Per-image risk matches the box-level miss rate only when images carry a single
ground-truth box; in general it is more conservative, which is the honest
behaviour for a safety budget defined per scene.

Pure-Python (no numpy/torch) so it runs on cached predictions anywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


class ConformalRiskController:
    """Choose a confidence threshold that controls the per-image miss rate."""

    def __init__(self, alpha: float = 0.1, iou_threshold: float = 0.5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.lambda_hat: Optional[float] = None
        self.feasible: Optional[bool] = None
        self._lambdas: List[float] = []

    # ------------------------------------------------------------------
    def _image_missed(self, pred: Dict, gt: Dict, lam: float) -> Optional[bool]:
        """Per-image miss indicator at threshold ``lam``.

        Returns ``True`` if the image has >= 1 ground-truth box left unmatched
        by the kept (score >= lam) predictions, ``False`` if all GT is matched,
        and ``None`` for a background-only image (no GT -> not counted).
        """
        g_boxes = list(gt.get("boxes", []))
        if not g_boxes:
            return None  # no GT box: cannot miss; excluded from the per-image rate
        g_labels = [int(x) for x in gt.get("labels", [])]

        p_boxes = list(pred.get("boxes", []))
        p_scores = [float(s) for s in pred.get("scores", [])]
        p_labels = [int(x) for x in pred.get("labels", [])]
        # keep confident predictions, strongest first (greedy, class-aware)
        kept = sorted((i for i in range(len(p_boxes)) if p_scores[i] >= lam),
                      key=lambda i: -p_scores[i])

        matched = [False] * len(g_boxes)
        for pi in kept:
            best_iou, best_g = 0.0, -1
            for gi in range(len(g_boxes)):
                if matched[gi] or g_labels[gi] != p_labels[pi]:
                    continue
                iou = _iou(p_boxes[pi], g_boxes[gi])
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou, best_g = iou, gi
            if best_g >= 0:
                matched[best_g] = True
        return matched.count(False) > 0

    def _n_images(self, ground_truths: List[Dict]) -> int:
        """Number of GT-bearing images = the exchangeable units of the guarantee."""
        return sum(1 for gt in ground_truths if list(gt.get("boxes", [])))

    def empirical_risk(self, predictions: List[Dict], ground_truths: List[Dict],
                       lam: float) -> float:
        """Per-image miss rate: fraction of GT-bearing images with >= 1 miss."""
        miss = 0
        n = 0
        for pred, gt in zip(predictions, ground_truths):
            ind = self._image_missed(pred, gt, lam)
            if ind is None:
                continue
            n += 1
            miss += 1 if ind else 0
        return miss / n if n else 0.0

    # ------------------------------------------------------------------
    def calibrate(self, cal_predictions: List[Dict], cal_ground_truths: List[Dict],
                  lambdas: Optional[Sequence[float]] = None) -> Optional[float]:
        """Fit lambda_hat on the calibration split.

        Returns the largest lambda whose finite-sample-corrected per-image miss
        rate is <= alpha, or ``None`` if no candidate qualifies (infeasible: the
        gate fails and no guarantee is issued). ``self.feasible`` records which.
        """
        if lambdas is None:
            lambdas = [round(0.01 * i, 2) for i in range(1, 100)]  # 0.01..0.99
        self._lambdas = sorted(lambdas)

        n = max(self._n_images(cal_ground_truths), 1)  # n = #GT-bearing images

        qualifying = []
        for lam in self._lambdas:
            r_hat = self.empirical_risk(cal_predictions, cal_ground_truths, lam)
            corrected = (n * r_hat + 1.0) / (n + 1.0)
            if corrected <= self.alpha:
                qualifying.append(lam)

        if qualifying:
            self.lambda_hat = max(qualifying)
            self.feasible = True
        else:
            # No threshold meets the budget: refuse to advertise a guarantee.
            self.lambda_hat = None
            self.feasible = False
        return self.lambda_hat

    def validate_coverage(self, test_predictions: List[Dict],
                          test_ground_truths: List[Dict],
                          lam: Optional[float] = None) -> Dict[str, object]:
        """Empirical test risk; controlled iff feasible and risk <= alpha.

        When calibration was infeasible (``lambda_hat is None``) we still report
        the best achievable per-image miss rate at the loosest grid threshold so
        the failure is quantified, but ``feasible`` and ``controlled`` are False
        and no ``lambda_hat`` is claimed.
        """
        if lam is None:
            lam = self.lambda_hat
        if self.feasible is None:
            raise RuntimeError("calibrate() must be called before validate_coverage()")

        if lam is None:  # infeasible calibration
            diag_lam = self._lambdas[0] if self._lambdas else 0.01
            risk = self.empirical_risk(test_predictions, test_ground_truths, diag_lam)
            return {
                "lambda_hat": None,
                "alpha": self.alpha,
                "feasible": False,
                "best_achievable_lambda": diag_lam,
                "test_risk": risk,                     # per-image miss rate at loosest lambda
                "test_coverage": 1.0 - risk,
                "controlled": False,                   # no guarantee was issued
            }

        risk = self.empirical_risk(test_predictions, test_ground_truths, lam)
        return {
            "lambda_hat": lam,
            "alpha": self.alpha,
            "feasible": True,
            "test_risk": risk,                         # per-image miss rate
            "test_coverage": 1.0 - risk,               # per-image detection rate
            "controlled": risk <= self.alpha,          # risk budget respected on test
        }


# ----------------------------------------------------------------------------
# Unit test: python3 src/risk/conformal.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    def make_split(n_imgs, seed):
        rng = random.Random(seed)
        preds, gts = [], []
        for _ in range(n_imgs):
            # one GT person box, one perfectly-localized prediction with random score
            box = [10.0, 10.0, 30.0, 50.0]
            score = rng.random()  # uniform[0,1]
            gts.append({"boxes": [box], "labels": [0]})
            preds.append({"boxes": [box], "scores": [score], "labels": [0]})
        return preds, gts

    cp, cg = make_split(400, seed=1)
    tp, tg = make_split(400, seed=2)

    crc = ConformalRiskController(alpha=0.2, iou_threshold=0.5)
    lam = crc.calibrate(cp, cg)
    cov = crc.validate_coverage(tp, tg)

    # Per-image risk monotone non-decreasing in lambda
    r_lo = crc.empirical_risk(cp, cg, 0.1)
    r_hi = crc.empirical_risk(cp, cg, 0.9)
    assert r_hi >= r_lo, (r_lo, r_hi)
    # n = #GT-bearing images = 400 (single box each)
    assert crc._n_images(cg) == 400, crc._n_images(cg)
    # With uniform scores, per-image miss(lambda) ~= lambda, so lambda_hat ~= alpha
    assert lam is not None and 0.05 <= lam <= 0.25, lam
    assert cov["feasible"] and cov["test_risk"] <= crc.alpha + 0.05, cov

    # Infeasibility path: an impossible budget must FAIL the gate (no lambda_hat)
    crc2 = ConformalRiskController(alpha=0.0, iou_threshold=0.5)
    lam2 = crc2.calibrate(cp, cg)
    cov2 = crc2.validate_coverage(tp, tg)
    assert lam2 is None and crc2.feasible is False, (lam2, crc2.feasible)
    assert cov2["feasible"] is False and cov2["controlled"] is False, cov2

    print("conformal self-test OK: alpha=%.2f lambda_hat=%.2f test_risk=%.3f coverage=%.3f"
          % (crc.alpha, lam, cov["test_risk"], cov["test_coverage"]))
    print("infeasible path OK: lambda_hat=%s feasible=%s best_risk=%.3f"
          % (cov2["lambda_hat"], cov2["feasible"], cov2["test_risk"]))
