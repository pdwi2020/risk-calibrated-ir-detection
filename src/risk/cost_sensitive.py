"""Cost-sensitive threshold optimization for risk-calibrated detection.

Cost function: C = c_FN·FN + c_FP·FP + c_Loc·LocErr + c_Def·Defer
  c_FN = 10 (missed detection — safety critical)
  c_FP = 1  (false alarm)
  c_Loc = 2 (localization error weight)
  c_Def = 0.5 (deferral/abstain cost)

Pure-Python (no numpy/torch) so it runs anywhere on cached predictions. Inputs
may be Python lists, numpy arrays, or torch tensors — each box/score/label is
coerced with float()/int() per element, which works for all three.
"""

from typing import Dict, List, Optional, Tuple


class CostSensitiveThreshold:
    """Optimize confidence threshold to minimize expected risk E[C(θ)]."""

    def __init__(
        self,
        c_fn: float = 10.0,
        c_fp: float = 1.0,
        c_loc: float = 2.0,
        c_def: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        self.c_fn = c_fn
        self.c_fp = c_fp
        self.c_loc = c_loc
        self.c_def = c_def
        self.iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(box_a, box_b) -> float:
        """IoU of two axis-aligned xyxy boxes."""
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def compute_cost(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        conf_threshold: float,
        uncertainty_threshold: Optional[float] = None,
    ) -> float:
        """Compute total cost C(θ) over a set of images.

        Args:
            predictions: per-image dicts with 'boxes' (N×4 xyxy), 'scores' (N,),
                'labels' (N,), and optionally 'uncertainty' (N,).
            ground_truths: per-image dicts with 'boxes' (M×4 xyxy), 'labels' (M,).
            conf_threshold: keep only predictions with score >= this value.
            uncertainty_threshold: if set, kept predictions whose uncertainty
                exceeds this are *deferred* (abstained on) rather than asserted.

        Matching is class-aware and greedy by descending score, IoU >= iou_threshold.

        Deferral semantics (so deferring can actually reduce cost): a deferred
        prediction is routed to human review — NOT scored as TP/FP. It costs
        c_Def, and any ground-truth box it would have matched is considered handled
        (not a missed detection). So deferring an uncertain prediction trades a
        possible c_FP (false alarm) or c_FN (miss) for the smaller c_Def.

        Returns:
            C = c_FN·FN + c_FP·FP + c_Loc·LocErr + c_Def·Defer summed over images.
        """
        total_fn = 0
        total_fp = 0
        total_defer = 0
        total_loc_err = 0.0

        for pred, gt in zip(predictions, ground_truths):
            p_boxes = list(pred.get("boxes", []))
            p_scores = [float(s) for s in pred.get("scores", [])]
            p_labels = [int(x) for x in pred.get("labels", [])]
            unc = pred.get("uncertainty")
            p_unc = [float(u) for u in unc] if unc is not None else None

            g_boxes = list(gt.get("boxes", []))
            g_labels = [int(x) for x in gt.get("labels", [])]
            n_gt = len(g_boxes)
            gt_resolved = [False] * n_gt  # matched by a confident OR deferred pred

            # Split kept predictions into confident vs deferred.
            confident: List[int] = []
            deferred: List[int] = []
            for i in range(len(p_boxes)):
                if p_scores[i] < conf_threshold:
                    continue
                if (uncertainty_threshold is not None and p_unc is not None
                        and p_unc[i] > uncertainty_threshold):
                    deferred.append(i)
                else:
                    confident.append(i)
            total_defer += len(deferred)

            def _greedy(order: List[int], count_fp: bool) -> None:
                nonlocal total_fp, total_loc_err
                for pi in order:
                    pl = p_labels[pi]
                    best_iou, best_g = 0.0, -1
                    for gi in range(n_gt):
                        if gt_resolved[gi] or g_labels[gi] != pl:
                            continue
                        iou = self._iou(p_boxes[pi], g_boxes[gi])
                        if iou >= self.iou_threshold and iou > best_iou:
                            best_iou, best_g = iou, gi
                    if best_g >= 0:
                        gt_resolved[best_g] = True
                        if count_fp:  # confident true positive
                            total_loc_err += (1.0 - best_iou)
                    elif count_fp:    # confident, matched nothing -> false alarm
                        total_fp += 1
                    # deferred preds matching nothing cost only c_Def (already counted)

            # Confident predictions get priority on GT, then deferred ones.
            _greedy(sorted(confident, key=lambda i: -p_scores[i]), count_fp=True)
            _greedy(sorted(deferred, key=lambda i: -p_scores[i]), count_fp=False)

            total_fn += sum(1 for r in gt_resolved if not r)

        return (
            self.c_fn * total_fn
            + self.c_fp * total_fp
            + self.c_loc * total_loc_err
            + self.c_def * total_defer
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def optimize_threshold(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        threshold_sweep: List[float] = None,
    ) -> Tuple[float, float]:
        """Find optimal threshold θ* = argmin E[C(θ)].

        Returns:
            (optimal_threshold, min_expected_cost).
        """
        if threshold_sweep is None:
            threshold_sweep = [round(0.1 * i, 1) for i in range(1, 10)]

        costs = [(theta, self.compute_cost(predictions, ground_truths, theta))
                 for theta in threshold_sweep]
        return min(costs, key=lambda x: x[1])

    def risk_adjusted_ap(
        self,
        map_score: float,
        expected_cost: float,
        alpha: float = 0.01,
    ) -> float:
        """RA-AP = mAP − alpha·E[C] at optimal threshold.

        Novel metric: penalizes high mAP that comes with high risk (high E[C]).
        """
        return map_score - alpha * expected_cost


# ----------------------------------------------------------------------------
# Unit test (hand-verified toy scene): python3 src/risk/cost_sensitive.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cs = CostSensitiveThreshold()  # c_fn=10, c_fp=1, c_loc=2, c_def=0.5, iou=0.5

    # One image: GT person@[0,0,10,10], car@[20,20,30,30], bike@[40,40,50,50].
    gt = [{"boxes": [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
           "labels": [0, 2, 1]}]
    # Preds: exact person (s=.9), exact car (s=.8), spurious person@[50,50,60,60] (s=.6).
    pred = [{"boxes": [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]],
             "scores": [0.9, 0.8, 0.6], "labels": [0, 2, 0],
             "uncertainty": [0.1, 0.1, 0.9]}]

    # θ=0.5, no deferral: TP=2 (LocErr 0), FP=1 (spurious), FN=1 (bike) -> 10+1 = 11
    assert abs(cs.compute_cost(pred, gt, 0.5) - 11.0) < 1e-6
    # θ=0.7: spurious (0.6) dropped -> FP=0, FN=1 -> 10
    assert abs(cs.compute_cost(pred, gt, 0.7) - 10.0) < 1e-6
    # θ=0.5 + defer uncertainty>0.5: spurious deferred -> FP 1->0, +c_def 0.5 -> 10.5
    assert abs(cs.compute_cost(pred, gt, 0.5, uncertainty_threshold=0.5) - 10.5) < 1e-6
    # argmin sweep prefers θ where cost is lowest (<= 10)
    theta, cost = cs.optimize_threshold(pred, gt)
    assert cost <= 10.0 + 1e-6, (theta, cost)
    # RA-AP sanity: higher cost lowers the score
    assert cs.risk_adjusted_ap(0.8, 100.0) < cs.risk_adjusted_ap(0.8, 0.0)

    print("cost_sensitive self-test OK: "
          "C(0.5)=11.0  C(0.7)=10.0  C(0.5,defer)=10.5  argmin=%.1f@θ=%.1f" % (cost, theta))
