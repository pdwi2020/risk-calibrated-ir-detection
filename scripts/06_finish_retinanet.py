"""Guarded RetinaNet continuation from a saved best checkpoint.

This is intentionally narrow: it resumes the interrupted RetinaNet work from a
known-good ``retinanet_best.pth`` checkpoint, appends epochs to the existing
metrics CSV, and preserves the old best checkpoint unless a later eval improves
mAP50.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._torch_train_utils import build_loaders, set_seed  # noqa: E402
from src.detectors.retinanet_wrapper import RetinaNetDetector  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continue RetinaNet from retinanet_best.pth and preserve best.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--flir-root", required=True)
    p.add_argument("--split-json", default="configs/flir_val_split.json")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--weights", default=None, help="Checkpoint to load; defaults to ckpt-dir/retinanet_best.pth")
    p.add_argument("--start-epoch", type=int, default=39)
    p.add_argument("--end-epoch", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5)
    return p.parse_args()


def _metric_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _best_map50(rows: list[dict[str, str]]) -> float:
    best = 0.0
    for row in rows:
        value = (row.get("mAP50") or "").strip()
        if not value:
            continue
        try:
            best = max(best, float(value))
        except ValueError:
            continue
    return best


def _append_row(csv_path: Path, row: dict[str, str]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "mAP50", "mAP50-95"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            inner = state.get(key)
            if isinstance(inner, dict):
                return inner
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict")
    return state


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else ckpt_dir / "retinanet_best.pth"
    best_path = ckpt_dir / "retinanet_best.pth"
    prior_best = ckpt_dir / "retinanet_best_prior_to_20260606_resume.pth"
    metrics_csv = ckpt_dir / "retinanet_metrics.csv"

    if not weights.exists():
        raise FileNotFoundError(weights)
    if best_path.exists() and not prior_best.exists():
        shutil.copy2(best_path, prior_best)

    print("[06_finish_retinanet] building dataloaders")
    train_loader, test_loader = build_loaders(
        flir_root=args.flir_root,
        split_json=args.split_json,
        batch=args.batch,
        workers=args.workers,
    )
    print(f"  train batches={len(train_loader)} test batches={len(test_loader)}")

    print(f"[06_finish_retinanet] loading {weights}")
    detector = RetinaNetDetector(n_classes=3, pretrained=False)
    detector.model.load_state_dict(_load_state(weights))
    detector.model.to(detector.device)

    optimizer = torch.optim.SGD(
        detector.model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_map50 = _best_map50(_metric_rows(metrics_csv))
    print(f"[06_finish_retinanet] current protected best mAP50={best_map50:.6f}")
    print(
        "[06_finish_retinanet] continuing epochs "
        f"{args.start_epoch}-{args.end_epoch} at lr={args.lr}"
    )

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        loss = float(detector.train_one_epoch(train_loader, optimizer))
        if not math.isfinite(loss):
            print(f"[06_finish_retinanet] non-finite loss at epoch {epoch}: {loss}")
            raise SystemExit(2)

        map50 = float("nan")
        map50_95 = float("nan")
        if epoch % args.eval_every == 0 or epoch == args.end_epoch:
            metrics = detector.evaluate(test_loader)
            map50 = float(metrics.get("mAP50", float("nan")))
            map50_95 = float(metrics.get("mAP50-95", float("nan")))

        _append_row(
            metrics_csv,
            {
                "epoch": str(epoch),
                "train_loss": f"{loss:.6f}",
                "mAP50": "" if math.isnan(map50) else f"{map50:.6f}",
                "mAP50-95": "" if math.isnan(map50_95) else f"{map50_95:.6f}",
            },
        )

        msg = f"[retinanet-resume] epoch {epoch}/{args.end_epoch} loss={loss:.4f}"
        if not math.isnan(map50):
            msg += f" mAP50={map50:.4f} mAP50-95={map50_95:.4f}"
        print(msg, flush=True)

        if epoch % 10 == 0 or epoch == args.end_epoch:
            epoch_path = ckpt_dir / f"retinanet_epoch{epoch}.pth"
            torch.save(detector.model.state_dict(), epoch_path)
            print(f"  -> saved checkpoint: {epoch_path}", flush=True)

        if not math.isnan(map50) and map50 > best_map50:
            best_map50 = map50
            torch.save(detector.model.state_dict(), best_path)
            print(f"  -> new best mAP50={best_map50:.4f} saved: {best_path}", flush=True)

    print(f"[06_finish_retinanet] done. protected/current best mAP50={best_map50:.6f}")
    print(f"  metrics: {metrics_csv}")
    print(f"  best:    {best_path}")


if __name__ == "__main__":
    main()
