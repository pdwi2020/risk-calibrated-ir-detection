"""Assemble the four-detector comparison table from eval/risk JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MODELS = [
    ("YOLOv8m", "results/yolov8m_flir_seed0/eval/yolov8m_test_map.json", "results/yolov8m_flir_seed0/eval/yolov8m_risk_summary.json"),
    ("Faster R-CNN", "results/faster_rcnn_flir_seed0/eval/frcnn_test_map.json", "results/faster_rcnn_flir_seed0/eval/frcnn_risk_summary.json"),
    ("RT-DETR", "results/rtdetr_flir_seed0/eval/rtdetr_test_map.json", "results/rtdetr_flir_seed0/eval/rtdetr_risk_summary.json"),
    ("RetinaNet", "results/retinanet_flir_seed0/eval/retinanet_test_map.json", "results/retinanet_flir_seed0/eval/retinanet_risk_summary.json"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, map_rel, risk_rel in DEFAULT_MODELS:
        map_path = root / map_rel
        risk_path = root / risk_rel
        if not map_path.exists() or not risk_path.exists():
            missing = []
            if not map_path.exists():
                missing.append(str(map_path))
            if not risk_path.exists():
                missing.append(str(risk_path))
            raise FileNotFoundError("; ".join(missing))
        map_data = _load(map_path)
        risk = _load(risk_path)
        rows.append(
            {
                "Detector": name,
                "mAP@0.5": map_data.get("mAP50"),
                "mAP50-95": map_data.get("mAP50-95"),
                "theta*": risk.get("theta_star"),
                "Cost reduction %": 100.0 * float(risk.get("cost_reduction_frac", 0.0)),
                "ECE method": risk.get("ece_best_method"),
                "ECE": risk.get("ece_test_calibrated"),
                "lambda_hat": risk.get("lambda_hat"),
                "Coverage": risk.get("test_coverage"),
                "Controlled": risk.get("conformal_controlled"),
                "RA-AP": risk.get("ra_ap"),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = []
        for key in headers:
            value = row[key]
            if key == "Cost reduction %":
                cells.append(_fmt_float(value, 1))
            elif key in {"Controlled", "ECE method", "Detector"}:
                cells.append(str(value))
            else:
                cells.append(_fmt_float(value, 4))
        out.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--out-csv", default="results/detector_comparison.csv")
    p.add_argument("--out-md", default="results/detector_comparison.md")
    args = p.parse_args()

    root = Path(args.root)
    rows = build_rows(root)
    write_csv(rows, root / args.out_csv)
    write_markdown(rows, root / args.out_md)
    print(f"[comparison] wrote {root / args.out_csv}")
    print(f"[comparison] wrote {root / args.out_md}")
    for row in rows:
        print(
            f"{row['Detector']}: mAP50={_fmt_float(row['mAP@0.5'])} "
            f"cost_reduction={_fmt_float(row['Cost reduction %'], 1)}% "
            f"RA-AP={_fmt_float(row['RA-AP'])}"
        )


if __name__ == "__main__":
    main()
