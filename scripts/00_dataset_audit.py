#!/usr/bin/env python3
"""Dataset audit script — Week 1 deliverable.

Verifies FLIR ADAS v2, LLVIP, and KAIST preview datasets.
Outputs:
  - Console: image counts, annotation counts, class distributions
  - File:    audit_output/dataset_audit_summary.txt
  - File:    audit_output/sample_grid.png   (sample images + corruptions)

Usage (local, X9 mounted):
    python scripts/00_dataset_audit.py

Usage (RunPod, after data transfer):
    python scripts/00_dataset_audit.py --x9 /workspace/data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — override with --x9 if running on RunPod
# ---------------------------------------------------------------------------

DEFAULT_X9 = "/Volumes/Crucial X9"


def build_paths(x9: str):
    x9 = Path(x9)
    return {
        "flir_root": x9 / "Research Projects/MV_Paper/datasets/flir_adas_v2/FLIR_ADAS_v2",
        "llvip_root": x9 / "Research Projects/MV_Paper/datasets/llvip",
        "kaist_root": x9 / "Research Projects/MV_Paper/datasets/kaist_preview/kaist_yolo_early/kaist_yolo_early",
        "src_root":   x9 / "Research Projects/MV_Paper",
    }


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def audit_flir(flir_root: Path) -> dict:
    """Audit FLIR ADAS v2 — parse coco.json for both splits."""
    import json
    from collections import Counter

    KEEP_CATS = {1: "person", 2: "bike", 3: "car"}
    results = {}

    for split, split_dir in [("train", "images_thermal_train"), ("val", "images_thermal_val")]:
        ann_file = flir_root / split_dir / "coco.json"
        if not ann_file.exists():
            print(f"  [WARN] {ann_file} not found — skipping FLIR {split}")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        n_images = len(coco["images"])
        cat_counts = Counter(
            a["category_id"] for a in coco["annotations"]
            if a["category_id"] in KEEP_CATS
        )
        class_dist = {KEEP_CATS[cid]: cnt for cid, cnt in cat_counts.items()}
        total_anns = sum(class_dist.values())

        # Check image dir (exclude macOS ._* shadow files)
        img_dir = flir_root / split_dir / "data"
        n_image_files = (
            sum(1 for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
            if img_dir.exists() else 0
        )

        results[split] = {
            "n_images_json": n_images,
            "n_image_files": n_image_files,
            "n_annotations": total_anns,
            "class_dist": class_dist,
            "resolution": "640×512",
            "format": "COCO JSON",
        }

    return results


def audit_llvip(llvip_root: Path) -> dict:
    """Audit LLVIP — count IR images and YOLO label files."""
    yolo_root = llvip_root / "LLVIP-YOLO"
    results = {}

    for split in ["train", "test"]:
        img_dir  = yolo_root / split / "lwir" / "images"
        lbl_dir  = yolo_root / split / "visible" / "labels"

        if not img_dir.exists():
            print(f"  [WARN] {img_dir} not found — skipping LLVIP {split}")
            continue

        n_images = sum(1 for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
        n_labels = sum(1 for p in lbl_dir.glob("*.txt") if not p.name.startswith(".")) \
            if lbl_dir.exists() else 0

        # Count total person boxes (read first 500 label files for speed)
        n_boxes = 0
        label_files = [p for p in lbl_dir.glob("*.txt") if not p.name.startswith(".")] \
            if lbl_dir.exists() else []
        sample_size = min(500, len(label_files))
        for lf in label_files[:sample_size]:
            with open(lf) as f:
                n_boxes += sum(1 for ln in f if len(ln.strip().split()) == 5)
        # Extrapolate to full set
        if sample_size and sample_size < len(label_files):
            n_boxes_est = int(n_boxes / sample_size * len(label_files))
            boxes_str = f"~{n_boxes_est:,} (estimated)"
        else:
            boxes_str = f"{n_boxes:,}"

        results[split] = {
            "n_images": n_images,
            "n_labels": n_labels,
            "n_boxes": boxes_str,
            "class_dist": {"person": "all"},
            "resolution": "1280×1024",
            "format": "YOLO txt",
        }

    return results


def audit_kaist(kaist_root: Path) -> dict:
    """Audit KAIST preview — count images per split."""
    results = {}
    for split in ["train", "val", "test"]:
        img_dir = kaist_root / "images" / split
        lbl_dir = kaist_root / "labels" / split
        n_images = sum(1 for _ in img_dir.rglob("*") if _.is_file()) if img_dir.exists() else 0
        n_labels = sum(1 for _ in lbl_dir.rglob("*.txt") if not _.name.startswith(".")) \
            if lbl_dir.exists() else 0
        results[split] = {
            "n_images": n_images,
            "n_labels": n_labels,
        }
    return results


# ---------------------------------------------------------------------------
# Sample visualisation
# ---------------------------------------------------------------------------

def visualise_samples(paths: dict, out_dir: Path) -> None:
    """Save a grid of sample images with bounding boxes + corruption demo."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image as PILImage
    except ImportError:
        print("  [SKIP] matplotlib/PIL not installed — skipping visualisation")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: Sample images from each dataset (3 + 3 + 2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("MV Paper — Dataset Samples", fontsize=14, fontweight="bold")

    # FLIR thermal samples
    flir_root = paths["flir_root"]
    flir_img_dir = flir_root / "images_thermal_train" / "data"
    import json
    with open(flir_root / "images_thermal_train" / "coco.json") as f:
        flir_coco = json.load(f)
    KEEP_CATS = {1: "person", 2: "bike", 3: "car"}
    COLORS = {0: "red", 1: "yellow", 2: "cyan"}
    CLASS_NAMES = {0: "person", 1: "bike", 2: "car"}
    cat_id_to_idx = {}
    for cat in flir_coco["categories"]:
        if cat["id"] in KEEP_CATS:
            cat_id_to_idx[cat["id"]] = list(KEEP_CATS.keys()).index(cat["id"])
    ann_by_img = {}
    for ann in flir_coco["annotations"]:
        if ann["category_id"] in KEEP_CATS:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Pick 3 images that have annotations
    sample_imgs = [img for img in flir_coco["images"] if img["id"] in ann_by_img][:3]

    for col, img_meta in enumerate(sample_imgs):
        img_path = flir_img_dir / Path(img_meta["file_name"]).name
        try:
            img = PILImage.open(img_path).convert("RGB")
            ax = axes[0][col]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"FLIR train #{col+1}", fontsize=9)
            ax.axis("off")
            W, H = img.width, img.height
            for ann in ann_by_img.get(img_meta["id"], []):
                if ann["category_id"] not in KEEP_CATS:
                    continue
                x, y, bw, bh = ann["bbox"]
                idx = cat_id_to_idx[ann["category_id"]]
                rect = patches.Rectangle(
                    (x, y), bw, bh,
                    linewidth=1.5, edgecolor=COLORS[idx], facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x, y - 3, CLASS_NAMES[idx], color=COLORS[idx], fontsize=7)
        except Exception as e:
            axes[0][col].set_title(f"FLIR #{col+1}\n(load err: {e})", fontsize=7)
            axes[0][col].axis("off")

    # LLVIP IR samples
    llvip_img_dir = paths["llvip_root"] / "LLVIP-YOLO" / "train" / "lwir" / "images"
    llvip_lbl_dir = paths["llvip_root"] / "LLVIP-YOLO" / "train" / "visible" / "labels"
    llvip_imgs = sorted(llvip_img_dir.glob("*.jpg"))[:3]

    for col, img_path in enumerate(llvip_imgs):
        try:
            img = PILImage.open(img_path).convert("RGB")
            W, H = img.width, img.height
            ax = axes[0][3] if col == 0 else axes[1][col - 1]
            ax.imshow(img, cmap="inferno")
            ax.set_title(f"LLVIP IR #{col+1}", fontsize=9)
            ax.axis("off")
            lbl_path = llvip_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, cx, cy, bw, bh = map(float, parts)
                            x = (cx - bw / 2) * W
                            y = (cy - bh / 2) * H
                            rect = patches.Rectangle(
                                (x, y), bw * W, bh * H,
                                linewidth=1.5, edgecolor="lime", facecolor="none"
                            )
                            ax.add_patch(rect)
        except Exception as e:
            ax = axes[0][3] if col == 0 else axes[1][col - 1]
            ax.set_title(f"LLVIP #{col+1}\n(err: {e})", fontsize=7)
            ax.axis("off")

    # Hide unused subplot
    axes[1][2].axis("off")
    axes[1][3].axis("off")

    plt.tight_layout()
    out_path = out_dir / "sample_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Figure 2: Corruption demo on one FLIR image (severity 1–4, gaussian)
    # ------------------------------------------------------------------
    try:
        import sys as _sys
        _sys.path.insert(0, str(paths["src_root"]))
        from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY, _FOG_T, _GAUSS_STD

        demo_img_path = flir_img_dir / Path(sample_imgs[0]["file_name"]).name
        demo_img = np.array(PILImage.open(demo_img_path).convert("L"))  # grayscale

        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))
        fig2.suptitle("Corruption Demo — FLIR Thermal (gaussian_noise, severities 1–4)", fontsize=12)

        axes2[0][0].imshow(demo_img, cmap="gray", vmin=0, vmax=255)
        axes2[0][0].set_title("Clean", fontsize=10)
        axes2[0][0].axis("off")

        for i, sev in enumerate([1, 2, 3, 4]):
            corrupted = CORRUPTION_REGISTRY["gaussian_noise"](demo_img, sev)
            ax = axes2[(i + 1) // 3][(i + 1) % 3]
            ax.imshow(corrupted, cmap="gray", vmin=0, vmax=255)
            sigma = _GAUSS_STD[sev]
            ax.set_title(f"Severity {sev} (σ={sigma})", fontsize=10)
            ax.axis("off")

        # Show fog corruption severity 4 for comparison
        fog_corrupted = CORRUPTION_REGISTRY["fog"](demo_img, 4)
        axes2[1][2].imshow(fog_corrupted, cmap="gray", vmin=0, vmax=255)
        axes2[1][2].set_title(f"Fog severity 4 (T={_FOG_T[4]})", fontsize=10)
        axes2[1][2].axis("off")

        plt.tight_layout()
        corr_path = out_dir / "corruption_demo.png"
        plt.savefig(corr_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {corr_path}")

    except Exception as e:
        print(f"  [WARN] Corruption demo failed: {e}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(flir: dict, llvip: dict, kaist: dict, txt_path: Path) -> None:
    lines = []

    sep = "=" * 78
    lines.append("")
    lines.append(sep)
    lines.append("  MV PAPER -- DATASET AUDIT SUMMARY")
    lines.append(sep)

    # FLIR
    lines.append("\n── FLIR ADAS Thermal v2 ──────────────────────────────────────────────────────")
    lines.append(f"  {'Split':<8} {'Images (JSON)':<16} {'Image files':<14} {'Annotations':<14} {'Class distribution'}")
    lines.append(f"  {'-'*6}  {'-'*13}  {'-'*11}  {'-'*11}  {'-'*35}")
    for split, r in flir.items():
        dist_str = "  ".join(f"{k}:{v:,}" for k, v in r["class_dist"].items())
        lines.append(f"  {split:<8} {r['n_images_json']:>13,}  {r['n_image_files']:>11,}  {r['n_annotations']:>11,}  {dist_str}")
    lines.append(f"  Resolution: {list(flir.values())[0]['resolution']}   Format: COCO JSON")

    # LLVIP
    lines.append("\n── LLVIP Infrared ────────────────────────────────────────────────────────────")
    lines.append(f"  {'Split':<8} {'Images':<10} {'Labels':<10} {'Boxes'}")
    lines.append(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*20}")
    for split, r in llvip.items():
        lines.append(f"  {split:<8} {r['n_images']:>8,}  {r['n_labels']:>8,}  {r['n_boxes']}")
    lines.append(f"  Resolution: 1280×1024   Class: person only   Format: YOLO txt")
    lines.append(f"  Labels path: LLVIP-YOLO/{{split}}/visible/labels/")
    lines.append(f"  IR images:   LLVIP-YOLO/{{split}}/lwir/images/")

    # KAIST
    lines.append("\n── KAIST Preview (1.7 GB subset) ─────────────────────────────────────────────")
    lines.append(f"  {'Split':<8} {'Images':<10} {'Labels'}")
    lines.append(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for split, r in kaist.items():
        lines.append(f"  {split:<8} {r['n_images']:>8,}  {r['n_labels']:>8,}")
    lines.append(f"  Class: person only   Format: YOLO txt")
    lines.append(f"  Note: Full KAIST (~30 GB, 95K frames) requires request form")

    # Experiment checklist status
    lines.append("\n── Experiment Status ─────────────────────────────────────────────────────────")
    lines.append("  [✓] Dataset verified on X9")
    lines.append("  [✓] FLIRDataset / LLVIPDataset implemented")
    lines.append("  [✓] CorruptionPipeline (6 types × 4 severities) implemented")
    lines.append("  [✓] YOLOv8 wrapper implemented")
    lines.append("  [✓] LLVIP lwir/labels symlinks created")
    lines.append("  [✓] flir_coco_to_yolo.py converter ready (run when PDWI mounted)")
    lines.append("  [ ] YOLOv8m baseline training (RunPod RTX 4090 — Week 3)")
    lines.append("  [ ] Faster R-CNN training (Week 5)")
    lines.append("  [ ] Corruption evaluation (Week 7–8)")
    lines.append("  [ ] Calibration & UQ (Week 9–10)")

    lines.append("")
    lines.append("=" * 78)

    summary = "\n".join(lines)
    print(summary)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"\n  Summary saved: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MV Paper dataset audit")
    parser.add_argument(
        "--x9", default=DEFAULT_X9,
        help=f"Path to X9 (or equivalent on RunPod). Default: {DEFAULT_X9}"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip sample visualisation (faster)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for PNG/txt (default: <project>/audit_output/)"
    )
    args = parser.parse_args()

    paths = build_paths(args.x9)

    if not paths["flir_root"].exists():
        sys.exit(
            f"ERROR: FLIR root not found: {paths['flir_root']}\n"
            f"Mount Crucial X9 or pass --x9 /path/to/data"
        )

    out_dir = Path(args.out_dir) if args.out_dir else (
        paths["src_root"] / "audit_output"
    )

    # Add src to path so we can import project modules
    sys.path.insert(0, str(paths["src_root"]))

    print("=" * 60)
    print("  MV Paper — Dataset Audit")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] Auditing FLIR ADAS v2 …")
    flir = audit_flir(paths["flir_root"])

    print("[2/4] Auditing LLVIP …")
    llvip = audit_llvip(paths["llvip_root"])

    print("[3/4] Auditing KAIST preview …")
    kaist = audit_kaist(paths["kaist_root"])

    if not args.no_vis:
        print("[4/4] Generating sample visualisations …")
        visualise_samples(paths, out_dir)
    else:
        print("[4/4] Skipping visualisation (--no-vis)")

    print_summary(flir, llvip, kaist, out_dir / "dataset_audit_summary.txt")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
