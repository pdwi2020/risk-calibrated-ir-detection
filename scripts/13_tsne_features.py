"""t-SNE feature visualization: FLIR vs LLVIP domain separation.

Extracts backbone features from YOLOv8m (FLIR-trained) for a balanced sample
of FLIR test images and LLVIP test images, runs t-SNE, and saves a
publication-quality PDF coloured by domain (and optionally person/other class
presence in GT labels).

Usage (from project root):
    python3 scripts/13_tsne_features.py \
        --flir-weights   results/yolov8m_flir_seed0/weights/best.pt \
        --flir-img-dir   /workspace/data/flir_yolo/val/images \
        --flir-lbl-dir   /workspace/data/flir_yolo/val/labels \
        --llvip-img-dir  /workspace/data/llvip/LLVIP-YOLO/test/lwir/images \
        --llvip-lbl-dir  /workspace/data/llvip/LLVIP-YOLO/test/lwir/labels \
        --n-per-domain 300 \
        --out paper/figs/fig_tsne.pdf
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── IEEE-style rc params ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    8,
    "figure.dpi":        200,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi":       300,
})

FLIR_COLOR  = "#004EA6"   # blue
LLVIP_COLOR = "#C00000"   # red


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    weights: str,
    img_paths: list[Path],
    device: str = "cuda",
) -> np.ndarray:
    """Extract global-average-pooled backbone features (SPPF layer 9) from YOLOv8m."""
    import torch
    import torch.nn.functional as F
    import cv2

    from ultralytics import YOLO

    model = YOLO(weights)
    yolo_model = model.model
    yolo_model.eval()
    yolo_model.to(device)

    feats: list[np.ndarray] = []
    hook_output: list = []

    def _hook(module, inp, out):
        # GAP over spatial dims → (B, C)
        hook_output.clear()
        hook_output.append(F.adaptive_avg_pool2d(out, 1).flatten(1).detach().cpu())

    # Layer 9 is SPPF (end of backbone) in all YOLOv8 variants
    handle = yolo_model.model[9].register_forward_hook(_hook)

    try:
        for path in img_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_rgb = np.stack([img] * 3, axis=-1)

            # Resize to 640×640 and normalise
            img_resized = cv2.resize(img_rgb, (640, 640))
            tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                yolo_model(tensor)

            if hook_output:
                feats.append(hook_output[0].numpy())   # (1, C)
    finally:
        handle.remove()

    return np.vstack(feats) if feats else np.empty((0, 512))


# ── Main ──────────────────────────────────────────────────────────────────────

def _has_person(lbl_dir: Path, stem: str) -> bool:
    """Return True if the YOLO label file contains a person (class 0) box."""
    p = lbl_dir / f"{stem}.txt"
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) == 0:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flir-weights",  required=True)
    ap.add_argument("--flir-img-dir",  required=True)
    ap.add_argument("--flir-lbl-dir",  required=True)
    ap.add_argument("--llvip-img-dir", required=True)
    ap.add_argument("--llvip-lbl-dir", required=True)
    ap.add_argument("--n-per-domain",  type=int, default=300,
                    help="Images sampled per domain (default 300).")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--out",           default="paper/figs/fig_tsne.pdf")
    ap.add_argument("--device",        default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    flir_img_dir  = Path(args.flir_img_dir)
    flir_lbl_dir  = Path(args.flir_lbl_dir)
    llvip_img_dir = Path(args.llvip_img_dir)
    llvip_lbl_dir = Path(args.llvip_lbl_dir)

    # Sample images
    flir_paths  = sorted(p for p in flir_img_dir.glob("*.jpg")
                         if not p.name.startswith("."))
    llvip_paths = sorted(p for p in llvip_img_dir.glob("*.jpg")
                         if not p.name.startswith("."))

    random.shuffle(flir_paths)
    random.shuffle(llvip_paths)
    flir_sample  = flir_paths[:args.n_per_domain]
    llvip_sample = llvip_paths[:args.n_per_domain]

    # Build person-presence masks
    flir_has_person  = [_has_person(flir_lbl_dir,  p.stem) for p in flir_sample]
    llvip_has_person = [_has_person(llvip_lbl_dir, p.stem) for p in llvip_sample]

    print(f"Extracting features: {len(flir_sample)} FLIR + {len(llvip_sample)} LLVIP images...")
    flir_feats  = extract_features(args.flir_weights, flir_sample,  device=args.device)
    llvip_feats = extract_features(args.flir_weights, llvip_sample, device=args.device)

    all_feats  = np.vstack([flir_feats, llvip_feats])
    n_flir     = len(flir_feats)
    n_llvip    = len(llvip_feats)

    print(f"Running t-SNE on {len(all_feats)} × {all_feats.shape[1]} feature matrix...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    all_feats_scaled = StandardScaler().fit_transform(all_feats)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=args.seed,
                init="pca", learning_rate="auto")
    emb = tsne.fit_transform(all_feats_scaled)

    emb_flir  = emb[:n_flir]
    emb_llvip = emb[n_flir:]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel A: domain separation
    ax = axes[0]
    ax.scatter(emb_flir[:, 0],  emb_flir[:, 1],
               c=FLIR_COLOR,  alpha=0.55, s=8, linewidths=0, label="FLIR ADAS v2")
    ax.scatter(emb_llvip[:, 0], emb_llvip[:, 1],
               c=LLVIP_COLOR, alpha=0.55, s=8, linewidths=0, label="LLVIP")
    ax.set_title("(a) Domain", fontsize=9)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # Panel B: person presence
    ax2 = axes[1]
    flir_hp  = np.array(flir_has_person[:n_flir])
    llvip_hp = np.array(llvip_has_person[:n_llvip])

    # FLIR: person vs no person
    ax2.scatter(emb_flir[flir_hp,  0],  emb_flir[flir_hp,  1],
                c=FLIR_COLOR,   alpha=0.65, s=8, linewidths=0,
                marker="o", label="FLIR (person)")
    ax2.scatter(emb_flir[~flir_hp, 0],  emb_flir[~flir_hp, 1],
                c=FLIR_COLOR,   alpha=0.25, s=8, linewidths=0,
                marker="x", label="FLIR (no person)")
    # LLVIP: all person
    ax2.scatter(emb_llvip[:, 0], emb_llvip[:, 1],
                c=LLVIP_COLOR,  alpha=0.55, s=8, linewidths=0,
                marker="o", label="LLVIP (person)")
    ax2.set_title("(b) Person Presence", fontsize=9)
    ax2.set_xlabel("t-SNE 1", fontsize=8)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.legend(fontsize=6, markerscale=2, loc="upper right")
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Save embeddings + labels for reproducibility
    np_out = out_path.with_suffix(".npz")
    np.savez_compressed(
        str(np_out),
        emb=emb,
        domain=np.array(["flir"] * n_flir + ["llvip"] * n_llvip),
        has_person=np.array(flir_has_person[:n_flir] + llvip_has_person[:n_llvip]),
    )
    print(f"Saved embeddings: {np_out}")


if __name__ == "__main__":
    main()
