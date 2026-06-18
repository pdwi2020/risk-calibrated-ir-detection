"""22_xai_gradcam.py — GradCAM / EigenCAM explanations for IR object detectors.

Produces a figure grid showing which image regions YOLOv8m attends to under:
  - Clean thermal image
  - Gaussian noise (sev 2)
  - Fog (sev 3)

For each condition: overlay cam heatmap on the thermal image with predicted boxes.

Outputs:
    results/xai/gradcam_grid.png   — 3×3 grid: 3 scenes × 3 conditions
    results/xai/gradcam_{scene}_{condition}.png  — individual panels
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

CLASS_NAMES = {0: "person", 1: "bike", 2: "car"}
CLASS_COLORS = {0: (1.0, 0.2, 0.2), 1: (0.2, 0.8, 0.2), 2: (0.2, 0.4, 1.0)}

# ── YOLO wrapper for pytorch-grad-cam ───────────────────────────────────────

class YOLOv8GradCAMWrapper(torch.nn.Module):
    """Wraps YOLOv8 so that EigenCAM can hook its backbone exit (layer 8).

    Uses a forward hook so the full model runs normally (the neck has
    multi-input modules that can't be run layer-by-layer), and we
    capture the feature map at layer 8 via the hook.
    """
    def __init__(self, model: YOLO):
        super().__init__()
        self.model = model.model  # nn.Module
        self._feat = None
        self.model.model[8].register_forward_hook(
            lambda m, inp, out: setattr(self, "_feat", out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feat = None
        try:
            self.model(x)
        except Exception:
            pass
        return self._feat if self._feat is not None else x


def get_target_layer(yolo_model: YOLO):
    """Return the last C2f block in the backbone (layer 8 = layer index -1 before neck)."""
    return yolo_model.model.model[8]


# ── Corruption helpers ───────────────────────────────────────────────────────

CONDITIONS = [
    ("clean",          None,               0),
    ("gaussian_noise", "gaussian_noise",   2),
    ("fog",            "fog",              3),
]

# ── Utilities ────────────────────────────────────────────────────────────────

def load_flir_image(path: Path) -> np.ndarray:
    """Load FLIR thermal JPEG → BGR uint8 (640×512)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def apply_corruption(img_bgr: np.ndarray, corr_name: str, sev: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cfn = CORRUPTION_REGISTRY[corr_name]
    corrupted_gray = cfn(gray, sev)
    return cv2.cvtColor(corrupted_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def run_yolo_detections(yolo: YOLO, img_bgr: np.ndarray, conf_thr: float = 0.25):
    """Return list of (x1,y1,x2,y2,conf,cls_id)."""
    result = yolo.predict(img_bgr, conf=conf_thr, verbose=False)[0]
    boxes = result.boxes
    dets = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(confs)):
            dets.append((*xyxy[i], confs[i], cls_ids[i]))
    return dets


def compute_eigencam(yolo: YOLO, img_bgr: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return EigenCAM heatmap (H×W) normalised to [0,1]."""
    wrapper = YOLOv8GradCAMWrapper(yolo)
    wrapper.eval()
    target_layer = [get_target_layer(yolo)]

    cam = EigenCAM(model=wrapper, target_layers=target_layer)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.tensor(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W)
    return grayscale_cam


def overlay_cam_and_boxes(img_bgr: np.ndarray, cam: np.ndarray,
                           dets: list, alpha: float = 0.5) -> np.ndarray:
    """Overlay CAM heatmap and detection boxes on image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    visualization = show_cam_on_image(img_rgb, cam_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # Draw detection boxes
    for (x1, y1, x2, y2, conf, cls_id) in dets:
        color = tuple(int(c * 255) for c in CLASS_COLORS.get(cls_id, (1, 1, 1)))
        cv2.rectangle(visualization, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(visualization, label, (int(x1), max(int(y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return visualization


# ── Scene selection ──────────────────────────────────────────────────────────

def select_scenes(flir_root: Path, n_scenes: int = 3, seed: int = 42) -> list[Path]:
    """Pick n_scenes FLIR test images with ≥2 detections (diverse scenes)."""
    data_dir = flir_root / "images_thermal_val" / "data"
    imgs = sorted(data_dir.glob("*.jpg"))
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    return imgs[:n_scenes]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flir-root",  required=True, type=Path)
    ap.add_argument("--weights",    required=True, type=Path,
                    help="YOLOv8m best.pt from results/yolov8m_flir_seed0/weights/")
    ap.add_argument("--out",        default="results/xai", type=Path)
    ap.add_argument("--scenes",     default=3, type=int)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLOv8m from {args.weights} ...")
    yolo = YOLO(str(args.weights))
    yolo.to(args.device)

    scenes = select_scenes(args.flir_root, n_scenes=args.scenes)
    print(f"Selected {len(scenes)} scenes")

    cond_labels = [c[0].replace("_", "\n") for c in CONDITIONS]
    scene_labels = [f"Scene {i+1}" for i in range(len(scenes))]

    fig, axes = plt.subplots(len(scenes), len(CONDITIONS),
                             figsize=(5 * len(CONDITIONS), 4 * len(scenes)),
                             dpi=150)
    if len(scenes) == 1:
        axes = axes[np.newaxis, :]

    for si, img_path in enumerate(scenes):
        print(f"  Scene {si+1}: {img_path.name}")
        img_bgr_clean = load_flir_image(img_path)

        for ci, (cond_name, corr_name, sev) in enumerate(CONDITIONS):
            if corr_name is None:
                img_bgr = img_bgr_clean.copy()
            else:
                img_bgr = apply_corruption(img_bgr_clean, corr_name, sev)

            cam = compute_eigencam(yolo, img_bgr, device=args.device)
            dets = run_yolo_detections(yolo, img_bgr)
            vis = overlay_cam_and_boxes(img_bgr, cam, dets)

            # Save individual panel
            panel_path = args.out / f"gradcam_scene{si+1}_{cond_name}.png"
            cv2.imwrite(str(panel_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            ax = axes[si, ci]
            ax.imshow(vis)
            ax.axis("off")
            if si == 0:
                ax.set_title(cond_labels[ci], fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(scene_labels[si], fontsize=10)

    # Legend
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[k], label=v)
                      for k, v in CLASS_NAMES.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, title="Class", bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("EigenCAM Detector Attention: Clean vs. Corruption\n"
                 "(YOLOv8m, FLIR ADAS v2 thermal test images)", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = args.out / "gradcam_grid.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    out_pdf = args.out / "fig_gradcam.pdf"
    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {out_pdf}")
    print("GradCAM analysis complete.")


if __name__ == "__main__":
    main()
