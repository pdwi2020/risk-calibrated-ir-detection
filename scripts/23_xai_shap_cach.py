"""23_xai_shap_cach.py — SHAP + Integrated Gradients explanations for CACH.

Three complementary views:
  A) SHAP on the MonotoneCalibrationHead
     Input: (raw_conf, embed_0 … embed_31)  — 33 features
     Target: delta = cal_conf − raw_conf  (calibration correction)
     Shows which embedding dimensions drive the correction and whether
     high/low confidence triggers different adjustments.

  B) Integrated Gradients on CorruptionEmbedNet
     Input: 64×64 image patch
     Target: L2 norm of embedding e (scalar → spatial attribution)
     Shows which image regions drive the corruption-severity encoding.
     Produces one panel per condition (clean / noise / fog).

  C) Corruption cluster in embedding space (PCA)
     Plots the 32-dim embeddings for 500 random images under all conditions,
     coloured by condition — shows CACH has learnt to separate them.

Outputs:
    results/xai/fig_shap_cach.pdf      — panel A
    results/xai/fig_ig_embed.pdf       — panel B
    results/xai/fig_embed_pca.pdf      — panel C
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.calibration.cach import CACH, preprocess_patch
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

CLASS_NAMES = {0: "person", 1: "bike", 2: "car"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_flir_images(flir_root: Path, n: int = 200, seed: int = 42) -> List[np.ndarray]:
    data_dir = flir_root / "images_thermal_val" / "data"
    paths = sorted(data_dir.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    imgs = []
    for p in paths[:n]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            imgs.append(g)
    return imgs


CONDITIONS_PCA = [
    ("clean",   None,             0),
    ("g_noise", "gaussian_noise", 2),
    ("impulse",  "impulse_noise", 2),
    ("fog",     "fog",            3),
    ("blur",    "motion_blur",    2),
]
COND_COLORS = ["#2196F3", "#F44336", "#FF9800", "#9C27B0", "#4CAF50"]


# ── Part A: SHAP on CalibrationHead ─────────────────────────────────────────

def build_calib_features(cach_model: CACH, imgs: List[np.ndarray],
                          preds_dir: Path, model_name: str,
                          n_samples: int = 2000, seed: int = 42,
                          device: str = "cpu") -> tuple:
    """Return X (n_samples × 33), delta (n_samples,) for SHAP.

    X[:,0] = raw_conf; X[:,1:] = embedding (32 dims).
    delta = cal_conf − raw_conf.
    """
    rng = random.Random(seed)
    # Collect raw_conf values from corruption_preds
    all_scores: list[float] = []
    for pred_file in sorted(preds_dir.glob(f"{model_name}_*.json")):
        records = json.loads(pred_file.read_text())
        for rec in records:
            all_scores.extend(rec.get("pred_scores", []))
        if len(all_scores) > n_samples * 10:
            break
    rng.shuffle(all_scores)
    all_scores = all_scores[:n_samples]
    if not all_scores:
        raise ValueError("No prediction scores found in corruption_preds.")

    scores_t = torch.tensor(all_scores, dtype=torch.float32).to(device)
    # Use random images for patches (approximation; CACH embedding is image-driven)
    sampled_imgs = [imgs[i % len(imgs)] for i in range(n_samples)]
    patches = torch.cat([preprocess_patch(img) for img in sampled_imgs]).to(device)  # (N,1,64,64)

    cach_model.eval()
    with torch.no_grad():
        embeds = cach_model.embed_net(patches)          # (N, 32)
        cal_scores = cach_model.calibrate(patches[0:1], scores_t)  # same embed per sample

        # Proper: embed per image
        cal_list = []
        for i in range(0, n_samples, 256):
            p_b = patches[i:i+256]
            s_b = scores_t[i:i+256]
            e_b = cach_model.embed_net(p_b)
            cal_b = cach_model.calib_head(s_b, e_b)
            cal_list.append(cal_b)
        cal_scores = torch.cat(cal_list)

    X = torch.cat([scores_t.unsqueeze(1), embeds], dim=1).cpu().numpy()  # (N, 33)
    delta = (cal_scores - scores_t).cpu().numpy()                          # (N,)
    return X, delta


class CalibHeadWrapper(torch.nn.Module):
    """Wraps MonotoneCalibrationHead so SHAP can call it with (N, 33) tensor."""
    def __init__(self, calib_head):
        super().__init__()
        self.calib_head = calib_head
        self.embed_dim = calib_head.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conf = x[:, 0]
        e    = x[:, 1:]
        return (self.calib_head(conf, e) - conf).unsqueeze(1)  # delta


def plot_shap(X: np.ndarray, delta: np.ndarray, out_path: Path, device: str = "cpu"):
    try:
        import shap
    except Exception as e:
        print(f"  shap import failed: {e} — skipping SHAP figure.")
        return

    feature_names = ["raw_conf"] + [f"embed_{i}" for i in range(X.shape[1] - 1)]

    # Use KernelSHAP (model-agnostic, works with any sklearn-like predict)
    # Subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    X_sub = X[idx]

    # Simple numpy predict function (much faster than torch for KernelSHAP)
    # We'll load model weights into a simple function
    def predict_delta(x_arr: np.ndarray) -> np.ndarray:
        return delta[idx[:len(x_arr)]] if len(x_arr) <= len(idx) else np.zeros(len(x_arr))

    # Use the real calibration head
    def predict_real(x_arr: np.ndarray) -> np.ndarray:
        t = torch.tensor(x_arr, dtype=torch.float32)
        conf = t[:, 0]
        e    = t[:, 1:].to(device)
        conf = conf.to(device)
        with torch.no_grad():
            cal = calib_head_global(conf, e)
            return (cal - conf).cpu().numpy()

    n_bg = min(100, len(X_sub))
    explainer = shap.KernelExplainer(predict_real, X_sub[:n_bg])
    shap_values = explainer.shap_values(X_sub[:200], nsamples=100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: mean |SHAP| bar plot (top 15 features)
    mean_abs = np.abs(shap_values).mean(0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    axes[0].barh(range(len(top_names)), top_vals[::-1],
                 color=plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(top_names))))
    axes[0].set_yticks(range(len(top_names)))
    axes[0].set_yticklabels(top_names[::-1], fontsize=9)
    axes[0].set_xlabel("Mean |SHAP value|  (calibration correction)", fontsize=10)
    axes[0].set_title("Feature Importance (SHAP)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: SHAP scatter for raw_conf (most interpretable)
    conf_shap = shap_values[:, 0]
    conf_vals = X_sub[:200, 0]
    sc = axes[1].scatter(conf_vals, conf_shap, c=conf_vals,
                         cmap="RdYlGn", alpha=0.6, s=20, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label="raw_conf")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("Raw confidence", fontsize=10)
    axes[1].set_ylabel("SHAP value (effect on calibration correction)", fontsize=10)
    axes[1].set_title("SHAP: raw_conf effect", fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.3)

    fig.suptitle("SHAP Explanation of CACH Calibration Head\n"
                 "Target: calibration correction = cal_conf − raw_conf",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP figure: {out_path}")


# Need global handle for KernelExplainer
calib_head_global = None


# ── Part B: Integrated Gradients on CorruptionEmbedNet ──────────────────────

def integrated_gradients(model_fn, x: torch.Tensor, baseline: torch.Tensor,
                          n_steps: int = 50) -> torch.Tensor:
    """Vanilla IG: returns attributions of same shape as x."""
    x.requires_grad_(False)
    alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)
    interp = baseline + alphas.view(-1, 1, 1, 1) * (x - baseline)  # (n_steps+1, C, H, W)
    interp.requires_grad_(True)
    out = model_fn(interp)  # (n_steps+1, embed_dim)
    # Scalar: L2 norm of embedding
    scalar = out.norm(dim=1).sum()
    scalar.backward()
    grads = interp.grad  # (n_steps+1, C, H, W)
    # Trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]).mean(0)  # (C, H, W)
    attr = (x - baseline) * avg_grads             # (C, H, W)
    return attr.detach()


def plot_ig(cach_model: CACH, flir_root: Path, out_path: Path, device: str = "cpu"):
    cond_pairs = [
        ("Clean",          None,             0),
        ("Gaussian Noise\n(sev 2)", "gaussian_noise", 2),
        ("Fog (sev 3)",    "fog",            3),
    ]

    # Pick one representative image
    data_dir = flir_root / "images_thermal_val" / "data"
    paths = sorted(data_dir.glob("*.jpg"))
    img_gray = cv2.imread(str(paths[7]), cv2.IMREAD_GRAYSCALE)

    baseline = torch.zeros(1, 1, 64, 64, device=device)
    embed_fn = lambda x: cach_model.embed_net(x)

    fig, axes = plt.subplots(2, len(cond_pairs), figsize=(5 * len(cond_pairs), 9))

    for ci, (label, corr_name, sev) in enumerate(cond_pairs):
        if corr_name is None:
            img_c = img_gray.copy()
        else:
            img_c = CORRUPTION_REGISTRY[corr_name](img_gray, sev)

        patch = preprocess_patch(img_c).to(device)  # (1,1,64,64)

        # IG attribution
        attr = integrated_gradients(embed_fn, patch.clone(), baseline, n_steps=50)
        attr_np = attr.squeeze().cpu().numpy()  # (64,64)
        # Absolute attribution (unsigned importance)
        attr_abs = np.abs(attr_np)
        attr_abs = (attr_abs - attr_abs.min()) / (attr_abs.max() + 1e-8)

        # Row 0: corrupted image
        img_show = cv2.resize(img_c.astype(np.uint8), (64, 64))
        axes[0, ci].imshow(img_show, cmap="gray", vmin=0, vmax=255)
        axes[0, ci].set_title(label, fontsize=10, fontweight="bold")
        axes[0, ci].axis("off")
        if ci == 0:
            axes[0, ci].set_ylabel("Input patch", fontsize=9)

        # Row 1: IG heatmap overlay
        img_rgb = np.stack([img_show]*3, axis=-1).astype(np.float32) / 255.0
        heatmap = plt.cm.hot(attr_abs)[:, :, :3]
        overlay = 0.55 * img_rgb + 0.45 * heatmap
        overlay = np.clip(overlay, 0, 1)
        axes[1, ci].imshow(overlay)
        axes[1, ci].axis("off")
        if ci == 0:
            axes[1, ci].set_ylabel("IG attribution", fontsize=9)

    fig.suptitle("Integrated Gradients — CorruptionEmbedNet\n"
                 "Regions driving the corruption-severity embedding",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IG figure: {out_path}")


# ── Part C: Embedding PCA ────────────────────────────────────────────────────

def plot_embed_pca(cach_model: CACH, imgs: List[np.ndarray],
                   out_path: Path, device: str = "cpu", n_per_cond: int = 100):
    all_embeds, all_labels = [], []

    for ci, (label, corr_name, sev) in enumerate(CONDITIONS_PCA):
        batch_imgs = imgs[:n_per_cond]
        patches = []
        for g in batch_imgs:
            if corr_name:
                g = CORRUPTION_REGISTRY[corr_name](g, sev)
            patches.append(preprocess_patch(g))
        patches_t = torch.cat(patches).to(device)  # (N,1,64,64)
        with torch.no_grad():
            e = cach_model.embed_net(patches_t).cpu().numpy()
        all_embeds.append(e)
        all_labels.extend([ci] * len(e))

    X_all = np.vstack(all_embeds)
    y_all = np.array(all_labels)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(7, 6))
    for ci, (label, _, _) in enumerate(CONDITIONS_PCA):
        mask = y_all == ci
        ax.scatter(Z[mask, 0], Z[mask, 1], c=COND_COLORS[ci],
                   label=label, alpha=0.6, s=20)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax.set_title("CACH CorruptionEmbedNet — PCA of Embeddings\n"
                 "(Each point = one image; colour = condition)", fontsize=11)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding PCA: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global calib_head_global

    ap = argparse.ArgumentParser()
    ap.add_argument("--cach-checkpoint", required=True, type=Path,
                    help="results/cach/cach_best.pt")
    ap.add_argument("--flir-root",       required=True, type=Path)
    ap.add_argument("--preds-dir",       required=True, type=Path)
    ap.add_argument("--out",             default="results/xai", type=Path)
    ap.add_argument("--model-name",      default="yolov8m")
    ap.add_argument("--device",          default="cpu")
    ap.add_argument("--skip-shap",       action="store_true",
                    help="Skip slow SHAP computation (for quick test)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading CACH from {args.cach_checkpoint} ...")
    cach = CACH.load(str(args.cach_checkpoint), map_location=device)
    cach.eval()
    calib_head_global = cach.calib_head

    imgs = load_flir_images(args.flir_root, n=200)
    print(f"Loaded {len(imgs)} FLIR images")

    # Part B: Integrated Gradients (fast, run first)
    print("\n--- Part B: Integrated Gradients on CorruptionEmbedNet ---")
    plot_ig(cach, args.flir_root, args.out / "fig_ig_embed.pdf", device=device)

    # Part C: PCA of embeddings (fast)
    print("\n--- Part C: Embedding PCA ---")
    plot_embed_pca(cach, imgs, args.out / "fig_embed_pca.pdf", device=device)

    # Part A: SHAP (slow — skip with --skip-shap for quick tests)
    if not args.skip_shap:
        print("\n--- Part A: SHAP on CalibrationHead ---")
        try:
            X, delta = build_calib_features(
                cach, imgs, args.preds_dir, args.model_name, n_samples=2000, device=device)
            plot_shap(X, delta, args.out / "fig_shap_cach.pdf", device=device)
        except Exception as e:
            print(f"  SHAP failed: {e}  (run with --skip-shap to skip)")
    else:
        print("  SHAP skipped (--skip-shap)")

    print(f"\nAll XAI outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
