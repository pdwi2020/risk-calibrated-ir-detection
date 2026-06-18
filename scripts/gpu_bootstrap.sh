#!/usr/bin/env bash
# gpu_bootstrap.sh -- one-time setup on a freshly-rented cloud GPU instance.
#
# Prepares the Python environment and pulls the FLIR + LLVIP datasets from
# Kaggle for the MV Paper (risk-calibrated IR detection). Idempotent: safe to
# re-run; populated targets are skipped.
#
# Works on BOTH:
#   - vast.ai  (PRIMARY; RTX 3090; base image
#               pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime; no /workspace)
#   - RunPod   (RTX 4090; /workspace network volume present)
#
# Configurable env vars (defaults in parentheses):
#   DATA_DIR           data root         (/workspace/data if writable, else ./data)
#   CKPT_DIR           checkpoint root   (/workspace/checkpoints if writable, else ./checkpoints)
#   FLIR_KAGGLE_SLUG   FLIR dataset slug (paritoshdwi2019/flir-adas-thermal-private)
#   LLVIP_KAGGLE_SLUG  LLVIP dataset slug(paritoshdwi2019/llvip-yolo-private)
#   KAGGLE_USERNAME / KAGGLE_KEY  -- used to write ~/.kaggle/kaggle.json if absent
#
# Usage:  bash gpu_bootstrap.sh

set -euo pipefail

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 0. Resolve writable dirs ------------------------------------------------
_pick_dir() {
  # $1 = preferred path, $2 = fallback. Echo whichever is creatable+writable.
  local pref="$1" fb="$2"
  if mkdir -p "$pref" 2>/dev/null && [ -w "$pref" ]; then
    printf '%s' "$pref"
  else
    printf '%s' "$fb"
  fi
}

DATA_DIR="${DATA_DIR:-$(_pick_dir /workspace/data ./data)}"
CKPT_DIR="${CKPT_DIR:-$(_pick_dir /workspace/checkpoints ./checkpoints)}"
FLIR_KAGGLE_SLUG="${FLIR_KAGGLE_SLUG:-paritoshdwi2019/flir-adas-thermal-private}"
LLVIP_KAGGLE_SLUG="${LLVIP_KAGGLE_SLUG:-paritoshdwi2019/llvip-yolo-private}"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

# --- 1. GPU check ------------------------------------------------------------
banner "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found -- is this a GPU instance?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- 2. Python dependencies --------------------------------------------------
banner "Python dependencies"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "torch not present -- installing CUDA 12.1 wheels..."
  "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
"$PY" -m pip install -q \
  ultralytics pycocotools albumentations opencv-python-headless mapie scipy scikit-learn

"$PY" - <<'PYEOF'
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available to torch")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB")
PYEOF

# --- 3. Kaggle credentials ---------------------------------------------------
banner "Kaggle credentials"
command -v kaggle >/dev/null 2>&1 || "$PY" -m pip install -q kaggle
KCFG="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KCFG" ]; then
  if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    mkdir -p "$HOME/.kaggle"
    printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$KCFG"
    echo "Wrote $KCFG from environment."
  else
    echo "ERROR: no Kaggle credentials found." >&2
    echo "  Supply ~/.kaggle/kaggle.json, OR set KAGGLE_USERNAME and KAGGLE_KEY." >&2
    exit 1
  fi
fi
chmod 600 "$KCFG" 2>/dev/null || true

# --- 4. Pull datasets --------------------------------------------------------
pull() {
  # $1 = kaggle slug, $2 = target subdir under DATA_DIR
  local slug="$1" name="$2" dest
  dest="$DATA_DIR/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "[$name] already populated at $dest -- skipping."
    return 0
  fi
  mkdir -p "$dest"
  echo "[$name] downloading $slug -> $dest"
  kaggle datasets download -d "$slug" -p "$dest" --unzip
}
banner "Datasets"
pull "$FLIR_KAGGLE_SLUG" flir
pull "$LLVIP_KAGGLE_SLUG" llvip

# --- 5. Summary --------------------------------------------------------------
banner "Summary"
echo "DATA_DIR=$DATA_DIR"
ls -1 "$DATA_DIR" 2>/dev/null || true
echo "CKPT_DIR=$CKPT_DIR"
echo
echo "Next: python scripts/01_train_yolo.py --data configs/flir_yolo.yaml"
