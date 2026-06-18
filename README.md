# Risk-Calibrated Infrared Object Detection under Low-SNR and Distribution Shift

**Authors:** Paritosh Dwivedi and Rajkumar S.
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India.

---

## Overview

This repository contains the research code for a method and evaluation framework for reliable object detection in thermal-infrared (IR) imagery under low-SNR conditions and distribution shift.

The core method is **CACH (Corruption-Adaptive Calibration Head)**: a ~20 K-parameter lightweight calibrator that attaches post-hoc to a frozen detector. A small CNN embeds corruption type and severity directly from the input image patch (no explicit corruption label required), and a monotonic MLP maps raw detection confidence to a calibrated score — all without retraining the backbone.

The evaluation framework integrates four complementary safety mechanisms:
- **Cost-sensitive confidence-threshold optimisation** — minimises a safety-weighted cost C = 10·FN + FP + 2·LocErr + 0.5·Defer.
- **Post-hoc isotonic calibration** — non-parametric reliability correction on held-out predictions.
- **Distribution-free conformal risk control (CRC)** — finite-sample guarantee on the missed-detection rate at a user-specified level alpha (Angelopoulos et al., 2022).
- **Risk-Adjusted Average Precision (RA-AP)** — a composite metric that penalises AP for residual calibration error and threshold instability.

The full pipeline is evaluated on three public thermal-IR benchmarks — **FLIR ADAS v2**, **LLVIP**, and **KAIST** — across four detectors: YOLOv8m, RT-DETR, Faster R-CNN, and RetinaNet.

> **Status:** Code accompanying a manuscript currently under review. Trained checkpoints, results files, and the paper are not included in this repository.

---

## Repository Structure

```
.
├── src/                         Core library (importable package)
│   ├── calibration/
│   │   ├── cach.py              CACH architecture: CorruptionEmbedNet + CalibrationHead
│   │   ├── isotonic.py          Post-hoc isotonic regression calibration
│   │   ├── temperature_scaling.py  Temperature scaling baseline
│   │   ├── mc_dropout.py        MC-Dropout uncertainty estimation
│   │   └── tta.py               Test-time augmentation wrapper
│   ├── corruption/
│   │   └── corruption_pipeline.py  Synthetic corruption transforms (noise, fog, blur, etc.)
│   ├── data/
│   │   ├── flir_dataset.py      FLIR ADAS v2 PyTorch dataset
│   │   ├── llvip_dataset.py     LLVIP PyTorch dataset
│   │   ├── llvip_splits.py      LLVIP train/val/test split utilities
│   │   └── splits.py            Generic split helpers
│   ├── detectors/
│   │   ├── yolov8_wrapper.py    Unified inference API — YOLOv8m
│   │   ├── yolov11_wrapper.py   Unified inference API — YOLOv11
│   │   ├── rtdetr_wrapper.py    Unified inference API — RT-DETR
│   │   ├── faster_rcnn_wrapper.py  Unified inference API — Faster R-CNN
│   │   └── retinanet_wrapper.py    Unified inference API — RetinaNet
│   └── risk/
│       ├── conformal.py         Conformal risk control (CRC) — miss-rate guarantee
│       └── cost_sensitive.py    Cost-sensitive threshold optimiser
│
├── scripts/                     End-to-end pipeline scripts (numbered by stage)
│   ├── 00_dataset_audit.py      Stage 0: Verify FLIR/LLVIP/KAIST splits and annotations
│   ├── 01_train_yolo.py         Stage 1a: Fine-tune YOLOv8m on FLIR ADAS v2
│   ├── 02_train_frcnn.py        Stage 1b: Fine-tune Faster R-CNN
│   ├── 03_train_retina_rtdetr.py Stage 1c: Fine-tune RetinaNet + RT-DETR
│   ├── 04_eval_detector.py      Stage 2: Evaluate all detectors (AP, AR, F1)
│   ├── 05_risk_eval.py          Stage 3: CRC threshold calibration + risk metrics
│   ├── 06_finish_retinanet.py   Stage 1d: Resume/finish RetinaNet training
│   ├── 07_detector_comparison.py Stage 2b: Cross-detector comparison table
│   ├── 08_cost_ablation.py      Stage 3b: Cost-ratio ablation (c_FN sweep)
│   ├── 09_bootstrap_ci.py       Stage 3c: Bootstrap 95% CIs for all metrics
│   ├── 10_run_corruption_eval.py Stage 4: Run all 6×4 corruption conditions
│   ├── 11_domain_transfer.py    Stage 5a: FLIR→LLVIP domain transfer evaluation
│   ├── 12_train_llvip.py        Stage 5b: Fine-tune detectors on LLVIP
│   ├── 13_tsne_features.py      Stage 5c: t-SNE of detector features across domains
│   ├── 14_corruption_calibration.py Stage 6a: Post-hoc calibration under corruption
│   ├── 14a_corruption_infer.py  Stage 6b: Cache per-detection records for CACH training
│   ├── 15_micro_calibration.py  Stage 6c: Cross-sensor micro-calibration (KAIST)
│   ├── 16_cost_ratio_grid.py    Stage 7a: c_FN/c_FP grid search
│   ├── 17_conditional_thresholds.py Stage 7b: Condition-specific optimal thresholds
│   ├── 18_raap_variants.py      Stage 7c: RA-AP ablation (penalty weight variants)
│   ├── 19_deferral_curve.py     Stage 7d: Deferral cost–recall tradeoff curve
│   ├── 20_train_cach.py         Stage 8a: Train CACH head (primary)
│   ├── 20b_train_cach_balanced.py   Stage 8b: CACH with TP/TN class-balanced sampling
│   ├── 20c_train_cach_efficient.py  Stage 8c: CACH lightweight variant
│   ├── 20d_train_cach_fast.py   Stage 8d: CACH fast-training schedule
│   ├── 21_train_yolov11.py      Stage 8e: YOLOv11 detector variant
│   ├── 22_xai_gradcam.py        Stage 9a: GradCAM saliency maps (detector)
│   ├── 23_xai_shap_cach.py      Stage 9b: SHAP analysis of CACH calibration decisions
│   ├── 24_evaluate_cach.py      Stage 9c: Full CACH evaluation (ECE, RA-AP, CRC)
│   ├── 25_cach_bootstrap_ci.py  Stage 9d: Bootstrap CIs for CACH vs baselines
│   ├── generate_corruption_plots.py  Figure generation — corruption performance curves
│   ├── generate_figures.py      Figure generation — paper figures (all stages)
│   ├── _torch_train_utils.py    Shared training utilities (LR scheduler, checkpointing)
│   ├── gpu_bootstrap.sh         GPU node setup script (RunPod / Vast.ai)
│   ├── vast_kaist_microcal.sh   Vast.ai launch: KAIST micro-calibration job
│   ├── vast_phase7_gpu.sh       Vast.ai launch: Phase 7 GPU job
│   └── vast_retinanet_finish_20260606.sh  Vast.ai launch: RetinaNet finish job
│
├── configs/                     YAML configs and dataset split files
│   ├── yolov8.yaml              YOLOv8m training config
│   ├── yolov11.yaml             YOLOv11 training config
│   ├── faster_rcnn.yaml         Faster R-CNN training config
│   ├── corruption.yaml          Corruption types and severity levels
│   ├── risk.yaml                CRC alpha, cost ratios, IoU thresholds
│   ├── flir_yolo.yaml           FLIR ADAS v2 YOLO dataset descriptor
│   ├── flir_yolo_calib.yaml     FLIR calibration split descriptor
│   ├── flir_yolo_test.yaml      FLIR test split descriptor
│   ├── llvip_yolo.yaml          LLVIP YOLO dataset descriptor
│   ├── flir_val_split.json      FLIR validation image list (deterministic split)
│   ├── llvip_val_split.json     LLVIP validation image list
│   ├── flir_val_calibration_images.txt  CRC calibration split image list
│   └── flir_val_test_images.txt CRC test split image list
│
└── notebooks/                   Exploratory Jupyter notebooks (one per paper phase)
    ├── 00_dataset_audit.ipynb
    ├── 01_yolov8_baseline.ipynb
    ├── 02_faster_rcnn.ipynb
    ├── 03_retinanet_rtdetr.ipynb
    ├── 04_corruption_pipeline.ipynb
    ├── 05_calibration_uq.ipynb
    ├── 06_cost_sensitive_risk.ipynb
    └── 07_domain_shift_eval.ipynb
```

---

## Installation

```bash
git clone https://github.com/pdwi2020/risk-calibrated-ir-detection.git
cd risk-calibrated-ir-detection
pip install -r requirements.txt
```

**Requirements summary:** Python 3.10, PyTorch >= 2.1.0, torchvision >= 0.16.0, ultralytics >= 8.0.0 (YOLOv8/v11 + RT-DETR), pycocotools, scikit-learn, scipy, albumentations, opencv-python.

---

## Datasets

The following datasets are **not included** in this repository. Download them from their official sources and adjust the root paths in `configs/` accordingly (several scripts also accept a `--flir-root`, `--llvip-root`, or `--kaist-root` flag):

| Dataset | Source | Notes |
|---|---|---|
| FLIR ADAS v2 | https://www.flir.com/oem/adas/adas-dataset-form/ | Primary benchmark; COCO annotations |
| LLVIP | https://bupt-ai-cz.github.io/LLVIP/ | Low-light visible-IR pairs |
| KAIST | https://soonminhwang.github.io/rgbt-ped-detection/ | RGB-T pedestrian, used for cross-sensor micro-calibration |

---

## Pipeline Usage

The numbered scripts in `scripts/` define a linear pipeline. Run them from the project root with the dataset root(s) as arguments. A GPU (CUDA) is required for Stages 1–8; Stages 3, 7, and part of 9 are CPU-only.

### Stage 0 — Dataset audit
```bash
python scripts/00_dataset_audit.py --x9 /path/to/data
```

### Stage 1 — Detector fine-tuning
```bash
python scripts/01_train_yolo.py --flir-root /path/to/flir/FLIR_ADAS_v2
python scripts/02_train_frcnn.py --flir-root /path/to/flir/FLIR_ADAS_v2
python scripts/03_train_retina_rtdetr.py --flir-root /path/to/flir/FLIR_ADAS_v2
```

### Stage 2 — Detector evaluation
```bash
python scripts/04_eval_detector.py --flir-root /path/to/flir/FLIR_ADAS_v2 \
    --checkpoint results/yolov8m_flir_best.pt
python scripts/07_detector_comparison.py
```

### Stage 3 — Cost/risk calibration
```bash
python scripts/05_risk_eval.py --split-json configs/flir_val_split.json
python scripts/08_cost_ablation.py
python scripts/09_bootstrap_ci.py
```

### Stage 4 — Corruption robustness
```bash
python scripts/10_run_corruption_eval.py --flir-root /path/to/flir/FLIR_ADAS_v2
```

### Stage 5 — Domain transfer
```bash
python scripts/11_domain_transfer.py
python scripts/12_train_llvip.py --llvip-root /path/to/llvip
python scripts/13_tsne_features.py
```

### Stage 6 — Post-hoc calibration + CACH data prep
```bash
python scripts/14_corruption_calibration.py
python scripts/14a_corruption_infer.py --flir-root /path/to/flir/FLIR_ADAS_v2
python scripts/15_micro_calibration.py --kaist-root /path/to/kaist
```

### Stage 7 — Risk metric ablations
```bash
python scripts/16_cost_ratio_grid.py
python scripts/17_conditional_thresholds.py
python scripts/18_raap_variants.py
python scripts/19_deferral_curve.py
```

### Stage 8 — CACH training
```bash
python scripts/20_train_cach.py \
    --flir-root /path/to/flir/FLIR_ADAS_v2 \
    --preds-dir results/corruption_preds \
    --split-json configs/flir_val_split.json \
    --out results/cach \
    --epochs 30 --batch 256 --lr 1e-3
```

### Stage 9 — XAI + CACH evaluation
```bash
python scripts/22_xai_gradcam.py
python scripts/23_xai_shap_cach.py
python scripts/24_evaluate_cach.py --cach-ckpt results/cach/cach_best.pt
python scripts/25_cach_bootstrap_ci.py
```

### Figure generation
```bash
python scripts/generate_corruption_plots.py
python scripts/generate_figures.py
```

---

## Key Modules

### `src/calibration/cach.py` — CACH architecture

```python
from src.calibration.cach import CorruptionEmbedNet, CalibrationHead
```

- `CorruptionEmbedNet` — small CNN (input: 64×64 grayscale patch, output: embedding of dim 32). ~12 K parameters.
- `CalibrationHead` — monotonic MLP mapping (conf, embedding) → calibrated confidence. Monotonicity enforced via non-negative weight parameterisation.

### `src/risk/conformal.py` — Conformal risk control

Implements the CRC threshold selector from Angelopoulos et al. (2022): given a calibration set of size n, finds the strictest threshold lambda such that the finite-sample-corrected missed-detection rate satisfies E[R_test] <= alpha.

### `src/risk/cost_sensitive.py` — Cost-sensitive threshold

Minimises expected cost C = 10·FN + FP + 2·LocErr + 0.5·Defer over a grid of confidence thresholds on the calibration split.

---

## Citation

If you use this code, please cite the accompanying paper (BibTeX will be added upon publication).

---

## License

Research code accompanying the manuscript; license to be finalized upon publication.
