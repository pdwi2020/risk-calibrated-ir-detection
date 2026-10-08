# Risk-Calibrated Infrared Object Detection under Low Signal-to-Noise Ratio and Distribution Shift

**Authors:** Paritosh Dwivedi and Rajkumar Soundrapandiyan.
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India.

---

## Overview

This repository contains the code, configurations, frozen data split and Lean 4 proofs for a post-hoc framework that turns the scores of a thermal-infrared (IR) object detector into decisions with stated, bounded risk. No detector retraining is needed.

The framework has four parts:
- **Scene-level conformal risk control.** A distribution-free rule on the per-image miss event (Angelopoulos et al., 2022). When the requested miss rate is not certifiable, the rule says so and reports the certifiable miss-rate frontier instead of passing off the loosest threshold.
- **Cost-sensitive confidence threshold.** Minimises the expected per-image cost C = c_FN·FN + c_FP·FP + c_Loc·L_loc + c_Def·Def, with c_FN = 10, c_FP = 1, c_Loc = 2, c_Def = 0.5.
- **Post-hoc isotonic calibration**, including corruption-pooled and few-shot cross-sensor recalibration.
- **Risk-Adjusted Average Precision (RA-AP)**, defined as RA-AP = mAP − C̄(θ*) / (c_FN · ḡ), where ḡ is the mean number of ground-truth objects per image. The denominator is the per-image cost of detecting nothing, so RA-AP has no tuning constant and is invariant to object density.

The repository also contains **CACH (Corruption-Adaptive Calibration Head)**, a learned module of about 20K parameters that infers the corruption from an image patch and predicts a per-image temperature and shift. Under the leakage-free, per-detector protocol, CACH does **not** beat a corruption-pooled isotonic baseline (mean ECE 0.098 vs. 0.068). The paper reports this as a negative result; the code is here so that it can be reproduced.

Everything is evaluated with four detectors (YOLOv8m, RT-DETR, Faster R-CNN, RetinaNet) on FLIR ADAS v2, LLVIP and KAIST, over six corruption types at four severities.

### Headline results (FLIR ADAS v2, frozen 572/572 calibration/test split)

| Result | Value |
|---|---|
| Expected-cost reduction of the cost-optimal threshold (YOLOv8m, four seeds) | 32.7 ± 0.6 % |
| ECE after isotonic calibration, all four detectors | ≤ 0.012 |
| Conformal miss-rate target α = 10 % | infeasible for all four detectors |
| Lowest certifiable per-image miss rate (RT-DETR) | 0.281 |
| RA-AP, clean (RT-DETR, highest) | 0.569 |
| CACH vs. corruption-pooled isotonic, mean ECE | 0.098 vs. 0.068 |
| 5-fold CV: YOLOv8m cost reduction / RT-DETR miss floor | 31.4 ± 1.7 % / 0.271 ± 0.034 |

> **Status:** Code accompanying a manuscript under review. Trained checkpoints, prediction caches and result files are not included.

---

## Repository Structure

```
.
├── src/                         Core library (importable package)
│   ├── calibration/
│   │   ├── cach.py              CACH: CorruptionEmbedNet + MonotoneCalibrationHead, losses, GT matching
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
│   │   ├── yolov8_wrapper.py    Unified inference API: YOLOv8m
│   │   ├── yolov11_wrapper.py   Unified inference API: YOLOv11
│   │   ├── rtdetr_wrapper.py    Unified inference API: RT-DETR
│   │   ├── faster_rcnn_wrapper.py  Unified inference API: Faster R-CNN
│   │   └── retinanet_wrapper.py    Unified inference API: RetinaNet
│   └── risk/
│       ├── conformal.py         Scene-level conformal risk control (miss-rate guarantee, infeasibility gate)
│       └── cost_sensitive.py    Cost-sensitive threshold optimiser
│
├── scripts/                     End-to-end pipeline scripts (numbered by stage)
│   ├── 00_dataset_audit.py      Stage 0: Verify FLIR/LLVIP/KAIST splits and annotations
│   ├── 01_train_yolo.py         Stage 1a: Fine-tune YOLOv8m on FLIR ADAS v2
│   ├── 02_train_frcnn.py        Stage 1b: Fine-tune Faster R-CNN
│   ├── 03_train_retina_rtdetr.py Stage 1c: Fine-tune RetinaNet + RT-DETR
│   ├── 04_eval_detector.py      Stage 2: Evaluate all detectors (AP, AR, F1)
│   ├── 05_risk_eval.py          Stage 3: Cost/risk evaluation on the calibration/test split
│   ├── 06_finish_retinanet.py   Stage 1d: Resume/finish RetinaNet training
│   ├── 07_detector_comparison.py Stage 2b: Cross-detector comparison table
│   ├── 08_cost_ablation.py      Stage 3b: Cost-ratio ablation (c_FN x c_FP grid)
│   ├── 09_bootstrap_ci.py       Stage 3c: Bootstrap 95% CIs for all metrics
│   ├── 10_run_corruption_eval.py Stage 4: Run all 6x4 corruption conditions
│   ├── 11_domain_transfer.py    Stage 5a: FLIR->LLVIP domain transfer evaluation
│   ├── 12_train_llvip.py        Stage 5b: Fine-tune detectors on LLVIP
│   ├── 13_tsne_features.py      Stage 5c: t-SNE of detector features across domains
│   ├── 14_corruption_calibration.py Stage 6a: Post-hoc calibration under corruption
│   ├── 14a_corruption_infer.py  Stage 6b: Cache per-detection records for CACH training
│   ├── 15_micro_calibration.py  Stage 6c: Few-shot cross-sensor recalibration (KAIST)
│   ├── 16_cost_ratio_grid.py    Stage 7a: c_FN/c_FP grid search
│   ├── 17_conditional_thresholds.py Stage 7b: Condition-specific optimal thresholds
│   ├── 18_raap_variants.py      Stage 7c: RA-AP variants
│   ├── 19_deferral_curve.py     Stage 7d: Deferral cost vs. recall trade-off
│   ├── 20_train_cach.py         Stage 8a: Train a pooled CACH head (original protocol)
│   ├── 20b_train_cach_all.sh    Stage 8a: Run 20_train_cach.py once per detector
│   ├── 20b_train_cach_balanced.py   Stage 8b: CACH with TP/FP class-balanced sampling
│   ├── 20c_train_cach_efficient.py  Stage 8c: CACH lightweight variant
│   ├── 20d_train_cach_fast.py   Stage 8d: CACH fast-training schedule
│   ├── 20e_train_cach_perdet.py Stage 8e: Per-detector CACH, calibration split only (leakage-free)
│   ├── 20f_train_cach_perdet_reg.py Stage 8f: 20e + identity-anchoring regulariser (--lambda-id)
│   ├── 21_train_yolov11.py      Stage 8g: YOLOv11 detector variant
│   ├── 22_corruption_robustness.py Stage 9a: Shared-baseline corruption error from seeded test caches
│   ├── 22_xai_gradcam.py        Stage 9b: GradCAM saliency maps (detector)
│   ├── 23_xai_shap_cach.py      Stage 9c: Attribution analysis of the CACH embedding
│   ├── 24_evaluate_cach.py      Stage 9d: No-cal / clean-cal / pooled / oracle / CACH comparison
│   ├── 25_cach_bootstrap_ci.py  Stage 9e: Bootstrap CIs for CACH vs. baselines
│   ├── 26_conformal_image_level.py Stage 10a: Scene-level conformal table and miss-rate frontier
│   ├── 27_flir_person_ap.py     Stage 10b: FLIR person-class AP@0.5 (in-domain reference for transfer)
│   ├── 28_kfold_cv.py           Stage 10c: 5-fold CV of isotonic ECE, cost reduction, miss floor
│   ├── lint_leakage.py          Leakage/integrity lint (exit code 0 iff every check passes)
│   ├── run_p5_local.sh          CPU runner: per-detector CACH training + evaluation
│   ├── run_p5_salvage.sh        CPU runner: identity-regularised CACH retrain
│   ├── run_p5_lambda_sweep.sh   CPU runner: regularisation-strength sweep
│   ├── run_p5_finish.sh         CPU runner: finish remaining per-detector heads + evaluation
│   ├── generate_corruption_plots.py  Figure generation: corruption performance curves
│   ├── generate_figures.py      Figure generation: paper figures
│   ├── regen_corruption_figs.py Regenerate corruption figures from the final CSVs
│   ├── regen_fig_cach.py        Regenerate the CACH figure from results/cach_eval.csv
│   ├── regen_dataset_figure.py  Dataset and corruption qualitative figure
│   ├── _torch_train_utils.py    Shared training utilities (LR scheduler, checkpointing)
│   ├── gpu_bootstrap.sh         GPU node setup script (RunPod / Vast.ai)
│   ├── vast_kaist_microcal.sh   Vast.ai launch: KAIST recalibration job
│   ├── vast_phase7_gpu.sh       Vast.ai launch: Phase 7 GPU job
│   └── vast_retinanet_finish_20260606.sh  Vast.ai launch: RetinaNet finish job
│
├── configs/                     YAML configs and dataset split files
│   ├── yolov8.yaml              YOLOv8m training config
│   ├── yolov11.yaml             YOLOv11 training config
│   ├── faster_rcnn.yaml         Faster R-CNN training config
│   ├── corruption.yaml          Corruption types and severity levels
│   ├── risk.yaml                Conformal alpha, cost weights, IoU thresholds
│   ├── flir_yolo.yaml           FLIR ADAS v2 YOLO dataset descriptor
│   ├── flir_yolo_calib.yaml     FLIR calibration split descriptor
│   ├── flir_yolo_test.yaml      FLIR test split descriptor
│   ├── llvip_yolo.yaml          LLVIP YOLO dataset descriptor
│   ├── flir_val_split.json      FLIR validation split (seed 42, deterministic)
│   ├── llvip_val_split.json     LLVIP validation image list
│   ├── flir_val_calibration_images.txt  Frozen calibration split (572 images)
│   └── flir_val_test_images.txt Frozen test split (572 images)
│
├── formal/                      Lean 4 + mathlib proofs of the paper's four propositions (see formal/README.md)
│
└── notebooks/                   Exploratory Jupyter notebooks (one per paper phase)
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

The datasets are **not included**. Download them from their official sources. Pass the dataset roots with `--flir-root`, `--llvip-root` or `--kaist-root`, or set `FLIR_ROOT` for the `run_p5_*.sh` runners and `regen_dataset_figure.py`. The YOLO descriptors in `configs/*.yaml` contain the authors' local paths; edit their `path:` field before training.

| Dataset | Source | Notes |
|---|---|---|
| FLIR ADAS v2 | https://www.flir.com/oem/adas/adas-dataset-form/ | Primary benchmark; COCO annotations |
| LLVIP | https://bupt-ai-cz.github.io/LLVIP/ | Low-light visible-IR pairs |
| KAIST | https://soonminhwang.github.io/rgbt-ped-detection/ | RGB-T pedestrian, used for cross-sensor recalibration |

All headline numbers use the frozen FLIR calibration/test split in `configs/flir_val_calibration_images.txt` and `configs/flir_val_test_images.txt` (572 images each). The split is shared across detectors and seeds, and no image appears in both halves.

---

## Pipeline Usage

The numbered scripts in `scripts/` define a linear pipeline. Run them from the repository root. A GPU (CUDA) is needed wherever detectors are trained or run (Stages 1, 2, 4, 5 and 6); the calibration, risk, conformal and CACH stages run on CPU from cached predictions.

### Stage 0: Dataset audit
```bash
python scripts/00_dataset_audit.py --x9 /path/to/data
```

### Stage 1: Detector fine-tuning
```bash
python scripts/01_train_yolo.py --flir-root /path/to/FLIR_ADAS_v2
python scripts/02_train_frcnn.py --flir-root /path/to/FLIR_ADAS_v2
python scripts/03_train_retina_rtdetr.py --flir-root /path/to/FLIR_ADAS_v2
```

### Stage 2: Detector evaluation
```bash
python scripts/04_eval_detector.py --flir-root /path/to/FLIR_ADAS_v2 \
    --checkpoint results/yolov8m_flir_best.pt
python scripts/07_detector_comparison.py
```

### Stage 3: Cost/risk evaluation
```bash
python scripts/05_risk_eval.py --split-json configs/flir_val_split.json
python scripts/08_cost_ablation.py
python scripts/09_bootstrap_ci.py
```

### Stage 4: Corruption robustness
```bash
python scripts/10_run_corruption_eval.py --flir-root /path/to/FLIR_ADAS_v2
python scripts/22_corruption_robustness.py
```

### Stage 5: Domain transfer
```bash
python scripts/11_domain_transfer.py
python scripts/12_train_llvip.py --llvip-root /path/to/llvip
python scripts/13_tsne_features.py
```

### Stage 6: Post-hoc calibration + CACH data preparation
```bash
python scripts/14_corruption_calibration.py
python scripts/14a_corruption_infer.py --flir-root /path/to/FLIR_ADAS_v2
python scripts/15_micro_calibration.py --kaist-root /path/to/kaist
```

### Stage 7: Risk metric ablations
```bash
python scripts/16_cost_ratio_grid.py
python scripts/17_conditional_thresholds.py
python scripts/18_raap_variants.py
python scripts/19_deferral_curve.py
```

### Stage 8: Per-detector CACH (leakage-free protocol used in the paper)
Needs the corruption prediction caches from Stage 6b in `results/corruption_preds`. Each runner trains one head per detector on the calibration split, copies it to `results/cach/<detector>_cach_best.pt`, and then runs `24_evaluate_cach.py` on the test split.
```bash
export FLIR_ROOT=/path/to/FLIR_ADAS_v2
bash scripts/run_p5_local.sh          # unregularised heads (20e)
bash scripts/run_p5_salvage.sh        # identity-regularised heads, lambda_id = 1.0 (20f): reported in the paper
bash scripts/run_p5_lambda_sweep.sh   # regularisation sweep, lambda_id = 0.1 and 0.3
python scripts/25_cach_bootstrap_ci.py
```

### Stage 9: Explainability
```bash
python scripts/22_xai_gradcam.py
python scripts/23_xai_shap_cach.py
```

### Stage 10: Scene-level conformal control, transfer reference and stability
```bash
python scripts/26_conformal_image_level.py --root . --alpha 0.10 --iou 0.5
python scripts/27_flir_person_ap.py \
    --preds results/yolov8m_flir_seed0/eval/yolov8m_test_predictions.json
python scripts/28_kfold_cv.py          # set MV_RESULTS_DIR if the caches live outside results/
python scripts/lint_leakage.py         # exit code 0 iff every leakage check passes
```

### Figure generation
```bash
python scripts/generate_figures.py
python scripts/regen_corruption_figs.py
python scripts/regen_fig_cach.py
python scripts/regen_dataset_figure.py
```

---

## Key Modules

### `src/risk/conformal.py`: scene-level conformal risk control

`ConformalRiskController` scores each image by its miss event and selects the largest threshold λ̂ on a finite grid whose finite-sample-corrected empirical risk (n·R̂(λ) + 1)/(n + 1) is at most α (Angelopoulos et al., 2022). If no grid threshold is feasible, it reports infeasibility rather than returning the loosest threshold.

### `src/risk/cost_sensitive.py`: cost-sensitive threshold

`CostSensitiveThreshold` minimises the expected per-image cost C over a grid of confidence thresholds on the calibration split.

### `src/calibration/cach.py`: CACH

- `CorruptionEmbedNet`: small CNN that embeds a 64×64 grayscale patch.
- `MonotoneCalibrationHead`: maps the embedding to a positive temperature T (softplus) and a shift b, giving p̂ = σ(T·logit p + b). The map is strictly increasing in p, so CACH never reorders detections (Proposition 3, proved in `formal/`).
- `identity_reg_loss`: identity-anchoring regulariser (T → 1, b → 0) used by `20f_train_cach_perdet_reg.py`.

---

## Formal proofs

`formal/` holds a Lean 4 + mathlib development of the paper's four propositions: well-posed scene-level conformal selection, RA-AP density invariance and ranking reversal, CACH order preservation, and corruption-metric fairness with K-fold stability. It is `sorry`-free and uses only the standard axioms. Build instructions are in [`formal/README.md`](formal/README.md).

---

## Citation

If you use this code, please cite the accompanying paper (BibTeX will be added upon publication).

---

## License

Research code accompanying the manuscript; license to be finalized upon publication.
