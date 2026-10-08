#!/usr/bin/env python3
"""Regenerate results/fig_cach.pdf from the FINAL leakage-free results/cach_eval.csv
(identity-regularised, per-detector CACH). Reproduces the two-panel ECE plot from
scripts/24_evaluate_cach.py:_make_figure without re-running the CACH evaluation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "results" / "cach_eval.csv"
OUT = REPO / "results" / "fig_cach.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 11,
})

df = pd.read_csv(CSV)
protocols = ["ece_nocal", "ece_clean", "ece_pooled", "ece_oracle", "ece_cach"]
labels = ["No-cal", "Clean-cal", "Pooled-cal", "Oracle", "CACH (ours)"]
colors = ["#9E9E9E", "#F44336", "#FF9800", "#4CAF50", "#2196F3"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

summary = df.groupby("corruption")[protocols].mean()
x = np.arange(len(summary))
width = 0.15
for i, (proto, lbl, col) in enumerate(zip(protocols, labels, colors)):
    axes[0].bar(x + i * width, summary[proto], width, label=lbl, color=col)
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels([c.replace("_", "\n") for c in summary.index], fontsize=9)
axes[0].set_ylabel("ECE (lower is better)")
axes[0].set_title("ECE by Corruption Type\n(averaged over detectors & severities)")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

by_sev = df.groupby("severity")[protocols].mean()
for proto, lbl, col in zip(protocols, labels, colors):
    axes[1].plot(by_sev.index, by_sev[proto], "o-", label=lbl, color=col, lw=1.5)
axes[1].set_xlabel("Corruption severity")
axes[1].set_ylabel("ECE (lower is better)")
axes[1].set_title("ECE vs Severity\n(averaged over detectors & corruptions)")
axes[1].set_xticks(sorted(df["severity"].unique()))
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT), bbox_inches="tight")
plt.savefig(str(OUT.with_suffix(".png")), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
