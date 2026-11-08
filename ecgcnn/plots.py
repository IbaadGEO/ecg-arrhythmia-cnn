"""Plots from saved run summaries (no training needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES, CLASS_NAMES


def recall_bars(summary, path, title):
    """summary: {config: seed_summary dict}. Bars = mean over seeds, whiskers = std over seeds."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cfgs, width = list(summary), 0.2
    for i, cfg in enumerate(cfgs):
        s = summary[cfg]
        mean = [s[f"recall_{c}"]["mean"] if s[f"recall_{c}"] else np.nan for c in CLASSES]
        std = [s[f"recall_{c}"]["std"] if s[f"recall_{c}"] else 0 for c in CLASSES]
        ax.bar(np.arange(4) + (i - (len(cfgs) - 1) / 2) * width, mean, width, yerr=std, capsize=3, label=cfg)
    ax.set_xticks(range(4), CLASS_NAMES); ax.set_ylabel("Recall (mean ± sd over seeds)")
    ax.set_ylim(0, 1.2); ax.legend(fontsize=8, ncol=2, loc="upper center"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def confusions(items, path):
    """items: [(title, raw 4x4 confusion)]. Shows counts and row-normalised %."""
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.5))
    for ax, (title, cm) in zip(np.atleast_1d(axes), items):
        cm = np.asarray(cm); rows = cm.sum(1, keepdims=True)
        norm = np.divide(cm, rows, out=np.zeros_like(cm, float), where=rows > 0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Predicted (argmax)"); ax.set_ylabel("Reference"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
