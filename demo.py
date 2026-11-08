"""Render an MP4 of the trained model reading an unseen patient's ECG, beat by beat.

usage: python demo.py [record] [model] [data_dir]
  record  a DS2 (test-patient) record, default 200
  model   a .pt file written by ecg.py, default results/cnn+rr_unweighted.pt
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

import ecg

rec = int(sys.argv[1]) if len(sys.argv) > 1 else 200
model_path = Path(sys.argv[2] if len(sys.argv) > 2 else "results/cnn+rr_unweighted.pt")
ecg.DATA = Path(sys.argv[3] if len(sys.argv) > 3 else "data/mitdb")
assert rec in ecg.DS2, "pick a test-patient record so the model has never seen this heart"

FS, FPS, SEG, VIEW = 360, 30, 20, 6  # Hz, video fps, seconds of ECG shown, seconds visible on screen
NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion"]
COL = ["#7fd68a", "#ffb347", "#ff5c5c", "#c58bff"]
BG, FG, DIM = "#0b0f14", "#e6edf3", "#7d8590"

X, R, y, pos, sig = ecg.load_record(rec)
use_rr = "rr" in model_path.stem
model = ecg.Net(ecg.N_RR if use_rr else 0)
model.load_state_dict(torch.load(model_path, map_location="cpu")); model.eval()
with torch.no_grad():
    pred = model(torch.tensor(X[:, None]), torch.tensor(R[:, :ecg.N_RR if use_rr else 0])).argmax(1).numpy()

# Show the SEG-second stretch with the most abnormal beats (cherry-picked for the demo; whole-record stats at the end).
starts = np.arange(0, len(sig) / FS - SEG, 1.0)
t0 = starts[np.argmax([((pos >= s * FS) & (pos < (s + SEG) * FS) & (y != 0)).sum() for s in starts])]
inseg = (pos >= t0 * FS) & (pos < (t0 + SEG) * FS)
# A beat is called once the model has its whole input: the 0.7 s window and the next R peak (for post-RR).
nxt = np.r_[pos[1:], pos[-1]]
ready = np.maximum(pos + ecg.HALF, nxt) / FS - t0

fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
INTRO, OUTRO = 3 * FPS, 5 * FPS
frames = INTRO + SEG * FPS + OUTRO


def card(lines):
    for i, (txt, size, col) in enumerate(lines):
        fig.text(0.5, 0.68 - i * 0.09, txt, ha="center", va="center", fontsize=size, color=col)


def stats(mask):
    rows = []
    for c in range(1, 4):
        n = (mask & (y == c)).sum()
        if n: rows.append((NAMES[c], (mask & (y == c) & (pred != 0)).sum(), n, COL[c]))
    n_norm = (mask & (y == 0)).sum()
    return rows, (mask & (y == 0) & (pred != 0)).sum(), n_norm


def draw(f):
    fig.clear()
    if f < INTRO:
        card([("Can a CNN spot abnormal heartbeats", 34, FG), ("in a patient it has never seen?", 34, FG),
              (f"MIT-BIH record {rec}  ·  held-out test patient  ·  {SEG} s of real ECG", 18, DIM),
              ("Each beat: AI prediction above, cardiologist label below", 18, DIM)])
        return
    if f >= INTRO + SEG * FPS:
        rows, fa, nn_ = stats(np.ones(len(y), bool))
        lines = [(f"Whole record {rec} ({len(sig) // FS // 60} min, {len(y)} beats)", 30, FG)]
        lines += [(f"{name}: {hit} / {n} flagged as abnormal", 24, col) for name, hit, n, col in rows]
        lines += [(f"Normal beats wrongly flagged: {fa} / {nn_} ({fa / max(nn_, 1):.0%})", 20, DIM),
                  ("Trained on 22 patients, tested on 22 others", 16, DIM)]
        card(lines)
        return
    now = (f - INTRO) / FPS  # seconds into the segment
    ax = fig.add_axes([0.04, 0.12, 0.70, 0.72], facecolor=BG)
    lo, hi = max(0.0, now - VIEW + 1), max(VIEW - 1, now) + 1
    a, b = int((t0 + lo) * FS), int((t0 + min(now, hi)) * FS)
    t = np.arange(a, b) / FS - t0
    ax.plot(t, sig[a:b], color="#3ddc84", lw=1.6)
    seg_sig = sig[int(t0 * FS):int((t0 + SEG) * FS)]
    y_lo, y_hi = np.percentile(seg_sig, 0.5) - 0.6, np.percentile(seg_sig, 99.5) + 0.9
    ax.set_xlim(lo, hi); ax.set_ylim(y_lo, y_hi)
    ax.axvline(now, color=DIM, lw=0.8, ls=":")
    for i in np.where(inseg & (ready <= now))[0]:
        x = pos[i] / FS - t0
        if x < lo: continue
        ok = pred[i] == y[i]
        ax.text(x, y_hi - 0.25, ecg.CLASSES[pred[i]], ha="center", va="center", fontsize=15, fontweight="bold",
                color=BG, bbox=dict(boxstyle="round,pad=0.3", fc=COL[pred[i]], ec="none"))
        ax.text(x, y_lo + 0.25, ecg.CLASSES[y[i]] + (" ✓" if ok else " ✗"), ha="center", va="center", fontsize=12,
                color=FG if ok else "#ff5c5c")
    ax.text(lo - 0.08, y_hi - 0.25, "AI", color=DIM, fontsize=11, va="center", ha="right", clip_on=False)
    ax.text(lo - 0.08, y_lo + 0.25, "Dr", color=DIM, fontsize=11, va="center", ha="right", clip_on=False)
    for s in ax.spines.values(): s.set_color("#30363d")
    ax.tick_params(colors=DIM); ax.set_yticks([]); ax.set_xlabel("seconds", color=DIM)
    fig.text(0.04, 0.92, f"1D CNN + RR features reading MIT-BIH record {rec} (unseen patient)", color=FG, fontsize=20)
    fig.text(0.04, 0.88, "N normal   S supraventricular   V ventricular   F fusion", color=DIM, fontsize=12)

    rows, fa, nn_ = stats(inseg & (ready <= now))
    fig.text(0.77, 0.80, "Abnormal beats caught", color=FG, fontsize=16)
    for k, (name, hit, n, col) in enumerate(rows):
        fig.text(0.77, 0.73 - k * 0.07, f"{name}", color=col, fontsize=14)
        fig.text(0.97, 0.73 - k * 0.07, f"{hit} / {n}", color=FG, fontsize=14, ha="right")
    fig.text(0.77, 0.73 - len(rows) * 0.07 - 0.03, "Normal beats", color=COL[0], fontsize=14)
    fig.text(0.97, 0.73 - len(rows) * 0.07 - 0.03, f"{nn_ - fa} / {nn_} ok", color=FG, fontsize=14, ha="right")


out = Path("results") / f"demo_record_{rec}.mp4"
writer = FFMpegWriter(fps=FPS, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])  # yuv420p: plays on LinkedIn/phones
with writer.saving(fig, str(out), dpi=120):
    for f in range(frames):
        draw(f); writer.grab_frame()
print(f"wrote {out}  (segment starts at {t0:.0f} s, {inseg.sum()} beats, {(y[inseg] != 0).sum()} abnormal)")
