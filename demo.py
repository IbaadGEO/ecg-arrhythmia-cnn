"""Render an MP4 of a checkpoint reading a recording it never saw, beat by beat.

usage: python demo.py --checkpoint runs/<run>/checkpoints/<name>.pt [--record 200] [--out file.mp4]
       python demo.py --checkpoint historical/v1/results/cnn+rr_unweighted.pt --legacy ...
"""
import argparse, shutil, sys
from pathlib import Path

import numpy as np

from ecgcnn import data, metrics
from ecgcnn.inference import STATUS, check_eligible, run_record
from ecgcnn.model import describe, load_checkpoint

FPS, SEG, VIEW = 30, 20, 6
COL = ["#7fd68a", "#ffb347", "#ff5c5c", "#c58bff"]
STATUS_COL = ["#e6edf3", "#ffb347", "#ff5c5c", "#ff5c5c"]
STATUS_SHORT = ["✓", "flagged, wrong class", "MISSED", "FALSE ALARM"]
BG, FG, DIM = "#0b0f14", "#e6edf3", "#7d8590"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--legacy", action="store_true", help="load a v1 bare state_dict (historical/v1)")
    ap.add_argument("--record", type=int, default=200)
    ap.add_argument("--data", default="data/mitdb")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true", help="overwrite an existing video")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH; install it (e.g. `winget install Gyan.FFmpeg`) to render the video")
    ckpt = Path(a.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    try:
        model, meta = load_checkpoint(ckpt, legacy=a.legacy)
    except ValueError as e:
        sys.exit(f"{e}\n(on the command line: add --legacy)")
    try:
        check_eligible(a.record, meta)
    except ValueError as e:
        sys.exit(str(e))
    out = Path(a.out) if a.out else ckpt.parent.parent / f"demo_record_{a.record}.mp4"
    if out.exists() and not a.force:
        sys.exit(f"{out} exists; pass --force to overwrite")
    render(data.load_record(a.data, a.record), model, meta, out)


def render(d, model, meta, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    thr = meta.get("alarm_threshold")
    r = run_record(d, model, thr)
    y, pos, sig, rec, fs = d["y"], d["pos"], d["sig"], d["rec"], data.FS
    alarm_rule = f"alarm if p(abnormal) ≥ {thr:.2f} (validation-selected)" if thr is not None else "alarm if predicted class ≠ N"
    starts = np.arange(0, len(sig) / fs - SEG, 1.0)
    t0 = starts[np.argmax([((pos >= s * fs) & (pos < (s + SEG) * fs) & (y != 0)).sum() for s in starts])]
    inseg = (pos >= t0 * fs) & (pos < (t0 + SEG) * fs)
    # label appears once the model has its full input: the window end and the next R peak (post-RR)
    ready = np.maximum(pos + data.HALF, np.r_[pos[1:], pos[-1]]) / fs - t0
    seg_sig = sig[int(t0 * fs):int((t0 + SEG) * fs)]
    y_lo, y_hi = np.percentile(seg_sig, 0.5) - 0.7, np.percentile(seg_sig, 99.5) + 1.0
    whole = metrics.from_confusion(metrics.confusion(y, r["pred"]))
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    INTRO, OUTRO = 4 * FPS, 7 * FPS

    def card(lines, top=0.80):
        for i, (txt, size, col) in enumerate(lines):
            fig.text(0.5, top - i * 0.075, txt, ha="center", va="center", fontsize=size, color=col)

    def draw(f):
        fig.clear()
        if f < INTRO:
            card([("Can a CNN spot abnormal heartbeats", 32, FG), ("in a person it was not trained on?", 32, FG),
                  (f"MIT-BIH record {rec}  ·  model: {describe(meta)}", 17, DIM),
                  ("Beat locations come from expert annotations (offline classification, not a live monitor)", 15, DIM),
                  ("This 20 s segment was chosen for its high concentration of annotated abnormal beats.", 15, "#ffb347"),
                  ("Whole-recording results follow at the end.", 15, DIM)], top=0.72)
            return
        if f >= INTRO + SEG * FPS:
            c = whole["counts"]
            lines = [(f"Whole record {rec}: {len(sig) // fs // 60} min, {c['total']} beats", 28, FG),
                     ("Exact class (4-class argmax):", 18, DIM)]
            lines += [(f"{name}: {c['correct'][k]} / {c['support'][k]} correct", 18, COL[i])
                      for i, (k, name) in enumerate(zip(data.CLASSES, data.CLASS_NAMES)) if c["support"][k]]
            fl = metrics.flag_metrics(y, r["p_abnormal"], thr) if thr is not None else None
            tp, ab = (fl["tp"], fl["abnormal"]) if fl else (c["abnormal_flagged"], c["abnormal"])
            fa = fl["fp"] if fl else c["normal_false_alarms"]
            n_alarm = int(r["alarm"].sum())
            lines += [("Abnormal alarm (any S/V/F):", 18, DIM),
                      (f"caught {tp} / {ab} abnormal beats  ·  {tp} of {n_alarm} alarms were real", 18, FG),
                      (f"false alarms on N-group beats: {fa} / {c['support']['N']} ({fa / max(c['support']['N'], 1):.1%})", 18, FG),
                      (f"{alarm_rule}  ·  research demo, not a medical device", 13, DIM)]
            card(lines, top=0.88)
            return
        now = (f - INTRO) / FPS
        ax = fig.add_axes([0.05, 0.13, 0.66, 0.70], facecolor=BG)
        lo, hi = max(0.0, now - VIEW + 1), max(VIEW - 1, now) + 1
        a, b = int((t0 + lo) * fs), int((t0 + min(now, hi)) * fs)
        ax.plot(np.arange(a, b) / fs - t0, sig[a:b], color="#3ddc84", lw=1.6)
        ax.set_xlim(lo, hi); ax.set_ylim(y_lo, y_hi); ax.axvline(now, color=DIM, lw=0.8, ls=":")
        for i in np.where(inseg & (ready <= now))[0]:
            x = pos[i] / fs - t0
            if x < lo: continue
            p, st = r["pred"][i], r["status"][i]
            ax.text(x, y_hi - 0.3, data.CLASSES[p], ha="center", va="center", fontsize=15, fontweight="bold", color=BG,
                    bbox=dict(boxstyle="round,pad=0.3", fc=COL[p], ec="white" if r["alarm"][i] else "none", lw=2))
            ax.text(x, y_lo + 0.45, data.CLASSES[y[i]], ha="center", va="center", fontsize=13, color=FG)
            ax.text(x, y_lo + 0.15, STATUS_SHORT[st], ha="center", va="center", fontsize=9 if st else 12,
                    color=STATUS_COL[st], fontweight="bold" if st >= 2 else "normal")
        ax.text(lo - 0.08, y_hi - 0.3, "AI", color=DIM, fontsize=11, va="center", ha="right", clip_on=False)
        ax.text(lo - 0.08, y_lo + 0.45, "Ref", color=DIM, fontsize=11, va="center", ha="right", clip_on=False)
        for s in ax.spines.values(): s.set_color("#30363d")
        ax.tick_params(colors=DIM); ax.set_yticks([]); ax.set_xlabel("seconds", color=DIM)
        fig.text(0.05, 0.93, f"{describe(meta)} reading MIT-BIH record {rec} (subject not in training)", color=FG, fontsize=18)
        fig.text(0.05, 0.895, "Segment chosen for a high concentration of annotated abnormal beats  ·  white box = alarm raised  ·  "
                 "N = AAMI N group, S supraventricular, V ventricular, F fusion", color=DIM, fontsize=10.5)
        k = inseg & (ready <= now)
        st, yk = r["status"][k], y[k]
        rows = [("Exact class correct", f"{(r['pred'][k] == yk).sum()} / {k.sum()}", FG),
                ("Abnormal beats alarmed", f"{((yk != 0) & r['alarm'][k]).sum()} / {(yk != 0).sum()}", FG),
                ("  …of which wrong class", f"{(st == 1).sum()}", STATUS_COL[1]),
                ("Missed abnormal", f"{(st == 2).sum()}", STATUS_COL[2]),
                ("False alarms (N group)", f"{(st == 3).sum()} / {(yk == 0).sum()}", STATUS_COL[3])]
        fig.text(0.74, 0.80, "This segment so far", color=FG, fontsize=16)
        for j, (lab, val, col) in enumerate(rows):
            fig.text(0.74, 0.73 - j * 0.065, lab, color=col, fontsize=13)
            fig.text(0.98, 0.73 - j * 0.065, val, color=FG, fontsize=13, ha="right")
        fig.text(0.74, 0.36, alarm_rule, color=DIM, fontsize=10, wrap=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=FPS, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])  # yuv420p plays on LinkedIn/phones
    with writer.saving(fig, str(out), dpi=120):
        for f in range(INTRO + SEG * FPS + OUTRO):
            draw(f); writer.grab_frame()
    print(f"wrote {out} (segment {t0:.0f}-{t0 + SEG:.0f} s, {inseg.sum()} beats, {(y[inseg] != 0).sum()} abnormal; "
          f"statuses in segment: {dict(zip(STATUS, np.bincount(r['status'][inseg], minlength=4).tolist()))})")


if __name__ == "__main__":
    main()
