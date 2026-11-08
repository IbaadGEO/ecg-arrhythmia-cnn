"""Interactive beat explorer.  Run:  streamlit run explorer.py

Loads saved checkpoints (no training), runs them on a recording whose subject the model never saw, and lets you
step through beats, inspect probabilities and RR intervals, and explore the abnormal-alarm threshold.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ecgcnn import data, metrics
from ecgcnn.inference import STATUS, beat_status, eligible_records
from ecgcnn.model import describe, load_checkpoint, predict_proba

ROOT = Path(__file__).parent
COL = ["#2e9e44", "#e08a00", "#d62828", "#8e44ad"]
STATUS_COL = ["#888888", "#e08a00", "#d62828", "#1f6feb"]

st.set_page_config(page_title="ECG beat explorer", layout="wide")


@st.cache_resource
def get_model(path, legacy):
    return load_checkpoint(path, legacy=legacy)


@st.cache_data
def get_record(data_dir, rec):
    return data.load_record(data_dir, rec)


@st.cache_data
def get_probs(path, legacy, data_dir, rec):
    model, _ = get_model(path, legacy)
    d = get_record(data_dir, rec)
    return predict_proba(model, d["X"], d["R"])


def checkpoints():
    found = {}
    for p in sorted(ROOT.glob("runs/*/checkpoints/*.pt")):
        found[f"{p.parent.parent.name} / {p.stem}"] = (str(p), False)
    for p in sorted(ROOT.glob("historical/v1/results/*.pt")):
        found[f"historical v1 (legacy) / {p.stem}"] = (str(p), True)
    return found


st.title("ECG beat explorer")
st.caption("MIT-BIH Arrhythmia Database · offline classification at expert-annotated beat locations · "
           "research demo, not a medical device")

with st.sidebar:
    data_dir = st.text_input("MIT-BIH folder", str(ROOT / "data" / "mitdb"))
    cks = checkpoints()
    if not cks:
        st.error("No checkpoints found under runs/*/checkpoints or historical/v1/results."); st.stop()
    ck_name = st.selectbox("Checkpoint", list(cks))
    path, legacy = cks[ck_name]
    model, meta = get_model(path, legacy)
    st.write(f"**Model:** {describe(meta)}")
    st.caption(f"Split: {meta['split']}")
    recs = [r for r in eligible_records(meta) if (Path(data_dir) / f"{r}.hea").exists()]
    if not recs:
        st.error(f"No eligible recordings found in {data_dir}."); st.stop()
    rec = st.selectbox("Recording (subject never seen by this model)", recs,
                       index=recs.index(200) if 200 in recs else 0)

    st.subheader("Abnormal alarm")
    st.caption("p_abnormal = 1 − p(N). The alarm is separate from the 4-class argmax prediction.")
    val_thr = meta.get("alarm_threshold")
    default = val_thr if val_thr is not None else 0.5
    thr = st.slider("Alarm threshold", 0.01, 0.99, float(default), 0.01)
    if val_thr is not None:
        st.write(f"Validation-selected threshold: **{val_thr:.2f}** ({meta.get('alarm_threshold_source')})")
    else:
        st.warning("This legacy checkpoint has no validation-selected threshold; 0.50 is an arbitrary default.")
    exploratory = val_thr is None or abs(thr - val_thr) > 1e-9
    if exploratory:
        st.warning("Exploratory: this threshold was set by hand while looking at test data. "
                   "Results at it are not a validated estimate.")

d = get_record(data_dir, rec)
probs = get_probs(path, legacy, data_dir, rec)
y, pos, sig, fs = d["y"], d["pos"], d["sig"], data.FS
pred, p_abn = probs.argmax(1), 1 - probs[:, 0]
alarm = p_abn >= thr
status = beat_status(y, pred, alarm)

# ---- whole-record results --------------------------------------------------------------------------------------
st.header(f"Record {rec}: whole-recording results")
m = metrics.from_confusion(metrics.confusion(y, pred))
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Exact class (4-class argmax)**: is the beat's AAMI class right?")
    st.dataframe(pd.DataFrame({
        "class": data.CLASS_NAMES,
        "reference beats": [m["counts"]["support"][c] for c in data.CLASSES],
        "correct": [m["counts"]["correct"][c] for c in data.CLASSES],
        "recall": [None if m[f"recall_{c}"] is None else round(m[f"recall_{c}"], 3) for c in data.CLASSES],
        "precision": [None if m[f"precision_{c}"] is None else round(m[f"precision_{c}"], 3) for c in data.CLASSES]}),
        hide_index=True)
    st.caption("Empty cells = undefined (class absent, or never predicted), not zero.")
with c2:
    st.markdown("**Abnormal alarm**: was an S/V/F beat flagged as *some* abnormality?")
    fl = metrics.flag_metrics(y, p_abn, thr)
    rows = [("threshold", f"{thr:.2f}" + (" (exploratory)" if exploratory else " (validation-selected)"))]
    rows += [("abnormal beats alarmed", f"{fl['tp']} / {fl['abnormal']}" + (f" ({fl['recall']:.1%})" if fl['recall'] is not None else "")),
             ("alarms that were abnormal", f"{fl['tp']} / {fl['tp'] + fl['fp']}" + (f" ({fl['precision']:.1%})" if fl['precision'] is not None else " (no alarms)")),
             ("N-group false alarms", f"{fl['fp']} / {fl['normal']} ({fl['normal_fpr']:.1%})")]
    if val_thr is not None and exploratory:
        v = metrics.flag_metrics(y, p_abn, val_thr)
        rows.append(("at validated threshold", f"recall {v['recall']:.1%}, precision "
                     + (f"{v['precision']:.1%}" if v['precision'] is not None else "n/a") + f", FPR {v['normal_fpr']:.1%}"))
    st.table(pd.DataFrame(rows, columns=["measure", "value"]).set_index("measure"))
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curve = [metrics.flag_metrics(y, p_abn, t) for t in grid]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(grid, [c["recall"] for c in curve], label="abnormal recall")
    ax.plot(grid, [c["precision"] if c["precision"] is not None else np.nan for c in curve], label="alarm precision")
    ax.axvline(thr, color="k", ls=":", lw=1, label="current")
    if val_thr is not None: ax.axvline(val_thr, color="g", ls="--", lw=1, label="validation-selected")
    ax.set_xlabel("threshold on p_abnormal"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.set_title(f"Record {rec} (test data: exploratory)", fontsize=9)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ---- beat navigation ----------------------------------------------------------------------------------------------
st.header("Beats")
FILTERS = {"all beats": np.ones(len(y), bool),
           "correct": status == 0,
           "4-class argmax wrong": pred != y,
           "abnormal, alarmed but wrong class": status == 1,
           "missed abnormal (no alarm)": status == 2,
           "false alarms (N group alarmed)": status == 3}
f1, f2 = st.columns([1, 2])
with f1:
    filt = st.radio("Show", list(FILTERS), format_func=lambda k: f"{k} ({FILTERS[k].sum()})", key="beat_filter")
idx = np.where(FILTERS[filt])[0]
if not len(idx):
    st.info("No beats match this filter."); st.stop()
with f2:
    jump = st.slider("Jump to time (minutes)", 0.0, round(len(sig) / fs / 60, 2), 0.0, 0.05)
    start = int(np.searchsorted(pos[idx], jump * 60 * fs))
    k = st.number_input(f"Beat {len(idx)} matching; step with ±", 0, len(idx) - 1, min(start, len(idx) - 1))
i = int(idx[k])
t = pos[i] / fs

left, right = st.columns([3, 2])
with left:
    a, b = max(0, pos[i] - 3 * fs), min(len(sig), pos[i] + 3 * fs)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(a, b) / fs, sig[a:b], color="#1a7f37", lw=1)
    lo_y, hi_y = sig[a:b].min(), sig[a:b].max()
    for j in np.where((pos >= a) & (pos < b))[0]:
        ax.text(pos[j] / fs, hi_y + 0.25, data.CLASSES[pred[j]], ha="center", color=COL[pred[j]], fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=STATUS_COL[status[j]] if status[j] else "none", lw=2 if alarm[j] else 1))
        ax.text(pos[j] / fs, lo_y - 0.35, data.CLASSES[y[j]], ha="center", color="black")
    ax.axvspan((pos[i] - data.HALF) / fs, (pos[i] + data.HALF) / fs, color="gold", alpha=0.2, label="model input window")
    ax.set_ylim(lo_y - 0.6, hi_y + 0.6); ax.set_xlabel("seconds"); ax.set_yticks([])
    ax.set_title("±3 s around the selected beat · top: argmax prediction (outlined = alarm) · bottom: reference", fontsize=9)
    ax.legend(loc="lower right", fontsize=7); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(np.arange(-data.HALF, data.HALF) / fs, d["X"][i], color="#333", lw=1)
    ax.axvline(0, color="gold"); ax.set_xlabel("seconds from R peak"); ax.set_yticks([])
    ax.set_title("Model input: 504-sample window, z-score normalised", fontsize=9)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with right:
    st.subheader(f"Beat at {int(t // 60)}:{t % 60:05.2f}")
    st.markdown(f"Reference: **{data.CLASS_NAMES[y[i]]} ({data.CLASSES[y[i]]})**  \n"
                f"Predicted (argmax): **{data.CLASS_NAMES[pred[i]]} ({data.CLASSES[pred[i]]})**  \n"
                f"Alarm: **{'yes' if alarm[i] else 'no'}** (p_abnormal {p_abn[i]:.3f} vs threshold {thr:.2f})  \n"
                f"Status: **{STATUS[status[i]]}**")
    st.markdown("Model probabilities (softmax outputs, not calibrated confidence):")
    st.bar_chart(pd.DataFrame({"probability": probs[i]}, index=data.CLASS_NAMES), horizontal=True, height=180)
    R = d["R"][i]
    st.markdown("RR intervals (uses the next beat and the whole-recording mean, so offline only):")
    st.table(pd.DataFrame({
        "interval": ["previous RR", "next RR", "mean of last 10 RR", "recording mean RR", "previous / next"],
        "seconds": [f"{R[0] * d['rr_mean'] / fs:.3f}", f"{R[1] * d['rr_mean'] / fs:.3f}",
                    f"{R[2] * d['rr_mean'] / fs:.3f}", f"{d['rr_mean'] / fs:.3f}", ""],
        "ratio": [f"{R[0]:.2f}", f"{R[1]:.2f}", f"{R[2]:.2f}", "1.00", f"{R[3]:.2f}"]}).set_index("interval"))
