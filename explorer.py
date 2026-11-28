"""Interactive beat explorer.  Run:  streamlit run explorer.py

Loads saved checkpoints (no training), runs them on a recording whose subject the model never saw, and lets you
step through beats, inspect probabilities and RR intervals, and explore the abnormal-alarm threshold.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v2 as components

from ecgcnn import data, explain, metrics
from ecgcnn.inference import STATUS, beat_status, eligible_records
from ecgcnn.model import describe, load_checkpoint, predict_proba

ROOT = Path(__file__).parent
# Okabe-Ito hues, distinguishable with common colour-vision deficiencies; N stays neutral as the majority class
COL = ["#e8e4da", "#56b4e9", "#e69f00", "#cc79a7"]
STATUS_COL = ["#8b949a", "#e69f00", "#d55e00", "#56b4e9"]
TEAL, GOLD, TEXT, PANEL = "#56c7b8", "#f2c94c", "#e8e4da", "#161c22"
plt.style.use(ROOT / "observatory.mplstyle")

st.set_page_config(page_title="ECG beat explorer", layout="wide")
# Only what .streamlit/config.toml cannot set: ECG-paper grid, uppercase headings, mono readouts, corner ticks.
st.html("""<style>
.stApp { background-color: #0b0f13; background-image:
  linear-gradient(rgba(214,222,230,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(214,222,230,.05) 1px, transparent 1px),
  linear-gradient(rgba(214,222,230,.02) 1px, transparent 1px), linear-gradient(90deg, rgba(214,222,230,.02) 1px, transparent 1px);
  background-size: 56px 56px, 56px 56px, 11.2px 11.2px, 11.2px 11.2px; }
h1, h2, h3 { text-transform: uppercase; letter-spacing: .1em; }
[data-testid="stCaptionContainer"] { color: #8b949a; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"],
[data-testid="stNumberInputContainer"] input { font-family: "JetBrains Mono", monospace; }
[data-testid="stTable"], [data-testid="stDataFrame"], [data-testid="stImage"], [data-testid="stVegaLiteChart"] {
  background: linear-gradient(#f2c94c, #f2c94c) top left / 10px 1px no-repeat, linear-gradient(#f2c94c, #f2c94c) top left / 1px 10px no-repeat,
    linear-gradient(rgba(214,222,230,.22), rgba(214,222,230,.22)) bottom right / 10px 1px no-repeat,
    linear-gradient(rgba(214,222,230,.22), rgba(214,222,230,.22)) bottom right / 1px 10px no-repeat, #161c22;
  border-radius: 4px; padding: 6px; }
[data-testid="stImage"], [data-testid="stVegaLiteChart"] { border: 1px solid rgba(214,222,230,.12); }
[data-testid="stTable"], [data-testid="stDataFrame"] { padding: 0; }
</style>""")

# Live canvas diagrams (ecgcnn/live.js): rendered in the page, fed with the model's real numbers.
LIVE = components.component("ecg_live", js=(ROOT / "ecgcnn" / "live.js").read_text(encoding="utf-8"), css="""
.live { border: 1px solid rgba(214,222,230,.12); border-radius: 4px; padding: 8px; background:
  linear-gradient(#f2c94c, #f2c94c) top left / 10px 1px no-repeat, linear-gradient(#f2c94c, #f2c94c) top left / 1px 10px no-repeat,
  linear-gradient(rgba(214,222,230,.22), rgba(214,222,230,.22)) bottom right / 10px 1px no-repeat,
  linear-gradient(rgba(214,222,230,.22), rgba(214,222,230,.22)) bottom right / 1px 10px no-repeat, #161c22; }
.bar { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
button { font: 600 12px "Barlow Semi Condensed", sans-serif; text-transform: uppercase; letter-spacing: .12em;
  border-radius: 4px; padding: 4px 12px; cursor: pointer; border: 1px solid rgba(214,222,230,.22); background: none; color: #e8e4da; }
button.play { background: #f2c94c; color: #0b0f13; border-color: #f2c94c; }
button:focus-visible { outline: 2px solid #56c7b8; outline-offset: 2px; }
.cap { font: 13px Manrope, sans-serif; color: #8b949a; margin-left: 4px; }
canvas { display: block; }
""")


def _plain(v):
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), 3).tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def live(key, scene, height, caption, alt=None, **d):
    """Mount one animated diagram; every value it draws comes from here."""
    LIVE(key=key, data=_plain({"scene": scene, "height": height, "caption": caption, "alt": alt or caption, **d}))


@st.cache_resource
def get_model(path, legacy):
    return load_checkpoint(path, legacy=legacy)


@st.cache_data(max_entries=4)  # a few records at a time keeps hosted memory bounded
def get_record(data_dir, rec):
    return data.load_record(data_dir, rec)


@st.cache_data(max_entries=16)
def get_probs(path, legacy, data_dir, rec):
    model, _ = get_model(path, legacy)
    d = get_record(data_dir, rec)
    return predict_proba(model, d["X"], d["R"])


@st.cache_data
def checkpoints():
    """Validation-selected checkpoint of each run first, so the app opens on it."""
    selected = {m.parent / json.loads(m.read_text())["selected_model"]["checkpoint"] for m in ROOT.glob("runs/*/metrics.json")}
    found = {}
    for p in sorted(ROOT.glob("runs/*/checkpoints/*.pt"), key=lambda p: (p not in selected, str(p))):
        found[f"{p.parent.parent.name} / {p.stem}" + (" (validation-selected)" if p in selected else "")] = (str(p), False)
    for p in sorted(ROOT.glob("historical/v1/results/*.pt")):
        found[f"historical v1 (legacy) / {p.stem}"] = (str(p), True)
    return found


st.title("ECG beat explorer")
st.caption("MIT-BIH Arrhythmia Database · offline classification at expert-annotated beat locations · "
           "research demo, not a medical device")

with st.sidebar:
    data_dir = str(ROOT / st.text_input("MIT-BIH folder (relative to the project, or absolute)", "data/mitdb"))
    cks = checkpoints()
    if not cks:
        st.error("No checkpoints found under runs/*/checkpoints or historical/v1/results."); st.stop()
    ck_name = st.selectbox("Checkpoint", list(cks))
    path, legacy = cks[ck_name]
    model, meta = get_model(path, legacy)
    st.write(f"**Model:** {describe(meta)}")
    st.caption(f"Split: {meta['split']}")
    recs = eligible_records(meta)  # missing files are fetched from PhysioNet on first use
    if not recs:
        st.error("No recording is eligible for this model."); st.stop()
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

try:
    with st.spinner(f"Loading record {rec} (downloaded from PhysioNet the first time)…"):
        data.fetch_record(data_dir, rec)
        d = get_record(data_dir, rec)
except Exception as e:  # network or file problems: say what failed instead of a traceback
    st.error(f"Could not load record {rec} from {data_dir}: {e}"); st.stop()
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
    ax.axvline(thr, color=TEXT, ls=":", lw=1, label="current")
    if val_thr is not None: ax.axvline(val_thr, color=GOLD, ls="--", lw=1, label="validation-selected")
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
    ax.plot(np.arange(a, b) / fs, sig[a:b], color=TEAL, lw=1)
    lo_y, hi_y = sig[a:b].min(), sig[a:b].max()
    for j in np.where((pos >= a) & (pos < b))[0]:
        ax.text(pos[j] / fs, hi_y + 0.25, data.CLASSES[pred[j]], ha="center", color=COL[pred[j]], fontweight="bold",
                bbox=dict(boxstyle="round", fc=PANEL, ec=STATUS_COL[status[j]] if status[j] else "none", lw=2 if alarm[j] else 1))
        ax.text(pos[j] / fs, lo_y - 0.35, data.CLASSES[y[j]], ha="center", color=TEXT)
    ax.axvspan((pos[i] - data.HALF) / fs, (pos[i] + data.HALF) / fs, color=GOLD, alpha=0.15, label="model input window")
    ax.set_ylim(lo_y - 0.6, hi_y + 0.6); ax.set_xlabel("seconds"); ax.set_yticks([])
    ax.set_title("±3 s around the selected beat · top: argmax prediction (outlined = alarm) · bottom: reference", fontsize=9)
    ax.legend(loc="lower right", fontsize=7); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(np.arange(-data.HALF, data.HALF) / fs, d["X"][i], color=TEAL, lw=1)
    ax.axvline(0, color=GOLD); ax.set_xlabel("seconds from R peak"); ax.set_yticks([])
    ax.set_title("Model input: 504-sample window, z-score normalised", fontsize=9)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with right:
    st.subheader(f"Beat at {int(t // 60)}:{t % 60:05.2f}")
    st.markdown(f"Reference: **{data.CLASS_NAMES[y[i]]} ({data.CLASSES[y[i]]})**  \n"
                f"Predicted (argmax): **{data.CLASS_NAMES[pred[i]]} ({data.CLASSES[pred[i]]})**  \n"
                f"Alarm: **{'yes' if alarm[i] else 'no'}** (p_abnormal {p_abn[i]:.3f} vs threshold {thr:.2f})  \n"
                f"Status: **{STATUS[status[i]]}**")
    with st.expander("Don't get it? See if you get this:"):
        head, lines = explain.plain_summary(y[i], pred[i], probs[i][pred[i]], p_abn[i], thr, status[i])
        st.markdown(f"#### {head}")
        st.markdown("\n".join(f"- {line}" for line in lines))
        st.progress(float(p_abn[i]), text=f"How unusual the computer thought this beat was: {p_abn[i]:.0%} "
                                          f"(alarm at {thr:.0%})")
        st.caption("In the trace on the left, the letters on top are the computer's guesses and the letters underneath "
                   "are the experts' labels. The shaded band is the stretch of signal the computer looked at. "
                   "Scores are the model's own numbers, not a medical certainty.")
    st.markdown("Model probabilities (softmax outputs, not calibrated confidence):")
    st.bar_chart(pd.DataFrame({"probability": probs[i]}, index=data.CLASS_NAMES), horizontal=True, height=180)
    R = d["R"][i]
    st.markdown("RR intervals (uses the next beat and the whole-recording mean, so offline only):")
    st.table(pd.DataFrame({
        "interval": ["previous RR", "next RR", "mean of last 10 RR", "recording mean RR", "previous / next"],
        "seconds": [f"{R[0] * d['rr_mean'] / fs:.3f}", f"{R[1] * d['rr_mean'] / fs:.3f}",
                    f"{R[2] * d['rr_mean'] / fs:.3f}", f"{d['rr_mean'] / fs:.3f}", ""],
        "ratio": [f"{R[0]:.2f}", f"{R[1]:.2f}", f"{R[2]:.2f}", "1.00", f"{R[3]:.2f}"]}).set_index("interval"))

# ---- how the machine learning works --------------------------------------------------------------------------------
st.header("How the machine learning works")
st.markdown(f"Everything below uses the beat you selected above (beat at {int(t // 60)}:{t % 60:05.2f}) and the real "
            f"model, a **{describe(meta)}**. The moving diagrams are drawn from the model's actual numbers. "
            "Nothing here is a mock-up.")
x0, r0 = d["X"][i], d["R"][i]
sec = np.arange(-data.HALF, data.HALF) / fs
parts = explain.score_parts(model, x0, r0)
top = int(parts["probs"].argmax())


def verdict(p_old, p_new):
    """One line comparing the unchanged beat with a changed one."""
    a, b = 1 - p_old[0], 1 - p_new[0]
    return (f"Chance of not ordinary: **{a:.0%} → {b:.0%}**. Alarm: **{'yes' if a >= thr else 'no'} → "
            f"{'yes' if b >= thr else 'no'}**. Top guess: **{data.CLASS_NAMES[int(np.argmax(p_old))]} → "
            f"{data.CLASS_NAMES[int(np.argmax(p_new))]}**.")


st.subheader("Watch it work")
st.markdown("The model reading 20 seconds of this recording as if it were live. A label appears on top as soon as the "
            "model has everything it needs for that beat"
            + (" (the full window and the next beat, because it uses the gap after the beat)" if model.n_rr else
               " (the full window)")
            + ", and the experts' label sits underneath. The panel on the right shows its percentages for the latest "
            "beat and keeps a running tally.")
a_s = max(0, int(pos[i] - 5 * fs)); b_s = min(len(sig), a_s + 20 * fs)
ready = pos + data.HALF
if model.n_rr:
    ready = np.maximum(ready, np.r_[pos[1:], pos[-1]])
seg = sig[a_s:b_s]
live("monitor", "monitor", 320, "Live: the model reading the recording beat by beat",
     alt="Animated ECG trace with the model's beat labels appearing in real time", sig=seg[::2], fs=fs / 2,
     dur=(b_s - a_s) / fs, lo=float(np.percentile(seg, 0.5)) - 0.2, hi=float(np.percentile(seg, 99.5)) + 0.2,
     thr=thr, classes=data.CLASSES, names=data.CLASS_NAMES,
     beats=[{"t": (pos[j] - a_s) / fs, "ready": (ready[j] - a_s) / fs, "ref": int(y[j]), "pred": int(pred[j]),
             "alarm": bool(alarm[j]), "pabn": float(p_abn[j]), "status": int(status[j]),
             "probs": np.round(probs[j], 3).tolist()} for j in np.where((pos >= a_s) & (pos < b_s))[0]])

st.subheader("1 · What goes in")
st.markdown(f"The model never sees the whole recording. It gets **{2 * data.HALF} measurements** (1.4 seconds) centred on "
            "one heartbeat, rescaled so that only the *shape* of the trace matters, not how tall it is or where it sits."
            + (" It also gets **4 timing numbers**: how long the gaps before and after this beat were, compared with "
               "this person's average." if model.n_rr else " This model uses the shape only, with no timing numbers."))
c_lo = max(0, int(pos[i] - 2 * fs)); raw = sig[c_lo:min(len(sig), int(pos[i] + 2 * fs))]
live("input", "input", 230, "Cutting out the window and rescaling it", raw=raw, x=x0,
     win=[int(pos[i] - data.HALF - c_lo), int(pos[i] + data.HALF - 1 - c_lo)])
with st.expander("Try it yourself: change the height, baseline or centring", expanded=True):
    t1, t2, t3 = st.columns(3)
    gain = t1.slider("Make the trace taller or shorter (×)", 0.25, 4.0, 1.0, 0.05)
    offset = t2.slider("Move the baseline up or down (mV)", -2.0, 2.0, 0.0, 0.05)
    shift = t3.slider("Centre the window off the beat (ms)", -200, 200, 0, 5)
    try:
        x_try = explain.window(sig, int(pos[i] + round(shift * fs / 1000)), gain, offset)
    except ValueError:
        st.warning("That shift runs past the edge of the recording."); x_try = x0
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        w0 = sig[pos[i] - data.HALF:pos[i] + data.HALF]
        ax.plot(sec, w0, color=TEXT, lw=0.8, alpha=0.4, label="recorded")
        ax.plot(sec, w0 * gain + offset, color=GOLD, lw=1, label="after your change")
        ax.set_ylabel("mV", fontsize=8); ax.set_xlabel("seconds"); ax.legend(fontsize=7, loc="upper right")
        ax.set_title("What you changed (millivolts)", fontsize=9); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.plot(sec, x0, color=TEXT, lw=0.8, alpha=0.4, label="original input")
        ax.plot(sec, x_try, color=TEAL, lw=1, label="what the model gets now")
        ax.set_yticks([]); ax.set_xlabel("seconds"); ax.legend(fontsize=7, loc="upper right")
        ax.set_title("What the model receives after rescaling", fontsize=9); fig.tight_layout(); st.pyplot(fig)
        plt.close(fig)
    st.markdown(verdict(parts["probs"], explain.score_parts(model, x_try, r0)["probs"]))
    st.caption("Height and baseline disappear in the rescaling, so they change nothing. Moving the window does change "
               "the input: the model learned from beats centred on their main spike.")

st.subheader("2 · Pattern detectors")
act = explain.first_layer(model, x0)
st.markdown("The first layer is a set of **32 small pattern detectors**, each only 7 measurements wide (about 0.02 s). "
            "Each one slides along the trace, multiplies the 7 measurements under it by its 7 weights and adds them up. "
            "The result is large where the trace matches the detector's pattern: a steep rise, a sharp peak, a slow dip. "
            "Nobody wrote these weights by hand. They were learned from the training recordings by repeatedly "
            "nudging them to make fewer mistakes.")
k1 = st.slider("Pick a detector (it starts on the one that lights up most for this beat)", 1, 32,
               int(act[:, 3:-3].max(1).argmax()) + 1)  # skip the zero-padded edges
live("conv", "conv", 250, f"Detector {k1} of 32 sliding along the beat", x=x0, act=act[k1 - 1], fs=fs, bn=True,
     w=model.cnn[0][0].weight[k1 - 1, 0].detach().numpy())
PRESETS = {"Sharp peak": [-1, -1, 0, 4, 0, -1, -1], "Upward slope": [-1, -1, -1, 0, 1, 1, 1],
           "Downward slope": [1, 1, 1, 0, -1, -1, -1], "Dip": [1, 1, 0, -4, 0, 1, 1]}
with st.expander("Try it yourself: design your own detector", expanded=True):
    preset = st.selectbox("Start from", list(PRESETS))
    wcols = st.columns(7)
    mine = np.array([wcols[j].slider(f"w{j + 1}", -4.0, 4.0, float(v), 0.5, key=f"w{j}_{preset}")
                     for j, v in enumerate(PRESETS[preset])])
    live("mine", "conv", 250, "Your detector on this beat (a plain multiply-and-add, then negatives set to 0)",
         x=x0, act=explain.my_detector(x0, mine), fs=fs, bn=False, w=mine)
    st.caption("Your detector is not part of the model. It works the same way as the model's first layer, without "
               "the learned scaling step.")

st.subheader("3 · Building bigger patterns")
blocks = explain.block_outputs(model, x0)
st.markdown("Three more layers combine those detectors into detectors for bigger patterns: a whole wave, or a "
            "wide, unusual spike. Each layer keeps half as much detail along time ("
            + " → ".join(str(n) for n in [2 * data.HALF] + [b.shape[1] for b in blocks]) + " positions) while the "
            f"number of detectors grows to {blocks[-1].shape[0]}. Finally each detector's output is averaged into "
            f"one number, giving {blocks[-1].shape[0]} numbers that summarise the beat's shape.")


def heat(b, rows=32):
    """A block's output scaled to 0..1, keeping the `rows` detectors that vary most, in their original order."""
    keep = np.sort(np.argsort(b.var(1))[-rows:])
    v = np.sqrt(b[keep] / (b.max() + 1e-9))  # square root so weak responses stay visible; order is unchanged
    return {"v": np.round(v, 2).ravel().tolist(), "rows": len(keep), "cols": b.shape[1], "total": b.shape[0]}


live("layers", "layers", 330, "The beat flowing through the four layers (brighter = stronger response)",
     x=x0, blocks=[heat(b) for b in blocks], feats=parts["feats"])
with st.expander("Try it yourself: follow one of the 128 summary numbers", expanded=True):
    k3 = st.slider(f"Summary number (starts on the one that pushed hardest towards {data.CLASS_NAMES[top]})", 1,
                   blocks[-1].shape[0], int(parts["contrib"][top].argmax()) + 1)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.plot(sec, x0, color=TEAL, lw=1)
        ax2 = ax.twinx()
        resp = blocks[-1][k3 - 1]
        ax2.bar(np.linspace(sec[0], sec[-1], len(resp)), resp, width=(sec[-1] - sec[0]) / len(resp) * 0.9,
                color=GOLD, alpha=0.45)
        ax2.axhline(resp.mean(), color=GOLD, lw=1, ls="--")
        ax2.set_yticks([]); ax2.grid(False); ax.set_yticks([]); ax.set_xlabel("seconds from the main spike")
        ax.set_title(f"Detector {k3} at its 31 positions; dashed line = its average, {resp.mean():.2f}", fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.table(pd.DataFrame({"type": data.CLASS_NAMES,
                               "push on the score": [f"{v:+.3f}" for v in parts["contrib"][:, k3 - 1]]})
                 .set_index("type"))
        st.caption("Its average times a learned weight for each type. All 128 pushes added together make the "
                   "'from the shape' column in step 4.")

st.subheader("4 · Making the decision")
st.markdown("Each of the four beat types gets a score: a weighted sum of the shape numbers"
            + (", plus a weighted sum of the timing numbers," if model.n_rr else "")
            + " plus a fixed starting point. The scores are then turned into percentages that add up to 100% "
            "(a step called *softmax*). The diagram builds this beat's scores piece by piece.")
with st.expander("Try it yourself: switch parts of the score off", expanded=True):
    o1, o2 = st.columns(2)
    no_shape = o1.checkbox("Ignore the shape numbers")
    no_time = o2.checkbox("Ignore the timing numbers", disabled=not model.n_rr)
    kept = parts["bias"] + (0 if no_shape else parts["shape"]) + (0 if no_time else parts["timing"])
    new4 = explain.softmax(kept)
    if no_shape or no_time:
        st.markdown(verdict(parts["probs"], new4))
        st.caption("This is arithmetic on the real scores, not a retrained model: it shows how much each part "
                   "contributes for this beat.")
live("decision", "decision", 260, "Scores built from their parts, then turned into percentages and the alarm",
     names=data.CLASS_NAMES, thr=thr, probs=new4, bias=parts["bias"],
     shape=None if no_shape else parts["shape"], timing=None if (no_time or not model.n_rr) else parts["timing"])
f2 = lambda v: [f"{x:+.2f}" for x in v]
cols = {"type": data.CLASS_NAMES, "from the shape": f2(parts["shape"])}
if model.n_rr:
    cols["from the timing"] = f2(parts["timing"])
cols |= {"starting point": f2(parts["bias"]), "total score": f2(parts["logits"]),
         "percentage": [f"{p:.1%}" for p in parts["probs"]]}
st.table(pd.DataFrame(cols).set_index("type"))
st.caption("The table shows the unchanged beat. The alarm uses one number: 100% minus the N-group percentage. If that "
           f"reaches the threshold ({thr:.2f}), the beat is flagged.")

st.subheader("5 · What does it rely on?")
st.markdown("Change the input and watch the model's answer move. The changed beats are not real recordings, so the "
            "results say something about this model, not about hearts.")
with st.expander("Try it yourself: flatten part of the beat or change its timing", expanded=True):
    e1, e2 = st.columns(2)
    with e1:
        hide = st.checkbox("Flatten part of the beat")
        lo, hi = st.slider("Stretch to flatten (seconds from the main spike)", float(sec[0]), float(sec[-1]),
                           (-0.1, 0.1), 0.01, disabled=not hide)
    x1 = x0.copy()
    if hide:
        x1[(sec >= lo) & (sec <= hi)] = 0.0  # 0 = the window's average level after rescaling
    with e2:
        if model.n_rr:
            pre = st.slider("Gap before this beat, compared with the person's average (below 1 = early)", 0.3, 2.0,
                            float(round(r0[0], 2)), 0.01)
            r1 = explain.with_pre_rr(r0, pre)
            st.caption(f"The real gap was {r0[0]:.2f}× the average. The other timing numbers stay as they were.")
        else:
            r1 = r0
            st.caption("This model has no timing input, so only the shape can be changed.")
    new = explain.score_parts(model, x1, r1)["probs"]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.plot(sec, x0, color=TEXT, lw=0.8, alpha=0.4, label="original")
        ax.plot(sec, x1, color=TEAL, lw=1.1, label="what the model gets")
        ax.set_yticks([]); ax.set_xlabel("seconds from the beat's main spike"); ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 2.4))
        yy = np.arange(len(data.CLASSES))
        ax.barh(yy + 0.2, parts["probs"], 0.4, color=TEXT, alpha=0.4, label="original")
        ax.barh(yy - 0.2, new, 0.4, color=TEAL, label="after your change")
        ax.set_yticks(yy, data.CLASS_NAMES, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown(verdict(parts["probs"], new))

width = (hi - lo) if hide else 0.2
st.markdown(f"**Automatic sweep:** a {width:.2f} s flattened stretch slides across the beat, and the curve records the "
            "model's answer at every position. Where the curve moves most is where the model is looking.")
centres = np.arange(0, 2 * data.HALF, 8)
hw = int(round(width * fs / 2))
live("sweep_shape", "sweep", 280, "Flattening each part of the beat in turn", mode="occlude", beat=x0, hw=hw,
     centres=centres, xs=sec[centres], pabn=explain.occlusion_sweep(model, x0, r0, centres, hw),
     base=float(1 - parts["probs"][0]), thr=thr, xlabel="position of the flattened stretch")
if model.n_rr:
    st.markdown("**Timing sweep:** the same beat, with the gap before it stretched from very early to very late.")
    pres = np.linspace(0.3, 2.0, 60)
    live("sweep_time", "sweep", 260, "Changing how early this beat arrives", mode="timing", xs=pres, post=float(r0[1]),
         pabn=explain.timing_sweep(model, x0, r0, pres), base=float(1 - parts["probs"][0]), thr=thr,
         xlabel="gap before the beat, from 0.3× to 2× the person's average")
