"""MIT-BIH loading: subjects, splits, AAMI labels, beat windows and RR features.

Pipeline = offline classification at expert-annotated beat locations. RR features use the *next* beat and the
*whole recording's* mean RR, so they are not available causally in a live monitor.
"""
from pathlib import Path

import numpy as np
import wfdb

FS = 360
CLASSES = ["N", "S", "V", "F"]
CLASS_NAMES = ["N group", "Supraventricular", "Ventricular", "Fusion"]
# AAMI EC57 grouping. The N *group* is not only sinus rhythm: it also contains bundle branch block (L, R)
# and atrial/nodal escape beats (e, j). Q (paced '/', fusion-of-paced 'f', unclassifiable 'Q') gets no class.
AAMI = {**dict.fromkeys("NLRej", 0), **dict.fromkeys("AaJS", 1), **dict.fromkeys("VE", 2), "F": 3}
BEATS = set(AAMI) | set("/fQ")  # every beat annotation counts when measuring RR intervals

ALL_RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
               122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
               222, 223, 228, 230, 231, 232, 233, 234]
PACED = {102, 104, 107, 217}
# 48 recordings, 47 subjects: 201 and 202 come from the same person (202.hea: "taken from the same analog tape").
SUBJECT = {r: r for r in ALL_RECORDS} | {202: 201}

# de Chazal et al. 2004 split. NOTE: 201 (DS1) and 202 (DS2) share a subject, so this split is NOT subject-disjoint.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
# Fixed, patient-disjoint validation partition inside DS1, used for early stopping, model and threshold selection.
# Chosen so every class is present (S 201, V 1111, F 30 beats) while 208 (almost all DS1 fusion beats), 209, 201,
# 207 and 118 (most DS1 S beats) stay in training. One fixed partition, not cross-validation.
VAL = [106, 122, 124, 205, 220, 223]

SPLITS = {
    # strict: DS1 trains, DS2 minus 202 tests -> 22 training recordings, 21 test recordings, no shared subjects
    "strict": {"train": [r for r in DS1 if r not in VAL], "val": VAL, "test": [r for r in DS2 if r != 202]},
    # conventional: original DS2 incl. 202, whose subject also appears in training (201). Kept for comparison.
    "conventional": {"train": [r for r in DS1 if r not in VAL], "val": VAL, "test": DS2},
}

HALF = 252  # samples each side of the R peak (0.7 s at 360 Hz) -> 504-sample window
N_RR = 4
PREPROCESSING = {"fs": FS, "half_window": HALF, "lead": "MLII", "window_norm": "z-score per window",
                 "rr_features": ["pre/mean", "post/mean", "local10/mean", "pre/post"],
                 "rr_mean": "whole-recording mean RR (non-causal)", "beat_locations": "expert annotations (.atr)"}


def subjects(records):
    return {SUBJECT[r] for r in records}


def check_subject_disjoint(**parts):
    """Raise if any two named partitions share a subject. Returns the partitions' subject sets."""
    subs = {k: subjects(v) for k, v in parts.items()}
    names = list(subs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = subs[a] & subs[b]
            if shared:
                raise ValueError(f"partitions '{a}' and '{b}' share subject(s) {sorted(shared)}")
    return subs


def rr_features(peaks):
    """Per beat: previous RR, next RR, mean of last 10 RRs (each / recording mean RR) and previous/next.
    Uses the next beat and the whole recording, so this is offline-only."""
    peaks = np.asarray(peaks)
    if len(peaks) < 3:
        raise ValueError(f"need at least 3 beats to compute RR features, got {len(peaks)}")
    rr = np.diff(peaks).astype(np.float64)
    if np.any(rr <= 0):
        raise ValueError("beat positions must be strictly increasing (found RR <= 0)")
    pre = np.r_[rr[0], rr]; post = np.r_[rr, rr[-1]]
    local = np.convolve(pre, np.ones(10) / 10)[:len(pre)]
    k = min(10, len(pre))
    local[:k] = pre[:k].cumsum() / np.arange(1, k + 1)
    m = rr.mean()
    return np.stack([pre / m, post / m, local / m, pre / post], 1).astype(np.float32)


def check_record_files(data_dir, rec):
    missing = [f"{rec}.{ext}" for ext in ("hea", "dat", "atr") if not (Path(data_dir) / f"{rec}.{ext}").exists()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)} in {data_dir} (download https://physionet.org/content/mitdb/1.0.0/)")


def load_record(data_dir, rec):
    """Returns dict: X (n, 504) z-scored windows, R (n, 4) RR features, y (n,) class ids, pos (n,) R-peak samples,
    sig (full MLII signal), rec. Beats too close to the recording edges are skipped."""
    check_record_files(data_dir, rec)
    r = wfdb.rdrecord(str(Path(data_dir) / str(rec)))
    if "MLII" not in r.sig_name:
        raise ValueError(f"record {rec} has no MLII lead (leads: {r.sig_name})")
    sig = r.p_signal[:, r.sig_name.index("MLII")]  # record 114 has MLII on channel 1
    ann = wfdb.rdann(str(Path(data_dir) / str(rec)), "atr")
    beats = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in BEATS]
    rrf = rr_features([s for s, _ in beats])
    keep = [i for i, (s, sym) in enumerate(beats) if sym in AAMI and HALF <= s < len(sig) - HALF]
    pos = np.array([beats[i][0] for i in keep])
    X = np.stack([sig[s - HALF:s + HALF] for s in pos]).astype(np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return {"X": X, "R": rrf[keep], "y": np.array([AAMI[beats[i][1]] for i in keep], np.int64),
            "pos": pos, "sig": sig, "rec": rec, "rr_mean": float(np.diff([s for s, _ in beats]).mean())}


def load_split(data_dir, records):
    """Concatenate records; `rec` holds each beat's record id (for per-patient metrics and bootstrap)."""
    parts = [load_record(data_dir, r) for r in records]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "R", "y")}
    out["rec"] = np.concatenate([np.full(len(p["y"]), p["rec"]) for p in parts])
    return out
