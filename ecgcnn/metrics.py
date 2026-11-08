"""Metrics computed from confusion matrices, with undefined values reported as None + reason (never 0)."""
import numpy as np

from .data import CLASSES, SUBJECT

K = len(CLASSES)


def confusion(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def _ratio(num, den, why):
    return (num / den, None) if den else (None, why)


def from_confusion(cm):
    """All headline metrics from one 4x4 confusion matrix (rows = reference, cols = argmax prediction).
    'class' metrics ask: right AAMI class? 'flag' metrics ask: called anything other than N?"""
    cm = np.asarray(cm)
    m, why = {}, {}

    def put(name, num, den, reason):
        v, r = _ratio(num, den, reason)
        m[name] = v
        if r: why[name] = r

    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    put("accuracy", tp.sum(), cm.sum(), "no beats")
    for k, c in enumerate(CLASSES):
        put(f"recall_{c}", tp[k], support[k], f"no reference {c} beats")
        put(f"precision_{c}", tp[k], predicted[k], f"no beats predicted {c}")
        put(f"f1_{c}", 2 * tp[k], 2 * tp[k] + (predicted[k] - tp[k]) + (support[k] - tp[k]), f"no {c} beats referenced or predicted")
    f1s = [m[f"f1_{c}"] for c in CLASSES]
    m["macro_f1"] = None if None in f1s else float(np.mean(f1s))
    if m["macro_f1"] is None: why["macro_f1"] = "F1 undefined for a class"
    recs = [m[f"recall_{c}"] for c in CLASSES]
    m["balanced_accuracy"] = None if None in recs else float(np.mean(recs))
    if m["balanced_accuracy"] is None: why["balanced_accuracy"] = "a class has no reference beats"
    flagged_abn, abn = cm[1:, 1:].sum(), cm[1:].sum()
    put("flag_recall", flagged_abn, abn, "no reference abnormal beats")
    put("flag_precision", flagged_abn, cm[:, 1:].sum(), "no beats flagged abnormal")
    put("normal_fpr", cm[0, 1:].sum(), support[0], "no reference N-group beats")
    m["counts"] = {"support": dict(zip(CLASSES, support.tolist())), "predicted": dict(zip(CLASSES, predicted.tolist())),
                   "correct": dict(zip(CLASSES, tp.tolist())), "abnormal": int(abn), "abnormal_flagged": int(flagged_abn),
                   "flags": int(cm[:, 1:].sum()), "normal_false_alarms": int(cm[0, 1:].sum()), "total": int(cm.sum())}
    m["confusion"] = cm.tolist()
    m["confusion_row_norm"] = [(row / row.sum()).round(4).tolist() if row.sum() else [None] * K for row in cm]
    m["undefined"] = why
    return m


def flag_metrics(y, p_abnormal, threshold):
    """Binary alarm: p_abnormal = 1 - p(N) >= threshold. Separate from the 4-class argmax."""
    alarm, abn = p_abnormal >= threshold, y != 0
    tp, fp = int((alarm & abn).sum()), int((alarm & ~abn).sum())
    rec, _ = _ratio(tp, int(abn.sum()), "")
    prec, _ = _ratio(tp, tp + fp, "")
    fpr, _ = _ratio(fp, int((~abn).sum()), "")
    f1 = 2 * tp / (2 * tp + fp + int((abn & ~alarm).sum())) if abn.any() or alarm.any() else None
    return {"threshold": float(threshold), "recall": rec, "precision": prec, "normal_fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "abnormal": int(abn.sum()), "normal": int((~abn).sum())}


def select_threshold(y, p_abnormal, grid=np.round(np.arange(0.02, 0.99, 0.01), 2)):
    """Threshold maximising alarm F1. Call this on VALIDATION data only."""
    return max((flag_metrics(y, p_abnormal, t) for t in grid), key=lambda r: r["f1"] or -1)


def per_record(y, pred, rec):
    rows = []
    for r in np.unique(rec):
        k = rec == r
        mm = from_confusion(confusion(y[k], pred[k]))
        rows.append({"record": int(r), "subject": SUBJECT[int(r)], **mm["counts"]["support"],
                     **{f"recall_{c}": mm[f"recall_{c}"] for c in CLASSES},
                     "flag_recall": mm["flag_recall"], "normal_fpr": mm["normal_fpr"],
                     "abnormal_flagged": mm["counts"]["abnormal_flagged"], "abnormal": mm["counts"]["abnormal"],
                     "normal_false_alarms": mm["counts"]["normal_false_alarms"]})
    return rows


BOOT_METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "flag_recall", "flag_precision", "normal_fpr"] + \
               [f"recall_{c}" for c in CLASSES] + [f"precision_{c}" for c in CLASSES]


def subject_bootstrap(per_rec_cms, records, n_boot=2000, seed=0):
    """Patient-cluster bootstrap. per_rec_cms: (n_seeds, n_records, K, K). Resamples SUBJECTS with replacement;
    in each resample, computes each metric per seed and averages over seeds. So the CI describes uncertainty from
    which patients were tested, for the seed-averaged metric; seed-to-seed spread is reported separately."""
    cms = np.asarray(per_rec_cms)
    subj = np.array([SUBJECT[r] for r in records])
    uniq = np.unique(subj)
    by_subj = np.stack([cms[:, subj == s].sum(1) for s in uniq], 1)  # (seeds, subjects, K, K)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in BOOT_METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        per_seed = [from_confusion(by_subj[s, pick].sum(0)) for s in range(len(cms))]
        for k in BOOT_METRICS:
            v = [p[k] for p in per_seed]
            vals[k].append(None if None in v else float(np.mean(v)))
    out = {}
    for k, v in vals.items():
        ok = np.array([x for x in v if x is not None])
        out[k] = {"ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None,
                  "undefined_resamples": len(v) - len(ok), "n_boot": n_boot, "n_subjects": len(uniq)}
        if len(v) - len(ok):
            out[k]["note"] = f"undefined in {len(v) - len(ok)} of {n_boot} resamples (class absent or never predicted); CI over the rest"
    return out


def seed_summary(per_seed_metrics):
    """Mean and sample std across seeds; None if undefined for any seed."""
    out = {}
    for k in BOOT_METRICS:
        v = [m[k] for m in per_seed_metrics]
        out[k] = None if None in v else {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                         "values": v}
    return out
