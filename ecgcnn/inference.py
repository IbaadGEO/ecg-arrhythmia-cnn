"""Run a checkpoint on one recording; shared by demo.py and explorer.py."""
import numpy as np

from . import data
from .model import predict_proba

STATUS = ["correct", "flagged, wrong class", "missed abnormal", "false alarm"]


def eligible_records(meta):
    """Non-paced records whose SUBJECT never appeared in the checkpoint's training or validation records."""
    seen = data.subjects(meta["train_records"] + meta.get("val_records", []))
    return [r for r in data.ALL_RECORDS if r not in data.PACED and data.SUBJECT[r] not in seen]


def check_eligible(rec, meta):
    if rec not in data.ALL_RECORDS:
        raise ValueError(f"{rec} is not an MIT-BIH record")
    if rec in data.PACED:
        raise ValueError(f"record {rec} is paced and excluded under AAMI convention")
    if rec not in eligible_records(meta):
        raise ValueError(f"record {rec} (subject {data.SUBJECT[rec]}) was seen by this model in training/validation; "
                         f"eligible: {eligible_records(meta)}")


def beat_status(y, pred, alarm):
    """0 correct | 1 abnormal & alarmed but wrong argmax class | 2 abnormal & no alarm | 3 N-group & alarm.
    N-group beats without alarm count as correct even if argmax picked another class below the threshold."""
    abn = y != 0
    s = np.zeros(len(y), int)
    s[abn & alarm & (pred != y)] = 1
    s[abn & ~alarm] = 2
    s[~abn & alarm] = 3
    return s


def run_record(rec_data, model, threshold=None):
    """rec_data from data.load_record. threshold=None -> alarm = argmax != N (used for v1 models)."""
    probs = predict_proba(model, rec_data["X"], rec_data["R"])
    pred, p_abn = probs.argmax(1), 1 - probs[:, 0]
    alarm = p_abn >= threshold if threshold is not None else pred != 0
    return {"probs": probs, "pred": pred, "p_abnormal": p_abn, "alarm": alarm,
            "status": beat_status(rec_data["y"], pred, alarm)}
