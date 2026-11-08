from pathlib import Path

import numpy as np
import pytest
import torch

from ecgcnn import data, metrics
from ecgcnn.model import Net, load_checkpoint, predict_proba, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


# --- subject separation -------------------------------------------------------------------------------------------
def test_201_202_same_subject():
    assert data.SUBJECT[202] == data.SUBJECT[201]
    assert len(set(data.SUBJECT.values())) == 47 and len(data.ALL_RECORDS) == 48


def test_strict_split_is_subject_disjoint_and_sized():
    sp = data.SPLITS["strict"]
    data.check_subject_disjoint(train=sp["train"], val=sp["val"], test=sp["test"])
    assert len(sp["train"]) + len(sp["val"]) == 22 and set(sp["train"] + sp["val"]) == set(data.DS1)
    assert len(sp["test"]) == 21 and 202 not in sp["test"]
    assert not (set(sp["train"] + sp["val"] + sp["test"]) & data.PACED)


def test_conventional_split_overlap_is_detected():
    sp = data.SPLITS["conventional"]
    with pytest.raises(ValueError, match="201"):
        data.check_subject_disjoint(train=sp["train"] + sp["val"], test=sp["test"])


# --- label mapping --------------------------------------------------------------------------------------------------
def test_aami_mapping():
    expect = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0, "A": 1, "a": 1, "J": 1, "S": 1, "V": 2, "E": 2, "F": 3}
    assert data.AAMI == expect
    for q in "/fQ":  # paced/unclassifiable: counted for RR, never labelled
        assert q not in data.AAMI and q in data.BEATS


# --- RR features ----------------------------------------------------------------------------------------------------
def test_rr_premature_beat():
    f = data.rr_features([0, 300, 600, 800, 1200, 1500])
    assert f.shape == (6, 4)
    assert f[3, 0] == pytest.approx(200 / 300) and f[3, 1] == pytest.approx(400 / 300) and f[3, 3] == pytest.approx(0.5)
    assert f[0, 0] == pytest.approx(1.0)  # first beat reuses the first RR
    assert f[2, 2] == pytest.approx(1.0)  # local mean of (300, 300, 300)


@pytest.mark.parametrize("peaks", [[], [5], [0, 300]])
def test_rr_needs_three_beats(peaks):
    with pytest.raises(ValueError, match="at least 3"):
        data.rr_features(peaks)


def test_rr_rejects_non_increasing():
    with pytest.raises(ValueError, match="RR <= 0"):
        data.rr_features([0, 300, 300, 600])


def test_missing_record_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="100.hea"):
        data.load_record(tmp_path, 100)


# --- metrics ----------------------------------------------------------------------------------------------------------
def test_metrics_flag_vs_class_and_undefined():
    y = np.array([0, 0, 1, 2, 2])
    pred = np.array([0, 2, 2, 2, 0])  # S flagged but wrong class; one V missed; one false alarm
    m = metrics.from_confusion(metrics.confusion(y, pred))
    assert m["recall_S"] == 0 and m["flag_recall"] == pytest.approx(2 / 3)
    assert m["flag_precision"] == pytest.approx(2 / 3) and m["normal_fpr"] == pytest.approx(0.5)
    assert m["recall_F"] is None and "F" in m["undefined"]["recall_F"]  # undefined, not zero
    assert m["balanced_accuracy"] is None


def test_subject_bootstrap_groups_201_202():
    cm = np.zeros((1, 2, 4, 4), int); cm[0, :, 0, 0] = 10; cm[0, :, 2, 2] = 1
    out = metrics.subject_bootstrap(cm, [201, 202], n_boot=50)
    assert out["accuracy"]["n_subjects"] == 1


# --- checkpoints -----------------------------------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = Net(4).eval()
    save_checkpoint(tmp_path / "m.pt", m, {"seed": 3, "train_records": [101], "val_records": [], "alarm_threshold": 0.4})
    m2, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["n_rr"] == 4 and meta["seed"] == 3 and meta["classes"] == data.CLASSES
    X, R = np.random.randn(5, 504).astype(np.float32), np.ones((5, 4), np.float32)
    assert np.allclose(predict_proba(m, X, R), predict_proba(m2, X, R), atol=1e-6)


LEGACY = ROOT / "historical/v1/results"


@pytest.mark.skipif(not LEGACY.exists(), reason="historical checkpoints not present")
@pytest.mark.parametrize("name,n_rr", [("cnn_unweighted.pt", 0), ("cnn+rr_unweighted.pt", 4)])
def test_legacy_checkpoints_need_explicit_flag(name, n_rr):
    with pytest.raises(ValueError, match="legacy=True"):
        load_checkpoint(LEGACY / name)
    model, meta = load_checkpoint(LEGACY / name, legacy=True)
    assert meta["n_rr"] == n_rr and meta["format"] == "legacy-v1" and meta["alarm_threshold"] is None
