import numpy as np
import pytest
import torch

from ecgcnn import explain
from ecgcnn.model import Net, predict_proba


@pytest.mark.parametrize("n_rr", [0, 4])
def test_score_parts_add_up_to_model_output(n_rr):
    torch.manual_seed(0)
    m = Net(n_rr).eval()
    x, r = np.random.default_rng(0).standard_normal((1, 504)).astype(np.float32), np.full((1, 4), 0.9, np.float32)
    p = explain.score_parts(m, x[0], r[0])
    assert np.allclose(p["shape"] + p["timing"] + p["bias"], p["logits"])
    assert np.allclose(p["probs"], predict_proba(m, x, r)[0], atol=1e-5)
    assert n_rr or not p["timing"].any()


def test_layer_shapes_follow_the_architecture():
    m = Net(0).eval()
    x = np.zeros(504, np.float32)
    assert explain.first_layer(m, x).shape == (32, 504)
    assert [b.shape for b in explain.block_outputs(m, x)] == [(32, 252), (64, 126), (128, 63), (128, 31)]


def test_with_pre_rr_keeps_features_consistent():
    r = np.array([1.0, 1.2, 1.0, 1.0 / 1.2], np.float32)
    r2 = explain.with_pre_rr(r, 0.6)
    assert r2[0] == pytest.approx(0.6) and r2[3] == pytest.approx(0.5) and r2[1:3].tolist() == r[1:3].tolist()
    assert r[0] == 1.0  # input untouched


@pytest.mark.parametrize("ref,pred,p_abn,status,head,did", [
    (0, 0, 0.1, 0, "classified this beat correctly", "did not"),
    (0, 1, 0.4, 0, "No alarm was needed", "did not"),
    (2, 1, 0.9, 1, "chose the wrong class", "did"),
    (2, 0, 0.2, 2, "Missed", "did not"),
    (0, 2, 0.8, 3, "False alarm", "did"),
])
def test_plain_summary_matches_status(ref, pred, p_abn, status, head, did):
    h, lines = explain.plain_summary(ref, pred, 0.7, p_abn, 0.5, status)
    assert head in h
    assert lines[0].endswith(f"**{explain.PLAIN[ref]}**.") and f"**{explain.PLAIN[pred]}**" in lines[1]
    assert f"**{did}** trigger" in lines[2]


def test_window_matches_loader_and_ignores_height_and_baseline():
    sig = np.sin(np.arange(3000) / 20) + np.random.default_rng(1).normal(0, 0.1, 3000)
    raw = sig[1000 - 252:1000 + 252].astype(np.float32)
    ref = (raw - raw.mean()) / (raw.std() + 1e-8)
    assert np.allclose(explain.window(sig, 1000), ref, atol=1e-5)
    assert np.allclose(explain.window(sig, 1000, gain=3.0, offset=-2.0), ref, atol=1e-4)
    with pytest.raises(ValueError):
        explain.window(sig, 100)


def test_my_detector_matches_a_conv_layer_with_relu():
    x, w = np.random.default_rng(2).standard_normal(504), np.array([-1, -1, 0, 3, 0, -1, -1.0])
    conv = torch.nn.functional.conv1d(torch.tensor(x)[None, None], torch.tensor(w)[None, None], padding=3)
    assert np.allclose(explain.my_detector(x, w), conv.relu()[0, 0].numpy())


def test_sweeps_reduce_to_the_unchanged_beat():
    torch.manual_seed(0)
    m = Net(4).eval()
    x, r = np.random.default_rng(3).standard_normal(504).astype(np.float32), np.array([0.8, 1.1, 1.0, 0.8 / 1.1], np.float32)
    base = 1 - explain.score_parts(m, x, r)["probs"][0]
    assert np.allclose(explain.occlusion_sweep(m, x, r, [-100, 900], 5), base, atol=1e-5)
    assert np.allclose(explain.timing_sweep(m, x, r, [0.8]), base, atol=1e-5)
    p = explain.score_parts(m, x, r)
    assert np.allclose(p["contrib"].sum(1), p["shape"]) and np.isclose(explain.softmax(p["logits"]).sum(), 1)
