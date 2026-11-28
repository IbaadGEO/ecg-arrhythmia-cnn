"""Look inside the CNN for one beat, and describe a result in plain language. Used by explorer.py."""
import numpy as np
import torch

from . import data
from .model import predict_proba

PLAIN = ["an ordinary-type beat", "an early beat from the heart's upper chambers",
         "a beat starting in the heart's lower chambers", "a blend of an ordinary and a lower-chamber beat"]


def _tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32))[None, None]


@torch.no_grad()
def first_layer(model, x):
    """First conv layer after batch-norm and ReLU, before pooling: (32 detectors, 504 samples), aligned with x."""
    return model.cnn[0][:3](_tensor(x))[0].numpy()


@torch.no_grad()
def block_outputs(model, x):
    """Output of each of the four conv blocks, as (channels, positions) arrays."""
    t, out = _tensor(x), []
    for blk in model.cnn[:4]:
        t = blk(t)
        out.append(t[0].numpy())
    return out


@torch.no_grad()
def score_parts(model, x, rr):
    """Split each class score (logit) exactly into a shape part (128 CNN features), a timing part (RR features)
    and the bias. Dropout is off in eval mode, so shape + timing + bias == the model's logits."""
    feats = model.cnn(_tensor(x))[0].numpy().astype(np.float64)
    lin = model.head[1]
    W, b = lin.weight.numpy().astype(np.float64), lin.bias.numpy().astype(np.float64)
    contrib = W[:, :feats.size] * feats  # each summary number's push on each class score
    shape = contrib.sum(1)
    timing = W[:, feats.size:] @ np.asarray(rr[:model.n_rr], np.float64) if model.n_rr else np.zeros_like(b)
    logits = shape + timing + b
    e = np.exp(logits - logits.max())
    return {"shape": shape, "timing": timing, "bias": b, "logits": logits, "probs": e / e.sum(), "feats": feats,
            "contrib": contrib}


def softmax(z):
    e = np.exp(np.asarray(z, np.float64) - np.max(z))
    return e / e.sum()


def zscore(w):
    w = np.asarray(w, np.float64)
    return ((w - w.mean()) / (w.std() + 1e-8)).astype(np.float32)


def window(sig, centre, gain=1.0, offset=0.0):
    """The model's input for a beat at sample `centre`, after an optional change of height and baseline."""
    if centre - data.HALF < 0 or centre + data.HALF > len(sig):
        raise ValueError("the window runs past the edge of the recording")
    return zscore(np.asarray(sig[centre - data.HALF:centre + data.HALF], np.float64) * gain + offset)


def my_detector(x, w):
    """A hand-made detector: slide weights w along x (zero-padded like the model) and keep positive sums (ReLU)."""
    w = np.asarray(w, np.float64)
    return np.maximum(np.correlate(np.pad(np.asarray(x, np.float64), len(w) // 2), w, "valid"), 0)


def occlusion_sweep(model, x, rr, centres, half_width):
    """p_abnormal with x flattened to 0 within half_width samples of each centre (one forward pass per centre)."""
    idx = np.arange(len(x))
    X = np.repeat(np.asarray(x, np.float32)[None], len(centres), 0)
    for j, c in enumerate(centres):
        X[j, np.abs(idx - c) <= half_width] = 0
    return 1 - predict_proba(model, X, np.repeat(np.asarray(rr, np.float32)[None], len(centres), 0))[:, 0]


def timing_sweep(model, x, rr, pres):
    """p_abnormal as the gap before the beat takes each value in pres (see with_pre_rr)."""
    R = np.stack([with_pre_rr(rr, p) for p in pres])
    return 1 - predict_proba(model, np.repeat(np.asarray(x, np.float32)[None], len(pres), 0), R)[:, 0]


def with_pre_rr(r, pre):
    """RR features with the previous gap set to `pre` (relative to the recording mean). The local 10-beat mean is
    left as it was, so this probes the model rather than simulating a real recording."""
    r = np.array(r, dtype=np.float32, copy=True)
    r[0], r[3] = pre, pre / r[1]
    return r


def plain_summary(ref, pred, p_pred, p_abn, thr, status):
    """Headline and sentences for a non-expert. status as in inference.beat_status."""
    if status == 0:
        head = ("The computer got this beat right." if pred == ref else
                "Right call: no alarm, although its single top guess named a different type.")
    else:
        head = ["", "Right to raise the alarm, but it named the wrong kind of beat.",
                "Missed: this beat was unusual and the computer did not raise an alarm.",
                "False alarm: this was an ordinary-type beat, but the computer raised an alarm."][status]
    alarm = p_abn >= thr
    lines = [f"Heart experts labelled it as **{PLAIN[ref]}**.",
             f"The computer's top guess was **{PLAIN[pred]}**, with a score of {p_pred:.0%}.",
             f"It rated the chance that the beat is not ordinary at **{p_abn:.0%}**. It raises an alarm at "
             f"{thr:.0%} or more, so it **{'did' if alarm else 'did not'}** raise one."]
    return head, lines
