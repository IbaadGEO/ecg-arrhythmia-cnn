"""Flag abnormal heartbeats in MIT-BIH Arrhythmia recordings with a 1D CNN.

Runs 4 experiments: {beat-level split, patient-level split} x {plain loss, class-weighted loss}
and writes per-class recall/precision + confusion matrices to results/.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mitdb")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
CLASSES = ["N", "S", "V", "F"]  # normal, supraventricular, ventricular, fusion
# AAMI EC57 grouping of MIT-BIH beat symbols. Q (paced/unclassifiable) is dropped.
AAMI = {**dict.fromkeys("NLRej", 0), **dict.fromkeys("AaJS", 1), **dict.fromkeys("VE", 2), "F": 3}
# de Chazal et al. 2004 inter-patient split; paced records 102, 104, 107, 217 excluded.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
assert not set(DS1) & set(DS2)
HALF = 252  # 0.7 s each side of the R peak at 360 Hz -> neighbouring beats are visible, so prematurity is too
EPOCHS, BATCH, SEED = 15, 256, 0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_record(rec):
    r = wfdb.rdrecord(str(DATA / str(rec)))
    sig = r.p_signal[:, r.sig_name.index("MLII")]  # record 114 has MLII on channel 1
    ann = wfdb.rdann(str(DATA / str(rec)), "atr")
    X, y = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI and HALF <= s < len(sig) - HALF:
            w = sig[s - HALF:s + HALF]
            X.append((w - w.mean()) / (w.std() + 1e-8))
            y.append(AAMI[sym])
    return np.array(X, np.float32), np.array(y, np.int64)


def load(recs):
    parts = [load_record(r) for r in recs]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def make_model():
    def block(i, o):
        return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, len(CLASSES)))


def train_eval(Xtr, ytr, Xte, yte, weighted):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = make_model().to(dev)
    w = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=ytr) if weighted else np.ones(len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xtr_t, ytr_t = torch.tensor(Xtr[:, None]), torch.tensor(ytr)
    # Fixed epoch count, no early stopping: the test patients are never used to pick a checkpoint.
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx].to(dev)), ytr_t[idx].to(dev))
            loss.backward(); opt.step()
        print(f"  epoch {ep + 1}/{EPOCHS} loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(torch.tensor(Xte[i:i + 4096, None]).to(dev)).argmax(1).cpu()
                          for i in range(0, len(Xte), 4096)]).numpy()
    rep = classification_report(yte, pred, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": rep["accuracy"],
            "recall": {c: rep[c]["recall"] for c in CLASSES},
            "precision": {c: rep[c]["precision"] for c in CLASSES},
            "support": {c: int(rep[c]["support"]) for c in CLASSES},
            "class_weights": dict(zip(CLASSES, np.round(w, 3).tolist())),
            "confusion": confusion_matrix(yte, pred, labels=range(len(CLASSES))).tolist()}


def main():
    Xa, ya = load(DS1); Xb, yb = load(DS2)
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    print(f"{len(y)} beats, class counts {dict(zip(CLASSES, np.bincount(y, minlength=4).tolist()))}")
    # Beat-level split leaks patients into the test set (same heart, same morphology) -> optimistic.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    splits = {"beat_split": (Xtr, ytr, Xte, yte), "patient_split": (Xa, ya, Xb, yb)}
    results = {"n_beats": int(len(y)), "class_counts": dict(zip(CLASSES, np.bincount(y, minlength=4).tolist()))}
    for split, (a, b, c, d) in splits.items():
        for weighted in (False, True):
            name = f"{split}/{'weighted' if weighted else 'unweighted'}"
            print(f"\n== {name}: train {len(b)} / test {len(d)}")
            r = results[name] = train_eval(a, b, c, d, weighted)
            print(f"  acc {r['accuracy']:.3f}  recall " + "  ".join(f"{k} {v:.3f}" for k, v in r["recall"].items()))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    plot(results)


def plot(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [k for k in results if "/" in k]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for i, run in enumerate(runs):
        ax.bar(np.arange(4) + (i - 1.5) * width, [results[run]["recall"][c] for c in CLASSES], width, label=run)
    ax.set_xticks(range(4), ["Normal", "Supraventricular", "Ventricular", "Fusion"])
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.2); ax.legend(fontsize=8, ncol=2, loc="upper center"); ax.set_title("Per-class recall on MIT-BIH")
    fig.tight_layout(); fig.savefig(OUT / "recall.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, run in zip(axes, ["patient_split/unweighted", "patient_split/weighted"]):
        cm = np.array(results[run]["confusion"]); norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(run)
    fig.tight_layout(); fig.savefig(OUT / "confusion_patient_split.png", dpi=150)


if __name__ == "__main__":
    main()
