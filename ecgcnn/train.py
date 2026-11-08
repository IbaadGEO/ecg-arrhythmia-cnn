"""Training with validation-based early stopping. The test partition is never passed in here."""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from . import data, metrics
from .model import Net, predict_proba

DEFAULTS = {"lr": 1e-3, "batch": 256, "max_epochs": 40, "patience": 6, "select_on": "val macro_f1"}


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # some CUDA ops have no deterministic kernel


def train(tr, va, use_rr, weighted, seed, hp=DEFAULTS, device=None, log=print):
    """tr/va: dicts from data.load_split. Returns (best model, history, class weights)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    model = Net(data.N_RR if use_rr else 0).to(device)
    w = compute_class_weight("balanced", classes=np.arange(len(data.CLASSES)), y=tr["y"]) if weighted \
        else np.ones(len(data.CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), hp["lr"])
    X, R, y = torch.tensor(tr["X"][:, None]), torch.tensor(tr["R"][:, :model.n_rr]), torch.tensor(tr["y"])
    gen = torch.Generator().manual_seed(seed)
    best, best_score, history, bad = None, -1.0, [], 0
    for ep in range(1, hp["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(y), generator=gen)
        total = 0.0
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(X[idx].to(device), R[idx].to(device)), y[idx].to(device))
            loss.backward(); opt.step()
            total += loss.item() * len(idx)
        model.eval()
        vm = metrics.from_confusion(metrics.confusion(va["y"], predict_proba(model, va["X"], va["R"]).argmax(1)))
        score = vm["macro_f1"] if vm["macro_f1"] is not None else -1.0
        history.append({"epoch": ep, "train_loss": total / len(y), "val_macro_f1": vm["macro_f1"],
                        "val_flag_recall": vm["flag_recall"], "val_normal_fpr": vm["normal_fpr"]})
        log(f"    epoch {ep:2d} loss {total / len(y):.4f}  val macroF1 {score:.3f}")
        if score > best_score:
            best_score, best, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best)
    return model.eval(), history, w
