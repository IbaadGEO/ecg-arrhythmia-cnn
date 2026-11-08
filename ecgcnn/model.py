"""1D CNN (optionally + RR features) and self-describing checkpoints."""
import torch
import torch.nn as nn

from . import data

FORMAT = "ecgcnn-checkpoint-v2"


class Net(nn.Module):
    """Same layer names as the v1 model, so v1 state_dicts load unchanged."""

    def __init__(self, n_rr):
        super().__init__()
        self.n_rr = n_rr

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.cnn = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + n_rr, len(data.CLASSES)))

    def forward(self, x, rr):  # x (n, 1, 504); rr (n, n_rr), n_rr may be 0
        return self.head(torch.cat([self.cnn(x), rr], 1))


def save_checkpoint(path, model, meta):
    """meta must describe how the model was made: config, seed, split, records, threshold..."""
    meta = {"format": FORMAT, "architecture": "Net", "n_rr": model.n_rr, "classes": data.CLASSES,
            "preprocessing": data.PREPROCESSING, **meta}
    torch.save({"meta": meta, "state_dict": model.state_dict()}, path)


# What we know about the v1 checkpoints (historical/v1): trained on all 22 DS1 records, seed 0, 15 fixed epochs,
# tested on the conventional DS1/DS2 split (201/202 shared subject), no validation set, no selected threshold.
LEGACY_V1_META = {"format": "legacy-v1", "architecture": "Net", "classes": data.CLASSES,
                  "preprocessing": data.PREPROCESSING, "split": "conventional (v1, no validation set)",
                  "train_records": data.DS1, "val_records": [], "seed": 0, "alarm_threshold": None,
                  "alarm_threshold_source": "none (v1 had no validation set)"}


def load_checkpoint(path, legacy=False, device="cpu"):
    """Returns (model in eval mode, meta). v1 files are bare state_dicts and need legacy=True."""
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and obj.get("meta", {}).get("format") == FORMAT:
        meta, sd = obj["meta"], obj["state_dict"]
        if meta["classes"] != data.CLASSES:
            raise ValueError(f"checkpoint class order {meta['classes']} != {data.CLASSES}")
    elif legacy:
        sd = obj
        # n_rr read from the weight shape, not the filename
        meta = {**LEGACY_V1_META, "n_rr": sd["head.1.weight"].shape[1] - 128, "config": None}
    else:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint; pass legacy=True to load a v1 state_dict")
    model = Net(meta["n_rr"])
    model.load_state_dict(sd)
    return model.to(device).eval(), meta


def describe(meta):
    rr = "CNN + RR" if meta["n_rr"] else "CNN"
    cfg = meta.get("config") or {}
    loss = {True: "class-weighted", False: "plain"}.get(cfg.get("weighted"), "")
    tag = "v1 legacy" if meta["format"] == "legacy-v1" else f"seed {meta.get('seed')}"
    return f"{rr}{', ' + loss if loss else ''} ({tag})"


@torch.no_grad()
def predict_proba(model, X, R, batch=4096, device=None):
    device = device or next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch, None]).to(device)
        r = torch.as_tensor(R[i:i + batch, :model.n_rr]).to(device)
        out.append(model(x, r).softmax(1).cpu())
    return torch.cat(out).numpy()
