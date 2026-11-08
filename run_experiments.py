"""Train the 4 configurations x N seeds on DS1 (minus validation), select on validation, evaluate on DS2.

usage: python run_experiments.py --out runs/strict_v2 [--seeds 0 1 2] [--data data/mitdb] [--n-boot 2000]

Each model is trained once and evaluated on both test protocols:
  strict        DS2 without 202 (21 recordings, no subject shared with training/validation)  <- headline
  conventional  full DS2 (22 recordings; 202 shares a subject with training record 201)       <- comparison only
"""
import argparse, json, platform, sys, time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from ecgcnn import data, metrics, plots
from ecgcnn.model import describe, predict_proba, save_checkpoint
from ecgcnn.train import DEFAULTS, train

CONFIGS = {"cnn/plain": (False, False), "cnn/weighted": (False, True),
           "cnn+rr/plain": (True, False), "cnn+rr/weighted": (True, True)}


def versions():
    v = {p: version(p) for p in ("torch", "numpy", "wfdb", "scikit-learn", "matplotlib")}
    v["python"] = sys.version.split()[0]; v["platform"] = platform.platform()
    v["cuda"] = torch.version.cuda
    v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return v


def evaluate(y, rec, probs, threshold):
    pred = probs.argmax(1)
    return {**metrics.from_confusion(metrics.confusion(y, pred)),
            "alarm_at_val_threshold": metrics.flag_metrics(y, 1 - probs[:, 0], threshold),
            "per_record": metrics.per_record(y, pred, rec),
            "per_record_confusion": {int(r): metrics.confusion(y[rec == r], pred[rec == r]).tolist() for r in np.unique(rec)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/mitdb")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-epochs", type=int, help="override (for smoke tests); recorded in config.json")
    args = ap.parse_args()
    hp = {**DEFAULTS, **({"max_epochs": args.max_epochs} if args.max_epochs else {})}
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite a previous run. Choose a new --out.")
    (out / "checkpoints").mkdir(parents=True); (out / "predictions").mkdir()

    sp = data.SPLITS["strict"]
    data.check_subject_disjoint(train=sp["train"], val=sp["val"], test=sp["test"])  # raises on overlap
    data.check_subject_disjoint(train=sp["train"], val=sp["val"])
    conv_shared = sorted(data.subjects(data.SPLITS["conventional"]["test"]) & data.subjects(sp["train"] + sp["val"]))
    config = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(sys.argv), "seeds": args.seeds,
              "configs": {c: {"use_rr": CONFIGS[c][0], "weighted": CONFIGS[c][1]} for c in args.configs},
              "hyperparameters": hp, "preprocessing": data.PREPROCESSING,
              "records": {"train": sp["train"], "val": sp["val"], "test_strict": sp["test"],
                          "test_conventional": data.SPLITS["conventional"]["test"]},
              "subject_map_exceptions": {r: s for r, s in data.SUBJECT.items() if r != s},
              "conventional_shared_subjects": conv_shared,
              "threshold_rule": "alarm if 1 - p(N) >= t; t maximises alarm F1 on validation",
              "versions": versions()}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print("loading records ...", flush=True)
    tr = data.load_split(args.data, sp["train"]); va = data.load_split(args.data, sp["val"])
    te = data.load_split(args.data, data.SPLITS["conventional"]["test"])
    strict_mask = np.isin(te["rec"], sp["test"])
    results = {}
    for cfg in args.configs:
        use_rr, weighted = CONFIGS[cfg]
        for seed in args.seeds:
            name = f"{cfg.replace('/', '_')}_seed{seed}"
            print(f"\n== {cfg} seed {seed}", flush=True)
            model, history, w = train(tr, va, use_rr, weighted, seed, hp)
            pv = predict_proba(model, va["X"], va["R"])
            thr = metrics.select_threshold(va["y"], 1 - pv[:, 0])
            meta = {"config": {"name": cfg, "use_rr": use_rr, "weighted": weighted}, "seed": seed,
                    "split": "strict (DS1 dev / DS2 minus 202 test)", "train_records": sp["train"],
                    "val_records": sp["val"], "test_records": sp["test"], "hyperparameters": hp,
                    "best_epoch": int(np.argmax([h["val_macro_f1"] or -1 for h in history]) + 1),
                    "class_weights": dict(zip(data.CLASSES, np.round(w, 4).tolist())),
                    "alarm_threshold": thr["threshold"], "alarm_threshold_source": "validation (max alarm F1)",
                    "versions": config["versions"]}
            save_checkpoint(out / "checkpoints" / f"{name}.pt", model, meta)
            pt = predict_proba(model, te["X"], te["R"])
            np.savez_compressed(out / "predictions" / f"{name}.npz", probs=pt.astype(np.float32), y=te["y"], rec=te["rec"])
            r = results[(cfg, seed)] = {
                "meta": meta, "history": history,
                "val": {**metrics.from_confusion(metrics.confusion(va["y"], pv.argmax(1))), "alarm": thr},
                "strict": evaluate(te["y"][strict_mask], te["rec"][strict_mask], pt[strict_mask], thr["threshold"]),
                "conventional": evaluate(te["y"], te["rec"], pt, thr["threshold"])}
            s = r["strict"]
            print(f"  best epoch {meta['best_epoch']}, val macroF1 {r['val']['macro_f1']:.3f}, threshold {thr['threshold']}"
                  f" | strict test: S {s['recall_S']:.3f} V {s['recall_V']:.3f} F {s['recall_F']:.3f}"
                  f" flag {s['flag_recall']:.3f} prec {s['flag_precision']:.3f} FPR {s['normal_fpr']:.3f}", flush=True)

    summary = {}
    for proto, recs in (("strict", sp["test"]), ("conventional", data.SPLITS["conventional"]["test"])):
        summary[proto] = {}
        for cfg in args.configs:
            runs = [results[(cfg, s)][proto] for s in args.seeds]
            cms = [[r["per_record_confusion"][rec] for rec in recs] for r in runs]
            summary[proto][cfg] = {"seed_mean_sd": metrics.seed_summary(runs),
                                   "patient_bootstrap": metrics.subject_bootstrap(cms, recs, args.n_boot),
                                   "alarm_at_val_threshold": [r["alarm_at_val_threshold"] for r in runs]}
    val_score = {cfg: float(np.mean([results[(cfg, s)]["val"]["macro_f1"] for s in args.seeds])) for cfg in args.configs}
    best_cfg = max(val_score, key=val_score.get)
    best_seed = max(args.seeds, key=lambda s: results[(best_cfg, s)]["val"]["macro_f1"])
    selection = {"rule": "config with highest mean validation macro-F1, then its seed with highest validation macro-F1",
                 "val_macro_f1_by_config": val_score, "config": best_cfg, "seed": best_seed,
                 "checkpoint": f"checkpoints/{best_cfg.replace('/', '_')}_seed{best_seed}.pt"}
    (out / "metrics.json").write_text(json.dumps(
        {"runs": {f"{c}/seed{s}": v for (c, s), v in results.items()}, "summary": summary, "selected_model": selection},
        indent=1, default=float))

    plots.recall_bars({c: summary["strict"][c]["seed_mean_sd"] for c in args.configs}, out / "recall_strict.png",
                      f"Per-class recall, strict split (21 test recordings), {len(args.seeds)} seeds")
    sel = results[(best_cfg, best_seed)]["strict"]
    base = results[(args.configs[0], best_seed)]["strict"]
    plots.confusions([(f"{args.configs[0]} seed {best_seed}", base["confusion"]),
                      (f"{best_cfg} seed {best_seed} (validation-selected)", sel["confusion"])], out / "confusion_strict.png")
    print(f"\nselected model: {selection['checkpoint']}  -> wrote {out}")


if __name__ == "__main__":
    main()
