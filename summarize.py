"""Turn a run's metrics.json into summary.md (all numbers in README/WALKTHROUGH come from here).

usage: python summarize.py runs/<run>
"""
import json, sys
from pathlib import Path

import numpy as np

CLASSES = ["N", "S", "V", "F"]


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def msd(s):
    return "n/a (undefined for a seed)" if s is None else f"{100 * s['mean']:.1f} ± {100 * s['std']:.1f}"


def ci(b):
    if b["ci95"] is None:
        return "n/a"
    lo, hi = b["ci95"]
    note = f" ({b['undefined_resamples']}/{b['n_boot']} undefined)" if b["undefined_resamples"] else ""
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]{note}"


def main(run):
    run = Path(run)
    M = json.loads((run / "metrics.json").read_text())
    C = json.loads((run / "config.json").read_text())
    seeds, cfgs = C["seeds"], list(C["configs"])
    L = [f"# Run summary: {run.name}", "",
         f"Created {C['created']} · seeds {seeds} · `{C['command']}`", "",
         f"- Training records ({len(C['records']['train'])}): {C['records']['train']}",
         f"- Validation records ({len(C['records']['val'])}): {C['records']['val']}",
         f"- Strict test records ({len(C['records']['test_strict'])}): {C['records']['test_strict']}",
         f"- Conventional test adds 202, which shares subject {C['conventional_shared_subjects']} with training.",
         f"- Versions: " + ", ".join(f"{k} {v}" for k, v in C["versions"].items()), "",
         "Values are **mean ± sd over seeds** (training randomness) and **95% patient-cluster bootstrap CI** of the "
         "seed-averaged metric (which patients were tested; subjects resampled with replacement). The two are separate.", ""]
    for proto in ("strict", "conventional"):
        S = M["summary"][proto]
        L += [f"## {proto} test protocol", "",
              "### Main metrics (4-class argmax; flag = any S/V/F predicted as non-N)", "",
              "| config | macro F1 | balanced acc. | accuracy | flag recall | flag precision | N-group FPR |", "|---|---|---|---|---|---|---|"]
        for c in cfgs:
            s, b = S[c]["seed_mean_sd"], S[c]["patient_bootstrap"]
            L.append(f"| {c} | " + " | ".join(f"{msd(s[k])} {ci(b[k])}" for k in
                     ("macro_f1", "balanced_accuracy", "accuracy", "flag_recall", "flag_precision", "normal_fpr")) + " |")
        L += ["", "### Per-class recall / precision (mean ± sd over seeds, 95% patient CI)", "",
              "| config | " + " | ".join(f"recall {c}" for c in CLASSES) + " | " + " | ".join(f"precision {c}" for c in CLASSES) + " |",
              "|---" * 9 + "|"]
        for c in cfgs:
            s, b = S[c]["seed_mean_sd"], S[c]["patient_bootstrap"]
            L.append(f"| {c} | " + " | ".join(f"{msd(s[f'recall_{k}'])} {ci(b[f'recall_{k}'])}" for k in CLASSES) + " | "
                     + " | ".join(f"{msd(s[f'precision_{k}'])} {ci(b[f'precision_{k}'])}" for k in CLASSES) + " |")
        L += ["", "### Per seed with counts", "",
              "| config | seed | best epoch | val macro F1 | " + " | ".join(f"{k} correct/support" for k in CLASSES)
              + " | F1 N/S/V/F | flagged/abnormal | real/flags | N false alarms | alarm @val thr: thr, recall, precision, FPR |",
              "|---" * 13 + "|"]
        for c in cfgs:
            for sd in seeds:
                R = M["runs"][f"{c}/seed{sd}"]; r = R[proto]; cn = r["counts"]; a = r["alarm_at_val_threshold"]
                L.append(f"| {c} | {sd} | {R['meta']['best_epoch']} | {R['val']['macro_f1']:.3f} | "
                         + " | ".join(f"{cn['correct'][k]}/{cn['support'][k]}" for k in CLASSES)
                         + " | " + "/".join("n/a" if r[f'f1_{k}'] is None else f"{r[f'f1_{k}']:.2f}" for k in CLASSES)
                         + f" | {cn['abnormal_flagged']}/{cn['abnormal']} | {cn['abnormal_flagged']}/{cn['flags']}"
                         + f" | {cn['normal_false_alarms']}/{cn['support']['N']}"
                         + f" | {a['threshold']:.2f}, {a['tp']}/{a['abnormal']} ({pct(a['recall'])}), {a['tp']}/{a['tp'] + a['fp']} ({pct(a['precision'])}), {a['fp']}/{a['normal']} ({pct(a['normal_fpr'])}) |")
        L.append("")
    sel = M["selected_model"]
    L += ["## Validation-selected model", "", f"Rule: {sel['rule']}.", "",
          "Mean validation macro F1: " + ", ".join(f"{c} {v:.3f}" for c, v in sel["val_macro_f1_by_config"].items()), "",
          f"Selected: **{sel['config']} seed {sel['seed']}** (`{sel['checkpoint']}`).", "",
          "### Per test patient (strict), selected model", "",
          "| record | N | S | V | F | recall S | recall V | recall F | flagged/abnormal | N false alarms |", "|---" * 10 + "|"]
    for row in M["runs"][f"{sel['config']}/seed{sel['seed']}"]["strict"]["per_record"]:
        L.append(f"| {row['record']} | {row['N']} | {row['S']} | {row['V']} | {row['F']} | {pct(row['recall_S'])} | "
                 f"{pct(row['recall_V'])} | {pct(row['recall_F'])} | {row['abnormal_flagged']}/{row['abnormal']} | "
                 f"{row['normal_false_alarms']}/{row['N']} |")
    (run / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {run / 'summary.md'}")


if __name__ == "__main__":
    main(sys.argv[1])
