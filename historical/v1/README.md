# Historical baseline (v1): kept as-is, not comparable to the strict-split runs

These files are the project's first version, moved here unchanged (code from commit `cba9219`). Read them as an exploratory baseline under the protocol below, not as strict-split results.

## Protocol used

- **Split:** "conventional" de Chazal DS1 → DS2. All 22 DS1 recordings trained the model and all 22 DS2 recordings tested it.
  **Correction:** records 201 (DS1) and 202 (DS2) come from the **same person**, so this split is *not* subject-disjoint. The original claim "tested on 22 people it never saw" was inaccurate: it was 22 test recordings, from 21 people the model never saw plus one it had seen.
- **No validation set.** 15 fixed epochs, a single seed (0), no checkpoint or threshold selection.
- **Test results were inspected** while the project was developed, including the RR-feature exploration below.
- The random beat split (70/30) also mixes beats from the same patient, *and* neighbouring 1.4 s windows overlap, so its numbers are optimistic for two reasons.

## Contents

| File | What it is |
|---|---|
| `code/ecg.py`, `code/demo.py` | v1 code (ran from the repo root with `import ecg`) |
| `results/metrics.json` | 8 runs: {beat split, patient split} × {plain, weighted} × {CNN, CNN+RR} |
| `results/*.pt` | bare state_dicts. Load with `load_checkpoint(path, legacy=True)` or `demo.py --legacy` |
| `results/recall.png`, `results/confusion_patient_split.png` | v1 plots |
| `results/demo_record_200.mp4` | v1 video (model CNN+RR plain). Its closing card says "tested on 22 others": see the correction above. It doesn't mark false alarms on individual beats or say the segment was chosen for its abnormal beats |
| `results/train.log` | console output |

## Verified numbers (from `results/metrics.json`)

v1 demo model `patient_split/cnn+rr/unweighted`, conventional DS2 (49,667 beats):

| metric | value | counts |
|---|---|---|
| abnormal-flag recall | 81.8% | 4,454 / 5,444 |
| abnormal-flag precision | 46.7% | 4,454 / 9,536 |
| N-group false-positive rate | 11.5% | 5,082 / 44,223 |
| S recall | 1.4% | 25 / 1,837 |
| V recall | 95.0% | 3,057 / 3,219 |
| F recall | 0.5% | 2 / 388 |

The v1 README's headline "class weighting raised ventricular recall 54% → 91%" (CNN-only, patient split) came from **one seed with no validation set**. It does **not** replicate under the revised protocol (see the main README).

The v1 README also described exploratory supraventricular experiments (late fusion, smaller CNN, CNN + RR-logistic ensemble) compared on a DS1 half-split. Those scripts were scratch files and were **not kept**. Treat those statements as unreproducible exploratory observations.
