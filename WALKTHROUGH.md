# Project Walkthrough: Classifying Heartbeats with a 1D CNN

This document goes through the project from start to finish: the question, the data, how it is split, each part of the code, the results and their limits. The [README](README.md) is the short version, and [runs/strict_2026-11-08/summary.md](runs/strict_2026-11-08/summary.md) holds every number with its counts.

---

## 1. The task

An **ECG** records the heart's electrical activity. Each beat makes a recognisable wave whose sharpest spike, the QRS complex, peaks at the **R peak**. The AAMI EC57 standard groups beat types into four classes:

| Class | What it contains | What it looks like |
|---|---|---|
| **N group** | Normal sinus beats **plus** bundle branch block (L, R) and atrial/nodal escape beats (e, j) | Mostly the regular shape and rhythm. Not every N-group beat is ordinary sinus rhythm |
| **S** supraventricular | Premature beats from the atria or AV junction (A, a, J, S) | Shape close to normal, but **early** |
| **V** ventricular | Premature ventricular contractions and ventricular escape (V, E) | Wide, tall, unusual QRS |
| **F** fusion | A normal and a ventricular beat firing together (F) | A mix of the two |

**The project's question:** does **class-weighted** cross-entropy improve **per-class recall** on the rare classes (S, V, F), compared with plain cross-entropy, for people the model was not trained on?

**What the model does:** for each beat whose **location a cardiologist annotated**, it decides which class it is. This is offline classification. It does not find beats by itself and does not run live.

### Two different questions

- **Exact class:** is a V beat called V? (4-class argmax, per-class recall and precision)
- **Abnormal flag:** is an S/V/F beat called *anything other than N*? (flag recall and precision)

A model can flag an S beat by calling it V. That counts as flagged, but the class is wrong. Both are reported, separately.

---

## 2. The data

- **MIT-BIH Arrhythmia Database:** 48 half-hour two-lead recordings at 360 Hz, from **47 people**. Records **201 and 202 come from the same person**; the `202.hea` header says it was "taken from the same analog tape as record 201".
- Initial beat labels were generated automatically. Two cardiologists then independently reviewed each recording and corrected the beat positions and classes (`.atr` files; Moody and Mark, 2005).
- The **MLII** lead is used. It is looked up by name, because record 114 stores it on the second channel.
- The 4 paced recordings (102, 104, 107, 217) are excluded by AAMI convention.

---

## 3. Splitting by person

[`ecgcnn/data.py:19-39`](ecgcnn/data.py:19)

```python
SUBJECT = {r: r for r in ALL_RECORDS} | {202: 201}
VAL = [106, 122, 124, 205, 220, 223]
SPLITS = {"strict": {...test: DS2 without 202}, "conventional": {...test: DS2}}
```

- **DS1/DS2** (de Chazal et al., 2004) is the standard inter-patient split. It puts 201 in DS1 and 202 in DS2, so the same person appears on both sides. Disjoint *record* IDs do not guarantee disjoint *people*.
- **Strict split (headline):** the 22 DS1 recordings are used for development (**16 train + 6 validation**), and **21 test recordings** (DS2 without 202) are used for testing. No person is in two partitions.
- **Conventional split:** full DS2, reported only for comparison, with the 201/202 exception documented.
- `check_subject_disjoint()` ([data.py:52](ecgcnn/data.py:52)) raises an error if any two partitions share a subject. It runs when experiments start ([run_experiments.py:56](run_experiments.py:56)), and the demo and explorer use the same subject map to decide which recordings a model may be shown ([inference.py:10](ecgcnn/inference.py:10)).
- **Validation partition:** fixed and chosen so all four classes are present (S 201, V 1,111, F 30 beats). Record 208, which holds nearly all DS1 fusion beats, and the S-heavy records 209, 201, 207 and 118 stay in training. It is a single partition, not cross-validation.

| Partition | Recordings | Beats (N / S / V / F) |
|---|---|---|
| Train | 16 | 37,553 (33,751 / 742 / 2,676 / 384) |
| Validation | 6 | 13,420 (12,078 / 201 / 1,111 / 30) |
| Strict test | 21 | 47,532 (42,163 / 1,782 / 3,200 / 387) |

**Honesty note:** DS2 results had already been looked at during the first version of the project (`historical/v1`). The revised protocol makes every choice on validation data, but DS2 is not a pristine holdout.

---

## 4. Preprocessing

[`data.py:64-112`](ecgcnn/data.py:64)

### 4.1 Beat windows

For each annotated beat in the four classes:
1. Take **504 samples** (±0.7 s, 1.4 s in total) centred on the annotated R peak.
2. **Z-score** the window, to remove differences in amplitude and baseline.
3. Skip beats too close to the recording's start or end to fit a full window.

Neighbouring windows **overlap**, because a 1.4 s window usually contains parts of the beats on either side. That matters for the random beat split used in v1: a test window can share samples with a training window, which makes that split even more optimistic.

### 4.2 RR-interval features

The RR interval is the time between consecutive R peaks. For each beat:

| Feature | Meaning |
|---|---|
| `pre / mean` | previous RR ÷ the recording's mean RR |
| `post / mean` | **next** RR ÷ mean RR |
| `local / mean` | mean of the last 10 RRs ÷ mean RR |
| `pre / post` | well below 1 means "early beat, then a pause" |

These features are **offline-only**. They use the **next** beat, and they divide by the **whole recording's** mean RR, which includes the future. The demo waits until the next beat has arrived before it shows a label, but that only hides the first problem; whole-record normalisation would still not be possible in a live monitor.

Input checks: fewer than 3 beats, or non-increasing beat positions, raise a clear `ValueError`. So do missing `.hea/.dat/.atr` files and a missing MLII lead. All of these are unit-tested.

---

## 5. The model

[`ecgcnn/model.py:10`](ecgcnn/model.py:10)

```
504-sample window
 → 4 × [Conv1d(kernel 7) → BatchNorm → ReLU → MaxPool(2)]   channels 32, 64, 128, 128
 → global average pool → 128 shape features
 → (+ 4 RR features in the CNN+RR models)
 → Dropout(0.3) → Linear → 4 class scores → softmax = model probabilities
```

About 190k parameters. The softmax outputs are model probabilities, **not** calibrated confidence.

### Checkpoints

[`model.py:27-57`](ecgcnn/model.py:27)

New checkpoints store the weights **and** their metadata: architecture, number of RR inputs, class order, preprocessing settings, split, train/validation/test records, seed, best epoch, class weights, the validation-selected alarm threshold and software versions. `load_checkpoint()` builds the model from that metadata, never from the filename.

The v1 checkpoints are bare weight files. They load only with an **explicit** `legacy=True` (`--legacy` on the command line), which attaches the known v1 metadata: trained on all of DS1, seed 0, no threshold. Their RR input size is read from the weight shapes.

---

## 6. Training and selection

[`ecgcnn/train.py`](ecgcnn/train.py)

- Adam, learning rate 1e-3, batch 256, up to **40 epochs**.
- **Class weights** (weighted configurations): inverse class frequency on the *training* partition, in sklearn's `balanced` style. Missing a rare beat costs more.
- **Early stopping on validation:** after each epoch, compute validation macro-F1. Keep the best checkpoint, and stop after 6 epochs without improvement.
  - Early stopping on **validation** data is the standard, valid way to choose a checkpoint.
  - Early stopping on **test** data would be leakage, because the test set would then shape the model.
- **Seeds:** each configuration is trained with seeds 0, 1 and 2. Seeds fix the initial weights and the order of training batches. CUDA is set to deterministic mode where PyTorch allows it, but a few GPU operations remain nondeterministic.
- **Alarm threshold** ([metrics.py:65](ecgcnn/metrics.py:65)): an alarm is raised when *p_abnormal = 1 − p(N)* ≥ *t*. *t* is chosen on **validation** data to maximise alarm F1. The alarm is a separate decision from the 4-class argmax.
- **Demo model:** the configuration with the highest mean validation macro-F1, then that configuration's best validation seed.

`run_experiments.py` trains each model once and evaluates it on both the strict and the conventional test sets. It refuses to overwrite an existing run directory, and it writes `config.json` (records, seeds, hyperparameters, versions) before training starts.

---

## 7. Metrics

[`ecgcnn/metrics.py`](ecgcnn/metrics.py)

Everything is computed from confusion matrices (rows = reference, columns = prediction):

| Metric | Question |
|---|---|
| Accuracy | fraction of all beats with the right class (dominated by the N group) |
| Recall per class | of the real X beats, how many were called X? |
| Precision per class | of the beats called X, how many really were X? |
| F1 per class | 2TP / (2TP + FP + FN) |
| Macro F1, balanced accuracy | unweighted means over the 4 classes (balanced accuracy = mean recall) |
| Flag recall | of the S/V/F beats, how many were called non-N? |
| Flag precision | of the beats called non-N, how many really were S/V/F? |
| N-group false-positive rate | of the N-group beats, how many were called non-N? |
| Raw and row-normalised confusion matrices, per-patient results | in `metrics.json` and `summary.md` |

**Undefined is not zero.** If a class is absent, or never predicted, its recall or precision is stored as `None` together with the reason. For example, when a seed never predicts F, its F precision shows as "n/a".

### Two kinds of uncertainty

- **Across seeds:** mean ± sample standard deviation over seeds 0-2, which measures training randomness.
- **Across patients:** a **patient-cluster bootstrap** ([metrics.py:87](ecgcnn/metrics.py:87)). Resample the 21 test *subjects* with replacement 2,000 times, recompute the seed-averaged metric each time and report the 2.5th-97.5th percentile. Beats from the same person are not independent, so beats are never resampled individually. If a metric is undefined in some resamples (for example no F beats drawn), the table says how many.

---

## 8. Results (strict split, 3 seeds)

Per-class recall, mean ± sd over seeds, with the 95% patient CI in brackets:

| Config | N group | S | V | F |
|---|---|---|---|---|
| CNN, plain | 87.9 ± 5.4 [79.9, 94.5] | 0.4 ± 0.2 [0.3, 1.7] | 90.4 ± 3.0 [80.1, 95.6] | 6.1 ± 10.6 [0.0, 7.7] |
| CNN, weighted | 82.4 ± 11.0 [75.6, 89.1] | 3.2 ± 1.8 [0.8, 16.0] | 83.5 ± 3.7 [68.0, 92.7] | 17.9 ± 26.2 [0.0, 18.8] |
| CNN+RR, plain | 86.0 ± 6.0 [77.2, 93.2] | 2.2 ± 1.5 [0.5, 10.7] | 92.2 ± 2.6 [82.7, 96.6] | 0.4 ± 0.7 [0.0, 0.5] |
| CNN+RR, weighted | 87.1 ± 3.7 [80.9, 93.1] | 3.4 ± 1.3 [0.6, 16.3] | 85.7 ± 6.4 [71.6, 93.5] | 12.1 ± 20.4 [0.0, 12.9] |

### What this shows

1. **Class weighting did not reliably improve rare-class recall here.**
   - S recall averages below 4% in every configuration; the best single run reaches 4.4%.
   - F recall rises on average, but that average is driven by one seed per configuration (seed 1: 48% CNN/weighted, 36% CNN+RR/weighted). The other seeds are below 6%.
   - V recall drops by about 7 points with weighting.
   - Macro F1 and balanced accuracy differ by less than the seed-to-seed spread.
2. **This differs from v1**, which reported V recall rising from 54% to 91% with weighting. v1 used one seed, a fixed 15 epochs and all 22 DS1 recordings for training. Here, checkpoints are chosen on validation macro-F1 and 6 recordings are held out for validation. Either change could explain the difference; neither was tested on its own.
3. **RR features show no consistent advantage** across seeds.
4. **Patient intervals are wide.** For example, flag recall for CNN+RR/weighted is 58.9% with a CI of [31.8, 85.4]. A few recordings hold most of the abnormal beats: record 232 has 1,382 of the 1,782 S test beats, and record 213 has 362 of the 387 F beats.

The validation-selected model is **CNN+RR, weighted, seed 1**, with threshold 0.56. On the strict test set:
- **Alarm recall** 63.9% (3,429 / 5,369)
- **Alarm precision** 37.7% (3,429 / 9,097)
- **N-group false alarms** 13.4% (5,668 / 42,163)
- **Exact-class recall** S 4.4%, V 80.8%, F 35.7%

---

## 9. Demo video

[`demo.py`](demo.py)

1. **Load the checkpoint** from its metadata, and refuse any record whose **subject** is in the checkpoint's training or validation records ([inference.py:16](ecgcnn/inference.py:16)). It also checks that ffmpeg exists, that the input files exist, and it won't overwrite an existing video without `--force`.
2. **Run the model** on the whole recording in batches.
3. **Pick the 20 s segment** with the most annotated abnormal beats. The video **says so on screen**, and it ends with whole-recording results.
4. **Draw each beat:**
   - top: the predicted class, outlined white when an alarm is raised
   - middle: the reference class
   - bottom: the status: ✓, *flagged, wrong class*, **MISSED** or **FALSE ALARM**
   The side panel counts exact-class correctness and alarm outcomes separately.
5. **Show each label only once the model's inputs exist:** the window end and the next R peak. This is presentation only; see §4.2.

The title comes from the loaded model, for example "CNN + RR, class-weighted (seed 1)". The v1 video is kept in `historical/v1/results/`.

---

## 10. Interactive explorer

[`explorer.py`](explorer.py) is a Streamlit app (`streamlit run explorer.py`).

- **Sidebar:** pick a checkpoint (new, or v1 through the legacy loader) and a recording from the eligible list.
- **Threshold slider** on p_abnormal. The validation-selected value is the default, and any other value shows an *exploratory* warning.
- **Whole-record panel:** an exact-class table (undefined cells left empty) and alarm results at the current threshold, plus the validated threshold for comparison and a recall/precision-vs-threshold curve.
- **Beat panel:**
  - Filters for all beats, correct, argmax wrong, alarmed-but-wrong-class, missed abnormal and false alarms.
  - Jump-to-time control and stepping through beats.
  - ±3 s context and the 504-sample model input.
  - Reference, prediction and alarm for the selected beat, with its model probabilities.
  - RR intervals in seconds and as ratios.
- **Caching:** records, models and probabilities are cached, so moving the slider doesn't rerun the model.

---

## 11. Limitations

- Expert-annotated beat locations only: no detector, no live operation.
- RR features look ahead (the next beat and the whole-recording mean).
- The validation partition is small and fixed. There is no cross-validation, and the F class is tiny there (30 beats).
- Test classes are concentrated in a few people (S → 232, F → 213).
- DS2 is not untouched (see §3).
- The N group is broader than "normal sinus".
- Research only, not a medical device.

### Optional extensions (not implemented)

- **Patient-specific baseline:** compare each beat with templates built from the same patient's earlier, unlabelled beats.
- **Noise robustness:** use the PhysioNet NST noise tracks. The prepared NST records are built from training records 118 and 119, so they can't serve as unseen-patient tests.
- **Automatic beat detection:** a detected-peak mode, scored by missed and extra beats.

---

## 12. Reproduce

```bash
.venv/Scripts/python -m pytest                          # 15 tests
.venv/Scripts/python run_experiments.py --out runs/new  # 4 configs × 3 seeds, ~5 min on RTX 3060 Ti
.venv/Scripts/python summarize.py runs/new
.venv/Scripts/python demo.py --checkpoint runs/new/checkpoints/<model>.pt --record 200
.venv/Scripts/streamlit run explorer.py --browser.gatherUsageStats false
```

## 13. Glossary

| Term | Meaning |
|---|---|
| R peak / RR interval | the beat's main spike / time between consecutive spikes |
| AAMI N group | N plus L, R, e, j beat labels; broader than normal sinus rhythm |
| Recall / precision | "did we find them?" / "were our calls right?" |
| Flag | any S/V/F beat predicted as a non-N class |
| Alarm | p_abnormal = 1 − p(N) ≥ threshold, a separate decision from argmax |
| Subject-disjoint | no person appears in more than one partition |
| Patient-cluster bootstrap | resampling whole patients to estimate uncertainty |
| Early stopping | keep the epoch with the best **validation** score |
