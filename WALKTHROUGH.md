# Project Walkthrough: Flagging Abnormal Heartbeats with a 1D CNN

This document goes through the project from start to finish: the problem, the data, each part of the code, the results and what they mean. The [README](README.md) is the short version; this is the long one.

---

## 1. The task

An **electrocardiogram (ECG)** records the heart's electrical activity as a wave. Each heartbeat makes a recognisable shape: a small bump (P wave), a sharp spike (the QRS complex, whose tallest point is the **R peak**), then another bump (T wave).

Most beats are normal. Some aren't, and cardiologists group the abnormal ones by where in the heart they start:

| Class | Name | What it is | How it looks |
|---|---|---|---|
| **N** | Normal | The heart's regular pacemaker fires as usual | The standard shape at a steady rhythm |
| **S** | Supraventricular | An early beat that starts in the upper chambers (atria) | Shape *nearly identical to normal*, but it **arrives early** |
| **V** | Ventricular | A beat that starts in the lower chambers (ventricles) | **Wide, tall, oddly shaped** spike; easy to see |
| **F** | Fusion | A normal beat and a ventricular beat firing at the same time | A mix of the two shapes |

**Goal:** build a model that reads a raw ECG recording and labels every beat as N, S, V or F, so the abnormal ones get flagged.

### Why this is harder than it sounds

1. **Class imbalance.** About 9 in 10 beats are normal. A model that always says "normal" scores about 90% accuracy and is useless. So the project measures **recall per class**: of all the real ventricular beats, what fraction did the model catch?
2. **New patients.** Every heart's ECG looks slightly different. A model that has seen beats from a patient can recognise that patient's shapes. What matters in practice is how it does on **patients it has never seen**.

---

## 2. The data: MIT-BIH Arrhythmia Database

- **48 recordings**, each **30 minutes** long, from 47 people, recorded at Boston's Beth Israel Hospital (1975–79).
- Sampled at **360 Hz**, meaning 360 readings per second, with two leads (two views of the heart). We use the **MLII** lead, which shows the QRS spike most clearly.
- **Every beat was labelled by two or more cardiologists.** That's about 110,000 labels, stored in `.atr` annotation files. Each label is a position (sample number of the R peak) plus a symbol such as `N`, `V` or `A`.

Files per recording (for example `100.*`): `.dat` is the signal, `.hea` is the header (sample rate, lead names) and `.atr` holds the cardiologist annotations.

The **Kaggle dataset** and the **Kachuee et al. paper** contain the same data already cut into beats. This project uses the raw recordings instead, because the Kaggle version doesn't say which patient each beat came from, so you can't split train and test by patient.

---

## 3. Project layout

```
ecg-arrhythmia-cnn/
├── ecg.py              # load data → cut beats → train 8 models → write results
├── demo.py             # render the demo video with a trained model
├── requirements.txt
├── README.md           # short summary
├── WALKTHROUGH.md      # this file
├── train.log           # full console output of the last training run
├── data/mitdb/         # raw recordings (not committed; download from PhysioNet)
└── results/
    ├── metrics.json                  # every number, for all 8 experiments
    ├── recall.png                    # per-class recall chart
    ├── confusion_patient_split.png   # confusion matrices
    ├── cnn_*.pt, cnn+rr_*.pt         # trained models (unseen-patient split only)
    └── demo_record_200.mp4           # the demo video
```

---

## 4. `ecg.py`, step by step

### 4.1 Settings and label groups ([ecg.py:17-31](ecg.py:17))

```python
AAMI = {**dict.fromkeys("NLRej", 0), **dict.fromkeys("AaJS", 1), **dict.fromkeys("VE", 2), "F": 3}
```

MIT-BIH uses about 15 beat symbols. The **AAMI EC57** standard is the industry convention for grouping them into the four classes:

- **N** ← `N` normal, `L`/`R` bundle branch block, `e`/`j` escape beats
- **S** ← `A` atrial premature, `a` aberrated atrial premature, `J` junctional premature, `S` supraventricular premature
- **V** ← `V` premature ventricular contraction, `E` ventricular escape
- **F** ← `F` fusion

Paced beats (`/`, `f`) and unclassifiable beats (`Q`) aren't given a class. They are still counted as heartbeats when measuring time between beats (`BEATS`, line 22).

```python
DS1 = [101, 106, ...]   # 22 training patients
DS2 = [100, 103, ...]   # 22 test patients
assert not set(DS1) & set(DS2)
```

This is the **patient split** from de Chazal et al. (2004), the standard one in the research literature. Four records (102, 104, 107, 217) come from patients with pacemakers and are left out by convention. The `assert` guarantees that no patient is in both lists.

`HALF = 252` sets the window size: 252 samples is 0.7 s on each side of the R peak.

### 4.2 RR-interval features ([ecg.py:33-46](ecg.py:33))

The **RR interval** is the time between one beat's R peak and the next. It captures *timing*, which is the main clue for supraventricular beats, since they arrive early.

For every beat the code computes 4 numbers:

| Feature | Meaning |
|---|---|
| `pre / mean` | Time since the previous beat, relative to this patient's average |
| `post / mean` | Time until the next beat, relative to the average |
| `local / mean` | Average of the last 10 RR intervals, relative to the average |
| `pre / post` | Previous gap divided by next gap; well below 1 means "early beat, then a pause" |

Dividing by the **patient's own average** matters. Different people have different resting heart rates, so a raw 0.6 s gap is normal for one person and early for another. Ratios carry over between patients.

A small **self-check** runs every time the module loads (lines 44-46). It builds a fake rhythm with one early beat and asserts that the features flag it.

### 4.3 Cutting beats into windows ([ecg.py:49-64](ecg.py:49))

```python
r = wfdb.rdrecord(...)                          # read the signal
sig = r.p_signal[:, r.sig_name.index("MLII")]   # choose the MLII lead
ann = wfdb.rdann(..., "atr")                    # read the cardiologist labels
```

For each labelled beat:

1. Take **504 samples** (1.4 s) centred on the R peak. The window is wide enough to include parts of the neighbouring beats.
2. **Z-score normalise** the window (subtract its mean, divide by its standard deviation). This removes differences in signal strength and baseline between recordings, so the model focuses on *shape*.
3. Store the window (`X`), its 4 RR features (`R`), its class (`y`) and its position (`pos`, used by the demo).

Beats too close to the start or end of a recording to fit a full window are skipped. The lead is looked up by name instead of assumed to be channel 0, because record 114 has its leads swapped.

**Result:** 100,640 beats: 90,052 N, 2,780 S, 7,006 V and 802 F.

### 4.4 The model ([ecg.py:67-77](ecg.py:67))

```
input: 504-sample window
  → Conv1d(1→32, kernel 7) → BatchNorm → ReLU → MaxPool(2)     # 252 long
  → Conv1d(32→64)  → BatchNorm → ReLU → MaxPool(2)             # 126
  → Conv1d(64→128) → BatchNorm → ReLU → MaxPool(2)             # 63
  → Conv1d(128→128)→ BatchNorm → ReLU → MaxPool(2)             # 31
  → global average pool → 128 numbers describing the beat's shape
  → [append 4 RR features]   (only in the "+RR" models)
  → Dropout(0.3) → Linear → 4 class scores
```

What each part does:

- **1D convolution:** a small filter (7 samples ≈ 20 ms) slides along the signal looking for local patterns such as a sharp rise or a wide dip. Early layers learn simple edges; deeper layers combine them into whole-beat features ("wide QRS", "inverted T").
- **BatchNorm:** keeps the values in a stable range, so training is faster and more reliable.
- **MaxPool:** halves the length each time, so deeper layers see a wider stretch of the beat.
- **Global average pool:** squashes the result into one fixed-length summary of the beat.
- **Dropout:** randomly switches off 30% of the features during training so the model can't depend too much on any single one.

About 190,000 trainable parameters, which is small by deep-learning standards. All 8 models train in about 4.5 minutes in total on one consumer GPU.

In the CNN-only models, `rr` has zero columns, so one class serves both versions.

### 4.5 Training and evaluation ([ecg.py:80-113](ecg.py:80))

- **Loss:** cross-entropy, the standard loss for classification.
- **Class weights** (line 85): with `weighted=True`, each class's loss is scaled by the inverse of how common it is. In DS1 the weights come out as N 0.28, S 13.5, V 3.4, F 30.8, so missing one fusion beat costs about 110× more than missing one normal beat. Without this, the model learns that "always say normal" is nearly always right.
- **Optimiser:** Adam, learning rate 0.001, batches of 256 beats, **15 epochs** (passes over the training data).
- **No early stopping** (line 89): the number of epochs is fixed in advance. Early stopping would mean checking the test patients during training to decide when to stop, which quietly leaks test information into the model.
- **Saving** (lines 100-101): only models trained on the patient split are saved, because those are the only ones that are honest to demo on a test patient.

Metrics recorded per experiment:

| Metric | Question it answers |
|---|---|
| Recall per class | Of the real V beats, what fraction were labelled V? |
| Precision per class | Of the beats labelled V, what fraction really were V? |
| **Flagged** (`abnormal_flag_recall`) | Of all abnormal beats, what fraction were called *anything other than normal*? This is the screening question: "did it raise the alarm?" |
| **False alarms** (`normal_false_alarm`) | Of the normal beats, what fraction were wrongly flagged? |
| Confusion matrix | The full table of true class against predicted class |

### 4.6 The experiments ([ecg.py:116-134](ecg.py:116))

`main()` runs **2 × 2 × 2 = 8** experiments:

| Choice | Option A | Option B |
|---|---|---|
| **Split** | Random beats (70/30). Leaky: the same patients are in train and test | **By patient** (DS1 → DS2). The honest one |
| **Loss** | Plain | Class-weighted |
| **Inputs** | CNN only (shape) | CNN + RR features (shape + timing) |

The random beat split is there **on purpose as a contrast**, to show how much it inflates the score.

### 4.7 Plots ([ecg.py:137](ecg.py:137))

`plot()` saves `recall.png` (recall per class for the 4 patient-split models) and `confusion_patient_split.png` (plain CNN next to CNN+RR). It reads only `metrics.json`, so the charts can be redrawn without retraining.

---

## 5. `demo.py`: the video

`python demo.py 200` produces `results/demo_record_200.mp4`.

1. **Check the patient is unseen** (line 22): the record must be in DS2, so the video never shows a patient the model trained on.
2. **Run the model on the whole 30-minute recording** in one go (lines 28-33). The CNN+RR plain model is the default because it had the best trade-off: 82% of abnormal beats flagged with 11.5% false alarms.
3. **Choose the 20 seconds with the most abnormal beats** (lines 36-38). That's deliberate selection for the clip, so the closing card shows the **whole-recording numbers** to keep it honest.
4. **Decide when each beat's label appears** (lines 40-41): only once the model would really have all its input, meaning the full window has been recorded *and* the next beat has arrived (the post-RR feature needs it). The video doesn't show the model "seeing the future".
5. **Draw each frame** (`draw`, line 62), 30 per second:
   - frames 0-3 s: intro card
   - frames 3-23 s: the ECG trace scrolling like a hospital monitor. Each beat shows the model's call above it (coloured box) and the cardiologist's label below it, with ✓ or ✗. A side panel keeps a running count of abnormal beats caught.
   - frames 23-28 s: results for the whole recording
6. **Encode** with ffmpeg as H.264, `yuv420p` (line 112), the format LinkedIn and phones play reliably.

---

## 6. Results

### Unseen patients

| Model | Loss | Accuracy | N | S | V | F | Flagged | False alarms |
|---|---|---|---|---|---|---|---|---|
| CNN | plain | 92.2% | 99.6% | 0.0% | 54.1% | 3.1% | 35.3% | 0.4% |
| CNN | weighted | 80.0% | 83.0% | 4.1% | **91.3%** | 1.3% | **81.0%** | 17.0% |
| CNN + RR | plain | 85.0% | 88.5% | 1.4% | **95.0%** | 0.5% | **81.8%** | **11.5%** |
| CNN + RR | weighted | 70.1% | 71.2% | 16.3% | 90.5% | 30.2% | 70.2% | 28.8% |

### What to take from it

1. **The model with the highest accuracy is the worst one.** The plain CNN scores 92% accuracy and misses almost half the ventricular beats and every supraventricular one. This is the imbalance trap the project was designed to expose.
2. **Class weighting works.** Ventricular recall goes from **54% to 91%**, and the share of abnormal beats flagged goes from **35% to 81%**. The cost is more false alarms (0.4% to 17%), and that is the trade-off a real system has to choose.
3. **Adding timing information reduces false alarms.** CNN+RR flags the same share of abnormal beats with fewer false alarms (11.5% against 17%).
4. **The random beat split overstates performance.** Supraventricular recall is 82-97% when patients overlap between train and test, and 0-16% when they don't. The model is recognising *people*, not *beat types*.
5. **Supraventricular beats remain unsolved.** Four fixes were tried, all compared on held-out *training* patients so the test set wasn't used to choose: RR features in the final layer, a separate RR branch, a smaller CNN with fewer epochs, and averaging with an RR-only model. None got above about 30%. The causes are a small amount of training data (about 940 supraventricular beats from a few patients) and one test patient (record 232) holding 75% of all supraventricular test beats.

### Honest caveats

- Each configuration was trained **once with one seed**. GPU training isn't bit-for-bit reproducible, so numbers move by a few points between runs.
- The RR features divide by the **whole recording's** average RR. A live monitor would have to use a running average built up over the first minutes.
- This is a portfolio project, **not a medical device**.

---

## 7. How to reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements.txt
# download https://physionet.org/content/mitdb/1.0.0/ → unzip to data/mitdb
.venv/Scripts/python ecg.py          # trains all 8 models, writes results/
.venv/Scripts/python demo.py 200     # renders the video (needs ffmpeg on PATH)
```

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **R peak** | The tallest spike of a heartbeat; used as the beat's position |
| **RR interval** | Time between two consecutive R peaks |
| **Lead (MLII)** | One view of the heart's electrical activity, from a particular electrode placement |
| **Recall** | Of the real positives, the fraction found. "Did we catch them?" |
| **Precision** | Of the predicted positives, the fraction correct. "Were the alarms real?" |
| **Class weights** | Making mistakes on rare classes cost more during training |
| **Inter-patient split** | Train and test sets contain different people |
| **Confusion matrix** | Table of true class (rows) against predicted class (columns) |
| **1D CNN** | A neural network that slides learned filters along a 1-dimensional signal |
| **Epoch** | One full pass over the training data |
