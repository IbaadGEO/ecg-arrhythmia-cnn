# ECG Arrhythmia Detection with a 1D CNN

A 1D convolutional network that reads raw ECG from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) and classifies each heartbeat as **Normal (N)**, **Supraventricular (S)**, **Ventricular (V)** or **Fusion (F)**, following the AAMI EC57 grouping.

The main point of the project is evaluation. About 9 in 10 beats are normal, so accuracy tells you very little. Every result below reports **per-class recall**, and the headline numbers come from an **inter-patient split**: the model is tested on people it never saw in training.

## Results

100,640 labelled beats from 44 recordings (the 4 paced records are excluded by AAMI convention).

| Split | Loss | Accuracy | Recall N | Recall S | Recall V | Recall F |
|---|---|---|---|---|---|---|
| Random beats (leaky) | plain | 98.3% | 99.1% | 82.6% | 98.1% | 63.9% |
| Random beats (leaky) | class-weighted | 94.1% | 94.2% | 85.0% | 95.5% | 92.5% |
| **By patient (honest)** | plain | 92.3% | 99.6% | **0.0%** | 55.8% | **0.0%** |
| **By patient (honest)** | class-weighted | 69.1% | 70.5% | 5.7% | **94.2%** | 2.1% |

![Per-class recall](results/recall.png)
![Confusion matrices, patient split](results/confusion_patient_split.png)

### What this shows

1. **Accuracy misleads.** On the patient split, the unweighted model reaches 92.3% accuracy while detecting no supraventricular or fusion beats at all. It mostly predicts "normal".
2. **Class weighting helps a lot where it can.** Ventricular recall goes from **55.8% to 94.2%**. When you treat every abnormal beat as one "flag" class, the share of abnormal beats flagged rises from **38.6% to 90.5%**.
3. **That improvement has a cost.** With weighting, 30% of normal beats are flagged as abnormal. Most supraventricular and fusion beats get flagged as *ventricular*, so they are caught, but given the wrong label.
4. **Splitting by beat inflates the score.** When beats from the same patient appear in both train and test, the model learns that person's waveform shape. S recall drops from 85% to 6% as soon as test patients are unseen. Many published results on MIT-BIH use the leaky split.

Supraventricular beats look almost the same as normal beats. What sets them apart is mostly *timing*: they arrive early. A CNN looking at waveform shape alone has trouble generalising that across patients. Adding RR-interval features (time since the previous beat, normalised to each patient's own rhythm) is the standard next step.

## Method

- **Loading:** `wfdb` reads each record's MLII lead (360 Hz) and the cardiologist beat annotations (`.atr`).
- **Windows:** 504 samples (±0.7 s) centred on each annotated R-peak, z-score normalised per window. The window is wide enough to include the neighbouring beats.
- **Labels:** N = {N, L, R, e, j}, S = {A, a, J, S}, V = {V, E}, F = {F}. Q / paced beats are dropped.
- **Split:** de Chazal et al. (2004) DS1 (22 patients, train) / DS2 (22 patients, test). The random-beat split is a 70/30 stratified split, included only for comparison.
- **Model:** 4 × (Conv1d k=7 → BatchNorm → ReLU → MaxPool) with 32/64/128/128 channels, then global average pool, dropout and a linear layer (~200k params).
- **Training:** Adam (lr 1e-3), batch 256, 15 epochs, seed 0. The epoch count is fixed with no early stopping, so test patients are never used to choose a checkpoint.
- **Class weights:** sklearn `balanced` weights (inverse frequency) passed into `CrossEntropyLoss`.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126   # or plain `pip install torch` for CPU
.venv/Scripts/pip install -r requirements.txt
# download https://physionet.org/content/mitdb/1.0.0/ and unzip to data/mitdb
.venv/Scripts/python ecg.py            # or: python ecg.py path/to/mitdb
```

All four experiments finish in a few minutes on a consumer GPU (RTX 3060 Ti). Outputs go to `results/`.

## References

- Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database*, IEEE EMB 2001. [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- de Chazal, O'Dwyer & Reilly, *Automatic classification of heartbeats using ECG morphology and heartbeat interval features*, IEEE TBME 2004 (inter-patient split)
- Kachuee, Fazeli & Sarrafzadeh, *ECG Heartbeat Classification: A Deep Transferable Representation*, 2018. [arXiv:1805.00794](https://arxiv.org/abs/1805.00794)
- [wfdb-python](https://github.com/MIT-LCP/wfdb-python)
