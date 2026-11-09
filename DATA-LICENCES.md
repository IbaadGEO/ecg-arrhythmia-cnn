# Data licences

The MIT licence in `LICENSE` covers the **code only**. Everything below is derived from the MIT-BIH Arrhythmia Database and keeps its terms. This table records what applies and what it requires of anyone reusing the files.

| Published file(s) | Derived from | Terms | What that requires |
|---|---|---|---|
| *Not included:* the raw recordings and annotations (`data/mitdb/`, git-ignored) | MIT-BIH Arrhythmia Database v1.0.0, PhysioNet (Moody and Mark, 2005) | **Open Data Commons Attribution License v1.0** (https://physionet.org/content/mitdb/view-license/1.0.0/) | Download them from PhysioNet. Cite the database DOI (10.13026/C2F305), Moody and Mark (2001) and the standard PhysioNet citation (Pollard et al., 2026) |
| `runs/*/checkpoints/*.pt`, `historical/v1/results/*.pt` | Model weights trained on MIT-BIH beat windows | Code output (MIT) produced from ODC-By data | Attribute the MIT-BIH Arrhythmia Database when reusing the weights |
| `runs/*/predictions/*.npz`, `runs/*/metrics.json`, `runs/*/summary.md`, `historical/v1/results/metrics.json` | Model outputs and metrics computed against the MIT-BIH reference annotations (the `.npz` files store reference class labels and record IDs) | ODC-By 1.0 (derived from the database) | Credit "MIT-BIH Arrhythmia Database, PhysioNet" and state that the data were modified (labels grouped into AAMI classes, beats windowed) |
| `runs/*/demo_record_200.mp4`, `historical/v1/results/demo_record_200.mp4` | Rendered ECG from MIT-BIH record 200 with its reference annotations, plus model predictions | ODC-By 1.0 for the ECG and annotations | Same credit as above; the videos show a cropped 20 s segment and whole-record counts |
| `docs/media/` (README animations and images) | Screen recordings of this project's explorer app; excerpts of the two demo videos; the per-class recall plots | ODC-By 1.0 for the ECG and labels shown; MIT for the rest | Same credit as above. These are adaptations: recorded with headless Chromium, cropped, resized, re-timed and re-encoded as animated WebP/GIF, JPEG and PNG |

Notes:

- PhysioNet's page for the database lists three citations to include: the database itself (Moody and Mark, 2005), the original publication (Moody and Mark, 2001) and the standard PhysioNet citation (Pollard et al., 2026). Full references are in the README.
- The Kaggle "ECG Heartbeat Categorization Dataset" (Kachuee, Fazeli and Sarrafzadeh, 2018) was not used and is not redistributed here.
