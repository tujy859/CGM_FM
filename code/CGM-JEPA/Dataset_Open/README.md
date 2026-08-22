---
language:
- en
license: cc-by-4.0
size_categories:
- 100K<n<1M
task_categories:
- time-series-forecasting
- feature-extraction
pretty_name: CGM-JEPA Pretraining Corpus
tags:
- continuous-glucose-monitor
- cgm
- self-supervised-learning
- representation-learning
- masked-prediction
- biosignal
- healthcare
- time-series
- pretraining
- jepa
configs:
- config_name: default
  data_files:
  - split: train
    path: cgm_initial_cohort.csv
---

# CGM-JEPA Pretraining Corpus

Continuous glucose monitor (CGM) time-series corpus used for self-supervised pretraining of **CGM-JEPA**, **X-CGM-JEPA**, **GluFormer**, and **TS2Vec** encoders in the paper [CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining](https://huggingface.co/papers/2605.00933).

**Code:** [https://github.com/cruiseresearchgroup/CGM-JEPA](https://github.com/cruiseresearchgroup/CGM-JEPA)

> Pretraining-only corpus. For the labeled downstream-evaluation cohorts (insulin resistance and β-cell dysfunction classification), see [`CRUISEResearchGroup/CGM-JEPA-Downstream`](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream). For pretrained model weights, see [`CRUISEResearchGroup/CGM-JEPA`](https://huggingface.co/CRUISEResearchGroup/CGM-JEPA).

## Quick start

### Option 1 — `datasets` library

```python
from datasets import load_dataset
ds = load_dataset("CRUISEResearchGroup/CGM-JEPA-Pretraining")
# DatasetDict({
#   train: Dataset({features: ['timestamp', 'glucose_value', 'subject'],
#                   num_rows: 389365})
# })
```

The natural unit for SSL pretraining is a 288-step (24-hour) window per `(subject, day)`; the code repo's `data_loaders/JEPALoader` performs the windowing on top of this raw flat CSV.

### Option 2 — file download for the code repo's pretraining scripts

```bash
huggingface-cli download CRUISEResearchGroup/CGM-JEPA-Pretraining \
  --repo-type dataset --local-dir Dataset_Open
```

Then from the [code repository](https://github.com/cruiseresearchgroup/CGM-JEPA):

```bash
# (Optional, only for X-CGM-JEPA) Precompute the Glucodensity cache once.
python -m utils.precompute_glucodensity \
  --data_path Dataset_Open/cgm_initial_cohort.csv \
  --output_path Dataset_Open/gluco_cache_stride288.pkl \
  --stride 288

# Pretrain
python -m pretrain.pretrain_x_cgm_jepa     # X-CGM-JEPA (cross-view)
python -m pretrain.pretrain_cgm_jepa       # CGM-JEPA (temporal only)
python -m pretrain.pretrain_gluformer      # GluFormer baseline
python -m pretrain.pretrain_ts2vec         # TS2Vec baseline
```

## Files

| File | Size | Purpose |
|---|---|---|
| `cgm_initial_cohort.csv` | ~12 MB | Raw CGM readings (5-min sampling) — primary pretraining input |

X-CGM-JEPA additionally needs a **Glucodensity cache** (~54 MB) which is *not* shipped here; it's derived deterministically from the CSV by [`utils/precompute_glucodensity.py`](https://github.com/cruiseresearchgroup/CGM-JEPA/blob/main/utils/precompute_glucodensity.py) in the code repository (one-time, ~few minutes on CPU).

## Schema

### `cgm_initial_cohort.csv`

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Reading time at 5-min resolution |
| `glucose_value` | float | Glucose value in mg/dL |
| `subject` | str | Subject identifier (`S01..S22` for Stanford; `colas_<subject>_<day>` for Colas subject-days) |

- **413 unique subject identifiers** (22 Stanford + 391 Colas subject-days = 206 unique Colas subjects across multiple days).
- **389,365 total readings**.
- Median 288 readings per subject (= one 24-hour window at 5-min sampling); max 18,527 readings for the longest Stanford continuous-CGM record.

## Cohort composition

| Source | Subjects | Rows | Share |
|---|---|---|---|
| Stanford CGM Study | 22 | 276,757 | 71.1 % |
| Colas et al. (2019) | 206 (391 subject-days) | 112,608 | 28.9 % |
| **Total** | **228 unique subjects** | **389,365** | 100 % |


## How this corpus was built

The corpus was assembled by [`scripts/preprocess_dataset.py`](https://github.com/cruiseresearchgroup/CGM-JEPA/blob/main/scripts/preprocess_dataset.py) in the code repository, from two upstream sources:

1. **Stanford CGM Study** (Metwally et al. 2025, *Nature Biomedical Engineering*) — data distributed through the [`Metabolic_Subphenotype_Predictor`](https://github.com/aametwally/Metabolic_Subphenotype_Predictor) repository under MIT license (files `filtered_cgm_03222026.csv`, `filtered_ogtt_…csv`, `filtered_metabolic_tests.csv`).
2. **Colas et al. (2019)** — Open-access CGM time series from *Detrended Fluctuation Analysis in the prediction of type 2 diabetes mellitus in patients at risk*, PLOS ONE 14(12):e0225817, distributed in the paper's Supporting Information files under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Stanford rows were smoothed onto a 5-min grid via cubic smoothing splines (`scipy.interpolate.make_smoothing_spline(lam=0.35)`). Sensor `"Low"`/`"High"` markers were replaced with the empirical numeric min/max. Colas daily windows were concatenated with prefixed subject IDs (`colas_<subject>_<day>`) to disambiguate them from Stanford subjects.

## Intended use

- Self-supervised pretraining of CGM encoders (masked-patch prediction, contrastive, autoencoding objectives).
- Comparison of CGM representation learning methods.

## License & attribution

This corpus is released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** to match the licenses of its two upstream sources:

- **Stanford CGM Study data** — released through [`Metabolic_Subphenotype_Predictor`](https://github.com/aametwally/Metabolic_Subphenotype_Predictor) under the MIT license, accompanying Metwally et al. 2025 (*Nature Biomedical Engineering*).
- **Colas et al. (2019)** — CGM time series from the PLOS ONE Supporting Information files, licensed under CC BY 4.0.

When using this dataset, please cite **all three sources**: the Stanford CGM Study (via `Metabolic_Subphenotype_Predictor`), Colas et al. 2019, and our CGM-JEPA paper.

## Citation

```bibtex
@article{muhammad2026cgm,
  title   = {CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining},
  author  = {Muhammad, Hada Melino and Li, Zechen and Salim, Flora and Metwally, Ahmed A},
  journal = {arXiv preprint arXiv:2605.00933},
  year    = {2026}
}
```