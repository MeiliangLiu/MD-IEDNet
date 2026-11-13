# MD-IEDNet

PyTorch implementation of **MD-IEDNet**, a morphology-aware directed graph network for interictal epileptiform discharge (IED) detection in multichannel EEG.

The repository includes binary classification on TUEV and VEPI, VEPI-to-TUEV transfer experiments, and five-class IED spatial classification on VEPI.


## Repository files

| File | Purpose |
| --- | --- |
| `model.py` | Binary MD-IEDNet architecture |
| `TUEV.py` | EDF/REC preprocessing and binary training on TUEV |
| `VEPI_2class.py` | Binary training on VEPI NPY segments |
| `CrossDataset.py` | Four VEPI-to-TUEV transfer protocols |
| `5class/model_5ied.py` | Five-class version of the architecture |
| `5class/train_vepi_5ied.py` | Patient-level five-class VEPI experiments |


## Data preparation

### TUEV

Set `data_root` to the directory containing the official `train/` and `eval/` folders. Each must contain matching `.edf` and `.rec` files, directly or in subdirectories.

| Example path | Meaning |
| --- | --- |
| `<data_root>/train/<patient_id>/<recording>.edf` | Training EEG recording |
| `<data_root>/train/<patient_id>/<recording>.rec` | Corresponding event annotations |
| `<data_root>/eval/<patient_id>/<recording>.edf` | Official evaluation EEG recording |
| `<data_root>/eval/<patient_id>/<recording>.rec` | Corresponding event annotations |

The first directory beneath each split is used as the patient identifier. If files are placed directly in a split folder, the recording filename stem is used instead. Preserve a genuine patient grouping: multiple recordings from one patient must not be treated as different patients.

`TUEV.py` groups annotations with identical start/stop times, extracts a centered 1-second window, constructs 20 bipolar TCP leads, resamples to 256 points, and applies per-segment, per-lead z-score normalization. Events containing any of SPSW, GPED, or PLED are labeled IED; EYEM, ARTF, and BCKG are non-IED unless the same interval also has an IED annotation.

Binary label convention throughout the repository:

| Label | Class | Probability used for metrics |
| --- | --- | --- |
| `0` | True-IED | `softmax(logits, dim=1)[:, 0]` |
| `1` | Non-IED | — |

The TCP order is:

```text
FP1-F7, F7-T3, T3-T5, T5-O1,
FP2-F8, F8-T4, T4-T6, T6-O2,
T3-C3, C3-CZ, CZ-C4, C4-T4,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2
```

A1/A2-dependent leads are excluded. 

### VEPI binary classification

Set `data_dir` to a directory containing `True-IED/` and `Non-IED/`, each with `.npy` segment files. The default patient regex `^([^_]+)_` treats the filename prefix before the first underscore as the patient identifier, for example `DA00100A` in `DA00100A_0_2000_500_0.npy`.


### VEPI five-class classification

The five-class loader expects these exact subdirectory names, in label order:

| Label | Directory |
| --- | --- |
| `0` | `Generalized-IED` |
| `1` | `Frontal-IED` |
| `2` | `Temporal-IED` |
| `3` | `Centro-Parietal-IED` |
| `4` | `Occipital-IED` |


## Running experiments

| Experiment | Script to run | Paths/settings to configure |
| --- | --- | --- |
| TUEV binary | `TUEV.py` | `data_root`, `cache_dir` |
| VEPI binary | `VEPI_2class.py` | `data_dir`, `group_regex` |
| VEPI → TUEV | `CrossDataset.py` | `vepi_data_dir`, `tuev_data_root`, `cache_dir`, `output_dir`, `vepi_channel_names`, `vepi_sfreq` |
| VEPI five-class | `5class/train_vepi_5ied.py` | `data_dir`, `result_dir`, `group_regex` |

All default experiments use training seeds 42–51 and a fixed split seed of 42.


### Default hyperparameters

These values describe the supplied code, rather than an independently verified paper reproduction.

| Parameter | TUEV binary | VEPI binary | VEPI five-class | Cross-dataset |
| --- | --- | --- | --- | --- |
| Input channels | 20 TCP | 19 | 19 | 20 TCP |
| Input samples | 256 | 2000 | 2000 | 2000 |
| Batch size | 96 | 256 | 256 | 128 |
| `node_dim` | 24 | 48 | 48 | 24 |
| `graph_hidden` | 48 | 128 | 128 | 48 |
| `rank_dim` | 12 | 12 | 12 | 12 |
| Dropout | 0.10 | 0.20 | 0.10 | 0.10 |
| Maximum learning rate | 1e-3 | 5e-3 | 1e-3 | See stages below |
| Maximum epochs | 100 | 100 | 100 | See stages below |
| Early-stopping patience | 15 | 15 | 15 | See stages below |
| Validation selection metric | AUPRC/AP | AUPRC/AP | Macro-F1 | AUPRC/AP |

All use AdamW, weight decay `1e-4`, cross-entropy loss, and OneCycleLR (`pct_start=0.10`, `div_factor=10`, `final_div_factor=100`). Five-class loss is unweighted by default; `use_class_weights=True` enables inverse-square-root training-frequency weighting. `time_bins` is a legacy field and does not control the encoded sequence length.

### Data splits and transfer protocols

- **TUEV:** patient-group train/validation partition within official training data; official evaluation data is reserved for final testing.
- **VEPI binary:** approximate 70%/10%/20% train/validation/test split with patient groups kept intact.
- **VEPI five-class:** approximately 70%/10%/20% of patients, with candidate splits selected to balance class counts and include all five classes in every split.
- **Transfer:** approximately 90%/10% source train/validation. From official TUEV training data, approximately 10% is used for gradient training, a separate 10% for validation, and 80% is unused. Fractions are approximate because patient groups are indivisible.

Thus the transfer setup uses target labels for both training and validation; the 10% training fraction is not the total target-label budget.

| Protocol/stage | Training data | Validation | Max LR | Epochs / patience |
| --- | --- | --- | --- | --- |
| Source pretraining | VEPI training split | VEPI validation | 1e-3 | 100 / 15 |
| Source-Only | Source-pretrained model; no target optimization | Source checkpoint selection | — | — |
| Pretrain-Finetune | Fine-tune all pretrained parameters on TUEV subset | TUEV validation | 1e-4 | 50 / 10 |
| Target-Only | TUEV subset, random initialization | TUEV validation | 1e-3 | 100 / 15 |
| Joint-Training | VEPI training split + TUEV subset, random initialization | TUEV validation | 1e-3 | 100 / 15 |

All four transfer protocols are evaluated on the official TUEV evaluation set. By default, joint training assigns equal total sampling probability to the two domains using replacement sampling.


## Metrics and outputs

Binary metrics are accuracy, IED precision, recall, specificity, F1, AUROC, and average precision (reported as AUPRC). Five-class precision, recall, specificity, F1, AUROC, and AP are macro-averaged; AUROC/AP use one-vs-rest probabilities. Accuracy is overall sample accuracy. Ten-seed summaries use sample standard deviation (`ddof=1`).

| Script | Outputs in the supplied implementation |
| --- | --- |
| `TUEV.py` | Preprocessing caches; printed metrics; Python return values from training functions |
| `VEPI_2class.py` | Printed metrics; Python return values from training functions |
| `CrossDataset.py` | `cross_dataset_config.json`, `cross_dataset_per_seed.csv`, `cross_dataset_summary.csv`; optional per-seed checkpoints |
| `5class/train_vepi_5ied.py` | `patient_split.csv`, `seed_metrics.csv`, `per_class_metrics.csv`, `confusion_seed<seed>.csv`, `summary_5ied.json` |