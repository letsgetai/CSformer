# Reproduction

This record separates paper-reported values from a single-seed run of the 28 CSformer configurations.
Paper values are from Table I of [Cross-Scale Attention for Long-term Time Series Forecasting](https://doi.org/10.1109/LSP.2024.3439103).

## Setup

| Item | Recorded setup |
|---|---|
| Experiment | `single-seed-legacy-v1` |
| Datasets / horizons | Seven datasets; 96, 192, 336, 720 |
| Seed | 2024 |
| Completed / failed | 28 / 0 |
| Hardware | One NVIDIA A100 |
| Software | Python 3.10.12, PyTorch 2.6.0+cu124, CUDA 12.4 |
| Selection | Lowest validation MSE |
| Test samples | All windows, including the final incomplete batch |
| Package versions | `requirements-cu124.lock.txt` |
| Data provenance | `metadata/dataset-sources.json` |

The result record includes dataset and raw-log SHA-256 values, all per-run parameters, and an
implementation source fingerprint. Original logs, predictions and checkpoints are not bundled.
CPU pipeline checks used macOS arm64, Python 3.14.6 and PyTorch 2.14.1; their exact environment is
in `requirements-cpu-macos.lock.txt`. Other platforms and multi-GPU operation were not validated.

### Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements-cu124.txt
```

## Data and configurations

The CSV files come from the [Autoformer dataset package](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).
The four ETT CSVs match the original ETDataset files byte-for-byte. `metadata/dataset-sources.json`
contains complete SHA-256 values and sizes. Scaling uses training data only. ETT uses 12/4/4
months for train/validation/test; Weather, Electricity and Traffic use 7:1:2 splits.

`metadata/paper-configs.json` contains all 28 configurations. `paper` contains explicitly reported
architectural settings; `archive_only` contains additional training parameters from the original
author scripts. The paper does not specify all training parameters.

## Validation protocol

- `legacy` (default): shuffled validation with `drop_last=True`, used for the recorded runs.
  The validation subset can vary by epoch, adding noise to checkpoint selection.
- `full`: unshuffled validation with `drop_last=False`. Every validation window is evaluated in
  a fixed order. This is recommended for new experiments and uses a separate `_val-full` output suffix.

Training uses shuffled, dropped-tail batches. Test and prediction loaders are unshuffled and
complete. Early stopping uses validation MSE only. The current training loop does not evaluate
the test set until the selected checkpoint is reloaded for final testing.

### Implementation version

The table below belongs to the recorded implementation `single-seed-legacy-v1`, identified by
`provenance.implementation_source_sha256` in `metadata/reproduced-results.json`. The source digest
covers the model, data and training Python files; its definition is included with the record.

The recorded implementation also traversed the test loader each epoch without using test loss
for checkpoint selection. That traversal has been removed in the current code. Constructing
DataLoader iterators consumes RNG, so removing it can change subsequent shuffle/dropout draws.
The mathematical model is unchanged, but identical seeds do not guarantee the same training
path. The current pipeline has separate CPU and A100 checks; it was not rerun over all 28 configurations.

Traffic has 862 channels. The recorded command supplied `enc_in=321`; under the reported settings
(`individual=0`, RevIN `affine=0`) this value does not affect CSformer predictions. Current
launchers use the correct channel count 862.

## Results

Values are single-run, seed 2024. Delta is reproduced minus paper; relative deltas divide by
paper values. Complete precision and both relative MSE/MAE deltas are in `metadata/reproduced-results.json`.

| Dataset | H | MSE | MAE | Paper MSE | Paper MAE | ΔMSE | ΔMAE | ΔMSE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.381637 | 0.404005 | 0.381 | 0.403 | +0.000637 | +0.001005 | +0.17 |
| ETTh1 | 192 | 0.413210 | 0.421945 | 0.413 | 0.421 | +0.000210 | +0.000945 | +0.05 |
| ETTh1 | 336 | 0.436233 | 0.440531 | 0.434 | 0.438 | +0.002233 | +0.002531 | +0.51 |
| ETTh1 | 720 | 0.456085 | 0.475955 | 0.448 | 0.470 | +0.008085 | +0.005955 | +1.80 |
| ETTh2 | 96 | 0.277094 | 0.339880 | 0.277 | 0.340 | +0.000094 | -0.000120 | +0.03 |
| ETTh2 | 192 | 0.337981 | 0.378518 | 0.337 | 0.379 | +0.000981 | -0.000482 | +0.29 |
| ETTh2 | 336 | 0.359872 | 0.397569 | 0.359 | 0.397 | +0.000872 | +0.000569 | +0.24 |
| ETTh2 | 720 | 0.391395 | 0.426844 | 0.392 | 0.426 | -0.000605 | +0.000844 | -0.15 |
| ETTm1 | 96 | 0.289966 | 0.343363 | 0.291 | 0.343 | -0.001034 | +0.000363 | -0.36 |
| ETTm1 | 192 | 0.329258 | 0.366247 | 0.329 | 0.366 | +0.000258 | +0.000247 | +0.08 |
| ETTm1 | 336 | 0.363952 | 0.387866 | 0.364 | 0.388 | -0.000048 | -0.000134 | -0.01 |
| ETTm1 | 720 | 0.419645 | 0.420932 | 0.419 | 0.421 | +0.000645 | -0.000068 | +0.15 |
| ETTm2 | 96 | 0.166481 | 0.257329 | 0.166 | 0.257 | +0.000481 | +0.000329 | +0.29 |
| ETTm2 | 192 | 0.224235 | 0.295472 | 0.223 | 0.295 | +0.001235 | +0.000472 | +0.55 |
| ETTm2 | 336 | 0.276008 | 0.330976 | 0.276 | 0.331 | +0.000008 | -0.000024 | +0.00 |
| ETTm2 | 720 | 0.374278 | 0.390069 | 0.370 | 0.387 | +0.004278 | +0.003069 | +1.16 |
| Weather | 96 | 0.149740 | 0.198049 | 0.149 | 0.197 | +0.000740 | +0.001049 | +0.50 |
| Weather | 192 | 0.193446 | 0.240630 | 0.196 | 0.244 | -0.002554 | -0.003370 | -1.30 |
| Weather | 336 | 0.247618 | 0.281896 | 0.246 | 0.283 | +0.001618 | -0.001104 | +0.66 |
| Weather | 720 | 0.324056 | 0.336527 | 0.323 | 0.335 | +0.001056 | +0.001527 | +0.33 |
| Traffic | 96 | 0.362223 | 0.254049 | 0.362 | 0.254 | +0.000223 | +0.000049 | +0.06 |
| Traffic | 192 | 0.384281 | 0.262774 | 0.384 | 0.263 | +0.000281 | -0.000226 | +0.07 |
| Traffic | 336 | 0.396581 | 0.267240 | 0.396 | 0.267 | +0.000581 | +0.000240 | +0.15 |
| Traffic | 720 | 0.430506 | 0.286945 | 0.430 | 0.287 | +0.000506 | -0.000055 | +0.12 |
| Electricity | 96 | 0.129689 | 0.222828 | 0.130 | 0.222 | -0.000311 | +0.000828 | -0.24 |
| Electricity | 192 | 0.148371 | 0.239665 | 0.148 | 0.240 | +0.000371 | -0.000335 | +0.25 |
| Electricity | 336 | 0.165422 | 0.257219 | 0.165 | 0.257 | +0.000422 | +0.000219 | +0.26 |
| Electricity | 720 | 0.202642 | 0.290181 | 0.203 | 0.290 | -0.000358 | +0.000181 | -0.18 |

26 of 28 MSE differences are at most 0.0026. This describes the observed differences; it is not
a predefined acceptance threshold. Weather-192 is 0.002554 below the paper.

## Deviations

### ETTh1, horizon 720

The single-seed MSE is 0.456085 versus 0.448 in the paper (+1.80%). An earlier experiment with
the same scale-3 setting reached 0.454304. Another experiment reached 0.447967 with scale 4, but
scale 4 uses 15 tokens and conflicts with the paper's scale-3 setting for ETTh1. It does not
explain the paper value. The discrepancy remains unresolved; no tuning was applied.

### ETTm2, horizon 720

The single-seed MSE is 0.374278 versus 0.370 in the paper (+1.16%). An earlier experiment with
the same recorded hyperparameters reached 0.370336. Framework numerics, stochastic checkpoint
selection and GPU nondeterminism are possible factors, but no controlled comparison has isolated
the cause. The actual framework version of the earlier experiment is unknown.

## Pipeline checks

The current code was tested on ETTh1-96 with seed 2024, legacy validation, up to 100 epochs and
patience 20. Both environments reproduced their saved-checkpoint metrics exactly after reload.

| Environment | MSE | MAE |
|---|---:|---:|
| macOS CPU | 0.380591 | 0.403055 |
| Linux / A100 | 0.380746 | 0.403328 |

These checks verify the training/evaluation path, not 28-configuration numerical equivalence.
Multi-seed statistics, baseline reproduction and the paper's GPU inference-efficiency table are
not included in this validation.

## Reproducing the run

```bash
python tools/run_reproduction.py \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity \
  --horizons 96 192 336 720 \
  --data-root /path/to/datasets --output /path/to/new-run-directory --device cuda
```

The runner requires a clean Git checkout and an output directory outside the source tree. It
records the current code revision, packages, data hashes, commands and metrics and stops if code,
dependencies or data change during execution. It selects the legacy validation protocol.

## Implementation and paper assets

The original model name `FTST` is retained as a checkpoint-compatible module; `CSformer` is the
paper-name entry point. The architecture image and editable draw.io source match paper Figure 1,
with only `MutilScale Patching` corrected to `Multi-scale Patching`. `Flatten Predicted Head` is
retained from the paper. README baseline means are computed from Table I and recorded in
`metadata/paper-comparison.json`; these are separate from the new single-seed results. The
latency comparison quotes Table II and has not been re-benchmarked here.
