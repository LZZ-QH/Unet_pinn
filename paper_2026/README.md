# Revised transient fatigue-variable workflow

This directory contains the implementation and compact reproducibility assets
for the revised manuscript. The release separates two life-network settings
that served different experimental purposes:

- `train_parameterized_ppinn.py` is the six-input parameterized P-PINN from the
  method section. Its inputs are `epsilon_a`, `sigma_mean`, `epsilon_f_prime`,
  `sigma_f_prime`, `c`, and `E`; `b=-0.12` is fixed.
- `compare_life_field_pinn_numerical_solvers.py` is the two-input, fixed-material
  pure-physics solver used for the reported 2M-pixel runtime table. This is the
  source of the reported 0.1226% mean relative error and 0.179 s CPU inference.

Keeping those settings explicit avoids presenting a case-calibrated benchmark
as though all material parameters varied during that benchmark.

## Repository map

```text
paper_2026/
  code/          label conversion, U-Net, P-PINN, solvers, OOD, validation
  configs/       portable U-Net and five-layout configurations
  results/       CSV/JSON values used by the revised manuscript
  sample_data/   compact external-validation fields (not the 2500-sample set)
  tests/         numerical and model plumbing checks
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r paper_2026/requirements.txt
```

For CUDA training, install the PyTorch build matching the local CUDA runtime
before installing the remaining requirements.

## Fast checks

Run the unit tests:

```bash
python -m unittest discover -s paper_2026/tests -v
```

Reproduce the external-validation table from the included compact fields:

```bash
python paper_2026/code/evaluate_external_validation_lowest_fraction.py
```

Expected evaluation values are approximately 17,920.80 cycles (10 Hz, 2 mm)
and 13,366.01 cycles (30 Hz, 2 mm), corresponding to 2.40% and 4.53% error.
The fatigue coefficient is calibrated only on the COMSOL no-vibration field.

Smoke-test the six-input pure-physics network:

```bash
python paper_2026/code/train_parameterized_ppinn.py --smoke --device cpu
```

## U-Net data schema and training

Each label file is named `unet_label_<sample>_256x256.npz` and stores:

- `Y[0]`: equivalent total strain amplitude, `epsilon_eq_total_a`;
- `Y[1]`: cycle mean stress in MPa, `sigma_mean_MPa`;
- `valid_mask`: valid chip-solder pixels;
- `power_grid_npz`: the matching portable power-grid path or basename;
- `X_grid`, `Y_grid`, `case_name`, and `channel_names`.

The main seven input channels are normalized power, active-power mask, solder
mask, x coordinate, y coordinate, log total power, and log maximum-chip power.

After placing label/power pairs under `paper_2026/data/main2500`, run:

```bash
python paper_2026/code/smoke_check.py \
  --label-dir paper_2026/data/main2500/labels

python paper_2026/code/train_jian_unet.py \
  --label-dir paper_2026/data/main2500/labels \
  --input-mode power_mask_xy_total \
  --epochs 120 --batch-size 8 --base-ch 32
```

Add `--no-attention` for the no-EMA variant. The 1-channel baseline uses
`--input-mode power`.

To convert COMSOL TSV exports into label NPZ files:

```bash
python paper_2026/code/build_jian_unet_labels_and_figures.py \
  --manifest path/to/case_manifest.csv \
  --result-root path/to/comsol_tsv_exports \
  --prefer-dense
```

COMSOL itself is proprietary and the author-provided `.mph` model is not
redistributed in this code directory. The converter starts from exported TSV
fields and documents the six-island interpolation step used by the paper.

## Life solver experiments

Train/evaluate the six-input parameterized P-PINN exactly with the manuscript
defaults (40,000 samples, 80/20 split, 2,500 epochs):

```bash
python paper_2026/code/train_parameterized_ppinn.py --device cuda
```

Sampling rejects and redraws parameter combinations whose CMB residual does
not bracket a physical root on `Nf=[1, 1e15]`. This endpoint equation check
does not generate numerical lifetime labels; bisection remains evaluation-only.

Run the fixed-material life-field benchmark after adding the full dataset and
the corresponding U-Net run metadata/checkpoint:

```bash
python paper_2026/code/compare_life_field_pinn_numerical_solvers.py \
  --label-dir paper_2026/data/main2500/labels \
  --run-dir paper_2026/artifacts/unet_7ch_ema \
  --max-samples 250 --max-pixels 2000000 --device cpu
```

The committed result files are archival measurements, not promises of identical
wall-clock time on different hardware. Accuracy should be comparable; runtime
depends on CPU, PyTorch, SciPy, process count, and warm-up protocol.

## Reported results

The exact tabulated inputs to the manuscript are under `results/`:

- U-Net ID/OOD45 ablation and power-factor sweep;
- 2M-pixel strict and accuracy-matched solver comparisons;
- data-only versus pure-physics neural-solver ablation;
- lowest-0.1% external-validation table.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for which script produces each
result and [DATA.md](DATA.md) for the public/private data boundary.
