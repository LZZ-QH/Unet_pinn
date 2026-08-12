# Data and model availability

This repository revision publishes the paper source code, portable
configuration, result tables, and the compact field arrays required to verify
the external-validation calculation.

The complete 2500-sample COMSOL-derived training set and U-Net checkpoints are
not committed to Git history in this update. The local project contains about
1.7 TB, mostly proprietary COMSOL `.mph` models and raw TSV exports; publishing
that workspace wholesale would expose third-party model material and produce an
unusable code repository.

The expected portable full-dataset layout is:

```text
paper_2026/data/main2500/
  labels/unet_label_<sample>_256x256.npz
  power/power_grid_<sample>_..._256x256.npz
```

`jian_dataset.py` resolves a power path stored in a label and also supports a
copied dataset where the matching power file is found by basename beside the
label directory. Absolute paths from the original servers are not required.

The author-provided COMSOL `.mph` model is intentionally excluded. Users with
an authorized model copy can export the solder-cycle TSV fields and run
`build_jian_unet_labels_and_figures.py`.
