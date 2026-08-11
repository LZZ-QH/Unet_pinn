# Unet_pinn

Code and reproducibility materials for **“Fatigue Life Prediction of High-Power
Chips using a Physics-Informed Deep Learning Methodology.”**

The current paper implementation is in [`paper_2026/`](paper_2026/README.md).
It supersedes the early endpoint-temperature/endpoint-stress prototype kept in
the repository root for provenance.

The updated workflow is:

```text
layout-aware power input
  -> transient COMSOL fatigue-variable labels
  -> 1/7-channel U-Net
  -> epsilon_eq_total_a(x,y), sigma_mean_MPa(x,y)
  -> CMB/Morrow numerical solver or pure-physics P-PINN
  -> Nf(x,y)
  -> lowest-0.1%-field device-life estimate
```

Start with the [paper-code README](paper_2026/README.md) for the environment,
data schema, commands, result provenance, and scope limitations.

> Note: the legacy directories describe the earlier power-to-temperature and
> power-to-von-Mises-stress study. They do not reproduce the revised manuscript.
