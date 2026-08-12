# Reproducibility map

| Manuscript item | Implementation | Archived values |
|---|---|---|
| 1ch / 7ch EMA / 7ch no-EMA U-Net | `code/train_jian_unet.py` | `results/unet/table_new_unet_ablation_metrics.csv` |
| OOD45 power sweep | `code/create_jian_ood_power_sweep_manifest.py` plus U-Net inference | `results/unet/table_new_ood45_power_sweep_by_factor.csv` |
| CMB life field | `code/compute_life_field_from_npz.py` | generated per run |
| Six-input parameterized P-PINN | `code/train_parameterized_ppinn.py` | generated per run |
| Fixed-material 2M P-PINN benchmark | `code/compare_life_field_pinn_numerical_solvers.py` | `results/life_solver/` |
| Data-only neural ablation | `code/train_data_only_mlp_vs_pure_physics_jian_2m.py` | `results/life_solver/data_only_vs_pure_physics_metrics.csv` |
| External lifetime validation | `code/evaluate_external_validation_lowest_fraction.py` | `results/external_validation/` |

The fixed-material benchmark uses `epsilon_f'=0.325`, `sigma_f'=263.535316568
MPa`, `b=-0.1443`, `c=-0.57`, and `E=31,200 MPa`. Its P-PINN learns only the
mapping from `(epsilon_a, sigma_mean)` to life for that calibrated equation.

The external validation performs a separate calibration on the COMSOL
no-vibration field. It obtains `sigma_f'` near 212.385 MPa for the lowest-0.1%
aggregation, then applies it without refitting to the U-Net 10 Hz and 30 Hz
fields. With 12,540 valid pixels, `ceil(0.001 * 12540) = 13` pixels are averaged.

The archived solver times were measured on the original experiment host. Use
the result JSON for hardware/software metadata and rerun on one machine before
making a new speed comparison.
