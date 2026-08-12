#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Jian total-power sweep OOD cases.

This package is separated from the mixed physical-condition OOD stress test.
It keeps cooling and solder material fixed, and sweeps the total chip power
from 0.80x to 1.20x around each layout-specific thermal center.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from create_jian_ood_phys_manifest import LAYOUTS, active_boxes, load_config, write_power_npz


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_2026" / "artifacts" / "ood_power_sweep_0p80_1p20"
POWER_DIR = OUT_DIR / "input_power_grid_256x256_ood_power_sweep"
FIG_DIR = OUT_DIR / "figures"
MANIFEST_PATH = OUT_DIR / "case_manifest_jian_ood_power_sweep_0p80_1p20.csv"

POWER_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
SAMPLE_STARTS = {
    "L0": 31001,
    "L4": 31401,
    "L5": 31501,
    "L6": 31601,
    "L7": 31701,
}


def factor_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        config = load_config(layout.config_path)
        boxes = active_boxes(config)
        base_chip_power = layout.center_total_power_w / 6.0
        start = SAMPLE_STARTS[layout.short]
        for offset, factor in enumerate(POWER_FACTORS):
            sample_id = start + offset
            tag = factor_tag(factor)
            chip_powers = np.full(6, base_chip_power * factor, dtype=float)
            case_name = f"{layout.short}_power_factor_{tag}_oodpower"
            power_path = POWER_DIR / f"power_grid_{sample_id}_{layout.layout_id}_power_factor_{tag}_oodpower_256x256.npz"
            write_power_npz(power_path, layout, boxes, chip_powers, 6800.0, 31.2)
            row = {
                "sample_id": sample_id,
                "case_name": case_name,
                "ood_category": "total_power_sweep",
                "ood_pattern": f"power_factor_{tag}",
                "total_power_W": float(chip_powers.sum()),
                "power_factor": factor,
                "ncycles": 5,
                "tstep_s": 2,
                "ton_s": 20,
                "toff_s": 20,
                "dense_nxy": 64,
                "dense_z_mm": "0.88,0.93,0.98",
                "h_W_m2K": 6800.0,
                "solder_E_GPa": 31.2,
                "dense_field": 1,
                "heat_source_mode": "six_active_silicon_domains_layout_aware",
                "power_grid_npz": str(power_path.relative_to(ROOT)),
                "layout_short": layout.short,
                "layout_id": layout.runner_layout_id,
                "layout_config_path": layout.runner_config_path,
            }
            for i, value in enumerate(chip_powers, start=1):
                row[f"chip{i}_power_W"] = float(value)
            rows.append(row)
    return rows


def write_manifest(rows: list[dict[str, object]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "case_name", "ood_category", "ood_pattern",
        "total_power_W", "power_factor", "ncycles", "tstep_s", "ton_s", "toff_s",
        "dense_nxy", "dense_z_mm",
        "chip1_power_W", "chip2_power_W", "chip3_power_W",
        "chip4_power_W", "chip5_power_W", "chip6_power_W",
        "h_W_m2K", "solder_E_GPa", "dense_field", "heat_source_mode",
        "power_grid_npz", "layout_short", "layout_id", "layout_config_path",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    factors = np.array(POWER_FACTORS)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for layout in LAYOUTS:
        vals = [row["total_power_W"] for row in rows if row["layout_short"] == layout.short]
        ax.plot(factors, vals, marker="o", label=layout.short)
    ax.set_xlabel("Power factor")
    ax.set_ylabel("Total chip power (W)")
    ax.set_title("Jian OOD total-power sweep cases")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5)
    fig.savefig(FIG_DIR / "jian_ood_power_sweep_total_power_summary.png", dpi=220)
    plt.close(fig)

    readme = FIG_DIR / "README_jian_ood_power_sweep_0p80_1p20.md"
    readme.write_text(
        "# Jian OOD total-power sweep 0.80--1.20\n\n"
        "This OOD package keeps cooling and solder material fixed, and varies only "
        "the total power level around each layout-specific thermal center.\n\n"
        f"- Manifest: `{MANIFEST_PATH}`\n"
        f"- Power grids: `{POWER_DIR}`\n"
        "- Layouts: L0, L4, L5, L6, L7\n"
        "- Factors: 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20\n"
        "- Cases: 45\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    write_manifest(rows)
    write_summary(rows)
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Wrote rows: {len(rows)}")
    print(f"Wrote summary: {FIG_DIR}")


if __name__ == "__main__":
    main()
