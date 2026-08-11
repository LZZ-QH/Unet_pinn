#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a small Jian physical-condition OOD manifest.

This set is intentionally separated from the main500 training labels and from
the spatial-power OOD40 set. It is a stress-test package for physical-condition
perturbations that are not explicit U-Net input channels, such as cooling
coefficient and solder Young's modulus.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "paper_2026" / "configs" / "layouts"
OUT_DIR = ROOT / "paper_2026" / "artifacts" / "ood_phys_stress_test"
POWER_DIR = OUT_DIR / "input_power_grid_256x256_ood_phys"
FIG_DIR = OUT_DIR / "figures"
MANIFEST_PATH = OUT_DIR / "case_manifest_jian_ood_phys_stress_test.csv"

GRID_X_MIN, GRID_X_MAX = -25.0, 25.0
GRID_Y_MIN, GRID_Y_MAX = -17.25, 17.25
GRID_N = 256
ACTIVE_THICKNESS_MM = 0.2


@dataclass(frozen=True)
class LayoutSpec:
    short: str
    layout_id: str
    config_path: Path
    sample_start: int
    center_total_power_w: float
    runner_layout_id: str
    runner_config_path: str


LAYOUTS = (
    LayoutSpec("L0", "L0_original_jian", CONFIG_DIR / "jian_layout_L0_original.json", 30001, 497.313595612, "", ""),
    LayoutSpec("L4", "L4_staggered_asymmetric", CONFIG_DIR / "jian_layout_L4_staggered_asymmetric.json", 30401, 332.575710, "L4_staggered_asymmetric", "paper_2026/configs/layouts/jian_layout_L4_staggered_asymmetric.json"),
    LayoutSpec("L5", "L5_irregular_scattered", CONFIG_DIR / "jian_layout_L5_irregular_scattered.json", 30501, 357.278906633, "L5_irregular_scattered", "paper_2026/configs/layouts/jian_layout_L5_irregular_scattered.json"),
    LayoutSpec("L6", "L6_diagonal_zigzag", CONFIG_DIR / "jian_layout_L6_diagonal_zigzag.json", 30601, 170.224002, "L6_diagonal_zigzag", "paper_2026/configs/layouts/jian_layout_L6_diagonal_zigzag.json"),
    LayoutSpec("L7", "L7_split_asymmetric_clusters", CONFIG_DIR / "jian_layout_L7_split_asymmetric_clusters.json", 30701, 293.949054, "L7_split_asymmetric_clusters", "paper_2026/configs/layouts/jian_layout_L7_split_asymmetric_clusters.json"),
)


CONDITIONS = (
    # name, power_factor, h_W_m2K, solder_E_GPa, category
    ("cooling_weak_h4800", 1.00, 4800.0, 31.2, "cooling"),
    ("cooling_strong_h9200", 1.00, 9200.0, 31.2, "cooling"),
    ("power_low_0p85", 0.85, 6800.0, 31.2, "total_power"),
    ("power_high_1p15", 1.15, 6800.0, 31.2, "total_power"),
    ("solder_E_low_25GPa", 1.00, 6800.0, 25.0, "material"),
    ("solder_E_high_38GPa", 1.00, 6800.0, 38.0, "material"),
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def active_boxes(config: dict) -> list[tuple[float, float, float, float]]:
    if "islands" in config and all("active_box" in item for item in config["islands"]):
        items = sorted(config["islands"], key=lambda item: int(item.get("idx", len(config["islands"]))))
        return [tuple(map(float, item["active_box"])) for item in items]
    centers = np.asarray(config["centers_xy_mm"], dtype=float)
    sx, sy = map(float, config.get("chip_active_size_mm", [6.0, 6.0]))
    return [(cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0) for cx, cy in centers]


def make_power_grid(boxes: list[tuple[float, float, float, float]], chip_powers_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(GRID_X_MIN, GRID_X_MAX, GRID_N, dtype=np.float32)
    ys = np.linspace(GRID_Y_MIN, GRID_Y_MAX, GRID_N, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    power = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for power_w, (x0, x1, y0, y1) in zip(chip_powers_w, boxes):
        area_mm2 = max((x1 - x0) * (y1 - y0), 1e-12)
        volume_m3 = area_mm2 * ACTIVE_THICKNESS_MM * 1e-9
        density = float(power_w) / volume_m3
        mask = (xg >= x0) & (xg <= x1) & (yg >= y0) & (yg <= y1)
        power[mask] = density
    return power, xg.astype(np.float32), yg.astype(np.float32)


def write_power_npz(path: Path, layout: LayoutSpec, boxes, chip_powers_w: np.ndarray, h: float, solder_e: float) -> None:
    power, xg, yg = make_power_grid(boxes, chip_powers_w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=power[None].astype(np.float32),
        power_grid=power.astype(np.float32),
        X_grid=xg,
        Y_grid=yg,
        chip_names=np.asarray([f"chip{i}" for i in range(1, 7)]),
        chip_powers_W=chip_powers_w.astype(np.float32),
        layout_id=np.asarray(layout.layout_id),
        layout_config_path=np.asarray(str(layout.config_path.relative_to(ROOT))),
        h_W_m2K=np.asarray(h, dtype=np.float32),
        solder_E_GPa=np.asarray(solder_e, dtype=np.float32),
    )


def build() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        config = load_config(layout.config_path)
        boxes = active_boxes(config)
        base_chip_power = layout.center_total_power_w / 6.0
        for offset, (condition, power_factor, h_value, solder_e_gpa, category) in enumerate(CONDITIONS):
            sample_id = layout.sample_start + offset
            chip_powers = np.full(6, base_chip_power * power_factor, dtype=float)
            case_name = f"{layout.short}_{condition}_oodphys"
            power_path = POWER_DIR / f"power_grid_{sample_id}_{layout.layout_id}_{condition}_oodphys_256x256.npz"
            write_power_npz(power_path, layout, boxes, chip_powers, h_value, solder_e_gpa)
            row = {
                "sample_id": sample_id,
                "case_name": case_name,
                "ood_category": category,
                "ood_pattern": condition,
                "total_power_W": float(chip_powers.sum()),
                "power_factor": power_factor,
                "ncycles": 5,
                "tstep_s": 2,
                "ton_s": 20,
                "toff_s": 20,
                "dense_nxy": 64,
                "dense_z_mm": "0.88,0.93,0.98",
                "h_W_m2K": h_value,
                "solder_E_GPa": solder_e_gpa,
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


def write_figures(rows: list[dict[str, object]]) -> None:
    import pandas as pd

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
    for ax, col, title, ylabel in [
        (axes[0], "total_power_W", "Total power", "W"),
        (axes[1], "h_W_m2K", "Cooling coefficient", r"W/(m$^2$ K)"),
        (axes[2], "solder_E_GPa", "Solder Young's modulus", "GPa"),
    ]:
        for layout, sub in df.groupby("layout_short"):
            ax.scatter([layout] * len(sub), sub[col], s=25, label=layout if col == "total_power_W" else None)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(1.7, 1.25))
    fig.savefig(FIG_DIR / "jian_ood_phys_condition_summary.png", dpi=220)
    plt.close(fig)

    readme = FIG_DIR / "README_jian_ood_phys_stress_test.md"
    readme.write_text(
        "# Jian-OOD-Phys stress-test set\n\n"
        "This package contains 30 cases: 5 layout families x 6 physical-condition perturbations.\n\n"
        "It is not mixed into the main500 training set. Cooling coefficient and solder Young's modulus are not explicit U-Net inputs, so these cases should be interpreted as limited robustness diagnostics rather than the primary generalization claim.\n\n"
        f"- Manifest: `{MANIFEST_PATH}`\n"
        f"- Power grids: `{POWER_DIR}`\n"
        "- Conditions: cooling weak/strong, total power low/high, solder Young's modulus low/high.\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = build()
    write_manifest(rows)
    write_figures(rows)
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Wrote rows: {len(rows)}")
    print(f"Wrote figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
