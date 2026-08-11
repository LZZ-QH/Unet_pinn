#!/usr/bin/env python3
"""Build Jian-geometry U-Net labels and visualizations from COMSOL TSV fields."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "paper_2026" / "data" / "manifests"
DEFAULT_DENSE_MANIFEST = (
    ROOT
    / "paper_2026" / "data" / "manifests" / "case_manifest_dense_batch.csv"
)
RESULT_ROOT = ROOT / "paper_2026" / "data" / "comsol_tsv"
OUT_ROOT = ROOT / "paper_2026" / "data" / "main2500"
LABEL_DIR = OUT_ROOT / "output_unet_256x256"
FIG_DIR = OUT_ROOT / "figures_interpolation_preview"
REPORT_DIR = OUT_ROOT / "reports"
MAKE_CASE_FIGURE = True

ACTIVE_CHIP_BOXES = [
    ("chip1_domain10", -21.5, -15.5, 6.25, 12.25),
    ("chip2_domain16", -2.8, 3.2, -8.85, -2.85),
    ("chip3_domain19", -1.2, 4.8, 2.65, 8.65),
    ("chip4_domain24", 6.4, 12.4, -8.85, -2.85),
    ("chip5_domain32", 14.8, 20.8, 2.65, 8.65),
    ("chip6_domain37", 15.6, 21.6, -8.85, -2.85),
]
SOLDER_MARGIN_MM = 0.85


def interpolation_regions(record: dict[str, str]) -> list[tuple[str, float, float, float, float]]:
    """Return solder-island interpolation boxes for a manifest record."""

    cfg_path = record.get("layout_config_path", "")
    if cfg_path:
        path = Path(cfg_path)
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            centers = np.asarray(cfg["centers_xy_mm"], dtype=float)
            solder_size = np.asarray(cfg.get("solder_island_size_mm", [7.5, 7.5]), dtype=float)
            half_x, half_y = solder_size[0] / 2.0, solder_size[1] / 2.0
            return [
                (f"chip{idx+1}_{cfg.get('layout_id', 'layout')}", cx - half_x, cx + half_x, cy - half_y, cy + half_y)
                for idx, (cx, cy) in enumerate(centers)
            ]

    return [
        (name, x0 - SOLDER_MARGIN_MM, x1 + SOLDER_MARGIN_MM, y0 - SOLDER_MARGIN_MM, y1 + SOLDER_MARGIN_MM)
        for name, x0, x1, y0, y1 in ACTIVE_CHIP_BOXES
    ]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def tsv_for_sample(sample_id: str, case_name: str, prefer_dense: bool) -> Path:
    if prefer_dense:
        dense_matches = sorted(
            RESULT_ROOT.glob(
                f"author_transient_Ac_50_A__case_{sample_id}_{case_name}"
                f"*_solder_cycle_field_dense*x*_z*.tsv"
            )
        )
        if dense_matches:
            return dense_matches[-1]

    if sample_id == "9101":
        dense_9101 = sorted(
            RESULT_ROOT.glob(
                "author_transient_Ac_50_A__case_9101_*"
                "_solder_cycle_field_dense*x*_z*.tsv"
            )
        )
        if prefer_dense and dense_9101:
            return dense_9101[-1]
        sparse_9101 = RESULT_ROOT / (
            "author_transient_Ac_50_A__n5_dt2_Esolder_31_2_GPa__h_6800_W_m_2_K"
            "__solid_on_chip_power_total_497_313595612W_solder_cycle_field.tsv"
        )
        if sparse_9101.exists():
            return sparse_9101

    matches = sorted(
        RESULT_ROOT.glob(
            f"author_transient_Ac_50_A__case_{sample_id}_{case_name}_n5_*_solder_cycle_field.tsv"
        )
    )
    if not matches:
        raise FileNotFoundError(f"No solder cycle field TSV found for {sample_id}_{case_name}")
    return matches[-1]


def combine_over_thickness(df: pd.DataFrame) -> pd.DataFrame:
    """Use max through-thickness value at each (x,y), matching the main workflow."""

    grouped = (
        df.groupby(["x_mm", "y_mm"], as_index=False)
        .agg(
            epsilon_eq_total_a=("epsilon_eq_total_a", "max"),
            sigma_mean_MPa=("sigma_mean_Pa", lambda s: float(np.max(s)) / 1e6),
            sigma_a_MPa=("sigma_a_Pa", lambda s: float(np.max(s)) / 1e6),
        )
        .sort_values(["x_mm", "y_mm"])
    )
    return grouped


def interpolate(points: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray):
    linear = griddata(points, values, (grid_x, grid_y), method="linear")
    nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
    valid = np.isfinite(linear)
    filled = np.where(valid, linear, nearest)
    return filled.astype(np.float32), valid


def interpolate_islandwise(
    combined: pd.DataFrame,
    value_column: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    regions: list[tuple[str, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each solder island separately to avoid nonphysical bridges."""

    out = np.zeros_like(grid_x, dtype=np.float32)
    valid_total = np.zeros_like(grid_x, dtype=bool)
    for _name, x0m, x1m, y0m, y1m in regions:
        point_mask = (
            (combined["x_mm"] >= x0m)
            & (combined["x_mm"] <= x1m)
            & (combined["y_mm"] >= y0m)
            & (combined["y_mm"] <= y1m)
        )
        region_mask = (
            (grid_x >= x0m)
            & (grid_x <= x1m)
            & (grid_y >= y0m)
            & (grid_y <= y1m)
        )
        local = combined.loc[point_mask]
        if len(local) < 3 or not np.any(region_mask):
            continue
        points = local[["x_mm", "y_mm"]].to_numpy(dtype=float)
        values = local[value_column].to_numpy(dtype=float)
        linear = griddata(points, values, (grid_x, grid_y), method="linear")
        nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
        valid = np.isfinite(linear) & region_mask
        filled = np.where(np.isfinite(linear), linear, nearest)
        out[region_mask] = filled[region_mask].astype(np.float32)
        valid_total |= valid
    return out, valid_total


def save_label(
    sample_id: str,
    case_name: str,
    record: dict[str, str],
    prefer_dense: bool,
) -> dict[str, object]:
    power_path = Path(record["power_grid_npz"])
    power_npz = np.load(power_path, allow_pickle=True)
    grid_x = np.asarray(power_npz["X_grid"], dtype=np.float32)
    grid_y = np.asarray(power_npz["Y_grid"], dtype=np.float32)
    power_grid = np.asarray(power_npz["power_grid"], dtype=np.float32)

    source_tsv = tsv_for_sample(sample_id, case_name, prefer_dense)
    df = pd.read_csv(source_tsv, sep="\t")
    combined = combine_over_thickness(df)
    points = combined[["x_mm", "y_mm"]].to_numpy(dtype=float)
    regions = interpolation_regions(record)

    epsilon_map, epsilon_valid = interpolate_islandwise(
        combined, "epsilon_eq_total_a", grid_x, grid_y, regions
    )
    sigma_map, sigma_valid = interpolate_islandwise(
        combined, "sigma_mean_MPa", grid_x, grid_y, regions
    )
    valid_mask = epsilon_valid & sigma_valid
    labels = np.stack([epsilon_map, sigma_map], axis=0).astype(np.float32)
    channel_names = np.asarray(["epsilon_eq_total_a", "sigma_mean_MPa"])

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = LABEL_DIR / f"unet_label_{sample_id}_256x256.npz"
    np.savez_compressed(
        out_npz,
        Y=labels,
        X_grid=grid_x,
        Y_grid=grid_y,
        valid_mask=valid_mask,
        channel_names=channel_names,
        source_file=source_tsv.name,
        power_grid_npz=power_path.name,
        case_name=case_name,
    )

    fig_path = FIG_DIR / f"jian_label_interpolation_{sample_id}_{case_name}.png"
    if MAKE_CASE_FIGURE:
        make_case_figure(
            fig_path,
            sample_id,
            case_name,
            grid_x,
            grid_y,
            power_grid,
            points,
            epsilon_map,
            sigma_map,
            valid_mask,
        )

    eps_valid = epsilon_map[valid_mask]
    sig_valid = sigma_map[valid_mask]
    return {
        "sample_id": sample_id,
        "case_name": case_name,
        "source_tsv": str(source_tsv),
        "power_grid_npz": str(power_path),
        "label_npz": str(out_npz),
        "figure": str(fig_path) if MAKE_CASE_FIGURE else "",
        "raw_rows": int(len(df)),
        "combined_xy_points": int(len(combined)),
        "valid_pixels": int(valid_mask.sum()),
        "valid_ratio": float(valid_mask.mean()),
        "epsilon_min_valid": float(eps_valid.min()),
        "epsilon_max_valid": float(eps_valid.max()),
        "sigma_mean_min_valid_MPa": float(sig_valid.min()),
        "sigma_mean_max_valid_MPa": float(sig_valid.max()),
    }


def imshow_grid(ax, grid_x, grid_y, data, title, cmap, label, mask=None):
    arr = np.array(data, dtype=float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    im = ax.imshow(
        arr,
        origin="lower",
        extent=[float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())],
        cmap=cmap,
        aspect="equal",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=8)
    return im


def make_case_figure(
    path: Path,
    sample_id: str,
    case_name: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power_grid: np.ndarray,
    points: np.ndarray,
    epsilon_map: np.ndarray,
    sigma_map: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), constrained_layout=True)
    fig.suptitle(
        f"Jian pilot sample {sample_id}: {case_name} (island-wise interpolation)",
        fontsize=13,
        fontweight="bold",
    )

    imshow_grid(
        axes[0, 0],
        grid_x,
        grid_y,
        power_grid / 1e10,
        "Input power density",
        "viridis",
        r"$10^{10}$ W/m$^3$",
    )
    axes[0, 0].scatter(points[:, 0], points[:, 1], s=3, c="white", alpha=0.45, linewidths=0)

    imshow_grid(
        axes[0, 1],
        grid_x,
        grid_y,
        epsilon_map,
        r"Island-wise $\varepsilon_{\mathrm{eq,total},a}$",
        "magma",
        "strain",
        valid_mask,
    )
    imshow_grid(
        axes[1, 0],
        grid_x,
        grid_y,
        sigma_map,
        r"Island-wise $\sigma_{\mathrm{mean}}$",
        "cividis",
        "MPa",
        valid_mask,
    )
    imshow_grid(
        axes[1, 1],
        grid_x,
        grid_y,
        valid_mask.astype(float),
        "Island-wise valid mask",
        "gray",
        "valid",
    )
    axes[1, 1].scatter(points[:, 0], points[:, 1], s=3, c="#2b6cb0", alpha=0.45, linewidths=0)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_overview(records: list[dict[str, object]], overview_limit: int) -> Path | None:
    shown = records[:overview_limit] if overview_limit > 0 else records
    if not shown:
        return None

    fig, axes = plt.subplots(len(shown), 3, figsize=(9.5, 2.25 * len(shown)), constrained_layout=True)
    if len(shown) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, rec in enumerate(shown):
        npz = np.load(rec["label_npz"], allow_pickle=True)
        power = np.load(rec["power_grid_npz"], allow_pickle=True)
        grid_x = npz["X_grid"]
        grid_y = npz["Y_grid"]
        mask = npz["valid_mask"].astype(bool)
        y = npz["Y"]
        power_grid = power["power_grid"] / 1e10
        titles = [
            f"{rec['sample_id']} {rec['case_name']}\npower density",
            r"$\varepsilon_{\mathrm{eq,total},a}$",
            r"$\sigma_{\mathrm{mean}}$ (MPa)",
        ]
        data = [power_grid, np.where(mask, y[0], np.nan), np.where(mask, y[1], np.nan)]
        cmaps = ["viridis", "magma", "cividis"]
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                data[col_idx],
                origin="lower",
                extent=[float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())],
                cmap=cmaps[col_idx],
                aspect="equal",
            )
            ax.set_title(titles[col_idx], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
    suffix = f"first{len(shown)}" if overview_limit > 0 else "all"
    out = FIG_DIR / f"jian_interpolation_overview_{suffix}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Jian-geometry 256x256 U-Net labels from COMSOL solder-cycle TSV fields."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DENSE_MANIFEST,
        help="Case manifest CSV. Use the dense-batch manifest for formal batch label generation.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Directory containing the COMSOL solder-cycle TSV exports.",
    )
    parser.add_argument(
        "--prefer-dense",
        action="store_true",
        help="Prefer dense64x64_z3 solder-cycle TSV files when available.",
    )
    parser.add_argument(
        "--overview-limit",
        type=int,
        default=5,
        help="Number of cases to include in the overview figure. Use 0 for all cases.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=LABEL_DIR,
        help="Directory for output unet_label_XXXX_256x256.npz files.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for interpolation preview figures.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory for summary CSV/Markdown reports.",
    )
    return parser.parse_args()


def main() -> None:
    global LABEL_DIR, FIG_DIR, REPORT_DIR, RESULT_ROOT
    args = parse_args()
    LABEL_DIR = args.label_dir
    FIG_DIR = args.fig_dir
    REPORT_DIR = args.report_dir
    RESULT_ROOT = args.result_root

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    records = []
    for sample_id in sorted(manifest):
        case_name = manifest[sample_id]["case_name"]
        records.append(save_label(sample_id, case_name, manifest[sample_id], args.prefer_dense))

    overview_path = make_overview(records, args.overview_limit)

    index_path = LABEL_DIR / "jian_unet_label_index.json"
    index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = REPORT_DIR / "jian_unet_label_interpolation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    md_path = REPORT_DIR / "jian_unet_label_interpolation_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Jian U-Net label interpolation summary\n\n")
        f.write("Each COMSOL solder-cycle TSV was projected onto the corresponding 256 x 256 Jian power-grid coordinates. The six solder islands are interpolated separately to avoid nonphysical bridges between disconnected solder regions. Island-wise linear interpolation defines `valid_mask`; nearest-neighbor fill is stored only inside each island region for numerical completeness.\n\n")
        f.write("| sample | case | raw rows | xy points | valid pixels | valid ratio | epsilon range | sigma_mean range |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for rec in records:
            f.write(
                f"| {rec['sample_id']} | `{rec['case_name']}` | {rec['raw_rows']} | "
                f"{rec['combined_xy_points']} | {rec['valid_pixels']} | {rec['valid_ratio']:.3f} | "
                f"{rec['epsilon_min_valid']:.7f}--{rec['epsilon_max_valid']:.7f} | "
                f"{rec['sigma_mean_min_valid_MPa']:.2f}--{rec['sigma_mean_max_valid_MPa']:.2f} MPa |\n"
            )
        f.write("\n## Figures\n\n")
        if overview_path is not None:
            f.write(f"- Overview: `{overview_path}`\n")
        for rec in records:
            f.write(f"- `{rec['sample_id']}`: `{rec['figure']}`\n")

    print(f"Wrote labels: {LABEL_DIR}")
    print(f"Wrote figures: {FIG_DIR}")
    print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
