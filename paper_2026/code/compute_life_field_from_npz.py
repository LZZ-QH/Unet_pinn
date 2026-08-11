#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Jian U-Net fatigue-variable labels into CMB/Morrow life fields.

This script is intentionally independent of the U-Net model.  It converts an
existing 256 x 256 label file containing epsilon_eq_total_a and sigma_mean_MPa
into a reference life field by solving the Morrow-corrected CMB equation with
vectorized bisection on every valid solder-layer pixel.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = DEFAULT_ROOT / "paper_2026" / "data" / "main2500" / "labels"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "paper_2026" / "artifacts" / "life_field_reference"


def solve_cmb_morrow_bisection(
    epsilon_a: np.ndarray,
    sigma_mean_mpa: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon_f_prime: float,
    sigma_f_prime_mpa: float,
    b: float,
    c: float,
    youngs_modulus_mpa: float,
    low: float = 1.0,
    high: float = 1.0e15,
    iterations: int = 120,
) -> np.ndarray:
    """Solve epsilon_a = eps_f'(2N)^c + ((sig_f'-sig_m)/E)(2N)^b."""

    eps = np.asarray(epsilon_a, dtype=np.float64)
    sig_m = np.asarray(sigma_mean_mpa, dtype=np.float64)
    mask = valid_mask.astype(bool) & np.isfinite(eps) & np.isfinite(sig_m) & (eps > 0)

    lo = np.full(eps.shape, low, dtype=np.float64)
    hi = np.full(eps.shape, high, dtype=np.float64)

    # Pixels outside the valid mask remain NaN.  The current parameter ranges
    # keep sigma_f' - sigma_mean positive for the Jian solder fields.
    def residual(nf: np.ndarray) -> np.ndarray:
        elastic = ((sigma_f_prime_mpa - sig_m) / youngs_modulus_mpa) * np.power(2.0 * nf, b)
        plastic = epsilon_f_prime * np.power(2.0 * nf, c)
        return plastic + elastic - eps

    # If high is not sufficient for unusually small epsilon, expand once.
    r_hi = residual(hi)
    if np.any(mask & (r_hi > 0)):
        hi[mask & (r_hi > 0)] = 1.0e20

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)
        go_right = r_mid > 0
        lo = np.where(mask & go_right, mid, lo)
        hi = np.where(mask & ~go_right, mid, hi)

    nf = 0.5 * (lo + hi)
    nf[~mask] = np.nan
    return nf


def load_fields(label_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    z = np.load(label_file, allow_pickle=True)
    names = [str(x) for x in z["channel_names"]]
    eps_idx = names.index("epsilon_eq_total_a")
    sig_idx = names.index("sigma_mean_MPa")
    return z["Y"][eps_idx], z["Y"][sig_idx], z["valid_mask"].astype(bool), names


def plot_life_overview(
    out_png: Path,
    epsilon_a: np.ndarray,
    sigma_mean: np.ndarray,
    log_life: np.ndarray,
    valid_mask: np.ndarray,
    sample_name: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    items = [
        (epsilon_a, r"$\varepsilon_{\mathrm{eq,total},a}$", "strain", "viridis"),
        (sigma_mean, r"$\sigma_{\mathrm{mean}}$", "MPa", "viridis"),
        (log_life, r"$\log_{10}(N_f)$", "cycles", "magma_r"),
    ]
    for ax, (arr, title, cbar_label, cmap) in zip(axes, items):
        shown = np.where(valid_mask, arr, np.nan)
        im = ax.imshow(shown, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(sample_name, fontsize=11)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def process_file(label_file: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, float | str]:
    eps, sig, mask, _names = load_fields(label_file)
    nf = solve_cmb_morrow_bisection(
        eps,
        sig,
        mask,
        epsilon_f_prime=args.epsilon_f_prime,
        sigma_f_prime_mpa=args.sigma_f_prime_mpa,
        b=args.b,
        c=args.c,
        youngs_modulus_mpa=args.youngs_modulus_mpa,
        iterations=args.iterations,
    )
    log_nf = np.log10(nf)
    sample_stem = label_file.stem.replace("unet_label_", "").replace("_256x256", "")
    npz_out = out_dir / f"life_field_{sample_stem}.npz"
    png_out = out_dir / f"life_field_{sample_stem}.png"
    np.savez_compressed(
        npz_out,
        Nf=nf.astype(np.float32),
        log10_Nf=log_nf.astype(np.float32),
        epsilon_eq_total_a=eps.astype(np.float32),
        sigma_mean_MPa=sig.astype(np.float32),
        valid_mask=mask,
        source_label=str(label_file),
        epsilon_f_prime=args.epsilon_f_prime,
        sigma_f_prime_mpa=args.sigma_f_prime_mpa,
        b=args.b,
        c=args.c,
        youngs_modulus_mpa=args.youngs_modulus_mpa,
    )
    plot_life_overview(png_out, eps, sig, log_nf, mask, label_file.name)
    finite = np.isfinite(nf) & mask
    min_idx = np.nanargmin(np.where(finite, nf, np.nan))
    min_y, min_x = np.unravel_index(min_idx, nf.shape)
    return {
        "sample": sample_stem,
        "source_label": str(label_file),
        "life_npz": str(npz_out),
        "life_png": str(png_out),
        "valid_pixels": int(finite.sum()),
        "Nf_min": float(np.nanmin(nf)),
        "Nf_median": float(np.nanmedian(nf[finite])),
        "Nf_mean": float(np.nanmean(nf[finite])),
        "log10_Nf_min": float(np.log10(np.nanmin(nf))),
        "argmin_x_pixel": int(min_x),
        "argmin_y_pixel": int(min_y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", nargs="*", default=["10052", "16152", "13353", "14276", "15044"])
    parser.add_argument("--epsilon-f-prime", type=float, default=0.325)
    parser.add_argument("--sigma-f-prime-mpa", type=float, default=263.535316568)
    parser.add_argument("--b", type=float, default=-0.1443)
    parser.add_argument("--c", type=float, default=-0.57)
    parser.add_argument("--youngs-modulus-mpa", type=float, default=31200.0)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in args.samples:
        label_file = args.label_dir / f"unet_label_{sample}_256x256.npz"
        if not label_file.exists():
            raise FileNotFoundError(label_file)
        rows.append(process_file(label_file, args.out_dir, args))

    summary_csv = args.out_dir / "life_field_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_csv}")
    for row in rows:
        print(
            f"{row['sample']}: Nf_min={row['Nf_min']:.6g}, "
            f"log10_min={row['log10_Nf_min']:.4f}, "
            f"argmin=({row['argmin_x_pixel']},{row['argmin_y_pixel']})"
        )


if __name__ == "__main__":
    main()
