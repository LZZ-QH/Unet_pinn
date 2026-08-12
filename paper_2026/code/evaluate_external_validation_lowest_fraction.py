#!/usr/bin/env python3
"""Reproduce the calibrated Jian et al. external-validation table.

The fatigue coefficient is calibrated once on the COMSOL no-vibration field
so that the mean of the lowest-life fraction equals 20,000 cycles.  That same
coefficient is then applied to the U-Net fields for the 10 Hz and 30 Hz cases.
No vibration-assisted lifetime is used during calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from compute_life_field_from_npz import solve_cmb_morrow_bisection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "paper_2026" / "sample_data" / "external_validation"
DEFAULT_OUT_DIR = ROOT / "paper_2026" / "artifacts" / "external_validation"


@dataclass(frozen=True)
class CMBParameters:
    epsilon_f_prime: float = 0.325
    b: float = -0.1443
    c: float = -0.57
    youngs_modulus_mpa: float = 31200.0


def lowest_fraction_mean(
    life: np.ndarray,
    mask: np.ndarray,
    fraction: float,
) -> tuple[float, int]:
    """Return the arithmetic mean of the lowest valid life-field values."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    valid = np.asarray(mask, dtype=bool) & np.isfinite(life) & (life > 0)
    values = np.sort(np.asarray(life, dtype=np.float64)[valid])
    if values.size == 0:
        raise ValueError("life field has no finite positive values inside the mask")
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(values[:count].mean()), count


def solve_life(
    epsilon_a: np.ndarray,
    sigma_mean_mpa: np.ndarray,
    mask: np.ndarray,
    sigma_f_prime_mpa: float,
    params: CMBParameters,
) -> np.ndarray:
    return solve_cmb_morrow_bisection(
        np.asarray(epsilon_a, dtype=np.float64),
        np.asarray(sigma_mean_mpa, dtype=np.float64),
        np.asarray(mask, dtype=bool),
        epsilon_f_prime=params.epsilon_f_prime,
        sigma_f_prime_mpa=sigma_f_prime_mpa,
        b=params.b,
        c=params.c,
        youngs_modulus_mpa=params.youngs_modulus_mpa,
        iterations=100,
    )


def calibrate_sigma_f_prime(
    epsilon_a: np.ndarray,
    sigma_mean_mpa: np.ndarray,
    mask: np.ndarray,
    target_cycles: float,
    fraction: float,
    params: CMBParameters,
    lower_mpa: float = 1.0,
    upper_mpa: float = 1000.0,
) -> tuple[float, float, int]:
    """Calibrate sigma_f' with a monotone outer bisection."""

    def objective(value: float) -> tuple[float, int]:
        field = solve_life(epsilon_a, sigma_mean_mpa, mask, value, params)
        return lowest_fraction_mean(field, mask, fraction)

    lower_value, _ = objective(lower_mpa)
    upper_value, _ = objective(upper_mpa)
    if not lower_value <= target_cycles <= upper_value:
        raise ValueError(
            "calibration target is outside the sigma_f' bracket: "
            f"{lower_value:.3f} <= {target_cycles:.3f} <= {upper_value:.3f} is false"
        )

    for _ in range(64):
        midpoint = 0.5 * (lower_mpa + upper_mpa)
        life_value, _ = objective(midpoint)
        if life_value < target_cycles:
            lower_mpa = midpoint
        else:
            upper_mpa = midpoint

    calibrated = 0.5 * (lower_mpa + upper_mpa)
    achieved, count = objective(calibrated)
    return calibrated, achieved, count


def relative_error_percent(prediction: float, reference: float) -> float:
    return float(abs(prediction - reference) / abs(reference) * 100.0)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    params = CMBParameters(
        epsilon_f_prime=args.epsilon_f_prime,
        b=args.b,
        c=args.c,
        youngs_modulus_mpa=args.youngs_modulus_mpa,
    )
    base = np.load(args.data_dir / "no_vibration_fields.npz", allow_pickle=False)
    base_mask = np.asarray(base["valid_mask"], dtype=bool)

    sigma_f_prime, achieved, pixel_count = calibrate_sigma_f_prime(
        base["target_fields"][0],
        base["target_fields"][1],
        base_mask,
        args.no_vibration_cycles,
        args.lowest_fraction,
        params,
    )

    predicted_base_life = solve_life(
        base["pred_fields"][0],
        base["pred_fields"][1],
        base_mask,
        sigma_f_prime,
        params,
    )
    predicted_base_value, _ = lowest_fraction_mean(
        predicted_base_life, base_mask, args.lowest_fraction
    )

    rows: list[dict[str, object]] = [
        {
            "case": "no vibration",
            "experimental_life": args.no_vibration_cycles,
            "predicted_life": args.no_vibration_cycles,
            "relative_error_percent": 0.0,
            "role": "calibration",
        }
    ]
    case_specs = (
        ("vertical 10 Hz, 2 mm", "vertical_10Hz_2mm_fields.npz", args.hz10_cycles),
        ("vertical 30 Hz, 2 mm", "vertical_30Hz_2mm_fields.npz", args.hz30_cycles),
    )
    diagnostic_rows: list[dict[str, object]] = []
    for case_name, filename, experimental_life in case_specs:
        case = np.load(args.data_dir / filename, allow_pickle=False)
        mask = np.asarray(case["valid_mask"], dtype=bool)
        pred_life = solve_life(
            case["pred_eps_total"],
            case["pred_sigma_mean"],
            mask,
            sigma_f_prime,
            params,
        )
        target_life = solve_life(
            case["target_eps_total"],
            case["target_sigma_mean"],
            mask,
            sigma_f_prime,
            params,
        )
        pred_value, pred_count = lowest_fraction_mean(pred_life, mask, args.lowest_fraction)
        target_value, _ = lowest_fraction_mean(target_life, mask, args.lowest_fraction)
        rows.append(
            {
                "case": case_name,
                "experimental_life": experimental_life,
                "predicted_life": pred_value,
                "relative_error_percent": relative_error_percent(pred_value, experimental_life),
                "role": "evaluation",
            }
        )
        diagnostic_rows.append(
            {
                "case": case_name,
                "critical_pixel_count": pred_count,
                "comsol_field_life": target_value,
                "unet_field_life": pred_value,
                "unet_vs_comsol_percent": relative_error_percent(pred_value, target_value),
            }
        )

    return {
        "method": "mean of lowest-life valid pixels",
        "lowest_fraction": args.lowest_fraction,
        "calibration": {
            "field_source": "COMSOL no-vibration target field",
            "target_cycles": args.no_vibration_cycles,
            "achieved_cycles": achieved,
            "critical_pixel_count": pixel_count,
            "sigma_f_prime_mpa": sigma_f_prime,
            "unet_no_vibration_diagnostic_cycles": predicted_base_value,
        },
        "cmb_parameters": asdict(params),
        "table_rows": rows,
        "diagnostics": diagnostic_rows,
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = summary["table_rows"]
    with (out_dir / "external_validation_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lowest-fraction", type=float, default=0.001)
    parser.add_argument("--no-vibration-cycles", type=float, default=20000.0)
    parser.add_argument("--hz10-cycles", type=float, default=17500.0)
    parser.add_argument("--hz30-cycles", type=float, default=14000.0)
    parser.add_argument("--epsilon-f-prime", type=float, default=0.325)
    parser.add_argument("--b", type=float, default=-0.1443)
    parser.add_argument("--c", type=float, default=-0.57)
    parser.add_argument("--youngs-modulus-mpa", type=float, default=31200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    write_outputs(summary, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
