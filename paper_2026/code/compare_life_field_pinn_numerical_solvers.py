#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare P-PINN and numerical solvers for Jian CMB/Morrow life fields.

The benchmark uses fatigue-variable fields

    epsilon_eq_total_a(x, y), sigma_mean_MPa(x, y)

and maps every valid solder-layer pixel to Nf through the calibrated
CMB/Morrow equation.  Strict bisection is used as the reference solution.
The P-PINN is trained only by the equation residual, while numerical solvers
are evaluated both under strict and relaxed convergence settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import brentq, least_squares


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"
DEFAULT_RUN_DIR = ROOT / "paper_2026" / "artifacts" / "unet_7ch_ema"
DEFAULT_OUT_DIR = ROOT / "paper_2026" / "artifacts" / "life_solver_benchmark"


@dataclass
class JianCMBConfig:
    epsilon_f_prime: float = 0.325
    sigma_f_prime_mpa: float = 263.535316568
    b: float = -0.1443
    c: float = -0.57
    youngs_modulus_mpa: float = 31200.0

    epsilon_a_range: tuple[float, float] = (1.0e-4, 4.0e-3)
    sigma_mean_range: tuple[float, float] = (0.0, 140.0)
    train_samples: int = 40000
    epochs: int = 2500
    batch_size: int = 1024
    lr: float = 1.0e-3
    seed: int = 42


class LifePINN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, ...] = (64, 64, 64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        final = nn.Linear(in_dim, 1)
        nn.init.constant_(final.bias, 4.5)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def residual_np(nf: np.ndarray, eps: np.ndarray, sig: np.ndarray, cfg: JianCMBConfig) -> np.ndarray:
    term_plastic = cfg.epsilon_f_prime * np.power(2.0 * nf, cfg.c)
    term_elastic = ((cfg.sigma_f_prime_mpa - sig) / cfg.youngs_modulus_mpa) * np.power(2.0 * nf, cfg.b)
    return eps - (term_plastic + term_elastic)


def solve_bisection_vectorized(
    eps: np.ndarray,
    sig: np.ndarray,
    cfg: JianCMBConfig,
    *,
    iterations: int = 120,
) -> np.ndarray:
    eps = np.asarray(eps, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    lo = np.ones_like(eps)
    hi = np.full_like(eps, 1.0e15)
    f_lo = residual_np(lo, eps, sig, cfg)
    f_hi = residual_np(hi, eps, sig, cfg)
    valid = np.isfinite(f_lo) & np.isfinite(f_hi) & (f_lo * f_hi <= 0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = residual_np(mid, eps, sig, cfg)
        left = valid & np.isfinite(f_mid) & (f_lo * f_mid < 0)
        right = valid & ~left
        hi[left] = mid[left]
        f_hi[left] = f_mid[left]
        lo[right] = mid[right]
        f_lo[right] = f_mid[right]
    out = 0.5 * (lo + hi)
    out[~valid] = np.nan
    return out


def solve_bisection_scalar(eps: float, sig: float, cfg: JianCMBConfig, *, iterations: int = 120) -> float:
    lo, hi = 1.0, 1.0e15
    f_lo = float(residual_np(np.array([lo]), np.array([eps]), np.array([sig]), cfg)[0])
    f_hi = float(residual_np(np.array([hi]), np.array([eps]), np.array([sig]), cfg)[0])
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return np.nan
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = float(residual_np(np.array([mid]), np.array([eps]), np.array([sig]), cfg)[0])
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def solve_log_bisection_scalar(eps: float, sig: float, cfg: JianCMBConfig, *, iterations: int = 15) -> float:
    lo, hi = 0.0, 15.0
    nf_lo = 10.0**lo
    nf_hi = 10.0**hi
    f_lo = float(residual_np(np.array([nf_lo]), np.array([eps]), np.array([sig]), cfg)[0])
    f_hi = float(residual_np(np.array([nf_hi]), np.array([eps]), np.array([sig]), cfg)[0])
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return np.nan
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        nf_mid = 10.0**mid
        f_mid = float(residual_np(np.array([nf_mid]), np.array([eps]), np.array([sig]), cfg)[0])
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 10.0 ** (0.5 * (lo + hi))


def solve_brent_strict(eps: float, sig: float, cfg: JianCMBConfig) -> float:
    def f(nf: float) -> float:
        return float(residual_np(np.array([nf]), np.array([eps]), np.array([sig]), cfg)[0])

    try:
        return float(brentq(f, 1.0, 1.0e15, xtol=1.0e-8, rtol=1.0e-12, maxiter=100))
    except Exception:
        return np.nan


def solve_brent_relaxed(eps: float, sig: float, cfg: JianCMBConfig) -> float:
    def f(nf: float) -> float:
        return float(residual_np(np.array([nf]), np.array([eps]), np.array([sig]), cfg)[0])

    try:
        return float(brentq(f, 1.0, 1.0e15, xtol=6.0e8, rtol=2.0e-2, maxiter=100))
    except Exception:
        return np.nan


def solve_trust_region_log(
    eps: float,
    sig: float,
    cfg: JianCMBConfig,
    *,
    strict: bool,
) -> float:
    def fun(log_nf_arr: np.ndarray) -> np.ndarray:
        nf = 10.0 ** float(log_nf_arr[0])
        r = float(residual_np(np.array([nf]), np.array([eps]), np.array([sig]), cfg)[0])
        if not np.isfinite(r):
            return np.array([1.0e6], dtype=float)
        return np.array([r], dtype=float)

    try:
        if strict:
            result = least_squares(
                fun,
                x0=np.array([4.5]),
                bounds=(np.array([0.0]), np.array([15.0])),
                method="trf",
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=200,
            )
        else:
            result = least_squares(
                fun,
                x0=np.array([5.0]),
                bounds=(np.array([0.0]), np.array([15.0])),
                method="trf",
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=6,
            )
        return float(10.0 ** result.x[0])
    except Exception:
        return np.nan


def physics_loss(model: LifePINN, x_norm: torch.Tensor, x_phys: torch.Tensor, cfg: JianCMBConfig) -> torch.Tensor:
    log_nf = model(x_norm)
    nf = torch.pow(10.0, log_nf)
    eps = x_phys[:, 0:1]
    sig = x_phys[:, 1:2]
    term_plastic = cfg.epsilon_f_prime * torch.pow(2.0 * nf, cfg.c)
    term_elastic = ((cfg.sigma_f_prime_mpa - sig) / cfg.youngs_modulus_mpa) * torch.pow(2.0 * nf, cfg.b)
    residual = eps - (term_plastic + term_elastic)
    return torch.mean((residual / eps.clamp_min(1.0e-12)) ** 2)


def generate_training_data(cfg: JianCMBConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    eps = rng.uniform(cfg.epsilon_a_range[0], cfg.epsilon_a_range[1], cfg.train_samples)
    sig = rng.uniform(cfg.sigma_mean_range[0], cfg.sigma_mean_range[1], cfg.train_samples)
    x = np.column_stack([eps, sig]).astype(np.float32)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1.0e-12] = 1.0
    return x, x_mean.astype(np.float32), x_std.astype(np.float32)


def train_or_load_pinn(out_dir: Path, cfg: JianCMBConfig, device: torch.device) -> tuple[LifePINN, np.ndarray, np.ndarray, Path]:
    ckpt_path = out_dir / "jian_pure_physics_life_pinn.pt"
    model = LifePINN(input_dim=2).to(device)
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        return model, np.asarray(payload["x_mean"], dtype=np.float32), np.asarray(payload["x_std"], dtype=np.float32), ckpt_path

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    x, x_mean, x_std = generate_training_data(cfg)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    x_phys_t = torch.tensor(x, dtype=torch.float32, device=device)
    x_norm_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(x_norm_t)
    model.train()
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for batch_start in range(0, n, cfg.batch_size):
            idx = perm[batch_start : batch_start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = physics_loss(model, x_norm_t[idx], x_phys_t[idx], cfg)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d}/{cfg.epochs}, physics_loss={epoch_loss / n:.6e}")
    train_time = time.perf_counter() - start
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "config": asdict(cfg),
            "train_time_s": train_time,
        },
        ckpt_path,
    )
    return model, x_mean, x_std, ckpt_path


def load_split_ids(run_dir: Path, split: str, max_samples: int | None) -> list[int]:
    payload = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    ids = [int(x) for x in payload[split]["sample_ids"]]
    return ids[:max_samples] if max_samples is not None else ids


def load_field_pixels(label_dir: Path, sample_ids: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    eps_parts: list[np.ndarray] = []
    sig_parts: list[np.ndarray] = []
    per_sample: list[dict[str, object]] = []
    for sid in sample_ids:
        path = label_dir / f"unet_label_{sid}_256x256.npz"
        with np.load(path, allow_pickle=True) as z:
            names = [str(x) for x in z["channel_names"]]
            eps_idx = names.index("epsilon_eq_total_a")
            sig_idx = names.index("sigma_mean_MPa")
            mask = z["valid_mask"].astype(bool)
            eps = z["Y"][eps_idx][mask].astype(np.float64)
            sig = z["Y"][sig_idx][mask].astype(np.float64)
            case_name = str(z["case_name"]) if "case_name" in z.files else ""
        eps_parts.append(eps)
        sig_parts.append(sig)
        per_sample.append({"sample_id": sid, "case_name": case_name, "valid_pixels": int(mask.sum())})
    eps_all = np.concatenate(eps_parts)
    sig_all = np.concatenate(sig_parts)
    info = {
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "per_sample": per_sample,
        "total_valid_pixels": int(eps_all.size),
        "epsilon_range": [float(eps_all.min()), float(eps_all.max())],
        "sigma_range_MPa": [float(sig_all.min()), float(sig_all.max())],
    }
    return eps_all, sig_all, info


def subsample(eps: np.ndarray, sig: np.ndarray, max_pixels: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_pixels is None or eps.size <= max_pixels:
        return eps, sig
    rng = np.random.default_rng(seed)
    idx = rng.choice(eps.size, size=max_pixels, replace=False)
    return eps[idx], sig[idx]


def predict_pinn(
    model: LifePINN,
    eps: np.ndarray,
    sig: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    outputs: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, eps.size, chunk_size):
        x = np.column_stack([eps[start : start + chunk_size], sig[start : start + chunk_size]]).astype(np.float32)
        x_norm = ((x - x_mean) / x_std).astype(np.float32)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.no_grad():
            pred_log = model(xt).detach().cpu().numpy().reshape(-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - tick
        outputs.append(pred_log)
    return np.concatenate(outputs), elapsed


def metrics_from_prediction(pred_nf: np.ndarray, ref_nf: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred_nf) & np.isfinite(ref_nf) & (pred_nf > 0) & (ref_nf > 0)
    rel = np.abs(pred_nf[valid] - ref_nf[valid]) / ref_nf[valid] * 100.0
    log_err = np.abs(np.log10(pred_nf[valid]) - np.log10(ref_nf[valid]))
    return {
        "valid_count": int(valid.sum()),
        "log10_Nf_MAE": float(np.mean(log_err)) if valid.any() else np.nan,
        "Nf_mean_relative_error_%": float(np.mean(rel)) if valid.any() else np.nan,
        "Nf_median_relative_error_%": float(np.median(rel)) if valid.any() else np.nan,
        "Nf_p95_relative_error_%": float(np.percentile(rel, 95)) if valid.any() else np.nan,
    }


def evaluate_scalar_solver(
    name: str,
    solver: Callable[[float, float, JianCMBConfig], float],
    eps: np.ndarray,
    sig: np.ndarray,
    ref_nf: np.ndarray,
    cfg: JianCMBConfig,
    note: str,
) -> dict[str, object]:
    pred = np.empty_like(ref_nf)
    start = time.perf_counter()
    for i, (e, s) in enumerate(zip(eps, sig)):
        pred[i] = solver(float(e), float(s), cfg)
    elapsed = time.perf_counter() - start
    row: dict[str, object] = {
        "method": name,
        "type": "scalar numerical solver",
        "setting": note,
        "pixels": int(eps.size),
        "runtime_s": float(elapsed),
        "time_per_pixel_us": float(elapsed / max(eps.size, 1) * 1.0e6),
    }
    row.update(metrics_from_prediction(pred, ref_nf))
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-pixels", type=int, default=500000)
    parser.add_argument("--scalar-pixels", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = JianCMBConfig()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, x_mean, x_std, ckpt_path = train_or_load_pinn(args.out_dir, cfg, device)

    sample_ids = load_split_ids(args.run_dir, args.split, args.max_samples)
    eps_all, sig_all, field_info = load_field_pixels(args.label_dir, sample_ids)
    eps_bench, sig_bench = subsample(eps_all, sig_all, args.max_pixels, cfg.seed)
    scalar_count = min(args.scalar_pixels, eps_bench.size)
    eps_scalar, sig_scalar = subsample(eps_bench, sig_bench, scalar_count, cfg.seed + 1)

    start = time.perf_counter()
    ref_nf = solve_bisection_vectorized(eps_bench, sig_bench, cfg, iterations=120)
    vector_bisection_time = time.perf_counter() - start
    start = time.perf_counter()
    ref_nf_scalar = solve_bisection_vectorized(eps_scalar, sig_scalar, cfg, iterations=120)
    scalar_reference_vector_time = time.perf_counter() - start

    pred_log, pinn_time = predict_pinn(model, eps_bench, sig_bench, x_mean, x_std, device, args.chunk_size)
    pred_nf = np.power(10.0, pred_log)

    main_rows: list[dict[str, object]] = []
    pinn_row: dict[str, object] = {
        "method": "pure-physics P-PINN",
        "type": "neural surrogate",
        "setting": "one batched forward pass after residual-only training",
        "pixels": int(eps_bench.size),
        "runtime_s": float(pinn_time),
        "time_per_pixel_us": float(pinn_time / max(eps_bench.size, 1) * 1.0e6),
    }
    pinn_row.update(metrics_from_prediction(pred_nf, ref_nf))
    main_rows.append(pinn_row)

    vec_row: dict[str, object] = {
        "method": "vectorized bisection",
        "type": "strict numerical reference",
        "setting": "120 iterations; vectorized NumPy implementation",
        "pixels": int(eps_bench.size),
        "runtime_s": float(vector_bisection_time),
        "time_per_pixel_us": float(vector_bisection_time / max(eps_bench.size, 1) * 1.0e6),
        "valid_count": int(np.isfinite(ref_nf).sum()),
        "log10_Nf_MAE": 0.0,
        "Nf_mean_relative_error_%": 0.0,
        "Nf_median_relative_error_%": 0.0,
        "Nf_p95_relative_error_%": 0.0,
    }
    main_rows.append(vec_row)

    scalar_rows: list[dict[str, object]] = []
    scalar_solvers: list[tuple[str, Callable[[float, float, JianCMBConfig], float], str]] = [
        ("scalar bisection strict", lambda e, s, c: solve_bisection_scalar(e, s, c, iterations=120), "120 iterations"),
        ("Brent strict", solve_brent_strict, "xtol=1e-8, rtol=1e-12"),
        ("log-domain trust-region strict", lambda e, s, c: solve_trust_region_log(e, s, c, strict=True), "max_nfev=200"),
        ("fixed-log-bisection 13iter", lambda e, s, c: solve_log_bisection_scalar(e, s, c, iterations=13), "13 log-domain iterations"),
        ("log-domain trust-region relaxed", lambda e, s, c: solve_trust_region_log(e, s, c, strict=False), "x0=5.0, max_nfev=6"),
    ]
    for name, solver, note in scalar_solvers:
        print(f"evaluating {name} on {scalar_count} pixels ...")
        scalar_rows.append(evaluate_scalar_solver(name, solver, eps_scalar, sig_scalar, ref_nf_scalar, cfg, note))

    # Put a vectorized reference row for the scalar subset too.
    scalar_rows.insert(
        0,
        {
            "method": "vectorized bisection reference on scalar subset",
            "type": "strict numerical reference",
            "setting": "120 iterations; vectorized NumPy implementation",
            "pixels": int(eps_scalar.size),
            "runtime_s": float(scalar_reference_vector_time),
            "time_per_pixel_us": float(scalar_reference_vector_time / max(eps_scalar.size, 1) * 1.0e6),
            "valid_count": int(np.isfinite(ref_nf_scalar).sum()),
            "log10_Nf_MAE": 0.0,
            "Nf_mean_relative_error_%": 0.0,
            "Nf_median_relative_error_%": 0.0,
            "Nf_p95_relative_error_%": 0.0,
        },
    )

    summary = {
        "benchmark": "Jian calibrated CMB/Morrow life-field solver comparison",
        "config": asdict(cfg),
        "pinn_checkpoint": str(ckpt_path),
        "label_dir": str(args.label_dir),
        "run_dir": str(args.run_dir),
        "split": args.split,
        "field_info": field_info,
        "benchmark_pixels": int(eps_bench.size),
        "scalar_solver_pixels": int(eps_scalar.size),
        "main_rows": main_rows,
        "scalar_rows": scalar_rows,
    }
    (args.out_dir / "solver_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / "solver_comparison_main_metrics.csv", main_rows)
    write_csv(args.out_dir / "solver_comparison_scalar_metrics.csv", scalar_rows)

    md_lines = [
        "# Jian Life-Field Solver Comparison",
        "",
        "This benchmark maps solder-layer fatigue-variable pixels to life values using the calibrated CMB/Morrow equation.",
        "Strict vectorized bisection is used as the reference solution. The P-PINN is trained only with the CMB/Morrow residual, without bisection labels.",
        "",
        "## Data",
        "",
        f"- Split: `{args.split}`",
        f"- Samples loaded: {field_info['sample_count']}",
        f"- Total valid pixels in loaded samples: {field_info['total_valid_pixels']}",
        f"- Benchmark pixels: {eps_bench.size}",
        f"- Scalar solver subset: {eps_scalar.size}",
        f"- Epsilon range: {field_info['epsilon_range']}",
        f"- Sigma range (MPa): {field_info['sigma_range_MPa']}",
        "",
        "## Main Batched Solver Results",
        "",
        "| Method | Pixels | Runtime (s) | Time / pixel (us) | log10(Nf) MAE | Mean relative error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        md_lines.append(
            f"| {row['method']} | {row['pixels']} | {row['runtime_s']:.6f} | "
            f"{row['time_per_pixel_us']:.4f} | {row['log10_Nf_MAE']:.6f} | "
            f"{row['Nf_mean_relative_error_%']:.4f}% |"
        )
    md_lines.extend(
        [
            "",
            "## Scalar Numerical Solver Subset",
            "",
            "| Method | Setting | Pixels | Runtime (s) | Time / pixel (us) | Mean relative error |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in scalar_rows:
        md_lines.append(
            f"| {row['method']} | {row['setting']} | {row['pixels']} | {row['runtime_s']:.6f} | "
            f"{row['time_per_pixel_us']:.4f} | {row['Nf_mean_relative_error_%']:.4f}% |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Strict numerical solvers provide the equation reference solution; the P-PINN is not claimed to be more accurate than a correctly converged solver.",
            "- The P-PINN advantage is the trained, batched forward mapping from fatigue-variable fields to life fields.",
            "- Relaxed numerical settings show the runtime/error trade-off when iterative solvers are stopped at a comparable engineering accuracy level.",
        ]
    )
    (args.out_dir / "solver_comparison_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
