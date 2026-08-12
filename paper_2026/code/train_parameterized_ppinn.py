#!/usr/bin/env python3
"""Train the six-input parameterized P-PINN described in the manuscript.

Inputs are [epsilon_a, sigma_mean, epsilon_f_prime, sigma_f_prime, c, E].
The Basquin exponent b is fixed.  Training uses only the relative residual of
the Morrow-corrected CMB equation; bisection is called after training solely
to report held-out accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "paper_2026" / "artifacts" / "parameterized_ppinn"
FEATURE_NAMES = (
    "epsilon_a",
    "sigma_mean_MPa",
    "epsilon_f_prime",
    "sigma_f_prime_MPa",
    "c",
    "E_MPa",
)


@dataclass(frozen=True)
class ParameterRanges:
    epsilon_a: tuple[float, float] = (1.0e-4, 4.0e-3)
    sigma_mean_mpa: tuple[float, float] = (0.0, 140.0)
    epsilon_f_prime: tuple[float, float] = (0.25, 0.55)
    sigma_f_prime_mpa: tuple[float, float] = (800.0, 2000.0)
    c: tuple[float, float] = (-0.8, -0.4)
    youngs_modulus_mpa: tuple[float, float] = (30000.0, 80000.0)
    b: float = -0.12


class ParameterizedLifePINN(nn.Module):
    def __init__(self, input_dim: int = 6) -> None:
        super().__init__()
        dims = (input_dim, 64, 64, 64, 32, 1)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.Tanh()))
        output = nn.Linear(dims[-2], dims[-1])
        nn.init.constant_(output.bias, 4.5)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_parameters(
    count: int,
    ranges: ParameterRanges,
    seed: int,
) -> np.ndarray:
    """Draw parameter combinations whose CMB root lies in [1, 1e15] cycles.

    The bracket check uses only the governing equation at the two endpoints;
    it does not calculate numerical lifetime labels for P-PINN training.
    """
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    while accepted_count < count:
        draw_count = max(256, count - accepted_count)
        columns = (
            rng.uniform(*ranges.epsilon_a, draw_count),
            rng.uniform(*ranges.sigma_mean_mpa, draw_count),
            rng.uniform(*ranges.epsilon_f_prime, draw_count),
            rng.uniform(*ranges.sigma_f_prime_mpa, draw_count),
            rng.uniform(*ranges.c, draw_count),
            rng.uniform(*ranges.youngs_modulus_mpa, draw_count),
        )
        values = np.column_stack(columns)
        valid = bracketed_root_mask(values, ranges.b)
        accepted.append(values[valid])
        accepted_count += int(valid.sum())
    return np.concatenate(accepted, axis=0)[:count].astype(np.float32)


def bracketed_root_mask(
    parameters: np.ndarray,
    b: float,
    lower_nf: float = 1.0,
    upper_nf: float = 1.0e15,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    epsilon_a, sigma_mean, epsilon_f_prime, sigma_f_prime, c, youngs_modulus = values.T

    def residual(nf: float) -> np.ndarray:
        modeled = epsilon_f_prime * np.power(2.0 * nf, c)
        modeled += ((sigma_f_prime - sigma_mean) / youngs_modulus) * np.power(2.0 * nf, b)
        return epsilon_a - modeled

    lower_residual = residual(lower_nf)
    upper_residual = residual(upper_nf)
    return (
        np.isfinite(lower_residual)
        & np.isfinite(upper_residual)
        & (lower_residual * upper_residual <= 0)
    )


def relative_physics_loss(
    model: nn.Module,
    normalized: torch.Tensor,
    physical: torch.Tensor,
    b: float,
) -> torch.Tensor:
    log10_nf = model(normalized)
    nf = torch.pow(10.0, log10_nf)
    epsilon_a = physical[:, 0:1]
    sigma_mean = physical[:, 1:2]
    epsilon_f_prime = physical[:, 2:3]
    sigma_f_prime = physical[:, 3:4]
    c = physical[:, 4:5]
    youngs_modulus = physical[:, 5:6]
    modeled = epsilon_f_prime * torch.pow(2.0 * nf, c)
    modeled = modeled + ((sigma_f_prime - sigma_mean) / youngs_modulus) * torch.pow(
        2.0 * nf, b
    )
    residual = (epsilon_a - modeled) / epsilon_a.clamp_min(1.0e-12)
    return torch.mean(residual.square())


def solve_bisection(parameters: np.ndarray, b: float, iterations: int = 100) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    epsilon_a, sigma_mean, epsilon_f_prime, sigma_f_prime, c, youngs_modulus = values.T
    lower = np.ones(values.shape[0], dtype=np.float64)
    upper = np.full(values.shape[0], 1.0e15, dtype=np.float64)

    def residual(nf: np.ndarray) -> np.ndarray:
        modeled = epsilon_f_prime * np.power(2.0 * nf, c)
        modeled += ((sigma_f_prime - sigma_mean) / youngs_modulus) * np.power(2.0 * nf, b)
        return epsilon_a - modeled

    lower_residual = residual(lower)
    upper_residual = residual(upper)
    valid = bracketed_root_mask(values, b)
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = residual(midpoint)
        left = valid & np.isfinite(midpoint_residual) & (lower_residual * midpoint_residual < 0)
        upper[left] = midpoint[left]
        lower[valid & ~left] = midpoint[valid & ~left]
        lower_residual[valid & ~left] = midpoint_residual[valid & ~left]
    result = 0.5 * (lower + upper)
    result[~valid] = np.nan
    return result


def accuracy_metrics(predicted_log10: np.ndarray, reference_nf: np.ndarray) -> dict[str, float]:
    predicted_nf = np.power(10.0, np.asarray(predicted_log10, dtype=np.float64))
    valid = np.isfinite(predicted_nf) & np.isfinite(reference_nf) & (reference_nf > 0)
    relative = np.abs(predicted_nf[valid] - reference_nf[valid]) / reference_nf[valid]
    log_error = np.abs(np.log10(predicted_nf[valid]) - np.log10(reference_nf[valid]))
    return {
        "valid_count": int(valid.sum()),
        "mean_relative_error_percent": float(100.0 * relative.mean()),
        "median_relative_error_percent": float(100.0 * np.median(relative)),
        "p95_relative_error_percent": float(100.0 * np.percentile(relative, 95)),
        "log10_mae": float(log_error.mean()),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    ranges = ParameterRanges()
    values = sample_parameters(args.samples, ranges, args.seed)
    permutation = np.random.default_rng(args.seed).permutation(len(values))
    values = values[permutation]
    split = int(round(len(values) * args.train_fraction))
    train_values = values[:split]
    test_values = values[split:]
    mean = train_values.mean(axis=0).astype(np.float32)
    std = np.maximum(train_values.std(axis=0), 1.0e-12).astype(np.float32)

    device = torch.device(args.device)
    train_physical = torch.as_tensor(train_values, device=device)
    train_normalized = torch.as_tensor((train_values - mean) / std, device=device)
    model = ParameterizedLifePINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_values), device=device)
        total = 0.0
        for start_index in range(0, len(order), args.batch_size):
            indices = order[start_index : start_index + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = relative_physics_loss(
                model, train_normalized[indices], train_physical[indices], ranges.b
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
        epoch_loss = total / len(train_values)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            history.append({"epoch": epoch, "physics_loss": epoch_loss})
            print(f"epoch {epoch}/{args.epochs}: physics_loss={epoch_loss:.6e}", flush=True)
    train_time = time.perf_counter() - start

    model.eval()
    test_normalized = torch.as_tensor((test_values - mean) / std, device=device)
    with torch.no_grad():
        predicted_log10 = model(test_normalized).cpu().numpy().reshape(-1)
    reference_nf = solve_bisection(test_values, ranges.b)
    metrics = accuracy_metrics(predicted_log10, reference_nf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "parameterized_ppinn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "normalization_mean": mean,
            "normalization_std": std,
            "parameter_ranges": asdict(ranges),
            "fixed_b": ranges.b,
        },
        checkpoint,
    )
    summary = {
        "model": "six-input parameterized pure-physics P-PINN",
        "feature_names": FEATURE_NAMES,
        "parameter_ranges": asdict(ranges),
        "samples": args.samples,
        "train_samples": len(train_values),
        "test_samples": len(test_values),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "train_time_seconds": train_time,
        "training_target": "relative CMB/Morrow residual only",
        "sampling_filter": "CMB residual changes sign on Nf=[1, 1e15]; rejected draws are resampled without numerical lifetime labels",
        "held_out_reference": "vectorized bisection used for evaluation only",
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override to 256 samples and two epochs for a fast plumbing check.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.samples = 256
        args.epochs = 2
        args.batch_size = 64
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    return args


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
