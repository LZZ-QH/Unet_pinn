#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a data-only MLP and compare it with pure-physics P-PINN on Jian 2M pixels.

This script aligns the loss-ablation experiment with the current Jian life-field
benchmark.  The data-only MLP uses the same two CMB/Morrow inputs and the same
network architecture as the pure-physics P-PINN, but it is trained on bisection
log10(Nf) labels instead of the physics residual.
"""

from __future__ import annotations

import csv
import json
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from compare_life_field_pinn_numerical_solvers import (
    DEFAULT_LABEL_DIR,
    DEFAULT_RUN_DIR,
    JianCMBConfig,
    LifePINN,
    generate_training_data,
    load_field_pixels,
    load_split_ids,
    metrics_from_prediction,
    predict_pinn,
    solve_bisection_vectorized,
    subsample,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PURE_PHYSICS_DIR = (
    ROOT
    / "paper_2026"
    / "artifacts"
    / "life_solver_benchmark"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "paper_2026"
    / "artifacts"
    / "life_solver_ablation"
)


def train_or_load_data_only_mlp(
    out_dir: Path,
    cfg: JianCMBConfig,
    train_device: torch.device,
) -> tuple[LifePINN, np.ndarray, np.ndarray, Path, dict[str, object]]:
    ckpt_path = out_dir / "jian_data_only_mlp_life_solver.pt"
    model = LifePINN(input_dim=2).to(train_device)
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=train_device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        return (
            model,
            np.asarray(payload["x_mean"], dtype=np.float32),
            np.asarray(payload["x_std"], dtype=np.float32),
            ckpt_path,
            dict(payload.get("training_summary", {})),
        )

    torch.manual_seed(cfg.seed)
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    x, x_mean, x_std = generate_training_data(cfg)
    start_label = time.perf_counter()
    ref_nf = solve_bisection_vectorized(x[:, 0], x[:, 1], cfg, iterations=120)
    label_time_s = time.perf_counter() - start_label
    y = np.log10(ref_nf).astype(np.float32).reshape(-1, 1)
    valid = np.isfinite(y.reshape(-1))
    x = x[valid]
    y = y[valid]
    x_norm = ((x - x_mean) / x_std).astype(np.float32)

    x_t = torch.tensor(x_norm, dtype=torch.float32, device=train_device)
    y_t = torch.tensor(y, dtype=torch.float32, device=train_device)
    n = len(x_t)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    loss_history: list[dict[str, float]] = []

    model.train()
    start_train = time.perf_counter()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=train_device)
        total_loss = 0.0
        for batch_start in range(0, n, cfg.batch_size):
            idx = perm[batch_start : batch_start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
        mean_loss = total_loss / n
        if epoch == 0 or (epoch + 1) % 250 == 0:
            loss_history.append({"epoch": float(epoch + 1), "train_mse": float(mean_loss)})
            print(f"data-only epoch {epoch + 1:4d}/{cfg.epochs}, train_mse={mean_loss:.6e}", flush=True)
    train_time_s = time.perf_counter() - start_train
    model.eval()

    training_summary = {
        "train_samples_requested": cfg.train_samples,
        "train_samples_valid": int(n),
        "label_generation_time_s": float(label_time_s),
        "train_time_s": float(train_time_s),
        "loss_history": loss_history,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "config": asdict(cfg),
            "training_summary": training_summary,
        },
        ckpt_path,
    )
    return model, x_mean, x_std, ckpt_path, training_summary


def load_pure_physics_model(ckpt_path: Path, device: torch.device) -> tuple[LifePINN, np.ndarray, np.ndarray]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LifePINN(input_dim=2).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return (
        model,
        np.asarray(payload["x_mean"], dtype=np.float32),
        np.asarray(payload["x_std"], dtype=np.float32),
    )


def eval_model_cpu(
    name: str,
    model: LifePINN,
    eps: np.ndarray,
    sig: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    ref_nf: np.ndarray,
    chunk_size: int,
) -> dict[str, object]:
    cpu = torch.device("cpu")
    model_cpu = LifePINN(input_dim=2).to(cpu)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    pred_log, elapsed = predict_pinn(model_cpu, eps, sig, x_mean, x_std, cpu, chunk_size)
    pred_nf = np.power(10.0, pred_log)
    row: dict[str, object] = {
        "method": name,
        "pixels": int(eps.size),
        "runtime_s": float(elapsed),
        "time_per_pixel_us": float(elapsed / max(eps.size, 1) * 1.0e6),
    }
    row.update(metrics_from_prediction(pred_nf, ref_nf))
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
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = JianCMBConfig()
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_chunk_size = 262144
    max_pixels = 2_000_000

    data_model, data_x_mean, data_x_std, data_ckpt, data_train_summary = train_or_load_data_only_mlp(
        out_dir, cfg, train_device
    )

    pure_ckpt = DEFAULT_PURE_PHYSICS_DIR / "jian_pure_physics_life_pinn.pt"
    pure_model, pure_x_mean, pure_x_std = load_pure_physics_model(pure_ckpt, train_device)

    sample_ids = load_split_ids(DEFAULT_RUN_DIR, "test", None)
    eps_all, sig_all, field_info = load_field_pixels(DEFAULT_LABEL_DIR, sample_ids)
    eps_bench, sig_bench = subsample(eps_all, sig_all, max_pixels, cfg.seed)

    start_ref = time.perf_counter()
    ref_nf = solve_bisection_vectorized(eps_bench, sig_bench, cfg, iterations=120)
    ref_runtime = time.perf_counter() - start_ref
    ref_row: dict[str, object] = {
        "method": "vectorized bisection reference",
        "pixels": int(eps_bench.size),
        "runtime_s": float(ref_runtime),
        "time_per_pixel_us": float(ref_runtime / max(eps_bench.size, 1) * 1.0e6),
        "valid_count": int(np.isfinite(ref_nf).sum()),
        "log10_Nf_MAE": 0.0,
        "Nf_mean_relative_error_%": 0.0,
        "Nf_median_relative_error_%": 0.0,
        "Nf_p95_relative_error_%": 0.0,
    }

    rows = [
        eval_model_cpu(
            "data-only MLP",
            data_model,
            eps_bench,
            sig_bench,
            data_x_mean,
            data_x_std,
            ref_nf,
            eval_chunk_size,
        ),
        eval_model_cpu(
            "pure-physics P-PINN",
            pure_model,
            eps_bench,
            sig_bench,
            pure_x_mean,
            pure_x_std,
            ref_nf,
            eval_chunk_size,
        ),
        ref_row,
    ]

    for row in rows:
        if row["method"] != "pure-physics P-PINN":
            continue
        pure_runtime = float(row["runtime_s"])
        for r in rows:
            r["relative_runtime_vs_pure_physics"] = (
                float(r["runtime_s"]) / pure_runtime if pure_runtime > 0 else np.nan
            )

    summary = {
        "benchmark": "Jian current 2M life-field data-only MLP vs pure-physics P-PINN",
        "purpose": "Align the loss-ablation experiment with the current Jian life-field benchmark.",
        "config": asdict(cfg),
        "label_dir": str(DEFAULT_LABEL_DIR),
        "run_dir": str(DEFAULT_RUN_DIR),
        "pure_physics_checkpoint": str(pure_ckpt),
        "data_only_checkpoint": str(data_ckpt),
        "field_info": field_info,
        "benchmark_pixels": int(eps_bench.size),
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "train_device": str(train_device),
            "cpu": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data_only_training": data_train_summary,
        "rows": rows,
    }
    (out_dir / "data_only_vs_pure_physics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "data_only_vs_pure_physics_metrics.csv", rows)

    md_lines = [
        "# Jian 2M Life-Field Loss Ablation: Data-Only MLP vs Pure-Physics P-PINN",
        "",
        "This experiment repeats the neural-solver ablation on the current Jian life-field benchmark.",
        "Both neural models use the same two CMB/Morrow inputs, the same MLP architecture, and the same CPU inference protocol.",
        "",
        "## Setup",
        "",
        f"- Synthetic CMB-input training samples: `{cfg.train_samples}`",
        f"- Training input range: `epsilon_a={cfg.epsilon_a_range}`, `sigma_mean={cfg.sigma_mean_range} MPa`",
        "- Data-only MLP training target: strict bisection `log10(Nf)` labels.",
        "- Pure-physics P-PINN training target: CMB/Morrow residual only, without bisection labels.",
        f"- Evaluation split: Jian U-Net `{DEFAULT_RUN_DIR.name}` test split.",
        f"- Benchmark pixels: `{eps_bench.size:,}`",
        "- Runtime shown below is CPU inference on the same machine/protocol; GPU is only used to accelerate data-only training.",
        "",
        "## Results",
        "",
        "| Method | Training target | Pixels | CPU runtime | Time / pixel | Mean relative error vs bisection | Relative runtime vs P-PINN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    target_map = {
        "data-only MLP": "supervised bisection log10(Nf)",
        "pure-physics P-PINN": "CMB/Morrow residual only",
        "vectorized bisection reference": "strict numerical reference",
    }
    for row in rows:
        err = (
            "reference"
            if row["method"] == "vectorized bisection reference"
            else f"{row['Nf_mean_relative_error_%']:.4f}%"
        )
        rel = f"{row['relative_runtime_vs_pure_physics']:.1f}x"
        md_lines.append(
            f"| {row['method']} | {target_map[row['method']]} | {row['pixels']:,} | "
            f"{row['runtime_s']:.6f} s | {row['time_per_pixel_us']:.4f} us/pixel | {err} | {rel} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The data-only MLP tests whether a supervised neural surrogate can learn the CMB/Morrow solver when strict bisection labels are provided.",
            "- The pure-physics P-PINN tests whether the same mapping can be learned from the equation residual alone.",
            "- This table should not be mixed with the older tail100 direct-field scalar ablation, because the present benchmark uses the current Jian 2M life-field pixel distribution.",
        ]
    )
    (out_dir / "data_only_vs_pure_physics_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
