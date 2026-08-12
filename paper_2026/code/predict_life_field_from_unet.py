#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict fatigue-variable fields with U-Net and convert them to life fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compute_life_field_from_npz import solve_cmb_morrow_bisection
from jian_dataset import JianFatigueDataset, discover_samples, input_channels
from unet import UNetPowerToFatigue


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "paper_2026" / "artifacts" / "unet_7ch_ema"
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "life_field_predictions"


def load_model(run_dir: Path, config: dict, device: torch.device) -> UNetPowerToFatigue:
    model = UNetPowerToFatigue(
        in_channels=input_channels(config["input_mode"]),
        out_channels=len(config["channels"]),
        base_ch=int(config["base_ch"]),
        use_attention=not bool(config.get("no_attention", False)),
        dropout=not bool(config.get("no_dropout", False)),
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_fields(
    model: UNetPowerToFatigue,
    dataset: JianFatigueDataset,
    item: int,
    stats: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    x, _y_norm, y_real, mask, sample_id, layout = dataset[item]
    with torch.no_grad():
        pred_norm = model(x[None].to(device)).cpu().numpy()[0]
    mean = np.asarray(stats["target"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["target"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    pred_real = pred_norm * std + mean
    return pred_real, y_real.numpy(), mask.numpy()[0].astype(bool), int(sample_id), str(layout)


def plot_prediction_life(
    out_png: Path,
    target_life: np.ndarray,
    pred_life: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
) -> None:
    target_log = np.log10(target_life)
    pred_log = np.log10(pred_life)
    abs_log_err = np.abs(pred_log - target_log)
    rel_err = np.abs(pred_life - target_life) / np.maximum(target_life, 1e-30) * 100.0
    arrays = [
        (target_log, r"Target $\log_{10}(N_f)$", "magma_r"),
        (pred_log, r"Predicted $\log_{10}(N_f)$", "magma_r"),
        (abs_log_err, r"$|\Delta \log_{10}(N_f)|$", "viridis"),
        (rel_err, r"$N_f$ relative error (%)", "viridis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True)
    for ax, (arr, name, cmap) in zip(axes, arrays):
        shown = np.where(valid_mask, arr, np.nan)
        im = ax.imshow(shown, origin="lower", cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", nargs="*", default=["10052", "16152", "13353", "14276", "15044"])
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-limit", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epsilon-f-prime", type=float, default=0.325)
    parser.add_argument("--sigma-f-prime-mpa", type=float, default=263.535316568)
    parser.add_argument("--b", type=float, default=-0.1443)
    parser.add_argument("--c", type=float, default=-0.57)
    parser.add_argument("--youngs-modulus-mpa", type=float, default=31200.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    stats = json.loads((args.run_dir / "normalization.json").read_text(encoding="utf-8"))
    samples = discover_samples(args.label_dir)
    sample_to_item = {str(sample.sample_id): i for i, sample in enumerate(samples)}
    if args.split:
        if args.split == "all":
            selected_ids = [str(sample.sample_id) for sample in samples]
        else:
            split_payload = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))
            selected_ids = [str(x) for x in split_payload[args.split]["sample_ids"]]
    else:
        selected_ids = [str(s) for s in args.samples]
    if args.max_samples is not None:
        selected_ids = selected_ids[: args.max_samples]
    selected_items = [sample_to_item[str(s)] for s in selected_ids]
    dataset = JianFatigueDataset(
        samples,
        list(range(len(samples))),
        stats,
        input_mode=config["input_mode"],
        channels=config["channels"],
    )
    device = torch.device(args.device)
    model = load_model(args.run_dir, config, device)

    rows = []
    for out_idx, item in enumerate(selected_items):
        pred, target, mask, sample_id, layout = predict_fields(model, dataset, item, stats, device)
        target_nf = solve_cmb_morrow_bisection(
            target[0],
            target[1],
            mask,
            epsilon_f_prime=args.epsilon_f_prime,
            sigma_f_prime_mpa=args.sigma_f_prime_mpa,
            b=args.b,
            c=args.c,
            youngs_modulus_mpa=args.youngs_modulus_mpa,
        )
        pred_nf = solve_cmb_morrow_bisection(
            np.maximum(pred[0], 1e-12),
            pred[1],
            mask,
            epsilon_f_prime=args.epsilon_f_prime,
            sigma_f_prime_mpa=args.sigma_f_prime_mpa,
            b=args.b,
            c=args.c,
            youngs_modulus_mpa=args.youngs_modulus_mpa,
        )
        finite = mask & np.isfinite(target_nf) & np.isfinite(pred_nf)
        rel_life = np.abs(pred_nf[finite] - target_nf[finite]) / np.maximum(target_nf[finite], 1e-30) * 100.0
        log_mae = np.mean(np.abs(np.log10(pred_nf[finite]) - np.log10(target_nf[finite])))
        eps_rel = np.sum(np.abs(pred[0][finite] - target[0][finite])) / np.sum(np.abs(target[0][finite])) * 100.0
        sig_rel = np.sum(np.abs(pred[1][finite] - target[1][finite])) / np.sum(np.abs(target[1][finite])) * 100.0
        target_min = float(np.nanmin(target_nf))
        pred_min = float(np.nanmin(pred_nf))
        target_arg = np.unravel_index(np.nanargmin(np.where(finite, target_nf, np.nan)), target_nf.shape)
        pred_arg = np.unravel_index(np.nanargmin(np.where(finite, pred_nf, np.nan)), pred_nf.shape)

        stem = f"{sample_id}_{layout}"
        np.savez_compressed(
            args.out_dir / f"predicted_life_field_{stem}.npz",
            target_Nf=target_nf.astype(np.float32),
            pred_Nf=pred_nf.astype(np.float32),
            target_fields=target.astype(np.float32),
            pred_fields=pred.astype(np.float32),
            valid_mask=mask,
            sample_id=sample_id,
            layout=layout,
        )
        if out_idx < args.plot_limit:
            plot_prediction_life(
                args.out_dir / f"predicted_life_field_{stem}.png",
                target_nf,
                pred_nf,
                mask,
                f"sample {sample_id} ({layout})",
            )
        rows.append(
            {
                "sample_id": sample_id,
                "layout": layout,
                "epsilon_rel_mae_%": float(eps_rel),
                "sigma_rel_mae_%": float(sig_rel),
                "log10_Nf_mae": float(log_mae),
                "Nf_rel_mae_%": float(np.mean(rel_life)),
                "target_Nf_min": target_min,
                "pred_Nf_min": pred_min,
                "Nf_min_rel_error_%": abs(pred_min - target_min) / max(target_min, 1e-30) * 100.0,
                "target_argmin_yx": str(tuple(int(v) for v in target_arg)),
                "pred_argmin_yx": str(tuple(int(v) for v in pred_arg)),
            }
        )

    csv_path = args.out_dir / "predicted_life_field_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    layout_rows = []
    for layout in sorted({row["layout"] for row in rows}):
        subset = [row for row in rows if row["layout"] == layout]
        layout_rows.append(
            {
                "layout": layout,
                "count": len(subset),
                "epsilon_rel_mae_mean_%": float(np.mean([r["epsilon_rel_mae_%"] for r in subset])),
                "sigma_rel_mae_mean_%": float(np.mean([r["sigma_rel_mae_%"] for r in subset])),
                "log10_Nf_mae_mean": float(np.mean([r["log10_Nf_mae"] for r in subset])),
                "Nf_rel_mae_mean_%": float(np.mean([r["Nf_rel_mae_%"] for r in subset])),
                "Nf_min_rel_error_mean_%": float(np.mean([r["Nf_min_rel_error_%"] for r in subset])),
            }
        )
    layout_csv = args.out_dir / "predicted_life_field_layout_summary.csv"
    with layout_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layout_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {layout_csv}")
    print("Layout summary:")
    for row in layout_rows:
        print(
            f"{row['layout']}: n={row['count']}, "
            f"field rel=({row['epsilon_rel_mae_mean_%']:.3f}%, {row['sigma_rel_mae_mean_%']:.3f}%), "
            f"logNf MAE={row['log10_Nf_mae_mean']:.5f}, "
            f"Nf rel={row['Nf_rel_mae_mean_%']:.3f}%, "
            f"Nf_min rel={row['Nf_min_rel_error_mean_%']:.3f}%"
        )
    for row in rows:
        print(
            f"{row['sample_id']} {row['layout']}: "
            f"field rel=({row['epsilon_rel_mae_%']:.3f}%, {row['sigma_rel_mae_%']:.3f}%), "
            f"logNf MAE={row['log10_Nf_mae']:.5f}, "
            f"Nf rel={row['Nf_rel_mae_%']:.3f}%, "
            f"Nf_min target/pred={row['target_Nf_min']:.4g}/{row['pred_Nf_min']:.4g}"
        )


if __name__ == "__main__":
    main()
