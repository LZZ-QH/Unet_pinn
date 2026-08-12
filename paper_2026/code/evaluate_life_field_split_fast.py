#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast split-level U-Net-to-life-field evaluation for the Jian dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from compute_life_field_from_npz import solve_cmb_morrow_bisection
from jian_dataset import JianFatigueDataset, discover_samples, input_channels
from unet import UNetPowerToFatigue


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "paper_2026" / "artifacts" / "unet_7ch_ema"
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(out_dir: Path, rows: list[dict], layout_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Layout grouped bars.
    layouts = [r["layout"] for r in layout_rows]
    x = np.arange(len(layouts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    ax.bar(x - width / 2, [r["field_rel_mean_%"] for r in layout_rows], width, label="field rel. MAE")
    ax.bar(x + width / 2, [r["Nf_rel_mae_mean_%"] for r in layout_rows], width, label=r"$N_f$ field rel. MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Life-field propagation error by layout")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "life_field_error_by_layout.png", dpi=220)
    plt.close(fig)

    # Worst sample ranking.
    top = sorted(rows, key=lambda r: r["Nf_rel_mae_%"], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    labels = [f"{r['sample_id']}\\n{r['layout']}" for r in top]
    ax.bar(np.arange(len(top)), [r["Nf_rel_mae_%"] for r in top], color="#4C78A8")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$N_f$ field relative MAE (%)")
    ax.set_title("Worst 20 test samples after U-Net-to-life propagation")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "life_field_worst20_ranking.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epsilon-f-prime", type=float, default=0.325)
    parser.add_argument("--sigma-f-prime-mpa", type=float, default=263.535316568)
    parser.add_argument("--b", type=float, default=-0.1443)
    parser.add_argument("--c", type=float, default=-0.57)
    parser.add_argument("--youngs-modulus-mpa", type=float, default=31200.0)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / f"life_field_{args.split}_fast")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    stats = json.loads((args.run_dir / "normalization.json").read_text(encoding="utf-8"))
    split_payload = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))
    samples = discover_samples(args.label_dir)
    sample_to_item = {sample.sample_id: i for i, sample in enumerate(samples)}
    if args.split == "all":
        indices = list(range(len(samples)))
    else:
        indices = [sample_to_item[int(sid)] for sid in split_payload[args.split]["sample_ids"]]

    dataset = JianFatigueDataset(
        samples,
        indices,
        stats,
        input_mode=config["input_mode"],
        channels=config["channels"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    model = load_model(args.run_dir, config, device)
    mean = torch.as_tensor(stats["target"]["mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(stats["target"]["std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)

    rows: list[dict] = []
    for x, _y_norm, y_real, mask, sample_ids, layouts in loader:
        x = x.to(device)
        with torch.no_grad():
            pred = (model(x) * std + mean).cpu().numpy()
        target = y_real.numpy()
        mask_np = mask.numpy()[:, 0].astype(bool)
        target_nf = solve_cmb_morrow_bisection(
            target[:, 0],
            target[:, 1],
            mask_np,
            epsilon_f_prime=args.epsilon_f_prime,
            sigma_f_prime_mpa=args.sigma_f_prime_mpa,
            b=args.b,
            c=args.c,
            youngs_modulus_mpa=args.youngs_modulus_mpa,
        )
        pred_nf = solve_cmb_morrow_bisection(
            np.maximum(pred[:, 0], 1e-12),
            pred[:, 1],
            mask_np,
            epsilon_f_prime=args.epsilon_f_prime,
            sigma_f_prime_mpa=args.sigma_f_prime_mpa,
            b=args.b,
            c=args.c,
            youngs_modulus_mpa=args.youngs_modulus_mpa,
        )
        for bidx in range(pred.shape[0]):
            finite = mask_np[bidx] & np.isfinite(target_nf[bidx]) & np.isfinite(pred_nf[bidx])
            eps_rel = np.sum(np.abs(pred[bidx, 0][finite] - target[bidx, 0][finite])) / np.sum(
                np.abs(target[bidx, 0][finite])
            ) * 100.0
            sig_rel = np.sum(np.abs(pred[bidx, 1][finite] - target[bidx, 1][finite])) / np.sum(
                np.abs(target[bidx, 1][finite])
            ) * 100.0
            log_mae = np.mean(np.abs(np.log10(pred_nf[bidx][finite]) - np.log10(target_nf[bidx][finite])))
            rel_life = np.abs(pred_nf[bidx][finite] - target_nf[bidx][finite]) / np.maximum(
                target_nf[bidx][finite], 1e-30
            ) * 100.0
            target_min = float(np.nanmin(target_nf[bidx]))
            pred_min = float(np.nanmin(pred_nf[bidx]))
            rows.append(
                {
                    "sample_id": int(sample_ids[bidx]),
                    "layout": str(layouts[bidx]),
                    "epsilon_rel_mae_%": float(eps_rel),
                    "sigma_rel_mae_%": float(sig_rel),
                    "field_rel_mean_%": float(0.5 * (eps_rel + sig_rel)),
                    "log10_Nf_mae": float(log_mae),
                    "Nf_rel_mae_%": float(np.mean(rel_life)),
                    "target_Nf_min": target_min,
                    "pred_Nf_min": pred_min,
                    "Nf_min_rel_error_%": float(abs(pred_min - target_min) / max(target_min, 1e-30) * 100.0),
                }
            )

    layout_rows: list[dict] = []
    for layout in sorted({r["layout"] for r in rows}):
        subset = [r for r in rows if r["layout"] == layout]
        layout_rows.append(
            {
                "layout": layout,
                "count": len(subset),
                "epsilon_rel_mae_mean_%": float(np.mean([r["epsilon_rel_mae_%"] for r in subset])),
                "sigma_rel_mae_mean_%": float(np.mean([r["sigma_rel_mae_%"] for r in subset])),
                "field_rel_mean_%": float(np.mean([r["field_rel_mean_%"] for r in subset])),
                "log10_Nf_mae_mean": float(np.mean([r["log10_Nf_mae"] for r in subset])),
                "Nf_rel_mae_mean_%": float(np.mean([r["Nf_rel_mae_%"] for r in subset])),
                "Nf_min_rel_error_mean_%": float(np.mean([r["Nf_min_rel_error_%"] for r in subset])),
            }
        )

    write_csv(out_dir / f"life_field_{args.split}_sample_summary.csv", rows)
    write_csv(out_dir / f"life_field_{args.split}_layout_summary.csv", layout_rows)
    plot_summary(out_dir, rows, layout_rows)

    total = {
        "split": args.split,
        "count": len(rows),
        "epsilon_rel_mae_mean_%": float(np.mean([r["epsilon_rel_mae_%"] for r in rows])),
        "sigma_rel_mae_mean_%": float(np.mean([r["sigma_rel_mae_%"] for r in rows])),
        "field_rel_mean_%": float(np.mean([r["field_rel_mean_%"] for r in rows])),
        "log10_Nf_mae_mean": float(np.mean([r["log10_Nf_mae"] for r in rows])),
        "Nf_rel_mae_mean_%": float(np.mean([r["Nf_rel_mae_%"] for r in rows])),
        "Nf_min_rel_error_mean_%": float(np.mean([r["Nf_min_rel_error_%"] for r in rows])),
    }
    (out_dir / f"life_field_{args.split}_total_summary.json").write_text(
        json.dumps(total, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(total, ensure_ascii=False, indent=2))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
