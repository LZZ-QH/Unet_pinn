#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a U-Net on the Jian 2024 multi-layout fatigue-variable dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from jian_dataset import (
    CHANNELS,
    INPUT_MODES,
    JianFatigueDataset,
    compute_stats,
    discover_samples,
    input_channels,
    split_samples,
    write_split_summary,
)
from unet import UNetPowerToFatigue, count_parameters


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"
DEFAULT_SAVE_DIR = ROOT / "paper_2026" / "artifacts" / "unet_runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, mask: torch.Tensor, loss: str, beta: float) -> torch.Tensor:
    mask_c = mask.expand_as(target_norm)
    if loss == "mse":
        err = (pred_norm - target_norm) ** 2
    elif loss == "l1":
        err = torch.abs(pred_norm - target_norm)
    elif loss == "huber":
        err = F.smooth_l1_loss(pred_norm, target_norm, beta=beta, reduction="none")
    else:
        raise ValueError(loss)
    return (err * mask_c).sum() / mask_c.sum().clamp_min(1.0)


def metrics_from_batch(
    pred_norm: torch.Tensor,
    target_real: torch.Tensor,
    mask: torch.Tensor,
    stats: dict,
    channel_names: tuple[str, ...],
) -> dict[str, float]:
    mean = torch.as_tensor(stats["target"]["mean"], device=pred_norm.device, dtype=pred_norm.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(stats["target"]["std"], device=pred_norm.device, dtype=pred_norm.dtype).view(1, -1, 1, 1)
    pred = pred_norm * std + mean
    mask_c = mask.expand_as(target_real)
    abs_err = torch.abs(pred - target_real) * mask_c
    abs_target = torch.abs(target_real) * mask_c
    denom = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
    out: dict[str, float] = {}
    for c, name in enumerate(channel_names):
        mae = abs_err[:, c].sum() / denom.sum()
        rel = abs_err[:, c].sum() / abs_target[:, c].sum().clamp_min(1e-12)
        out[f"mae_{name}"] = float(mae.detach().cpu())
        out[f"relmae_{name}"] = float((100.0 * rel).detach().cpu())
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict,
    channel_names: tuple[str, ...],
    optimizer: torch.optim.Optimizer | None,
    loss_name: str,
    huber_beta: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for x, y_norm, y_real, mask, _sid, _layout in tqdm(loader, leave=False, desc="train" if train else "eval"):
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        y_real = y_real.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = masked_loss(pred, y_norm, mask, loss_name, huber_beta)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = int(x.shape[0])
        count += batch
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch
        batch_metrics = metrics_from_batch(pred.detach(), y_real, mask, stats, channel_names)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def data_parallel_if_needed(model: nn.Module, device: torch.device, gpu_ids: list[int] | None) -> nn.Module:
    if device.type != "cuda":
        return model
    available = torch.cuda.device_count()
    if gpu_ids is None:
        gpu_ids = list(range(available)) if available > 1 else []
    gpu_ids = [idx for idx in gpu_ids if 0 <= idx < available]
    return nn.DataParallel(model, device_ids=gpu_ids) if len(gpu_ids) > 1 else model


def state_dict_for_save(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="power_mask_xy_total")
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS))
    parser.add_argument("--holdout-layout", default=None, help="Use one layout as test set, for example L7.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--loss", choices=["mse", "huber", "l1"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-dropout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one train and one eval batch only.")
    args = parser.parse_args()

    set_seed(args.seed)
    samples = discover_samples(args.label_dir)
    if not samples:
        raise SystemExit(f"No labels found in {args.label_dir}")

    splits = split_samples(
        samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        holdout_layout=args.holdout_layout,
    )
    train_idx, val_idx, test_idx = splits
    stats = compute_stats(samples, train_idx, channels=args.channels)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"jian_unet_{args.input_mode}"
    if args.holdout_layout:
        run_name += f"_holdout_{args.holdout_layout}"
    if args.no_attention:
        run_name += "_noattention"
    run_dir = args.save_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_split_summary(run_dir / "splits.json", samples, splits)
    (run_dir / "normalization.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    train_ds = JianFatigueDataset(samples, train_idx, stats, input_mode=args.input_mode, channels=args.channels)
    val_ds = JianFatigueDataset(samples, val_idx, stats, input_mode=args.input_mode, channels=args.channels)
    test_ds = JianFatigueDataset(samples, test_idx, stats, input_mode=args.input_mode, channels=args.channels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = UNetPowerToFatigue(
        in_channels=input_channels(args.input_mode),
        out_channels=len(args.channels),
        base_ch=args.base_ch,
        use_attention=not args.no_attention,
        dropout=not args.no_dropout,
    ).to(device)
    model = data_parallel_if_needed(model, device, args.gpu_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    config = vars(args).copy()
    config.update(
        {
            "label_dir": str(args.label_dir),
            "run_dir": str(run_dir),
            "sample_count": len(samples),
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "test_count": len(test_idx),
            "input_channels": input_channels(args.input_mode),
            "parameter_count": count_parameters(model.module if isinstance(model, nn.DataParallel) else model),
            "layouts": sorted({s.layout for s in samples}),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2, default=str))

    best_score = float("inf")
    best_path = run_dir / "best.pth"
    rows: list[dict] = []
    max_epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, stats, tuple(args.channels), optimizer, args.loss, args.huber_beta)
        val_metrics = run_epoch(model, val_loader, device, stats, tuple(args.channels), None, args.loss, args.huber_beta)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["loss"] < best_score:
            best_score = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": state_dict_for_save(model),
                    "stats": stats,
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_score,
                },
                best_path,
            )
        if args.smoke:
            break

    final_path = run_dir / "final.pth"
    torch.save(
        {
            "model_state_dict": state_dict_for_save(model),
            "stats": stats,
            "config": config,
            "epoch": rows[-1]["epoch"],
            "best_val_loss": best_score,
        },
        final_path,
    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, stats, tuple(args.channels), None, args.loss, args.huber_beta)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "training_log.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", best_path)
    print("test", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
