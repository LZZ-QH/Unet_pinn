#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick data/model plumbing check for the Jian U-Net dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jian_dataset import (
    CHANNELS,
    INPUT_MODES,
    JianFatigueDataset,
    compute_stats,
    discover_samples,
    input_channels,
    split_samples,
)
from unet import UNetPowerToFatigue, count_parameters


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="power_mask_xy_total")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-ch", type=int, default=16)
    args = parser.parse_args()

    samples = discover_samples(args.label_dir)
    if len(samples) < 3:
        raise SystemExit(
            f"Need at least three portable label/power pairs in {args.label_dir}; "
            "pass --label-dir explicitly after downloading the dataset."
        )
    train_idx, val_idx, test_idx = split_samples(samples, seed=42)
    stats = compute_stats(samples, train_idx, channels=CHANNELS)
    ds = JianFatigueDataset(samples, train_idx[: max(args.batch_size, 2)], stats, input_mode=args.input_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    x, y_norm, y_real, mask, sample_ids, layouts = next(iter(loader))
    model = UNetPowerToFatigue(
        in_channels=input_channels(args.input_mode),
        out_channels=len(CHANNELS),
        base_ch=args.base_ch,
        use_attention=True,
    )
    with torch.no_grad():
        pred = model(x)
    print("samples", len(samples), "splits", len(train_idx), len(val_idx), len(test_idx))
    print("batch x", tuple(x.shape), "y_norm", tuple(y_norm.shape), "mask", tuple(mask.shape))
    print("pred", tuple(pred.shape), "params", count_parameters(model))
    print("sample_ids", [int(v) for v in sample_ids], "layouts", list(layouts))
    print("target real min/max", float(y_real.min()), float(y_real.max()))
    print("input min/max", float(x.min()), float(x.max()))


if __name__ == "__main__":
    main()
