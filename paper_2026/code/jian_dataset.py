#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset and split utilities for the Jian 2024 multi-layout U-Net data."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset


CHANNELS = ("epsilon_eq_total_a", "sigma_mean_MPa")
LAYOUT_ORDER = ("L0", "L4", "L5", "L6", "L7")
INPUT_MODES = (
    "power",
    "power_mask_xy",
    "power_mask_xy_total",
    "power_mask_xy_layout",
    "power_mask_xy_total_geom",
)


def sample_id_from_name(path: str | Path) -> int | None:
    match = re.search(r"unet_label_(\d+)_256x256", Path(path).name)
    return int(match.group(1)) if match else None


def layout_from_sample_id(sample_id: int) -> str:
    if 10001 <= sample_id <= 10500:
        return "L0"
    if 20001 <= sample_id <= 20099:
        return "L0"
    if 16001 <= sample_id <= 16500:
        return "L4"
    if 20401 <= sample_id <= 20499:
        return "L4"
    if 13001 <= sample_id <= 13500:
        return "L5"
    if 20501 <= sample_id <= 20599:
        return "L5"
    if 14001 <= sample_id <= 14500:
        return "L6"
    if 20601 <= sample_id <= 20699:
        return "L6"
    if 15001 <= sample_id <= 15500:
        return "L7"
    if 20701 <= sample_id <= 20799:
        return "L7"
    return "unknown"


def _scalar_str(value) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


@dataclass(frozen=True)
class JianSample:
    sample_id: int
    label_file: Path
    power_file: Path
    case_name: str
    layout: str


def resolve_power_path(label_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    # Fallbacks for portable datasets where labels and power grids share a parent.
    candidates = [
        label_dir.parent / "power" / path.name,
        label_dir / path.name,
        label_dir.parent / path.name,
        label_dir.parent / "input_power_grid_256x256_layout_scaled_main500" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Power grid not found: {raw_path}")


def discover_samples(label_dir: str | Path) -> list[JianSample]:
    label_dir = Path(label_dir)
    samples: list[JianSample] = []
    for label_file in sorted(label_dir.glob("unet_label_*_256x256.npz")):
        sample_id = sample_id_from_name(label_file)
        if sample_id is None:
            continue
        z = np.load(label_file, allow_pickle=True)
        power_file = resolve_power_path(label_dir, _scalar_str(z["power_grid_npz"]))
        case_name = _scalar_str(z["case_name"])
        layout = layout_from_sample_id(sample_id)
        samples.append(JianSample(sample_id, label_file, power_file, case_name, layout))
    return samples


def split_samples(
    samples: Sequence[JianSample],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    holdout_layout: str | None = None,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    if holdout_layout:
        holdout = {holdout_layout}
        pool_by_layout: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            if sample.layout in holdout:
                test.append(idx)
            else:
                pool_by_layout.setdefault(sample.layout, []).append(idx)
        for indices in pool_by_layout.values():
            indices = list(indices)
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * val_fraction)))
            val.extend(indices[:n_val])
            train.extend(indices[n_val:])
        return sorted(train), sorted(val), sorted(test)

    by_layout: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        by_layout.setdefault(sample.layout, []).append(idx)
    for indices in by_layout.values():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
        n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])
    return sorted(train), sorted(val), sorted(test)


def _load_power_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    if "power_grid" in z:
        power = z["power_grid"].astype(np.float32)
    elif "X" in z:
        power = z["X"].astype(np.float32)
        if power.ndim == 3:
            power = power[0]
    else:
        raise KeyError(f"No power_grid or X in {path}")
    chip_powers = z["chip_powers_W"].astype(np.float32) if "chip_powers_W" in z else np.zeros(6, np.float32)
    return power, z["X_grid"].astype(np.float32), z["Y_grid"].astype(np.float32), chip_powers


def compute_stats(samples: Sequence[JianSample], indices: Iterable[int], channels: Sequence[str] = CHANNELS) -> dict:
    powers: list[np.ndarray] = []
    log_totals: list[list[float]] = []
    target_chunks: list[np.ndarray] = []

    for idx in indices:
        sample = samples[idx]
        label = np.load(sample.label_file, allow_pickle=True)
        names = [str(x) for x in label["channel_names"]]
        channel_idx = [names.index(name) for name in channels]
        mask = label["valid_mask"].astype(bool)
        target = label["Y"][channel_idx].astype(np.float64)
        target_chunks.append(target[:, mask])
        power, _xg, _yg, chip_powers = _load_power_grid(sample.power_file)
        active = power > 1e-12
        if np.any(active):
            powers.append(power[active].astype(np.float64))
        total = float(chip_powers.sum()) if chip_powers.size else float(power.sum())
        max_chip = float(chip_powers.max()) if chip_powers.size else float(power.max())
        log_totals.append([math.log10(max(total, 1e-12)), math.log10(max(max_chip, 1e-12))])

    power_all = np.concatenate(powers) if powers else np.ones(1, dtype=np.float64)
    power_median = float(np.median(power_all))
    power_iqr = float(np.percentile(power_all, 75) - np.percentile(power_all, 25))
    if abs(power_iqr) < 1e-12:
        power_iqr = 1.0

    target_all = np.concatenate(target_chunks, axis=1)
    target_mean = target_all.mean(axis=1)
    target_std = np.maximum(target_all.std(axis=1), 1e-12)
    log_arr = np.asarray(log_totals, dtype=np.float64)
    log_std = np.maximum(log_arr.std(axis=0), 1e-12)
    return {
        "channels": list(channels),
        "layout_order": list(LAYOUT_ORDER),
        "power": {
            "method": "nonzero_median_iqr",
            "median": power_median,
            "iqr": power_iqr,
            "log_total_mean": log_arr.mean(axis=0).astype(float).tolist(),
            "log_total_std": log_std.astype(float).tolist(),
        },
        "target": {
            "method": "valid_pixel_zscore",
            "mean": target_mean.astype(float).tolist(),
            "std": target_std.astype(float).tolist(),
        },
    }


def _component_summary(mask: np.ndarray, values: np.ndarray | None = None) -> list[dict]:
    labels, n_labels = ndi.label(mask.astype(bool))
    out: list[dict] = []
    for lab in range(1, n_labels + 1):
        yy, xx = np.nonzero(labels == lab)
        if yy.size == 0:
            continue
        item = {
            "label": lab,
            "cy": float(yy.mean()),
            "cx": float(xx.mean()),
            "y0": int(yy.min()),
            "y1": int(yy.max()),
            "x0": int(xx.min()),
            "x1": int(xx.max()),
            "area": int(yy.size),
        }
        if values is not None:
            vals = values[labels == lab]
            nz = vals[np.abs(vals) > 1e-12]
            item["value"] = float(nz.mean()) if nz.size else 0.0
        out.append(item)
    # Stable rank order independent of connected-component labeling details.
    out.sort(key=lambda item: (item["cy"], item["cx"]))
    return out


def _normalize_rank(rank: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return float(2.0 * rank / (total - 1) - 1.0)


GEOMETRY_CONTEXT_VERSION = "geom_context_v1"
GEOMETRY_CONTEXT_CHANNELS = (
    "island_rank",
    "nearest_chip_rank",
    "local_island_x",
    "local_island_y",
    "island_width",
    "island_height",
    "distance_to_solder_edge",
    "nearest_chip_power_ratio",
    "neighbor_power_ratio",
)


def geometry_cache_path(label_file: Path) -> Path:
    sample_id = sample_id_from_name(label_file)
    if sample_id is None:
        raise ValueError(f"Cannot parse sample id from {label_file}")
    return label_file.parent / GEOMETRY_CONTEXT_VERSION / f"geom_context_{sample_id}_256x256.npz"


def _geometry_context_channels(power: np.ndarray, active: np.ndarray, valid_mask: np.ndarray) -> list[np.ndarray]:
    """Build continuous geometry/context maps for layout-holdout generalization.

    The maps are derived from the current sample only, so they can also be
    computed for a layout family that was never seen during training.
    """

    h, w = valid_mask.shape
    dtype = np.float32
    yy_grid, xx_grid = np.indices((h, w), dtype=np.float32)

    island_rank = np.zeros((h, w), dtype=dtype)
    nearest_chip_rank = np.zeros((h, w), dtype=dtype)
    local_x = np.zeros((h, w), dtype=dtype)
    local_y = np.zeros((h, w), dtype=dtype)
    island_width = np.zeros((h, w), dtype=dtype)
    island_height = np.zeros((h, w), dtype=dtype)
    solder_edge_distance = np.zeros((h, w), dtype=dtype)
    nearest_chip_power_ratio = np.zeros((h, w), dtype=dtype)
    neighbor_power_ratio = np.zeros((h, w), dtype=dtype)

    islands = _component_summary(valid_mask)
    chips = _component_summary(active, power)
    chip_values = np.asarray([float(item.get("value", 0.0)) for item in chips], dtype=np.float32)
    positive_values = chip_values[chip_values > 1e-12]
    mean_chip_power = float(positive_values.mean()) if positive_values.size else 1.0

    chip_centers = np.asarray([[item["cy"], item["cx"]] for item in chips], dtype=np.float32) if chips else np.zeros((0, 2), dtype=np.float32)

    island_labels, _ = ndi.label(valid_mask.astype(bool))
    for island_idx, island in enumerate(islands):
        lab = int(island["label"])
        component = island_labels == lab
        if not np.any(component):
            continue

        y0, y1 = int(island["y0"]), int(island["y1"])
        x0, x1 = int(island["x0"]), int(island["x1"])
        width = max(float(x1 - x0 + 1), 1.0)
        height = max(float(y1 - y0 + 1), 1.0)

        island_rank[component] = _normalize_rank(island_idx, len(islands))
        local_x[component] = 2.0 * ((xx_grid[component] - x0) / max(width - 1.0, 1.0)) - 1.0
        local_y[component] = 2.0 * ((yy_grid[component] - y0) / max(height - 1.0, 1.0)) - 1.0
        island_width[component] = width / float(w)
        island_height[component] = height / float(h)

        # Distance inside each solder island; normalized to island size.
        dist = ndi.distance_transform_edt(component).astype(dtype)
        solder_edge_distance[component] = dist[component] / max(min(width, height), 1.0)

        if chip_centers.shape[0] == 0:
            continue
        center = np.asarray([float(island["cy"]), float(island["cx"])], dtype=np.float32)
        distances = np.linalg.norm(chip_centers - center[None, :], axis=1)
        nearest = int(np.argmin(distances))
        nearest_chip_rank[component] = _normalize_rank(nearest, len(chips))
        nearest_value = float(chip_values[nearest]) if chip_values.size else 0.0
        nearest_chip_power_ratio[component] = nearest_value / max(mean_chip_power, 1e-12)

        if len(chips) > 1:
            weights = 1.0 / np.maximum(distances, 1.0)
            weights[nearest] = 0.0
            denom = float(weights.sum())
            if denom > 1e-12:
                neighbor_power = float((weights * chip_values).sum() / denom)
                neighbor_power_ratio[component] = neighbor_power / max(mean_chip_power, 1e-12)

    return [
        island_rank,
        nearest_chip_rank,
        local_x,
        local_y,
        island_width,
        island_height,
        solder_edge_distance,
        nearest_chip_power_ratio,
        neighbor_power_ratio,
    ]


class JianFatigueDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[JianSample],
        indices: Sequence[int],
        stats: dict,
        input_mode: str = "power_mask_xy_total",
        channels: Sequence[str] = CHANNELS,
        cache: bool = False,
    ) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input_mode={input_mode}; choose from {INPUT_MODES}")
        self.samples = list(samples)
        self.indices = list(indices)
        self.stats = stats
        self.input_mode = input_mode
        self.channels = tuple(channels)
        self.cache = cache
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _make_input(
        self,
        power: np.ndarray,
        valid_mask: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        chip_powers: np.ndarray,
        layout: str,
        geom_extra: np.ndarray | None = None,
    ) -> np.ndarray:
        pstats = self.stats["power"]
        power_norm = power.copy().astype(np.float32)
        active = power_norm > 1e-12
        power_norm[active] = (power_norm[active] - float(pstats["median"])) / float(pstats["iqr"])
        channels = [power_norm]
        if self.input_mode in {"power_mask_xy", "power_mask_xy_total", "power_mask_xy_layout", "power_mask_xy_total_geom"}:
            channels.append(active.astype(np.float32))
            channels.append(valid_mask.astype(np.float32))
            x_scale = max(float(np.max(np.abs(x_grid))), 1e-12)
            y_scale = max(float(np.max(np.abs(y_grid))), 1e-12)
            channels.append((x_grid / x_scale).astype(np.float32))
            channels.append((y_grid / y_scale).astype(np.float32))
        if self.input_mode in {"power_mask_xy_total", "power_mask_xy_layout", "power_mask_xy_total_geom"}:
            total = float(chip_powers.sum()) if chip_powers.size else float(power.sum())
            max_chip = float(chip_powers.max()) if chip_powers.size else float(power.max())
            log_values = np.asarray(
                [math.log10(max(total, 1e-12)), math.log10(max(max_chip, 1e-12))],
                dtype=np.float32,
            )
            mean = np.asarray(pstats["log_total_mean"], dtype=np.float32)
            std = np.asarray(pstats["log_total_std"], dtype=np.float32)
            log_values = (log_values - mean) / np.maximum(std, 1e-12)
            channels.extend([np.full_like(power_norm, log_values[0]), np.full_like(power_norm, log_values[1])])
        if self.input_mode == "power_mask_xy_total_geom":
            if geom_extra is None:
                channels.extend(_geometry_context_channels(power, active, valid_mask))
            else:
                geom_extra = np.asarray(geom_extra, dtype=np.float32)
                if geom_extra.shape[0] != len(GEOMETRY_CONTEXT_CHANNELS):
                    raise ValueError(f"Expected {len(GEOMETRY_CONTEXT_CHANNELS)} geometry channels, got {geom_extra.shape}")
                channels.extend([geom_extra[i] for i in range(geom_extra.shape[0])])
        if self.input_mode == "power_mask_xy_layout":
            for name in LAYOUT_ORDER:
                channels.append(np.full_like(power_norm, 1.0 if layout == name else 0.0))
        return np.stack(channels, axis=0).astype(np.float32)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        if self.cache and idx in self._cache:
            return self._cache[idx]
        sample = self.samples[idx]
        label = np.load(sample.label_file, allow_pickle=True)
        names = [str(x) for x in label["channel_names"]]
        channel_idx = [names.index(name) for name in self.channels]
        target = label["Y"][channel_idx].astype(np.float32)
        mask = label["valid_mask"].astype(bool)
        power, x_grid, y_grid, chip_powers = _load_power_grid(sample.power_file)
        geom_extra = None
        if self.input_mode == "power_mask_xy_total_geom":
            cache_path = geometry_cache_path(sample.label_file)
            if cache_path.exists():
                geom_extra = np.load(cache_path, allow_pickle=False)["geom"]
        x = self._make_input(power, mask, x_grid, y_grid, chip_powers, sample.layout, geom_extra=geom_extra)
        y = target.copy()
        mean = np.asarray(self.stats["target"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(self.stats["target"]["std"], dtype=np.float32).reshape(-1, 1, 1)
        y_norm = (y - mean) / np.maximum(std, 1e-12)
        out = (
            torch.from_numpy(x),
            torch.from_numpy(y_norm.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(mask[None].astype(np.float32)),
            int(sample.sample_id),
            sample.layout,
        )
        if self.cache:
            self._cache[idx] = out
        return out


def input_channels(input_mode: str) -> int:
    if input_mode == "power":
        return 1
    if input_mode == "power_mask_xy":
        return 5
    if input_mode == "power_mask_xy_total":
        return 7
    if input_mode == "power_mask_xy_layout":
        return 12
    if input_mode == "power_mask_xy_total_geom":
        return 16
    raise ValueError(input_mode)


def write_split_summary(path: str | Path, samples: Sequence[JianSample], splits: tuple[list[int], list[int], list[int]]) -> None:
    path = Path(path)
    names = ("train", "val", "test")
    payload = {}
    for name, indices in zip(names, splits):
        layout_counts: dict[str, int] = {}
        sample_ids: list[int] = []
        for idx in indices:
            sample = samples[idx]
            sample_ids.append(sample.sample_id)
            layout_counts[sample.layout] = layout_counts.get(sample.layout, 0) + 1
        payload[name] = {"count": len(indices), "layout_counts": layout_counts, "sample_ids": sample_ids}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
