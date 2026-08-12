#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute geometry-context channels for Jian multi-layout U-Net inputs."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jian_dataset import (
    GEOMETRY_CONTEXT_CHANNELS,
    GEOMETRY_CONTEXT_VERSION,
    _geometry_context_channels,
    _load_power_grid,
    discover_samples,
    geometry_cache_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = ROOT / "paper_2026" / "data" / "main2500" / "labels"


def _run_one(args: tuple[int, str, str, bool]) -> dict:
    sample_id, label_file_s, power_file_s, overwrite = args
    label_file = Path(label_file_s)
    power_file = Path(power_file_s)
    out_path = geometry_cache_path(label_file)
    if out_path.exists() and not overwrite:
        return {"sample_id": sample_id, "status": "skipped", "path": str(out_path)}

    label = np.load(label_file, allow_pickle=True)
    valid_mask = label["valid_mask"].astype(bool)
    power, _x_grid, _y_grid, _chip_powers = _load_power_grid(power_file)
    active = power > 1e-12
    geom = np.stack(_geometry_context_channels(power, active, valid_mask), axis=0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        geom=geom,
        channel_names=np.asarray(GEOMETRY_CONTEXT_CHANNELS),
        version=np.asarray(GEOMETRY_CONTEXT_VERSION),
        sample_id=np.asarray(sample_id),
        source_label=np.asarray(str(label_file)),
        source_power=np.asarray(str(power_file)),
    )
    tmp_path.replace(out_path)
    return {
        "sample_id": sample_id,
        "status": "written",
        "path": str(out_path),
        "shape": "x".join(str(v) for v in geom.shape),
        "min": float(geom.min()),
        "max": float(geom.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N limit for smoke tests.")
    args = parser.parse_args()

    samples = discover_samples(args.label_dir)
    if args.limit > 0:
        samples = samples[: args.limit]
    tasks = [(s.sample_id, str(s.label_file), str(s.power_file), args.overwrite) for s in samples]
    if not tasks:
        raise SystemExit(f"No labels found in {args.label_dir}")

    rows: list[dict] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="geometry context"):
            rows.append(_run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_run_one, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="geometry context"):
                rows.append(fut.result())

    rows.sort(key=lambda item: int(item["sample_id"]))
    cache_dir = geometry_cache_path(samples[0].label_file).parent
    report_json = cache_dir / "geometry_context_precompute_summary.json"
    report_csv = cache_dir / "geometry_context_precompute_summary.csv"
    payload = {
        "label_dir": str(args.label_dir),
        "cache_dir": str(cache_dir),
        "version": GEOMETRY_CONTEXT_VERSION,
        "channel_names": list(GEOMETRY_CONTEXT_CHANNELS),
        "total": len(rows),
        "written": sum(1 for r in rows if r["status"] == "written"),
        "skipped": sum(1 for r in rows if r["status"] == "skipped"),
    }
    report_json.write_text(json.dumps({"summary": payload, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
