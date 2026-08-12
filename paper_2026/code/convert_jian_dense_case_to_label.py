#!/usr/bin/env python3
"""Convert one completed Jian dense COMSOL TSV into a 256x256 U-Net label.

This is a thin, recoverable wrapper around build_jian_unet_labels_and_figures.py.
It is intended for batch production: after one COMSOL case finishes, call this
script with the same manifest and sample id to immediately create the training
label NPZ and preview figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_jian_unet_labels_and_figures as builder  # noqa: E402


DEFAULT_LABEL_DIR = ROOT / "paper_2026/data/main2500/labels"
DEFAULT_FIG_DIR = ROOT / "paper_2026/artifacts/label_previews"
DEFAULT_REPORT_DIR = ROOT / "paper_2026/artifacts/label_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one Jian dense TSV into unet_label_<sample>_256x256.npz."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest containing sample_id/case_name.")
    parser.add_argument("--sample-id", required=True, help="Sample id to convert.")
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Directory containing the COMSOL solder-cycle TSV exports.",
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefer-dense", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the NPZ already exists.")
    parser.add_argument("--no-figure", action="store_true", help="Suppress per-sample figure generation.")
    parser.add_argument(
        "--figure-interval",
        type=int,
        default=int(os.environ.get("JIAN_LABEL_FIG_INTERVAL", "20")),
        help=(
            "Generate a preview figure only when sample_id is divisible by this interval. "
            "Use 1 for every sample, or 0 with --no-figure-like behavior."
        ),
    )
    return parser.parse_args()


def read_existing_records(index_path: Path) -> list[dict[str, object]]:
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_reports(records: list[dict[str, object]], report_dir: Path, label_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    index_path = label_dir / "jian_unet_label_index.json"
    index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    if not records:
        return
    csv_path = report_dir / "jian_unet_label_interpolation_summary.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    md_path = report_dir / "jian_unet_label_interpolation_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Jian U-Net label interpolation summary\n\n")
        f.write(
            "This report is incrementally maintained by "
            "`scripts/convert_jian_dense_case_to_label.py` during batch production.\n\n"
        )
        f.write("| sample | case | raw rows | xy points | valid pixels | valid ratio | epsilon range | sigma_mean range |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for rec in sorted(records, key=lambda r: int(r["sample_id"])):
            f.write(
                f"| {rec['sample_id']} | `{rec['case_name']}` | {rec['raw_rows']} | "
                f"{rec['combined_xy_points']} | {rec['valid_pixels']} | {rec['valid_ratio']:.3f} | "
                f"{rec['epsilon_min_valid']:.7f}--{rec['epsilon_max_valid']:.7f} | "
                f"{rec['sigma_mean_min_valid_MPa']:.2f}--{rec['sigma_mean_max_valid_MPa']:.2f} MPa |\n"
            )


def main() -> None:
    args = parse_args()
    manifest = builder.load_manifest(args.manifest)
    sample_id = str(args.sample_id)
    if sample_id not in manifest:
        raise SystemExit(f"sample_id {sample_id} not found in manifest: {args.manifest}")

    args.label_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    out_npz = args.label_dir / f"unet_label_{sample_id}_256x256.npz"
    if out_npz.exists() and not args.force:
        print(f"SKIP existing label: {out_npz}")
        return

    try:
        sample_int = int(sample_id)
    except ValueError:
        sample_int = -1
    make_figure = (
        not args.no_figure
        and args.figure_interval > 0
        and (args.figure_interval == 1 or sample_int % args.figure_interval == 0)
    )

    builder.LABEL_DIR = args.label_dir
    builder.FIG_DIR = args.fig_dir
    builder.REPORT_DIR = args.report_dir
    builder.RESULT_ROOT = args.result_root
    builder.MAKE_CASE_FIGURE = make_figure

    case_name = manifest[sample_id]["case_name"]
    record = builder.save_label(sample_id, case_name, manifest[sample_id], prefer_dense=args.prefer_dense)

    index_path = args.label_dir / "jian_unet_label_index.json"
    records = read_existing_records(index_path)
    merged: dict[str, dict[str, object]] = {
        str(rec.get("sample_id")): rec for rec in records if "sample_id" in rec
    }
    merged[sample_id] = record
    write_reports(list(merged.values()), args.report_dir, args.label_dir)

    print(f"Wrote label: {record['label_npz']}")
    if make_figure:
        print(f"Wrote figure: {record['figure']}")
    else:
        print(f"Skipped figure: sample_id={sample_id}, figure_interval={args.figure_interval}")


if __name__ == "__main__":
    main()
