#!/usr/bin/env python3
"""Parse training log .txt files (Model: lines) and produce per-metric heatmaps.

Usage:
    python plot_results.py Milleville_mixing_results.txt
    python plot_results.py Hussein_datamixing_results.txt --value_column test_miou

Outputs a subdirectory named <dataset>_<strategy>/ containing:
  - results.csv          (all parsed test metrics, one row per model)
  - test_miou_heatmap.png
  - test_dice_macro_heatmap.png
  - ...one PNG and SVG per test_* metric (or just the specified --value_column)
"""

import argparse
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_log(text: str) -> pd.DataFrame:
    """Parse 'Model: name, key: value, ...' lines into a DataFrame.

    Skips class_weights (contains commas inside brackets).
    Extracts dataset, strategy, num_real, num_syn from model_name.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("Model:"):
            continue
        parts = line[len("Model:"):].split(", ")
        row = {"model_name": parts[0].strip()}
        for p in parts[1:]:
            if p.startswith("class_weights"):
                break  # comma inside [ ] would corrupt split; stop here
            key, val = p.split(": ", 1)
            row[key.strip()] = val.strip()
        rows.append(row)

    df = pd.DataFrame(rows)

    # Extract structured fields from name: e.g. Milleville_mixing_190_real_0_synth
    m = df["model_name"].str.extract(
        r"^(?P<dataset>\w+)_(?P<strategy>\w+)_(?P<num_real>\d+)_real_(?P<num_syn>\d+)_synth$"
    )
    df = pd.concat([df, m], axis=1)
    df["num_real"] = df["num_real"].astype(int)
    df["num_syn"] = df["num_syn"].astype(int)

    # Coerce all test_* metric columns to numeric
    for col in df.columns:
        if col.startswith("test_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_heatmap(data: pd.DataFrame, value_col: str, out_dir: str) -> None:
    """Produce a single-panel heatmap in the style of vis_metrics.py.

    - viridis cmap, annotated with 3-decimal values
    - y-axis labels are percentage-of-max-real with n= count
    - x-axis is raw synthetic sample count
    - best cell annotated bold
    """
    pivot = (
        data.pivot(index="num_real", columns="num_syn", values=value_col)
        .sort_index(ascending=False)       # most real on top
        .sort_index(axis=1, ascending=True)  # least synth on left
    )

    total_real = data["num_real"].max()

    # Format y labels as "XX% (n=NNN)"
    pivot.index = [
        f"{round(v / total_real * 100)}% (n={v})" if total_real > 0 else f"n={v}"
        for v in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(pivot))))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=pivot.round(3).to_numpy(),  # display rounding only
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_col.replace("test_", "").replace("_", " ")},
    )

    # Bold the cell(s) holding the global max
    ax_vals = pivot.to_numpy()
    mask = ~np.isnan(ax_vals)
    if mask.any():
        best = np.nanmax(ax_vals)
        for row_i, col_j in zip(*np.where(mask & (ax_vals == best))):
            # seaborn places text at the cell centre; find by position
            texts = ax.texts
            idx = row_i * len(pivot.columns) + col_j  # row-major order matches annot grid
            if idx < len(texts):
                texts[idx].set_fontweight("bold")

    dataset = data["dataset"].iloc[0]
    strategy = data["strategy"].iloc[0]
    ax.set_title(f"{dataset} {strategy}: {value_col.replace('test_', '').replace('_', ' ')}", fontsize=11)
    ax.set_xlabel("synthetic samples", fontsize=9)
    ax.set_ylabel("real samples", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{value_col}_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"  saved {out_path}")
    
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    print(f"  saved {svg_path}")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot per-metric heatmaps from training log .txt files."
    )
    ap.add_argument("log_path", type=str, help="Path to results .txt with 'Model:' lines")
    ap.add_argument(
        "--value_column", "-m", type=str, default=None,
        help="Plot only this metric (e.g. test_miou). Default: all test_* columns."
    )
    args = ap.parse_args()

    with open(args.log_path) as fh:
        text = fh.read()

    df = parse_log(text)
    if df.empty:
        raise SystemExit(f"No 'Model:' lines found in {args.log_path}")

    # Determine output subdirectory from the (single) dataset+strategy combo
    out_dir = f"{df['dataset'].iloc[0]}_{df['strategy'].iloc[0]}".lower()
    os.makedirs(out_dir, exist_ok=True)

    # Co-output: full CSV for reviewer convenience
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} models)")

    # Determine which metrics to plot
    if args.value_column:
        cols = [args.value_column]
        if not cols[0].startswith("test_"):
            raise SystemExit(f"Unknown or non-test metric: {cols[0]}")
    else:
        cols = sorted(c for c in df.columns if c.startswith("test_"))

    print(f"Plotting {len(cols)} metric(s) into {out_dir}/:")
    for col in cols:
        make_heatmap(df, col, out_dir)


if __name__ == "__main__":
    main()
