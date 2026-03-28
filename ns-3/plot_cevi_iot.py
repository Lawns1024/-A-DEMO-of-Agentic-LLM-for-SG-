#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CEVI IoT MAC metrics from CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cevi_metrics.csv"),
        help="Path to cevi_metrics.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cevi-metrics"),
        help="Output file prefix (without extension)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    normalized_columns = {col.strip(): col for col in df.columns}
    expected = {
        "Time_s": "time_s",
        "Throughput_Mbps": "throughput_mbps",
        "CollisionRate": "collision_rate",
    }
    missing = [src for src in expected if src not in normalized_columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.rename(columns={normalized_columns[src]: dst for src, dst in expected.items()})

    # 同一时间戳有多个节点记录，取均值生成全局曲线
    df = df[df["time_s"] >= 1.5].copy()
    grouped = (
        df.groupby("time_s", as_index=False)
        .agg({"throughput_mbps": "mean", "collision_rate": "mean"})
        .sort_values("time_s")
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    axes[0].plot(grouped["time_s"], grouped["throughput_mbps"], color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[0].set_title("CEVI IoT MAC Metrics")

    axes[1].plot(grouped["time_s"], grouped["collision_rate"], color="#d62728", linewidth=2)
    axes[1].set_ylabel("Collision Rate")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()

    out_prefix = args.out.expanduser().resolve()
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
