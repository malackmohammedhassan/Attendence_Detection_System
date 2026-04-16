"""
Generate IEEE-paper-ready tables and charts from real benchmark data.

Source of truth:
  backend/results/dashboard.db (benchmark_results table)

Outputs:
  backend/exports/ieee/
    - latest_completed_runs.csv
    - latest_full_eval_runs.csv
    - model_summary.csv
    - figure_latency_mean_ms.png
    - figure_latency_p95_ms.png
    - figure_f1_full_eval.png
    - figure_cpu_memory_full_eval.png
    - figure_fps_vs_latency_scatter.png

Usage:
  cd backend
  python scripts/generate_ieee_results.py
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "results" / "dashboard.db"
OUT_DIR = ROOT_DIR / "exports" / "ieee"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_cpu_memory_chart(
    labels: list[str],
    cpu_values: list[float],
    mem_values: list[float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    cpu_bars = ax.bar(x - width / 2, cpu_values, width, label="CPU Avg (%)")
    mem_bars = ax.bar(x + width / 2, mem_values, width, label="Memory Avg (MB)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Full-Eval Resource Usage by Run")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bar in cpu_bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar in mem_bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_scatter(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    colors = {
        "scratch_cnn": "#1f77b4",
        "mtcnn": "#d62728",
    }

    for row in rows:
        latency = _to_float(row.get("latency_mean_ms"))
        fps = _to_float(row.get("avg_fps"))
        if latency is None or fps is None:
            continue
        model_name = str(row.get("model_name"))
        label = f"{model_name}-{str(row.get('run_id'))[:8]}"
        ax.scatter(
            latency,
            fps,
            color=colors.get(model_name, "#2ca02c"),
            alpha=0.8,
            s=60,
        )
        ax.annotate(label, (latency, fps), fontsize=7, xytext=(5, 4), textcoords="offset points")

    ax.set_title("FPS vs Mean Latency (Completed Runs)")
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Average FPS")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT run_id, model_name, status, frame_size, is_full_eval, frames_evaluated,
               precision, recall, f1, false_positives,
               cpu_avg, memory_avg_mb, avg_fps,
               latency_mean_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms,
               duration_sec, created_at
        FROM benchmark_results
        WHERE status = 'completed'
        ORDER BY created_at DESC
        """
    )
    completed_rows = [dict(r) for r in cur.fetchall()]

    full_eval_rows = [row for row in completed_rows if int(row.get("is_full_eval") or 0) == 1]

    cur.execute(
        """
        SELECT model_name,
               COUNT(*) AS runs,
               AVG(latency_mean_ms) AS latency_mean_ms,
               AVG(latency_p95_ms) AS latency_p95_ms,
               AVG(avg_fps) AS avg_fps,
               AVG(precision) AS precision,
               AVG(recall) AS recall,
               AVG(f1) AS f1
        FROM benchmark_results
        WHERE status = 'completed'
        GROUP BY model_name
        ORDER BY runs DESC, model_name ASC
        """
    )
    model_summary_rows = [dict(r) for r in cur.fetchall()]

    _save_csv(OUT_DIR / "latest_completed_runs.csv", completed_rows)
    _save_csv(OUT_DIR / "latest_full_eval_runs.csv", full_eval_rows)
    _save_csv(OUT_DIR / "model_summary.csv", model_summary_rows)

    if model_summary_rows:
        labels = [str(row["model_name"]) for row in model_summary_rows]

        mean_latency = [_to_float(row.get("latency_mean_ms")) or 0.0 for row in model_summary_rows]
        p95_latency = [_to_float(row.get("latency_p95_ms")) or 0.0 for row in model_summary_rows]
        _save_bar_chart(
            labels,
            mean_latency,
            "Average Mean Latency by Model (ms)",
            "Latency (ms)",
            OUT_DIR / "figure_latency_mean_ms.png",
        )
        _save_bar_chart(
            labels,
            p95_latency,
            "Average P95 Latency by Model (ms)",
            "Latency (ms)",
            OUT_DIR / "figure_latency_p95_ms.png",
        )

    if full_eval_rows:
        labels = [f"{row['model_name']}-{str(row['run_id'])[:8]}" for row in full_eval_rows]
        f1_values = [_to_float(row.get("f1")) or 0.0 for row in full_eval_rows]
        cpu_values = [_to_float(row.get("cpu_avg")) or 0.0 for row in full_eval_rows]
        mem_values = [_to_float(row.get("memory_avg_mb")) or 0.0 for row in full_eval_rows]

        _save_bar_chart(
            labels,
            f1_values,
            "F1 Score by Full-Eval Run",
            "F1 Score",
            OUT_DIR / "figure_f1_full_eval.png",
        )
        _save_cpu_memory_chart(labels, cpu_values, mem_values, OUT_DIR / "figure_cpu_memory_full_eval.png")

    if completed_rows:
        _save_scatter(completed_rows, OUT_DIR / "figure_fps_vs_latency_scatter.png")

    print(f"Completed runs: {len(completed_rows)}")
    print(f"Full-eval runs: {len(full_eval_rows)}")
    print(f"Model summary rows: {len(model_summary_rows)}")
    print(f"Saved IEEE artifacts to: {OUT_DIR}")


if __name__ == "__main__":
    main()
