from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .metrics import regression_metrics

sns.set_style("whitegrid")


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    sample_size: Optional[int] = 200_000,
    annotate_r2: bool = True,
    annotation_metrics: Optional[Dict[str, float]] = None,
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return

    if sample_size is not None and y_true.size > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(y_true.size, size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    if min_val == max_val:
        max_val = min_val + 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.3, edgecolor="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="crimson", linewidth=1.5)
    plt.xlabel("Actual expression")
    plt.ylabel("Predicted expression")
    plt.title(title)
    if annotate_r2:
        if annotation_metrics is None:
            metrics = regression_metrics(y_true, y_pred)
            mean_override = False
        else:
            metrics = annotation_metrics
            mean_override = True
        def _fmt(value: float) -> str:
            return "nan" if not np.isfinite(value) else f"{value:.3f}"

        lines = []
        r2_val = metrics.get("r2") if metrics is not None else None
        if r2_val is not None:
            lines.append(f"$R^2={_fmt(r2_val)}$")
        pearson_val = metrics.get("pearson") if metrics is not None else None
        if pearson_val is not None:
            if mean_override:
                label = "Mean per-gene Pearson"
            else:
                label = "Pearson"
            lines.append(f"{label}={_fmt(pearson_val)}")
        spearman_val = metrics.get("spearman") if metrics is not None else None
        if spearman_val is not None:
            if mean_override:
                label = "Mean per-gene Spearman"
            else:
                label = "Spearman"
            lines.append(f"{label}={_fmt(spearman_val)}")
        text = "\n".join(lines)
        if text:
            plt.text(
                min_val + 0.05 * (max_val - min_val),
                max_val - 0.12 * (max_val - min_val),
                text,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_pred - y_true
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Residual (prediction - actual)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: Iterable[str], output_path: Path, title: str) -> None:
    if importances.size == 0:
        return
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = np.asarray(list(feature_names))[indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_names)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metric_table(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric,value"] + [f"{name},{value}" for name, value in metrics.items()]
    output_path.write_text("\n".join(lines) + "\n")


def plot_residual_barplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: Sequence[str],
    output_path: Path,
    title: str,
    top_n: int = 30,
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        return
    if y_true.size == 0:
        return

    residuals = y_pred - y_true
    mae = np.nanmean(np.abs(residuals), axis=0)
    mean_res = np.nanmean(residuals, axis=0)

    if mae.size == 0:
        return

    order = np.argsort(mae)[::-1]
    limit = min(top_n, mae.size)
    idx = order[:limit]

    selected_genes = np.asarray(gene_names)[idx]
    selected_mae = mae[idx]
    selected_mean = mean_res[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.0, limit * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bar_colors = ["#d62728" if val >= 0 else "#1f77b4" for val in selected_mean]
    y_positions = np.arange(limit)
    ax.barh(y_positions, selected_mae, color=bar_colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(selected_genes)
    ax.invert_yaxis()  # put largest residuals at the top for readability
    ax.set_xlabel("Mean absolute residual")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_boxplot(
    values: Sequence[float],
    output_path: Path,
    title: str,
    metric_label: str,
    axes: plt.Axes | None = None,
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return

    if axes is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6.5, 4.0))
        ax = plt.gca()
        save_and_close = True
    else:
        ax = axes
        save_and_close = False

    sns.boxplot(y=arr, color="#4C72B0", orient="v", ax=ax)
    sns.stripplot(y=arr, orient="v", color="#1f77b4", alpha=0.7, size=4, jitter=0.15, ax=ax)
    ax.set_ylabel(metric_label)
    ax.set_xticks([])
    ax.set_title(title)
    if save_and_close:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_training_history_curves(
    history: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    include_train: bool = True,
    include_val: bool = True,
) -> None:
    if history.empty or "epoch" not in history.columns:
        return

    metric_clean = metric.lower()
    if metric_clean == "loss":
        train_col = "train_loss"
        val_col = "val_loss"
    else:
        train_col = f"train_{metric_clean}"
        val_col = f"val_{metric_clean}"

    curves = []
    labels = []
    if include_train and train_col in history:
        curves.append((history[train_col], "Train"))
    if include_val and val_col in history:
        curves.append((history[val_col], "Validation"))

    if not curves:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    for series, label in curves:
        plt.plot(history["epoch"], series, label=label, linewidth=2.0)
    plt.xlabel("Epoch")
    ylabel = "Loss" if metric_clean == "loss" else metric_clean.capitalize()
    plt.ylabel(ylabel)
    plt.title(title)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
