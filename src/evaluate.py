"""Evaluation metrics and publication-ready chart generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["FAKE", "REAL"]


def evaluate_predictions(y_true: Any, y_pred: Any, y_probability_real: Any) -> dict[str, Any]:
    """Calculate binary and aggregate metrics; REAL (1) is the positive class."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "positive_class": "REAL (1)",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability_real)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(metrics: dict[str, Any], title: str, output_path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
        annot_kws={"size": 13, "weight": "bold"},
    )
    axis.set_title(title, fontsize=14, weight="bold", pad=14)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    curves: dict[str, tuple[Any, Any]], metrics: dict[str, dict[str, Any]], output_path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    colors = {"logistic_regression": "#2563eb", "naive_bayes": "#0f766e"}
    labels = {"logistic_regression": "Logistic Regression", "naive_bayes": "Naive Bayes"}
    for model_key, (y_true, probabilities) in curves.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        auc_value = metrics[model_key]["test"]["roc_auc"]
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.4,
            color=colors[model_key],
            label=f"{labels[model_key]} (AUC = {auc_value:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random classifier")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Test-set ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(
    all_metrics: dict[str, dict[str, Any]],
    curves: dict[str, tuple[Any, Any]],
    results_dir: Path,
    selected_model: str,
) -> None:
    """Persist machine-readable results, summary CSV, and evaluation figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_criterion": "Highest validation macro F1; Logistic Regression wins exact ties.",
        "selected_model": selected_model,
        "positive_class": "REAL (1)",
        "models": all_metrics,
    }
    (results_dir / "evaluation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    display_names = {
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
    }
    rows: list[dict[str, Any]] = []
    for model_key, partitions in all_metrics.items():
        row: dict[str, Any] = {"model": display_names[model_key]}
        for split in ("validation", "test"):
            for metric in ("accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1", "roc_auc"):
                row[f"{split}_{metric}"] = partitions[split][metric]
        row["selected_for_deployment"] = model_key == selected_model
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "model_comparison.csv", index=False)

    save_confusion_matrix(
        all_metrics["logistic_regression"]["test"],
        "Logistic Regression — Test Confusion Matrix",
        results_dir / "confusion_matrix_lr.png",
    )
    save_confusion_matrix(
        all_metrics["naive_bayes"]["test"],
        "Multinomial Naive Bayes — Test Confusion Matrix",
        results_dir / "confusion_matrix_nb.png",
    )
    save_roc_curves(curves, all_metrics, results_dir / "roc_curve.png")

