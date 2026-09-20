"""Train, select, evaluate, and persist the VeriNews AI models."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.dataset import BINARY_LABEL_NAMES, RANDOM_SEED, prepare_dataset
from src.evaluate import evaluate_predictions, save_evaluation_artifacts
from src.paths import MODELS_DIR, RESULTS_DIR, ensure_project_directories
from src.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Build both classifiers with identical leakage-safe TF-IDF settings."""
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "naive_bayes": Pipeline(
            [("tfidf", _tfidf()), ("classifier", MultinomialNB(alpha=1.0))]
        ),
    }


def _positive_probability(model: Pipeline, texts: object) -> object:
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(texts)[:, classes.index(1)]


def train(force_download: bool = False) -> dict[str, object]:
    """Run the full reproducible training and held-out evaluation workflow."""
    ensure_project_directories()
    splits, dataset_stats = prepare_dataset(force_download=force_download)
    x_train = splits["train"]["statement"]
    y_train = splits["train"]["binary_label"]
    x_validation = splits["validation"]["statement"]
    y_validation = splits["validation"]["binary_label"]
    x_test = splits["test"]["statement"]
    y_test = splits["test"]["binary_label"]

    pipelines = build_pipelines()
    all_metrics: dict[str, dict[str, object]] = {}
    curves: dict[str, tuple[object, object]] = {}
    artifact_names = {
        "logistic_regression": "logistic_regression.joblib",
        "naive_bayes": "naive_bayes.joblib",
    }

    for model_key, pipeline in pipelines.items():
        LOGGER.info("Training %s on %d samples", model_key, len(x_train))
        pipeline.fit(x_train, y_train)
        validation_prediction = pipeline.predict(x_validation)
        validation_probability = _positive_probability(pipeline, x_validation)
        test_prediction = pipeline.predict(x_test)
        test_probability = _positive_probability(pipeline, x_test)
        all_metrics[model_key] = {
            "validation": evaluate_predictions(
                y_validation, validation_prediction, validation_probability
            ),
            "test": evaluate_predictions(y_test, test_prediction, test_probability),
        }
        curves[model_key] = (y_test, test_probability)
        joblib.dump(pipeline, MODELS_DIR / artifact_names[model_key], compress=3)

    # Stable tie break: Logistic Regression is listed first and wins exact ties.
    selected_model = max(
        pipelines,
        key=lambda key: (all_metrics[key]["validation"]["macro_f1"], key == "logistic_regression"),
    )
    shutil.copy2(MODELS_DIR / artifact_names[selected_model], MODELS_DIR / "best_model.joblib")
    save_evaluation_artifacts(all_metrics, curves, RESULTS_DIR, selected_model)

    selected_pipeline = pipelines[selected_model]
    classifier = selected_pipeline.named_steps["classifier"]
    metadata = {
        "project": "VeriNews AI",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "selected_model": selected_model,
        "selected_model_display_name": (
            "Logistic Regression" if selected_model == "logistic_regression" else "Multinomial Naive Bayes"
        ),
        "selection_criterion": "Highest validation macro F1; Logistic Regression wins exact ties.",
        "class_mapping": {str(key): value for key, value in BINARY_LABEL_NAMES.items()},
        "classifier_classes": [int(value) for value in classifier.classes_],
        "training_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(x_test),
        "tfidf": {
            "ngram_range": [1, 2],
            "max_features": 20_000,
            "min_df": 2,
            "sublinear_tf": True,
            "lowercase": False,
            "fitted_vocabulary_size": len(selected_pipeline.named_steps["tfidf"].vocabulary_),
        },
        "model_hyperparameters": {
            "logistic_regression": pipelines["logistic_regression"].named_steps["classifier"].get_params(),
            "naive_bayes": pipelines["naive_bayes"].named_steps["classifier"].get_params(),
        },
        "software_versions": {
            package: version(package)
            for package in ("scikit-learn", "pandas", "numpy", "joblib")
        },
        "dataset_summary": {
            "source": dataset_stats["source"],
            "processed_split_counts": dataset_stats["processed_split_counts"],
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "selected_model": selected_model,
        "metrics": all_metrics,
        "dataset_statistics": dataset_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download", action="store_true", help="Ignore processed CSVs and reacquire LIAR."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = train(force_download=args.force_download)
    except Exception:
        LOGGER.exception("Training failed")
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

