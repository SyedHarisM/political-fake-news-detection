"""Safe artifact loading and prediction helpers for UI and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.paths import MODELS_DIR
from src.preprocessing import preprocess_text

CLASS_NAMES = {0: "FAKE", 1: "REAL"}


class PredictionError(RuntimeError):
    """A user-safe prediction or artifact error."""


def validate_input(text: Any, minimum_characters: int = 12) -> str:
    cleaned = preprocess_text(text)
    if not cleaned:
        raise PredictionError("Please enter a political statement or headline.")
    if len(cleaned) < minimum_characters or len(cleaned.split()) < 3:
        raise PredictionError("Please enter a more complete statement (at least 3 words and 12 characters).")
    return str(text).strip()


def load_model(model_path: Path | None = None) -> Any:
    path = model_path or MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise PredictionError("The trained model is missing. Run `python -m src.train` first.")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise PredictionError("The saved model could not be loaded. Retrain the artifacts.") from exc
    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise PredictionError("The saved artifact is not a compatible probabilistic classifier.")
    return model


def load_metadata() -> dict[str, Any]:
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def predict_statement(text: Any, model: Any | None = None) -> dict[str, Any]:
    original_text = validate_input(text)
    pipeline = model or load_model()
    try:
        predicted_id = int(pipeline.predict([original_text])[0])
        probability_values = pipeline.predict_proba([original_text])[0]
        classifier = pipeline.named_steps["classifier"]
        classes = [int(value) for value in classifier.classes_]
    except Exception as exc:
        raise PredictionError("The model could not analyze this statement.") from exc
    probabilities = {
        CLASS_NAMES[class_id]: float(probability_values[index])
        for index, class_id in enumerate(classes)
    }
    if predicted_id not in CLASS_NAMES:
        raise PredictionError(f"The model returned an unsupported class: {predicted_id}")
    confidence = probabilities[CLASS_NAMES[predicted_id]]
    if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-6):
        raise PredictionError("The model returned invalid probability values.")
    return {
        "label_id": predicted_id,
        "label": CLASS_NAMES[predicted_id],
        "confidence": confidence,
        "probabilities": probabilities,
        "normalized_text": preprocess_text(original_text),
    }

