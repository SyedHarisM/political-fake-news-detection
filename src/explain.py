"""Local LIME explanations for the exact deployed pipeline."""

from __future__ import annotations

from typing import Any

from lime.lime_text import LimeTextExplainer

from src.dataset import RANDOM_SEED


def explain_prediction(
    text: str,
    model: Any,
    num_features: int = 10,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """Explain a prediction, preserving the fitted classifier's class ordering."""
    classifier_classes = [int(value) for value in model.named_steps["classifier"].classes_]
    class_names = ["FAKE" if value == 0 else "REAL" for value in classifier_classes]
    predicted_class = int(model.predict([text])[0])
    class_position = classifier_classes.index(predicted_class)
    explainer = LimeTextExplainer(class_names=class_names, random_state=RANDOM_SEED)
    explanation = explainer.explain_instance(
        text,
        model.predict_proba,
        labels=(class_position,),
        num_features=num_features,
        num_samples=num_samples,
    )
    contributions = [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in explanation.as_list(label=class_position)
    ]
    return {
        "predicted_class": predicted_class,
        "explained_class_name": class_names[class_position],
        "contributions": contributions,
        "intercept": float(explanation.intercept[class_position]),
        "local_prediction": float(explanation.local_pred[0]),
    }
