"""VeriNews AI — Streamlit interface for political statement classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.explain import explain_prediction
from src.paths import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.predict import PredictionError, load_metadata, load_model, predict_statement

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="VeriNews AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root { --navy:#071a33; --blue:#2563eb; --cyan:#38bdf8; --ink:#172033; }
    .stApp { background: linear-gradient(180deg,#f4f7fb 0%,#ffffff 60%); color:var(--ink); }
    [data-testid="stSidebar"] { background: #071a33; }
    [data-testid="stSidebar"] * { color: #eef6ff !important; }
    [data-testid="stSidebar"] .stRadio label { padding: .35rem 0; }
    .hero { background: linear-gradient(125deg,#071a33,#123b70 62%,#1769aa); color:white;
            padding:2.15rem 2.4rem; border-radius:22px; margin-bottom:1.3rem;
            box-shadow:0 16px 40px rgba(7,26,51,.18); }
    .hero h1 { font-size:2.65rem; margin:0 0 .35rem; letter-spacing:-.04em; }
    .hero p { margin:0; color:#dbeafe; font-size:1.08rem; }
    .eyebrow { color:#7dd3fc; font-size:.78rem; letter-spacing:.13em; font-weight:750; }
    .card { background:white; color:var(--ink); border:1px solid #dfe7f1; border-radius:18px; padding:1.25rem 1.35rem;
            box-shadow:0 8px 24px rgba(15,23,42,.06); margin:.45rem 0 1rem; }
    .result-real { background:#ecfdf5; border:1px solid #a7f3d0; border-radius:18px; padding:1.3rem; }
    .result-fake { background:#fff1f2; border:1px solid #fecdd3; border-radius:18px; padding:1.3rem; }
    .result-title { font-size:2rem; font-weight:800; letter-spacing:-.03em; }
    .muted { color:#526176; font-size:.92rem; }
    .pipeline { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:1rem 0; }
    .step { background:#eaf2ff; border:1px solid #bfdbfe; color:#123b70; padding:.62rem .82rem;
            border-radius:12px; font-weight:650; }
    .arrow { color:#64748b; font-weight:700; }
    div.stButton > button { background:#2563eb; color:white; border-radius:11px; border:0;
                           font-weight:700; padding:.65rem 1.2rem; }
    div.stButton > button:hover { background:#1d4ed8; color:white; }
    [data-testid="stMetric"] { background:white; border:1px solid #e2e8f0; padding:1rem;
                              border-radius:14px; box-shadow:0 5px 15px rgba(15,23,42,.04); }
    [data-testid="stWidgetLabel"] p { color:var(--ink); font-weight:650; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></section>',
        unsafe_allow_html=True,
    )


def probability_chart(probabilities: dict[str, float]) -> None:
    frame = pd.DataFrame(
        {"Class": list(probabilities), "Probability": [value * 100 for value in probabilities.values()]}
    )
    fig, axis = plt.subplots(figsize=(7, 2.25))
    colors = ["#e11d48" if name == "FAKE" else "#059669" for name in frame["Class"]]
    bars = axis.barh(frame["Class"], frame["Probability"], color=colors, height=0.52)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Estimated probability (%)")
    axis.grid(axis="x", alpha=.18)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, frame["Probability"]):
        axis.text(min(value + 1.2, 92), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", weight="bold")
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def explanation_chart(contributions: list[dict], predicted_label: str) -> None:
    if not contributions:
        st.info("LIME did not identify stable word-level contributions for this input.")
        return
    frame = pd.DataFrame(contributions).sort_values("weight")
    fig, axis = plt.subplots(figsize=(8, max(3.3, len(frame) * .42)))
    colors = ["#2563eb" if value > 0 else "#f59e0b" for value in frame["weight"]]
    axis.barh(frame["feature"], frame["weight"], color=colors)
    axis.axvline(0, color="#64748b", linewidth=.8)
    axis.set_xlabel(f"Contribution toward/away from explained class: {predicted_label}")
    axis.set_ylabel("Word or phrase")
    axis.grid(axis="x", alpha=.15)
    axis.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.caption(
        "Blue bars support the explained class; amber bars push away from it. "
        "LIME is a local approximation of this one prediction, not factual evidence."
    )


def analyze_page() -> None:
    hero(
        "UBIT · NATURAL LANGUAGE PROCESSING",
        "VeriNews AI",
        "Political Fake News and Misinformation Detection Using NLP",
    )
    st.markdown(
        '<div class="card"><b>What this tool does</b><br><span class="muted">'
        "It compares the linguistic patterns in an English political statement with patterns learned from "
        "the LIAR benchmark. It is a classifier—not an independent fact-checking service.</span></div>",
        unsafe_allow_html=True,
    )

    examples_path = PROCESSED_DIR / "held_out_examples.csv"
    example_text = ""
    if examples_path.exists():
        try:
            examples = pd.read_csv(examples_path)
            options = ["Write my own statement"] + [
                f"Held-out example {index + 1} · {row.original_label}"
                for index, row in examples.head(6).iterrows()
            ]
            selected = st.selectbox("Optional demonstration input", options)
            if selected != options[0]:
                example_text = str(examples.iloc[options.index(selected) - 1]["statement"])
                st.caption("The original LIAR label is shown in the selector; the statement was not used for training.")
        except Exception:
            LOGGER.exception("Could not load demo examples")

    statement = st.text_area(
        "Political statement or headline",
        value=example_text,
        height=155,
        placeholder="Paste or type an English political statement here…",
        help="For meaningful results, enter a complete statement rather than one or two words.",
    )
    analyze = st.button("Analyze statement", type="primary", use_container_width=False)
    if not analyze:
        st.markdown(
            '<div class="pipeline"><span class="step">Input text</span><span class="arrow">→</span>'
            '<span class="step">Preprocessing</span><span class="arrow">→</span>'
            '<span class="step">TF-IDF</span><span class="arrow">→</span>'
            '<span class="step">Classifier</span><span class="arrow">→</span>'
            '<span class="step">LIME explanation</span></div>',
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Analyzing linguistic patterns…"):
            model = get_model()
            prediction = predict_statement(statement, model=model)
        result_class = "result-real" if prediction["label"] == "REAL" else "result-fake"
        icon = "✓" if prediction["label"] == "REAL" else "!"
        st.markdown(
            f'<div class="{result_class}"><div class="muted">MODEL PREDICTION</div>'
            f'<div class="result-title">{icon} {prediction["label"]}</div>'
            f'<b>{prediction["confidence"]:.1%} estimated probability for the predicted class</b></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([.9, 1.1], gap="large")
        with left:
            st.subheader("Class probabilities")
            probability_chart(prediction["probabilities"])
        with right:
            st.subheader("Local explanation")
            try:
                with st.spinner("Generating LIME explanation…"):
                    explanation = explain_prediction(statement, model)
                explanation_chart(explanation["contributions"], explanation["explained_class_name"])
            except Exception:
                LOGGER.exception("LIME explanation failed")
                st.warning("The classification succeeded, but a local explanation could not be generated for this input.")
        st.warning(
            "This output reflects learned language patterns and is not independently verified factual truth. "
            "Consult reliable primary sources and professional fact-checkers before drawing conclusions."
        )
    except PredictionError as exc:
        st.error(str(exc))
    except Exception:
        LOGGER.exception("Unexpected analysis failure")
        st.error("The analysis could not be completed. Check the model artifacts and application log.")


def performance_page() -> None:
    hero("HELD-OUT EVALUATION", "Model performance", "Transparent results loaded from the saved experiment artifacts")
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    stats_path = RESULTS_DIR / "dataset_statistics.json"
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if not all(path.exists() for path in (metrics_path, stats_path, comparison_path)):
        st.error("Evaluation artifacts are missing. Run `python -m src.train` from the project root.")
        return
    try:
        metrics = read_json(metrics_path)
        stats = read_json(stats_path)
        comparison = pd.read_csv(comparison_path)
    except Exception:
        LOGGER.exception("Could not read evaluation artifacts")
        st.error("The saved evaluation results are incompatible or damaged. Retrain the project artifacts.")
        return

    counts = stats["processed_split_counts"]
    cols = st.columns(4)
    cols[0].metric("Training statements", f"{counts['train']:,}")
    cols[1].metric("Validation statements", f"{counts['validation']:,}")
    cols[2].metric("Test statements", f"{counts['test']:,}")
    cols[3].metric("Deployed model", metrics["selected_model"].replace("_", " ").title())

    st.subheader("Model comparison")
    display_columns = [
        "model", "validation_macro_f1", "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_macro_f1", "test_roc_auc", "selected_for_deployment",
    ]
    styled = comparison[display_columns].rename(columns=lambda value: value.replace("_", " ").title())
    numeric = styled.select_dtypes(include="number").columns
    st.dataframe(styled.style.format({column: "{:.3f}" for column in numeric}), use_container_width=True, hide_index=True)
    st.caption(
        "Binary precision, recall, F1, and ROC-AUC use REAL (1) as the positive class. "
        "Deployment selection used validation macro-F1 only; the test set remained untouched until final evaluation."
    )

    st.subheader("Binary class distribution")
    distribution_rows = []
    for split, values in stats["binary_class_distributions"].items():
        for label, count in values.items():
            distribution_rows.append({"Split": split.title(), "Class": label, "Count": count})
    distribution = pd.DataFrame(distribution_rows).pivot(index="Split", columns="Class", values="Count")
    st.bar_chart(distribution, color=["#e11d48", "#059669"])

    st.subheader("Confusion matrices")
    left, right = st.columns(2)
    with left:
        st.image(str(RESULTS_DIR / "confusion_matrix_lr.png"), caption="Logistic Regression on held-out test data", use_container_width=True)
    with right:
        st.image(str(RESULTS_DIR / "confusion_matrix_nb.png"), caption="Multinomial Naive Bayes on held-out test data", use_container_width=True)
    st.subheader("Receiver operating characteristic")
    st.image(str(RESULTS_DIR / "roc_curve.png"), caption="ROC curves on the untouched test split", use_container_width=True)

    with st.expander("Dataset mapping and methodology"):
        st.markdown(
            "- **FAKE (0):** pants-fire, false, barely-true\n"
            "- **REAL (1):** mostly-true, true\n"
            "- **Excluded:** half-true, because it is not cleanly binary\n\n"
            "Official LIAR train/validation/test partitions were preserved. TF-IDF was fit only on training text. "
            "Both classifiers used identical partitions; model choice was based on validation macro-F1."
        )


def how_it_works_page() -> None:
    hero("SYSTEM ARCHITECTURE", "How it works", "From raw language to an interpretable local prediction")
    st.markdown(
        '<div class="pipeline"><span class="step">1 · Input text</span><span class="arrow">→</span>'
        '<span class="step">2 · Normalize</span><span class="arrow">→</span>'
        '<span class="step">3 · TF-IDF</span><span class="arrow">→</span>'
        '<span class="step">4 · Classify</span><span class="arrow">→</span>'
        '<span class="step">5 · Explain with LIME</span></div>', unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Preprocessing")
        st.write("URLs and formatting noise are removed, text is lowercased, and whitespace is normalized. Negations are retained because they can change meaning.")
        st.markdown("### TF-IDF representation")
        st.write("Unigrams and bigrams become weighted numeric features. Terms receive more weight when important to a statement but less common across the training corpus.")
        st.markdown("### Logistic Regression")
        st.write("A linear probabilistic classifier that learns a weight for each text feature and combines those weights to estimate class probabilities.")
    with right:
        st.markdown("### Multinomial Naive Bayes")
        st.write("A probabilistic text classifier that models how strongly TF-IDF features are associated with each class under a conditional-independence assumption.")
        st.markdown("### Model selection")
        st.write("Both models are compared on validation macro-F1, giving both classes equal importance. The final test split is used only after selection.")
        st.markdown("### LIME")
        st.write("LIME perturbs words in the submitted statement, observes changes in the real model's probabilities, and fits a small local approximation. It explains model behavior, not factual truth.")


def about_page() -> None:
    hero("UNIVERSITY OF KARACHI", "About the project", "A final-year BSCS Natural Language Processing course prototype")
    st.markdown("### Political Fake News and Misinformation Detection Using Natural Language Processing and Explainable AI")
    university, project = st.columns(2, gap="large")
    with university:
        st.markdown(
            '<div class="card"><b>Academic context</b><br><br>University of Karachi<br>'
            'Department: UBIT<br>Course: Natural Language Processing<br>Submission: 25 September 2026</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Group members")
        st.write("Syed Haris Mujtaba — EB23210006029")
        st.write("Sarim Ali Khan — EB23210006116")
        st.write("Arhama Musharraf — EB23210006017")
    with project:
        st.markdown("#### Purpose and scope")
        st.write("This educational prototype demonstrates preprocessing, TF-IDF feature engineering, supervised text classification, held-out evaluation, and local explainability on a public political-statements benchmark.")
        st.markdown("#### Technologies")
        st.write("Python, pandas, scikit-learn, LIAR, Logistic Regression, Multinomial Naive Bayes, LIME, Matplotlib, Seaborn, and Streamlit.")
        st.markdown("#### Important limitations")
        st.warning("The classifier is trained primarily on short English-language U.S. political statements from LIAR. It may not generalize reliably to Pakistani politics, Urdu, recent events, satire, or full-length news articles. A prediction is not a fact-check.")


def main() -> None:
    with st.sidebar:
        st.markdown("## ◈ VeriNews AI")
        st.caption("Explainable political-text classification")
        page = st.radio(
            "Navigate",
            ("Analyze", "Model Performance", "How It Works", "About the Project"),
            label_visibility="collapsed",
        )
        metadata = load_metadata()
        if metadata:
            st.divider()
            st.caption(f"Deployed: {metadata.get('selected_model_display_name', 'trained model')}")
            st.caption(f"Vocabulary: {metadata.get('tfidf', {}).get('fitted_vocabulary_size', 0):,} features")
    pages = {
        "Analyze": analyze_page,
        "Model Performance": performance_page,
        "How It Works": how_it_works_page,
        "About the Project": about_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
