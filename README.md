# VeriNews AI

**Political Fake News and Misinformation Detection Using Natural Language Processing and Explainable AI**

VeriNews AI is a reproducible university NLP project that classifies the linguistic patterns of short English political statements as **FAKE** or **REAL**. It uses locally trained TF-IDF pipelines, compares Logistic Regression with Multinomial Naive Bayes, and explains individual predictions with LIME.

> **Important:** A result is a model prediction, not independently verified factual truth. The system learns correlations in the LIAR dataset and must not be used as a definitive fact-checking service.

## Academic information

- University: University of Karachi
- Department: UBIT
- Course: Natural Language Processing
- Submission deadline: 25 September 2026
- Syed Haris Mujtaba — EB23210006029
- Sarim Ali Khan — EB23210006116
- Arhama Musharraf — EB23210006017

## Implemented features

- Programmatic LIAR acquisition with a Hugging Face-first strategy and an automatic fallback to the dataset authors' TSV archive
- Validation of the actual six labels before binary conversion
- Preservation of the official train, validation, and test partitions
- Shared deterministic preprocessing for training and inference
- Leakage-safe `scikit-learn` pipelines containing TF-IDF and each classifier
- Logistic Regression and Multinomial Naive Bayes training
- Validation macro-F1 model selection with a documented tie-break
- Test accuracy, precision, recall, F1, macro/weighted F1, classification reports, confusion matrices, and ROC-AUC
- Saved model pipelines, metadata, JSON/CSV results, and PNG charts
- Real local LIME explanations against the deployed classifier
- Four-page styled Streamlit application with analysis, performance, workflow, and project information
- Held-out demonstration inputs and automated tests

## Dataset and label mapping

[LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) is a public benchmark of 12,791 labeled political statements. The loader first tries the Hugging Face aliases `ucsbnlp/liar` and `liar`. With current `datasets` versions, the repository's legacy Python loading script is unsupported, so the tested run used the original archive hosted by the dataset authors.

The original labels are mapped as follows:

| Binary class | ID | Original LIAR labels |
|---|---:|---|
| FAKE | 0 | `pants-fire`, `false`, `barely-true` |
| REAL | 1 | `mostly-true`, `true` |
| Excluded | — | `half-true` |

`half-true` is excluded because mixed factual accuracy is not cleanly represented by a binary target. This grouping is an experimental simplification, not a claim that all statements in one group have identical factual status.

The tested processed counts are 8,126 training, 1,036 validation, and 1,002 test statements (10,164 total after exclusion).

## Architecture

```text
English political statement
          ↓
Deterministic preprocessing
          ↓
TF-IDF (unigrams + bigrams, at most 20,000 features)
          ↓
Selected probabilistic classifier
          ↓
FAKE / REAL probabilities + LIME local explanation
```

Preprocessing removes URLs and non-text formatting, lowercases text, and normalizes whitespace. Stop words are not blindly removed; important negations such as “not,” “never,” and “no” remain available to the model. The vectorizer is fitted only on training data.

## Project structure

```text
app.py                         Streamlit application
src/dataset.py                 Acquisition, validation, mapping, and statistics
src/preprocessing.py           Shared deterministic text normalization
src/train.py                   End-to-end training and model selection
src/evaluate.py                Metrics, tables, confusion matrices, ROC curves
src/predict.py                 Artifact loading, validation, inference
src/explain.py                 LIME integration
data/processed/                Processed official splits and demo examples
models/                        Trained pipelines and metadata
results/                       Actual metrics and charts
tests/                         Automated tests
docs/IMPLEMENTATION_SUMMARY.md Detailed record for the report and viva
```

## Windows installation

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, it is not necessary to activate the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

The following command acquires LIAR, trains both models, selects the deployment model on validation macro-F1, evaluates both models once on the held-out test split, and writes all artifacts:

```powershell
python -m src.train
```

Force dataset reacquisition when needed:

```powershell
python -m src.train --force-download
```

The application never retrains during startup or prediction. It loads `models/best_model.joblib`.

## Run the application

```powershell
streamlit run app.py
```

Without environment activation:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Run tests

```powershell
python -m pytest -q
```

The suite checks preprocessing, both trainable pipeline definitions, ordered probability output, validation errors, and inference from the saved deployment artifact.

## Explainable AI

`LimeTextExplainer` perturbs words in the submitted text and repeatedly calls the deployed pipeline's real `predict_proba` method. The class-name list is created from the fitted classifier's `classes_` order. The displayed signed feature weights describe a local approximation of that prediction only; they are not factual evidence or global model rules.

## Results from the verified run

| Model | Validation macro-F1 | Test accuracy | Test F1 (REAL) | Test macro-F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6566 | 0.6517 | 0.6219 | 0.6495 | 0.7013 |
| Multinomial Naive Bayes | 0.6153 | 0.6497 | 0.5483 | 0.6311 | 0.6918 |

The selected deployment model is **Logistic Regression**. Binary precision, recall, and F1 treat **REAL (1)** as the positive class. Full-precision values and classification reports are in `results/evaluation_metrics.json`.

## Limitations

- LIAR consists mainly of short English-language U.S. political statements.
- The model may not generalize to Pakistani politics, Urdu, recent events, satire, or full-length articles.
- Text-only classification lacks sources, dates, evidence, speaker context, and live knowledge.
- Dataset labels and the project’s binary grouping contain unavoidable ambiguity.
- LIME explanations are local approximations and can vary with perturbation settings.
- A probability is model confidence under this learned decision function, not the probability that a claim is objectively true.

## Troubleshooting

- **“The trained model is missing”** — run `python -m src.train` from the repository root.
- **Dataset download fails** — check internet access and retry with `--force-download`. The loader reports errors from both supported sources instead of creating synthetic data.
- **PowerShell activation is blocked** — call `.venv\Scripts\python.exe` directly as shown above.
- **Incompatible model artifact** — reinstall the pinned-compatible requirements and retrain; serialized scikit-learn objects are version-sensitive.
- **Port already in use** — run `streamlit run app.py --server.port 8502`.
- **LIME fails on one statement** — classification remains available and the UI presents a readable explanation warning.

