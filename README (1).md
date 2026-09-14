# Breast Cancer Diagnosis Classification — with Explainable AI (SHAP)

## Clinical problem
Given measurements from a breast mass biopsy, predict whether the mass is malignant or benign. Framed as
a decision-support tool — it flags likely-malignant cases for priority review rather than replacing a
pathologist's judgment.

## Approach
1. **Data:** Breast Cancer Wisconsin (Diagnostic) dataset — real, public clinical data, 569 patients,
   30 features derived from digitized biopsy images (bundled with scikit-learn, fully reproducible).
2. **Models:** Logistic Regression (interpretable baseline) vs. Random Forest (captures non-linear
   patterns, at the cost of being a "black box" — which is why SHAP is applied to it).
3. **Evaluation with a clinical lens:** prioritizes **recall on the malignant class**, not just accuracy —
   a missed cancer (false negative) is far more costly than a false alarm.
4. **Explainability:** SHAP `TreeExplainer` for (a) global feature importance across all patients, and
   (b) a local force-plot explaining one individual patient's prediction — the piece a clinician actually
   needs to trust a specific call.

## Key result
Both models perform strongly; SHAP confirms the model relies on clinically sensible features (worst
concave points, worst perimeter, worst radius — consistent with known malignancy indicators), which is a
useful sanity check that it hasn't learned a spurious correlation.

## Files
- `healthcare_classification.ipynb` — full analysis notebook (code, metrics, SHAP plots, findings)
- `healthcare_classification.html` — static rendered version, viewable without Jupyter
- `requirements.txt` — Python dependencies

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, SHAP

## How to run
```bash
pip install -r requirements.txt
jupyter notebook healthcare_classification.ipynb
```

## Clinical caveats
This is a decision-support demo, not a diagnostic device. A real deployment would require regulatory
approval (e.g. TGA in Australia), validation on external hospital data, drift monitoring, and a
human-in-the-loop review process.

## Next steps
Calibrate predicted probabilities, test threshold-tuning to push recall further (trading off more false
positives for fewer missed cancers), and validate SHAP explanations with a clinical domain expert.
