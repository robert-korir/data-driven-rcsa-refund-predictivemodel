# Data-Driven Multi-KRI RCSA Streamlit App

## Repository files

Place these files in the root of the GitHub repository:

- `app.py`
- `requirements.txt`
- `Final_Multi_KRI_Predictive_Model.pkl`
- `RCSA_MultiKRI_Configuration.json`
- `sample_multi_kri_input.csv` (optional test file)

## Streamlit Cloud deployment

1. Push all files above to the same GitHub repository and branch.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Input CSV

The app expects these columns:

- `date`
- `KRI3_waiting_refunds_after_14_days`
- `KRI4_chargebacks`
- `KRI5_historical_refund_operations`

KRI3 may be blank on non-snapshot dates. KRI4 and KRI5 need consecutive daily observations for the predictive model.

## Model

Selected predictive algorithm: Ridge Regression.

Prediction target: next-day Historical Refund Operations.

The included model was created with scikit-learn 1.8.0. Keep the same scikit-learn version in `requirements.txt` when deploying this model artifact.

## Proof-of-concept thresholds

KRI3 — Waiting Refunds After 14 Days:
- Q1 = 35.50
- Q3 = 45.00

KRI4 — Daily Chargebacks:
- Q1 = 194.00
- Q3 = 1534.75

For both current-risk KRIs:
- Green: value <= Q1
- Amber: Q1 < value <= Q3
- Red: value > Q3

Composite score:
- KRI3 weight = 0.50
- KRI4 weight = 0.50
- Green: score < 1.50
- Amber: 1.50 <= score < 2.50
- Red: score >= 2.50

These are proof-of-concept settings and require organizational validation before production use.
