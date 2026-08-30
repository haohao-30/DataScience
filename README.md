# Daily Gold Price H1–H7 Forecasting App

Submission-ready Streamlit application for comparing four already-finalized regression models—Ridge, KNN, SVR and XGBoost—and producing direct cumulative return forecasts for gold prices.

## What the app predicts

Each fitted horizon Pipeline predicts a cumulative return independently for H1 through H7. Here, H1–H7 mean **recorded observations ahead**, not guaranteed calendar days. Prices are reconstructed in the dataset's original units:

```text
Predicted Price Hh = Current Price × (1 + Predicted Return Hh)
```

H2–H7 do not use earlier forecasts recursively.

## App modes

- **Existing Evaluation Date:** replays the saved leakage-safe walk-forward predictions for one of the 609 common Evaluation origins. It never calls the all-data deployment models. Actual outcomes are labelled as values revealed after prediction.
- **Manual Input:** accepts exactly seven known fields and sends one verified 22-predictor row to each selected saved deployment Pipeline. No model or scaler is fitted in the app.

The featured result always identifies the official comparison winner from `comparison/Model_Comparison_Configuration.json`. The historical comparison remains based on saved Evaluation evidence, not current manual inputs.

## Feature construction

The user enters Current Price, Open, High, Low, Volume, Price Lag 1 and Price Lag 2. The app creates the remaining 15 predictors using definitions verified numerically against multiple canonical rows, including Evaluation rows:

- percentage-change Current CHG and Return Lags 1–6;
- inclusive 7- and 30-observation rolling price means;
- 7- and 30-observation rolling return sample standard deviations (`ddof=1`);
- 7- and 30-observation price momentum;
- the latest stored lagged USD Index return and US 10Y real-yield change.

The two external values are coursework values, not live data. Startup stops if parity with the canonical predictors fails.

## Leakage safety

The app does not train, tune, resplit, shuffle or regenerate Evaluation predictions. Historical mode only reads saved expanding walk-forward CSVs. Manual mode only predicts with final fitted deployment Pipelines. Targets never enter the manual feature row.

## Folder structure

```text
Daily_Gold_Price_Streamlit_App/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── data/
│   └── Gold_Price_Final_ModelData_External_Return_H1_H7.csv
├── models/
│   ├── Ridge_H1_H7_Deployment.joblib
│   ├── KNN_H1_H7_Deployment.joblib
│   ├── SVR_H1_H7_Deployment.joblib
│   └── XGBoost_H1_H7_Deployment.joblib
├── evidence/
│   ├── predictions/     # four saved walk-forward CSVs
│   ├── metrics/         # four saved metric CSVs
│   └── configurations/  # four model JSON files
├── comparison/
│   ├── Model_Comparison_Metrics.csv
│   ├── Model_Ranking.csv
│   └── Model_Comparison_Configuration.json
├── src/
│   ├── charts.py
│   ├── constants.py
│   ├── contracts.py
│   ├── data_access.py
│   ├── feature_builder.py
│   ├── prediction.py
│   └── xgb_portable.py
└── tests/test_app_contracts.py
```

## Local installation and run

Use Python 3.12 or 3.13 from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run verification first when desired:

```bash
python -m pytest -q
```

## GitHub and Streamlit Community Cloud deployment

1. Create a private or public GitHub repository and upload this folder without changing its internal paths.
2. Confirm all four `.joblib` files are committed. The largest is below GitHub's ordinary 100 MB file limit, so Git LFS is not required.
3. In Streamlit Community Cloud, choose the repository, branch and `app.py` as the entry point.
4. Deploy. Streamlit installs the pinned packages from `requirements.txt`.
5. Confirm that the page shows `Startup validation: PASS` before using prediction controls.

The supplied XGBoost artifact was originally created in Colab/Linux. Its fitted model-only tree bytes were placed in a small prediction-only cross-platform wrapper so the same saved trees load on Windows and Linux. This was a serialization conversion only—no fitting, retuning or data change occurred. Exact prediction equality before and after re-saving is tested during conversion.

## Limitations

- Educational forecasting prototype; not financial advice.
- H1–H7 are recorded observations, not guaranteed calendar days.
- No live data is fetched; manual external predictors use latest stored coursework values.
- Deployment forecasts can be weakened by regime shift.
- Forecasts are uncertain and are not guaranteed prices.
- Historical and manual modes use different valid model states and may give different outputs for otherwise similar visible fields.
