from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Gold_Price_Final_ModelData_External_Return_H1_H7.csv"
MODEL_NAMES = ("Ridge", "KNN", "SVR", "XGBoost")
HORIZONS = tuple(range(1, 8))

MODEL_PATHS = {name: ROOT / "models" / f"{name}_H1_H7_Deployment.joblib" for name in MODEL_NAMES}
PREDICTION_PATHS = {
    name: ROOT / "evidence" / "predictions" / f"{name}_H1_H7_WalkForward_Predictions.csv"
    for name in MODEL_NAMES
}
METRIC_PATHS = {
    name: ROOT / "evidence" / "metrics" / f"{name}_H1_H7_Metrics.csv"
    for name in MODEL_NAMES
}
CONFIG_PATHS = {
    name: ROOT / "evidence" / "configurations" / f"{name}_H1_H7_Configuration.json"
    for name in MODEL_NAMES
}
COMPARISON_METRICS_PATH = ROOT / "comparison" / "Model_Comparison_Metrics.csv"
RANKING_PATH = ROOT / "comparison" / "Model_Ranking.csv"
COMPARISON_CONFIG_PATH = ROOT / "comparison" / "Model_Comparison_Configuration.json"

VISIBLE_FIELDS = (
    "Current_Price",
    "Current_Open",
    "Current_High",
    "Current_Low",
    "Current_Volume",
    "Price_Lag1",
    "Price_Lag2",
)

PREDICTORS = (
    "Current_Price", "Current_Open", "Current_High", "Current_Low", "Current_Volume",
    "Price_Lag1", "Price_Lag2", "Current_CHG", "Return_Lag1", "Return_Lag2",
    "Return_Lag3", "Return_Lag4", "Return_Lag5", "Return_Lag6", "MA_7", "MA_30",
    "Volatility_7", "Volatility_30", "Momentum_7", "Momentum_30",
    "USD_Index_Return_Lag1", "US10Y_Real_Yield_Change_Lag1",
)

PREDICTION_COLUMNS = (
    "Model", "Horizon", "Evaluation_Step", "Origin_Date", "Target_Date", "Current_Price",
    "Actual_Return", "Predicted_Return", "Actual_Price", "Predicted_Price",
    "Persistence_Return", "Persistence_Price", "Absolute_Price_Error", "Squared_Price_Error",
    "Actual_Direction", "Predicted_Direction", "Eligible_Training_Rows",
)

METRIC_COLUMNS = (
    "Model", "Horizon", "N", "Price_RMSE", "Price_MAE", "Price_MAPE_Percent",
    "Price_NRMSE_Range", "Price_R2", "Return_RMSE", "Return_MAE", "Return_R2",
    "Directional_Accuracy_Percent", "Always_Up_Accuracy_Percent", "Persistence_Price_RMSE",
    "Persistence_Price_MAE", "RMSE_Skill_vs_Persistence", "MAE_Skill_vs_Persistence",
    "Best_Params",
)

TARGET_PREFIXES = ("Target_",)

