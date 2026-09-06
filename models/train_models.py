"""
Task 3: Building Predictive Financial Models

Trains:
  1. Portfolio / stock return regressor (XGBoost, falls back to sklearn
     GradientBoostingRegressor if xgboost isn't installed)
  2. Stock movement classifier (up/down next period)
  3. Risk clustering model (KMeans over fundamentals -> risk buckets)

Saves trained models + a metrics report to models/saved/.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE, "..", "data", "raw")
SAVE_DIR = os.path.join(BASE, "saved")


def build_features():
    prices = pd.read_csv(os.path.join(RAW_DIR, "stock_prices.csv"), parse_dates=["Date"])
    fundamentals = pd.read_csv(os.path.join(RAW_DIR, "fundamentals.csv"))
    macro = pd.read_csv(os.path.join(RAW_DIR, "macro_indicators.csv"), parse_dates=["date"])

    prices = prices.sort_values(["ticker", "Date"])
    prices["daily_return"] = prices.groupby("ticker")["Close"].pct_change()
    prices["volatility_20d"] = prices.groupby("ticker")["daily_return"].transform(
        lambda s: s.rolling(20).std())
    prices["momentum_20d"] = prices.groupby("ticker")["Close"].transform(
        lambda s: s.pct_change(20))

    # forward 5-day return = prediction target
    prices["fwd_return_5d"] = prices.groupby("ticker")["Close"].transform(
        lambda s: s.shift(-5) / s - 1)
    prices["direction_up"] = (prices["fwd_return_5d"] > 0).astype(int)

    df = prices.merge(fundamentals, on="ticker", how="left")
    latest_macro = macro.sort_values("date").iloc[-1]
    for col in ["interest_rate", "gdp_growth", "cpi_inflation"]:
        df[col] = latest_macro[col]

    df = df.dropna(subset=["daily_return", "volatility_20d", "momentum_20d", "fwd_return_5d"])
    return df, fundamentals


FEATURE_COLS = ["volatility_20d", "momentum_20d", "pe_ratio", "beta", "debt_equity",
                 "roi", "interest_rate", "gdp_growth", "cpi_inflation"]


def train_return_model(df):
    X = df[FEATURE_COLS]
    y = df["fwd_return_5d"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAS_XGB:
        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return model, {"rmse": rmse, "n_test": len(y_test), "engine": "xgboost" if HAS_XGB else "sklearn-gbr"}


def train_direction_model(df):
    X = df[FEATURE_COLS]
    y = df["direction_up"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if HAS_XGB:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
                               eval_metric="logloss")
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))
    return model, {"accuracy": acc, "f1": f1, "n_test": len(y_test),
                    "engine": "xgboost" if HAS_XGB else "sklearn-gbc"}


def train_risk_clusters(fundamentals, k=3):
    feats = fundamentals[["pe_ratio", "beta", "debt_equity", "roi", "dividend_yield"]].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Order clusters by mean beta so label 0 = lowest risk, k-1 = highest
    order = np.argsort([feats["beta"][labels == c].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    risk_labels = [remap[l] for l in labels]

    result = fundamentals[["ticker"]].copy()
    result["risk_cluster"] = risk_labels
    risk_names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    result["risk_label"] = result["risk_cluster"].map(lambda c: risk_names.get(c, f"Cluster {c}"))
    return km, scaler, result


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Building feature set ...")
    df, fundamentals = build_features()

    print("Training return regression model ...")
    return_model, return_metrics = train_return_model(df)

    print("Training direction classification model ...")
    direction_model, direction_metrics = train_direction_model(df)

    print("Training risk clustering model ...")
    risk_model, risk_scaler, risk_table = train_risk_clusters(fundamentals)

    with open(os.path.join(SAVE_DIR, "return_model.pkl"), "wb") as f:
        pickle.dump(return_model, f)
    with open(os.path.join(SAVE_DIR, "direction_model.pkl"), "wb") as f:
        pickle.dump(direction_model, f)
    with open(os.path.join(SAVE_DIR, "risk_model.pkl"), "wb") as f:
        pickle.dump({"kmeans": risk_model, "scaler": risk_scaler}, f)
    risk_table.to_csv(os.path.join(SAVE_DIR, "risk_clusters.csv"), index=False)

    metrics = {"return_model": return_metrics, "direction_model": direction_metrics,
               "feature_cols": FEATURE_COLS}
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nModels + risk clusters saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
