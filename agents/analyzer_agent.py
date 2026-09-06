"""
Analyzer Agent: runs the Task-3 predictive models and quantitative
portfolio analytics (diversification, risk concentration) over a
user-supplied portfolio (list of {ticker, weight}).
"""
import os
import pickle
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)
SAVED_DIR = os.path.join(BASE, "..", "models", "saved")
RAW_DIR = os.path.join(BASE, "..", "data", "raw")


class AnalyzerAgent:
    def __init__(self):
        with open(os.path.join(SAVED_DIR, "return_model.pkl"), "rb") as f:
            self.return_model = pickle.load(f)
        with open(os.path.join(SAVED_DIR, "direction_model.pkl"), "rb") as f:
            self.direction_model = pickle.load(f)
        self.risk_table = pd.read_csv(os.path.join(SAVED_DIR, "risk_clusters.csv"))
        self.fundamentals = pd.read_csv(os.path.join(RAW_DIR, "fundamentals.csv"))
        self.prices = pd.read_csv(os.path.join(RAW_DIR, "stock_prices.csv"), parse_dates=["Date"])
        self.macro = pd.read_csv(os.path.join(RAW_DIR, "macro_indicators.csv"), parse_dates=["date"])

    def _latest_features(self, ticker):
        p = self.prices[self.prices["ticker"] == ticker].sort_values("Date").copy()
        p["daily_return"] = p["Close"].pct_change()
        vol = p["daily_return"].rolling(20).std().iloc[-1]
        mom = p["Close"].pct_change(20).iloc[-1]
        fund = self.fundamentals[self.fundamentals["ticker"] == ticker].iloc[0]
        macro_row = self.macro.sort_values("date").iloc[-1]
        return {
            "volatility_20d": vol, "momentum_20d": mom, "pe_ratio": fund["pe_ratio"],
            "beta": fund["beta"], "debt_equity": fund["debt_equity"], "roi": fund["roi"],
            "interest_rate": macro_row["interest_rate"], "gdp_growth": macro_row["gdp_growth"],
            "cpi_inflation": macro_row["cpi_inflation"],
        }

    def predict_ticker(self, ticker):
        feats = pd.DataFrame([self._latest_features(ticker)])
        pred_return = float(self.return_model.predict(feats)[0])
        pred_up_prob = float(self.direction_model.predict_proba(feats)[0][1])
        risk_row = self.risk_table[self.risk_table["ticker"] == ticker]
        risk_label = risk_row["risk_label"].iloc[0] if len(risk_row) else "Unknown"
        return {
            "ticker": ticker,
            "predicted_5d_return": round(pred_return, 4),
            "prob_price_up": round(pred_up_prob, 3),
            "risk_label": risk_label,
        }

    def analyze_portfolio(self, portfolio):
        """portfolio: list of {'ticker': str, 'weight': float (0-1)}"""
        rows = []
        for holding in portfolio:
            t = holding["ticker"]
            pred = self.predict_ticker(t)
            fund = self.fundamentals[self.fundamentals["ticker"] == t].iloc[0]
            rows.append({**pred, "weight": holding["weight"], "sector": fund["sector"],
                          "beta": fund["beta"]})
        df = pd.DataFrame(rows)

        weighted_return = float((df["predicted_5d_return"] * df["weight"]).sum())
        weighted_beta = float((df["beta"] * df["weight"]).sum())
        sector_weights = df.groupby("sector")["weight"].sum().to_dict()
        risk_weights = df.groupby("risk_label")["weight"].sum().to_dict()

        # Simple diversification gap: sectors present vs. all known sectors, and any
        # single-sector overweight above 40%.
        overweight_sectors = {s: w for s, w in sector_weights.items() if w > 0.4}

        return {
            "holdings": df.to_dict(orient="records"),
            "portfolio_predicted_5d_return": round(weighted_return, 4),
            "portfolio_beta": round(weighted_beta, 3),
            "sector_allocation": sector_weights,
            "risk_allocation": risk_weights,
            "diversification_gaps": overweight_sectors,
        }
