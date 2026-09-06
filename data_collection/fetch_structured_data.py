"""
Task 1: Data Collection - Structured Data
Fetches stock prices, fundamentals, and ratios from Yahoo Finance / Financial
Modeling Prep. This is REAL client code -- it requires outbound internet
access and (for FMP) an API key. If those aren't available in your
environment, use `generate_synthetic_data.py` instead, which produces
data in the exact same schema so the rest of the pipeline is unaffected.

Usage:
    python fetch_structured_data.py --tickers AAPL MSFT GOOGL --out ../data/raw
"""
import argparse
import os
import sys
import json
import pandas as pd

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def fetch_yfinance(tickers, period="1y"):
    """Fetch historical prices + basic info via yfinance."""
    import yfinance as yf

    frames = []
    infos = []
    for t in tickers:
        tk = yf.Ticker(t)
        hist = tk.history(period=period).reset_index()
        hist["ticker"] = t
        frames.append(hist)

        info = tk.info or {}
        infos.append({
            "ticker": t,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
        })
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fundamentals = pd.DataFrame(infos)
    return prices, fundamentals


def fetch_fmp_fundamentals(tickers, api_key):
    """Fetch analyst ratings / fundamentals from Financial Modeling Prep."""
    import requests

    rows = []
    for t in tickers:
        url = f"{FMP_BASE_URL}/rating/{t}?apikey={api_key}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data:
            rows.append(data[0])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "JPM", "XOM"])
    parser.add_argument("--out", default="../data/raw")
    parser.add_argument("--fmp-key", default=os.environ.get("FMP_API_KEY"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Fetching Yahoo Finance data for {args.tickers} ...")
    prices, fundamentals = fetch_yfinance(args.tickers)
    prices.to_csv(os.path.join(args.out, "stock_prices.csv"), index=False)
    fundamentals.to_csv(os.path.join(args.out, "fundamentals_yf.csv"), index=False)

    if args.fmp_key:
        print("Fetching Financial Modeling Prep ratings ...")
        ratings = fetch_fmp_fundamentals(args.tickers, args.fmp_key)
        ratings.to_csv(os.path.join(args.out, "analyst_ratings.csv"), index=False)
    else:
        print("No FMP_API_KEY set -- skipping analyst ratings fetch.")

    print("Done. Files written to", args.out)


if __name__ == "__main__":
    main()
