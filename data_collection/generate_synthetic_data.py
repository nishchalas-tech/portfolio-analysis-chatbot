"""
Generates realistic-looking synthetic data in the SAME schema that
fetch_structured_data.py / fetch_unstructured_data.py would produce,
so the rest of the pipeline (embeddings, models, agents, UI) can be
built, tested, and demoed without live internet/API access.

Swap this out for the real fetch_*.py scripts once you have API keys
and outbound network access.
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "JPM", "XOM", "TSLA", "JNJ", "KO", "NVDA", "PG",
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "BAJFINANCE",
    "MARUTI", "ASIANPAINT", "SUNPHARMA", "TITAN", "WIPRO", "ADANIENT",
    "TATAMOTORS", "AXISBANK",
]
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "JPM": "Financials", "XOM": "Energy",
    "TSLA": "Consumer Discretionary", "JNJ": "Healthcare",
    "KO": "Consumer Staples", "PG": "Consumer Staples",
    "RELIANCE": "Energy", "TCS": "Technology", "HDFCBANK": "Financials",
    "INFY": "Technology", "ICICIBANK": "Financials", "HINDUNILVR": "Consumer Staples",
    "SBIN": "Financials", "BHARTIARTL": "Telecom", "ITC": "Consumer Staples",
    "KOTAKBANK": "Financials", "LT": "Industrials", "BAJFINANCE": "Financials",
    "MARUTI": "Consumer Discretionary", "ASIANPAINT": "Consumer Discretionary",
    "SUNPHARMA": "Healthcare", "TITAN": "Consumer Discretionary",
    "WIPRO": "Technology", "ADANIENT": "Industrials",
    "TATAMOTORS": "Consumer Discretionary", "AXISBANK": "Financials",
}

def generate_price_history(tickers, days=252):
    rows = []
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    for t in tickers:
        price = np.random.uniform(50, 400)
        vol = np.random.uniform(0.01, 0.035)
        for d in dates:
            price *= (1 + np.random.normal(0.0004, vol))
            rows.append({
                "Date": d, "ticker": t, "Close": round(price, 2),
                "Volume": int(np.random.uniform(1e6, 5e7)),
            })
    return pd.DataFrame(rows)


def generate_fundamentals(tickers):
    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "market_cap": round(np.random.uniform(20e9, 3e12), 0),
            "pe_ratio": round(np.random.uniform(8, 45), 2),
            "sector": SECTORS.get(t, "Other"),
            "beta": round(np.random.uniform(0.5, 1.8), 2),
            "dividend_yield": round(np.random.uniform(0, 0.04), 4),
            "debt_equity": round(np.random.uniform(0.1, 2.0), 2),
            "roi": round(np.random.uniform(-0.05, 0.30), 4),
        })
    return pd.DataFrame(rows)


def generate_macro():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=24, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "interest_rate": np.round(np.random.uniform(3.5, 5.5, len(dates)), 2),
        "gdp_growth": np.round(np.random.uniform(-1.0, 3.5, len(dates)), 2),
        "cpi_inflation": np.round(np.random.uniform(1.5, 6.0, len(dates)), 2),
    })


FIN_DOC_TEMPLATES = {
    "news": [
        "{t} shares moved after quarterly results showed {dir} revenue growth of {pct}% "
        "year-over-year, with analysts pointing to {theme} as a key driver.",
        "Market analysts raised concerns about {t}'s exposure to {theme}, citing "
        "{dir} margins and a P/E ratio that remains {level} relative to sector peers.",
    ],
    "filing": [
        "Item 1A Risk Factors: {t}'s business is subject to risks including {theme}, "
        "competitive pressure, and macroeconomic headwinds such as rising interest rates. "
        "Management believes diversification across product lines partially mitigates these risks.",
        "MD&A: {t} reported {dir} operating cash flow compared to the prior fiscal year, "
        "driven primarily by {theme} and disciplined cost management.",
    ],
    "blog": [
        "Investment strategy note: allocating to {t} can improve diversification for portfolios "
        "overweight in {theme}-sensitive sectors. A common approach is to pair it with lower-beta "
        "holdings to manage overall portfolio volatility.",
        "Risk management perspective: {t} carries {level} volatility historically, so position sizing "
        "and correlation with existing holdings matter more than the standalone return forecast.",
    ],
}
THEMES = ["supply chain disruption", "AI infrastructure spending", "consumer demand softness",
          "regulatory scrutiny", "energy price volatility", "interest rate sensitivity",
          "international expansion", "input cost inflation"]


def generate_financial_text_docs(tickers, out_dir, n_per_ticker=4):
    os.makedirs(out_dir, exist_ok=True)
    doc_id = 0
    for t in tickers:
        for _ in range(n_per_ticker):
            doc_type = np.random.choice(list(FIN_DOC_TEMPLATES.keys()))
            template = np.random.choice(FIN_DOC_TEMPLATES[doc_type])
            text = template.format(
                t=t,
                dir=np.random.choice(["strong", "weak", "modest", "declining"]),
                pct=round(np.random.uniform(-10, 25), 1),
                theme=np.random.choice(THEMES),
                level=np.random.choice(["elevated", "below-average", "in line with", "highly"]),
            )
            fname = os.path.join(out_dir, f"{t}_{doc_type}_{doc_id}.txt")
            with open(fname, "w") as f:
                f.write(text)
            doc_id += 1
    print(f"Wrote {doc_id} synthetic financial text documents to {out_dir}")


def main():
    base = os.path.dirname(__file__)
    raw_dir = os.path.join(base, "..", "data", "raw")
    docs_dir = os.path.join(base, "..", "data", "processed", "financial_docs")
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating synthetic price history ...")
    generate_price_history(TICKERS).to_csv(os.path.join(raw_dir, "stock_prices.csv"), index=False)

    print("Generating synthetic fundamentals ...")
    generate_fundamentals(TICKERS).to_csv(os.path.join(raw_dir, "fundamentals.csv"), index=False)

    print("Generating synthetic macro indicators ...")
    generate_macro().to_csv(os.path.join(raw_dir, "macro_indicators.csv"), index=False)

    print("Generating synthetic financial text corpus (news/filings/blogs) ...")
    generate_financial_text_docs(TICKERS, docs_dir)

    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
