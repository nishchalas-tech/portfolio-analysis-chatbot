"""
Task 1: Data Collection - Unstructured Data (SEC filings, news, blogs)

Real client code for pulling 10-K/10-Q filing text from SEC EDGAR's full-text
search + submission API, and for scraping/parsing article text with
BeautifulSoup. Requires outbound internet access.

Usage:
    python fetch_unstructured_data.py --cik 0000320193 --out ../data/processed/financial_docs
"""
import argparse
import os
import re
import requests
from bs4 import BeautifulSoup

SEC_HEADERS = {"User-Agent": "Portfolio-RAG-Bot research@example.com"}


def fetch_sec_filings(cik, out_dir, limit=3):
    """Pull the most recent 10-K/10-Q filing text for a company (by CIK)."""
    cik = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    forms = data["filings"]["recent"]
    saved = 0
    for i, form_type in enumerate(forms["form"]):
        if form_type in ("10-K", "10-Q") and saved < limit:
            accession = forms["accessionNumber"][i].replace("-", "")
            doc = forms["primaryDocument"][i]
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            )
            filing_resp = requests.get(filing_url, headers=SEC_HEADERS, timeout=15)
            text = BeautifulSoup(filing_resp.content, "html.parser").get_text(" ", strip=True)
            fname = os.path.join(out_dir, f"{cik}_{form_type}_{forms['accessionNumber'][i]}.txt")
            with open(fname, "w") as f:
                f.write(text)
            saved += 1
    print(f"Saved {saved} filing(s) for CIK {cik}")


def fetch_article_text(url, out_dir, name):
    """Scrape and clean article text from a news/blog URL."""
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.content, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
    fname = os.path.join(out_dir, f"{name}.txt")
    with open(fname, "w") as f:
        f.write(text)
    print(f"Saved article -> {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", help="SEC CIK number, e.g. 0000320193 for Apple")
    parser.add_argument("--article-url", help="A news/blog article URL to scrape")
    parser.add_argument("--out", default="../data/processed/financial_docs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.cik:
        fetch_sec_filings(args.cik, args.out)
    if args.article_url:
        fetch_article_text(args.article_url, args.out, "article_" + str(abs(hash(args.article_url))))
