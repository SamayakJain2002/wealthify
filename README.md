# Wealthify

A portfolio analytics studio for Indian equities, built in Python and Streamlit. It is an educational project, not investment advice.

**Live app:** _add your Streamlit link here_

## What it does

- **Sector scoreboard:** 1, 3 and 12 month performance across 21 NSE sectors.
- **Build a portfolio two ways:** pick your own stocks (any NSE-listed stock, up to 40), or answer a few questions and get recommended sectors and stocks.
- **Expected returns:** DCF-based (bear, neutral and bull cases), a blend with a market-implied return, or historical.
- **Stock analyser:** price with moving averages, key ratios, and a three-stage DCF with a sensitivity table.
- **Risk and scenarios:** beta, stress tests and a Monte Carlo simulation.
- **Backtest:** against the Nifty 50, with an out-of-sample option.
- **Tax and rebalancing:** short-term and long-term capital gains (FY 2026-27 rates), rebalancing trades and tax-loss harvesting.

## Screenshots

![Sectors](docs/1-sectors.png)
![Portfolio](docs/2-portfolio.png)
![DCF](docs/3-dcf.png)
![Risk](docs/4-risk.png)
![Tax](docs/5-tax.png)

## How it works

- Prices and fundamentals come from Yahoo Finance (`yfinance`); the stock list comes from NSE.
- Weights come from a constrained optimiser (SciPy SLSQP) with per-stock and per-sector limits.
- Risk (volatility and correlation) is estimated from past prices.

## Limitations

- Yahoo Finance data is free, delayed and sometimes incomplete.
- The DCF is simplified, makes no adjustment for debt or cash, and does not suit banks.
- The tax page assumes a single purchase date and ignores surcharge, cess and brokerage.
- Optimisers amplify errors in their inputs. Results are estimates, not forecasts.

## Run locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Disclaimer

For educational purposes only. Nothing here is investment advice or a recommendation to buy or sell.

Made by Samayak Jain.
