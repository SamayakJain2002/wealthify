import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

try:  # first run only: write the theme file (applies after one restart of the app)
    cfg = Path(__file__).parent / ".streamlit" / "config.toml"
    if not cfg.exists():
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text('[theme]\nbase="light"\nprimaryColor="#2F6FED"\nbackgroundColor="#FFFFFF"\n'
                       'secondaryBackgroundColor="#F5F8FC"\ntextColor="#0B1F3A"\n')
except Exception:
    pass

st.set_page_config(page_title="Wealthify", page_icon="◆", layout="wide")

GREEN, RED = "#0E9F6E", "#E5484D"
TONES = ["#0F2A4A", "#2F6FED", "#5B9BFF", "#0E9F6E", "#7ED7B5", "#F5A524", "#E5484D", "#8FA3BF", "#C9D6EA", "#3B4B66"]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0');
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#FFFFFF !important; color:#0B1F3A; font-family:'Manrope',sans-serif; scroll-behavior:smooth; }
[data-testid="stHeader"], [data-testid="stToolbar"] { display:none; }
[data-testid="stMainBlockContainer"] { padding-top:92px; max-width:1180px; }
h1, h2, h3 { color:#0B1F3A; letter-spacing:-0.02em; }
.topbar { position:fixed; top:0; left:0; right:0; z-index:99999; display:flex; align-items:center; gap:28px;
  padding:12px 40px; background:rgba(255,255,255,.92); backdrop-filter:blur(10px); border-bottom:1px solid #E3EAF3; }
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; color:#0B1F3A; }
.mark { width:34px; height:34px; border-radius:11px; background:#0F2A4A; color:#fff; display:grid; place-items:center; font-weight:800; }
.links { display:flex; gap:4px; flex:1; flex-wrap:wrap; }
.links a { padding:8px 14px; border-radius:10px; color:#5B6B82 !important; text-decoration:none; font-weight:600; font-size:.92rem; }
.links a:hover { background:#F1F5FB; color:#0F2A4A !important; }
.who { display:flex; align-items:center; gap:10px; font-size:.8rem; color:#5B6B82; line-height:1.25; }
.who b { display:block; color:#0B1F3A; font-size:.9rem; }
.avatar { width:38px; height:38px; border-radius:50%; background:#E8F0FF; color:#2F6FED; display:grid; place-items:center; font-weight:800; }
.anchor { display:block; height:1px; scroll-margin-top:96px; }
.hero { border-radius:28px; padding:44px 48px; margin-bottom:8px; color:#fff;
  background:radial-gradient(1100px 380px at 92% -25%, #2F6FED 0%, rgba(47,111,237,0) 62%), #0F2A4A; }
.hero h1 { color:#fff; font-size:2.5rem; font-weight:800; margin:0 0 10px 0; max-width:660px; }
.hero p { color:#C6D4EA; max-width:560px; margin:0 0 24px 0; font-size:1.02rem; }
.chips { display:flex; gap:12px; flex-wrap:wrap; }
.chip { padding:10px 16px; border-radius:14px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.18); font-weight:600; }
.head { display:flex; align-items:center; gap:14px; margin:56px 0 18px 0; }
.head h2 { margin:0; font-size:1.65rem; font-weight:800; }
.head p { margin:2px 0 0 0; color:#5B6B82; }
.ico { width:46px; height:46px; border-radius:14px; background:#EAF1FF; color:#2F6FED; display:grid; place-items:center; overflow:hidden; flex:none; }
.material-symbols-rounded { font-family:'Material Symbols Rounded'; font-size:26px; font-weight:normal; line-height:1;
  white-space:nowrap; font-variation-settings:'FILL' 0,'wght' 500; }
.kpi { border:1px solid #E3EAF3; border-radius:18px; padding:16px 20px; background:#fff; box-shadow:0 1px 2px rgba(15,42,74,.04); }
.kpi span { color:#5B6B82; font-weight:600; font-size:.85rem; }
.kpi b { display:block; font-size:1.65rem; font-weight:800; margin-top:2px; }
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #E3EAF3 !important; border-radius:20px !important;
  background:#fff; box-shadow:0 4px 18px rgba(15,42,74,.05); }
input, textarea, div[data-baseweb="select"] > div { border-radius:12px !important; border-color:#E3EAF3 !important;
  background:#fff !important; color:#0B1F3A !important; }
span[data-baseweb="tag"] { background:#EAF1FF !important; color:#0F2A4A !important; border-radius:8px !important; }
.stButton > button { background:#0F2A4A !important; color:#fff !important; border:0 !important; border-radius:12px !important;
  padding:.6rem 1.4rem; font-weight:700 !important; }
.stButton > button:hover { background:#2F6FED !important; }
.foot { margin:64px 0 24px 0; padding-top:20px; border-top:1px solid #E3EAF3; color:#5B6B82; font-size:.85rem; }
</style>""", unsafe_allow_html=True)

# Sector index (Yahoo ticker) and a curated list of large NSE stocks per sector
SECTORS = {
    "IT": ("^CNXIT", ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE"]),
    "Banks": ("^NSEBANK", ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK"]),
    "PSU banks": ("^CNXPSUBANK", ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB"]),
    "Financial services": ("^CNXFIN", ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN", "PFC", "RECLTD"]),
    "Insurance": (None, ["SBILIFE", "HDFCLIFE", "ICICIGI", "ICICIPRULI", "LICI"]),
    "Pharma": ("^CNXPHARMA", ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "TORNTPHARM", "AUROPHARMA", "ZYDUSLIFE"]),
    "Healthcare services": (None, ["APOLLOHOSP", "MAXHEALTH", "FORTIS", "LALPATHLAB", "SYNGENE"]),
    "Auto": ("^CNXAUTO", ["MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY"]),
    "FMCG": ("^CNXFMCG", ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "TATACONSUM", "MARICO"]),
    "Consumer and retail": (None, ["TITAN", "TRENT", "DMART", "ASIANPAINT", "PAGEIND", "JUBLFOOD"]),
    "Consumer durables": (None, ["HAVELLS", "VOLTAS", "DIXON", "CROMPTON", "BLUESTARCO"]),
    "Energy": ("^CNXENERGY", ["RELIANCE", "ONGC", "COALINDIA", "BPCL", "IOC", "GAIL", "OIL"]),
    "Power": (None, ["NTPC", "POWERGRID", "TATAPOWER", "NHPC", "ADANIPOWER", "JSWENERGY"]),
    "Metals": ("^CNXMETAL", ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "NMDC"]),
    "Infrastructure": ("^CNXINFRA", ["LT", "ADANIPORTS", "RVNL", "IRCON", "NCC"]),
    "Capital goods and defence": (None, ["SIEMENS", "ABB", "BEL", "HAL", "BHEL", "CUMMINSIND", "MAZDOCK"]),
    "Cement": (None, ["ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT"]),
    "Chemicals": (None, ["SRF", "PIDILITIND", "UPL", "DEEPAKNTR", "AARTIIND", "TATACHEM"]),
    "Realty": ("^CNXREALTY", ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA"]),
    "Telecom": (None, ["BHARTIARTL", "INDUSTOWER", "IDEA", "TATACOMM"]),
    "Media": ("^CNXMEDIA", ["SUNTV", "ZEEL", "PVRINOX", "TV18BRDCST"]),
}
SECTOR_OF = {s + ".NS": k for k, (_, lst) in SECTORS.items() for s in lst}
PROFILES = {"Conservative": (0.10, "minvol"), "Moderate": (0.15, "sharpe"), "Aggressive": (0.25, "sharpe")}
DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "LT", "SUNPHARMA", "MARUTI"]


@st.cache_data(ttl=86400)
def load_universe():
    """All NSE equity-series stocks: NSE download, then local EQUITY_L.csv, then the curated list."""
    df = None
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        local = Path(__file__).with_name("EQUITY_L.csv")
        if local.exists():
            df = pd.read_csv(local)
    if df is None:
        syms = sorted({s for _, lst in SECTORS.values() for s in lst})
        df = pd.DataFrame({"SYMBOL": syms, "NAME OF COMPANY": syms})
    df.columns = df.columns.str.strip()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"]
    df = df[["SYMBOL", "NAME OF COMPANY"]].dropna().copy()
    df["label"] = df["NAME OF COMPANY"].str.title() + " (" + df["SYMBOL"] + ")"
    df["ticker"] = df["SYMBOL"] + ".NS"
    return df.sort_values("label").reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_prices(tickers, years):
    data = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, thresh=int(len(data) * 0.8))
    return data.dropna()


@st.cache_data(ttl=3600)
def sector_table():
    """Uses the Nifty sector index where Yahoo has one, else an equal-weight basket of the sector's stocks."""
    def row(s):
        s = s.dropna()
        if len(s) < 70:
            return None
        return {"1M": s.iloc[-1] / s.iloc[-22] - 1, "3M": s.iloc[-1] / s.iloc[-64] - 1, "1Y": s.iloc[-1] / s.iloc[0] - 1}

    rows = {}
    idx = {v[0]: k for k, v in SECTORS.items() if v[0]}
    px = yf.download(list(idx), period="1y", auto_adjust=True, progress=False)["Close"]
    for t, name in idx.items():
        if t in px.columns and row(px[t]):
            rows[name] = row(px[t])
    rest = [k for k in SECTORS if k not in rows]
    if rest:
        cp = yf.download([s + ".NS" for k in rest for s in SECTORS[k][1]], period="1y", auto_adjust=True, progress=False)["Close"]
        for k in rest:
            cols = [s + ".NS" for s in SECTORS[k][1] if s + ".NS" in cp.columns]
            if cols:
                basket = cp[cols].ffill().pct_change().mean(axis=1).add(1).cumprod()
                if row(basket):
                    rows[k] = row(basket)
    return (pd.DataFrame(rows).T * 100).round(1) if rows else pd.DataFrame()


@st.cache_data(ttl=3600)
def nifty(start):
    return yf.download("^NSEI", start=start, auto_adjust=True, progress=False)["Close"].squeeze().dropna()


def expected_returns(px, rets, rf, method, case="Neutral", horizon=3):
    hist = rets.mean() * 252
    if method == "Historical":
        return hist, None
    mk = nifty(str(px.index[0].date())).pct_change().reindex(rets.index)
    ok = mk.notna()
    beta = (rets[ok].apply(lambda c: np.cov(c, mk[ok])[0, 1]) / mk[ok].var()) if ok.sum() >= 60 else pd.Series(1.0, index=rets.columns)
    capm = rf + beta * 0.055
    if not method.startswith("DCF"):
        return 0.5 * hist.clip(-0.30, 0.30) + 0.5 * capm, None
    mu, basis = capm.copy(), pd.Series("Market-implied", index=capm.index)
    prefetch(list(px.columns))
    for t in px.columns:
        info, fcf, g = fundamentals(t)
        gg, rr, tt = scen_defaults(g)[case]
        v = dcf_value(fcf, info.get("sharesOutstanding"), gg / 100, rr / 100, tt / 100)
        if v and v > 0:
            mu[t] = min(max((v / float(px[t].iloc[-1])) ** (1 / horizon) - 1, -0.30), 0.40)
            basis[t] = "DCF"
    return mu, basis


def optimise(rets, cap, goal, rf, target=None, groups=None, gcap=0.35, mu_in=None, minw=0.0):
    mu, cov = (rets.mean().values * 252 if mu_in is None else mu_in), rets.cov().values * 252
    n = len(mu)
    cap = max(cap, 1.0 / n + 0.02)

    def vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if groups is not None:
        gc = max(gcap, 1.0 / len(set(groups)) + 0.05)
        for g in set(groups):
            idx = np.array([i for i, x in enumerate(groups) if x == g])
            cons.append({"type": "ineq", "fun": lambda w, idx=idx: gc - w[idx].sum()})
    x0, bnds, res, met = np.full(n, 1.0 / n), [(min(minw, 0.9 / n), cap)] * n, None, None
    if target is not None:
        res = minimize(vol, x0, method="SLSQP", bounds=bnds,
                       constraints=cons + [{"type": "ineq", "fun": lambda w: w @ mu - target}])
        met = bool(res.success)
        if not res.success:
            res = None
    if res is None:
        f = vol if goal == "minvol" else (lambda w: -(w @ mu - rf) / vol(w))
        res = minimize(f, x0, method="SLSQP", bounds=bnds, constraints=cons)
    w = np.clip(res.x, 0, None)
    return w / w.sum(), met


def head(anchor, icon, title, sub):
    st.markdown(f"<div id='{anchor}' class='anchor'></div><div class='head'><div class='ico'>"
                f"<span class='material-symbols-rounded'>{icon}</span></div><div><h2>{title}</h2><p>{sub}</p></div></div>",
                unsafe_allow_html=True)


def kpi(label, value):
    st.markdown(f"<div class='kpi'><span>{label}</span><b>{value}</b></div>", unsafe_allow_html=True)


@st.cache_resource
def _store():
    return {}


def _fetch(t):
    """Plain function (no Streamlit calls) so it can run in threads. Returns info, normalised FCF, past revenue growth."""
    info, series, g = {}, [], None
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        cf = tk.cashflow
        if "Free Cash Flow" in cf.index:
            series = cf.loc["Free Cash Flow"].dropna().tolist()
        elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            series = (cf.loc["Operating Cash Flow"] + cf.loc["Capital Expenditure"]).dropna().tolist()
        rev = tk.financials.loc["Total Revenue"].dropna()
        if len(rev) >= 3 and rev.iloc[-1] > 0:
            g = float((rev.iloc[0] / rev.iloc[-1]) ** (1 / (len(rev) - 1)) - 1)
    except Exception:
        pass
    return info, (float(np.mean(series[:3])) if series else None), g


def fundamentals(t):
    d = _store()
    hit = d.get(t)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    res = _fetch(t)
    if res[0]:
        d[t] = (time.time(), res)
    return res


def prefetch(tickers):
    """Fetch company data for many stocks in parallel (much faster than one by one)."""
    d = _store()
    todo = [t for t in tickers if not (d.get(t) and time.time() - d[t][0] < 3600)]
    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for t, res in zip(todo, ex.map(_fetch, todo)):
                if res[0]:
                    d[t] = (time.time(), res)


def dcf_full(fcf, shares, g, rate, tg):
    """Three stages: growth g for years 1-5, fading linearly to tg over years 6-10, then a terminal value.
    Returns (value per share, share of value from the terminal stage)."""
    if not fcf or fcf <= 0 or not shares or rate <= tg:
        return None
    flows, cf = [], fcf
    for i in range(1, 11):
        cf *= 1 + (g if i <= 5 else g + (tg - g) * (i - 5) / 5)
        flows.append(cf)
    pv = sum(f / (1 + rate) ** i for i, f in enumerate(flows, 1))
    tv = flows[-1] * (1 + tg) / (rate - tg) / (1 + rate) ** 10
    return (pv + tv) / shares, tv / (pv + tv)


def dcf_value(*args):
    r = dcf_full(*args)
    return r[0] if r else None


def scen_defaults(g):
    """(growth %, discount rate %, long-term growth %) for each case, starting from past revenue growth."""
    b = min(max(g if g is not None else 0.10, 0.03), 0.25) * 100
    return {"Bear": (round(b * 0.5, 1), 14.0, 3.0), "Neutral": (round(b, 1), 12.0, 4.5),
            "Bull": (round(min(b * 1.4, 30.0), 1), 11.0, 6.0)}


def view_picker(key):
    return st.radio("Scenario view", ["Bear", "Neutral", "Bull", "Compare"], index=3, horizontal=True, key=key)


def scenario_block(p):
    st.markdown("**Expected return by scenario (from DCF values)**")
    view = view_picker("v_pf")
    yrs = st.slider("Years for the price to reach fair value", 1, 5, 3)
    if st.button("Estimate bear, neutral and bull returns"):
        out, miss = {"Bear": 0.0, "Neutral": 0.0, "Bull": 0.0}, []
        prefetch([t for t, wt in p["weights"].items() if wt >= 0.01])
        with st.spinner("Fetching company data and running DCFs (first run is slow)..."):
            for t, wt in p["weights"].items():
                base = float(p["mu"][t]) if "mu" in p else 0.12
                r = {"Bear": base, "Neutral": base, "Bull": base}
                if wt >= 0.01:
                    info, fcf, g = fundamentals(t)
                    price = float(p["prices"][t].iloc[-1])
                    for n, (gg, rr, tt) in scen_defaults(g).items():
                        v = dcf_value(fcf, info.get("sharesOutstanding"), gg / 100, rr / 100, tt / 100)
                        if v and v > 0:
                            r[n] = min(max((v / price) ** (1 / yrs) - 1, -0.30), 0.40)
                        elif t not in miss:
                            miss.append(t)
                for n in out:
                    out[n] += wt * r[n]
        st.session_state["scen"] = dict(out=out, miss=miss, yrs=yrs)
    sc = st.session_state.get("scen")
    if sc:
        names = ["Bear", "Neutral", "Bull"] if view == "Compare" else [view]
        for c, n in zip(st.columns(len(names)), names):
            with c:
                kpi(f"{n} case portfolio return", f"{sc['out'][n] * 100:.1f}% a year")
        st.caption(f"Each stock's DCF value is converted to a yearly return over {sc['yrs']} years (capped between -30% and +40%). "
                   f"{len(sc['miss'])} stock(s) such as banks or loss-makers have no usable DCF and keep the market-implied estimate "
                   "in every case. Holdings under 1% also keep it. Estimates only, not forecasts.")


def build(px, profile, equal, target, use_groups, amount, rf, method="Historical", case="Neutral", horizon=3, minw=0.0, asked=()):
    if px.shape[1] < 3:
        st.error("Not enough price data came back. Try other stocks or a shorter history.")
        return
    rets = px.pct_change().dropna()
    cap, goal = PROFILES[profile]
    met = None
    with st.spinner("Preparing expected returns (DCF runs are slow the first time)..."):
        mu, basis = expected_returns(px, rets, rf, method, case, horizon)
    if equal:
        w = np.full(px.shape[1], 1.0 / px.shape[1])
    else:
        groups = [SECTOR_OF.get(t, "Other") for t in px.columns] if use_groups else None
        w, met = optimise(rets, cap, goal, rf, target, groups, mu_in=mu.values, minw=minw)
    st.session_state.pop("scen", None)
    st.session_state["portfolio"] = dict(weights=pd.Series(w, index=px.columns), prices=px, rf=rf,
                                         amount=amount, profile=profile, target=target, met=met, mu=mu, method=method, basis=basis, case=case, horizon=horizon,
                                         dropped=[t.replace(".NS", "") for t in asked if t not in px.columns])


def show(p):
    w, px = p["weights"], p["prices"]
    rets = px.pct_change().dropna()
    cov = rets.cov() * 252
    mu = p.get("mu", rets.mean() * 252)
    r = float(w.values @ mu.values)
    v = float(np.sqrt(w.values @ cov.values @ w.values))
    curve = (1 + rets @ w).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    if p.get("dropped"):
        st.warning("Left out because Yahoo has too little price history: " + ", ".join(p["dropped"]) + ". Try a shorter history.")
    k = st.columns(4)
    with k[0]:
        kpi("Expected return", f"{r * 100:.1f}%")
    with k[1]:
        kpi("Volatility", f"{v * 100:.1f}%")
    with k[2]:
        kpi("Sharpe ratio", f"{(r - p['rf']) / v:.2f}")
    with k[3]:
        kpi("Worst historical fall", f"{dd * 100:.1f}%")
    st.write("")
    if p["target"] is not None:
        if p["met"]:
            st.success(f"On these return estimates this mix reaches your {p['target']:.0%} aim at the lowest risk the optimiser could find.")
        else:
            st.warning(f"These return estimates did not support {p['target']:.0%} within your risk limits. "
                       "This is the best risk-adjusted mix instead. Aiming higher usually means accepting bigger losses.")
    sec_of = [SECTOR_OF.get(t, "Other") for t in w.index]
    tbl = pd.DataFrame({"Sector": sec_of, "Weight %": (w * 100).round(1).values,
                        "Amount (Rs)": (w * p["amount"]).round(0).values, "Return %": (mu * 100).round(1).values,
                        "Volatility %": (np.sqrt(np.diag(cov.values)) * 100).round(1)},
                       index=[t.replace(".NS", "") for t in w.index])
    if p.get("basis") is not None:
        tbl["Basis"] = p["basis"].reindex(w.index).values
    tbl = tbl[tbl["Weight %"] >= 0.1].sort_values("Weight %", ascending=False)
    sw = (w.groupby(sec_of).sum() * 100).sort_values(ascending=False)
    c1, c2 = st.columns([3, 2])
    c1.dataframe(tbl, width="stretch", height=380)
    pie = go.Figure(go.Pie(labels=sw.index, values=sw.values.round(1), hole=0.6, marker=dict(colors=TONES)))
    pie.update_layout(template="plotly_white", height=380, margin=dict(t=10, b=10), font=dict(family="Manrope"),
                      legend=dict(font=dict(size=11)))
    c2.plotly_chart(pie, width="stretch")
    if str(p.get("method", "")).startswith("DCF"):
        st.caption(f"Expected returns come from DCF fair values ({p['case']} case), turned into a yearly return over {p['horizon']} years. "
                   "Risk (volatility and correlation) still comes from past prices, because there is no forward-looking source for it. "
                   "Stocks without a usable DCF, such as banks, use a market-implied estimate and are marked in the Basis column.")
    else:
        st.caption("Historical mode uses past average returns, which can overstate the future after a strong run. Blend mode averages them "
                   "(capped at 30%) with a market-implied return (risk-free rate + beta x 5.5%). Neither is a forecast.")


universe = load_universe()
sec = sector_table()

nav = [("sectors", "Sectors"), ("build", "Build portfolio"), ("stocks", "Stock analyser"),
       ("risk", "Risk"), ("backtest", "Backtest"), ("tax", "Tax")]
links = "".join(f"<a href='#{a}' target='_self'>{t}</a>" for a, t in nav)
st.markdown(f"<div class='topbar'><div class='brand'><div class='mark'>W</div>Wealthify</div>"
            f"<div class='links'>{links}</div><div class='who'><div class='avatar'>SJ</div>"
            f"<div>Made by<b>Samayak Jain</b></div></div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='hero'><h1>Build a portfolio you can explain.</h1>"
            f"<p>Pick your own stocks or answer a few questions. Wealthify builds the portfolio from NSE stocks "
            f"and shows its risk.</p><div class='chips'><div class='chip'>{len(universe):,} NSE stocks</div>"
            f"<div class='chip'>{len(SECTORS)} sectors</div><div class='chip'>Up to 40 holdings</div></div></div>",
            unsafe_allow_html=True)

# ---------- Sector scoreboard ----------
head("sectors", "bar_chart", "Sector scoreboard", "Which parts of the market have been strong or weak lately.")
with st.container(border=True):
    if sec.empty:
        st.warning("Sector data could not be loaded from Yahoo Finance. Refresh in a minute.")
    else:
        s = sec.sort_values("3M")
        fig = go.Figure(go.Bar(x=s["3M"], y=s.index, orientation="h", text=[f"{x:+.1f}%" for x in s["3M"]],
                               textposition="outside", marker_color=[GREEN if x >= 0 else RED for x in s["3M"]]))
        fig.update_layout(template="plotly_white", height=max(380, 26 * len(s) + 80), margin=dict(t=10, b=10, l=10, r=50),
                          xaxis_title="3-month return (%)", font=dict(family="Manrope"))
        a, b = st.columns([3, 2])
        a.plotly_chart(fig, width="stretch")
        b.dataframe(sec.sort_values("3M", ascending=False).rename(columns=lambda c: c + " %"), width="stretch")

# ---------- Build portfolio ----------
head("build", "pie_chart", "Build your portfolio", "Choose a path. Both end in the same portfolio view.")
mode = st.radio("Path", ["I will pick the stocks", "Recommend a portfolio for me"], horizontal=True,
                label_visibility="collapsed")
labels = universe["label"].tolist()
defaults = [x for x in labels if any(x.endswith(f"({s})") for s in DEFAULT_SYMBOLS)]
with st.container(border=True):
    z1, z2, z3 = st.columns(3)
    amount = z1.number_input("Amount to invest (Rs)", 100000, 1000000000, 5000000, step=100000)
    rf = z2.number_input("Risk-free rate (%)", 0.0, 12.0, 6.5, step=0.1) / 100
    years = z3.slider("Price history (years)", 1, 5, 3)
    method = st.radio("Expected returns based on", ["DCF-based (future)", "Blend with market", "Historical"], horizontal=True)
    case, horizon = "Neutral", 3
    if method.startswith("DCF"):
        m1, m2 = st.columns(2)
        case = m1.radio("DCF case to build on", ["Bear", "Neutral", "Bull"], index=1, horizontal=True)
        horizon = m2.slider("Years for the price to reach fair value", 1, 5, 3, key="h_build")
    minw = st.slider("Minimum weight per selected stock (%)", 0.0, 5.0, 1.0, 0.5) / 100
    if mode == "I will pick the stocks":
        picked = st.multiselect("Pick 3 to 40 stocks (type to search all NSE stocks)", labels, default=defaults)
        y1, y2 = st.columns(2)
        profile = y1.selectbox("Risk profile", list(PROFILES), index=1)
        weighting = y2.radio("Weighting", ["Optimised", "Equal"], horizontal=True)
        if st.button("Build portfolio"):
            if not 3 <= len(picked) <= 40:
                st.warning("Pick between 3 and 40 stocks.")
            else:
                tk = tuple(universe.set_index("label").loc[picked, "ticker"])
                with st.spinner("Downloading prices from Yahoo Finance..."):
                    px = get_prices(tk, years)
                build(px, profile, weighting == "Equal", None, False, amount, rf, method, case, horizon, minw, tk)
    else:
        y1, y2 = st.columns(2)
        profile = y1.selectbox("How do you feel about risk?", list(PROFILES), index=1)
        target = y2.slider("Yearly return you are aiming for (%)", 8, 30, 15) / 100
        likes = st.multiselect("Sectors you like (optional)", list(SECTORS))
        avoid = st.multiselect("Sectors to avoid (optional)", [x for x in SECTORS if x not in likes])
        n = st.slider("Number of stocks", 10, 40, 25)
        if st.button("Recommend and build"):
            pool = [x for x in SECTORS if x not in avoid]
            if len(pool) < 3:
                st.warning("Keep at least 3 sectors so the portfolio stays diversified.")
            else:
                score = sec.rank(pct=True)[["3M", "1Y"]].mean(axis=1) if not sec.empty else pd.Series(dtype=float)
                order = sorted(pool, key=lambda x: -(score.get(x, 0.5) + (0.3 if x in likes else 0)))
                k = max({"Conservative": 6, "Moderate": 5, "Aggressive": 4}[profile], -(-n // 7))
                chosen = order[:max(3, min(len(pool), max(k, len(likes))))]
                cands = tuple(x + ".NS" for c in chosen for x in SECTORS[c][1])
                with st.spinner("Screening stocks in the best sectors..."):
                    pa = get_prices(cands, years)
                mom = pa.iloc[-1] / pa.iloc[-126] - 1
                per = -(-n // len(chosen))
                sel = []
                for c in chosen:
                    cols = [t for t in pa.columns if SECTOR_OF.get(t) == c]
                    sel += list(mom[cols].sort_values(ascending=False).index[:per])
                sel = sorted(sel, key=lambda t: -mom[t])[:n]
                st.caption("Sectors used: " + ", ".join(chosen))
                build(pa[sel], profile, False, target, True, amount, rf, method, case, horizon, minw, tuple(sel))

p = st.session_state.get("portfolio")
if p:
    with st.container(border=True):
        show(p)
        scenario_block(p)

def stats(v):
    r, yrs = v.pct_change().dropna(), max(len(v) / 252, 0.1)
    return {"Return a year %": round(((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
            "Volatility %": round(r.std() * np.sqrt(252) * 100, 1),
            "Worst fall %": round(float((v / v.cummax() - 1).min()) * 100, 1)}


def fmt(v, mult=1, suffix="", dec=1):
    return "n/a" if v is None else f"{v * mult:,.{dec}f}{suffix}"


# ---------- Stock analyser ----------
head("stocks", "search", "Stock analyser", "Valuation, ratios and charts for each holding.")
with st.container(border=True):
    if not p:
        st.info("Build a portfolio above first. This page uses it.")
    else:
        pick = st.selectbox("Choose a holding", [x.replace(".NS", "") for x in p["prices"].columns])
        tkr = pick + ".NS"
        ser = p["prices"][tkr]
        info, fcf, g = fundamentals(tkr)
        fig = go.Figure()
        fig.add_scatter(x=ser.index, y=ser.values, name="Price", line=dict(color="#0F2A4A", width=2))
        for days, col in [(50, "#2F6FED"), (200, "#F5A524")]:
            fig.add_scatter(x=ser.index, y=ser.rolling(days).mean(), name=f"{days}-day average", line=dict(color=col, width=1.5))
        fig.update_layout(template="plotly_white", height=340, margin=dict(t=10, b=10), font=dict(family="Manrope"),
                          yaxis_title="Price (Rs)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Sector: {info.get('sector', 'n/a')}")
        rows = [("Market cap (Rs crore)", fmt(info.get("marketCap"), 1e-7, "", 0)),
                ("P/E (trailing)", fmt(info.get("trailingPE"), 1, "x")), ("P/E (forward)", fmt(info.get("forwardPE"), 1, "x")),
                ("Price to book", fmt(info.get("priceToBook"), 1, "x")), ("Return on equity", fmt(info.get("returnOnEquity"), 100, "%")),
                ("Debt to equity", fmt(info.get("debtToEquity"))), ("52-week high (Rs)", fmt(info.get("fiftyTwoWeekHigh"))),
                ("52-week low (Rs)", fmt(info.get("fiftyTwoWeekLow")))]
        for i in (0, 4):
            for c, (lab, val) in zip(st.columns(4), rows[i:i + 4]):
                with c:
                    kpi(lab, val)
            st.write("")
        st.markdown("**DCF valuation: bear, neutral and bull**")
        shares = info.get("sharesOutstanding")
        if not fcf or fcf <= 0 or not shares:
            st.warning("A DCF is not meaningful here: free cash flow or share data is missing or negative. "
                       "This is common for banks and financial companies.")
        else:
            dflt = scen_defaults(g)
            st.caption((f"Starting growth is based on this company's past revenue growth of {g * 100:.0f}% a year. " if g is not None
                        else "Past revenue growth was not available, so 10% is the starting point. ")
                       + "Adjust any case below, and use research reports you trust for the sector outlook.")
            last = float(ser.iloc[-1])
            mcap = last * shares
            with st.expander("Data used in this DCF (check it against a source you trust)"):
                st.markdown(
                    f"- **Price:** Rs {last:,.1f}, the last close from Yahoo Finance ({ser.index[-1].date()})\n"
                    f"- **Shares outstanding (Yahoo):** {shares / 1e7:,.1f} crore, so market cap is about Rs {mcap / 1e7:,.0f} crore\n"
                    f"- **Starting free cash flow:** Rs {fcf / 1e7:,.0f} crore, the average of up to 3 recent years from Yahoo's cash flow statement\n"
                    f"- **Free cash flow yield:** {fcf / mcap * 100:.1f}% of market cap\n"
                    f"- **Past revenue growth:** " + (f"{g * 100:.1f}% a year (Yahoo, about 3 to 4 years). Used only to set the default growth, not in the valuation."
                                                      if g is not None else "not available") + "\n"
                    f"- **Sector:** {info.get('sector', 'n/a')}")
                yc = info.get("marketCap")
                if yc and abs(mcap / yc - 1) > 0.3:
                    st.warning("Yahoo's share count and its market cap disagree by more than 30%, so the share data may be wrong. "
                               "Treat the per-share values with suspicion.")
                if info.get("sector") in ("Financial Services", "Real Estate"):
                    st.warning("Free cash flow DCFs are unreliable for banks, lenders and real estate companies, because their "
                               "operating cash flow includes loans and project cycles.")
            view = view_picker("v_sa")
            asm = {}
            with st.expander("Adjust the assumptions for each case"):
                for c, name in zip(st.columns(3), ["Bear", "Neutral", "Bull"]):
                    with c:
                        st.markdown(f"**{name}**")
                        gg = st.slider("Growth, next 5 years (%)", 0.0, 30.0, dflt[name][0], 0.5, key=f"{tkr}{name}g")
                        rr = st.slider("Discount rate (%)", 2.0, 18.0, dflt[name][1], 0.5, key=f"{tkr}{name}r")
                        tt = st.slider("Long-term growth (%)", 0.0, 15.0, dflt[name][2], 0.5, key=f"{tkr}{name}t")
                        asm[name] = (gg, rr, tt)
            ordered = (asm["Bear"][0] <= asm["Neutral"][0] <= asm["Bull"][0] and asm["Bear"][2] <= asm["Neutral"][2] <= asm["Bull"][2]
                       and asm["Bear"][1] >= asm["Neutral"][1] >= asm["Bull"][1])
            if not ordered:
                st.warning("Check the order of your cases: Bear should have the lowest growth and the highest discount rate, "
                           "Bull the opposite, and Neutral in between.")
            price = float(ser.iloc[-1])
            vals = {n: dcf_value(fcf, shares, a[0] / 100, a[1] / 100, a[2] / 100) for n, a in asm.items()}
            names = ["Bear", "Neutral", "Bull"] if view == "Compare" else [view]
            for c, n in zip(st.columns(len(names)), names):
                with c:
                    if vals[n] is None:
                        st.error(f"{n}: the discount rate must be above long-term growth.")
                    else:
                        kpi(f"{n} case value per share", f"Rs {vals[n]:,.0f} ({(vals[n] / price - 1) * 100:+.0f}%)")
            if view == "Compare" and all(vals.values()):
                fig = go.Figure(go.Bar(x=list(vals), y=[round(v) for v in vals.values()], marker_color=[RED, "#8FA3BF", GREEN]))
                fig.add_hline(y=price, line_dash="dash", line_color="#0F2A4A", annotation_text=f"Price {price:,.0f}")
                fig.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=10), font=dict(family="Manrope"),
                                  yaxis_title="Value per share (Rs)")
                st.plotly_chart(fig, width="stretch")
            for n in names:
                if vals[n] and not 0.33 <= vals[n] / price <= 3:
                    st.warning(f"{n} case: the DCF value is {vals[n] / price:.1f}x the current price. That usually means the inputs "
                               "(especially a small gap between the discount rate and long-term growth) or Yahoo's data need checking.")
            sc_case = "Neutral" if view == "Compare" else view
            gg, rr, tt = asm[sc_case]
            full = dcf_full(fcf, shares, gg / 100, rr / 100, tt / 100)
            if full:
                st.caption(f"{sc_case} case: {full[1] * 100:.0f}% of the value comes from the long-term (terminal) stage."
                           + (" That is high, so treat the result with care." if full[1] > 0.75 else ""))
                st.caption(f"Terminal multiple: the long-term stage values the business at about "
                           f"{(1 + tt / 100) / ((rr - tt) / 100):.0f}x its year-10 cash flow.")
                rates = [rr + d for d in (-2, -1, 0, 1, 2)]
                tgs = [tt + d for d in (-1, 0, 1)]
                grid = pd.DataFrame({f"LT growth {x:.1f}%": [dcf_value(fcf, shares, gg / 100, r_ / 100, x / 100) for r_ in rates]
                                     for x in tgs}, index=[f"Discount {r_:.1f}%" for r_ in rates])
                st.markdown(f"**Sensitivity: value per share, {sc_case} case**")
                st.dataframe(grid.astype(float).round(0), width="stretch")
            if any(asm[n][2] > 7 or asm[n][1] < 6.5 for n in names):
                st.warning("Long-term growth above about 7% means outgrowing the economy forever, and a discount rate below "
                           "about 6.5% is under the risk-free rate. Treat those results with suspicion.")
            st.caption("Three stages: your growth rate for 5 years, a fade to the long-term rate over the next 5 years, then a long-term stage, "
                       "all discounted back. The starting cash flow is the average of up to 3 recent years of free cash flow. "
                       "No adjustment is made for debt or cash, which can matter a lot for leveraged companies. Small changes in "
                       "these inputs move the answer a lot. A learning tool, not a target price.")


# ---------- Risk ----------
head("risk", "shield", "Risk and scenarios", "How bad could it get, and what range of outcomes is realistic?")
with st.container(border=True):
    if not p:
        st.info("Build a portfolio above first. This page uses it.")
    else:
        w, px = p["weights"], p["prices"]
        port = px.pct_change().dropna() @ w
        nf = nifty(str(px.index[0].date())).pct_change().reindex(port.index).dropna()
        beta = float(np.cov(port.loc[nf.index], nf)[0, 1] / nf.var()) if len(nf) > 60 else float("nan")
        month = float(((1 + port).rolling(21).apply(np.prod, raw=True) - 1).min())
        k = st.columns(3)
        with k[0]:
            kpi("Beta vs Nifty 50", "n/a" if np.isnan(beta) else f"{beta:.2f}")
        with k[1]:
            kpi("Worst day", f"{port.min() * 100:.1f}%")
        with k[2]:
            kpi("Worst month", f"{month * 100:.1f}%")
        st.markdown("**Stress test: if the market falls**")
        if np.isnan(beta):
            st.warning("Nifty 50 data could not be loaded, so the stress test is unavailable.")
        else:
            sh = [-10, -20, -30]
            st.dataframe(pd.DataFrame({"Market move %": sh, "Estimated portfolio move %": [round(beta * s, 1) for s in sh],
                                       "Estimated change (Rs)": [round(p["amount"] * beta * s / 100) for s in sh]}),
                         width="stretch", hide_index=True)
        st.markdown("**One-year Monte Carlo (4,000 simulations)**")
        paths = p["amount"] * np.cumprod(1 + np.random.default_rng(42).choice(port.values, size=(4000, 252)), axis=1)
        qs = [5, 25, 50, 75, 95]
        pct = np.percentile(paths, qs, axis=0)
        fig = go.Figure()
        for i, (q, c) in enumerate(zip(qs, [RED, "#8FA3BF", "#0F2A4A", "#8FA3BF", GREEN])):
            fig.add_scatter(y=pct[i], name=f"{q}th percentile", line=dict(color=c, width=3 if q == 50 else 1.5))
        fig.update_layout(template="plotly_white", height=340, margin=dict(t=10, b=10), font=dict(family="Manrope"),
                          xaxis_title="Trading days ahead", yaxis_title="Portfolio value (Rs)")
        st.plotly_chart(fig, width="stretch")
        fin = paths[:, -1]
        m = st.columns(4)
        with m[0]:
            kpi("Median outcome", f"Rs {np.median(fin):,.0f}")
        with m[1]:
            kpi("Bad case (5th pct)", f"Rs {np.percentile(fin, 5):,.0f}")
        with m[2]:
            kpi("Good case (95th pct)", f"Rs {np.percentile(fin, 95):,.0f}")
        with m[3]:
            kpi("Chance of a loss", f"{(fin < p['amount']).mean() * 100:.0f}%")
        st.caption("Resamples your portfolio's own past daily returns. It assumes the future looks like the past, "
                   "which markets often break. Beta-based stress results are rough estimates.")

# ---------- Backtest ----------
head("backtest", "history", "Backtest", "How this portfolio would have done against the Nifty 50.")
with st.container(border=True):
    if not p:
        st.info("Build a portfolio above first. This page uses it.")
    else:
        w, px = p["weights"], p["prices"]
        test = st.radio("Test type", ["Full history", "Out-of-sample"], horizontal=True)
        st.caption("Full history reuses the data the weights were built on, so it flatters the result. Out-of-sample "
                   "rebuilds the portfolio on the first 70% of the dates (with your risk profile) and tests it on the last 30%.")
        tp, wv = px, w
        if test == "Out-of-sample":
            cut = int(len(px) * 0.7)
            cap, goal = PROFILES[p["profile"]]
            w2, _ = optimise(px.iloc[:cut].pct_change().dropna(), cap, goal, p["rf"])
            tp, wv = px.iloc[cut:], pd.Series(w2, index=px.columns)
        series = {"Your portfolio": ((tp / tp.iloc[0]) @ wv) * 100}
        nf = nifty(str(tp.index[0].date())).reindex(tp.index).ffill().dropna()
        if len(nf) > 20:
            series["Nifty 50"] = nf / nf.iloc[0] * 100
        fig = go.Figure()
        for (name, v), c in zip(series.items(), ["#0F2A4A", "#2F6FED"]):
            fig.add_scatter(x=v.index, y=v.values, name=name, line=dict(color=c, width=2.5))
        fig.update_layout(template="plotly_white", height=340, margin=dict(t=10, b=10), font=dict(family="Manrope"),
                          yaxis_title="Value of 100 invested")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame({n: stats(v) for n, v in series.items()}).T, width="stretch")

def cg_tax(g, long_term):
    """Listed equity, FY 2026-27: short-term 20%; long-term 12.5% above Rs 1.25 lakh a year. Excludes surcharge and cess."""
    return max(0.0, g - 125000) * 0.125 if long_term else max(0.0, g) * 0.20


# ---------- Tax and rebalancing ----------
head("tax", "receipt_long", "Tax and rebalancing", "Capital gains tax, rebalancing trades and tax-loss harvesting for Indian listed equity.")
with st.container(border=True):
    if not p:
        st.info("Build a portfolio above first. This page uses it.")
    else:
        w, px = p["weights"], p["prices"]
        r1, r2 = st.columns(2)
        with r1:
            kpi("Short-term gains (held 12 months or less)", "20%")
        with r2:
            kpi("Long-term gains (over 12 months; first Rs 1.25 lakh a year exempt)", "12.5%")
        st.write("")
        default_m = int(min(60, max(1, round((px.index[-1] - px.index[0]).days / 30.4))))
        held = st.slider("How long have you held these stocks (months)", 1, 60, default_m)
        long_term = held > 12
        cost = w * p["amount"]
        val = cost * (px.iloc[-1] / px.iloc[0])
        cur = val / val.sum()
        trade = (w - cur) * val.sum()
        gain = ((-trade).clip(lower=0) / val) * (val - cost)
        net = float(gain.sum())
        k = st.columns(4)
        with k[0]:
            kpi("Value today", f"Rs {val.sum():,.0f}")
        with k[1]:
            kpi("Gain since purchase", f"Rs {val.sum() - cost.sum():,.0f}")
        with k[2]:
            kpi("Gain realised by rebalancing", f"Rs {net:,.0f}")
        with k[3]:
            kpi("Estimated tax", f"Rs {cg_tax(net, long_term):,.0f}")
        st.write("")
        st.dataframe(pd.DataFrame({"If these gains are treated as": ["Short-term (20%)", "Long-term (12.5% above Rs 1.25 lakh)"],
                                   "Tax on rebalancing gains (Rs)": [round(cg_tax(net, False)), round(cg_tax(net, True))]}),
                     width="stretch", hide_index=True)
        if net > 0 and not long_term and cg_tax(net, False) > cg_tax(net, True):
            st.info(f"Waiting until the holding passes 12 months would cut the tax on these gains from "
                    f"Rs {cg_tax(net, False):,.0f} to Rs {cg_tax(net, True):,.0f}.")
        tbl = pd.DataFrame({"Action": np.where(trade > 0, "Buy", "Sell"), "Target %": (w * 100).round(1), "Drifted %": (cur * 100).round(1),
                            "Trade (Rs)": trade.round(0), "Gain realised (Rs)": gain.round(0)})
        tbl.index = [t.replace(".NS", "") for t in tbl.index]
        st.dataframe(tbl[tbl["Trade (Rs)"].abs() > 1000].sort_values("Trade (Rs)"), width="stretch", height=300)
        st.caption("Positive trade = buy, negative = sell. Gain realised is only counted on the part you sell, so buys show 0.")
        st.markdown("**Tax-loss harvesting**")
        unreal = val - cost
        losers = unreal[unreal < 0]
        loss = float(-losers.sum())
        h = st.columns(3)
        with h[0]:
            kpi("Unrealised losses you could book", f"Rs {loss:,.0f}")
        with h[1]:
            kpi("Estimated tax saved", f"Rs {cg_tax(net, long_term) - cg_tax(net - loss, long_term):,.0f}")
        with h[2]:
            kpi("Tax-free gains still available", f"Rs {max(0.0, 125000 - net) if long_term else 0:,.0f}")
        if losers.empty:
            st.info("No holding is below its purchase value in this window, so there is nothing to harvest.")
        else:
            lt = pd.DataFrame({"Unrealised loss (Rs)": losers.round(0), "Loss %": ((losers / cost[losers.index]) * 100).round(1)})
            lt.index = [t.replace(".NS", "") for t in lt.index]
            st.dataframe(lt.sort_values("Unrealised loss (Rs)"), width="stretch")
        st.caption("How it works: a short-term loss can be set off against short-term and long-term gains; a long-term loss only "
                   "against long-term gains. Unused losses carry forward for 8 years if you file your return on time. You can also "
                   "book up to Rs 1.25 lakh of long-term gains a year tax-free and buy back to reset your cost price. Brokerage, "
                   "STT and missing a price move are real costs. As far as I know there is no formal wash-sale rule for shares, "
                   "but authorities can challenge artificial transactions. This page is simplified (one purchase date, no surcharge, "
                   "cess or brokerage) and rates can change, so check with a tax professional.")

st.markdown("<div class='foot'>For educational purposes only. Nothing here is investment advice or a recommendation "
            "to buy or sell. Market data comes from Yahoo Finance and NSE and may be delayed or incomplete. "
            "Past performance does not guarantee future results. Made by Samayak Jain.</div>", unsafe_allow_html=True)
