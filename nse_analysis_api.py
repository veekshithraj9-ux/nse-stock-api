"""
NSE Stock Dashboard (Streamlit) — reuses the indicator functions in nse_analysis_api.py
Run locally:  streamlit run streamlit_app.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nse_analysis_api import (
    calculate_rsi,
    calculate_macd,
    calculate_moving_averages,
    calculate_bollinger_bands,
    get_technical_snapshot,
    get_fundamentals,
)

st.set_page_config(page_title="NSE Stock Dashboard", page_icon="📈", layout="wide")


# ── Helpers ────────────────────────────────────────────────────────────────
def to_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str) -> dict:
    return get_fundamentals(yf.Ticker(ticker))


def signal_score(t: dict):
    """Simple rule-based score from the technical snapshot. Educational only."""
    score, reasons = 0, []
    rsi = t.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            score += 1; reasons.append(f"RSI {rsi} is oversold (<30)")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI {rsi} is overbought (>70)")
        else:
            reasons.append(f"RSI {rsi} is neutral")
    if t.get("macd_histogram") is not None:
        if t["macd_histogram"] > 0:
            score += 1; reasons.append("MACD histogram positive (bullish momentum)")
        else:
            score -= 1; reasons.append("MACD histogram negative (bearish momentum)")
    if t.get("ma_trend") == "bullish":
        score += 1; reasons.append("50-DMA above 200-DMA (golden-cross regime)")
    elif t.get("ma_trend") == "bearish":
        score -= 1; reasons.append("50-DMA below 200-DMA (death-cross regime)")
    if t.get("ma_50") and t.get("current_price"):
        if t["current_price"] > t["ma_50"]:
            score += 1; reasons.append("Price above 50-DMA")
        else:
            score -= 1; reasons.append("Price below 50-DMA")
    adx = t.get("adx_14")
    if adx is not None:
        reasons.append(f"ADX {adx}: " + ("strong trend" if adx > 25 else "weak/no trend"))
    label = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"
    return score, label, reasons


def build_chart(hist: pd.DataFrame, months: int, show_bb: bool) -> go.Figure:
    close = hist["Close"]
    ma50, ma200 = calculate_moving_averages(close)
    rsi = calculate_rsi(close)
    macd_line, sig_line, macd_hist = calculate_macd(close)
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = middle + 2 * std, middle - 2 * std

    n = months * 21
    h = hist.tail(n)
    idx = h.index

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(x=idx, open=h["Open"], high=h["High"], low=h["Low"],
                                 close=h["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma50.loc[idx], name="MA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma200.loc[idx], name="MA200", line=dict(width=1.5)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=idx, y=upper.loc[idx], name="BB upper",
                                 line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=lower.loc[idx], name="BB lower",
                                 line=dict(width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_line.loc[idx], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=sig_line.loc[idx], name="Signal"), row=3, col=1)
    fig.add_trace(go.Bar(x=idx, y=macd_hist.loc[idx], name="Histogram"), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def backtest_ma_cross(hist: pd.DataFrame, short=50, long=200):
    close = hist["Close"]
    ma_s, ma_l = close.rolling(short).mean(), close.rolling(long).mean()
    position = (ma_s > ma_l).astype(int).shift(1).fillna(0)  # trade next day, no look-ahead
    ret = close.pct_change().fillna(0)
    strat = (1 + ret * position).cumprod()
    bh = (1 + ret).cumprod()
    years = max((close.index[-1] - close.index[0]).days / 365.25, 0.01)

    def stats(curve):
        cagr = curve.iloc[-1] ** (1 / years) - 1
        mdd = (curve / curve.cummax() - 1).min()
        return round(cagr * 100, 2), round(mdd * 100, 2), round((curve.iloc[-1] - 1) * 100, 2)

    trades = int((position.diff().abs() == 1).sum())
    return strat, bh, stats(strat), stats(bh), trades


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📈 NSE Dashboard")
page = st.sidebar.radio("Go to", ["Stock Analysis", "Watchlist", "Compare Stocks", "Backtest"])
st.sidebar.caption("Data: Yahoo Finance (may be delayed). Educational use only — not investment advice.")


# ── Page 1: Stock Analysis ─────────────────────────────────────────────────
if page == "Stock Analysis":
    st.title("Stock Analysis")
    c1, c2, c3 = st.columns([2, 1, 1])
    symbol = c1.text_input("NSE symbol", "RELIANCE")
    months = c2.slider("Chart months", 1, 24, 6)
    show_bb = c3.checkbox("Bollinger Bands", True)

    if symbol:
        ticker = to_ticker(symbol)
        hist = load_history(ticker)
        if hist.empty:
            st.error(f"No data found for {ticker}. Check the symbol.")
        else:
            tech, ca_flag = get_technical_snapshot(hist)
            score, label, reasons = signal_score(tech)

            if ca_flag:
                st.warning("A >15% single-day move was detected in the last 2 years (split/demerger?). "
                           "MA and Bollinger values may be distorted.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price (₹)", tech["current_price"])
            m2.metric("RSI (14)", tech["rsi_14"])
            m3.metric("% of 52W high", tech["pct_of_52w_high"])
            m4.metric("Signal", label, f"score {score:+d}")

            st.plotly_chart(build_chart(hist, months, show_bb), use_container_width=True)

            tab1, tab2, tab3 = st.tabs(["Why this signal", "All technicals", "Fundamentals"])
            with tab1:
                for r in reasons:
                    st.write("•", r)
            with tab2:
                st.dataframe(pd.DataFrame(tech.items(), columns=["Indicator", "Value"]),
                             use_container_width=True, hide_index=True)
            with tab3:
                with st.spinner("Loading fundamentals..."):
                    f = load_fundamentals(ticker)
                st.dataframe(pd.DataFrame(f.items(), columns=["Metric", "Value"]),
                             use_container_width=True, hide_index=True)

            st.download_button("Download price history (CSV)", hist.to_csv().encode(),
                               file_name=f"{ticker}_history.csv", mime="text/csv")


# ── Page 2: Watchlist ──────────────────────────────────────────────────────
elif page == "Watchlist":
    st.title("Watchlist")
    raw = st.text_area("Symbols (comma-separated, max 15)", "RELIANCE, TCS, INFY, HDFCBANK, ITC")
    if st.button("Scan"):
        symbols = [s.strip() for s in raw.split(",") if s.strip()][:15]
        rows = []
        bar = st.progress(0.0)
        for i, s in enumerate(symbols):
            t = to_ticker(s)
            try:
                h = load_history(t)
                if h.empty:
                    rows.append({"Symbol": t, "Signal": "No data"})
                else:
                    tech, flag = get_technical_snapshot(h)
                    score, label, _ = signal_score(tech)
                    rows.append({
                        "Symbol": t, "Price": tech["current_price"], "RSI": tech["rsi_14"],
                        "MACD hist": tech["macd_histogram"], "MA trend": tech["ma_trend"],
                        "% of 52W high": tech["pct_of_52w_high"], "Score": score,
                        "Signal": label, "Corp. action?": flag,
                    })
            except Exception as e:
                rows.append({"Symbol": t, "Signal": f"Error: {e}"})
            bar.progress((i + 1) / len(symbols))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           file_name="watchlist.csv", mime="text/csv")


# ── Page 3: Compare ────────────────────────────────────────────────────────
elif page == "Compare Stocks":
    st.title("Compare Stocks")
    raw = st.text_input("Symbols (comma-separated)", "RELIANCE, TCS, INFY")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    if raw:
        series = {}
        for s in [x.strip() for x in raw.split(",") if x.strip()][:8]:
            t = to_ticker(s)
            h = load_history(t, period)
            if not h.empty:
                series[t] = h["Close"] / h["Close"].iloc[0] * 100
        if series:
            df = pd.DataFrame(series)
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
            fig.update_layout(title="Performance rebased to 100", height=500,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            ret = (df.iloc[-1] - 100).round(2).rename("Return %").to_frame()
            st.dataframe(ret, use_container_width=True)
            st.subheader("Correlation of daily returns")
            st.dataframe(df.pct_change().corr().round(2), use_container_width=True)
        else:
            st.error("No data found for those symbols.")


# ── Page 4: Backtest ───────────────────────────────────────────────────────
else:
    st.title("Moving-Average Crossover Backtest")
    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("NSE symbol", "RELIANCE")
    short = c2.number_input("Short MA", 5, 100, 50)
    long = c3.number_input("Long MA", 20, 300, 200)
    period = c4.selectbox("History", ["2y", "5y", "10y"], index=1)

    if symbol and short < long:
        t = to_ticker(symbol)
        h = load_history(t, period)
        if h.empty:
            st.error(f"No data for {t}.")
        else:
            strat, bh, s_stats, b_stats, trades = backtest_ma_cross(h, int(short), int(long))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strat.index, y=strat, name=f"MA {short}/{long} strategy"))
            fig.add_trace(go.Scatter(x=bh.index, y=bh, name="Buy & hold"))
            fig.update_layout(title="Growth of ₹1", height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(
                {"CAGR %": [s_stats[0], b_stats[0]],
                 "Max drawdown %": [s_stats[1], b_stats[1]],
                 "Total return %": [s_stats[2], b_stats[2]]},
                index=["Strategy", "Buy & hold"]), use_container_width=True)
            st.caption(f"Number of position changes: {trades}. Ignores brokerage, taxes and slippage.")
    elif short >= long:
        st.warning("Short MA must be smaller than long MA.")
