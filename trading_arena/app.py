from pathlib import Path
import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Silver Commodity Forecasting Agent",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,.25);
    background: linear-gradient(135deg, rgba(60,70,90,.28), rgba(30,35,45,.28));
    margin-bottom: 1rem;
}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .35rem 0 0 0; color: #AAB4C3;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px;
    padding: .8rem;
    background: rgba(128,128,128,.06);
}
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "model_data.json"

@st.cache_data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def num(x, default=math.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def money(x):
    return f"${x:,.2f}"

def pct(x):
    return f"{x:.2f}%"

def make_df(test):
    df = pd.DataFrame({
        "Date": pd.to_datetime(test["dates"]),
        "Actual": pd.to_numeric(test["actuals"], errors="coerce")
    })
    for name, values in test["predictions"].items():
        df[name] = pd.to_numeric(values, errors="coerce")
    return df

def signal(actual, predicted, threshold):
    if pd.isna(actual) or pd.isna(predicted) or actual == 0:
        return "HOLD"
    move = (predicted - actual) / actual
    if move >= threshold:
        return "BUY"
    if move <= -threshold:
        return "SELL"
    return "HOLD"

def simulate(frame, model_name, cash, qty, threshold):
    cash = float(cash)
    shares = 0
    trades = []
    equity = []

    for _, row in frame.iterrows():
        price = num(row["Actual"])
        forecast = num(row[model_name])

        if math.isnan(price) or math.isnan(forecast):
            equity.append(cash + shares * (0 if math.isnan(price) else price))
            continue

        action = signal(price, forecast, threshold)

        if action == "BUY" and cash >= price * qty:
            cash -= price * qty
            shares += qty
            trades.append([row["Date"], "BUY", qty, price, cash])
        elif action == "SELL" and shares >= qty:
            cash += price * qty
            shares -= qty
            trades.append([row["Date"], "SELL", qty, price, cash])

        equity.append(cash + shares * price)

    final_price = num(frame["Actual"].iloc[-1])
    final_value = cash + shares * final_price
    trades_df = pd.DataFrame(
        trades,
        columns=["Date", "Action", "Quantity", "Price", "Cash After"]
    )
    equity_df = pd.DataFrame({
        "Date": frame["Date"],
        "Portfolio": equity
    })
    return final_value, shares, trades_df, equity_df

if not DATA_FILE.exists():
    st.error(f"model_data.json not found next to app.py.\n\nExpected:\n{DATA_FILE}")
    st.stop()

data = load_data(str(DATA_FILE))
test = data["test_data"]
predictions = test["predictions"]
metrics = test.get("metrics", {})
scenarios = data.get("scenarios", {})
df = make_df(test)
models = list(predictions.keys())

st.sidebar.title("🥈 Forecast Controls")
selected_model = st.sidebar.selectbox(
    "Prediction model",
    models,
    index=models.index("Ensemble_Switching") if "Ensemble_Switching" in models else 0
)
days = st.sidebar.slider(
    "Recent days shown",
    min_value=60,
    max_value=len(df),
    value=min(500, len(df)),
    step=20
)
threshold = st.sidebar.slider(
    "Trading signal threshold",
    min_value=0.002,
    max_value=0.05,
    value=0.01,
    step=0.002
)

st.markdown("""
<div class="hero">
<h1>🥈 Silver Commodity Price Forecasting Agent</h1>
<p>Forecast comparison, market analytics, risk indicators and historical trading simulation.</p>
</div>
""", unsafe_allow_html=True)

latest = df.iloc[-1]
latest_actual = num(latest["Actual"])
latest_pred = num(latest[selected_model])
latest_signal = signal(latest_actual, latest_pred, threshold)

best_name = None
best_rmse = None
for name, vals in metrics.items():
    if isinstance(vals, dict) and "RMSE" in vals:
        r = num(vals["RMSE"])
        if not math.isnan(r) and (best_rmse is None or r < best_rmse):
            best_name, best_rmse = name, r

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest Price", money(latest_actual))
k2.metric(selected_model, money(latest_pred))
k3.metric("Signal", latest_signal)
k4.metric("Best RMSE", best_name or "N/A")
k5.metric("RMSE", f"{best_rmse:.4f}" if best_rmse is not None else "N/A")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Overview",
    "🤖 Model Comparison",
    "⚠️ Risk & Market",
    "💰 Trading Arena",
    "🌎 Crisis Scenarios"
])

with tab1:
    view = df.tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["Date"], y=view["Actual"],
        name="Actual Price", mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=view["Date"], y=view[selected_model],
        name=selected_model, mode="lines", line=dict(dash="dash")
    ))
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis_title="Silver Price"
    )
    st.plotly_chart(fig, width="stretch")

    diff = latest_pred - latest_actual
    diff_pct = (diff / latest_actual * 100) if latest_actual else math.nan
    st.subheader("Latest Forecast Snapshot")
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Date", "Actual", "Forecast",
            "Difference", "Difference %", "Signal"
        ],
        "Value": [
            latest["Date"].date(),
            money(latest_actual),
            money(latest_pred),
            money(diff),
            pct(diff_pct),
            latest_signal
        ]
    }), width="stretch", hide_index=True)

with tab2:
    st.subheader("Model Performance")
    rows = []
    for name, vals in metrics.items():
        if isinstance(vals, dict):
            rows.append({
                "Model": name,
                "RMSE": vals.get("RMSE"),
                "MAE": vals.get("MAE"),
                "MAPE": vals.get("MAPE"),
                "Directional Accuracy": vals.get("Directional_Accuracy"),
                "R2": vals.get("R2"),
            })
    metric_df = pd.DataFrame(rows)
    st.dataframe(metric_df, width="stretch", hide_index=True)

    if not metric_df.empty:
        ranked = metric_df.dropna(subset=["RMSE"]).sort_values("RMSE")
        bar = go.Figure(go.Bar(x=ranked["Model"], y=ranked["RMSE"]))
        bar.update_layout(template="plotly_dark", height=360,
                          yaxis_title="RMSE", xaxis_title="Model")
        st.plotly_chart(bar, width="stretch")

    chosen = st.multiselect(
        "Models to compare",
        models,
        default=models[:min(4, len(models))]
    )
    cmp = df.tail(days)
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Scatter(
        x=cmp["Date"], y=cmp["Actual"], name="Actual", line=dict(width=3)
    ))
    for name in chosen:
        comp_fig.add_trace(go.Scatter(
            x=cmp["Date"], y=cmp[name], name=name
        ))
    comp_fig.update_layout(
        template="plotly_dark",
        height=520,
        hovermode="x unified"
    )
    st.plotly_chart(comp_fig, width="stretch")

    download_df = cmp[["Date", "Actual"] + chosen]
    st.download_button(
        "Download selected predictions",
        download_df.to_csv(index=False),
        "selected_predictions.csv",
        "text/csv"
    )

with tab3:
    a, b, c = st.columns(3)
    if "volatility" in test:
        a.metric("Latest Volatility", f'{num(test["volatility"][-1]):.2f}%')
    else:
        a.metric("Latest Volatility", "N/A")
    if "fed_rate" in test:
        b.metric("Latest Fed Rate", f'{num(test["fed_rate"][-1]):.2f}%')
    else:
        b.metric("Latest Fed Rate", "N/A")
    if "gold_ratio" in test:
        c.metric("Gold/Silver Ratio", f'{num(test["gold_ratio"][-1]):.2f}')
    else:
        c.metric("Gold/Silver Ratio", "N/A")

    col1, col2 = st.columns(2)
    with col1:
        if "volatility" in test:
            v = pd.DataFrame({
                "Date": df["Date"], "Volatility": test["volatility"]
            }).set_index("Date")
            st.markdown("**Volatility**")
            st.line_chart(v)
        if "gold_ratio" in test:
            g = pd.DataFrame({
                "Date": df["Date"], "Gold/Silver Ratio": test["gold_ratio"]
            }).set_index("Date")
            st.markdown("**Gold / Silver Ratio**")
            st.line_chart(g)
    with col2:
        if "fed_rate" in test:
            f = pd.DataFrame({
                "Date": df["Date"], "Fed Rate": test["fed_rate"]
            }).set_index("Date")
            st.markdown("**Federal Funds Rate**")
            st.line_chart(f)
        if "trends" in test:
            t = pd.DataFrame({
                "Date": df["Date"], "Google Trends": test["trends"]
            }).set_index("Date")
            st.markdown("**Google Trends**")
            st.line_chart(t)

    if test.get("regimes"):
        st.subheader("Market Regimes")
        st.info(f'Latest regime: {test["regimes"][-1]}')
        regime_df = pd.DataFrame({
            "Date": df["Date"],
            "Regime": test["regimes"]
        })
        st.dataframe(regime_df.tail(30), width="stretch", hide_index=True)

with tab4:
    st.subheader("Historical Trading Arena")
    st.caption("This is a simulated replay of the exported historical test predictions; no real orders are placed.")

    c1, c2, c3 = st.columns(3)
    capital = c1.number_input(
        "Starting capital ($)", min_value=1000.0,
        value=100000.0, step=5000.0
    )
    quantity = c2.number_input(
        "Units per trade", min_value=1,
        value=10, step=1
    )
    sim_threshold = c3.slider(
        "Simulation threshold", 0.002, 0.05, float(threshold), 0.002
    )

    final_value, shares, trades_df, equity_df = simulate(
        df, selected_model, capital, quantity, sim_threshold
    )
    pnl = final_value - capital
    roi = (pnl / capital * 100) if capital else 0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Final Portfolio", money(final_value))
    p2.metric("P/L", money(pnl), f"{roi:.2f}%")
    p3.metric("Holdings", f"{shares} units")
    p4.metric("Trades", len(trades_df))

    eq = go.Figure(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Portfolio"],
        name="Portfolio Value",
        fill="tozeroy"
    ))
    eq.update_layout(
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(eq, width="stretch")

    if trades_df.empty:
        st.info("No trades were generated under the selected threshold.")
    else:
        st.dataframe(trades_df, width="stretch", hide_index=True)
        st.download_button(
            "Download trade history",
            trades_df.to_csv(index=False),
            "trade_history.csv",
            "text/csv"
        )

with tab5:
    st.subheader("Crisis Scenario Explorer")
    if scenarios:
        names = list(scenarios.keys())
        chosen_scenario = st.selectbox("Scenario", names)
        sc = scenarios[chosen_scenario]
        sc_dates = pd.to_datetime(sc.get("dates", []))
        prices = sc.get("prices", [])
        if len(sc_dates) and prices:
            sf = go.Figure(go.Scatter(
                x=sc_dates, y=prices, name="Silver Price"
            ))
            sf.update_layout(
                template="plotly_dark",
                height=420,
                hovermode="x unified"
            )
            st.plotly_chart(sf, width="stretch")

            s1, s2, s3 = st.columns(3)
            start = num(prices[0])
            end = num(prices[-1])
            change = ((end / start) - 1) * 100 if start else 0
            s1.metric("Start Price", money(start))
            s2.metric("End Price", money(end))
            s3.metric("Scenario Change", f"{change:.2f}%")
        with st.expander("Show scenario data"):
            st.json(sc)
    else:
        st.info("No scenario data was found.")

# import streamlit as st

st.set_page_config(page_title="Commodity Price Forecasting Agent", layout="wide")

st.sidebar.title("👨‍💻 About")
st.sidebar.write("Developed by Harshit (B.Tech, Information Technology)")

st.sidebar.link_button(
    "⭐ GitHub Repository",
    "https://github.com/YOUR_USERNAME/commodity-price-forecasting-agent"
)
st.markdown("---")
st.caption(
    "Silver Commodity Price Forecasting Agent • Historical analytics and educational simulation only"
)

