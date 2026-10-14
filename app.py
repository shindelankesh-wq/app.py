import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
)

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="USD/INR Prediction Engine",
    page_icon="📈",
    layout="wide",
)

st.title("USD/INR Intraday Prediction Engine")
st.caption(
    "15-minute and 60-minute forecasts | "
    "Signal-only prototype | No order placement"
)

with st.sidebar:
    st.header("Settings")
    instrument_token = st.number_input(
        "Kite instrument token",
        min_value=0,
        value=0,
        step=1,
    )
    instrument_label = st.text_input(
        "Contract / expiry",
        value="USDINR futures",
    )
    lookback_days = st.slider(
        "Historical lookback (days)", 30, 90, 60, 15
    )
    min_train_rows = st.slider(
        "Minimum training samples", 100, 800, 250, 50
    )
    prob_threshold = st.slider(
        "Directional probability threshold",
        0.50, 0.75, 0.57, 0.01
    )
    min_rr = st.slider(
        "Minimum planned reward/risk",
        1.0, 3.0, 1.5, 0.25
    )


def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


@st.cache_resource
def connect_kite(api_key, access_token):
    if not api_key or not access_token or KiteConnect is None:
        return None, "Credentials or KiteConnect package missing."
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        kite.profile()  # Authentication check; no orders placed.
        return kite, None
    except Exception as exc:
        return None, f"Kite authentication failed: {exc}"


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(raw):
    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    df["ret_1"] = close.pct_change()
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ema_9_gap"] = close / close.ewm(span=9).mean() - 1
    df["ema_21_gap"] = close / close.ewm(span=21).mean() - 1
    df["rsi_14"] = calculate_rsi(close)

    previous = close.shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous).abs(),
            (df["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close
    df["range_pct"] = (df["high"] - df["low"]) / close

    if "volume" in df and df["volume"].notna().any():
        df["volume_change"] = (
            df["volume"].pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )
    else:
        df["volume_change"] = 0.0

    if (
        "volume" in df
        and df["volume"].fillna(0).sum() > 0
    ):
        typical = (df["high"] + df["low"] + close) / 3
        dates = df["date"].dt.date
        volume = df["volume"].fillna(0)
        cumulative_volume = volume.groupby(dates).cumsum()
        cumulative_pv = (typical * volume).groupby(dates).cumsum()
        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)
        df["vwap_gap"] = close / vwap - 1
    else:
        df["vwap_gap"] = 0.0

    return df


FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "ema_9_gap", "ema_21_gap",
    "rsi_14", "atr_pct", "range_pct",
    "volume_change", "vwap_gap",
]


def train_and_predict(features, horizon_bars, minimum_rows):
    data = features.copy()
    data["future_return"] = (
        data["close"].shift(-horizon_bars) / data["close"] - 1
    )

    volatility = data["ret_1"].std()
    if pd.isna(volatility):
        raise ValueError("Not enough candles to estimate volatility.")

    neutral_band = max(
        float(volatility) * np.sqrt(horizon_bars) * 0.20,
        0.00002,
    )

    data["target"] = np.where(
        data["future_return"] > neutral_band, 1,
        np.where(
            data["future_return"] < -neutral_band, 0, np.nan
        ),
    )

    clean = data.dropna(
        subset=FEATURES + ["target"]
    ).copy()

    if len(clean) < minimum_rows:
        raise ValueError(
            f"Only {len(clean)} usable samples; "
            f"at least {minimum_rows} required."
        )

    # Chronological split: do not randomly shuffle time-series data.
    split = int(len(clean) * 0.80)
    train = clean.iloc[:split]
    test = clean.iloc[split:]

    if train["target"].nunique() < 2:
        raise ValueError("Training data contains only one direction.")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
        )),
    ])
    model.fit(train[FEATURES], train["target"].astype(int))

    classes = list(model.named_steps["classifier"].classes_)
    up_index = classes.index(1)

    current = features.dropna(subset=FEATURES).iloc[[-1]]
    p_up = float(
        model.predict_proba(current[FEATURES])[0][up_index]
    )

    test_predictions = model.predict(test[FEATURES])
    test_probabilities = model.predict_proba(test[FEATURES])[
        :, up_index
    ]

    metrics = {
        "samples": len(clean),
        "holdout": len(test),
        "accuracy": accuracy_score(
            test["target"].astype(int), test_predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            test["target"].astype(int), test_predictions
        ),
        "brier": (
            brier_score_loss(
                test["target"].astype(int), test_probabilities
            )
            if test["target"].nunique() == 2
            else np.nan
        ),
    }

    return p_up, metrics


def get_direction(probability):
    if probability >= prob_threshold:
        return "UP BIAS"
    if probability <= 1 - prob_threshold:
        return "DOWN BIAS"
    return "RANGE / WAIT"


def price(value):
    if value is None or not np.isfinite(value):
        return "—"
    return f"₹{value:,.4f}"


api_key = get_secret("KITE_API_KEY")
access_token = get_secret("KITE_ACCESS_TOKEN")
kite, connection_error = connect_kite(api_key, access_token)

if kite is None:
    st.warning(
        "LIVE DATA UNAVAILABLE. No live price or forecast is fabricated."
    )
    st.info(
        "Configure KITE_API_KEY and KITE_ACCESS_TOKEN "
        "in Streamlit Secrets after deployment."
    )
    if connection_error:
        st.caption(connection_error)

if kite is not None and instrument_token > 0:
    if st.button("Fetch candles and calculate forecasts",
                  type="primary"):
        try:
            now = datetime.now(IST)
            start = now - timedelta(days=lookback_days)

            with st.spinner("Fetching Kite historical candles..."):
                candles = kite.historical_data(
                    instrument_token=int(instrument_token),
                    from_date=start.replace(tzinfo=None),
                    to_date=now.replace(tzinfo=None),
                    interval="5minute",
                    continuous=False,
                    oi=False,
                )

            if len(candles) < 100:
                raise ValueError(
                    "Too few candles. Verify token, contract expiry "
                    "and data access."
                )

            raw = pd.DataFrame(candles)
            features = make_features(raw)

            # 3 x 5-minute candles = 15 minutes.
            p15, metrics15 = train_and_predict(
                features, 3, min_train_rows
            )
            # 12 x 5-minute candles = 60 minutes.
            p60, metrics60 = train_and_predict(
                features, 12, min_train_rows
            )

            st.session_state["raw"] = raw
            st.session_state["features"] = features
            st.session_state["p15"] = p15
            st.session_state["p60"] = p60
            st.session_state["metrics15"] = metrics15
            st.session_state["metrics60"] = metrics60
            st.session_state["fetched_at"] = now.strftime(
                "%Y-%m-%d %H:%M:%S IST"
            )
            st.session_state["contract"] = instrument_label

        except Exception as exc:
            st.error(f"Forecast unavailable: {exc}")

if "features" in st.session_state:
    features = st.session_state["features"]
    last = features.iloc[-1]
    p15 = st.session_state["p15"]
    p60 = st.session_state["p60"]

    st.caption(
        f"Contract: {st.session_state['contract']} | "
        f"Fetched: {st.session_state['fetched_at']} | "
        f"Last candle: {last['date']}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last candle close", price(last["close"]))
    c2.metric("15-minute up probability", f"{p15:.1%}",
              get_direction(p15))
    c3.metric("60-minute up probability", f"{p60:.1%}",
              get_direction(p60))
    c4.metric("ATR (14 x 5m)", price(last["atr_14"]))

    st.caption(
        "Model probabilities are estimates, not probabilities of profit."
    )

    def make_plan(probability):
        close = float(last["close"])
        atr = float(last["atr_14"])

        if not np.isfinite(atr) or atr <= 0:
            return "WAIT — ATR unavailable", close, np.nan, np.nan, np.nan

        bias = get_direction(probability)

        if bias == "UP BIAS":
            stop = close - atr
            target = close + max(1.5 * atr, min_rr * atr)
            rr = (target - close) / (close - stop)
            status = "CONDITIONAL LONG"
        elif bias == "DOWN BIAS":
            stop = close + atr
            target = close - max(1.5 * atr, min_rr * atr)
            rr = (close - target) / (stop - close)
            status = "CONDITIONAL SHORT"
        else:
            return "WAIT — no directional edge", close, np.nan, np.nan, np.nan

        return status, close, target, stop, rr

    tab15, tab60, tabdiag = st.tabs([
        "15-minute setup",
        "60-minute setup",
        "Diagnostics",
    ])

    for tab, probability, metrics, label in [
        (tab15, p15, st.session_state["metrics15"], "15 minutes"),
        (tab60, p60, st.session_state["metrics60"], "60 minutes"),
    ]:
        with tab:
            status, entry, target, stop, rr = make_plan(probability)
            st.subheader(label)
            st.markdown(f"## {status}")

            a, b, c, d = st.columns(4)
            a.metric("Reference entry", price(entry))
            b.metric("Illustrative target", price(target))
            c.metric("Illustrative stop", price(stop))
            d.metric(
                "Planned R:R",
                f"{rr:.2f}:1" if np.isfinite(rr) else "—",
            )

            st.write(
                f"Up estimate: **{probability:.1%}** | "
                f"Down estimate: **{1-probability:.1%}**"
            )
            st.caption(
                "Reference entry is the last historical candle close, "
                "not a live executable quote. Confirm a fresh quote "
                "before acting. Costs and slippage are not included."
            )

    with tabdiag:
        rows = []
        for label, metrics in [
            ("15-minute", st.session_state["metrics15"]),
            ("60-minute", st.session_state["metrics60"]),
        ]:
            rows.append({
                "Horizon": label,
                "Samples": metrics["samples"],
                "Holdout rows": metrics["holdout"],
                "Accuracy": metrics["accuracy"],
                "Balanced accuracy": metrics["balanced_accuracy"],
                "Brier score": metrics["brier"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.warning(
            "A single holdout does not prove profitability. "
            "Walk-forward testing, realistic fees, slippage and "
            "paper trading are required before relying on signals."
        )

    chart = go.Figure(data=[
        go.Candlestick(
            x=features["date"],
            open=features["open"],
            high=features["high"],
            low=features["low"],
            close=features["close"],
            name="5-minute candles",
        )
    ])
    chart.update_layout(
        height=450,
        xaxis_rangeslider_visible=False,
        title="Historical USD/INR futures candles",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.download_button(
        "Export candles as CSV",
        data=st.session_state["raw"].to_csv(index=False).encode(),
        file_name="usdinr_candles.csv",
        mime="text/csv",
    )

elif kite is not None:
    st.info(
        "Enter the correct instrument token for the desired "
        "USD/INR futures expiry in the sidebar."
    )

st.divider()
st.caption(
    "Signal-only prototype. No orders are placed. "
    "Verify contract details, data freshness, transaction costs "
    "and execution risks independently."
)
