import streamlit as st
import numpy as np
import pandas as pd
import pywt
import ccxt
import time

st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling on Real BTC")
st.markdown("Live data from Kraken • Adaptive windows • Honest history limits")

# ==============================================================================
# HELPERS
# ==============================================================================
def sharpe_ratio(rets, periods_per_year=252 * 24):
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    sd = np.std(rets, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 2:
        return 0.02
    log_rets = np.log(prices[1:] / prices[:-1])
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) < 2:
        return 0.02
    vol = np.std(log_rets, ddof=1)
    return float(vol) if np.isfinite(vol) and vol > 0 else 0.02


def adaptive_window_size(n, requested_window, min_window=64):
    if n < min_window + 10:
        return None
    max_allowed = max(min_window, int(n * 0.45))
    return int(min(requested_window, max_allowed))


# ==============================================================================
# FETCH REAL BTC DATA
# ==============================================================================
@st.cache_data(ttl=1800)
def fetch_real_btc_data(target_bars=3500, symbol="BTC/USD", timeframe="1h"):
    try:
        exchange = ccxt.kraken({
            "enableRateLimit": True,
            "timeout": 30000,
        })
        exchange.load_markets()

        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        now = exchange.milliseconds()
        since = now - target_bars * tf_ms

        all_rows = []
        seen_ts = set()
        last_last_ts = None
        logs = []

        for i in range(20):
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=720)
            if not candles:
                logs.append(f"Call {i+1}: no candles returned")
                break

            batch = pd.DataFrame(
                candles, columns=["ts", "open", "high", "low", "close", "volume"]
            ).drop_duplicates(subset="ts").sort_values("ts")

            if batch.empty:
                logs.append(f"Call {i+1}: empty batch after dedup")
                break

            first_ts = int(batch["ts"].iloc[0])
            last_ts = int(batch["ts"].iloc[-1])
            logs.append(
                f"Call {i+1}: {len(batch)} rows from "
                f"{exchange.iso8601(first_ts)} to {exchange.iso8601(last_ts)}"
            )

            new_rows = [row for row in batch.values.tolist() if row[0] not in seen_ts]
            if not new_rows:
                logs.append("No new timestamps; stopping")
                break

            for row in new_rows:
                seen_ts.add(row[0])
            all_rows.extend(new_rows)

            if last_last_ts is not None and last_ts <= last_last_ts:
                logs.append("Pagination stalled; stopping")
                break

            last_last_ts = last_ts
            since = last_ts + tf_ms

            if len(all_rows) >= target_bars:
                break

            time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.25)

        if not all_rows:
            return {"ok": False, "error": "No candles returned from Kraken", "df": None, "prices": None, "meta": {"logs": logs}}

        df = pd.DataFrame(
            all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
        ).drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        prices = df["close"].to_numpy(dtype=float)

        return {
            "ok": True,
            "error": None,
            "df": df,
            "prices": prices,
            "meta": {
                "fetched_bars": len(df),
                "days": len(df) / 24.0,
                "requested_bars": target_bars,
                "history_limited": len(df) < target_bars,
                "first_iso": df["datetime"].iloc[0].isoformat(),
                "last_iso": df["datetime"].iloc[-1].isoformat(),
                "logs": logs,
            },
        }

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "df": None, "prices": None, "meta": {"logs": []}}


# ==============================================================================
# WAVELET / LABELING
# ==============================================================================
@st.cache_data(show_spinner=False)
def causal_wavelet_denoise(prices_tuple, window_size):
    prices = np.asarray(prices_tuple, dtype=float)
    n = len(prices)
    if n == 0:
        return prices.copy()

    denoised = prices.copy()
    wavelet_name = "db4"
    wavelet = pywt.Wavelet(wavelet_name)

    if window_size >= n:
        window_size = max(32, n - 1)

    for i in range(window_size - 1, n):
        window_data = prices[i - window_size + 1:i + 1]
        if len(window_data) < 16:
            denoised[i] = window_data[-1]
            continue

        max_level = pywt.dwt_max_level(len(window_data), wavelet.dec_len)
        level = max(1, min(3, max_level))

        coeffs = pywt.wavedec(window_data, wavelet_name, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
        uthresh = sigma * np.sqrt(2 * np.log(len(window_data))) if sigma > 0 else 0.0

        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs_thresh, wavelet_name)
        rec = np.asarray(rec).flatten()[:len(window_data)]

        denoised[i] = rec[-1] if len(rec) else window_data[-1]

    return denoised


def auto_labeling(data_tuple, timestamp_tuple, w):
    data_list = np.asarray(data_tuple, dtype=float).flatten()
    timestamps = pd.Series(timestamp_tuple)

    if len(data_list) == 0:
        return np.array([], dtype=float)

    if len(data_list) != len(timestamps):
        m = min(len(data_list), len(timestamps))
        data_list = data_list[:m]
        timestamps = timestamps.iloc[:m].reset_index(drop=True)

    n = len(data_list)
    labels = np.zeros(n, dtype=float)

    FP = data_list[0]
    x_H = data_list[0]
    x_L = data_list[0]
    HT = timestamps.iloc[0]
    LT = timestamps.iloc[0]
    Cid = 0
    FP_N = 0

    for i in range(n):
        if data_list[i] > FP + FP * w:
            x_H = data_list[i]
            HT = timestamps.iloc[i]
            FP_N = i
            Cid = 1
            break
        if data_list[i] < FP - FP * w:
            x_L = data_list[i]
            LT = timestamps.iloc[i]
            FP_N = i
            Cid = -1
            break

    for i in range(max(FP_N, 1), n):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]
                HT = timestamps.iloc[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                mask = ((timestamps > LT) & (timestamps <= HT)).to_numpy()
                labels[mask] = 1
                x_L = data_list[i]
                LT = timestamps.iloc[i]
                Cid = -1

        elif Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]
                LT = timestamps.iloc[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                mask = ((timestamps > HT) & (timestamps <= LT)).to_numpy()
                labels[mask] = -1
                x_H = data_list[i]
                HT = timestamps.iloc[i]
                Cid = 1

    if Cid == 0:
        labels[:] = 0
    else:
        labels = np.where(labels == 0, Cid, labels)

    return labels


def strategy_returns_from_labels(prices, labels, tc_bps=22):
    prices = np.asarray(prices, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(prices) < 2 or len(labels) < 2:
        return np.array([], dtype=float)

    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]

    if len(pos) != len(asset_rets):
        m = min(len(pos), len(asset_rets))
        pos = pos[:m]
        asset_rets = asset_rets[:m]

    strat_rets = pos * asset_rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets = strat_rets.copy()
    strat_rets[position_changes] -= tc_bps / 10000.0
    strat_rets = strat_rets[np.isfinite(strat_rets)]
    return strat_rets


def evaluate_single_window(prices, requested_window, tc_bps, periods_per_year=252 * 24):
    prices = np.asarray(prices, dtype=float)
    eff_window = adaptive_window_size(len(prices), requested_window)

    if eff_window is None:
        return np.nan, None

    denoised = causal_wavelet_denoise(tuple(prices), eff_window)
    w = max(rogers_satchell_volatility_approx(prices) * 1.8, 1e-6)
    labels = auto_labeling(tuple(denoised), tuple(np.arange(len(prices))), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)

    if len(strat_rets) < 5:
        return np.nan, eff_window

    return sharpe_ratio(strat_rets, periods_per_year), eff_window


def prices_from_resampled_returns(base_prices, rng):
    base_prices = np.asarray(base_prices, dtype=float)
    if len(base_prices) < 2:
        return base_prices.copy()

    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    orig_rets = orig_rets[np.isfinite(orig_rets)]
    if len(orig_rets) == 0:
        return base_prices.copy()

    boot_rets = rng.choice(orig_rets, size=len(base_prices) - 1, replace=True)
    prices = np.empty(len(base_prices), dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


def build_equity_curve(prices, requested_window, tc_bps, timestamps=None):
    prices = np.asarray(prices, dtype=float)
    eff_window = adaptive_window_size(len(prices), requested_window)

    if eff_window is None or len(prices) < 20:
        return pd.DataFrame(), eff_window

    denoised = causal_wavelet_denoise(tuple(prices), eff_window)
    w = max(rogers_satchell_volatility_approx(prices) * 1.8, 1e-6)
    labels = auto_labeling(tuple(denoised), tuple(np.arange(len(prices))), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)

    if len(strat_rets) == 0:
        return pd.DataFrame(), eff_window

    equity = np.cumprod(1.0 + np.concatenate(([0.0], strat_rets)))

    if timestamps is None or len(timestamps) != len(equity):
        df_eq = pd.DataFrame({"Bar": np.arange(len(equity)), "Equity": equity})
    else:
        df_eq = pd.DataFrame({"Time": pd.to_datetime(timestamps[:len(equity)]), "Equity": equity})

    return df_eq, eff_window


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("Settings")
target_bars = st.sidebar.slider("Requested number of 1h bars", 500, 5000, 3500, step=100)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Requested Window Size", value=420, min_value=64, max_value=1000, step=20)
use_oos = st.sidebar.checkbox("Use Out-of-Sample Split", value=True)
oos_pct = st.sidebar.slider("OOS %", 20, 50, 35, step=5)
n_boot = st.sidebar.slider("Bootstrap Runs", 50, 1000, 300, step=50)

# ==============================================================================
# MAIN
# ==============================================================================
if st.button("🚀 Fetch Real BTC Data & Run Full Test", type="primary"):
    result = fetch_real_btc_data(target_bars=target_bars)

    if not result["ok"]:
        st.error(result["error"])
        st.stop()

    df = result["df"]
    prices = result["prices"]
    meta = result["meta"]

    if meta["history_limited"]:
        st.warning(
            f"Kraken returned {meta['fetched_bars']} bars (~{meta['days']:.1f} days), "
            f"not the requested {meta['requested_bars']}."
        )
    else:
        st.success(f"Fetched {meta['fetched_bars']} bars from Kraken.")

    min_required = 120
    if len(prices) < min_required:
        st.error(f"Not enough data to run. Need at least {min_required} bars, got {len(prices)}.")
        st.stop()

    if use_oos:
        split_idx = int(len(prices) * (1 - oos_pct / 100.0))
        split_idx = max(split_idx, 80)
        split_idx = min(split_idx, len(prices) - 80)

        if split_idx >= len(prices) - 20:
            st.error("OOS split left too little test data. Lower OOS %.")
            st.stop()

        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
        test_times = df["datetime"].iloc[split_idx:].reset_index(drop=True)

        train_sharpe, train_win = evaluate_single_window(train_prices, fixed_window, tc_bps)
        oos_sharpe, test_win = evaluate_single_window(test_prices, fixed_window, tc_bps)
        equity_df, eq_win = build_equity_curve(test_prices, fixed_window, tc_bps, timestamps=test_times)

    else:
        train_prices = prices
        test_prices = prices
        test_times = df["datetime"].reset_index(drop=True)

        train_sharpe, train_win = evaluate_single_window(prices, fixed_window, tc_bps)
        oos_sharpe, test_win = np.nan, None
        equity_df, eq_win = build_equity_curve(prices, fixed_window, tc_bps, timestamps=test_times)

    seed = 12345
    rng = np.random.default_rng(seed)

    progress_bar = st.progress(0, text="Running bootstrap...")
    boot_vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        boot_p = prices_from_resampled_returns(train_prices, rng)
        boot_sharpe, _ = evaluate_single_window(boot_p, fixed_window, tc_bps)
        boot_vals[i] = boot_sharpe if np.isfinite(boot_sharpe) else np.nan
        progress_bar.progress((i + 1) / n_boot, text=f"Running bootstrap... {i + 1}/{n_boot}")

    progress_bar.empty()

    boot_vals = boot_vals[np.isfinite(boot_vals)]
    if np.isfinite(train_sharpe) and len(boot_vals) > 10:
        p_value = np.mean(boot_vals >= train_sharpe)
        q95 = np.quantile(boot_vals, 0.95)
    else:
        p_value = np.nan
        q95 = np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetched Bars", f"{len(prices)}")
    c2.metric("Train Window Used", "n/a" if train_win is None else f"{train_win}")
    c3.metric("Train Sharpe", "nan" if not np.isfinite(train_sharpe) else f"{train_sharpe:.4f}")
    c4.metric("OOS Sharpe", "nan" if not np.isfinite(oos_sharpe) else f"{oos_sharpe:.4f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Equity Window Used", "n/a" if eq_win is None else f"{eq_win}")
    c6.metric("p-value", "nan" if not np.isfinite(p_value) else f"{p_value:.1%}")
    c7.metric("Boot 95% Quantile", "nan" if not np.isfinite(q95) else f"{q95:.4f}")

    if np.isfinite(train_sharpe) and np.isfinite(q95):
        if train_sharpe > q95:
            st.success("✅ Train Sharpe is above the 95% bootstrap quantile.")
        elif train_sharpe > np.nanmedian(boot_vals):
            st.info("ℹ️ Strategy is better than median bootstrap, but not strongly significant.")
        else:
            st.warning("⚠️ Strategy still looks fragile / likely overfit.")
    else:
        st.info("ℹ️ Bootstrap significance is inconclusive, but the strategy did run.")

    st.subheader("📈 Equity Curve")
    if equity_df.empty:
        st.warning("Could not build equity curve because the effective test sample is still too short.")
    else:
        if "Time" in equity_df.columns:
            st.line_chart(equity_df.set_index("Time")[["Equity"]], use_container_width=True)
        else:
            st.line_chart(equity_df.set_index("Bar")[["Equity"]], use_container_width=True)

    st.subheader("Recent BTC Data")
    st.dataframe(df[["datetime", "open", "high", "low", "close", "volume"]].tail(50), use_container_width=True)

else:
    st.info("👆 Click the button to fetch BTC data and run the test.")

st.caption("BTC/USD 1h data from Kraken via CCXT • Adaptive evaluation avoids empty OOS charts")
