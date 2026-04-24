import streamlit as st
import numpy as np
import pandas as pd
import pywt
import ccxt
import time

st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling on Real BTC")
st.markdown("Live data from Kraken • Honest history limits • High transaction costs")

# ==============================================================================
# HELPERS
# ==============================================================================
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def sharpe_ratio(rets, periods_per_year=252 * 24):
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
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
    if not np.isfinite(vol) or vol <= 0:
        return 0.02
    return vol


# ==============================================================================
# FETCH REAL BTC DATA
# ==============================================================================
@st.cache_data(ttl=1800)
def fetch_real_btc_data(target_bars=3500, symbol="BTC/USD", timeframe="1h"):
    """
    Kraken often exposes only a recent OHLC window.
    This function fetches as much as Kraken actually serves and reports honestly.
    """
    try:
        exchange = ccxt.kraken({
            "enableRateLimit": True,
            "timeout": 30000,
        })
        exchange.load_markets()

        if symbol not in exchange.symbols:
            raise ValueError(f"{symbol} is not available on Kraken")

        if not exchange.has.get("fetchOHLCV", False):
            raise ValueError("Exchange does not support fetchOHLCV")

        if timeframe not in exchange.timeframes:
            raise ValueError(f"Timeframe {timeframe} not supported by Kraken")

        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        now = exchange.milliseconds()

        # Ask for target depth ago, but Kraken may ignore deep history.
        since = now - target_bars * tf_ms

        all_rows = []
        seen_ts = set()
        max_calls = 20
        limit = 720  # practical recent-window size for Kraken behavior
        last_last_ts = None

        status_lines = []
        calls_used = 0

        for _ in range(max_calls):
            calls_used += 1
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if not candles:
                status_lines.append("No candles returned; stopping.")
                break

            batch_df = pd.DataFrame(
                candles, columns=["ts", "open", "high", "low", "close", "volume"]
            ).drop_duplicates(subset="ts").sort_values("ts")

            if batch_df.empty:
                status_lines.append("Empty batch after deduplication; stopping.")
                break

            batch_rows = batch_df.values.tolist()
            new_rows = [row for row in batch_rows if row[0] not in seen_ts]

            first_ts = int(batch_df["ts"].iloc[0])
            last_ts = int(batch_df["ts"].iloc[-1])

            status_lines.append(
                f"Call {calls_used}: got {len(batch_rows)} rows "
                f"from {exchange.iso8601(first_ts)} to {exchange.iso8601(last_ts)}"
            )

            if not new_rows:
                status_lines.append("No new timestamps returned; stopping.")
                break

            for row in new_rows:
                seen_ts.add(row[0])
            all_rows.extend(new_rows)

            if last_last_ts is not None and last_ts <= last_last_ts:
                status_lines.append("Pagination stalled (last timestamp did not advance); stopping.")
                break

            last_last_ts = last_ts
            since = last_ts + tf_ms

            if len(all_rows) >= target_bars:
                status_lines.append(f"Reached requested target of {target_bars} bars.")
                break

            time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.25)

        if not all_rows:
            return {
                "ok": False,
                "error": "No candles returned from Kraken.",
                "df": None,
                "prices": None,
                "meta": {"logs": status_lines},
            }

        df = pd.DataFrame(
            all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
        ).drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"]).reset_index(drop=True)
        prices = df["close"].to_numpy(dtype=float)

        available_bars = len(df)
        available_days = available_bars / 24.0

        meta = {
            "exchange": "Kraken",
            "symbol": symbol,
            "timeframe": timeframe,
            "requested_bars": int(target_bars),
            "fetched_bars": int(available_bars),
            "days": float(available_days),
            "first_ts": int(df["ts"].iloc[0]),
            "last_ts": int(df["ts"].iloc[-1]),
            "first_iso": exchange.iso8601(int(df["ts"].iloc[0])),
            "last_iso": exchange.iso8601(int(df["ts"].iloc[-1])),
            "logs": status_lines,
            "history_limited": available_bars < target_bars,
        }

        return {
            "ok": True,
            "error": None,
            "df": df,
            "prices": prices,
            "meta": meta,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "df": None,
            "prices": None,
            "meta": {"logs": []},
        }


# ==============================================================================
# WAVELET / LABELING
# ==============================================================================
@st.cache_data(show_spinner=False)
def causal_wavelet_denoise(prices_tuple, window_size):
    prices = np.asarray(prices_tuple, dtype=float)

    if len(prices) == 0:
        return prices.copy()

    denoised = prices.copy()

    wavelet = "db4"
    max_level = pywt.dwt_max_level(window_size, pywt.Wavelet(wavelet).dec_len)
    level = min(4, max_level) if max_level >= 1 else 1

    for i in range(window_size - 1, len(prices)):
        window_data = prices[i - window_size + 1:i + 1]

        coeffs = pywt.wavedec(window_data, wavelet, level=level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 if len(detail) else 0.0
        uthresh = sigma * np.sqrt(2 * np.log(len(window_data))) if sigma > 0 else 0.0

        coeffs_thresh = [coeffs[0]]
        for c in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(c, uthresh, mode="soft"))

        window_denoised = pywt.waverec(coeffs_thresh, wavelet)
        window_denoised = np.asarray(window_denoised).flatten()[:len(window_data)]
        denoised[i] = window_denoised[-1]

    return denoised


def auto_labeling(data_tuple, timestamp_tuple, w):
    data_list = np.asarray(data_tuple, dtype=float).flatten()
    timestamps = pd.Series(timestamp_tuple)

    if len(data_list) == 0:
        return np.array([], dtype=float)

    if len(data_list) != len(timestamps):
        min_len = min(len(data_list), len(timestamps))
        data_list = data_list[:min_len]
        timestamps = timestamps.iloc[:min_len].reset_index(drop=True)

    n = len(data_list)
    labels = np.zeros(n, dtype=float)

    FP = data_list[0]
    x_H = data_list[0]
    HT = timestamps.iloc[0]
    x_L = data_list[0]
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

    labels = np.where(labels == 0, Cid, labels)
    return labels


def strategy_returns_from_labels(prices, labels, tc_bps=22):
    prices = np.asarray(prices, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(prices) < 2 or len(labels) < 2:
        return np.array([], dtype=float)

    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]
    strat_rets = pos * asset_rets

    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets = strat_rets.copy()
    strat_rets[position_changes] -= tc_bps / 10000.0

    strat_rets = strat_rets[np.isfinite(strat_rets)]
    return strat_rets


def evaluate_single_window(prices, window_size, tc_bps, periods_per_year=252 * 24):
    prices = np.asarray(prices, dtype=float)

    if len(prices) <= max(window_size, 10):
        return np.nan

    denoised = causal_wavelet_denoise(tuple(prices), int(window_size))
    w = rogers_satchell_volatility_approx(prices) * 1.8
    w = max(w, 1e-6)

    idx = np.arange(len(prices))
    labels = auto_labeling(tuple(denoised), tuple(idx), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)

    if len(strat_rets) < 2:
        return np.nan

    return sharpe_ratio(strat_rets, periods_per_year)


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


def build_equity_curve(prices, window_size, tc_bps):
    prices = np.asarray(prices, dtype=float)
    if len(prices) <= max(window_size, 10):
        return pd.DataFrame()

    denoised = causal_wavelet_denoise(tuple(prices), int(window_size))
    w = rogers_satchell_volatility_approx(prices) * 1.8
    w = max(w, 1e-6)

    labels = auto_labeling(tuple(denoised), tuple(np.arange(len(prices))), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)

    if len(strat_rets) == 0:
        return pd.DataFrame()

    equity = np.cumprod(1.0 + np.concatenate(([0.0], strat_rets)))
    out = pd.DataFrame({
        "Bar": np.arange(len(equity)),
        "Equity": equity
    })
    return out


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("Settings")
target_bars = st.sidebar.slider("Requested number of 1h bars", 500, 5000, 3500, step=100)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Fixed Window Size", value=420, min_value=64, max_value=1000, step=20)
use_oos = st.sidebar.checkbox("Use Out-of-Sample Split", value=True)
oos_pct = st.sidebar.slider("OOS %", 20, 50, 35, step=5)
n_boot = st.sidebar.slider("Bootstrap Runs", 50, 1000, 300, step=50)

st.sidebar.caption("Kraken may return only recent OHLC history for 1h candles.")

# ==============================================================================
# MAIN RUN
# ==============================================================================
if st.button("🚀 Fetch Real BTC Data & Run Full Test", type="primary"):
    with st.spinner("Fetching real BTC data from Kraken..."):
        result = fetch_real_btc_data(target_bars=target_bars, symbol="BTC/USD", timeframe="1h")

    if not result["ok"]:
        st.error(result["error"])
        st.stop()

    df = result["df"]
    prices = result["prices"]
    meta = result["meta"]

    if meta["history_limited"]:
        st.warning(
            f"Kraken returned only {meta['fetched_bars']} bars (~{meta['days']:.1f} days), "
            f"not the requested {meta['requested_bars']} bars."
        )
    else:
        st.success(
            f"Fetched {meta['fetched_bars']} bars (~{meta['days']:.1f} days) "
            f"from {meta['first_iso']} to {meta['last_iso']}."
        )

    with st.expander("Fetch details"):
        st.write(f"Exchange: {meta['exchange']}")
        st.write(f"Symbol: {meta['symbol']}")
        st.write(f"Timeframe: {meta['timeframe']}")
        st.write(f"First candle: {meta['first_iso']}")
        st.write(f"Last candle: {meta['last_iso']}")
        for line in meta["logs"]:
            st.text(line)

    min_required = max(int(fixed_window) + 50, 300)
    if len(prices) < min_required:
        st.error(
            f"Not enough data to run. Need at least {min_required} bars for window={fixed_window}, "
            f"but got {len(prices)}."
        )
        st.stop()

    seed = 12345
    rng = np.random.default_rng(seed)

    with st.spinner("Running strategy evaluation..."):
        if use_oos:
            split_idx = int(len(prices) * (1 - oos_pct / 100.0))
            split_idx = max(split_idx, fixed_window + 25)
            split_idx = min(split_idx, len(prices) - max(100, fixed_window // 2))

            if split_idx <= fixed_window or split_idx >= len(prices) - 20:
                st.error("Invalid split after applying safety checks. Try a smaller window or lower OOS %.")
                st.stop()

            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]

            train_sharpe = evaluate_single_window(train_prices, fixed_window, tc_bps)
            oos_sharpe = evaluate_single_window(test_prices, fixed_window, tc_bps)

            equity_df = build_equity_curve(test_prices, fixed_window, tc_bps)
        else:
            train_prices = prices
            test_prices = prices

            train_sharpe = evaluate_single_window(prices, fixed_window, tc_bps)
            oos_sharpe = np.nan

            equity_df = build_equity_curve(prices, fixed_window, tc_bps)

    progress_bar = st.progress(0, text="Running bootstrap...")
    best_boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        boot_p = prices_from_resampled_returns(train_prices, rng)
        best_boot[i] = evaluate_single_window(boot_p, fixed_window, tc_bps)
        progress_bar.progress((i + 1) / n_boot, text=f"Running bootstrap... {i + 1}/{n_boot}")

    progress_bar.empty()

    valid_boot = best_boot[np.isfinite(best_boot)]
    if len(valid_boot) == 0 or not np.isfinite(train_sharpe):
        p_value = np.nan
        q95 = np.nan
    else:
        p_value = np.mean(valid_boot >= train_sharpe)
        q95 = np.quantile(valid_boot, 0.95)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fetched Bars", f"{len(prices)}")
    col2.metric("Fixed Window", f"{fixed_window}")
    col3.metric("Train Sharpe", "nan" if not np.isfinite(train_sharpe) else f"{train_sharpe:.4f}")
    col4.metric("OOS Sharpe", "nan" if not np.isfinite(oos_sharpe) else f"{oos_sharpe:.4f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Bootstrap Runs", f"{n_boot}")
    col6.metric("p-value", "nan" if not np.isfinite(p_value) else f"{p_value:.1%}")
    col7.metric("Boot 95% Quantile", "nan" if not np.isfinite(q95) else f"{q95:.4f}")

    if np.isfinite(train_sharpe) and np.isfinite(q95):
        if train_sharpe > q95 * 0.9:
            st.success("✅ Result looks relatively strong versus bootstrap baseline.")
        else:
            st.warning("⚠️ Strategy still looks fragile / likely overfit.")
    else:
        st.warning("⚠️ Could not compute a stable bootstrap comparison.")

    st.subheader("📈 Equity Curve")
    if equity_df.empty:
        st.warning("Could not build equity curve for the selected setup.")
    else:
        st.line_chart(equity_df.set_index("Bar")[["Equity"]], use_container_width=True)

    st.subheader("Price Sample")
    preview = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    st.dataframe(preview.tail(50), use_container_width=True)

else:
    st.info("👆 Click the button to fetch real BTC data and run the test.")

st.caption("BTC/USD 1h data from Kraken via CCXT • Fetches actual available history only")
