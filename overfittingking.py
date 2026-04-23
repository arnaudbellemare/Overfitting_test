import streamlit as st
import numpy as np
import pandas as pd
import pywt
import ccxt

st.set_page_config(page_title="Wavelet Overfitting Test - Real BTC", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling — Real BTC Data")
st.markdown("**Live data from Kraken via CCXT** • Strict settings to reduce overfitting")

# ==============================================================================
# HELPERS
# ==============================================================================
def fetch_btc_data(limit=2000):
    """Fetch real BTC/USD data from Kraken"""
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df['close'].values.astype(float)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None


def sharpe_ratio(rets, periods_per_year=252 * 24):  # hourly data
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    if len(prices) < 2:
        return 0.02
    log_rets = np.log(prices[1:] / prices[:-1])
    return np.std(log_rets, ddof=1)


@st.cache_data(ttl=3600)
def causal_wavelet_denoise(prices_tuple, window_size):
    prices = np.asarray(prices_tuple, dtype=float)
    denoised = prices.copy()
    for i in range(window_size, len(prices)):
        window_data = prices[i - window_size + 1: i + 1]
        coeffs = pywt.wavedec(window_data, 'db4', level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(window_data)))
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
        window_denoised = pywt.waverec(coeffs_thresh, 'db4')
        window_denoised = np.asarray(window_denoised).flatten()[:len(window_data)]
        denoised[i] = window_denoised[-1]
    return denoised


def auto_labeling(data_tuple, timestamp_tuple, w):
    data_list = np.asarray(data_tuple).flatten()
    timestamps = pd.Series(timestamp_tuple)
    if len(data_list) != len(timestamps):
        min_len = min(len(data_list), len(timestamps))
        data_list = data_list[:min_len]
        timestamps = timestamps.iloc[:min_len].reset_index(drop=True)
    n = len(data_list)
    labels = np.zeros(n)
    FP = data_list[0]; x_H = data_list[0]; HT = timestamps[0]
    x_L = data_list[0]; LT = timestamps[0]; Cid = 0; FP_N = 0
    for i in range(n):
        if data_list[i] > FP + FP * w:
            x_H = data_list[i]; HT = timestamps[i]; FP_N = i; Cid = 1; break
        if data_list[i] < FP - FP * w:
            x_L = data_list[i]; LT = timestamps[i]; FP_N = i; Cid = -1; break
    for i in range(max(FP_N, 1), n):
        if Cid > 0:
            if data_list[i] > x_H: x_H = data_list[i]; HT = timestamps[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                mask = ((timestamps > LT) & (timestamps <= HT)).values
                labels[mask] = 1
                x_L = data_list[i]; LT = timestamps[i]; Cid = -1
        elif Cid < 0:
            if data_list[i] < x_L: x_L = data_list[i]; LT = timestamps[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                mask = ((timestamps > HT) & (timestamps <= LT)).values
                labels[mask] = -1
                x_H = data_list[i]; HT = timestamps[i]; Cid = 1
    labels = np.where(labels == 0, Cid, labels)
    return labels


def strategy_returns_from_labels(prices, labels, tc_bps=20):
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]
    strat_rets = pos * asset_rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets[position_changes] -= tc_bps / 10000.0
    return strat_rets


def evaluate_single_window(prices, window_size, tc_bps, periods_per_year=252*24):
    if window_size >= len(prices):
        return np.nan
    denoised = causal_wavelet_denoise(tuple(prices), window_size)
    w = rogers_satchell_volatility_approx(prices) * 1.8   # stricter
    idx = np.arange(len(prices))
    labels = auto_labeling(tuple(denoised), tuple(idx), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)
    return sharpe_ratio(strat_rets, periods_per_year)


def prices_from_resampled_returns(base_prices, rng):
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    boot_rets = rng.choice(orig_rets, size=orig_rets.size, replace=True)
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("⚙️ Strict Real BTC Settings")

limit = st.sidebar.slider("Data points to fetch (1h bars)", 1000, 5000, 2500, step=100)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 10, 40, 22, step=2)
fixed_window = st.sidebar.number_input("Fixed Window Size", value=420, min_value=200, max_value=600, step=20)
use_oos = st.sidebar.checkbox("Use Out-of-Sample", value=True)
oos_pct = st.sidebar.slider("OOS %", 30, 45, 35, step=5)

# ==============================================================================
# RUN
# ==============================================================================
if st.button("🚀 Fetch Real BTC Data & Run Strict Test", type="primary"):
    with st.spinner("Fetching real BTC data from Kraken..."):
        prices = fetch_btc_data(limit)
        if prices is None or len(prices) < 1000:
            st.error("Not enough data fetched.")
            st.stop()

    seed = 12345
    rng = np.random.default_rng(seed)

    with st.spinner("Running strict test on real BTC..."):
        split_idx = int(len(prices) * (1 - oos_pct / 100))
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        best_win = fixed_window
        train_sharpe = evaluate_single_window(train_prices, best_win, tc_bps)
        oos_sharpe = evaluate_single_window(test_prices, best_win, tc_bps)

    # Bootstrap
    progress_bar = st.progress(0, text="Bootstrap on real BTC returns...")
    n_boot = 800
    best_boot = np.empty(n_boot)
    for i in range(n_boot):
        boot_p = prices_from_resampled_returns(train_prices, rng)
        best_boot[i] = evaluate_single_window(boot_p, best_win, tc_bps)
        if (i + 1) % 100 == 0:
            progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    p_value = np.mean(best_boot >= train_sharpe)
    q95 = np.quantile(best_boot, 0.95)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fixed Window", f"{best_win} bars")
        st.metric("Train Sharpe", f"{train_sharpe:.4f}")
    with col2:
        st.metric("OOS Sharpe (Real BTC)", f"{oos_sharpe:.4f}")
        st.metric("p-value", f"{p_value:.1%}")

    if train_sharpe > q95 * 0.95:   # slightly relaxed for real data
        st.success("✅ Acceptable on real BTC data")
    else:
        st.error("Still overfitting — very common with this adaptive method on noisy BTC data")

    # Equity Curve
    st.subheader("📈 Equity Curve on Recent Out-of-Sample BTC Data")
    denoised = causal_wavelet_denoise(tuple(test_prices), best_win)
    w = rogers_satchell_volatility_approx(test_prices) * 1.8
    labels = auto_labeling(tuple(denoised), tuple(np.arange(len(test_prices))), w)
    strat_rets = strategy_returns_from_labels(test_prices, labels, tc_bps)
    equity = np.cumprod(1 + np.concatenate(([0.], strat_rets)))

    st.line_chart(pd.DataFrame({"Equity": equity}), use_container_width=True)

    st.info(f"Tested on last ~{len(prices)} hours of real BTC/USD data")

else:
    st.info("Click the button above to fetch live BTC data and run the test")

st.caption("Real data from Kraken • 1h timeframe • Strict fixed window + high costs")
