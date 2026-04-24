import streamlit as st
import numpy as np
import pandas as pd
import pywt
import ccxt
from datetime import datetime, timedelta

st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling on Real BTC Data")
st.markdown("**Live Kraken BTC/USD 1h data with pagination**")

# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_real_btc_data(limit=5000):
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        all_ohlcv = []
        since = None
        remaining = limit

        with st.spinner(f"Fetching up to {limit} BTC 1h bars..."):
            while remaining > 0:
                ohlcv = exchange.fetch_ohlcv(
                    'BTC/USD', 
                    timeframe='1h', 
                    limit=min(1000, remaining),
                    since=since
                )
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # next batch
                remaining -= len(ohlcv)
                
                if len(ohlcv) < 1000:
                    break  # no more data

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        
        prices = df['close'].values.astype(float)
        days = len(prices) // 24
        
        st.success(f"✅ Successfully fetched **{len(prices)}** real BTC 1h bars (~{days} days)")
        return prices
        
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None


# ==============================================================================
# CORE FUNCTIONS (same as before)
# ==============================================================================
def sharpe_ratio(rets, periods_per_year=252*24):
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    if len(prices) < 2:
        return 0.02
    log_rets = np.log(prices[1:] / prices[:-1])
    return np.std(log_rets, ddof=1)


@st.cache_data
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


def strategy_returns_from_labels(prices, labels, tc_bps=22):
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
    w = rogers_satchell_volatility_approx(prices) * 1.8
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
st.sidebar.header("Settings")
limit = st.sidebar.slider("Target number of 1h bars", 1000, 6000, 4000, step=200)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Fixed Window Size", value=420, min_value=250, max_value=600, step=20)
use_oos = st.sidebar.checkbox("Use Out-of-Sample", value=True)
oos_pct = st.sidebar.slider("OOS %", 30, 45, 35, step=5)

# ==============================================================================
if st.button("🚀 Fetch Real BTC Data & Run Test", type="primary"):
    prices = fetch_real_btc_data(limit)
    if prices is None or len(prices) < 800:
        st.stop()

    seed = 12345
    rng = np.random.default_rng(seed)

    with st.spinner("Running test on real BTC data..."):
        split_idx = int(len(prices) * (1 - oos_pct / 100))
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        best_win = fixed_window
        train_sharpe = evaluate_single_window(train_prices, best_win, tc_bps)
        oos_sharpe = evaluate_single_window(test_prices, best_win, tc_bps)

    # Bootstrap
    progress_bar = st.progress(0, text="Bootstrap...")
    n_boot = 600
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
        st.metric("Fixed Window", f"{best_win}")
        st.metric("Train Sharpe", f"{train_sharpe:.4f}")
    with col2:
        st.metric("OOS Sharpe", f"{oos_sharpe:.4f}")
        st.metric("p-value", f"{p_value:.1%}")

    if train_sharpe > q95 * 0.9:
        st.success("✅ Relatively good on real BTC")
    else:
        st.warning("⚠️ Still overfitting (expected for this method on BTC)")

    # Equity Curve
    st.subheader("📈 Equity Curve - Recent Out-of-Sample")
    denoised = causal_wavelet_denoise(tuple(test_prices), best_win)
    w = rogers_satchell_volatility_approx(test_prices) * 1.8
    labels = auto_labeling(tuple(denoised), tuple(np.arange(len(test_prices))), w)
    strat_rets = strategy_returns_from_labels(test_prices, labels, tc_bps)
    equity = np.cumprod(1 + np.concatenate(([0.], strat_rets)))

    st.line_chart(pd.DataFrame({"Equity": equity}), use_container_width=True)

else:
    st.info("Click the button to fetch real BTC data")

st.caption("Pagination-enabled fetch from Kraken • 1h BTC/USD")
