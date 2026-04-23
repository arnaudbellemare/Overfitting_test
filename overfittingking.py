import streamlit as st
import numpy as np
import pandas as pd
import pywt

st.set_page_config(page_title="Wavelet Overfitting Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling Overfitting Check")
st.markdown("**Conservative Mode** — Higher costs + Fixed window to fight overfitting")

# ==============================================================================
# HELPER FUNCTIONS (same as before)
# ==============================================================================
def simulate_gbm_prices(nobs, mu_annual, vol_annual, s0=100.0, periods_per_year=252, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    drift = (mu_annual - 0.5 * vol_annual ** 2) * dt
    shock_scale = vol_annual * np.sqrt(dt)
    log_rets = drift + shock_scale * rng.standard_normal(nobs - 1)
    prices = np.empty(nobs, dtype=float)
    prices[0] = s0
    prices[1:] = s0 * np.exp(np.cumsum(log_rets))
    return prices


def sharpe_ratio(rets, periods_per_year=252):
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    if len(prices) < 2:
        return 0.01
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


def strategy_returns_from_labels(prices, labels, tc_bps=15):
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]
    strat_rets = pos * asset_rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets[position_changes] -= tc_bps / 10000.0
    return strat_rets


def evaluate_single_window(prices, window_size, tc_bps, periods_per_year=252):
    if window_size >= len(prices):
        return np.nan
    denoised = causal_wavelet_denoise(tuple(prices), window_size)
    w = rogers_satchell_volatility_approx(prices)
    idx = np.arange(len(prices))
    labels = auto_labeling(tuple(denoised), tuple(idx), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)
    return sharpe_ratio(strat_rets, periods_per_year)


def test_wavelet_range(prices, window_sizes, tc_bps, periods_per_year=252):
    sharpes = np.full(len(window_sizes), np.nan, dtype=float)
    for i, window in enumerate(window_sizes):
        if window >= len(prices): continue
        sharpes[i] = evaluate_single_window(prices, window, tc_bps, periods_per_year)
    imax = np.nanargmax(sharpes)
    return {"windows": np.array(window_sizes, dtype=int), "sharpes": sharpes,
            "best_window": int(window_sizes[imax]), "best_sharpe": float(sharpes[imax])}


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
st.sidebar.header("⚙️ Conservative Settings")

data_mode = st.sidebar.radio("Data Source", ["Simulated GBM", "Upload CSV (BTC etc.)"])

if data_mode == "Simulated GBM":
    nobs = st.sidebar.slider("Number of bars", 1000, 5000, 2500, step=100)
    mu_annual = st.sidebar.slider("Annual drift μ", 0.0, 0.30, 0.08, 0.01)
    vol_annual = st.sidebar.slider("Annual volatility σ", 0.10, 0.80, 0.45, 0.01)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'close' column", type=["csv"])
    if uploaded_file is None:
        st.warning("Upload CSV...")
        st.stop()
    df = pd.read_csv(uploaded_file)
    prices = df['close'].dropna().values.astype(float)
    st.sidebar.success(f"Loaded {len(prices)} bars")

tc_bps = st.sidebar.slider("Transaction cost (bps)", 5, 30, 15, step=1)
use_oos = st.sidebar.checkbox("Use true Out-of-Sample", value=True)
oos_pct = st.sidebar.slider("OOS %", 25, 40, 30, step=5)

fixed_window = st.sidebar.number_input("Fixed Causal Window", value=320, min_value=128, max_value=512, step=32)

n_boot = st.sidebar.slider("Bootstrap replications", 500, 1500, 800, step=100)

# ==============================================================================
# RUN
# ==============================================================================
if st.button("🚀 Run Conservative Overfitting Test", type="primary"):
    seed = 12345
    rng = np.random.default_rng(seed)

    with st.spinner("Running..."):
        if data_mode == "Simulated GBM":
            prices = simulate_gbm_prices(nobs, mu_annual, vol_annual, seed=seed)

        split_idx = int(len(prices) * (1 - oos_pct / 100))
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        # Fixed window only (no search)
        best_win = fixed_window
        train_sharpe = evaluate_single_window(train_prices, best_win, tc_bps)
        result_train = {"best_window": best_win, "best_sharpe": train_sharpe,
                        "windows": np.array([best_win]), "sharpes": np.array([train_sharpe])}

        full_sharpe = evaluate_single_window(prices, best_win, tc_bps)
        oos_sharpe = evaluate_single_window(test_prices, best_win, tc_bps)

    # Bootstrap
    progress_bar = st.progress(0, text="Bootstrap...")
    best_boot_sharpes = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_prices = prices_from_resampled_returns(train_prices, rng)
        best_boot_sharpes[i] = evaluate_single_window(boot_prices, best_win, tc_bps)
        if (i + 1) % max(1, n_boot // 10) == 0:
            progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    p_value = np.mean(best_boot_sharpes >= result_train["best_sharpe"])
    q50, q90, q95, q99 = np.quantile(best_boot_sharpes, [0.50, 0.90, 0.95, 0.99])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fixed Window", f"{best_win} bars")
        st.metric("Train Sharpe", f"{result_train['best_sharpe']:.4f}")
    with col2:
        st.metric("OOS Sharpe", f"{oos_sharpe:.4f}")
        st.metric("p-value", f"{p_value:.1%}")

    if result_train["best_sharpe"] > q95:
        st.success("✅ Low risk of overfitting")
    else:
        st.warning("⚠️ Still overfitting — this method is very strong at fitting noise")

    # Auto Equity Curve
    st.subheader("📈 Equity Curve on Out-of-Sample Data")
    denoised = causal_wavelet_denoise(tuple(test_prices), best_win)
    w = rogers_satchell_volatility_approx(test_prices)
    idx = np.arange(len(test_prices))
    labels = auto_labeling(tuple(denoised), tuple(idx), w)
    strat_rets = strategy_returns_from_labels(test_prices, labels, tc_bps)
    equity = np.cumprod(1 + np.concatenate(([0.], strat_rets)))

    eq_df = pd.DataFrame({"Bar": np.arange(len(equity)), "Equity": equity})
    st.line_chart(eq_df.set_index("Bar"), use_container_width=True)

    st.info(f"**Current Settings:** Fixed window = {best_win}, TC = {tc_bps} bps, OOS = {oos_pct}%")

else:
    st.info("Adjust settings on the left and click **Run**")

st.caption("Conservative setup to reduce overfitting on Bitcoin-type data")
