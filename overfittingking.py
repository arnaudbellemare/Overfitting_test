import streamlit as st
import numpy as np
import pandas as pd
import pywt

st.set_page_config(page_title="Wavelet Auto-Labeling Overfitting Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling Overfitting Check")
st.markdown("""
This app performs a **bootstrap-based overfitting test** for the **causal wavelet denoising + trend auto-labeling** strategy 
(from your Streamlit dashboard Tab 3).  

It answers: *Is the best performance we get by tuning the causal window size real, or just data-mining luck?*
""")

# ==============================================================================
# Helper Functions
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
    if sd == 0.0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    if len(prices) < 2:
        return 0.01
    log_rets = np.log(prices[1:] / prices[:-1])
    return np.std(log_rets, ddof=1)


@st.cache_data
def causal_wavelet_denoise(prices_tuple, window_size):
    """Strictly causal wavelet denoising."""
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
    """Same trend auto-labeling logic as your dashboard."""
    data_list = np.asarray(data_tuple).flatten()
    timestamps = pd.Series(timestamp_tuple)

    if len(data_list) != len(timestamps):
        min_len = min(len(data_list), len(timestamps))
        data_list = data_list[:min_len]
        timestamps = timestamps.iloc[:min_len].reset_index(drop=True)

    n = len(data_list)
    labels = np.zeros(n)
    FP = data_list[0]
    x_H = data_list[0]
    HT = timestamps[0]
    x_L = data_list[0]
    LT = timestamps[0]
    Cid = 0
    FP_N = 0

    for i in range(n):
        if data_list[i] > FP + FP * w:
            x_H = data_list[i]; HT = timestamps[i]; FP_N = i; Cid = 1; break
        if data_list[i] < FP - FP * w:
            x_L = data_list[i]; LT = timestamps[i]; FP_N = i; Cid = -1; break

    for i in range(max(FP_N, 1), n):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]; HT = timestamps[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                mask = ((timestamps > LT) & (timestamps <= HT)).values
                labels[mask] = 1
                x_L = data_list[i]; LT = timestamps[i]; Cid = -1
        elif Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]; LT = timestamps[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                mask = ((timestamps > HT) & (timestamps <= LT)).values
                labels[mask] = -1
                x_H = data_list[i]; HT = timestamps[i]; Cid = 1

    labels = np.where(labels == 0, Cid, labels)
    return labels


def strategy_returns_from_labels(prices, labels):
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]  # one-bar lag → causal
    return pos * asset_rets


def test_wavelet_range(prices, window_sizes, periods_per_year=252):
    sharpes = np.full(len(window_sizes), np.nan, dtype=float)

    for i, window in enumerate(window_sizes):
        if window >= len(prices):
            continue
        denoised = causal_wavelet_denoise(tuple(prices), window)
        w = rogers_satchell_volatility_approx(prices)
        idx = np.arange(len(prices))
        labels = auto_labeling(tuple(denoised), tuple(idx), w)

        strat_rets = strategy_returns_from_labels(prices, labels)
        sharpes[i] = sharpe_ratio(strat_rets, periods_per_year)

    imax = np.nanargmax(sharpes)
    return {
        "windows": np.array(window_sizes, dtype=int),
        "sharpes": sharpes,
        "best_window": int(window_sizes[imax]),
        "best_sharpe": float(sharpes[imax]),
    }


def prices_from_resampled_returns(base_prices, rng):
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    boot_rets = rng.choice(orig_rets, size=orig_rets.size, replace=True)
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


# ==============================================================================
# Sidebar Configuration
# ==============================================================================
st.sidebar.header("⚙️ Simulation & Test Settings")

nobs = st.sidebar.slider("Number of observations (bars)", 300, 2000, 500, step=50)
mu_annual = st.sidebar.slider("Annual drift (μ)", 0.0, 0.30, 0.10, 0.01)
vol_annual = st.sidebar.slider("Annual volatility (σ)", 0.10, 0.60, 0.30, 0.01)

window_min = st.sidebar.slider("Min causal window size", 32, 128, 64, step=16)
window_max = st.sidebar.slider("Max causal window size", 128, 512, 384, step=16)
window_step = st.sidebar.slider("Window step size", 16, 64, 32, step=16)

n_boot = st.sidebar.slider("Bootstrap replications", 500, 5000, 2000, step=500)
periods_per_year = 252
seed = 12345

window_sizes = np.arange(window_min, window_max + 1, window_step)

if len(window_sizes) == 0 or window_sizes.max() >= nobs:
    st.error("Invalid window range – max window must be smaller than number of observations.")
    st.stop()

# ==============================================================================
# Run Button
# ==============================================================================
if st.button("🚀 Run Overfitting Test", type="primary"):
    rng = np.random.default_rng(seed)

    with st.spinner("Simulating price series and running real-data test..."):
        prices = simulate_gbm_prices(nobs, mu_annual, vol_annual, seed=seed)
        result_real = test_wavelet_range(prices, window_sizes, periods_per_year)

    progress_bar = st.progress(0, text="Running bootstrap replications...")
    best_boot_sharpes = np.empty(n_boot, dtype=float)
    best_boot_windows = np.empty(n_boot, dtype=int)

    for i in range(n_boot):
        boot_prices = prices_from_resampled_returns(prices, rng)
        result_boot = test_wavelet_range(boot_prices, window_sizes, periods_per_year)
        best_boot_sharpes[i] = result_boot["best_sharpe"]
        best_boot_windows[i] = result_boot["best_window"]

        if (i + 1) % max(1, n_boot // 20) == 0:
            progress_bar.progress((i + 1) / n_boot, text=f"Bootstrap {i+1}/{n_boot}...")

    progress_bar.empty()

    # ==============================================================================
    # Results
    # ==============================================================================
    p_value = np.mean(best_boot_sharpes >= result_real["best_sharpe"])
    q50, q90, q95, q99 = np.quantile(best_boot_sharpes, [0.50, 0.90, 0.95, 0.99])
    imax_boot = np.argmax(best_boot_sharpes)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Original Simulated Series")
        st.metric("Best causal window", f"{result_real['best_window']} bars")
        st.metric("Best Sharpe ratio", f"{result_real['best_sharpe']:.4f}")

    with col2:
        st.subheader("📊 Bootstrap Distribution (Random Data)")
        st.metric("Median best Sharpe", f"{q50:.4f}")
        st.metric("90th percentile", f"{q90:.4f}")
        st.metric("95th percentile", f"{q95:.4f}")
        st.metric("99th percentile", f"{q99:.4f}")
        st.metric("p-value (Pr[random ≥ real])", f"{p_value:.4%}")

    st.markdown("---")

    if result_real["best_sharpe"] > q95:
        st.success("✅ The best Sharpe stands out significantly in the tail → possible genuine edge (still needs real-market validation).")
    else:
        st.warning("⚠️ The best Sharpe is **not unusual** compared to what parameter search finds on pure noise. Strong evidence of overfitting.")

    # Top windows on original data
    topn = min(10, len(window_sizes))
    idx = np.argsort(result_real["sharpes"])[::-1][:topn]

    st.subheader(f"🏆 Top {topn} Causal Window Sizes on Original Series")
    df_top = pd.DataFrame({
        "Causal Window": result_real["windows"][idx],
        "Sharpe Ratio": result_real["sharpes"][idx]
    })
    st.dataframe(df_top.style.format({"Sharpe Ratio": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.info("""
    **Interpretation**:  
    This test shows how much performance you can "manufacture" just by searching over different causal window sizes on random data.  
    If your real-data best Sharpe is only around the median or 90th percentile of the bootstrap, the rule is likely overfit.
    """)

    # Optional: Show all sharpes plot
    if st.checkbox("Show Sharpe vs Window Size plot (original series)"):
        fig_df = pd.DataFrame({
            "Window Size": result_real["windows"],
            "Sharpe": result_real["sharpes"]
        })
        st.line_chart(fig_df.set_index("Window Size"))

else:
    st.info("Adjust parameters in the sidebar and click **Run Overfitting Test** to begin.")

st.caption("Built with causal wavelet denoising + auto-labeling from your dashboard. Uses 252 trading days/year and one-bar lag for causality.")
