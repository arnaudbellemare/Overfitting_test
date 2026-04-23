import streamlit as st
import numpy as np
import pandas as pd
import pywt

st.set_page_config(page_title="Wavelet Auto-Labeling Overfitting Test", layout="wide")
st.title("🌊 Causal Wavelet + Auto-Labeling Overfitting Check")
st.markdown("""
**Now with transaction costs + true Out-of-Sample evaluation**  
This version greatly reduces the chance of seeing inflated Sharpes due to overfitting.
""")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def simulate_gbm_prices(nobs, mu_annual, vol_annual, s0=100.0, periods_per_year=252, seed=None):
    """Simulate GBM price series."""
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
    """Annualized Sharpe (rf = 0)."""
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods_per_year) * np.mean(rets) / sd


def rogers_satchell_volatility_approx(prices):
    """Close-only volatility proxy used as threshold w."""
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
    """Exact same trend auto-labeling logic as your original dashboard."""
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
            x_H = data_list[i]
            HT = timestamps[i]
            FP_N = i
            Cid = 1
            break
        if data_list[i] < FP - FP * w:
            x_L = data_list[i]
            LT = timestamps[i]
            FP_N = i
            Cid = -1
            break

    for i in range(max(FP_N, 1), n):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]
                HT = timestamps[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                mask = ((timestamps > LT) & (timestamps <= HT)).values
                labels[mask] = 1
                x_L = data_list[i]
                LT = timestamps[i]
                Cid = -1
        elif Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]
                LT = timestamps[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                mask = ((timestamps > HT) & (timestamps <= LT)).values
                labels[mask] = -1
                x_H = data_list[i]
                HT = timestamps[i]
                Cid = 1

    labels = np.where(labels == 0, Cid, labels)
    return labels


def strategy_returns_from_labels(prices, labels, tc_bps=10):
    """Compute strategy returns with one-bar lag + transaction costs."""
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]  # causal lag
    strat_rets = pos * asset_rets

    # Transaction costs on every position change
    # Prepend initial zero position to correctly detect entry at the very first bar
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    # position_changes now has exactly the same length as strat_rets (N-1)
    strat_rets[position_changes] -= tc_bps / 10000.0
    return strat_rets


def evaluate_single_window(prices, window_size, tc_bps, periods_per_year=252):
    """Evaluate one fixed window (used for OOS and final reporting)."""
    if window_size >= len(prices):
        return np.nan
    denoised = causal_wavelet_denoise(tuple(prices), window_size)
    w = rogers_satchell_volatility_approx(prices)
    idx = np.arange(len(prices))
    labels = auto_labeling(tuple(denoised), tuple(idx), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)
    return sharpe_ratio(strat_rets, periods_per_year)


def test_wavelet_range(prices, window_sizes, tc_bps, periods_per_year=252):
    """Optimize over window sizes (used on training data only)."""
    sharpes = np.full(len(window_sizes), np.nan, dtype=float)
    for i, window in enumerate(window_sizes):
        if window >= len(prices):
            continue
        sharpes[i] = evaluate_single_window(prices, window, tc_bps, periods_per_year)
    imax = np.nanargmax(sharpes)
    return {
        "windows": np.array(window_sizes, dtype=int),
        "sharpes": sharpes,
        "best_window": int(window_sizes[imax]),
        "best_sharpe": float(sharpes[imax]),
    }


def prices_from_resampled_returns(base_prices, rng):
    """Bootstrap: resample returns with replacement."""
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    boot_rets = rng.choice(orig_rets, size=orig_rets.size, replace=True)
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("⚙️ Simulation & Test Settings")

nobs = st.sidebar.slider("Number of bars (observations)", 500, 5000, 2000, step=100)
mu_annual = st.sidebar.slider("Annual drift μ", 0.0, 0.30, 0.10, 0.01)
vol_annual = st.sidebar.slider("Annual volatility σ", 0.10, 0.60, 0.30, 0.01)
tc_bps = st.sidebar.slider("Transaction cost (bps round-trip)", 0, 30, 10, step=1)

use_oos = st.sidebar.checkbox("✅ Use true Out-of-Sample split (highly recommended)", value=True)
oos_pct = st.sidebar.slider("Out-of-Sample %", 20, 40, 30, step=5) if use_oos else 0

window_min = st.sidebar.slider("Min causal window size", 64, 256, 128, step=32)
window_max = st.sidebar.slider("Max causal window size", 192, 512, 384, step=32)
window_step = st.sidebar.slider("Window step size", 16, 64, 32, step=16)
n_boot = st.sidebar.slider("Bootstrap replications", 500, 3000, 1000, step=500)

window_sizes = np.arange(window_min, window_max + 1, window_step)

if len(window_sizes) == 0 or window_sizes.max() >= nobs:
    st.error("Window range is invalid. Max window must be smaller than number of bars.")
    st.stop()

# ==============================================================================
# RUN BUTTON
# ==============================================================================
if st.button("🚀 Run Full Overfitting Test", type="primary"):
    seed = 12345
    rng = np.random.default_rng(seed)

    with st.spinner("Generating price series and running optimization..."):
        prices = simulate_gbm_prices(nobs, mu_annual, vol_annual, seed=seed)

        # Split into train / test if OOS is enabled
        if use_oos:
            split_idx = int(len(prices) * (1 - oos_pct / 100))
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
        else:
            train_prices = prices
            test_prices = None

        # Optimize only on training data
        result_train = test_wavelet_range(train_prices, window_sizes, tc_bps)

        # Evaluate best window on full series and on OOS
        full_sharpe = evaluate_single_window(prices, result_train["best_window"], tc_bps)
        oos_sharpe = evaluate_single_window(test_prices, result_train["best_window"], tc_bps) if use_oos else np.nan

    # ==============================================================================
    # BOOTSTRAP (only on training portion)
    # ==============================================================================
    progress_bar = st.progress(0, text="Running bootstrap on training data...")
    best_boot_sharpes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        boot_prices = prices_from_resampled_returns(train_prices, rng)
        result_boot = test_wavelet_range(boot_prices, window_sizes, tc_bps)
        best_boot_sharpes[i] = result_boot["best_sharpe"]

        if (i + 1) % max(1, n_boot // 20) == 0:
            progress_bar.progress((i + 1) / n_boot)

    progress_bar.empty()

    # ==============================================================================
    # RESULTS
    # ==============================================================================
    p_value = np.mean(best_boot_sharpes >= result_train["best_sharpe"])
    q50, q90, q95, q99 = np.quantile(best_boot_sharpes, [0.50, 0.90, 0.95, 0.99])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Training Results (Optimization)")
        st.metric("Best causal window", f"{result_train['best_window']} bars")
        st.metric("Best Sharpe (train)", f"{result_train['best_sharpe']:.4f}")

    with col2:
        st.subheader("📊 Bootstrap (Random Training Data)")
        st.metric("Median best Sharpe", f"{q50:.4f}")
        st.metric("90th percentile", f"{q90:.4f}")
        st.metric("95th percentile", f"{q95:.4f}")
        st.metric("99th percentile", f"{q99:.4f}")
        st.metric("p-value (random ≥ real)", f"{p_value:.1%}")

    st.markdown("---")

    # True OOS result
    if use_oos:
        st.subheader("🔥 True Out-of-Sample Performance")
        col_oos1, col_oos2 = st.columns(2)
        col_oos1.metric("OOS Sharpe (best window)", f"{oos_sharpe:.4f}")
        col_oos2.metric("Full-series Sharpe (for reference)", f"{full_sharpe:.4f}")

    # Interpretation
    if result_train["best_sharpe"] > q95:
        st.success("✅ Best window stands out strongly → low risk of overfitting.")
    else:
        st.warning("⚠️ The best Sharpe is **not unusual** compared to parameter search on pure noise. Strong evidence of overfitting.")

    # Top windows table
    topn = min(10, len(window_sizes))
    idx = np.argsort(result_train["sharpes"])[::-1][:topn]
    st.subheader(f"🏆 Top {topn} Causal Window Sizes (Training Data)")
    df_top = pd.DataFrame({
        "Causal Window": result_train["windows"][idx],
        "Sharpe Ratio": result_train["sharpes"][idx]
    })
    st.dataframe(df_top.style.format({"Sharpe Ratio": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.info("""
    **How to avoid overfitting (what this app now does):**
    • True OOS split (optimize only on first part)
    • Realistic transaction costs
    • Bootstrap only on training data
    • Much larger dataset (2000+ bars recommended)
    """)

    # Optional equity curve for best window on OOS
    if use_oos and st.checkbox("Show equity curve for best window on Out-of-Sample period"):
        best_window = result_train["best_window"]
        denoised = causal_wavelet_denoise(tuple(test_prices), best_window)
        w = rogers_satchell_volatility_approx(test_prices)
        idx = np.arange(len(test_prices))
        labels = auto_labeling(tuple(denoised), tuple(idx), w)
        strat_rets = strategy_returns_from_labels(test_prices, labels, tc_bps)
        equity = np.cumprod(1 + strat_rets)

        fig_df = pd.DataFrame({
            "Bar": np.arange(len(equity)),
            "Equity": equity
        })
        st.line_chart(fig_df.set_index("Bar"), use_container_width=True)

else:
    st.info("👈 Adjust settings in the sidebar and click **Run Full Overfitting Test**")

st.caption("Causal wavelet denoising + trend auto-labeling • 252 trading days/year • Strictly causal • Transaction costs applied")
