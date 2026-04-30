import time
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pywt
import streamlit as st


TIMEFRAME = "1h"
PRIMARY_SYMBOL = "BTC/USD"
SECONDARY_SYMBOLS = ["ETH/USD"]
PERIODS_PER_YEAR = 252 * 24
FETCH_EXCHANGES = [
    {"id": "kraken", "symbol": PRIMARY_SYMBOL, "limit": 720},
    {"id": "coinbase", "symbol": PRIMARY_SYMBOL, "limit": 300},
    {"id": "bitstamp", "symbol": PRIMARY_SYMBOL, "limit": 1000},
]


st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("Causal Wavelet + Honest BTC Validation")
st.markdown("Live BTC/USD 1h data via ccxt with baselines, block bootstrap, and cross-asset checks.")


def _safe_sharpe(rets: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if rets.size < 2:
        return np.nan
    sd = np.std(rets, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.sqrt(periods_per_year) * np.mean(rets) / sd)


def _volatility_band(prices: np.ndarray, multiplier: float = 1.8) -> float:
    if prices.size < 2:
        return 0.02
    log_rets = np.diff(np.log(prices))
    sigma = np.std(log_rets, ddof=1)
    if np.isnan(sigma) or sigma <= 0:
        return 0.02
    return float(max(sigma * multiplier, 1e-4))


def _fetch_from_exchange(
    exchange_id: str,
    symbol: str,
    target_bars: int,
    per_request_limit: int,
    warmup_bars: int = 256,
) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    meta: Dict[str, str] = {"exchange": exchange_id, "symbol": symbol}
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        exchange.load_markets()
        if not exchange.has.get("fetchOHLCV"):
            meta["error"] = "fetchOHLCV not supported"
            return None, meta

        timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
        total_needed = target_bars + warmup_bars
        request_limit = max(50, min(per_request_limit, total_needed))
        now_ms = exchange.milliseconds()
        since = now_ms - total_needed * timeframe_ms

        all_candles: List[List[float]] = []
        max_loops = max(3, int(np.ceil(total_needed / request_limit)) + 5)

        for _ in range(max_loops):
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=since,
                limit=request_limit,
            )
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            next_since = last_ts + timeframe_ms
            if next_since <= since:
                break
            since = next_since

            if len(candles) < request_limit or last_ts >= now_ms - timeframe_ms:
                break

            time.sleep(exchange.rateLimit / 1000.0)

        if not all_candles:
            meta["error"] = "no candles returned"
            return None, meta

        df = pd.DataFrame(
            all_candles,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.tail(total_needed).copy()

        meta["bars"] = str(len(df))
        meta["start"] = df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
        meta["end"] = df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        return df, meta
    except Exception as exc:
        meta["error"] = str(exc)
        return None, meta


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_data(
    target_bars: int = 4000, symbol: str = PRIMARY_SYMBOL
) -> Tuple[Optional[pd.DataFrame], Dict[str, str], List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    best_df: Optional[pd.DataFrame] = None
    best_meta: Dict[str, str] = {}

    for cfg in FETCH_EXCHANGES:
        df, meta = _fetch_from_exchange(
            exchange_id=cfg["id"],
            symbol=symbol,
            target_bars=target_bars,
            per_request_limit=cfg["limit"],
        )
        attempts.append(meta)

        if df is not None and (best_df is None or len(df) > len(best_df)):
            best_df = df
            best_meta = meta

        if df is not None and len(df) >= target_bars:
            return df, meta, attempts

    return best_df, best_meta, attempts


@st.cache_data(show_spinner=False)
def causal_wavelet_denoise(prices_tuple: Tuple[float, ...], window_size: int) -> np.ndarray:
    prices = np.asarray(prices_tuple, dtype=float)
    if prices.size == 0:
        return prices

    denoised = prices.copy()
    wavelet = pywt.Wavelet("db4")

    for i in range(window_size - 1, len(prices)):
        window_data = prices[i - window_size + 1 : i + 1]
        max_level = pywt.dwt_max_level(len(window_data), wavelet.dec_len)
        level = min(4, max_level)
        if level < 1:
            denoised[i] = window_data[-1]
            continue

        coeffs = pywt.wavedec(window_data, wavelet, level=level, mode="symmetric")
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
        uthresh = sigma * np.sqrt(2.0 * np.log(len(window_data))) if sigma > 0 else 0.0
        coeffs_thresh = [coeffs[0]] + [
            pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]
        ]
        rebuilt = pywt.waverec(coeffs_thresh, wavelet, mode="symmetric")
        denoised[i] = np.asarray(rebuilt).flatten()[: len(window_data)][-1]

    return denoised


def auto_labeling(data: np.ndarray, timestamps: np.ndarray, w: float) -> np.ndarray:
    prices = np.asarray(data, dtype=float).flatten()
    ts = np.asarray(timestamps)
    n = min(prices.size, ts.size)
    prices = prices[:n]
    ts = ts[:n]

    labels = np.zeros(n, dtype=float)
    fp = prices[0]
    x_h = fp
    x_l = fp
    ht = ts[0]
    lt = ts[0]
    cid = 0
    fp_n = 0

    for i in range(n):
        if prices[i] > fp * (1.0 + w):
            x_h = prices[i]
            ht = ts[i]
            fp_n = i
            cid = 1
            break
        if prices[i] < fp * (1.0 - w):
            x_l = prices[i]
            lt = ts[i]
            fp_n = i
            cid = -1
            break

    for i in range(max(fp_n, 1), n):
        if cid > 0:
            if prices[i] > x_h:
                x_h = prices[i]
                ht = ts[i]
            if prices[i] < x_h * (1.0 - w) and lt < ht:
                labels[(ts > lt) & (ts <= ht)] = 1.0
                x_l = prices[i]
                lt = ts[i]
                cid = -1
        elif cid < 0:
            if prices[i] < x_l:
                x_l = prices[i]
                lt = ts[i]
            if prices[i] > x_l * (1.0 + w) and ht <= lt:
                labels[(ts > ht) & (ts <= lt)] = -1.0
                x_h = prices[i]
                ht = ts[i]
                cid = 1

    return np.where(labels == 0, cid, labels)


def strategy_returns_from_labels(prices: np.ndarray, labels: np.ndarray, tc_bps: int = 22) -> np.ndarray:
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]
    strat_rets = pos * asset_rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets[position_changes] -= tc_bps / 10000.0
    return strat_rets


def final_oos_returns(
    train_prices: np.ndarray, test_prices: np.ndarray, window_size: int, tc_bps: int
) -> np.ndarray:
    combined = np.concatenate([train_prices[-window_size:], test_prices])
    denoised = causal_wavelet_denoise(tuple(combined), window_size)
    denoised_test = denoised[window_size:]
    w = _volatility_band(train_prices)
    labels = auto_labeling(denoised_test, np.arange(denoised_test.size), w)
    return strategy_returns_from_labels(test_prices, labels, tc_bps)


def walk_forward_score(
    prices: np.ndarray,
    window_size: int,
    tc_bps: int,
    n_splits: int = 4,
    min_train_bars: int = 800,
    test_bars: int = 240,
) -> float:
    fold_scores: List[float] = []
    n = prices.size
    total_test_bars = n_splits * test_bars

    if n < min_train_bars + total_test_bars:
        return np.nan

    train_end = n - total_test_bars
    for _ in range(n_splits):
        test_end = train_end + test_bars
        train_slice = prices[:train_end]
        test_slice = prices[train_end:test_end]

        if test_slice.size < test_bars or train_slice.size < max(min_train_bars, window_size + 10):
            return np.nan

        rets = final_oos_returns(train_slice, test_slice, window_size, tc_bps)
        score = _safe_sharpe(rets)
        if np.isnan(score):
            return np.nan
        fold_scores.append(score)
        train_end = test_end

    return float(np.mean(fold_scores)) if fold_scores else np.nan


def prices_from_resampled_returns(base_prices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    boot_rets = rng.choice(orig_rets, size=orig_rets.size, replace=True)
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


def prices_from_block_bootstrap(
    base_prices: np.ndarray, rng: np.random.Generator, block_size: int
) -> np.ndarray:
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    if orig_rets.size == 0:
        return base_prices.copy()

    block_size = max(2, min(block_size, orig_rets.size))
    blocks: List[np.ndarray] = []
    needed = orig_rets.size

    while sum(block.size for block in blocks) < needed:
        start = int(rng.integers(0, max(1, orig_rets.size - block_size + 1)))
        block = orig_rets[start : start + block_size]
        if block.size == 0:
            continue
        blocks.append(block)

    boot_rets = np.concatenate(blocks)[:needed]
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


def buy_and_hold_returns(prices: np.ndarray) -> np.ndarray:
    return prices[1:] / prices[:-1] - 1.0


def moving_average_crossover_returns(
    prices: np.ndarray, tc_bps: int, fast_window: int = 24, slow_window: int = 72
) -> np.ndarray:
    if prices.size <= slow_window + 1:
        return np.array([])

    series = pd.Series(prices)
    fast_ma = series.rolling(fast_window).mean()
    slow_ma = series.rolling(slow_window).mean()
    pos = np.where(fast_ma > slow_ma, 1.0, -1.0)
    pos = pd.Series(pos).fillna(0.0).to_numpy()
    rets = buy_and_hold_returns(prices)
    strat = pos[:-1] * rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos[:-1])))) > 0
    strat[position_changes] -= tc_bps / 10000.0
    return strat


def choose_best_window(
    prices: np.ndarray, candidate_windows: List[int], tc_bps: int
) -> Tuple[Optional[int], pd.DataFrame]:
    rows = []
    for window in candidate_windows:
        score = walk_forward_score(prices, window, tc_bps)
        rows.append({"window": window, "walk_forward_sharpe": score})

    score_df = pd.DataFrame(rows).sort_values(
        by=["walk_forward_sharpe", "window"],
        ascending=[False, True],
        na_position="last",
    )
    if score_df.empty or score_df["walk_forward_sharpe"].isna().all():
        return None, score_df
    return int(score_df.iloc[0]["window"]), score_df


def evaluate_cross_asset(
    target_bars: int,
    symbol: str,
    best_win: int,
    tc_bps: int,
    oos_pct: int,
) -> Dict[str, object]:
    df, meta, attempts = fetch_market_data(target_bars, symbol)
    min_needed = max(1200, best_win + 300)
    result: Dict[str, object] = {
        "symbol": symbol,
        "exchange": meta.get("exchange", ""),
        "status": "ok",
        "attempts": attempts,
    }

    if df is None or len(df) < min_needed:
        result["status"] = "insufficient_data"
        return result

    prices = df["close"].to_numpy(dtype=float)
    split_idx = int(len(prices) * (1 - oos_pct / 100))
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    rets = final_oos_returns(train_prices, test_prices, best_win, tc_bps)
    result["holdout_sharpe"] = _safe_sharpe(rets)
    result["bars"] = len(df)
    return result


st.sidebar.header("Settings")
target_bars = st.sidebar.slider("Target bars", 1500, 5000, 3500, step=100)
tc_bps = st.sidebar.slider("Transaction cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Center window size", value=420, min_value=200, max_value=900, step=20)
window_span = st.sidebar.slider("Window search span", 40, 240, 120, step=20)
window_step = st.sidebar.select_slider("Window search step", options=[10, 20, 30, 40, 60], value=20)
use_oos = st.sidebar.checkbox("Use final holdout", value=True)
oos_pct = st.sidebar.slider("Holdout %", 20, 45, 35, step=5)
n_boot = st.sidebar.slider("Bootstrap runs", 100, 800, 300, step=50)
bootstrap_mode = st.sidebar.selectbox("Bootstrap mode", ["Block bootstrap", "IID bootstrap"], index=0)
block_size = st.sidebar.slider("Bootstrap block size", 12, 168, 48, step=12)
run_cross_asset = st.sidebar.checkbox("Run cross-asset check (ETH/USD)", value=True)


if st.button("Fetch Real BTC Data & Run Test", type="primary"):
    with st.spinner("Fetching BTC/USD candles from available exchanges..."):
        df, fetch_meta, fetch_attempts = fetch_market_data(target_bars, PRIMARY_SYMBOL)

    min_needed = max(1200, fixed_window + 300)
    if df is None or len(df) < min_needed:
        st.error("Fetch failed or returned too little data for an honest test.")
        if fetch_attempts:
            st.write(pd.DataFrame(fetch_attempts))
        st.stop()

    prices = df["close"].to_numpy(dtype=float)
    shown_df = df.tail(target_bars).copy()
    start_dt = shown_df["datetime"].iloc[0]
    end_dt = shown_df["datetime"].iloc[-1]
    st.success(
        f"Fetched {len(shown_df)} BTC/USD 1h bars from {start_dt:%Y-%m-%d %H:%M UTC} to {end_dt:%Y-%m-%d %H:%M UTC} using {fetch_meta.get('exchange', 'unknown')}."
    )
    if fetch_attempts:
        with st.expander("Fetch diagnostics"):
            st.dataframe(pd.DataFrame(fetch_attempts), use_container_width=True, hide_index=True)

    candidate_windows = sorted(
        {
            w
            for w in range(
                max(160, fixed_window - window_span),
                min(960, fixed_window + window_span) + 1,
                window_step,
            )
        }
    )

    if use_oos:
        split_idx = int(len(prices) * (1 - oos_pct / 100))
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
    else:
        train_prices = prices
        test_prices = prices
        st.warning("Holdout is disabled. Any score below is in-sample and should not be trusted as evidence.")

    with st.spinner("Selecting the window using walk-forward validation on the training set..."):
        best_win, score_df = choose_best_window(train_prices, candidate_windows, tc_bps)

    if best_win is None:
        st.error("Could not score the candidate windows. Increase target bars or reduce the search range.")
        st.stop()

    train_cv_sharpe = float(score_df.iloc[0]["walk_forward_sharpe"])

    if use_oos:
        wavelet_rets = final_oos_returns(train_prices, test_prices, best_win, tc_bps)
        wavelet_sharpe = _safe_sharpe(wavelet_rets)
        equity = np.cumprod(1.0 + np.concatenate(([0.0], wavelet_rets)))
    else:
        denoised = causal_wavelet_denoise(tuple(train_prices), best_win)
        labels = auto_labeling(denoised, np.arange(train_prices.size), _volatility_band(train_prices))
        wavelet_rets = strategy_returns_from_labels(train_prices, labels, tc_bps)
        wavelet_sharpe = _safe_sharpe(wavelet_rets)
        equity = np.cumprod(1.0 + np.concatenate(([0.0], wavelet_rets)))

    benchmark_prices = test_prices if use_oos else train_prices
    buy_hold_rets = buy_and_hold_returns(benchmark_prices)
    ma_rets = moving_average_crossover_returns(benchmark_prices, tc_bps)
    buy_hold_sharpe = _safe_sharpe(buy_hold_rets)
    ma_sharpe = _safe_sharpe(ma_rets)

    rng = np.random.default_rng(12345)
    progress_bar = st.progress(0, text="Running bootstrap under the null...")
    best_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        if bootstrap_mode == "Block bootstrap":
            boot_prices = prices_from_block_bootstrap(train_prices, rng, block_size)
        else:
            boot_prices = prices_from_resampled_returns(train_prices, rng)
        boot_best_win, boot_scores = choose_best_window(boot_prices, candidate_windows, tc_bps)
        best_boot[i] = float(boot_scores.iloc[0]["walk_forward_sharpe"]) if boot_best_win is not None else np.nan
        progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    valid_boot = best_boot[~np.isnan(best_boot)]
    p_value = float(np.mean(valid_boot >= train_cv_sharpe)) if valid_boot.size else np.nan
    q95 = float(np.quantile(valid_boot, 0.95)) if valid_boot.size else np.nan

    cross_asset_rows: List[Dict[str, object]] = []
    if run_cross_asset and use_oos:
        with st.spinner("Running cross-asset validation..."):
            for symbol in SECONDARY_SYMBOLS:
                cross_asset_rows.append(
                    evaluate_cross_asset(target_bars, symbol, best_win, tc_bps, oos_pct)
                )

    report_checks = [
        ("Train CV > bootstrap 95%", bool(not np.isnan(q95) and train_cv_sharpe > q95)),
        ("Holdout Sharpe > 0", bool(not np.isnan(wavelet_sharpe) and wavelet_sharpe > 0)),
        ("Beat buy-and-hold", bool(not np.isnan(wavelet_sharpe) and not np.isnan(buy_hold_sharpe) and wavelet_sharpe > buy_hold_sharpe)),
        ("Beat MA baseline", bool(not np.isnan(wavelet_sharpe) and not np.isnan(ma_sharpe) and wavelet_sharpe > ma_sharpe)),
    ]
    if cross_asset_rows:
        cross_asset_ok = any(
            row.get("status") == "ok" and not np.isnan(row.get("holdout_sharpe", np.nan)) and row.get("holdout_sharpe", np.nan) > 0
            for row in cross_asset_rows
        )
        report_checks.append(("Positive on secondary asset", cross_asset_ok))

    passed_checks = sum(1 for _, ok in report_checks if ok)
    total_checks = len(report_checks)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected window", f"{best_win} bars")
        st.metric("Train CV Sharpe", f"{train_cv_sharpe:.4f}")
    with col2:
        st.metric("Wavelet Sharpe", f"{wavelet_sharpe:.4f}")
        st.metric("Bootstrap p-value", "n/a" if np.isnan(p_value) else f"{p_value:.1%}")
    with col3:
        st.metric("Buy/Hold Sharpe", f"{buy_hold_sharpe:.4f}")
        st.metric("MA Sharpe", f"{ma_sharpe:.4f}")
    with col4:
        st.metric("Checks passed", f"{passed_checks}/{total_checks}")
        st.metric("Bootstrap mode", "block" if bootstrap_mode == "Block bootstrap" else "iid")

    st.subheader("Pass / Fail Report")
    report_df = pd.DataFrame(
        [{"check": label, "result": "PASS" if ok else "FAIL"} for label, ok in report_checks]
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    if passed_checks == total_checks:
        st.success("This run passed every configured robustness check.")
    elif passed_checks >= max(2, total_checks - 1):
        st.warning("This run is somewhat promising, but at least one robustness check still failed.")
    else:
        st.error("This run looks fragile. The strategy is not surviving enough robustness checks yet.")

    st.subheader("Window Search Results")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("Baseline Comparison")
    baseline_df = pd.DataFrame(
        [
            {"strategy": "Wavelet labels", "sharpe": wavelet_sharpe},
            {"strategy": "Buy and hold", "sharpe": buy_hold_sharpe},
            {"strategy": "24/72 MA crossover", "sharpe": ma_sharpe},
        ]
    )
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    if cross_asset_rows:
        st.subheader("Cross-Asset Check")
        st.dataframe(pd.DataFrame(cross_asset_rows), use_container_width=True, hide_index=True)

    st.subheader("Equity Curve")
    chart_label = "Holdout Equity" if use_oos else "In-sample Equity"
    st.line_chart(pd.DataFrame({chart_label: equity}), use_container_width=True)

    st.caption(
        "Window selection uses only the training set. Final holdout, baselines, bootstrap, and cross-asset checks are reported separately."
    )
else:
    st.info("Click the button to fetch real BTC data and run a stricter validation pass.")
