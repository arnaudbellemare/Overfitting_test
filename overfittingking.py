import time
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pywt
import streamlit as st


TIMEFRAME = "1h"
SYMBOL = "BTC/USD"
MS_PER_HOUR = 60 * 60 * 1000
PERIODS_PER_YEAR = 252 * 24
FETCH_EXCHANGES = [
    {"id": "kraken", "symbol": "BTC/USD", "limit": 720},
    {"id": "coinbase", "symbol": "BTC/USD", "limit": 300},
    {"id": "bitstamp", "symbol": "BTC/USD", "limit": 1000},
]


st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("Causal Wavelet + Honest BTC Validation")
st.markdown("Live BTC/USD 1h data via ccxt with walk-forward training and true holdout testing.")


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
def fetch_real_btc_data(target_bars: int = 4000) -> Tuple[Optional[pd.DataFrame], Dict[str, str], List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    best_df: Optional[pd.DataFrame] = None
    best_meta: Dict[str, str] = {}

    for cfg in FETCH_EXCHANGES:
        df, meta = _fetch_from_exchange(
            exchange_id=cfg["id"],
            symbol=cfg["symbol"],
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
def causal_wavelet_denoise(prices_tuple: tuple[float, ...], window_size: int) -> np.ndarray:
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


def evaluate_single_window(prices: np.ndarray, window_size: int, tc_bps: int) -> float:
    min_required = max(window_size + 10, 300)
    if prices.size < min_required:
        return np.nan

    denoised = causal_wavelet_denoise(tuple(prices), window_size)
    w = _volatility_band(prices)
    labels = auto_labeling(denoised, np.arange(prices.size), w)
    strat_rets = strategy_returns_from_labels(prices, labels, tc_bps)
    return _safe_sharpe(strat_rets)


def walk_forward_score(
    prices: np.ndarray,
    window_size: int,
    tc_bps: int,
    n_splits: int = 4,
    min_train_bars: int = 800,
    test_bars: int = 240,
) -> float:
    fold_scores: list[float] = []
    n = prices.size
    first_test_end = n_splits * test_bars

    if n < min_train_bars + first_test_end:
        return np.nan

    train_end = n - first_test_end
    for _ in range(n_splits):
        test_end = train_end + test_bars
        train_slice = prices[:train_end]
        test_slice = prices[train_end:test_end]

        if test_slice.size < test_bars or train_slice.size < max(min_train_bars, window_size + 10):
            return np.nan

        combined = np.concatenate([train_slice[-window_size:], test_slice])
        denoised = causal_wavelet_denoise(tuple(combined), window_size)
        denoised_test = denoised[window_size:]
        w = _volatility_band(train_slice)
        labels = auto_labeling(denoised_test, np.arange(denoised_test.size), w)
        rets = strategy_returns_from_labels(test_slice, labels, tc_bps)
        score = _safe_sharpe(rets)
        if np.isnan(score):
            return np.nan
        fold_scores.append(score)
        train_end = test_end

    if not fold_scores:
        return np.nan
    return float(np.mean(fold_scores))


def prices_from_resampled_returns(base_prices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    orig_rets = base_prices[1:] / base_prices[:-1] - 1.0
    boot_rets = rng.choice(orig_rets, size=orig_rets.size, replace=True)
    prices = np.empty(base_prices.size, dtype=float)
    prices[0] = base_prices[0]
    prices[1:] = prices[0] * np.cumprod(1.0 + boot_rets)
    return prices


def choose_best_window(
    prices: np.ndarray, candidate_windows: list[int], tc_bps: int
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


def final_oos_returns(train_prices: np.ndarray, test_prices: np.ndarray, window_size: int, tc_bps: int) -> np.ndarray:
    combined = np.concatenate([train_prices[-window_size:], test_prices])
    denoised = causal_wavelet_denoise(tuple(combined), window_size)
    denoised_test = denoised[window_size:]
    w = _volatility_band(train_prices)
    labels = auto_labeling(denoised_test, np.arange(denoised_test.size), w)
    return strategy_returns_from_labels(test_prices, labels, tc_bps)


st.sidebar.header("Settings")
target_bars = st.sidebar.slider("Target bars", 1500, 5000, 3500, step=100)
tc_bps = st.sidebar.slider("Transaction cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Center window size", value=420, min_value=200, max_value=900, step=20)
window_span = st.sidebar.slider("Window search span", 40, 240, 120, step=20)
window_step = st.sidebar.select_slider("Window search step", options=[10, 20, 30, 40, 60], value=20)
use_oos = st.sidebar.checkbox("Use final holdout", value=True)
oos_pct = st.sidebar.slider("Holdout %", 20, 45, 35, step=5)
n_boot = st.sidebar.slider("Bootstrap runs", 100, 800, 300, step=50)


if st.button("Fetch Real BTC Data & Run Test", type="primary"):
    with st.spinner("Fetching BTC/USD candles from available exchanges..."):
        df, fetch_meta, fetch_attempts = fetch_real_btc_data(target_bars)

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
        oos_rets = final_oos_returns(train_prices, test_prices, best_win, tc_bps)
        oos_sharpe = _safe_sharpe(oos_rets)
        equity = np.cumprod(1.0 + np.concatenate(([0.0], oos_rets)))
    else:
        oos_rets = strategy_returns_from_labels(
            train_prices,
            auto_labeling(
                causal_wavelet_denoise(tuple(train_prices), best_win),
                np.arange(train_prices.size),
                _volatility_band(train_prices),
            ),
            tc_bps,
        )
        oos_sharpe = _safe_sharpe(oos_rets)
        equity = np.cumprod(1.0 + np.concatenate(([0.0], oos_rets)))

    rng = np.random.default_rng(12345)
    progress_bar = st.progress(0, text="Running bootstrap under the null...")
    best_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_prices = prices_from_resampled_returns(train_prices, rng)
        boot_best_win, boot_scores = choose_best_window(boot_prices, candidate_windows, tc_bps)
        best_boot[i] = (
            float(boot_scores.iloc[0]["walk_forward_sharpe"])
            if boot_best_win is not None
            else np.nan
        )
        progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    valid_boot = best_boot[~np.isnan(best_boot)]
    p_value = float(np.mean(valid_boot >= train_cv_sharpe)) if valid_boot.size else np.nan
    q95 = float(np.quantile(valid_boot, 0.95)) if valid_boot.size else np.nan

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected window", f"{best_win} bars")
        st.metric("Train CV Sharpe", f"{train_cv_sharpe:.4f}")
    with col2:
        st.metric("Holdout Sharpe" if use_oos else "In-sample Sharpe", f"{oos_sharpe:.4f}")
        st.metric("Bootstrap p-value", "n/a" if np.isnan(p_value) else f"{p_value:.1%}")
    with col3:
        st.metric("Train bars", f"{len(train_prices)}")
        st.metric("Holdout bars" if use_oos else "Scored bars", f"{len(test_prices)}")

    st.subheader("Window Search Results")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    if not np.isnan(q95):
        if train_cv_sharpe > q95 and use_oos and oos_sharpe > 0:
            st.success("The model cleared the bootstrap threshold on training CV and stayed positive on the holdout.")
        elif use_oos and oos_sharpe <= 0:
            st.warning("The selected window did not hold up on the final holdout. Treat this as likely overfitting.")
        else:
            st.warning("The training CV edge is not clearly above the bootstrap null. This still looks fragile.")

    st.subheader("Equity Curve")
    chart_label = "Holdout Equity" if use_oos else "In-sample Equity"
    st.line_chart(pd.DataFrame({chart_label: equity}), use_container_width=True)

    st.caption(
        "Fetch source: exchange selected via ccxt. Window selection uses only the training set. Holdout evaluation is performed once after selection."
    )
else:
    st.info("Click the button to fetch real BTC data and run a stricter validation pass.")
