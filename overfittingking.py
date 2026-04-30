import sqlite3
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
DB_PATH = "wavelet_features.db"
FETCH_EXCHANGES = [
    {"id": "kraken", "symbol": PRIMARY_SYMBOL, "limit": 720},
    {"id": "coinbase", "symbol": PRIMARY_SYMBOL, "limit": 300},
    {"id": "bitstamp", "symbol": PRIMARY_SYMBOL, "limit": 1000},
]


st.set_page_config(page_title="Real BTC Wavelet Test", layout="wide")
st.title("Causal Wavelet + Feature-Filtered Validation")
st.markdown("Live BTC/USD 1h data via ccxt with baselines, block bootstrap, SQLite logging, and feature filters.")


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def execute_with_retry(fn, attempts: int = 5, delay: float = 0.25):
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def init_db() -> None:
    def _init() -> None:
        with open_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (exchange, symbol, timeframe, ts)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    orderbook_imbalance REAL,
                    vwap_gap REAL,
                    ema_slope REAL,
                    volume_z REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, ts)
                )
                """
            )

    execute_with_retry(_init)


def store_ohlcv(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> None:
    if df.empty:
        return
    rows = [
        (
            exchange,
            symbol,
            timeframe,
            int(row.ts),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for row in df.itertuples(index=False)
    ]

    def _store() -> None:
        with open_db() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv
                (exchange, symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    execute_with_retry(_store)


def store_feature_snapshot(
    exchange: str,
    symbol: str,
    timeframe: str,
    ts: int,
    orderbook_imbalance: float,
    vwap_gap: float,
    ema_slope: float,
    volume_z: float,
) -> None:
    def _store() -> None:
        with open_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_snapshots
                (exchange, symbol, timeframe, ts, orderbook_imbalance, vwap_gap, ema_slope, volume_z)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange,
                    symbol,
                    timeframe,
                    ts,
                    orderbook_imbalance,
                    vwap_gap,
                    ema_slope,
                    volume_z,
                ),
            )

    execute_with_retry(_store)


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


def fetch_orderbook_imbalance(exchange_id: str, symbol: str, depth: int = 10) -> float:
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        order_book = exchange.fetch_order_book(symbol, limit=depth)
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bid_vol = sum(float(bid[1]) for bid in bids[:depth]) if bids else 0.0
        ask_vol = sum(float(ask[1]) for ask in asks[:depth]) if asks else 0.0
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / denom)
    except Exception:
        return 0.0


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


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["cum_vol"] = feat["volume"].cumsum()
    feat["cum_pv"] = (feat["close"] * feat["volume"]).cumsum()
    feat["vwap"] = feat["cum_pv"] / feat["cum_vol"].replace(0, np.nan)
    feat["vwap_gap"] = (feat["close"] / feat["vwap"]) - 1.0
    feat["ema_fast"] = feat["close"].ewm(span=12, adjust=False).mean()
    feat["ema_slow"] = feat["close"].ewm(span=48, adjust=False).mean()
    feat["ema_slope"] = feat["ema_fast"].pct_change(3).fillna(0.0)
    vol_mean = feat["volume"].rolling(24, min_periods=12).mean()
    vol_std = feat["volume"].rolling(24, min_periods=12).std(ddof=1)
    feat["volume_z"] = ((feat["volume"] - vol_mean) / vol_std.replace(0, np.nan)).fillna(0.0)
    return feat


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


def amplitude_segment_labels(
    feature_df: pd.DataFrame, minamp: float, tinactive: int, scale: float = 1e4
) -> np.ndarray:
    prices = feature_df["close"].to_numpy(dtype=float)
    n = prices.size
    if n == 0:
        return np.array([], dtype=float)

    cumr = np.log(prices / prices[0]) * scale
    labels = np.zeros(n, dtype=float)
    epsilon = 1e-6

    istart = 0
    icursor = 0
    imin = 0
    imax = 0
    vmin = cumr[0]
    vmax = cumr[0]

    while icursor < n:
        v = cumr[icursor]
        if (vmax - vmin) + epsilon >= minamp and imin > imax and (v - vmin) + epsilon >= minamp:
            labels[istart:imax] = 0.0
            labels[imax : imin + 1] = -1.0
            istart = imin
            imax = icursor
            vmax = v
        elif (vmax - vmin) + epsilon >= minamp and imax > imin and (vmax - v) + epsilon >= minamp:
            labels[istart:imin] = 0.0
            labels[imin : imax + 1] = 1.0
            istart = imax
            imin = icursor
            vmin = v
        elif imax > imin and (icursor - imax) >= tinactive and v <= vmax:
            if (vmax - vmin) + epsilon >= minamp:
                labels[istart:imin] = 0.0
                labels[imin : imax + 1] = 1.0
                labels[imax + 1 : icursor + 1] = 0.0
            else:
                labels[istart : icursor + 1] = 0.0
            istart = icursor
            imax = icursor
            imin = icursor
            vmax = v
            vmin = v
        elif imin > imax and (icursor - imin) >= tinactive and v >= vmin:
            if (vmax - vmin) + epsilon >= minamp:
                labels[istart:imax] = 0.0
                labels[imax : imin + 1] = -1.0
                labels[imin + 1 : icursor + 1] = 0.0
            else:
                labels[istart : icursor + 1] = 0.0
            istart = icursor
            imax = icursor
            imin = icursor
            vmax = v
            vmin = v

        if v >= vmax:
            imax = icursor
            vmax = v
        if v <= vmin:
            imin = icursor
            vmin = v
        icursor += 1

    if (vmax - vmin) + epsilon >= minamp:
        if imin > imax:
            labels[istart:imax] = 0.0
            labels[imax : imin + 1] = -1.0
        elif imax > imin:
            labels[istart:imin] = 0.0
            labels[imin : imax + 1] = 1.0

    ipos = 0
    while ipos < n:
        direction = labels[ipos]
        if direction == 0.0:
            ipos += 1
            continue
        seg_start = ipos
        while ipos < n and labels[ipos] == direction:
            ipos += 1
        seg_end = ipos - 1
        x = np.arange(seg_start, seg_end + 1) - seg_start
        y = cumr[seg_start : seg_end + 1]
        if y.size < 2 or np.any(np.isnan(y)) or np.any(np.isinf(y)):
            labels[seg_start : seg_end + 1] = 0.0
            continue
        try:
            beta = np.polyfit(x, y, 1)[0]
        except Exception:
            labels[seg_start : seg_end + 1] = 0.0
            continue
        max_dist = direction * beta * x
        if np.max(max_dist) + epsilon < minamp:
            labels[seg_start : seg_end + 1] = 0.0

    return labels


def amplitude_ensemble_labels(
    feature_df: pd.DataFrame,
    base_thresholds: List[float],
    tinactive: int,
    use_dynamic_scaling: bool,
    vol_window: int = 24,
) -> np.ndarray:
    if not base_thresholds:
        return np.zeros(len(feature_df), dtype=float)

    if not use_dynamic_scaling:
        label_stack = [
            amplitude_segment_labels(feature_df, float(threshold), tinactive)
            for threshold in base_thresholds
        ]
    else:
        prices = feature_df["close"].to_numpy(dtype=float)
        log_rets = np.diff(np.log(prices), prepend=np.log(prices[0]))
        rolling_vol = pd.Series(log_rets).rolling(vol_window, min_periods=max(6, vol_window // 2)).std(ddof=1)
        vol_scale = (rolling_vol / rolling_vol.median()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_scale = vol_scale.clip(lower=0.5, upper=2.0).to_numpy(dtype=float)
        label_stack = []
        for threshold in base_thresholds:
            scaled_threshold = float(threshold)
            dynamic_df = feature_df.copy()
            dynamic_df["close"] = feature_df["close"].to_numpy(dtype=float)
            dynamic_labels = np.zeros(len(dynamic_df), dtype=float)
            for i in range(len(dynamic_df)):
                local_threshold = scaled_threshold * vol_scale[i]
                window_df = dynamic_df.iloc[: i + 1]
                dynamic_labels[i] = amplitude_segment_labels(window_df, local_threshold, tinactive)[-1]
            label_stack.append(dynamic_labels)

    stacked = np.vstack(label_stack)
    mean_signal = stacked.mean(axis=0)
    return np.where(mean_signal > 0.05, 1.0, np.where(mean_signal < -0.05, -1.0, 0.0))


def estimate_workload_units(
    target_bars: int,
    n_boot: int,
    run_cross_asset: bool,
    use_amplitude_ensemble: bool,
    use_amplitude_dynamic_scaling: bool,
) -> int:
    units = target_bars + (n_boot * 10)
    if run_cross_asset:
        units += target_bars
    if use_amplitude_ensemble:
        units += target_bars
    if use_amplitude_dynamic_scaling:
        units += target_bars * 8
        units += n_boot * 25
    return units


def apply_feature_filters(
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
) -> np.ndarray:
    filt = labels.astype(float).copy()
    if use_vwap_filter:
        long_ok = feature_df["vwap_gap"].to_numpy() > 0
        short_ok = feature_df["vwap_gap"].to_numpy() < 0
        filt = np.where((filt > 0) & long_ok, filt, np.where(filt > 0, 0.0, filt))
        filt = np.where((filt < 0) & short_ok, filt, np.where(filt < 0, 0.0, filt))
    if use_ema_filter:
        slope = feature_df["ema_slope"].to_numpy()
        filt = np.where((filt > 0) & (slope > 0), filt, np.where(filt > 0, 0.0, filt))
        filt = np.where((filt < 0) & (slope < 0), filt, np.where(filt < 0, 0.0, filt))
    if use_volume_filter:
        vol_ok = feature_df["volume_z"].to_numpy() >= min_volume_z
        filt = np.where(vol_ok, filt, 0.0)
    return filt


def strategy_returns_from_labels(prices: np.ndarray, labels: np.ndarray, tc_bps: int = 22) -> np.ndarray:
    asset_rets = prices[1:] / prices[:-1] - 1.0
    pos = labels[:-1]
    strat_rets = pos * asset_rets
    position_changes = np.abs(np.diff(np.concatenate(([0.0], pos)))) > 0
    strat_rets[position_changes] -= tc_bps / 10000.0
    return strat_rets


def build_labels_and_features(prices: np.ndarray, feature_df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    denoised = causal_wavelet_denoise(tuple(prices), window_size)
    w = _volatility_band(prices)
    labels = auto_labeling(denoised, np.arange(prices.size), w)
    return denoised, labels


def build_signal_labels(
    feature_df: pd.DataFrame,
    signal_kind: str,
    signal_param: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
) -> np.ndarray:
    prices = feature_df["close"].to_numpy(dtype=float)
    if signal_kind == "wavelet":
        _, labels = build_labels_and_features(prices, feature_df, int(signal_param))
        return labels
    if signal_kind == "amplitude":
        return amplitude_segment_labels(feature_df, float(signal_param), amplitude_tinactive)
    if signal_kind == "amplitude_ensemble":
        if amplitude_ensemble_thresholds:
            ranked = sorted(amplitude_ensemble_thresholds, key=lambda x: (abs(x - float(signal_param)), x))
            keep_n = max(1, 2 * amplitude_ensemble_width + 1)
            thresholds = sorted(ranked[:keep_n])
        else:
            thresholds = [float(signal_param)]
        return amplitude_ensemble_labels(
            feature_df,
            thresholds,
            amplitude_tinactive,
            use_dynamic_scaling=amplitude_dynamic_scaling,
        )
    raise ValueError(f"Unsupported signal kind: {signal_kind}")


def compute_strategy_returns(
    feature_df: pd.DataFrame,
    signal_kind: str,
    signal_param: float,
    tc_bps: int,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
) -> Dict[str, np.ndarray]:
    prices = feature_df["close"].to_numpy(dtype=float)
    raw_labels = build_signal_labels(
        feature_df,
        signal_kind,
        signal_param,
        amplitude_tinactive,
        amplitude_ensemble_thresholds,
        amplitude_dynamic_scaling,
        amplitude_ensemble_width,
    )
    filtered_labels = apply_feature_filters(
        raw_labels,
        feature_df,
        use_vwap_filter=use_vwap_filter,
        use_ema_filter=use_ema_filter,
        use_volume_filter=use_volume_filter,
        min_volume_z=min_volume_z,
    )
    return {
        "raw": strategy_returns_from_labels(prices, raw_labels, tc_bps),
        "filtered": strategy_returns_from_labels(prices, filtered_labels, tc_bps),
        "raw_labels": raw_labels,
        "filtered_labels": filtered_labels,
    }


def final_oos_strategy_returns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    signal_kind: str,
    signal_param: float,
    tc_bps: int,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
) -> Dict[str, np.ndarray]:
    overlap = int(signal_param) if signal_kind == "wavelet" else 0
    combined = pd.concat([train_df.tail(overlap), test_df], ignore_index=True)
    all_labels = build_signal_labels(
        combined,
        signal_kind,
        signal_param,
        amplitude_tinactive,
        amplitude_ensemble_thresholds,
        amplitude_dynamic_scaling,
        amplitude_ensemble_width,
    )
    test_labels = all_labels[overlap:]
    test_features = combined.iloc[overlap:].reset_index(drop=True)
    filtered_labels = apply_feature_filters(
        test_labels,
        test_features,
        use_vwap_filter=use_vwap_filter,
        use_ema_filter=use_ema_filter,
        use_volume_filter=use_volume_filter,
        min_volume_z=min_volume_z,
    )
    test_prices = test_df["close"].to_numpy(dtype=float)
    return {
        "raw": strategy_returns_from_labels(test_prices, test_labels, tc_bps),
        "filtered": strategy_returns_from_labels(test_prices, filtered_labels, tc_bps),
        "raw_labels": test_labels,
        "filtered_labels": filtered_labels,
    }


def walk_forward_score(
    feature_df: pd.DataFrame,
    signal_kind: str,
    signal_param: float,
    tc_bps: int,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
    n_splits: int = 4,
    min_train_bars: int = 800,
    test_bars: int = 240,
) -> float:
    fold_scores: List[float] = []
    n = len(feature_df)
    total_test_bars = n_splits * test_bars
    if n < min_train_bars + total_test_bars:
        return np.nan

    train_end = n - total_test_bars
    for _ in range(n_splits):
        test_end = train_end + test_bars
        train_slice = feature_df.iloc[:train_end].reset_index(drop=True)
        test_slice = feature_df.iloc[train_end:test_end].reset_index(drop=True)
        overlap = int(signal_param) if signal_kind == "wavelet" else 0
        if len(test_slice) < test_bars or len(train_slice) < max(min_train_bars, overlap + 10):
            return np.nan

        rets = final_oos_strategy_returns(
            train_slice,
            test_slice,
            signal_kind,
            signal_param,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            amplitude_ensemble_thresholds,
            amplitude_dynamic_scaling,
            amplitude_ensemble_width,
        )["filtered"]
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


def feature_df_from_prices(prices: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "close": prices,
            "volume": np.ones_like(prices, dtype=float),
        }
    )
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["ts"] = np.arange(len(df))
    df["datetime"] = pd.to_datetime(df["ts"], unit="h", utc=True)
    return add_causal_features(df)


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


def choose_best_param(
    feature_df: pd.DataFrame,
    candidate_params: List[float],
    signal_kind: str,
    tc_bps: int,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
) -> Tuple[Optional[int], pd.DataFrame]:
    rows = []
    for param in candidate_params:
        score = walk_forward_score(
            feature_df,
            signal_kind,
            param,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            amplitude_ensemble_thresholds,
            amplitude_dynamic_scaling,
            amplitude_ensemble_width,
        )
        rows.append({"param": param, "walk_forward_sharpe": score})

    score_df = pd.DataFrame(rows).sort_values(
        by=["walk_forward_sharpe", "param"],
        ascending=[False, True],
        na_position="last",
    )
    if score_df.empty or score_df["walk_forward_sharpe"].isna().all():
        return None, score_df
    best_param = score_df.iloc[0]["param"]
    if signal_kind == "wavelet":
        return int(best_param), score_df
    return float(best_param), score_df


def evaluate_cross_asset(
    target_bars: int,
    symbol: str,
    signal_kind: str,
    best_param: float,
    tc_bps: int,
    oos_pct: int,
    use_vwap_filter: bool,
    use_ema_filter: bool,
    use_volume_filter: bool,
    min_volume_z: float,
    amplitude_tinactive: int,
    amplitude_ensemble_thresholds: Optional[List[float]] = None,
    amplitude_dynamic_scaling: bool = False,
    amplitude_ensemble_width: int = 2,
) -> Dict[str, object]:
    df, meta, _ = fetch_market_data(target_bars, symbol)
    overlap = int(best_param) if signal_kind == "wavelet" else 0
    min_needed = max(1200, overlap + 300)
    result: Dict[str, object] = {
        "symbol": symbol,
        "exchange": meta.get("exchange", ""),
        "status": "ok",
    }
    if df is None or len(df) < min_needed:
        result["status"] = "insufficient_data"
        return result

    feat = add_causal_features(df)
    split_idx = int(len(feat) * (1 - oos_pct / 100))
    train_df = feat.iloc[:split_idx].reset_index(drop=True)
    test_df = feat.iloc[split_idx:].reset_index(drop=True)
    rets = final_oos_strategy_returns(
        train_df,
        test_df,
        signal_kind,
        best_param,
        tc_bps,
        use_vwap_filter,
        use_ema_filter,
        use_volume_filter,
        min_volume_z,
        amplitude_tinactive,
        amplitude_ensemble_thresholds,
        amplitude_dynamic_scaling,
        amplitude_ensemble_width,
    )["filtered"]
    result["holdout_sharpe"] = _safe_sharpe(rets)
    result["bars"] = len(feat)
    return result


init_db()

st.sidebar.header("Settings")
target_bars = st.sidebar.slider("Target bars", 1500, 5000, 3500, step=100)
tc_bps = st.sidebar.slider("Transaction cost (bps)", 15, 40, 22, step=1)
fixed_window = st.sidebar.number_input("Center window size", value=420, min_value=200, max_value=900, step=20)
window_span = st.sidebar.slider("Window search span", 40, 240, 120, step=20)
window_step = st.sidebar.select_slider("Window search step", options=[10, 20, 30, 40, 60], value=20)
amplitude_center = st.sidebar.number_input("Center amplitude threshold", value=10.0, min_value=2.0, max_value=40.0, step=1.0)
amplitude_span = st.sidebar.slider("Amplitude search span", 2, 20, 6, step=2)
amplitude_step = st.sidebar.select_slider("Amplitude search step", options=[1.0, 2.0, 3.0, 5.0], value=1.0)
amplitude_tinactive = st.sidebar.slider("Amplitude inactivity bars", 3, 48, 10, step=1)
use_amplitude_ensemble = st.sidebar.checkbox("Use multi-threshold amplitude ensemble", value=True)
use_amplitude_dynamic_scaling = st.sidebar.checkbox("Use volatility-aware amplitude scaling", value=False)
ensemble_width = st.sidebar.slider("Amplitude ensemble width", 1, 4, 2, step=1)
use_oos = st.sidebar.checkbox("Use final holdout", value=True)
oos_pct = st.sidebar.slider("Holdout %", 20, 45, 35, step=5)
n_boot = st.sidebar.slider("Bootstrap runs", 50, 800, 100, step=50)
bootstrap_mode = st.sidebar.selectbox("Bootstrap mode", ["Block bootstrap", "IID bootstrap"], index=0)
block_size = st.sidebar.slider("Bootstrap block size", 12, 168, 48, step=12)
run_cross_asset = st.sidebar.checkbox("Run cross-asset check (ETH/USD)", value=False)
use_vwap_filter = st.sidebar.checkbox("Use VWAP confirmation", value=True)
use_ema_filter = st.sidebar.checkbox("Use EMA slope confirmation", value=True)
use_volume_filter = st.sidebar.checkbox("Use volume z-score filter", value=False)
min_volume_z = st.sidebar.slider("Min volume z-score", -1.0, 2.0, 0.0, step=0.1)


if st.button("Fetch Real BTC Data & Run Test", type="primary"):
    with st.spinner("Fetching BTC/USD candles from available exchanges..."):
        df, fetch_meta, fetch_attempts = fetch_market_data(target_bars, PRIMARY_SYMBOL)

    min_needed = max(1200, fixed_window + 300)
    if df is None or len(df) < min_needed:
        st.error("Fetch failed or returned too little data for an honest test.")
        if fetch_attempts:
            st.write(pd.DataFrame(fetch_attempts))
        st.stop()

    feat_df = add_causal_features(df)
    store_ohlcv(df[["ts", "open", "high", "low", "close", "volume"]], fetch_meta.get("exchange", "unknown"), PRIMARY_SYMBOL, TIMEFRAME)

    live_orderbook_imbalance = fetch_orderbook_imbalance(fetch_meta.get("exchange", "kraken"), PRIMARY_SYMBOL)
    latest_row = feat_df.iloc[-1]
    store_feature_snapshot(
        exchange=fetch_meta.get("exchange", "unknown"),
        symbol=PRIMARY_SYMBOL,
        timeframe=TIMEFRAME,
        ts=int(latest_row["ts"]),
        orderbook_imbalance=live_orderbook_imbalance,
        vwap_gap=float(latest_row["vwap_gap"]),
        ema_slope=float(latest_row["ema_slope"]),
        volume_z=float(latest_row["volume_z"]),
    )

    shown_df = feat_df.tail(target_bars).copy()
    start_dt = shown_df["datetime"].iloc[0]
    end_dt = shown_df["datetime"].iloc[-1]
    st.success(
        f"Fetched {len(shown_df)} BTC/USD 1h bars from {start_dt:%Y-%m-%d %H:%M UTC} to {end_dt:%Y-%m-%d %H:%M UTC} using {fetch_meta.get('exchange', 'unknown')}."
    )
    if fetch_attempts:
        with st.expander("Fetch diagnostics"):
            st.dataframe(pd.DataFrame(fetch_attempts), width="stretch", hide_index=True)

    st.subheader("Latest Live Feature Snapshot")
    live_cols = st.columns(4)
    with live_cols[0]:
        st.metric("Orderbook imbalance", f"{live_orderbook_imbalance:.3f}")
    with live_cols[1]:
        st.metric("VWAP gap", f"{latest_row['vwap_gap']:.3%}")
    with live_cols[2]:
        st.metric("EMA slope", f"{latest_row['ema_slope']:.3%}")
    with live_cols[3]:
        st.metric("Volume z-score", f"{latest_row['volume_z']:.2f}")
    st.caption("The live orderbook snapshot is logged to SQLite for future study. It is not backfilled historically, so it is not used in the historical holdout test yet.")

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
    candidate_amplitudes = sorted(
        {
            round(a, 4)
            for a in np.arange(
                max(2.0, amplitude_center - amplitude_span),
                amplitude_center + amplitude_span + amplitude_step * 0.5,
                amplitude_step,
            )
        }
    )

    if use_oos:
        split_idx = int(len(feat_df) * (1 - oos_pct / 100))
        train_df = feat_df.iloc[:split_idx].reset_index(drop=True)
        test_df = feat_df.iloc[split_idx:].reset_index(drop=True)
    else:
        train_df = feat_df.reset_index(drop=True)
        test_df = feat_df.reset_index(drop=True)
        st.warning("Holdout is disabled. Any score below is in-sample and should not be trusted as evidence.")

    workload_units = estimate_workload_units(
        target_bars,
        n_boot,
        run_cross_asset,
        use_amplitude_ensemble,
        use_amplitude_dynamic_scaling,
    )
    if workload_units > 12000:
        st.warning(
            "This configuration is heavy and may feel slow on Streamlit Cloud. "
            "Try lower bootstrap runs, disable cross-asset, or turn off dynamic amplitude scaling."
        )

    amplitude_signal_kind = "amplitude_ensemble" if use_amplitude_ensemble else "amplitude"

    with st.spinner("Selecting best wavelet and amplitude parameters on the training set..."):
        best_win, wavelet_score_df = choose_best_param(
            train_df,
            candidate_windows,
            "wavelet",
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
        )
        best_amp, amplitude_score_df = choose_best_param(
            train_df,
            candidate_amplitudes,
            amplitude_signal_kind,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            candidate_amplitudes if use_amplitude_ensemble else None,
            use_amplitude_dynamic_scaling,
            ensemble_width,
        )

    if best_win is None or best_amp is None:
        st.error("Could not score one or more candidate parameter grids. Increase target bars or reduce the search range.")
        st.stop()

    wavelet_train_cv_sharpe = float(wavelet_score_df.iloc[0]["walk_forward_sharpe"])
    amplitude_train_cv_sharpe = float(amplitude_score_df.iloc[0]["walk_forward_sharpe"])

    if use_oos:
        wavelet_out = final_oos_strategy_returns(
            train_df,
            test_df,
            "wavelet",
            best_win,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
        )
        amplitude_out = final_oos_strategy_returns(
            train_df,
            test_df,
            amplitude_signal_kind,
            best_amp,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            candidate_amplitudes if use_amplitude_ensemble else None,
            use_amplitude_dynamic_scaling,
            ensemble_width,
        )
    else:
        wavelet_out = compute_strategy_returns(
            train_df,
            "wavelet",
            best_win,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
        )
        amplitude_out = compute_strategy_returns(
            train_df,
            amplitude_signal_kind,
            best_amp,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            candidate_amplitudes if use_amplitude_ensemble else None,
            use_amplitude_dynamic_scaling,
            ensemble_width,
        )

    wavelet_raw_sharpe = _safe_sharpe(wavelet_out["raw"])
    wavelet_filtered_sharpe = _safe_sharpe(wavelet_out["filtered"])
    amplitude_raw_sharpe = _safe_sharpe(amplitude_out["raw"])
    amplitude_filtered_sharpe = _safe_sharpe(amplitude_out["filtered"])
    wavelet_filtered_equity = np.cumprod(1.0 + np.concatenate(([0.0], wavelet_out["filtered"])))
    amplitude_filtered_equity = np.cumprod(1.0 + np.concatenate(([0.0], amplitude_out["filtered"])))

    benchmark_prices = test_df["close"].to_numpy(dtype=float) if use_oos else train_df["close"].to_numpy(dtype=float)
    buy_hold_sharpe = _safe_sharpe(buy_and_hold_returns(benchmark_prices))
    ma_sharpe = _safe_sharpe(moving_average_crossover_returns(benchmark_prices, tc_bps))

    rng = np.random.default_rng(12345)
    progress_bar = st.progress(0, text="Running bootstrap under the null...")
    wavelet_best_boot = np.empty(n_boot, dtype=float)
    amplitude_best_boot = np.empty(n_boot, dtype=float)
    train_prices = train_df["close"].to_numpy(dtype=float)
    for i in range(n_boot):
        if bootstrap_mode == "Block bootstrap":
            boot_prices = prices_from_block_bootstrap(train_prices, rng, block_size)
        else:
            boot_prices = prices_from_resampled_returns(train_prices, rng)
        boot_feat = feature_df_from_prices(boot_prices)
        boot_best_win, boot_wavelet_scores = choose_best_param(
            boot_feat,
            candidate_windows,
            "wavelet",
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
        )
        boot_best_amp, boot_amplitude_scores = choose_best_param(
            boot_feat,
            candidate_amplitudes,
            amplitude_signal_kind,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
            amplitude_tinactive,
            candidate_amplitudes if use_amplitude_ensemble else None,
            use_amplitude_dynamic_scaling,
            ensemble_width,
        )
        wavelet_best_boot[i] = float(boot_wavelet_scores.iloc[0]["walk_forward_sharpe"]) if boot_best_win is not None else np.nan
        amplitude_best_boot[i] = float(boot_amplitude_scores.iloc[0]["walk_forward_sharpe"]) if boot_best_amp is not None else np.nan
        progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    valid_wavelet_boot = wavelet_best_boot[~np.isnan(wavelet_best_boot)]
    valid_amplitude_boot = amplitude_best_boot[~np.isnan(amplitude_best_boot)]
    wavelet_p_value = float(np.mean(valid_wavelet_boot >= wavelet_train_cv_sharpe)) if valid_wavelet_boot.size else np.nan
    wavelet_q95 = float(np.quantile(valid_wavelet_boot, 0.95)) if valid_wavelet_boot.size else np.nan
    amplitude_p_value = float(np.mean(valid_amplitude_boot >= amplitude_train_cv_sharpe)) if valid_amplitude_boot.size else np.nan
    amplitude_q95 = float(np.quantile(valid_amplitude_boot, 0.95)) if valid_amplitude_boot.size else np.nan

    cross_asset_rows: List[Dict[str, object]] = []
    if run_cross_asset and use_oos:
        with st.spinner("Running cross-asset validation..."):
            for symbol in SECONDARY_SYMBOLS:
                cross_asset_rows.append(
                    {
                        "symbol": symbol,
                        "wavelet_filtered_sharpe": evaluate_cross_asset(
                            target_bars,
                            symbol,
                            "wavelet",
                            best_win,
                            tc_bps,
                            oos_pct,
                            use_vwap_filter,
                            use_ema_filter,
                            use_volume_filter,
                            min_volume_z,
                            amplitude_tinactive,
                        ).get("holdout_sharpe", np.nan),
                        "amplitude_filtered_sharpe": evaluate_cross_asset(
                            target_bars,
                            symbol,
                            amplitude_signal_kind,
                            best_amp,
                            tc_bps,
                            oos_pct,
                            use_vwap_filter,
                            use_ema_filter,
                            use_volume_filter,
                            min_volume_z,
                            amplitude_tinactive,
                            candidate_amplitudes if use_amplitude_ensemble else None,
                            use_amplitude_dynamic_scaling,
                            ensemble_width,
                        ).get("holdout_sharpe", np.nan),
                    }
                )

    def build_report_checks(
        name: str,
        train_cv_sharpe: float,
        q95: float,
        raw_sharpe: float,
        filtered_sharpe: float,
    ) -> List[Tuple[str, bool]]:
        checks = [
            (f"{name}: Train CV > bootstrap 95%", bool(not np.isnan(q95) and train_cv_sharpe > q95)),
            (f"{name}: Filtered holdout Sharpe > 0", bool(not np.isnan(filtered_sharpe) and filtered_sharpe > 0)),
            (f"{name}: Filtered > raw", bool(not np.isnan(filtered_sharpe) and not np.isnan(raw_sharpe) and filtered_sharpe > raw_sharpe)),
            (f"{name}: Filtered > buy-and-hold", bool(not np.isnan(filtered_sharpe) and not np.isnan(buy_hold_sharpe) and filtered_sharpe > buy_hold_sharpe)),
            (f"{name}: Filtered > MA baseline", bool(not np.isnan(filtered_sharpe) and not np.isnan(ma_sharpe) and filtered_sharpe > ma_sharpe)),
        ]
        if cross_asset_rows:
            if name == "Wavelet":
                cross_ok = any(not np.isnan(row.get("wavelet_filtered_sharpe", np.nan)) and row.get("wavelet_filtered_sharpe", np.nan) > 0 for row in cross_asset_rows)
            else:
                cross_ok = any(not np.isnan(row.get("amplitude_filtered_sharpe", np.nan)) and row.get("amplitude_filtered_sharpe", np.nan) > 0 for row in cross_asset_rows)
            checks.append((f"{name}: Positive on secondary asset", cross_ok))
        return checks

    wavelet_report_checks = build_report_checks(
        "Wavelet",
        wavelet_train_cv_sharpe,
        wavelet_q95,
        wavelet_raw_sharpe,
        wavelet_filtered_sharpe,
    )
    amplitude_report_checks = build_report_checks(
        "Amplitude",
        amplitude_train_cv_sharpe,
        amplitude_q95,
        amplitude_raw_sharpe,
        amplitude_filtered_sharpe,
    )
    report_checks = wavelet_report_checks + amplitude_report_checks

    passed_checks = sum(1 for _, ok in report_checks if ok)
    total_checks = len(report_checks)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Wavelet window", f"{best_win} bars")
        st.metric("Wavelet train CV", f"{wavelet_train_cv_sharpe:.4f}")
    with col2:
        st.metric("Amplitude threshold", f"{best_amp:.2f}")
        st.metric("Amplitude train CV", f"{amplitude_train_cv_sharpe:.4f}")
    with col3:
        st.metric("Buy/Hold Sharpe", f"{buy_hold_sharpe:.4f}")
        st.metric("MA Sharpe", f"{ma_sharpe:.4f}")
    with col4:
        st.metric("Checks passed", f"{passed_checks}/{total_checks}")
        st.metric(
            "Wavelet / Amp p-value",
            f"{wavelet_p_value:.1%} / {amplitude_p_value:.1%}" if not (np.isnan(wavelet_p_value) or np.isnan(amplitude_p_value)) else "n/a",
        )

    st.subheader("Pass / Fail Report")
    report_df = pd.DataFrame(
        [{"check": label, "result": "PASS" if ok else "FAIL"} for label, ok in report_checks]
    )
    st.dataframe(report_df, width="stretch", hide_index=True)

    st.subheader("Wavelet Window Search")
    st.dataframe(wavelet_score_df, width="stretch", hide_index=True)

    st.subheader("Amplitude Threshold Search")
    st.dataframe(amplitude_score_df, width="stretch", hide_index=True)

    st.subheader("Strategy Comparison")
    compare_df = pd.DataFrame(
        [
            {"strategy": "Raw wavelet", "sharpe": wavelet_raw_sharpe},
            {"strategy": "Feature-filtered wavelet", "sharpe": wavelet_filtered_sharpe},
            {"strategy": "Raw amplitude", "sharpe": amplitude_raw_sharpe},
            {"strategy": "Feature-filtered amplitude" if not use_amplitude_ensemble else "Feature-filtered amplitude ensemble", "sharpe": amplitude_filtered_sharpe},
            {"strategy": "Buy and hold", "sharpe": buy_hold_sharpe},
            {"strategy": "24/72 MA crossover", "sharpe": ma_sharpe},
        ]
    )
    st.dataframe(compare_df, width="stretch", hide_index=True)

    filter_state_df = pd.DataFrame(
        [
            {"filter": "VWAP confirmation", "enabled": use_vwap_filter},
            {"filter": "EMA slope confirmation", "enabled": use_ema_filter},
            {"filter": "Volume z-score", "enabled": use_volume_filter},
            {"filter": "Min volume z", "enabled": min_volume_z},
            {"filter": "Amplitude inactivity bars", "enabled": amplitude_tinactive},
            {"filter": "Use amplitude ensemble", "enabled": use_amplitude_ensemble},
            {"filter": "Use dynamic amplitude scaling", "enabled": use_amplitude_dynamic_scaling},
            {"filter": "Amplitude ensemble width", "enabled": ensemble_width},
        ]
    )
    st.subheader("Filter Configuration")
    st.dataframe(filter_state_df, width="stretch", hide_index=True)

    if cross_asset_rows:
        st.subheader("Cross-Asset Check")
        st.dataframe(pd.DataFrame(cross_asset_rows), width="stretch", hide_index=True)

    st.subheader("Filtered Equity Curves")
    chart_label = "Holdout Equity" if use_oos else "In-sample Equity"
    st.line_chart(
        pd.DataFrame(
            {
                f"Wavelet {chart_label}": wavelet_filtered_equity,
                f"Amplitude {chart_label}": amplitude_filtered_equity,
            }
        ),
        width="stretch",
    )

    st.caption(
        "Candles and live feature snapshots are stored in SQLite. Historical backtests currently use candle-derived features only; live orderbook imbalance is logged for future out-of-sample research. Wavelet and amplitude signals are both tested under the same holdout, bootstrap, baseline, and cross-asset framework."
    )
else:
    st.info("Click the button to fetch real BTC data and run the feature-filtered validation pass.")
