import time
from typing import Optional, Tuple
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
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
st.markdown("Live Kraken BTC/USD 1h data with walk-forward training and true holdout testing.")
st.markdown("Live BTC/USD 1h data via ccxt with walk-forward training and true holdout testing.")


def _safe_sharpe(rets: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return float(max(sigma * multiplier, 1e-4))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_real_btc_data(target_bars: int = 4000) -> Optional[pd.DataFrame]:
def _fetch_from_exchange(
    exchange_id: str,
    symbol: str,
    target_bars: int,
    per_request_limit: int,
    warmup_bars: int = 256,
) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    meta: Dict[str, str] = {"exchange": exchange_id, "symbol": symbol}
    try:
        exchange = ccxt.kraken({"enableRateLimit": True})
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        exchange.load_markets()
        if not exchange.has.get("fetchOHLCV"):
            meta["error"] = "fetchOHLCV not supported"
            return None, meta

        timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
        request_limit = min(720, target_bars)
        warmup_bars = 256
        total_needed = target_bars + warmup_bars
        request_limit = max(50, min(per_request_limit, total_needed))
        now_ms = exchange.milliseconds()
        since = now_ms - total_needed * timeframe_ms

        all_candles: list[list[float]] = []
        max_loops = max(3, int(np.ceil(total_needed / request_limit)) + 3)
        all_candles: List[List[float]] = []
        max_loops = max(3, int(np.ceil(total_needed / request_limit)) + 5)

        for _ in range(max_loops):
            candles = exchange.fetch_ohlcv(
                SYMBOL,
                symbol,
                timeframe=TIMEFRAME,
                since=since,
                limit=request_limit,
            time.sleep(exchange.rateLimit / 1000.0)

        if not all_candles:
            return None
            meta["error"] = "no candles returned"
            return None, meta

        df = pd.DataFrame(
            all_candles,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.tail(total_needed).copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

        return df
    except Exception:
        return None
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


if st.button("Fetch Real BTC Data & Run Test", type="primary"):
    with st.spinner("Fetching Kraken BTC/USD candles..."):
        df = fetch_real_btc_data(target_bars)
    with st.spinner("Fetching BTC/USD candles from available exchanges..."):
        df, fetch_meta, fetch_attempts = fetch_real_btc_data(target_bars)

    if df is None or len(df) < 1200:
        st.error("Fetch failed or returned too little data. Try again in a moment.")
    min_needed = max(1200, fixed_window + 300)
    if df is None or len(df) < min_needed:
        st.error("Fetch failed or returned too little data for an honest test.")
        if fetch_attempts:
            st.write(pd.DataFrame(fetch_attempts))
        st.stop()

    prices = df["close"].to_numpy(dtype=float)
    start_dt = shown_df["datetime"].iloc[0]
    end_dt = shown_df["datetime"].iloc[-1]
    st.success(
        f"Fetched {len(shown_df)} BTC/USD 1h bars from {start_dt:%Y-%m-%d %H:%M UTC} to {end_dt:%Y-%m-%d %H:%M UTC}."
        f"Fetched {len(shown_df)} BTC/USD 1h bars from {start_dt:%Y-%m-%d %H:%M UTC} to {end_dt:%Y-%m-%d %H:%M UTC} using {fetch_meta.get('exchange', 'unknown')}."
    )
    if fetch_attempts:
        with st.expander("Fetch diagnostics"):
            st.dataframe(pd.DataFrame(fetch_attempts), use_container_width=True, hide_index=True)

    candidate_windows = sorted(
        {
    st.line_chart(pd.DataFrame({chart_label: equity}), use_container_width=True)

    st.caption(
        "Fetch source: Kraken via ccxt. Window selection uses only the training set. Holdout evaluation is performed once after selection."
        "Fetch source: exchange selected via ccxt. Window selection uses only the training set. Holdout evaluation is performed once after selection."
    )
else:
    st.info("Click the button to fetch real BTC data and run a stricter validation pass.")
