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


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def store_ohlcv(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> None:
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
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
    conn.executemany(
        """
        INSERT OR REPLACE INTO ohlcv
        (exchange, symbol, timeframe, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def store_feature_snapshot(
    exchange: str,
    symbol: str,
    timeframe: str,
