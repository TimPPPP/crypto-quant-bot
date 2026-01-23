from __future__ import annotations

import logging
import os
from typing import Optional

import requests

QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT = int(os.getenv("QUESTDB_PORT", "9000"))
QUESTDB_HTTP = f"http://{QUESTDB_HOST}:{QUESTDB_PORT}/exec"

logger = logging.getLogger("backtest.fix_candles_1h")


def _exec(query: str, session: Optional[requests.Session] = None) -> None:
    sess = session or requests.Session()
    resp = sess.get(QUESTDB_HTTP, params={"query": query}, timeout=60)
    resp.raise_for_status()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sess = requests.Session()

    logger.info("Creating candles_1h_dedup (if needed)...")
    _exec(
        """
        CREATE TABLE IF NOT EXISTS candles_1h_dedup (
            timestamp TIMESTAMP,
            symbol SYMBOL capacity 256 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            turnover DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """,
        session=sess,
    )

    logger.info("Deduplicating candles_1h into candles_1h_dedup...")
    _exec(
        """
        INSERT INTO candles_1h_dedup
        SELECT
            timestamp,
            symbol,
            last(open) AS open,
            last(high) AS high,
            last(low) AS low,
            last(close) AS close,
            last(volume) AS volume,
            last(turnover) AS turnover
        FROM candles_1h
        GROUP BY timestamp, symbol;
        """,
        session=sess,
    )

    logger.info("✅ Deduped table ready: candles_1h_dedup")


if __name__ == "__main__":
    main()
