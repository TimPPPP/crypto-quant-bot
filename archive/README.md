# Archive Directory

This directory contains historical and outdated files that have been moved out of the main codebase for cleanup purposes.

## Directory Structure

### collectors/
**Archived:** December 13, 2025
- `ingest_1m.py` - Old Hyperliquid 1-minute candle ingester
  - **Reason:** Hyperliquid API only provides ~3.5 days of historical data
  - **Replacement:** `src/collectors/ingest_okx.py` (primary), `ingest_coinbase.py` (secondary)

### scripts/
**Archived:** December 13, 2025
- `ingest_all.py` - Old parallel ingestion script
  - **Reason:** Used the archived `ingest_1m.py`
  - **Replacement:** `scripts/ingest_all.py` (new multi-source orchestrator)

### notebooks/
**Archived:** December 13, 2025 & January 15, 2026
- `02_strategy_validation.ipynb` - Strategy validation notebook
- `01_data_check.ipynb` - Data quality check notebook
  - **Reason:** Historical analysis from earlier development
  - **Current:** Use `research/backtest/run_simulation.py` for backtesting

### old_results/
**Archived:** December 13, 2025 & January 15, 2026
- 52 result directories from December 2025 and early January 2026
  - **Reason:** Historical backtest runs, keeping recent results only
  - **Current:** Active results in `results/` directory (74 directories)

### research/
**Archived:** December 13, 2025
- Nested research directory with brain_scan.png
  - **Reason:** Old visualization from early development

---

## Current Active Components

### Data Ingestion (`src/collectors/`)
| File | Purpose | Priority |
|------|---------|----------|
| `ingest_okx.py` | OKX 1m candles (365 days) | Primary |
| `ingest_coinbase.py` | Coinbase 1m candles | Secondary |
| `ingest_binance.py` | Binance candles (geo-restricted) | Fallback |
| `ingest_hyperliquid.py` | Hyperliquid (~3.5 days only) | Native tokens |
| `ingest_funding.py` | Funding rates | Live trading |
| `merge_sources.py` | Multi-source data merging | Export |
| `base_ingester.py` | Abstract base class | Core |
| `symbol_mapping.py` | Exchange symbol mapping | Core |

### Backtesting (`src/backtest/`)
| File | Purpose |
|------|---------|
| `config_backtest.py` | Single source of truth for all parameters |
| `pnl_engine.py` | Numba-accelerated P&L calculation |
| `signal_generation.py` | Kalman z-scores, entry/exit signals |
| `position_sizing.py` | Conviction-based sizing |
| `accountant_filter.py` | Signal filtering |
| `performance_report.py` | Metrics and reporting |

### Live Trading (`src/`)
| File | Purpose |
|------|---------|
| `main.py` | Trading bot orchestrator |
| `execution/exchange_client.py` | Exchange API wrapper |
| `execution/risk.py` | Risk management |
| `execution/executor.py` | Order execution |

---

## Notes

These files are archived for reference purposes. They can be deleted permanently if disk space is needed, as all critical functionality has been replaced by newer implementations.

**Last cleanup:** January 15, 2026
