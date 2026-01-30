# src/backtest/diagnostics.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.diagnostics")


def _parse_pair(pair: str) -> Tuple[str, str]:
    sep = getattr(cfg, "PAIR_ID_SEPARATOR", "-")
    if sep not in pair:
        raise ValueError(f"Invalid pair_id '{pair}'. Expected like 'ETH-BTC'.")
    y, x = pair.split(sep, 1)
    y, x = y.strip(), x.strip()
    if not y or not x:
        raise ValueError(f"Invalid pair_id '{pair}'.")
    return y, x


def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def plot_pair_diagnosis(
    *,
    run_dir: Path,
    pair_id: str,
    test_df: pd.DataFrame,
    z_score: pd.DataFrame,
    beta: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    expected_profit: Optional[pd.DataFrame] = None,
    spread_volatility: Optional[pd.DataFrame] = None,
    pnl_mode: str = "price",  # "price" or "log"
    save: bool = True,
) -> Path:
    """
    Step I — Pair-level deep dive.

    Produces a 2-panel plot:
      1) Spread with entry/exit markers
      2) Z-score with thresholds + entry/exit markers

    Output
    ------
    results/run_id/plots/pair_<id>_diagnosis.png
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if pair_id not in z_score.columns:
        raise KeyError(f"{pair_id} not found in z_score columns.")
    if pair_id not in beta.columns:
        raise KeyError(f"{pair_id} not found in beta columns.")
    if pair_id not in entries.columns or pair_id not in exits.columns:
        raise KeyError(f"{pair_id} not found in entries/exits columns.")

    # Align everything to test_df index
    idx = test_df.index
    zs = z_score[pair_id].reindex(idx)
    be = beta[pair_id].reindex(idx)
    en = entries[pair_id].reindex(idx).fillna(False).astype(bool)
    ex = exits[pair_id].reindex(idx).fillna(False).astype(bool)

    coin_y, coin_x = _parse_pair(pair_id)

    if coin_y not in test_df.columns or coin_x not in test_df.columns:
        raise KeyError(f"Missing columns in test_df for {pair_id}: need {coin_y}, {coin_x}")

    y = pd.to_numeric(test_df[coin_y], errors="coerce").reindex(idx)
    x = pd.to_numeric(test_df[coin_x], errors="coerce").reindex(idx)

    if pnl_mode == "log":
        y = np.log(y.where(y > 0))
        x = np.log(x.where(x > 0))

    # Spread uses time-varying beta (diagnostic). In PnL engine you lock beta at entry.
    spread = y - (be * x)

    # Thresholds from config (or fallbacks)
    entry_z = float(getattr(cfg, "ENTRY_Z", 2.0))
    exit_z = float(getattr(cfg, "EXIT_Z", 0.5))
    stop_z = float(getattr(cfg, "STOP_LOSS_Z", 4.5))

    # Optional overlays
    ep = expected_profit[pair_id].reindex(idx) if expected_profit is not None and pair_id in expected_profit.columns else None
    sv = spread_volatility[pair_id].reindex(idx) if spread_volatility is not None and pair_id in spread_volatility.columns else None

    # Marker positions
    entry_times = idx[en.to_numpy()]
    exit_times = idx[ex.to_numpy()]

    # Plot
    fig = plt.figure(figsize=(14, 8))

    # --- Panel 1: Spread ---
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(idx, spread.values, label="Spread (Y - beta*X)")
    if len(entry_times) > 0:
        ax1.scatter(entry_times, spread.loc[entry_times].values, marker="^", label="Entry")
    if len(exit_times) > 0:
        ax1.scatter(exit_times, spread.loc[exit_times].values, marker="v", label="Exit")

    title = f"{pair_id} Diagnosis — Spread"
    ax1.set_title(title)
    ax1.set_ylabel("Spread")
    ax1.legend(loc="best")

    # Optional second y-axis for expected_profit or spread_volatility (to avoid clutter)
    if ep is not None or sv is not None:
        ax1b = ax1.twinx()
        if sv is not None:
            ax1b.plot(idx, sv.values, linestyle="--", label="Spread Vol (std)")
        if ep is not None:
            ax1b.plot(idx, ep.values, linestyle=":", label="Expected Profit")
        ax1b.set_ylabel("Vol / Expected Profit")
        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1b.legend(h1 + h2, l1 + l2, loc="upper right")

    # --- Panel 2: Z-score ---
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(idx, zs.values, label="Z-score")

    # Threshold lines
    ax2.axhline(entry_z, linestyle="--", label=f"+Entry ({entry_z})")
    ax2.axhline(-entry_z, linestyle="--", label=f"-Entry ({-entry_z})")
    ax2.axhline(exit_z, linestyle=":", label=f"+Exit ({exit_z})")
    ax2.axhline(-exit_z, linestyle=":", label=f"-Exit ({-exit_z})")
    ax2.axhline(stop_z, linestyle="-.", label=f"+Stop ({stop_z})")
    ax2.axhline(-stop_z, linestyle="-.", label=f"-Stop ({-stop_z})")

    if len(entry_times) > 0:
        ax2.scatter(entry_times, zs.loc[entry_times].values, marker="^", label="Entry")
    if len(exit_times) > 0:
        ax2.scatter(exit_times, zs.loc[exit_times].values, marker="v", label="Exit")

    ax2.set_title(f"{pair_id} Diagnosis — Z-score")
    ax2.set_ylabel("Z")
    ax2.set_xlabel("Time")
    ax2.legend(loc="best")

    plt.tight_layout()

    filename = f"pair_{_sanitize_filename(pair_id)}_diagnosis.png"
    out_path = plots_dir / filename

    if save:
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("✅ Saved pair diagnosis: %s", out_path)
    else:
        # If caller wants to view interactively
        logger.info("Pair diagnosis generated (not saved).")
    return out_path
