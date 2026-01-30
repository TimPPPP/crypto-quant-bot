"""
Adaptive parameter tuning for live trading and backtests.

Uses rolling trade performance to adjust a small set of strategy/risk knobs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config_adaptive as adaptive_cfg

logger = logging.getLogger("AdaptiveTuner")


@dataclass
class TradeStats:
    trade_count: int
    win_rate: float
    total_return: float
    avg_return: float
    median_return: float
    return_std: float
    sharpe_like: float
    max_drawdown: float
    stop_loss_rate: float
    profit_factor: float
    hold_hours_avg: Optional[float]
    hold_hours_p90: Optional[float]


class TradeHistoryStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("recorded_ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_since(self, since_ts: datetime) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        results: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(rec.get("exit_ts") or rec.get("recorded_ts"))
                if ts is None:
                    continue
                if ts >= since_ts:
                    results.append(rec)
        return results


class AdaptiveController:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        window_days: Optional[int] = None,
        min_trades: Optional[int] = None,
        trades_path: Optional[str] = None,
        overrides_path: Optional[str] = None,
    ) -> None:
        self.enabled = adaptive_cfg.ADAPTIVE_ENABLED if enabled is None else bool(enabled)
        self.window_days = adaptive_cfg.ADAPTIVE_WINDOW_DAYS if window_days is None else int(window_days)
        self.min_trades = adaptive_cfg.ADAPTIVE_MIN_TRADES if min_trades is None else int(min_trades)

        trades_path = trades_path or adaptive_cfg.ADAPTIVE_TRADES_PATH
        overrides_path = overrides_path or adaptive_cfg.ADAPTIVE_OVERRIDES_PATH
        self.trade_store = TradeHistoryStore(trades_path)
        self.overrides_path = Path(overrides_path)
        self.overrides_path.parent.mkdir(parents=True, exist_ok=True)

        self.last_update_ts = _parse_ts(self._load_overrides().get("last_update_ts"))

    def record_trade(self, record: Dict[str, Any]) -> None:
        self.trade_store.append(record)

    def maybe_update_live(self, current_params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[TradeStats]]:
        if not self.enabled:
            return None, None
        now = datetime.now(timezone.utc)
        if self.last_update_ts and (now - self.last_update_ts) < timedelta(days=self.window_days):
            return None, None

        since_ts = now - timedelta(days=self.window_days)
        trades = self.trade_store.load_since(since_ts)
        stats = compute_trade_stats(trades)
        if stats.trade_count < self.min_trades:
            logger.info("Adaptive update skipped (trades=%d < min=%d)", stats.trade_count, self.min_trades)
            return None, stats

        overrides = recommend_overrides(current_params, stats)
        if not overrides:
            return None, stats

        payload = {
            "last_update_ts": now.isoformat(),
            "overrides": overrides,
            "stats": stats.__dict__,
        }
        self._save_overrides(payload)
        self.last_update_ts = now
        return overrides, stats

    def maybe_update_backtest(
        self,
        *,
        current_params: Dict[str, Any],
        window_id: str,
        trades: Iterable[float],
        stop_loss_rate: float,
        hold_hours: Optional[Iterable[float]] = None,
        log_path: Path,
    ) -> Tuple[Optional[Dict[str, Any]], TradeStats]:
        stats = compute_trade_stats(
            [{"pnl_return": float(r)} for r in trades],
            stop_loss_rate=stop_loss_rate,
            hold_hours=hold_hours,
        )
        if stats.trade_count < self.min_trades:
            return None, stats

        overrides = recommend_overrides(current_params, stats)
        if overrides:
            _append_jsonl(
                log_path,
                {
                    "window_id": window_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "overrides": overrides,
                    "stats": stats.__dict__,
                },
            )
        return overrides, stats

    def _load_overrides(self) -> Dict[str, Any]:
        if not self.overrides_path.exists():
            return {}
        try:
            return json.loads(self.overrides_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load_saved_overrides(self) -> Dict[str, Any]:
        payload = self._load_overrides()
        if isinstance(payload, dict):
            return payload.get("overrides", {}) or {}
        return {}

    def _save_overrides(self, payload: Dict[str, Any]) -> None:
        self.overrides_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_trade_stats(
    trades: List[Dict[str, Any]],
    *,
    stop_loss_rate: Optional[float] = None,
    hold_hours: Optional[Iterable[float]] = None,
) -> TradeStats:
    returns: List[float] = []
    stop_loss_hits = 0
    hold_values: List[float] = []
    for rec in trades:
        ret = rec.get("pnl_return")
        if ret is None:
            continue
        try:
            ret_f = float(ret)
        except (TypeError, ValueError):
            continue
        returns.append(ret_f)
        if rec.get("exit_reason") == "STOP LOSS":
            stop_loss_hits += 1
        hold_value = rec.get("hold_hours")
        if hold_value is None:
            entry_ts = _parse_ts(rec.get("entry_ts"))
            exit_ts = _parse_ts(rec.get("exit_ts"))
            if entry_ts and exit_ts:
                hold_value = (exit_ts - entry_ts).total_seconds() / 3600.0
        if hold_value is not None:
            try:
                hold_values.append(float(hold_value))
            except (TypeError, ValueError):
                pass

    if hold_hours is not None:
        for value in hold_hours:
            try:
                hold_values.append(float(value))
            except (TypeError, ValueError):
                continue

    trade_count = len(returns)
    if trade_count == 0:
        return TradeStats(
            trade_count=0,
            win_rate=0.0,
            total_return=0.0,
            avg_return=0.0,
            median_return=0.0,
            return_std=0.0,
            sharpe_like=0.0,
            max_drawdown=0.0,
            stop_loss_rate=stop_loss_rate or 0.0,
            profit_factor=0.0,
            hold_hours_avg=None,
            hold_hours_p90=None,
        )

    wins = sum(1 for r in returns if r > 0)
    total_return = sum(returns)
    avg_return = total_return / trade_count
    median_return = _median(returns)
    return_std = _std(returns)
    sharpe_like = avg_return / return_std if return_std > 1e-10 else 0.0
    max_drawdown = _max_drawdown(returns)
    profit_factor = _profit_factor(returns)

    hold_hours_avg = None
    hold_hours_p90 = None
    if hold_values:
        hold_hours_avg = sum(hold_values) / len(hold_values)
        hold_hours_p90 = _percentile(hold_values, 0.9)

    if stop_loss_rate is None:
        stop_loss_rate = stop_loss_hits / trade_count if trade_count else 0.0

    return TradeStats(
        trade_count=trade_count,
        win_rate=wins / trade_count,
        total_return=total_return,
        avg_return=avg_return,
        median_return=median_return,
        return_std=return_std,
        sharpe_like=sharpe_like,
        max_drawdown=max_drawdown,
        stop_loss_rate=stop_loss_rate,
        profit_factor=profit_factor,
        hold_hours_avg=hold_hours_avg,
        hold_hours_p90=hold_hours_p90,
    )


def recommend_overrides(current: Dict[str, Any], stats: TradeStats) -> Dict[str, Any]:
    hold_too_long = stats.hold_hours_avg is not None and stats.hold_hours_avg > adaptive_cfg.ADAPTIVE_RISK_OFF_MAX_HOLD_HOURS
    hold_too_short = stats.hold_hours_avg is not None and stats.hold_hours_avg < adaptive_cfg.ADAPTIVE_RISK_OFF_MIN_HOLD_HOURS
    risk_off = (
        stats.total_return < 0
        or stats.win_rate < adaptive_cfg.ADAPTIVE_RISK_OFF_WIN_RATE
        or stats.max_drawdown > adaptive_cfg.ADAPTIVE_RISK_OFF_DD
        or stats.stop_loss_rate > adaptive_cfg.ADAPTIVE_RISK_OFF_STOP_LOSS_RATE
        or stats.profit_factor < adaptive_cfg.ADAPTIVE_RISK_OFF_PROFIT_FACTOR
        or hold_too_long
        or hold_too_short
    )
    hold_ok = stats.hold_hours_avg is None or stats.hold_hours_avg <= adaptive_cfg.ADAPTIVE_RISK_ON_MAX_HOLD_HOURS
    risk_on = (
        stats.total_return > adaptive_cfg.ADAPTIVE_RISK_ON_TOTAL_RETURN
        and stats.win_rate > adaptive_cfg.ADAPTIVE_RISK_ON_WIN_RATE
        and stats.sharpe_like > adaptive_cfg.ADAPTIVE_RISK_ON_SHARPE
        and stats.max_drawdown < adaptive_cfg.ADAPTIVE_RISK_ON_DD
        and stats.profit_factor > adaptive_cfg.ADAPTIVE_RISK_ON_PROFIT_FACTOR
        and stats.median_return > 0
        and hold_ok
    )

    if not (risk_off or risk_on):
        return {}

    overrides: Dict[str, Any] = {}
    entry_z = float(current.get("ENTRY_Z", 2.3))
    exit_z = float(current.get("EXIT_Z", 0.5))
    min_profit = float(current.get("MIN_PROFIT_HURDLE", 0.012))
    max_pos = int(current.get("MAX_PORTFOLIO_POSITIONS", 8))
    max_pos_coin = int(current.get("MAX_POSITIONS_PER_COIN", 2))
    stop_loss_pct = float(current.get("STOP_LOSS_PCT", 0.025))

    if risk_off:
        entry_z += adaptive_cfg.ENTRY_Z_STEP
        exit_z += adaptive_cfg.EXIT_Z_STEP
        min_profit += adaptive_cfg.MIN_PROFIT_STEP
        max_pos -= adaptive_cfg.MAX_POSITIONS_STEP
        max_pos_coin -= adaptive_cfg.MAX_POSITIONS_PER_COIN_STEP
        stop_loss_pct -= adaptive_cfg.STOP_LOSS_PCT_STEP
    elif risk_on:
        entry_z -= adaptive_cfg.ENTRY_Z_STEP
        exit_z -= adaptive_cfg.EXIT_Z_STEP
        min_profit -= adaptive_cfg.MIN_PROFIT_STEP
        max_pos += adaptive_cfg.MAX_POSITIONS_STEP
        max_pos_coin += adaptive_cfg.MAX_POSITIONS_PER_COIN_STEP
        stop_loss_pct += adaptive_cfg.STOP_LOSS_PCT_STEP

    entry_z = _clamp(entry_z, *adaptive_cfg.ENTRY_Z_BOUNDS)
    exit_z = _clamp(exit_z, *adaptive_cfg.EXIT_Z_BOUNDS)
    min_profit = _clamp(min_profit, *adaptive_cfg.MIN_PROFIT_HURDLE_BOUNDS)
    max_pos = int(_clamp(max_pos, *adaptive_cfg.MAX_PORTFOLIO_POSITIONS_BOUNDS))
    max_pos_coin = int(_clamp(max_pos_coin, *adaptive_cfg.MAX_POSITIONS_PER_COIN_BOUNDS))
    stop_loss_pct = _clamp(stop_loss_pct, *adaptive_cfg.STOP_LOSS_PCT_BOUNDS)

    overrides["ENTRY_Z"] = entry_z
    overrides["EXIT_Z"] = exit_z
    overrides["MIN_PROFIT_HURDLE"] = min_profit
    overrides["MAX_PORTFOLIO_POSITIONS"] = max_pos
    overrides["MAX_POSITIONS_PER_COIN"] = max_pos_coin
    overrides["STOP_LOSS_PCT"] = stop_loss_pct

    return overrides


def apply_overrides_to_backtest(cfg_module: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(cfg_module, key):
            setattr(cfg_module, key, value)


def apply_overrides_to_live(bot: Any, overrides: Dict[str, Any]) -> None:
    entry_z = overrides.get("ENTRY_Z")
    exit_z = overrides.get("EXIT_Z")

    if entry_z is not None:
        for kf in bot.active_pairs.values():
            kf.entry_z_threshold = float(entry_z)
        for kf in bot.retiring_pairs.values():
            kf.entry_z_threshold = float(entry_z)

    if exit_z is not None:
        bot.exit_z_short = float(exit_z)
        bot.exit_z_long = -float(exit_z)

    if overrides.get("MIN_PROFIT_HURDLE") is not None:
        bot.trade_executor.MIN_NET_PROFIT = float(overrides["MIN_PROFIT_HURDLE"])

    if overrides.get("MAX_PORTFOLIO_POSITIONS") is not None:
        bot.risk_engine.max_positions = int(overrides["MAX_PORTFOLIO_POSITIONS"])

    if overrides.get("MAX_POSITIONS_PER_COIN") is not None:
        bot.risk_engine.max_positions_per_coin = int(overrides["MAX_POSITIONS_PER_COIN"])

    if overrides.get("STOP_LOSS_PCT") is not None:
        bot.risk_engine.stop_loss_pct = float(overrides["STOP_LOSS_PCT"])


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        ts = datetime.fromisoformat(str(value))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def _std(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def _median(values: List[float]) -> float:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct))
    return float(values[max(0, min(idx, len(values) - 1))])


def _profit_factor(values: List[float]) -> float:
    gains = sum(v for v in values if v > 0)
    losses = sum(v for v in values if v < 0)
    if losses == 0:
        return 99.0 if gains > 0 else 0.0
    return gains / abs(losses)


def _max_drawdown(returns: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
