#!/usr/bin/env python3
"""Swing Reversion Grid strategy template for the trading contest."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Import base infrastructure from the shared template bundle.
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "base-bot-template")
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/app/base"

sys.path.insert(0, BASE_PATH)

from strategy_interface import BaseStrategy, Portfolio, Signal, register_strategy  # type: ignore  # noqa: E402
from exchange_interface import MarketSnapshot  # type: ignore  # noqa: E402


@dataclass
class GridFrame:
    """Holds the anchor reference and computed grid levels."""

    anchor: float
    atr: float
    buy_levels: List[float]
    sell_levels: List[float]


class SwingGridStrategy(BaseStrategy):
    """Mean-reversion grid that scales into weakness and trims into strength."""

    def __init__(self, config: Dict[str, Any], exchange) -> None:
        super().__init__(config=config, exchange=exchange)
        self.anchor_period = max(60, int(config.get("anchor_period", 288)))
        self.atr_period = max(20, int(config.get("atr_period", 56)))
        self.grid_steps = max(1, int(config.get("grid_steps", 4)))
        self.atr_multiplier = float(config.get("atr_multiplier", 1.1))
        self.base_allocation = float(config.get("base_allocation", 0.12))
        self.max_position_fraction = float(config.get("max_position_fraction", 0.35))
        self.min_order_notional = float(config.get("min_order_notional", 150.0))
        self.min_atr = float(config.get("min_atr", 5.0))
        self.cooldown_minutes = max(0, int(config.get("cooldown_minutes", 30)))
        self.use_rsi_filter = bool(config.get("use_rsi_filter", False))
        self.rsi_period = max(5, int(config.get("rsi_period", 14)))
        self.low_rsi = float(config.get("low_rsi", 38.0))
        self.high_rsi = float(config.get("high_rsi", 62.0))
        configured_min_history = int(config.get("min_history_bars", 120))
        self.min_history_bars = max(60, configured_min_history)
        self.symbol = str(config.get("symbol", "BTC-USD"))
        raw_preload = config.get("preload_history_bars", 0)
        try:
            configured_preload = int(raw_preload)
        except (TypeError, ValueError):
            configured_preload = 0

        baseline_preload = max(self.min_history_bars, self.anchor_period + 20, self.atr_period + 20, self.rsi_period + 20)
        if configured_preload > 0:
            self.preload_history_bars = max(configured_preload, baseline_preload)
        else:
            self.preload_history_bars = baseline_preload
        self._history_buffer: List[float] = []
        self._max_history_bars = max(
            self.preload_history_bars * 2,
            self.anchor_period * 2,
            self.atr_period * 3,
            self.min_history_bars * 3,
            400,
        )

        self._filled_sizes: Dict[int, float] = {level: 0.0 for level in range(1, self.grid_steps + 1)}
        self._last_trade_time: Optional[datetime] = None

        self.last_signal_data: Dict[str, Any] = {"status": "initialized"}

    def prepare(self) -> None:
        self.last_signal_data = {"status": "ready"}
        self._preload_history()

    def generate_signal(self, market: MarketSnapshot, portfolio: Portfolio) -> Signal:
        prices = self._integrate_prices(market.prices)
        required = self._required_history()
        if len(prices) < required:
            self.last_signal_data = {
                "reason": "insufficient_history",
                "bars": len(prices),
                "needed": required,
            }
            return Signal("hold", reason="Insufficient history")

        anchor = self._sma(prices, self.anchor_period)
        atr = max(self.min_atr, self._atr(prices, self.atr_period))
        frame = self._build_frame(anchor, atr)

        now = self._with_timezone(market.timestamp)
        cooldown_ok = self._last_trade_time is None or now >= self._last_trade_time + timedelta(minutes=self.cooldown_minutes)

        rsi_value = self._rsi(prices, self.rsi_period)
        rsi_gate = True
        if self.use_rsi_filter and rsi_value is not None:
            rsi_gate = rsi_value <= self.low_rsi or rsi_value >= self.high_rsi

        equity = portfolio.cash + portfolio.quantity * market.current_price
        max_position_value = equity * self.max_position_fraction
        current_position_value = portfolio.quantity * market.current_price
        remaining_capacity = max(0.0, max_position_value - current_position_value)

        self.last_signal_data = {
            "price": round(market.current_price, 2),
            "anchor": round(frame.anchor, 2),
            "atr": round(frame.atr, 2),
            "rsi": round(rsi_value, 2) if rsi_value is not None else None,
            "remaining_capacity": round(remaining_capacity, 2),
            "history_bars": len(prices),
        }

        for level in range(self.grid_steps, 0, -1):
            held = self._filled_sizes.get(level, 0.0)
            if held <= 0:
                continue
            target = frame.sell_levels[level - 1]
            if market.current_price >= target:
                size = min(held, portfolio.quantity)
                if size <= 0:
                    continue
                self.last_signal_data.update({"action": "sell", "level": level, "target": round(target, 2)})
                return Signal("sell", size=size, reason=f"level_{level}_reversion", target_price=target)

        if not rsi_gate:
            self.last_signal_data["reason"] = "rsi_filter"
            return Signal("hold", reason="RSI filter active")
        if not cooldown_ok:
            self.last_signal_data["reason"] = "cooldown"
            return Signal("hold", reason="Cooldown")

        allocation_value = equity * self.base_allocation
        order_budget = min(allocation_value, remaining_capacity, portfolio.cash)

        for level in range(1, self.grid_steps + 1):
            if self._filled_sizes.get(level, 0.0) > 0:
                continue
            trigger = frame.buy_levels[level - 1]
            if market.current_price <= trigger and order_budget >= self.min_order_notional:
                size = order_budget / market.current_price
                if size <= 0:
                    continue
                self.last_signal_data.update({"action": "buy", "level": level, "entry": round(trigger, 2), "size": round(size, 6)})
                return Signal("buy", size=size, reason=f"level_{level}_entry", target_price=frame.sell_levels[level - 1])

        self.last_signal_data["reason"] = "no_trigger"
        return Signal("hold", reason="No grid trigger")

    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        timestamp = self._with_timezone(timestamp)
        self._last_trade_time = timestamp

        level = self._extract_level(signal.reason)
        if level is None or execution_size <= 0:
            return

        if signal.action == "buy":
            self._filled_sizes[level] = self._filled_sizes.get(level, 0.0) + execution_size
        elif signal.action == "sell":
            current = self._filled_sizes.get(level, 0.0)
            self._filled_sizes[level] = max(0.0, current - execution_size)

    def get_state(self) -> Dict[str, Any]:
        return {
            "filled_sizes": self._filled_sizes,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        raw = state.get("filled_sizes") or {}
        self._filled_sizes = {int(k): float(v) for k, v in raw.items()}
        stamp = state.get("last_trade_time")
        if stamp:
            self._last_trade_time = self._with_timezone(datetime.fromisoformat(stamp))

    def _required_history(self) -> int:
        return max(self.min_history_bars, self.anchor_period + 5, self.atr_period + 5, self.rsi_period + 5)

    def _build_frame(self, anchor: float, atr: float) -> GridFrame:
        buy = [anchor - self.atr_multiplier * atr * step for step in range(1, self.grid_steps + 1)]
        sell = [anchor + self.atr_multiplier * atr * step for step in range(1, self.grid_steps + 1)]
        return GridFrame(anchor=anchor, atr=atr, buy_levels=buy, sell_levels=sell)

    def _preload_history(self) -> None:
        if self.preload_history_bars <= 0:
            self.last_signal_data["preload"] = "skipped"
            return
        if not self.exchange or not hasattr(self.exchange, "fetch_market_snapshot"):
            self.last_signal_data["preload"] = "unavailable"
            return

        limit = max(self.preload_history_bars, self._required_history())
        try:
            snapshot = self.exchange.fetch_market_snapshot(self.symbol, limit=limit)
        except Exception as exc:  # noqa: BLE001 - surface preload issues for operators
            self.last_signal_data["preload_error"] = str(exc)
            return

        self._history_buffer = list(snapshot.prices)[-self._max_history_bars:]
        self.last_signal_data["preload_bars"] = len(self._history_buffer)

    def _integrate_prices(self, snapshot_prices: List[float]) -> List[float]:
        series = list(snapshot_prices)
        if not series:
            return list(self._history_buffer)

        if not self._history_buffer:
            self._history_buffer = series[-self._max_history_bars:]
            return list(self._history_buffer)

        max_overlap = min(len(self._history_buffer), len(series))
        overlap = 0
        for span in range(max_overlap, 0, -1):
            if self._history_buffer[-span:] == series[:span]:
                overlap = span
                break

        tail = series if overlap == 0 else series[overlap:]
        if tail:
            self._history_buffer.extend(tail)
        if len(self._history_buffer) > self._max_history_bars:
            self._history_buffer = self._history_buffer[-self._max_history_bars:]

        return list(self._history_buffer)

    @staticmethod
    def _sma(values: List[float], period: int) -> float:
        window = values[-period:]
        return sum(window) / len(window)

    @staticmethod
    def _atr(values: List[float], period: int) -> float:
        available = len(values)
        if available <= 1:
            return 0.0
        effective_period = min(period, available - 1)
        span = values[-(effective_period + 1):]
        if len(span) <= 1:
            return 0.0
        ranges = [abs(curr - prev) for prev, curr in zip(span, span[1:])]
        return sum(ranges) / len(ranges)

    @staticmethod
    def _rsi(values: List[float], period: int) -> Optional[float]:
        if len(values) <= period:
            return None
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            change = values[i] - values[i - 1]
            if change >= 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0
        rs = (gains / period) / (losses / period)
        return 100.0 - (100.0 / (1 + rs))

    @staticmethod
    def _extract_level(reason: Optional[str]) -> Optional[int]:
        if not reason or "level_" not in reason:
            return None
        try:
            return int(reason.split("level_")[-1].split("_")[0])
        except ValueError:
            return None

    @staticmethod
    def _with_timezone(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)


register_strategy("swing_grid", lambda cfg, ex: SwingGridStrategy(cfg, ex))
