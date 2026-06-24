from __future__ import annotations

from typing import Any

import polars as pl

from quant_lab.core.events import Event, EventBus, MarketEvent, SignalEvent
from quant_lab.strategy.base import Strategy, register_strategy


@register_strategy
class TrendFollowing(Strategy):
    """Moving average crossover or channel breakout.

    Params:
        mode: "crossover" or "breakout" (default: "crossover")
        fast_period: fast MA window (default: 10)
        slow_period: slow MA window (default: 30)
        breakout_period: for breakout mode (default: 20)
    """

    def __init__(self, event_bus: EventBus, params: dict[str, Any]) -> None:
        super().__init__(event_bus, params)
        self.mode: str = params.get("mode", "crossover")
        self.fast_period: int = params.get("fast_period", 10)
        self.slow_period: int = params.get("slow_period", 30)
        self.breakout_period: int = params.get("breakout_period", 20)
        self._in_position: set[str] = set()

    @property
    def name(self) -> str:
        return "TrendFollowing"

    def warmup_period(self) -> int:
        if self.mode == "crossover":
            return self.slow_period
        return self.breakout_period

    def on_market(self, event: Event) -> None:
        assert isinstance(event, MarketEvent)
        self._bar_count += 1
        max_window = self.slow_period + 2 if self.mode == "crossover" else self.breakout_period + 2
        self._update_history(event.bars, max_window)

        if self._bar_count < self.warmup_period():
            return

        history = self.get_history_df()
        if history.is_empty():
            return

        if self.mode == "crossover":
            self._crossover_signals(history, event)
        else:
            self._breakout_signals(history, event)

    def _crossover_signals(self, history: pl.DataFrame, event: MarketEvent) -> None:
        signals = (
            history.sort("trade_date")
            .group_by("ts_code")
            .agg(
                pl.col("close").tail(self.fast_period).mean().alias("fast_ma"),
                pl.col("close").tail(self.slow_period).mean().alias("slow_ma"),
                pl.col("close")
                .shift(1)
                .tail(self.fast_period)
                .mean()
                .alias("prev_fast_ma"),
                pl.col("close")
                .shift(1)
                .tail(self.slow_period)
                .mean()
                .alias("prev_slow_ma"),
            )
        )

        for row in signals.iter_rows(named=True):
            symbol = row["ts_code"]
            fast_ma = row["fast_ma"]
            slow_ma = row["slow_ma"]
            prev_fast = row["prev_fast_ma"]
            prev_slow = row["prev_slow_ma"]

            if prev_fast is None or prev_slow is None:
                continue

            cross_up = prev_fast <= prev_slow and fast_ma > slow_ma
            cross_down = prev_fast >= prev_slow and fast_ma < slow_ma

            if cross_up and symbol not in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=1.0,
                    )
                )
                self._in_position.add(symbol)

            elif cross_down and symbol in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=0.0,
                    )
                )
                self._in_position.discard(symbol)

    def _breakout_signals(self, history: pl.DataFrame, event: MarketEvent) -> None:
        signals = (
            history.sort("trade_date")
            .group_by("ts_code")
            .agg(
                pl.col("high").tail(self.breakout_period).max().alias("channel_high"),
                pl.col("low").tail(self.breakout_period).min().alias("channel_low"),
                pl.col("close").last().alias("last_close"),
            )
        )

        for row in signals.iter_rows(named=True):
            symbol = row["ts_code"]
            last_close = row["last_close"]
            channel_high = row["channel_high"]
            channel_low = row["channel_low"]

            if last_close >= channel_high and symbol not in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=1.0,
                    )
                )
                self._in_position.add(symbol)

            elif last_close <= channel_low and symbol in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=0.0,
                    )
                )
                self._in_position.discard(symbol)
