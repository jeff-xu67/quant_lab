from __future__ import annotations

from typing import Any

import polars as pl

from quant_lab.core.events import Event, EventBus, MarketEvent, SignalEvent
from quant_lab.strategy.base import Strategy, register_strategy


@register_strategy
class MomentumRotation(Strategy):
    """Rank ETFs by N-day return, hold top K, rebalance every M days.

    Params:
        lookback: momentum calculation window (default: 20)
        top_k: number of ETFs to hold (default: 3)
        rebalance_days: rebalance frequency (default: 20)
    """

    def __init__(self, event_bus: EventBus, params: dict[str, Any]) -> None:
        super().__init__(event_bus, params)
        self.lookback: int = params.get("lookback", 20)
        self.top_k: int = params.get("top_k", 3)
        self.rebalance_days: int = params.get("rebalance_days", 20)
        self._current_holdings: set[str] = set()

    @property
    def name(self) -> str:
        return "MomentumRotation"

    def warmup_period(self) -> int:
        return self.lookback

    def on_market(self, event: Event) -> None:
        assert isinstance(event, MarketEvent)
        self._bar_count += 1
        self._update_history(event.bars, self.lookback + 1)

        if self._bar_count < self.warmup_period():
            return

        if (self._bar_count - self.warmup_period()) % self.rebalance_days != 0:
            return

        history = self.get_history_df()
        if history.is_empty():
            return

        momentum = (
            history.sort("trade_date")
            .group_by("ts_code")
            .agg(
                pl.col("close").first().alias("close_start"),
                pl.col("close").last().alias("close_end"),
            )
            .with_columns(
                ((pl.col("close_end") / pl.col("close_start")) - 1.0).alias("momentum")
            )
            .sort("momentum", descending=True)
        )

        top_symbols = set(momentum.head(self.top_k)["ts_code"].to_list())

        for symbol in self._current_holdings - top_symbols:
            self.event_bus.publish(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    direction=0.0,
                )
            )

        for symbol in top_symbols:
            self.event_bus.publish(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    direction=1.0,
                    strength=1.0 / self.top_k,
                )
            )

        self._current_holdings = top_symbols
