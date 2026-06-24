from __future__ import annotations

from typing import Any

import polars as pl

from quant_lab.core.events import Event, EventBus, MarketEvent, SignalEvent
from quant_lab.strategy.base import Strategy, register_strategy


@register_strategy
class MeanReversion(Strategy):
    """Buy when price deviates N std below MA, sell on reversion.

    Params:
        ma_period: moving average window (default: 20)
        entry_zscore: z-score threshold to buy (default: -2.0)
        exit_zscore: z-score threshold to exit (default: 0.0)
    """

    def __init__(self, event_bus: EventBus, params: dict[str, Any]) -> None:
        super().__init__(event_bus, params)
        self.ma_period: int = params.get("ma_period", 20)
        self.entry_zscore: float = params.get("entry_zscore", -2.0)
        self.exit_zscore: float = params.get("exit_zscore", 0.0)
        self._in_position: set[str] = set()

    @property
    def name(self) -> str:
        return "MeanReversion"

    def warmup_period(self) -> int:
        return self.ma_period

    def on_market(self, event: Event) -> None:
        assert isinstance(event, MarketEvent)
        self._bar_count += 1
        self._update_history(event.bars, self.ma_period + 1)

        if self._bar_count < self.warmup_period():
            return

        history = self.get_history_df()
        if history.is_empty():
            return

        stats = (
            history.sort("trade_date")
            .group_by("ts_code")
            .agg(
                pl.col("close").mean().alias("ma"),
                pl.col("close").std().alias("std"),
                pl.col("close").last().alias("last_close"),
            )
            .filter(pl.col("std") > 0)
            .with_columns(
                ((pl.col("last_close") - pl.col("ma")) / pl.col("std")).alias("zscore")
            )
        )

        for row in stats.iter_rows(named=True):
            symbol = row["ts_code"]
            zscore = row["zscore"]

            if zscore <= self.entry_zscore and symbol not in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=1.0,
                        strength=abs(zscore) / abs(self.entry_zscore),
                    )
                )
                self._in_position.add(symbol)

            elif zscore >= self.exit_zscore and symbol in self._in_position:
                self.event_bus.publish(
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        direction=0.0,
                    )
                )
                self._in_position.discard(symbol)
