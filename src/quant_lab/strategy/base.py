from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from quant_lab.core.events import Event, EventBus

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Decorator to register a strategy class by its class name."""
    STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


class Strategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, event_bus: EventBus, params: dict[str, Any]) -> None:
        self.event_bus = event_bus
        self.params = params
        self._history: list[pl.DataFrame] = []
        self._bar_count = 0

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def on_market(self, event: Event) -> None: ...

    def warmup_period(self) -> int:
        return 0

    def _update_history(self, bars: pl.DataFrame, max_window: int) -> None:
        self._history.append(bars)
        if len(self._history) > max_window:
            self._history = self._history[-max_window:]

    def get_history_df(self) -> pl.DataFrame:
        if not self._history:
            return pl.DataFrame()
        return pl.concat(self._history)
