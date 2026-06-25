from __future__ import annotations

from datetime import date

import polars as pl

from quant_lab.core.events import MarketEvent
from quant_lab.db import QuantLabDB


class DataFeed:
    """Loads universe data from DuckDB into Polars, iterates bar-by-bar."""

    def __init__(
        self,
        db: QuantLabDB,
        symbols: list[str],
        start_date: str,
        end_date: str,
        adj_price: bool = True,
    ) -> None:
        self.db = db
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

        self._data = db.get_universe_pl(symbols, start_date, end_date)

        if adj_price:
            self._apply_adj_factor()

        self._dates: list[date] = (
            self._data.select("trade_date")
            .unique()
            .sort("trade_date")
            .to_series()
            .to_list()
        )
        self._index = 0

    def _apply_adj_factor(self) -> None:
        self._data = self._data.with_columns(
            (pl.col("open") * pl.col("adj_factor")).alias("open"),
            (pl.col("high") * pl.col("adj_factor")).alias("high"),
            (pl.col("low") * pl.col("adj_factor")).alias("low"),
            (pl.col("close") * pl.col("adj_factor")).alias("close"),
        )

    def has_next(self) -> bool:
        return self._index < len(self._dates)

    def next(self) -> MarketEvent:
        current_date = self._dates[self._index]
        bars = self._data.filter(pl.col("trade_date") == current_date)
        self._index += 1
        return MarketEvent(timestamp=current_date, bars=bars)

    def reset(self) -> None:
        self._index = 0

    @property
    def total_bars(self) -> int:
        return len(self._dates)
