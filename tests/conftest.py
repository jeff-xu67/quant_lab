from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_lab.core.events import EventBus


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def sample_bars():
    """Sample market data for 3 symbols over 5 days."""
    symbols = ["ETF_A", "ETF_B", "ETF_C"]
    dates = [date(2023, 1, 2 + i) for i in range(5)]
    rows = []
    base_prices = {"ETF_A": 3.0, "ETF_B": 5.0, "ETF_C": 1.5}

    for i, d in enumerate(dates):
        for sym in symbols:
            bp = base_prices[sym]
            rows.append(
                {
                    "ts_code": sym,
                    "trade_date": d,
                    "open": bp + i * 0.01,
                    "high": bp + i * 0.02 + 0.05,
                    "low": bp + i * 0.01 - 0.03,
                    "close": bp + i * 0.015,
                    "vol": 1000000.0,
                    "amount": 3000000.0,
                    "adj_factor": 1.0,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def sample_bars_by_date(sample_bars):
    """Return dict of date -> bars DataFrame."""
    result = {}
    for d in sample_bars["trade_date"].unique().sort().to_list():
        result[d] = sample_bars.filter(pl.col("trade_date") == d)
    return result
