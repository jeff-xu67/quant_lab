from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_lab.metrics.calculator import compute_metrics


def test_compute_metrics_positive_return():
    # 10 days, steady growth
    dates = [date(2023, 1, i + 2) for i in range(10)]
    equity = [100_000 * (1.001**i) for i in range(10)]

    equity_curve = pl.DataFrame({"date": dates, "equity": equity})
    trade_log = pl.DataFrame(
        schema={
            "timestamp": pl.Date,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "commission": pl.Float64,
            "slippage_cost": pl.Float64,
        }
    )

    metrics = compute_metrics(equity_curve, trade_log)
    assert metrics.total_return > 0
    assert metrics.annualized_return > 0
    assert metrics.max_drawdown <= 0 or abs(metrics.max_drawdown) < 1e-10
    assert metrics.volatility >= 0


def test_compute_metrics_with_drawdown():
    # Equity goes up then down
    equity = [100_000, 110_000, 120_000, 100_000, 105_000]
    dates = [date(2023, 1, i + 2) for i in range(5)]

    equity_curve = pl.DataFrame({"date": dates, "equity": equity})
    trade_log = pl.DataFrame(
        schema={
            "timestamp": pl.Date,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "commission": pl.Float64,
            "slippage_cost": pl.Float64,
        }
    )

    metrics = compute_metrics(equity_curve, trade_log)
    # Max drawdown: (100000 - 120000) / 120000 = -16.67%
    assert metrics.max_drawdown < -0.15
    assert abs(metrics.total_return - 0.05) < 1e-10


def test_compute_metrics_trade_stats():
    dates = [date(2023, 1, i + 2) for i in range(5)]
    equity = [100_000.0, 100_100.0, 100_200.0, 100_300.0, 100_400.0]
    equity_curve = pl.DataFrame({"date": dates, "equity": equity})

    trade_log = pl.DataFrame(
        {
            "timestamp": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 3), date(2023, 1, 4)],
            "symbol": ["ETF_A", "ETF_A", "ETF_B", "ETF_B"],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "quantity": [100.0, 100.0, 200.0, 200.0],
            "price": [3.0, 3.1, 5.0, 4.9],
            "commission": [0.09, 0.09, 0.30, 0.30],
            "slippage_cost": [0.01, 0.01, 0.02, 0.02],
        }
    )

    metrics = compute_metrics(equity_curve, trade_log)
    assert metrics.trade_count == 2
    assert metrics.win_rate == 0.5  # 1 win, 1 loss


def test_empty_equity_curve():
    equity_curve = pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64})
    trade_log = pl.DataFrame(
        schema={
            "timestamp": pl.Date,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "commission": pl.Float64,
            "slippage_cost": pl.Float64,
        }
    )
    metrics = compute_metrics(equity_curve, trade_log)
    assert metrics.total_return == 0.0
    assert metrics.trade_count == 0
