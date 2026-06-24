from __future__ import annotations

from datetime import date

import polars as pl

from quant_lab.core.events import (
    EventBus,
    EventType,
    FillEvent,
    MarketEvent,
    OrderSide,
    SignalEvent,
)
from quant_lab.portfolio.portfolio import Portfolio
from quant_lab.portfolio.position import Position


def test_position_buy():
    pos = Position(symbol="ETF_A")
    pos.update(100, 3.0, OrderSide.BUY)
    assert pos.quantity == 100
    assert pos.avg_cost == 3.0
    assert pos.is_open


def test_position_buy_and_sell():
    pos = Position(symbol="ETF_A")
    pos.update(100, 3.0, OrderSide.BUY)
    pos.update(100, 3.5, OrderSide.SELL)
    assert pos.quantity == 0
    assert pos.realized_pnl == 50.0  # (3.5 - 3.0) * 100
    assert not pos.is_open


def test_position_multiple_buys():
    pos = Position(symbol="ETF_A")
    pos.update(100, 3.0, OrderSide.BUY)
    pos.update(100, 4.0, OrderSide.BUY)
    assert pos.quantity == 200
    assert pos.avg_cost == 3.5


def test_portfolio_on_market_tracks_equity(event_bus):
    portfolio = Portfolio(event_bus=event_bus, initial_cash=100_000.0)
    bars = pl.DataFrame(
        {"ts_code": ["ETF_A"], "trade_date": [date(2023, 1, 2)], "close": [3.0]}
    )
    event = MarketEvent(timestamp=date(2023, 1, 2), bars=bars)
    portfolio.on_market(event)

    curve = portfolio.get_equity_curve()
    assert curve.height == 1
    assert curve["equity"][0] == 100_000.0


def test_portfolio_on_fill_updates_cash(event_bus):
    portfolio = Portfolio(event_bus=event_bus, initial_cash=100_000.0, commission_rate=0.001)
    fill = FillEvent(
        timestamp=date(2023, 1, 2),
        symbol="ETF_A",
        side=OrderSide.BUY,
        quantity=1000,
        fill_price=3.0,
        commission=3.0,
        slippage_cost=0.5,
    )
    portfolio.on_fill(fill)

    assert portfolio.cash == 100_000.0 - 3000.0 - 3.0
    assert portfolio.positions["ETF_A"].quantity == 1000
    assert portfolio.positions["ETF_A"].avg_cost == 3.0


def test_portfolio_signal_generates_order(event_bus):
    orders = []
    event_bus.subscribe(EventType.ORDER, lambda e: orders.append(e))

    portfolio = Portfolio(event_bus=event_bus, initial_cash=100_000.0, max_position_pct=0.25)
    # Set a price first
    portfolio._current_prices["ETF_A"] = 3.0

    signal = SignalEvent(timestamp=date(2023, 1, 2), symbol="ETF_A", direction=1.0)
    portfolio.on_signal(signal)
    event_bus.drain()

    assert len(orders) == 1
    assert orders[0].symbol == "ETF_A"
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity > 0
