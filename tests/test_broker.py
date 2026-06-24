from __future__ import annotations

from datetime import date

import polars as pl

from quant_lab.core.events import (
    EventBus,
    EventType,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)
from quant_lab.execution.broker import SimulatedBroker
from quant_lab.execution.slippage import FixedSlippage


def test_broker_fill_on_close():
    event_bus = EventBus()
    fills = []
    event_bus.subscribe(EventType.FILL, lambda e: fills.append(e))

    broker = SimulatedBroker(
        event_bus=event_bus,
        slippage_model=FixedSlippage(bps=0),
        commission_rate=0.001,
        fill_on_next_open=False,
    )

    bars = pl.DataFrame(
        {
            "ts_code": ["ETF_A"],
            "trade_date": [date(2023, 1, 2)],
            "open": [3.0],
            "high": [3.1],
            "low": [2.9],
            "close": [3.05],
        }
    )
    market = MarketEvent(timestamp=date(2023, 1, 2), bars=bars)
    broker.on_market(market)

    order = OrderEvent(
        timestamp=date(2023, 1, 2),
        symbol="ETF_A",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    broker.on_order(order)
    event_bus.drain()

    assert len(fills) == 1
    assert fills[0].fill_price == 3.05
    assert fills[0].quantity == 100


def test_broker_fill_on_next_open():
    event_bus = EventBus()
    fills = []
    event_bus.subscribe(EventType.FILL, lambda e: fills.append(e))

    broker = SimulatedBroker(
        event_bus=event_bus,
        slippage_model=FixedSlippage(bps=0),
        commission_rate=0.001,
        fill_on_next_open=True,
    )

    # Day 1: place order
    bars1 = pl.DataFrame(
        {
            "ts_code": ["ETF_A"],
            "trade_date": [date(2023, 1, 2)],
            "open": [3.0],
            "high": [3.1],
            "low": [2.9],
            "close": [3.05],
        }
    )
    broker.on_market(MarketEvent(timestamp=date(2023, 1, 2), bars=bars1))

    order = OrderEvent(
        timestamp=date(2023, 1, 2),
        symbol="ETF_A",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    broker.on_order(order)
    event_bus.drain()
    assert len(fills) == 0  # Not filled yet

    # Day 2: fills at open
    bars2 = pl.DataFrame(
        {
            "ts_code": ["ETF_A"],
            "trade_date": [date(2023, 1, 3)],
            "open": [3.08],
            "high": [3.2],
            "low": [3.0],
            "close": [3.15],
        }
    )
    broker.on_market(MarketEvent(timestamp=date(2023, 1, 3), bars=bars2))
    event_bus.drain()

    assert len(fills) == 1
    assert fills[0].fill_price == 3.08


def test_fixed_slippage():
    slippage = FixedSlippage(bps=10)
    # Buy: price goes up
    buy_price = slippage.calculate(100.0, 100, OrderSide.BUY)
    assert buy_price > 100.0
    assert abs(buy_price - 100.1) < 1e-10

    # Sell: price goes down
    sell_price = slippage.calculate(100.0, 100, OrderSide.SELL)
    assert sell_price < 100.0
    assert abs(sell_price - 99.9) < 1e-10
