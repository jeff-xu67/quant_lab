from __future__ import annotations

from datetime import date

import polars as pl

from quant_lab.core.events import (
    EventBus,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
)


def test_event_bus_subscribe_and_publish(event_bus):
    received = []

    def handler(event):
        received.append(event)

    event_bus.subscribe(EventType.SIGNAL, handler)
    signal = SignalEvent(timestamp=date(2023, 1, 1), symbol="ETF_A", direction=1.0)
    event_bus.publish(signal)
    event_bus.drain()

    assert len(received) == 1
    assert received[0] is signal


def test_event_bus_only_routes_to_correct_handlers(event_bus):
    signals = []
    fills = []

    event_bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
    event_bus.subscribe(EventType.FILL, lambda e: fills.append(e))

    event_bus.publish(SignalEvent(timestamp=date(2023, 1, 1), symbol="A", direction=1.0))
    event_bus.publish(
        FillEvent(
            timestamp=date(2023, 1, 1),
            symbol="A",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=3.0,
            commission=0.09,
            slippage_cost=0.01,
        )
    )
    event_bus.drain()

    assert len(signals) == 1
    assert len(fills) == 1


def test_event_bus_fifo_ordering(event_bus):
    order = []

    event_bus.subscribe(EventType.SIGNAL, lambda e: order.append(e.symbol))
    event_bus.publish(SignalEvent(timestamp=date(2023, 1, 1), symbol="first", direction=1.0))
    event_bus.publish(SignalEvent(timestamp=date(2023, 1, 1), symbol="second", direction=1.0))
    event_bus.drain()

    assert order == ["first", "second"]


def test_market_event_type():
    bars = pl.DataFrame({"ts_code": ["A"], "close": [3.0]})
    event = MarketEvent(timestamp=date(2023, 1, 1), bars=bars)
    assert event.event_type == EventType.MARKET


def test_order_event_type():
    event = OrderEvent(
        timestamp=date(2023, 1, 1),
        symbol="A",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    assert event.event_type == EventType.ORDER
