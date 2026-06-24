from __future__ import annotations

import logging

import polars as pl

from quant_lab.core.events import (
    Event,
    EventBus,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)
from quant_lab.execution.slippage import SlippageModel

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """Simulates order execution with slippage and commission.

    Supports two fill modes:
    - fill_on_next_open=True: orders queue and fill at next bar's open (realistic)
    - fill_on_next_open=False: orders fill immediately at current bar's close
    """

    def __init__(
        self,
        event_bus: EventBus,
        slippage_model: SlippageModel,
        commission_rate: float = 0.0003,
        fill_on_next_open: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self.slippage_model = slippage_model
        self.commission_rate = commission_rate
        self.fill_on_next_open = fill_on_next_open

        self._current_bars: pl.DataFrame | None = None
        self._pending_orders: list[OrderEvent] = []

    def on_market(self, event: Event) -> None:
        """Store current bars and fill pending orders from previous bar."""
        assert isinstance(event, MarketEvent)
        self._current_bars = event.bars

        if self.fill_on_next_open and self._pending_orders:
            orders_to_fill = self._pending_orders[:]
            self._pending_orders.clear()
            for order in orders_to_fill:
                self._fill_order(order, use_open=True)

    def on_order(self, event: Event) -> None:
        assert isinstance(event, OrderEvent)

        if self.fill_on_next_open:
            if event.order_type == OrderType.MARKET:
                self._pending_orders.append(event)
            else:
                self._pending_orders.append(event)
        else:
            self._fill_order(event, use_open=False)

    def _fill_order(self, order: OrderEvent, use_open: bool) -> None:
        if self._current_bars is None:
            logger.warning(f"No market data available for {order.symbol}")
            return

        bar = self._current_bars.filter(pl.col("ts_code") == order.symbol)
        if bar.is_empty():
            logger.warning(f"No bar data for {order.symbol} on current date")
            return

        row = bar.row(0, named=True)

        if order.order_type == OrderType.MARKET:
            base_price = row["open"] if use_open else row["close"]
        else:
            base_price = row["open"] if use_open else row["close"]
            if order.side == OrderSide.BUY and order.limit_price is not None:
                if base_price > order.limit_price:
                    if row["low"] > order.limit_price:
                        self._pending_orders.append(order)
                        return
                    base_price = order.limit_price
            elif order.side == OrderSide.SELL and order.limit_price is not None:
                if base_price < order.limit_price:
                    if row["high"] < order.limit_price:
                        self._pending_orders.append(order)
                        return
                    base_price = order.limit_price

        fill_price = self.slippage_model.calculate(base_price, order.quantity, order.side)
        slippage_cost = abs(fill_price - base_price) * order.quantity
        commission = fill_price * order.quantity * self.commission_rate

        fill = FillEvent(
            timestamp=self._current_bars.row(0, named=True)["trade_date"],
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )
        self.event_bus.publish(fill)
