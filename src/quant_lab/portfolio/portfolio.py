from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from quant_lab.core.events import (
    Event,
    EventBus,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
)
from quant_lab.portfolio.position import Position

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    timestamp: date
    cash: float
    positions_value: float
    total_equity: float


class Portfolio:
    """Manages positions, cash, equity and converts signals to orders."""

    def __init__(
        self,
        event_bus: EventBus,
        initial_cash: float = 1_000_000.0,
        position_sizing: str = "equal_weight",
        max_position_pct: float = 0.25,
        commission_rate: float = 0.0003,
    ) -> None:
        self.event_bus = event_bus
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.position_sizing = position_sizing
        self.max_position_pct = max_position_pct
        self.commission_rate = commission_rate

        self._current_prices: dict[str, float] = {}
        self._equity_history: list[PortfolioSnapshot] = []
        self._trade_log: list[dict[str, Any]] = []
        self._pending_signals: list[SignalEvent] = []

    def on_market(self, event: Event) -> None:
        assert isinstance(event, MarketEvent)
        for row in event.bars.iter_rows(named=True):
            self._current_prices[row["ts_code"]] = row["close"]

        self._equity_history.append(
            PortfolioSnapshot(
                timestamp=event.timestamp,
                cash=self.cash,
                positions_value=self._positions_value(),
                total_equity=self.total_equity,
            )
        )

    def on_signal(self, event: Event) -> None:
        assert isinstance(event, SignalEvent)

        if event.direction == 0.0:
            pos = self.positions.get(event.symbol)
            if pos and pos.is_open:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                )
                self.event_bus.publish(order)
            return

        target_value = self._compute_target_value(event)
        current_value = self._current_position_value(event.symbol)
        delta_value = target_value - current_value

        if abs(delta_value) < 100:
            return

        price = self._current_prices.get(event.symbol, 0.0)
        if price <= 0:
            return

        quantity = abs(delta_value) / price
        # ETF最小交易单位100份
        quantity = int(quantity / 100) * 100
        if quantity <= 0:
            return

        side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(quantity),
        )
        self.event_bus.publish(order)

    def on_fill(self, event: Event) -> None:
        assert isinstance(event, FillEvent)

        if event.symbol not in self.positions:
            self.positions[event.symbol] = Position(symbol=event.symbol)

        pos = self.positions[event.symbol]
        pos.update(event.quantity, event.fill_price, event.side)

        cost = event.fill_price * event.quantity
        if event.side == OrderSide.BUY:
            self.cash -= cost + event.commission
        else:
            self.cash += cost - event.commission

        self._trade_log.append(
            {
                "timestamp": event.timestamp,
                "symbol": event.symbol,
                "side": event.side.value,
                "quantity": event.quantity,
                "price": event.fill_price,
                "commission": event.commission,
                "slippage_cost": event.slippage_cost,
            }
        )
        logger.debug(
            f"FILL {event.side.value} {event.symbol} qty={event.quantity:.0f} "
            f"price={event.fill_price:.4f} comm={event.commission:.2f}"
        )

    @property
    def total_equity(self) -> float:
        return self.cash + self._positions_value()

    def _positions_value(self) -> float:
        return sum(
            pos.quantity * self._current_prices.get(sym, 0.0)
            for sym, pos in self.positions.items()
            if pos.is_open
        )

    def _current_position_value(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        if not pos or not pos.is_open:
            return 0.0
        return pos.quantity * self._current_prices.get(symbol, 0.0)

    def _compute_target_value(self, signal: SignalEvent) -> float:
        equity = self.total_equity
        if self.position_sizing == "equal_weight":
            target = equity * self.max_position_pct
        elif self.position_sizing == "signal_weight":
            target = equity * self.max_position_pct * abs(signal.strength)
        else:  # fixed_fraction
            target = equity * self.max_position_pct
        return target * signal.direction

    def get_equity_curve(self) -> pl.DataFrame:
        if not self._equity_history:
            return pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64})
        return pl.DataFrame(
            {
                "date": [s.timestamp for s in self._equity_history],
                "equity": [float(s.total_equity) for s in self._equity_history],
                "cash": [float(s.cash) for s in self._equity_history],
                "positions_value": [float(s.positions_value) for s in self._equity_history],
            }
        )

    def get_trade_log(self) -> pl.DataFrame:
        if not self._trade_log:
            return pl.DataFrame(
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
        return pl.DataFrame(self._trade_log)
