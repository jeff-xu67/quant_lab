from __future__ import annotations

import logging

from quant_lab.core.events import EventBus, EventType
from quant_lab.data.feed import DataFeed
from quant_lab.execution.broker import SimulatedBroker
from quant_lab.portfolio.portfolio import Portfolio
from quant_lab.strategy.base import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main event loop: drives the simulation forward bar-by-bar."""

    def __init__(
        self,
        event_bus: EventBus,
        data_feed: DataFeed,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: SimulatedBroker,
    ) -> None:
        self.event_bus = event_bus
        self.data_feed = data_feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = broker
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        self.event_bus.subscribe(EventType.MARKET, self.broker.on_market)
        self.event_bus.subscribe(EventType.MARKET, self.portfolio.on_market)
        self.event_bus.subscribe(EventType.MARKET, self.strategy.on_market)
        self.event_bus.subscribe(EventType.SIGNAL, self.portfolio.on_signal)
        self.event_bus.subscribe(EventType.ORDER, self.broker.on_order)
        self.event_bus.subscribe(EventType.FILL, self.portfolio.on_fill)

    def run(self) -> None:
        logger.info(
            f"Backtest starting: {self.data_feed.total_bars} bars, "
            f"strategy={self.strategy.name}"
        )
        bar_count = 0
        while self.data_feed.has_next():
            market_event = self.data_feed.next()
            self.event_bus.publish(market_event)
            self.event_bus.drain()
            bar_count += 1

        logger.info(
            f"Backtest complete. {bar_count} bars processed. "
            f"Final equity: {self.portfolio.total_equity:,.2f}"
        )
