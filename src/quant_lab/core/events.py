from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any, Callable

import polars as pl


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    timestamp: date
    bars: pl.DataFrame

    @property
    def event_type(self) -> EventType:
        return EventType.MARKET


@dataclass(frozen=True, slots=True)
class SignalEvent:
    timestamp: date
    symbol: str
    direction: float  # +1.0 long, -1.0 short, 0.0 exit
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL


@dataclass(frozen=True, slots=True)
class OrderEvent:
    timestamp: date
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None = None

    @property
    def event_type(self) -> EventType:
        return EventType.ORDER


@dataclass(frozen=True, slots=True)
class FillEvent:
    timestamp: date
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage_cost: float

    @property
    def event_type(self) -> EventType:
        return EventType.FILL


Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
EventHandler = Callable[[Event], None]


class EventBus:
    """Central event dispatch with FIFO queue and type-based subscriptions."""

    def __init__(self) -> None:
        self._queue: deque[Event] = deque()
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        self._queue.append(event)

    def process_next(self) -> bool:
        if not self._queue:
            return False
        event = self._queue.popleft()
        for handler in self._handlers[event.event_type]:
            handler(event)
        return True

    def drain(self) -> None:
        while self.process_next():
            pass

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0
