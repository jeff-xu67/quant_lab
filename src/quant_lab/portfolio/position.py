from __future__ import annotations

from dataclasses import dataclass

from quant_lab.core.events import OrderSide


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def update(self, fill_qty: float, fill_price: float, side: OrderSide) -> None:
        if side == OrderSide.BUY:
            total_cost = self.avg_cost * self.quantity + fill_price * fill_qty
            self.quantity += fill_qty
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0.0
        else:
            sell_qty = min(fill_qty, self.quantity)
            self.realized_pnl += (fill_price - self.avg_cost) * sell_qty
            self.quantity -= sell_qty
            if self.quantity < 1e-9:
                self.quantity = 0.0
                self.avg_cost = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-9
