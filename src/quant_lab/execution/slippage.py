from __future__ import annotations

from abc import ABC, abstractmethod

from quant_lab.core.events import OrderSide


class SlippageModel(ABC):
    @abstractmethod
    def calculate(self, price: float, quantity: float, side: OrderSide) -> float:
        """Return slippage-adjusted fill price."""
        ...


class FixedSlippage(SlippageModel):
    """Fixed basis points slippage."""

    def __init__(self, bps: float = 5.0) -> None:
        self.bps = bps

    def calculate(self, price: float, quantity: float, side: OrderSide) -> float:
        factor = 1 + self.bps / 10000 if side == OrderSide.BUY else 1 - self.bps / 10000
        return price * factor


class VolumeSlippage(SlippageModel):
    """Slippage proportional to trade size vs average volume."""

    def __init__(self, impact_factor: float = 0.1, avg_volume: float = 1e6) -> None:
        self.impact_factor = impact_factor
        self.avg_volume = avg_volume

    def calculate(self, price: float, quantity: float, side: OrderSide) -> float:
        volume_ratio = quantity / self.avg_volume
        impact_bps = self.impact_factor * volume_ratio * 10000
        factor = 1 + impact_bps / 10000 if side == OrderSide.BUY else 1 - impact_bps / 10000
        return price * factor
