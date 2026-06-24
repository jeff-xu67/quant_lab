from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_cash: float = 1_000_000.0
    benchmark: str | None = None


@dataclass
class UniverseConfig:
    mode: str = "static"
    symbols: list[str] = field(default_factory=list)


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    commission_rate: float = 0.0003
    slippage_model: str = "fixed"
    slippage_bps: float = 5.0
    fill_on_next_open: bool = True
    min_trade_amount: float = 100.0


@dataclass
class PortfolioConfig:
    position_sizing: str = "equal_weight"
    max_position_pct: float = 0.25
    max_total_exposure: float = 1.0


@dataclass
class ExperimentConfig:
    backtest: BacktestConfig
    universe: UniverseConfig
    strategy: StrategyConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    metrics: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            backtest=BacktestConfig(**raw["backtest"]),
            universe=UniverseConfig(**raw.get("universe", {})),
            strategy=StrategyConfig(**raw["strategy"]),
            execution=ExecutionConfig(**raw.get("execution", {})),
            portfolio=PortfolioConfig(**raw.get("portfolio", {})),
            metrics=raw.get("metrics", {}),
            logging=raw.get("logging", {}),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "backtest": {
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
                "initial_cash": self.backtest.initial_cash,
                "benchmark": self.backtest.benchmark,
            },
            "universe": {
                "mode": self.universe.mode,
                "symbols": self.universe.symbols,
            },
            "strategy": {
                "name": self.strategy.name,
                "params": self.strategy.params,
            },
            "execution": {
                "commission_rate": self.execution.commission_rate,
                "slippage_model": self.execution.slippage_model,
                "slippage_bps": self.execution.slippage_bps,
                "fill_on_next_open": self.execution.fill_on_next_open,
                "min_trade_amount": self.execution.min_trade_amount,
            },
            "portfolio": {
                "position_sizing": self.portfolio.position_sizing,
                "max_position_pct": self.portfolio.max_position_pct,
                "max_total_exposure": self.portfolio.max_total_exposure,
            },
            "metrics": self.metrics,
            "logging": self.logging,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
