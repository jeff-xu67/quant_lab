from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from quant_lab.core.engine import BacktestEngine
from quant_lab.core.events import EventBus
from quant_lab.data.feed import DataFeed
from quant_lab.db import QuantLabDB
from quant_lab.execution.broker import SimulatedBroker
from quant_lab.execution.slippage import FixedSlippage, VolumeSlippage
from quant_lab.experiment.config import ExperimentConfig
from quant_lab.metrics.calculator import BacktestMetrics, compute_metrics
from quant_lab.portfolio.portfolio import Portfolio
from quant_lab.strategy import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a single backtest run."""

    def __init__(
        self,
        config: ExperimentConfig,
        db_path: str = "data/quant_lab.duckdb",
        results_dir: str = "results",
    ) -> None:
        self.config = config
        self.db_path = db_path
        self.results_dir = Path(results_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]

    def run(self) -> dict[str, Any]:
        db = QuantLabDB(self.db_path)
        try:
            event_bus = EventBus()

            data_feed = DataFeed(
                db=db,
                symbols=self.config.universe.symbols,
                start_date=self.config.backtest.start_date,
                end_date=self.config.backtest.end_date,
            )

            strategy_name = self.config.strategy.name
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(
                    f"Unknown strategy: {strategy_name}. "
                    f"Available: {list(STRATEGY_REGISTRY.keys())}"
                )
            strategy_cls = STRATEGY_REGISTRY[strategy_name]
            strategy = strategy_cls(event_bus=event_bus, params=self.config.strategy.params)

            portfolio = Portfolio(
                event_bus=event_bus,
                initial_cash=self.config.backtest.initial_cash,
                position_sizing=self.config.portfolio.position_sizing,
                max_position_pct=self.config.portfolio.max_position_pct,
                commission_rate=self.config.execution.commission_rate,
            )

            slippage_model = self._build_slippage_model()
            broker = SimulatedBroker(
                event_bus=event_bus,
                slippage_model=slippage_model,
                commission_rate=self.config.execution.commission_rate,
                fill_on_next_open=self.config.execution.fill_on_next_open,
            )

            self._ohlc_data = data_feed._data

            engine = BacktestEngine(event_bus, data_feed, strategy, portfolio, broker)
            engine.run()

            equity_curve = portfolio.get_equity_curve()
            trade_log = portfolio.get_trade_log()

            rf = self.config.metrics.get("risk_free_rate", 0.02)
            tdays = self.config.metrics.get("trading_days_per_year", 242)
            metrics = compute_metrics(equity_curve, trade_log, rf, tdays)

            self._save_results(equity_curve, trade_log, metrics)
            logger.info(f"\n{metrics.summary()}")

            return metrics.to_dict()
        finally:
            db.close()

    def _save_results(
        self,
        equity_curve: pl.DataFrame,
        trade_log: pl.DataFrame,
        metrics: BacktestMetrics,
    ) -> None:
        run_dir = self.results_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.config.to_yaml(run_dir / "config.yaml")
        equity_curve.write_parquet(run_dir / "equity_curve.parquet")
        if not trade_log.is_empty():
            trade_log.write_parquet(run_dir / "trades.parquet")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Results saved to {run_dir}")

    def _build_slippage_model(self):
        model = self.config.execution.slippage_model
        if model == "fixed":
            return FixedSlippage(bps=self.config.execution.slippage_bps)
        elif model == "volume":
            return VolumeSlippage()
        raise ValueError(f"Unknown slippage model: {model}")
