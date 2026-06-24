"""CLI entry point for running backtests."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_lab.experiment.config import ExperimentConfig
from quant_lab.experiment.logger import setup_logging
from quant_lab.experiment.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest experiment")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--db-path", default="data/quant_lab.duckdb")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results")
    parser.add_argument("--show-report", action="store_true", help="Open report in browser")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(**config.logging)

    runner = ExperimentRunner(config, db_path=args.db_path, results_dir=args.results_dir)
    metrics = runner.run()

    if args.show_report:
        import polars as pl

        from quant_lab.metrics.calculator import BacktestMetrics
        from quant_lab.metrics.visualization import (
            create_report,
            generate_html_report,
            plot_kline_page,
        )

        run_dir = Path(args.results_dir) / runner.run_id
        equity_curve = pl.read_parquet(run_dir / "equity_curve.parquet")
        trades_file = run_dir / "trades.parquet"
        trade_log = pl.read_parquet(trades_file) if trades_file.exists() else pl.DataFrame()

        bm = BacktestMetrics(**{k: v for k, v in metrics.items()})

        # Build benchmark (CSI 300) normalized price series
        benchmark_symbol = config.backtest.benchmark
        benchmark_df = None
        if benchmark_symbol and benchmark_symbol in runner._ohlc_data["ts_code"].unique().to_list():
            bench_raw = (
                runner._ohlc_data
                .filter(pl.col("ts_code") == benchmark_symbol)
                .sort("trade_date")
                .select(
                    pl.col("trade_date").alias("date"),
                    pl.col("close"),
                )
            )
            if not bench_raw.is_empty():
                benchmark_df = bench_raw

        fig_report = create_report(
            equity_curve, trade_log, bm,
            benchmark=benchmark_df,
            title=f"Backtest: {config.strategy.name}",
        )
        fig_kline = plot_kline_page(
            ohlc_data=runner._ohlc_data,
            trade_log=trade_log,
            symbols=config.universe.symbols,
            ma_periods=(5, 10, 20, 60),
        )

        html_path = str(run_dir / "report.html")
        generate_html_report(
            fig_report,
            fig_kline,
            title=f"Backtest: {config.strategy.name}",
            output_path=html_path,
        )
        webbrowser.open(f"file://{Path(html_path).resolve()}")
        print(f"Report: {html_path}")


if __name__ == "__main__":
    main()
