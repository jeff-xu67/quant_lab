from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import plotly.graph_objects as go


class RunComparison:
    """Load and compare multiple backtest runs."""

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)

    def list_runs(self) -> pl.DataFrame:
        runs = []
        if not self.results_dir.exists():
            return pl.DataFrame(schema={"run_id": pl.Utf8, "strategy": pl.Utf8})

        for run_dir in sorted(self.results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                metrics = json.load(f)
            metrics["run_id"] = run_dir.name
            runs.append(metrics)

        if not runs:
            return pl.DataFrame(schema={"run_id": pl.Utf8})
        return pl.DataFrame(runs)

    def compare(self, run_ids: list[str]) -> pl.DataFrame:
        rows = []
        for run_id in run_ids:
            metrics_file = self.results_dir / run_id / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                data["run_id"] = run_id
                rows.append(data)
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"run_id": pl.Utf8})

    def plot_equity_comparison(self, run_ids: list[str]) -> go.Figure:
        fig = go.Figure()
        for run_id in run_ids:
            eq_file = self.results_dir / run_id / "equity_curve.parquet"
            if not eq_file.exists():
                continue
            df = pl.read_parquet(eq_file)
            equity = df["equity"].to_list()
            base = equity[0] if equity else 1.0
            fig.add_trace(
                go.Scatter(
                    x=df["date"].to_list(),
                    y=[e / base for e in equity],
                    mode="lines",
                    name=run_id[:20],
                )
            )

        fig.update_layout(
            title="Equity Curve Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig
