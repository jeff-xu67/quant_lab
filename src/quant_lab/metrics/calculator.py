from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    trade_count: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    volatility: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_return": round(self.total_return, 4),
            "annualized_return": round(self.annualized_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "trade_count": self.trade_count,
            "avg_trade_return": round(self.avg_trade_return, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "volatility": round(self.volatility, 4),
        }

    def summary(self) -> str:
        lines = [
            "=" * 45,
            f"{'Backtest Performance Summary':^45}",
            "=" * 45,
            f"  Total Return:        {self.total_return:>10.2%}",
            f"  Annualized Return:   {self.annualized_return:>10.2%}",
            f"  Max Drawdown:        {self.max_drawdown:>10.2%}",
            f"  Max DD Duration:     {self.max_drawdown_duration:>10d} days",
            f"  Sharpe Ratio:        {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:       {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:        {self.calmar_ratio:>10.3f}",
            f"  Volatility:          {self.volatility:>10.2%}",
            "-" * 45,
            f"  Trade Count:         {self.trade_count:>10d}",
            f"  Win Rate:            {self.win_rate:>10.2%}",
            f"  Profit Factor:       {self.profit_factor:>10.3f}",
            f"  Avg Trade Return:    {self.avg_trade_return:>10.2%}",
            f"  Avg Win:             {self.avg_win:>10.2%}",
            f"  Avg Loss:            {self.avg_loss:>10.2%}",
            "=" * 45,
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: pl.DataFrame,
    trade_log: pl.DataFrame,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 242,
) -> BacktestMetrics:
    if equity_curve.is_empty() or equity_curve.height < 2:
        return _empty_metrics()

    equity = equity_curve["equity"].to_numpy()
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0]) - 1.0
    n_days = len(equity) - 1
    n_years = n_days / trading_days_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax
    max_drawdown = float(np.min(drawdowns))

    # Max drawdown duration
    dd_duration = _max_drawdown_duration(equity)

    # Volatility
    volatility = float(np.std(returns, ddof=1) * np.sqrt(trading_days_per_year))

    # Sharpe
    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = returns - daily_rf
    sharpe = (
        float(np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(trading_days_per_year))
        if np.std(excess_returns, ddof=1) > 0
        else 0.0
    )

    # Sortino
    downside = returns[returns < daily_rf] - daily_rf
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-10
    sortino = (
        float((np.mean(returns) - daily_rf) / downside_std * np.sqrt(trading_days_per_year))
        if downside_std > 1e-10
        else 0.0
    )

    # Calmar
    calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    # Trade metrics
    trade_count, win_rate, profit_factor, avg_trade_return, avg_win, avg_loss = _trade_metrics(
        trade_log
    )

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        max_drawdown_duration=dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_count=trade_count,
        avg_trade_return=avg_trade_return,
        avg_win=avg_win,
        avg_loss=avg_loss,
        volatility=volatility,
    )


def _max_drawdown_duration(equity: np.ndarray) -> int:
    cummax = np.maximum.accumulate(equity)
    in_drawdown = equity < cummax
    max_dur = 0
    current_dur = 0
    for dd in in_drawdown:
        if dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def _trade_metrics(
    trade_log: pl.DataFrame,
) -> tuple[int, float, float, float, float, float]:
    if trade_log.is_empty():
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Pair trades: match buys and sells for the same symbol
    buys = trade_log.filter(pl.col("side") == "BUY").sort("timestamp")
    sells = trade_log.filter(pl.col("side") == "SELL").sort("timestamp")

    if sells.is_empty():
        return buys.height, 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute round-trip returns per symbol
    trade_returns: list[float] = []
    for symbol in trade_log["symbol"].unique().to_list():
        sym_buys = buys.filter(pl.col("symbol") == symbol)
        sym_sells = sells.filter(pl.col("symbol") == symbol)

        n_pairs = min(sym_buys.height, sym_sells.height)
        for i in range(n_pairs):
            buy_price = sym_buys["price"][i]
            sell_price = sym_sells["price"][i]
            ret = (sell_price - buy_price) / buy_price
            trade_returns.append(ret)

    if not trade_returns:
        return trade_log.height, 0.0, 0.0, 0.0, 0.0, 0.0

    arr = np.array(trade_returns)
    trade_count = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    win_rate = len(wins) / trade_count if trade_count > 0 else 0.0
    avg_trade_return = float(np.mean(arr))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    return trade_count, win_rate, profit_factor, avg_trade_return, avg_win, avg_loss


def _empty_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        max_drawdown_duration=0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        trade_count=0,
        avg_trade_return=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        volatility=0.0,
    )
