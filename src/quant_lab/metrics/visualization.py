from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant_lab.metrics.calculator import BacktestMetrics

MA_COLORS = {
    5: "#FF6B6B",
    10: "#FFA500",
    20: "#4ECDC4",
    30: "#45B7D1",
    60: "#9B59B6",
    120: "#2C3E50",
}


def plot_equity_curve(
    equity_curve: pl.DataFrame,
    benchmark: pl.DataFrame | None = None,
    title: str = "Equity Curve",
) -> go.Figure:
    fig = go.Figure()
    dates = equity_curve["date"].to_list()
    equity = equity_curve["equity"].to_list()

    # Normalize to 1.0
    base = equity[0] if equity else 1.0
    normalized = [e / base for e in equity]

    fig.add_trace(
        go.Scatter(x=dates, y=normalized, mode="lines", name="Strategy", line=dict(width=2))
    )

    if benchmark is not None and not benchmark.is_empty():
        bench_dates = benchmark["date"].to_list()
        bench_eq = benchmark["equity"].to_list()
        bench_base = bench_eq[0] if bench_eq else 1.0
        bench_norm = [e / bench_base for e in bench_eq]
        fig.add_trace(
            go.Scatter(
                x=bench_dates,
                y=bench_norm,
                mode="lines",
                name="Benchmark",
                line=dict(width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_drawdown(equity_curve: pl.DataFrame) -> go.Figure:
    equity = equity_curve["equity"].to_numpy()
    dates = equity_curve["date"].to_list()
    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="red", width=1),
            fillcolor="rgba(255,0,0,0.2)",
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
    )
    return fig


def plot_monthly_returns(equity_curve: pl.DataFrame) -> go.Figure:
    df = equity_curve.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )

    monthly = (
        df.group_by(["year", "month"])
        .agg(
            pl.col("equity").first().alias("start_eq"),
            pl.col("equity").last().alias("end_eq"),
        )
        .with_columns(
            ((pl.col("end_eq") / pl.col("start_eq")) - 1.0).alias("return")
        )
        .sort(["year", "month"])
    )

    years = sorted(monthly["year"].unique().to_list())
    months = list(range(1, 13))
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    z_data = []
    for year in years:
        row = []
        for month in months:
            val = monthly.filter((pl.col("year") == year) & (pl.col("month") == month))
            if val.is_empty():
                row.append(None)
            else:
                row.append(round(val["return"][0] * 100, 2))
        z_data.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=month_names,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z_data],
            texttemplate="%{text}",
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_white",
    )
    return fig


def plot_trade_analysis(trade_log: pl.DataFrame) -> go.Figure:
    if trade_log.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    sells = trade_log.filter(pl.col("side") == "SELL")
    if sells.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No closed trades", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trade P&L Distribution", "Commission Cost"))

    # Approximate P&L from sells (price * quantity as proxy)
    amounts = (sells["price"] * sells["quantity"]).to_list()
    fig.add_trace(
        go.Histogram(x=amounts, nbinsx=30, name="Trade Amounts"),
        row=1, col=1,
    )

    commissions = sells["commission"].to_list()
    fig.add_trace(
        go.Histogram(x=commissions, nbinsx=20, name="Commissions"),
        row=1, col=2,
    )

    fig.update_layout(template="plotly_white", title="Trade Analysis", showlegend=False)
    return fig


def create_report(
    equity_curve: pl.DataFrame,
    trade_log: pl.DataFrame,
    metrics: BacktestMetrics,
    benchmark: pl.DataFrame | None = None,
    title: str = "Backtest Report",
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Equity Curve",
            "Trade Analysis",
            "Drawdown",
            "Monthly Returns",
            "Metrics Summary",
            "Rolling Sharpe (60d)",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "table"}, {"type": "xy"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    # Panel 1: Equity curve with benchmark
    dates = equity_curve["date"].to_list()
    equity = equity_curve["equity"].to_list()
    base = equity[0] if equity else 1.0
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[e / base for e in equity],
            mode="lines",
            name="Strategy",
            line=dict(color="#1a73e8", width=2),
        ),
        row=1, col=1,
    )

    if benchmark is not None and not benchmark.is_empty():
        bench_dates = benchmark["date"].to_list()
        bench_close = benchmark["close"].to_list()
        bench_base = bench_close[0] if bench_close else 1.0
        fig.add_trace(
            go.Scatter(
                x=bench_dates,
                y=[c / bench_base for c in bench_close],
                mode="lines",
                name="CSI 300",
                line=dict(color="#999", width=1.5, dash="dash"),
            ),
            row=1, col=1,
        )

    # Panel 2: Trade histogram (swapped from old position)
    if not trade_log.is_empty():
        sells = trade_log.filter(pl.col("side") == "SELL")
        if not sells.is_empty():
            amounts = (sells["price"] * sells["quantity"]).to_list()
            fig.add_trace(
                go.Histogram(
                    x=amounts, nbinsx=20, name="Trades",
                    marker_color="rgba(26,115,232,0.6)",
                    marker_line=dict(color="rgba(26,115,232,0.9)", width=1),
                ),
                row=1, col=2,
            )

    # Panel 3: Drawdown
    eq_arr = np.array(equity)
    cummax = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - cummax) / cummax * 100
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd.tolist(),
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="#E8443A", width=1),
            fillcolor="rgba(232,68,58,0.12)",
        ),
        row=2, col=1,
    )

    # Panel 4: Monthly returns heatmap
    df_monthly = equity_curve.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    monthly = (
        df_monthly.group_by(["year", "month"])
        .agg(
            pl.col("equity").first().alias("start_eq"),
            pl.col("equity").last().alias("end_eq"),
        )
        .with_columns(((pl.col("end_eq") / pl.col("start_eq")) - 1.0).alias("return"))
        .sort(["year", "month"])
    )
    years = sorted(monthly["year"].unique().to_list())
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = []
    for year in years:
        row = []
        for m in range(1, 13):
            val = monthly.filter((pl.col("year") == year) & (pl.col("month") == m))
            row.append(val["return"][0] * 100 if not val.is_empty() else None)
        z.append(row)
    fig.add_trace(
        go.Heatmap(z=z, x=months_names, y=[str(y) for y in years],
                   colorscale="RdYlGn", zmid=0, showscale=False),
        row=2, col=2,
    )

    # Panel 5: Metrics table (swapped to bottom-left, beautified)
    metrics_dict = metrics.to_dict()
    pct_keys = {
        "total_return", "annualized_return", "max_drawdown",
        "win_rate", "volatility", "avg_trade_return", "avg_win", "avg_loss",
    }
    metric_labels = []
    metric_values = []
    for k, v in metrics_dict.items():
        label = k.replace("_", " ").title()
        metric_labels.append(label)
        if isinstance(v, float):
            if k in pct_keys:
                metric_values.append(f"{v * 100:.2f}%")
            else:
                metric_values.append(f"{v:.3f}")
        else:
            metric_values.append(str(v))

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#1a73e8",
                font=dict(color="white", size=12, family="Arial"),
                align=["left", "right"],
                height=28,
                line=dict(color="#1565c0", width=1),
            ),
            cells=dict(
                values=[metric_labels, metric_values],
                fill_color=[["#f8f9fa" if i % 2 == 0 else "#ffffff" for i in range(len(metric_labels))]],
                font=dict(color="#333", size=11, family="Consolas, monospace"),
                align=["left", "right"],
                height=24,
                line=dict(color="#e8e8e8", width=1),
            ),
            columnwidth=[0.6, 0.4],
        ),
        row=3, col=1,
    )

    # Panel 6: Rolling Sharpe
    if len(equity) > 60:
        returns = np.diff(eq_arr) / eq_arr[:-1]
        window = 60
        rolling_sharpe = []
        for i in range(window, len(returns)):
            chunk = returns[i - window : i]
            s = float(np.mean(chunk) / np.std(chunk, ddof=1) * np.sqrt(242)) if np.std(chunk, ddof=1) > 0 else 0
            rolling_sharpe.append(s)
        fig.add_trace(
            go.Scatter(
                x=dates[window + 1:], y=rolling_sharpe, mode="lines",
                name="Rolling Sharpe",
                line=dict(color="#9B59B6", width=1.5),
            ),
            row=3, col=2,
        )
        fig.add_shape(
            type="line", y0=0, y1=0, x0=0, x1=1,
            xref="x5 domain", yref="y5",
            line=dict(color="#bbb", width=1, dash="dot"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#333")),
        template="plotly_white",
        height=1050,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom",
            font=dict(size=11),
        ),
        margin=dict(t=70, b=30, l=50, r=30),
    )
    return fig


def plot_kline_page(
    ohlc_data: pl.DataFrame,
    trade_log: pl.DataFrame,
    symbols: list[str],
    ma_periods: Sequence[int] = (5, 10, 20, 60),
) -> go.Figure:
    """Professional K-line chart with timeframe switching and trade signal arrows.

    Features:
    - Daily / Weekly / Monthly K-line switching via buttons
    - Symbol switching via dropdown
    - MA lines toggled by clicking legend
    - Buy/Sell arrows with tail for visibility
    """
    # Pre-compute all timeframe data per symbol
    all_data: dict[str, dict[str, pl.DataFrame]] = {}
    for symbol in symbols:
        sym = ohlc_data.filter(pl.col("ts_code") == symbol).sort("trade_date")
        if sym.is_empty():
            all_data[symbol] = {"daily": pl.DataFrame(), "weekly": pl.DataFrame(), "monthly": pl.DataFrame()}
            continue
        all_data[symbol] = {
            "daily": sym,
            "weekly": _resample_ohlc(sym, "1w"),
            "monthly": _resample_ohlc(sym, "1mo"),
        }

    timeframes = ["daily", "weekly", "monthly"]
    tf_labels = ["Day", "Week", "Month"]
    n_ma = len(ma_periods)
    # Per (symbol, timeframe): candlestick + volume + n_ma MAs + buy_arrows + sell_arrows = 2 + n_ma + 2
    traces_per_tf = 2 + n_ma + 2
    traces_per_symbol = traces_per_tf * len(timeframes)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    for sym_idx, symbol in enumerate(symbols):
        for tf_idx, tf in enumerate(timeframes):
            df = all_data[symbol][tf]
            is_default = sym_idx == 0 and tf_idx == 0
            visible = True if is_default else False

            if df.is_empty():
                for _ in range(traces_per_tf):
                    fig.add_trace(go.Scatter(x=[], y=[], visible=False), row=1, col=1)
                continue

            dates = df["trade_date"].to_list()
            opens = df["open"].to_list()
            highs = df["high"].to_list()
            lows = df["low"].to_list()
            closes = df["close"].to_list()
            vols = df["vol"].to_list()

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=dates,
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    name="Price",
                    visible=visible,
                    increasing_line_color="#E8443A",
                    increasing_fillcolor="#E8443A",
                    decreasing_line_color="#2DB757",
                    decreasing_fillcolor="#2DB757",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Volume bars
            vol_colors = ["rgba(232,68,58,0.5)" if c >= o else "rgba(45,183,87,0.5)" for o, c in zip(opens, closes)]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=vols,
                    marker_color=vol_colors,
                    marker_line_width=0,
                    name="Volume",
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Moving averages
            for ma_period in ma_periods:
                if df.height >= ma_period:
                    ma_vals = df.with_columns(
                        pl.col("close").rolling_mean(window_size=ma_period).alias("ma")
                    )["ma"].to_list()
                else:
                    ma_vals = [None] * df.height

                color = MA_COLORS.get(ma_period, "#888")
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=ma_vals,
                        mode="lines",
                        name=f"MA{ma_period}",
                        line=dict(width=1.3, color=color),
                        visible=visible,
                        showlegend=is_default,
                        legendgroup=f"ma{ma_period}",
                    ),
                    row=1,
                    col=1,
                )

            # Buy arrows (line + marker = arrow shape)
            buy_x, buy_y, buy_base = _get_arrow_points(df, trade_log, symbol, "BUY")
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="arrow-up",
                        size=12,
                        color="#2DB757",
                        line=dict(width=1.5, color="#1a8c3e"),
                        angleref="up",
                    ),
                    visible=visible,
                    showlegend=is_default,
                    legendgroup="buy",
                    hovertemplate="<b>BUY</b><br>%{x}<br>Price: %{text}<extra></extra>",
                    text=[f"{p:.3f}" for p in buy_base] if buy_base else [],
                ),
                row=1,
                col=1,
            )

            # Sell arrows
            sell_x, sell_y, sell_base = _get_arrow_points(df, trade_log, symbol, "SELL")
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="arrow-down",
                        size=12,
                        color="#E8443A",
                        line=dict(width=1.5, color="#b0201a"),
                        angleref="up",
                    ),
                    visible=visible,
                    showlegend=is_default,
                    legendgroup="sell",
                    hovertemplate="<b>SELL</b><br>%{x}<br>Price: %{text}<extra></extra>",
                    text=[f"{p:.3f}" for p in sell_base] if sell_base else [],
                ),
                row=1,
                col=1,
            )

    # Store metadata for JS-based controls in generate_html_report
    total_traces = traces_per_symbol * len(symbols)

    def _make_visible(sym_idx: int, tf_idx: int) -> list[bool]:
        vis = [False] * total_traces
        start = sym_idx * traces_per_symbol + tf_idx * traces_per_tf
        for i in range(traces_per_tf):
            vis[start + i] = True
        return vis

    fig.update_layout(
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#FFFFFF",
        height=700,
        margin=dict(t=60, b=40, l=65, r=20),
        font=dict(family="Arial, sans-serif", size=11, color="#333"),
        hovermode="x unified",
        meta={
            "kline_symbols": symbols,
            "kline_tf_labels": tf_labels,
            "kline_traces_per_tf": traces_per_tf,
            "kline_traces_per_symbol": traces_per_symbol,
            "kline_total_traces": total_traces,
        },
        xaxis=dict(
            rangeslider_visible=False,
            showgrid=True,
            gridcolor="#ECECEC",
            gridwidth=0.5,
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor="#ECECEC",
            gridwidth=0.5,
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="#ECECEC",
            gridwidth=0.5,
            side="right",
        ),
        yaxis2=dict(
            title="Vol",
            showgrid=False,
            side="right",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.0,
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )

    return fig


def _resample_ohlc(df: pl.DataFrame, period: str) -> pl.DataFrame:
    """Resample daily OHLC to weekly or monthly."""
    resampled = (
        df.sort("trade_date")
        .group_by_dynamic("trade_date", every=period)
        .agg(
            pl.col("ts_code").first(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("vol").sum(),
        )
        .sort("trade_date")
    )
    return resampled


def _get_arrow_points(
    df: pl.DataFrame,
    trade_log: pl.DataFrame,
    symbol: str,
    side: str,
) -> tuple[list, list, list]:
    """Get arrow marker positions. Returns (dates, y_positions, actual_prices)."""
    if trade_log.is_empty():
        return [], [], []

    trades = trade_log.filter((pl.col("symbol") == symbol) & (pl.col("side") == side))
    if trades.is_empty():
        return [], [], []

    dates = trades["timestamp"].to_list()
    y_positions = []
    prices = []
    for d in dates:
        bar = df.filter(pl.col("trade_date") == d)
        if not bar.is_empty():
            if side == "BUY":
                y_positions.append(bar["low"][0] * 0.985)
            else:
                y_positions.append(bar["high"][0] * 1.015)
            prices.append(bar["close"][0])
        else:
            y_positions.append(None)
            prices.append(0)
    return dates, y_positions, prices


def generate_html_report(
    report_fig: go.Figure,
    kline_fig: go.Figure,
    title: str = "Backtest Report",
    output_path: str | None = None,
) -> str:
    """Generate a single HTML file with tab navigation between report and K-line pages."""
    import json as _json

    report_html = report_fig.to_html(full_html=False, include_plotlyjs=False)
    kline_html = kline_fig.to_html(full_html=False, include_plotlyjs=False)

    meta = kline_fig.layout.meta or {}
    symbols_js = _json.dumps(meta.get("kline_symbols", []))
    tf_labels_js = _json.dumps(meta.get("kline_tf_labels", ["Day", "Week", "Month"]))
    traces_per_tf = meta.get("kline_traces_per_tf", 8)
    traces_per_symbol = meta.get("kline_traces_per_symbol", 24)
    total_traces = meta.get("kline_total_traces", 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; }}
.tab-bar {{
    display: flex;
    background: #fff;
    border-bottom: 2px solid #e0e0e0;
    padding: 0 20px;
    position: sticky;
    top: 0;
    z-index: 100;
}}
.tab-btn {{
    padding: 12px 24px;
    border: none;
    background: none;
    font-size: 14px;
    font-weight: 500;
    color: #666;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
}}
.tab-btn:hover {{ color: #333; background: #f8f8f8; }}
.tab-btn.active {{ color: #1a73e8; border-bottom-color: #1a73e8; }}
.tab-content {{ display: none; padding: 16px 20px; background: #fff; margin: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.tab-content.active {{ display: block; }}
h1 {{ padding: 16px 20px 0; font-size: 18px; color: #333; }}
.kline-controls {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 0;
    border-bottom: 1px solid #eee;
    margin-bottom: 8px;
}}
.kline-controls label {{ font-size: 12px; color: #666; font-weight: 500; }}
.kline-controls select {{
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 12px;
    background: #fff;
    cursor: pointer;
}}
.tf-group {{ display: flex; gap: 2px; }}
.tf-btn {{
    padding: 5px 14px;
    border: 1px solid #ddd;
    background: #fff;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
}}
.tf-btn:first-child {{ border-radius: 4px 0 0 4px; }}
.tf-btn:last-child {{ border-radius: 0 4px 4px 0; }}
.tf-btn.active {{ background: #1a73e8; color: #fff; border-color: #1a73e8; }}
.tf-btn:hover:not(.active) {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('report')">Performance Report</button>
    <button class="tab-btn" onclick="switchTab('kline')">K-Line Charts</button>
</div>
<div id="tab-report" class="tab-content active">{report_html}</div>
<div id="tab-kline" class="tab-content">
    <div class="kline-controls">
        <label>Symbol</label>
        <select id="kline-symbol-select" onchange="klineUpdate()"></select>
        <label>Timeframe</label>
        <div class="tf-group" id="kline-tf-group"></div>
    </div>
    {kline_html}
</div>
<script>
function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
    window.dispatchEvent(new Event('resize'));
}}

(function() {{
    var symbols = {symbols_js};
    var tfLabels = {tf_labels_js};
    var tracesPerTf = {traces_per_tf};
    var tracesPerSymbol = {traces_per_symbol};
    var totalTraces = {total_traces};
    var curSym = 0, curTf = 0;

    var select = document.getElementById('kline-symbol-select');
    symbols.forEach(function(s, i) {{
        var opt = document.createElement('option');
        opt.value = i; opt.textContent = s;
        select.appendChild(opt);
    }});

    var tfGroup = document.getElementById('kline-tf-group');
    tfLabels.forEach(function(lbl, i) {{
        var btn = document.createElement('button');
        btn.className = 'tf-btn' + (i === 0 ? ' active' : '');
        btn.textContent = lbl;
        btn.onclick = function() {{
            curTf = i;
            tfGroup.querySelectorAll('.tf-btn').forEach(function(b) {{ b.classList.remove('active'); }});
            btn.classList.add('active');
            klineUpdate();
        }};
        tfGroup.appendChild(btn);
    }});

    window.klineUpdate = function() {{
        curSym = parseInt(select.value);
        var vis = new Array(totalTraces).fill(false);
        var start = curSym * tracesPerSymbol + curTf * tracesPerTf;
        for (var i = 0; i < tracesPerTf; i++) vis[start + i] = true;
        var klineDiv = document.getElementById('tab-kline').querySelector('.plotly-graph-div');
        if (klineDiv) Plotly.restyle(klineDiv, {{'visible': vis.map(function(v){{ return v; }})}});
    }};
}})();
</script>
</body>
</html>"""

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html
