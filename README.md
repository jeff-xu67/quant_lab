# Quant Lab - ETF 量化回测实验室

事件驱动的 ETF 量化回测框架，支持动量轮动、均值回复、趋势跟踪三种策略，提供完整的指标计算、可视化和实验管理能力。

## 特性

- **事件驱动回测引擎** — Market → Signal → Order → Fill 事件链，支持当bar收盘/次bar开盘两种成交模式
- **统一策略接口** — 继承 `Strategy` 基类即可快速开发新策略，装饰器注册后 YAML 直接引用
- **三种内置策略** — 动量轮动 (MomentumRotation)、均值回复 (MeanReversion)、趋势跟踪 (TrendFollowing)
- **完整指标体系** — 收益率、最大回撤、Sharpe、Sortino、Calmar、胜率、盈亏比等 14 项指标
- **Plotly 可视化** — 权益曲线、回撤图、月度收益热力图、综合报告面板
- **实验管理** — YAML 配置驱动、运行结果自动存档、多次运行对比
- **高性能数据层** — DuckDB 存储 + Polars 零拷贝读取，2000+ 只 ETF 秒级加载

## 技术栈

| 用途 | 库 |
|------|-----|
| 数据存储 | DuckDB |
| 数据处理 | Polars |
| 数值计算 | NumPy |
| 可视化 | Plotly |
| 配置管理 | PyYAML |

## 项目结构

```
quant_lab/
├── configs/                    # YAML 配置文件
│   ├── default.yaml
│   └── examples/
│       ├── momentum_rotation.yaml
│       ├── mean_reversion.yaml
│       └── trend_following.yaml
├── data/                       # 数据目录 (gitignored)
│   ├── etf/                    # 原始 CSV
│   └── quant_lab.duckdb        # DuckDB 数据库
├── results/                    # 回测结果 (gitignored)
├── scripts/
│   ├── build_etf_db.py         # 数据库构建脚本
│   └── run_backtest.py         # 回测 CLI 入口
├── src/quant_lab/
│   ├── db.py                   # 数据访问层 (DuckDB + Polars)
│   ├── core/
│   │   ├── events.py           # 事件类型 + EventBus
│   │   └── engine.py           # 回测引擎主循环
│   ├── data/
│   │   └── feed.py             # DataFeed (逐bar发射 MarketEvent)
│   ├── strategy/
│   │   ├── base.py             # Strategy ABC + 注册表
│   │   ├── momentum_rotation.py
│   │   ├── mean_reversion.py
│   │   └── trend_following.py
│   ├── portfolio/
│   │   ├── portfolio.py        # 持仓管理 + 仓位控制
│   │   └── position.py         # 单标的持仓
│   ├── execution/
│   │   ├── broker.py           # 模拟经纪商
│   │   └── slippage.py         # 滑点模型
│   ├── metrics/
│   │   ├── calculator.py       # 统计指标计算
│   │   └── visualization.py    # Plotly 图表
│   └── experiment/
│       ├── config.py           # YAML 配置加载
│       ├── runner.py           # 实验编排
│       ├── comparison.py       # 多次运行对比
│       └── logger.py           # 日志配置
└── tests/                      # pytest 单元测试
```

## 快速开始

### 1. 安装

```bash
pip install -e ".[dev]"
```

### 2. 构建数据库

```bash
python scripts/build_etf_db.py --csv-dir data/etf --db-path data/quant_lab.duckdb
```

### 3. 运行回测

```bash
# 使用默认配置 (动量轮动)
python scripts/run_backtest.py configs/default.yaml

# 运行趋势跟踪策略
python scripts/run_backtest.py configs/examples/trend_following.yaml

# 运行并打开 Plotly 可视化报告
python scripts/run_backtest.py configs/examples/momentum_rotation.yaml --show-report
```

输出示例：

```
=============================================
        Backtest Performance Summary
=============================================
  Total Return:            33.30%
  Annualized Return:        5.91%
  Max Drawdown:           -15.46%
  Max DD Duration:            546 days
  Sharpe Ratio:             0.354
  Sortino Ratio:            0.489
  Calmar Ratio:             0.382
  Volatility:              12.92%
---------------------------------------------
  Trade Count:                 82
  Win Rate:                45.12%
  Profit Factor:            1.969
  Avg Trade Return:         1.65%
  Avg Win:                  7.42%
  Avg Loss:                -3.10%
=============================================
```

### 4. 对比多次运行

```python
from quant_lab.experiment.comparison import RunComparison

comp = RunComparison("results")
print(comp.list_runs())           # 查看所有运行
fig = comp.plot_equity_comparison(["run_id_1", "run_id_2"])
fig.show()
```

## 配置说明

所有回测参数通过 YAML 配置文件管理：

```yaml
backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000.0

universe:
  symbols:
    - "510300.SH"
    - "510500.SH"
    - "159915.SZ"

strategy:
  name: MomentumRotation       # 策略类名
  params:
    lookback: 20               # 动量回望窗口
    top_k: 3                   # 持仓数量
    rebalance_days: 20         # 轮动周期

execution:
  commission_rate: 0.0003      # 手续费 3bps
  slippage_model: fixed        # fixed / volume
  slippage_bps: 5.0            # 滑点基点
  fill_on_next_open: true      # true=次bar开盘成交 (推荐)

portfolio:
  position_sizing: equal_weight  # equal_weight / signal_weight / fixed_fraction
  max_position_pct: 0.30
```

## 策略开发

继承 `Strategy` 基类并使用 `@register_strategy` 装饰器即可添加新策略：

```python
from quant_lab.strategy.base import Strategy, register_strategy
from quant_lab.core.events import Event, MarketEvent, SignalEvent

@register_strategy
class MyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "MyStrategy"

    def warmup_period(self) -> int:
        return self.params.get("window", 20)

    def on_market(self, event: Event) -> None:
        assert isinstance(event, MarketEvent)
        self._bar_count += 1
        self._update_history(event.bars, self.warmup_period() + 1)

        if self._bar_count < self.warmup_period():
            return

        # 计算信号逻辑...
        self.event_bus.publish(
            SignalEvent(timestamp=event.timestamp, symbol="510300.SH", direction=1.0)
        )
```

注册后即可在 YAML 中通过 `name: MyStrategy` 引用。

## 架构设计

```
DataFeed.next()
  └─→ MarketEvent
        ├─→ Broker.on_market()       # 成交 pending orders (fill_on_next_open 模式)
        ├─→ Portfolio.on_market()    # 更新价格，记录权益快照
        └─→ Strategy.on_market()     # 计算信号
              └─→ SignalEvent
                    └─→ Portfolio.on_signal()   # 仓位管理 → 生成订单
                          └─→ OrderEvent
                                └─→ Broker.on_order()    # 成交模拟
                                      └─→ FillEvent
                                            └─→ Portfolio.on_fill()  # 更新持仓和现金
```

## 数据接口

```python
from quant_lab import QuantLabDB

db = QuantLabDB("data/quant_lab.duckdb")

# Pandas 接口
df = db.get_etf_daily("510300.SH", start_date="2020-01-01")

# Polars 接口 (推荐，零拷贝)
df = db.get_etf_daily_pl(["510300.SH", "510500.SH"], start_date="2020-01-01")

# 批量获取
df = db.get_universe_pl(symbols, "2020-01-01", "2024-12-31")

db.close()
```

## 数据库表结构

表名: `etf_daily`，主键: `(ts_code, trade_date)`

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | TEXT | 标的代码 |
| trade_date | DATE | 交易日 |
| open/high/low/close | DOUBLE | OHLC 价格 |
| pre_close | DOUBLE | 前收盘价 |
| change/pct_chg | DOUBLE | 涨跌额/涨跌幅 |
| vol/amount | DOUBLE | 成交量/成交额 |
| adj_factor | DOUBLE | 复权因子 |
| total_share/total_size | DOUBLE | 份额/规模 |
| nav | DOUBLE | 净值 |
| fund_name | TEXT | 基金名称 |
| exchange | TEXT | 交易所 |
| fund_category | TEXT | 基金类别 |

## 测试

```bash
python -m pytest tests/ -v
```

## License

Private
