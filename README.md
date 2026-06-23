Quant Lab - ETF DuckDB 数据库

目标
- 将 data/etf 目录下的 ETF CSV 文件统一导入 DuckDB。
- 提供统一数据接口，支持按代码和日期范围查询。

目录
- src/quant_lab/db.py: DuckDB 数据访问层
- scripts/build_etf_db.py: 批量导入脚本
- requirements.txt: Python 依赖

1. 安装依赖
在项目根目录执行：

pip install -r requirements.txt

2. 构建数据库
在项目根目录执行：

python scripts/build_etf_db.py --csv-dir data/etf --db-path data/quant_lab.duckdb

说明
- 默认会读取 data/etf/*.csv
- 使用主键 (ts_code, trade_date) 去重
- 重复导入时会自动更新已有记录

3. 在代码中使用统一接口

from quant_lab import QuantLabDB

# 连接数据库
qdb = QuantLabDB("data/quant_lab.duckdb")

# 按代码取日线（可加日期范围）
df = qdb.get_etf_daily("150010.SZ", start_date="2010-03-11", end_date="2010-12-31")
print(df.head())

# 列出全量标的与时间范围
print(qdb.list_symbols().head())

# 执行原生 SQL
print(qdb.query("SELECT ts_code, COUNT(*) AS n FROM etf_daily GROUP BY ts_code ORDER BY n DESC LIMIT 10"))

qdb.close()

4. 统一表结构
表名: etf_daily

核心字段
- ts_code, trade_date
- open, high, low, close, pre_close, change, pct_chg
- vol, amount, adj_factor
- total_share, total_size, nav
- fund_name, exchange, fund_category
- source_file, load_time

5. 常用 SQL 示例

# 查看某只 ETF 最新 20 条
SELECT *
FROM etf_daily
WHERE ts_code = '150010.SZ'
ORDER BY trade_date DESC
LIMIT 20;

# 查看某天全市场成交额前 20
SELECT ts_code, trade_date, amount
FROM etf_daily
WHERE trade_date = DATE '2020-01-02'
ORDER BY amount DESC
LIMIT 20;
