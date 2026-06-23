from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class QuantLabDB:
    """统一的ETF数据访问层（DuckDB）"""

    def __init__(self, db_path: str | Path = "data/quant_lab.duckdb") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS etf_daily (
                ts_code TEXT,
                trade_date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                pre_close DOUBLE,
                change DOUBLE,
                pct_chg DOUBLE,
                vol DOUBLE,
                amount DOUBLE,
                adj_factor DOUBLE,
                total_share DOUBLE,
                total_size DOUBLE,
                nav DOUBLE,
                fund_name TEXT,
                exchange TEXT,
                fund_category TEXT,
                source_file TEXT,
                load_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_etf_daily_trade_date
            ON etf_daily(trade_date)
            """
        )

    def import_etf_csv_dir(self, csv_dir: str | Path, pattern: str = "*.csv") -> int:
        """批量导入目录下ETF CSV，返回导入（或更新）行数。"""
        csv_dir = Path(csv_dir)
        files = sorted(csv_dir.glob(pattern))
        if not files:
            return 0

        total_rows = 0
        for file_path in files:
            total_rows += self.import_one_csv(file_path)
        return total_rows

    def import_one_csv(self, csv_path: str | Path) -> int:
        """导入单个CSV，支持重复导入（按主键覆盖更新）。"""
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        expected_columns = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "adj_factor",
            "total_share",
            "total_size",
            "nav",
            "fund_name",
            "exchange",
            "fund_category",
        ]

        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"CSV字段缺失: {missing} | 文件: {csv_path}")

        # 统一日期格式：20240131 -> DATE
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")

        # 文件名作为来源追踪
        df["source_file"] = csv_path.name

        self.conn.register("tmp_etf_df", df)
        self.conn.execute(
            """
            INSERT INTO etf_daily AS t
            SELECT
                ts_code,
                trade_date,
                open,
                high,
                low,
                close,
                pre_close,
                change,
                pct_chg,
                vol,
                amount,
                adj_factor,
                total_share,
                total_size,
                nav,
                fund_name,
                exchange,
                fund_category,
                source_file,
                CURRENT_TIMESTAMP
            FROM tmp_etf_df
            WHERE trade_date IS NOT NULL
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                pre_close = EXCLUDED.pre_close,
                change = EXCLUDED.change,
                pct_chg = EXCLUDED.pct_chg,
                vol = EXCLUDED.vol,
                amount = EXCLUDED.amount,
                adj_factor = EXCLUDED.adj_factor,
                total_share = EXCLUDED.total_share,
                total_size = EXCLUDED.total_size,
                nav = EXCLUDED.nav,
                fund_name = EXCLUDED.fund_name,
                exchange = EXCLUDED.exchange,
                fund_category = EXCLUDED.fund_category,
                source_file = EXCLUDED.source_file,
                load_time = CURRENT_TIMESTAMP
            """
        )
        self.conn.unregister("tmp_etf_df")

        return len(df)

    def get_etf_daily(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """统一查询接口：按代码和可选日期范围获取日线。"""
        sql = """
            SELECT *
            FROM etf_daily
            WHERE ts_code = ?
        """
        params: list[object] = [ts_code]

        if start_date:
            sql += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND trade_date <= ?"
            params.append(end_date)

        sql += " ORDER BY trade_date"
        return self.conn.execute(sql, params).fetch_df()

    def list_symbols(self) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT ts_code, MIN(trade_date) AS start_date, MAX(trade_date) AS end_date, COUNT(*) AS bars
            FROM etf_daily
            GROUP BY ts_code
            ORDER BY ts_code
            """
        ).fetch_df()

    def query(self, sql: str) -> pd.DataFrame:
        """保留原生SQL能力，便于研究实验。"""
        return self.conn.execute(sql).fetch_df()
