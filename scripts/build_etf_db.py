from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_lab import QuantLabDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DuckDB database from ETF CSV files")
    parser.add_argument("--csv-dir", default="data/etf", help="ETF CSV directory")
    parser.add_argument("--db-path", default="data/quant_lab.duckdb", help="DuckDB file path")
    parser.add_argument("--pattern", default="*.csv", help="CSV filename glob pattern")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    db = QuantLabDB(args.db_path)
    try:
        imported = db.import_etf_csv_dir(csv_dir, pattern=args.pattern)
        symbols = db.list_symbols()
        print(f"Imported/updated rows: {imported}")
        print(f"Total symbols: {len(symbols)}")

        if len(symbols) > 0:
            print("Top 10 symbols overview:")
            print(symbols.head(10).to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
