#!/usr/bin/env python3
"""
Export all tables from the PostgreSQL database into a local DuckDB file.

Each public-schema table is read via psycopg2, loaded into a pandas
DataFrame, and written into DuckDB with the same table name.

Usage:
    python pg_to_duckdb.py
    python pg_to_duckdb.py --output /path/to/output.duckdb
"""

import argparse
import configparser
from pathlib import Path

import duckdb
import pandas as pd
import psycopg2

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "data" / "phi" / "real_time" / "activate.duckdb"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output DuckDB file path")
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
    return parser.parse_args()


def get_db_connection(secrets_path: str):
    cfg = configparser.ConfigParser()
    cfg.read(secrets_path)
    def _strip(val: str) -> str:
        return val.strip('"')

    return psycopg2.connect(
        host=_strip(cfg["database"]["server"]),
        port=int(_strip(cfg["database"]["port"])),
        dbname=_strip(cfg["database"]["database"]),
        user=_strip(cfg["user"]["user"]),
        password=_strip(cfg["user"]["password"]),
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Connect to PostgreSQL --------------------------------------------
    print("Connecting to PostgreSQL ...")
    pg_conn = get_db_connection(args.secrets)
    cur = pg_conn.cursor()

    # List all public tables
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )
    tables = [row[0] for row in cur.fetchall()]
    print(f"Found {len(tables)} tables: {', '.join(tables)}")

    # --- Connect to DuckDB -----------------------------------------------
    if output_path.exists():
        output_path.unlink()
    duck = duckdb.connect(str(output_path))

    # --- Copy each table --------------------------------------------------
    for table in tables:
        print(f"  {table} ...", end=" ", flush=True)
        df = pd.read_sql(f'SELECT * FROM "{table}"', pg_conn)
        duck.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
        print(f"{len(df)} rows")

    cur.close()
    pg_conn.close()
    duck.close()

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(tables)} tables to {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
