#!/usr/bin/env python3
"""List all tables and their columns from the database."""

import argparse
import configparser
import psycopg2


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets", help="Path to database_secrets.txt", default="/ksg/kehl_mm_data/mmai/v22/data/phi/database_secrets.txt")
    args = parser.parse_args()
    print(f"Using secrets from: {args.secrets}")
    conn = get_db_connection(args.secrets)
    cur = conn.cursor()

    # Get all tables in the public schema
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )
    tables = [row[0] for row in cur.fetchall()]

    for table in tables:
        print(f"\n=== {table} ===")
        cur.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,),
        )
        for col_name, data_type, nullable in cur.fetchall():
            null_flag = "" if nullable == "YES" else " NOT NULL"
            print(f"  {col_name}: {data_type}{null_flag}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
