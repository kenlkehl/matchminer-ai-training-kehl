#!/usr/bin/env python3
"""
Sample a random email from the activate_emails table and print it.
"""

import argparse
import configparser
import subprocess
import tempfile
from pathlib import Path

import psycopg2

SECRETS_FILE = Path(__file__).resolve().parents[3] / "data" / "phi" / "database_secrets.txt"


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
    parser.add_argument("--secrets", type=str, default=str(SECRETS_FILE),
                        help="Path to database_secrets.txt")
    args = parser.parse_args()

    conn = get_db_connection(args.secrets)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, mrn, subject, body, cc, email_from, email_not_send, "
        "provider_email_not_send_reason, other_email_not_send_reason, "
        "email_sent, date_sent, date_created, redcap_record_id, "
        "execution_timestamp, hipaa_logged, recipient_npi "
        "FROM activate_emails "
        "WHERE body IS NOT NULL AND body != '' "
        "ORDER BY random() LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        print("No emails found in activate_emails.")
        return

    columns = [
        "id", "mrn", "subject", "body", "cc", "email_from", "email_not_send",
        "provider_email_not_send_reason", "other_email_not_send_reason",
        "email_sent", "date_sent", "date_created", "redcap_record_id",
        "execution_timestamp", "hipaa_logged", "recipient_npi",
    ]

    data = dict(zip(columns, row))

    print("=" * 80)
    print("RANDOM EMAIL FROM activate_emails")
    print("=" * 80)
    for col, val in data.items():
        print(f"{col}: {val}")

    # Write body to a temp HTML file and open in browser
    tmp_dir = Path("/ksg/kehl_mm_data/active_serial/data/phi/temp_files")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="email_", dir=tmp_dir, delete=False
    ) as f:
        f.write(data["body"])
        tmp_path = f.name

    print(f"\nSaved email body to {tmp_path}")
    subprocess.Popen(["xdg-open", tmp_path],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
