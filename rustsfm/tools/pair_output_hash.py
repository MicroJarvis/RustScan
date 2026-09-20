#!/usr/bin/env python3
"""Hashes the normalized pair outputs of a RustSFM matching database.

The hash covers every ``matches`` and ``two_view_geometries`` row in ascending
``pair_id`` order, including the raw blobs, so any change in matched pairs,
match indices, inlier masks or recovered models changes the digest. SQLite page
layout is deliberately excluded, which makes the digest comparable across runs
that wrote the same logical output.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def _update_table(
    digest: "hashlib._Hash", connection: sqlite3.Connection, table: str
) -> int:
    columns = _table_columns(connection, table)
    if "pair_id" not in columns:
        raise SystemExit(f"table {table} has no pair_id column")
    ordered = ["pair_id"] + sorted(column for column in columns if column != "pair_id")
    digest.update(f"table:{table}\n".encode())
    digest.update(("columns:" + ",".join(ordered) + "\n").encode())
    rows = 0
    selection = ", ".join(ordered)
    for row in connection.execute(
        f"SELECT {selection} FROM {table} ORDER BY pair_id ASC"
    ):
        for column, value in zip(ordered, row):
            digest.update(f"{column}=".encode())
            if value is None:
                digest.update(b"null")
            elif isinstance(value, bytes):
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
            elif isinstance(value, int):
                digest.update(f"i{value}".encode())
            elif isinstance(value, float):
                # Hash the exact bits so tiny numeric drift cannot hide.
                digest.update(b"f")
                digest.update(
                    __import__("struct").pack("<d", value)
                )
            else:
                digest.update(f"s{value}".encode())
            digest.update(b"\x00")
        digest.update(b"\n")
        rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    arguments = parser.parse_args()
    if not arguments.database.is_file():
        raise SystemExit(f"missing database: {arguments.database}")
    digest = hashlib.sha256()
    connection = sqlite3.connect(f"file:{arguments.database}?mode=ro", uri=True)
    try:
        counts = {
            table: _update_table(digest, connection, table)
            for table in ("matches", "two_view_geometries")
        }
    finally:
        connection.close()
    for table, rows in counts.items():
        print(f"{table} rows: {rows}")
    print(f"pairs SHA-256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
