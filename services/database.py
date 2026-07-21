"""Shared SQLite connection policy for concurrent web and worker access."""

import sqlite3


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection
