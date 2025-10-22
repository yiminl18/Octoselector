import sqlite3
import pandas as pd

def execute_sqlite_query(db_path: str, query: str):
    """Run a SQL query on a SQLite DB and return a pandas DataFrame, or None on error."""
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)
    except Exception:
        return None

def get_tables(path: str):
    """Return all table names as a list of (name,) tuples from the SQLite database."""
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return cur.fetchall()

def get_table_schema(path: str, table_name: str):
    """Return PRAGMA table_info rows for a given table."""
    # Quote table name defensively for PRAGMA context
    safe_table = table_name.replace('"', '""')
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{safe_table}");')
        return cur.fetchall()

def concat_schema(path: str) -> str:
    """Concatenate table names and column lists into a single schema string."""
    parts = []
    for table in get_tables(path):
        tname = table[0]
        cols = [col[1] for col in get_table_schema(path, tname)]
        parts.append(f"Table Name: {tname}; Columns: {','.join(cols)};")
    return " \n".join(parts)