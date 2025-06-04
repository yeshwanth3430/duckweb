import pandas as pd
from pathlib import Path
import duckdb
from datetime import datetime, timedelta
import os
import numpy as np

class DatabaseManager:
    def __init__(self, db_path="data/market_data.duckdb"):
        """Initialize the database manager for DuckDB"""
        self.db_path = db_path
        self._ensure_db_exists()
        self._initialize_tables()

    def _ensure_db_exists(self):
        """Ensure the DuckDB file and its directory exist"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        # Just touch the file to ensure it exists
        Path(self.db_path).touch(exist_ok=True)

    def _initialize_tables(self):
        """Initialize the database tables if they don't exist"""
        conn = duckdb.connect(self.db_path)
        for index in ['nifty', 'banknifty', 'sensex']:
            # Daily table
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {index}_daily (
                datetime TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """)
            # Hourly table
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {index}_hourly (
                datetime TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """)
            # 1-minute table
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {index}_1min (
                datetime TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """)
        conn.close()

    def get_table_names(self):
        """Get list of available tables in the DuckDB database"""
        conn = duckdb.connect(self.db_path)
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        return [t[0] for t in tables]

    def get_available_dates(self, table_name):
        """Get the earliest and latest dates available for a given table"""
        try:
            conn = duckdb.connect(self.db_path)
            query = f"SELECT MIN(datetime), MAX(datetime) FROM {table_name}"
            result = conn.execute(query).fetchone()
            conn.close()
            if result and result[0] and result[1]:
                return pd.to_datetime(result[0]), pd.to_datetime(result[1])
            return None, None
        except Exception as e:
            print(f"Error getting available dates: {str(e)}")
            return None, None

    def get_index_data(self, index_type, date_range, timeframe="daily"):
        """Get index data for the specified date range and timeframe from DuckDB"""
        try:
            conn = duckdb.connect(self.db_path)
            start_date, end_date = date_range
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            table_name = f"{index_type}_{timeframe}" if timeframe != "daily" else f"{index_type}_daily"
            query = f"""
            SELECT datetime, open, high, low, close, volume
            FROM {table_name}
            WHERE datetime BETWEEN ? AND ?
            ORDER BY datetime
            """
            df = conn.execute(query, [start_str, end_str]).df()
            conn.close()
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    # Sample data and import methods can be adapted for DuckDB if needed

def delete_rows_by_date(table_name, date_str, db_path='data/market_data.duckdb'):
    """
    Delete all rows from the specified table where the date part of 'datetime' matches date_str (format: 'YYYY-MM-DD').
    """
    import duckdb
    try:
        conn = duckdb.connect(db_path)
        query = f"""
        DELETE FROM {table_name}
        WHERE CAST(datetime AS DATE) = '{date_str}'
        """
        conn.execute(query)
        conn.close()
        print(f"Deleted rows from {table_name} where date = {date_str}")
        return True
    except Exception as e:
        print(f"Error deleting rows: {e}")
        return False
