import duckdb
import sqlite3
import pandas as pd
from datetime import datetime

def import_sensex_1min():
    # Connect to DuckDB
    duck_conn = duckdb.connect('/Users/pyeshwanthreddy/projects/DUCK/1MIN_SENSEX.duckdb')
    
    # Read data from DuckDB
    df = duck_conn.execute("SELECT * FROM sensex_1min").fetchdf()
    
    if df.empty:
        print("No data found in DuckDB table")
        return False
    
    # Print column names for debugging
    print("Columns in DuckDB table:", df.columns.tolist())
    
    # Select only the required columns
    required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_columns]
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect('data/market_data.db')
    
    # Convert datetime if needed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Insert data into SQLite
    df.to_sql('sensex_1min', sqlite_conn, if_exists='append', index=False)
    
    # Close connections
    sqlite_conn.close()
    duck_conn.close()
    
    print(f"Successfully imported {len(df)} rows of Sensex 1-minute data")
    return True

if __name__ == "__main__":
    import_sensex_1min() 