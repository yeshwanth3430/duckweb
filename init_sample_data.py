import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import duckdb
from config import SAMPLE_DB_PATH, DEFAULT_INDICES, DEFAULT_TIMEFRAMES

def generate_sample_data(start_date, end_date, timeframe):
    """Generate sample market data for testing"""
    if timeframe == "daily":
        freq = "D"
    elif timeframe == "hourly":
        freq = "H"
    else:  # 1min
        freq = "min"

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate random price data
    base_price = 10000
    volatility = 0.02
    
    data = []
    for date in dates:
        # Generate random price movement
        price_change = np.random.normal(0, volatility)
        base_price *= (1 + price_change)
        
        # Generate OHLCV data
        high = base_price * (1 + abs(np.random.normal(0, 0.005)))
        low = base_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price * (1 + np.random.normal(0, 0.002))
        close = base_price * (1 + np.random.normal(0, 0.002))
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def initialize_sample_database():
    """Initialize the sample database with test data"""
    # Connect to DuckDB
    conn = duckdb.connect(SAMPLE_DB_PATH)
    
    # Generate data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create tables and insert sample data for each index and timeframe
    for index in DEFAULT_INDICES:
        for timeframe in DEFAULT_TIMEFRAMES:
            table_name = f"{index}_{timeframe}"
            
            # Create table
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                datetime TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """)
            
            # Generate and insert sample data
            df = generate_sample_data(start_date, end_date, timeframe)
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    
    conn.close()
    print("Sample database initialized successfully!")

if __name__ == "__main__":
    initialize_sample_database() 