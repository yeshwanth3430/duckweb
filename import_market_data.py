import duckdb
import pandas as pd
from pathlib import Path

def import_market_data():
    # Source file paths
    source_files = {
        'nifty': {
            '1min': '/Users/pyeshwanthreddy/projects/DUCK/numbers.duckdb',
            '5min': '/Users/pyeshwanthreddy/projects/DUCK/5min_nifty.duckdb',
            '15min': '/Users/pyeshwanthreddy/projects/DUCK/15MIN_NIFTY.duckdb',
            '1hour': '/Users/pyeshwanthreddy/projects/DUCK/1HOU_NIFTY.duckdb',
            'daily': '/Users/pyeshwanthreddy/projects/DUCK/DAY_NIFTY.duckdb'
        },
        'banknifty': {
            '1min': '/Users/pyeshwanthreddy/projects/DUCK/numbers.duckdb',
            '5min': '/Users/pyeshwanthreddy/projects/DUCK/5MIN_BANKNIFTY.duckdb',
            '15min': '/Users/pyeshwanthreddy/projects/DUCK/15MIN_BANKNIFTY.duckdb',
            '1hour': '/Users/pyeshwanthreddy/projects/DUCK/1HOU_BANKNIFTY.duckdb',
            'daily': '/Users/pyeshwanthreddy/projects/DUCK/DAY_BANKNIFTY.duckdb'
        },
        'sensex': {
            '1min': '/Users/pyeshwanthreddy/projects/DUCK/numbers.duckdb',
            '5min': '/Users/pyeshwanthreddy/projects/DUCK/5MIN_SENSEX.duckdb',
            '15min': '/Users/pyeshwanthreddy/projects/DUCK/15MIN_SENSEX.duckdb',
            '1hour': '/Users/pyeshwanthreddy/projects/DUCK/1HOU_SENSEX.duckdb',
            'daily': '/Users/pyeshwanthreddy/projects/DUCK/DAY_SENSEX.duckdb'
        }
    }

    # Target database
    target_db = 'data/market_data.duckdb'
    
    # Ensure target directory exists
    Path(target_db).parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to target database
    target_conn = duckdb.connect(target_db)
    
    # Create tables if they don't exist
    for index in ['nifty', 'banknifty', 'sensex']:
        for timeframe in ['1min', '5min', '15min', '1hour', 'daily']:
            table_name = f"{index}_{timeframe}"
            target_conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                datetime TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """)
    
    # Import data for each index and timeframe
    for index, timeframes in source_files.items():
        for timeframe, source_file in timeframes.items():
            try:
                # Connect to source database
                source_conn = duckdb.connect(source_file)
                
                # Get table name based on index and timeframe
                if timeframe == '1min':
                    # For 1min data, we need to filter the spot_data table
                    query = f"""
                    SELECT datetime, open, high, low, close, volume 
                    FROM spot_data 
                    WHERE stock_code = '{index.upper()}'
                    """
                else:
                    # For other timeframes, use the table name directly
                    query = f"""
                    SELECT datetime, open, high, low, close, volume 
                    FROM {index}_{timeframe}
                    """
                
                # Read data from source
                df = source_conn.execute(query).df()
                
                if not df.empty:
                    # Ensure datetime is in correct format
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    # Remove duplicates
                    df = df.drop_duplicates(subset=['datetime'])
                    # Insert data into target database
                    target_table = f"{index}_{timeframe}"
                    target_conn.execute(f"DELETE FROM {target_table}")  # Clear existing data
                    target_conn.register('temp_df', df)
                    target_conn.execute(f"INSERT INTO {target_table} SELECT * FROM temp_df")
                    target_conn.unregister('temp_df')
                    print(f"Successfully imported {len(df)} rows to {target_table}")
                else:
                    print(f"No data found for {index}_{timeframe}")
                
                source_conn.close()
                
            except Exception as e:
                print(f"Error importing {index}_{timeframe}: {str(e)}")
    
    target_conn.close()
    print("Data import completed!")

if __name__ == "__main__":
    import_market_data() 