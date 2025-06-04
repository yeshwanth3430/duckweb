import pandas as pd
import numpy as np

def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands for the given dataframe.
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV data
    period (int): Period for moving average calculation
    std_dev (float): Number of standard deviations for the bands
    
    Returns:
    pd.DataFrame: DataFrame with Bollinger Bands
    """
    df = df.copy()
    
    # Calculate middle band (SMA)
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    df['bb_std'] = df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
    
    # Calculate bandwidth
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Calculate %B
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return pd.DataFrame({
        'bb_middle': df['bb_middle'],
        'bb_upper': df['bb_upper'],
        'bb_lower': df['bb_lower'],
        'bb_bandwidth': df['bb_bandwidth'],
        'bb_percent_b': df['bb_percent_b']
    })

def get_bollinger_signals(df):
    """
    Generate trading signals based on Bollinger Bands.
    
    Parameters:
    df (pd.DataFrame): DataFrame with Bollinger Bands
    
    Returns:
    pd.DataFrame: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=df.index)
    
    # Price crosses above upper band (bearish)
    signals['bb_upper_cross'] = np.where(
        (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1)),
        'Bearish',
        np.where(
            (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1)),
            'Bullish',
            'Neutral'
        )
    )
    
    # Bandwidth expansion/contraction
    signals['bb_bandwidth_signal'] = np.where(
        df['bb_bandwidth'] > df['bb_bandwidth'].rolling(window=20).mean(),
        'Expanding',
        'Contracting'
    )
    
    return signals 