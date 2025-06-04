import pandas as pd
import numpy as np

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Calculate the fast and slow EMAs
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    })

def get_macd_signals(df):
    """Generate trading signals based on MACD"""
    signals = pd.DataFrame(index=df.index)
    
    # MACD line crosses above signal line (bullish)
    signals['macd_cross'] = np.where(
        (df['macd_line'] > df['signal_line']) & 
        (df['macd_line'].shift(1) <= df['signal_line'].shift(1)),
        'Bullish',
        np.where(
            (df['macd_line'] < df['signal_line']) & 
            (df['macd_line'].shift(1) >= df['signal_line'].shift(1)),
            'Bearish',
            'Neutral'
        )
    )
    
    # MACD histogram crosses zero (bullish/bearish)
    signals['histogram_cross'] = np.where(
        (df['histogram'] > 0) & (df['histogram'].shift(1) <= 0),
        'Bullish',
        np.where(
            (df['histogram'] < 0) & (df['histogram'].shift(1) >= 0),
            'Bearish',
            'Neutral'
        )
    )
    
    return signals
