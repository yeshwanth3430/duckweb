import pandas as pd
import numpy as np

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    df = df.copy()
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    # Calculate SuperTrend
    hl2 = (df['high'] + df['low']) / 2
    
    # Calculate upper and lower bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize SuperTrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    # Calculate SuperTrend
    for i in range(period, len(df)):
        curr, prev = i, i-1
        
        if df['close'][curr] > upper_band[prev]:
            direction[curr] = 1
        elif df['close'][curr] < lower_band[prev]:
            direction[curr] = -1
        else:
            direction[curr] = direction[prev]
            
            if direction[curr] == 1 and lower_band[curr] < lower_band[prev]:
                lower_band[curr] = lower_band[prev]
            if direction[curr] == -1 and upper_band[curr] > upper_band[prev]:
                upper_band[curr] = upper_band[prev]
        
        if direction[curr] == 1:
            supertrend[curr] = lower_band[curr]
        else:
            supertrend[curr] = upper_band[curr]
    
    return pd.DataFrame({
        'supertrend': supertrend,
        'direction': direction,
        'upper_band': upper_band,
        'lower_band': lower_band
    })

def get_supertrend_signals(df):
    """Generate trading signals based on SuperTrend"""
    signals = pd.DataFrame(index=df.index)
    
    # SuperTrend direction change
    signals['supertrend_signal'] = np.where(
        df['direction'] == 1,
        'Bullish',
        'Bearish'
    )
    
    # SuperTrend crossover
    signals['supertrend_cross'] = np.where(
        (df['direction'] == 1) & (df['direction'].shift(1) == -1),
        'Bullish',
        np.where(
            (df['direction'] == -1) & (df['direction'].shift(1) == 1),
            'Bearish',
            'Neutral'
        )
    )
    
    return signals
