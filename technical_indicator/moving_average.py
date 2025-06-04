import pandas as pd
import numpy as np

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_moving_averages(df, periods=[20, 50, 200]):
    """Calculate multiple moving averages for a dataframe"""
    df = df.copy()
    
    for period in periods:
        df[f'sma_{period}'] = calculate_sma(df['close'], period)
        df[f'ema_{period}'] = calculate_ema(df['close'], period)
    
    return df

def get_moving_average_signals(df):
    """Generate trading signals based on moving average crossovers"""
    signals = pd.DataFrame(index=df.index)
    
    # SMA 20 vs 50 crossover
    signals['sma_20_50_cross'] = np.where(
        df['sma_20'] > df['sma_50'],
        'Bullish',
        'Bearish'
    )
    
    # SMA 50 vs 200 crossover
    signals['sma_50_200_cross'] = np.where(
        df['sma_50'] > df['sma_200'],
        'Bullish',
        'Bearish'
    )
    
    # EMA 20 vs 50 crossover
    signals['ema_20_50_cross'] = np.where(
        df['ema_20'] > df['ema_50'],
        'Bullish',
        'Bearish'
    )
    
    return signals
