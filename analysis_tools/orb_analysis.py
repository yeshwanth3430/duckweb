import pandas as pd
import numpy as np

def identify_orb(df, range_minutes=15):
    """
    Identify Open Range Break (ORB) patterns in the given dataframe.
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV data
    range_minutes (int): Number of minutes to consider for the open range
    
    Returns:
    pd.DataFrame: DataFrame with ORB signals and statistics
    """
    df = df.copy()
    
    # Ensure datetime index
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Calculate open range high and low
    df['range_high'] = df['high'].rolling(window=range_minutes).max()
    df['range_low'] = df['low'].rolling(window=range_minutes).min()
    
    # Identify breakouts
    df['orb_up'] = df['close'] > df['range_high'].shift(1)
    df['orb_down'] = df['close'] < df['range_low'].shift(1)
    
    # Calculate range size
    df['range_size'] = ((df['range_high'] - df['range_low']) / df['range_low']) * 100
    
    # Calculate breakout strength
    df['breakout_strength'] = np.where(
        df['orb_up'],
        ((df['close'] - df['range_high'].shift(1)) / df['range_high'].shift(1)) * 100,
        np.where(
            df['orb_down'],
            ((df['range_low'].shift(1) - df['close']) / df['range_low'].shift(1)) * 100,
            0
        )
    )
    
    # Reset index to keep datetime as column
    df.reset_index(inplace=True)
    
    return df

def analyze_orb(df, range_minutes=15):
    """
    Analyze ORB patterns and return statistics.
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV data
    range_minutes (int): Number of minutes to consider for the open range
    
    Returns:
    dict: Dictionary containing ORB statistics
    """
    df = identify_orb(df, range_minutes)
    
    # Basic statistics
    total_bars = len(df)
    orb_up = df['orb_up'].sum()
    orb_down = df['orb_down'].sum()
    total_orb = orb_up + orb_down
    
    # Range size analysis
    range_sizes = df[df['orb_up'] | df['orb_down']]['range_size']
    size_stats = {
        'mean': range_sizes.mean(),
        'median': range_sizes.median(),
        'min': range_sizes.min(),
        'max': range_sizes.max(),
        'std': range_sizes.std()
    }
    
    # Breakout strength analysis
    breakout_strengths = df[df['orb_up'] | df['orb_down']]['breakout_strength']
    strength_stats = {
        'mean': breakout_strengths.mean(),
        'median': breakout_strengths.median(),
        'min': breakout_strengths.min(),
        'max': breakout_strengths.max(),
        'std': breakout_strengths.std()
    }
    
    # Success rate analysis (price continues in breakout direction)
    df['next_bar_return'] = df['close'].shift(-1) / df['close'] - 1
    df['successful_up'] = df['orb_up'] & (df['next_bar_return'] > 0)
    df['successful_down'] = df['orb_down'] & (df['next_bar_return'] < 0)
    
    success_stats = {
        'up_success_rate': (df['successful_up'].sum() / orb_up * 100) if orb_up > 0 else 0,
        'down_success_rate': (df['successful_down'].sum() / orb_down * 100) if orb_down > 0 else 0,
        'total_success_rate': ((df['successful_up'].sum() + df['successful_down'].sum()) / total_orb * 100) if total_orb > 0 else 0
    }
    
    return {
        'total_bars': total_bars,
        'orb_up': orb_up,
        'orb_down': orb_down,
        'total_orb': total_orb,
        'size_stats': size_stats,
        'strength_stats': strength_stats,
        'success_stats': success_stats
    }

def get_orb_summary(df, range_minutes=15):
    """
    Create a summary dataframe of ORB patterns.
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV data
    range_minutes (int): Number of minutes to consider for the open range
    
    Returns:
    pd.DataFrame: Summary dataframe of ORB patterns
    """
    df = identify_orb(df, range_minutes)
    orb_df = df[df['orb_up'] | df['orb_down']].copy()
    
    if orb_df.empty:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame({
        'datetime': orb_df['datetime'],
        'open': orb_df['open'],
        'high': orb_df['high'],
        'low': orb_df['low'],
        'close': orb_df['close'],
        'range_size': orb_df['range_size'],
        'breakout_strength': orb_df['breakout_strength'],
        'orb_up': orb_df['orb_up'],
        'orb_down': orb_df['orb_down'],
        'successful': orb_df['successful_up'] | orb_df['successful_down']
    })
    
    return summary_df 