import pandas as pd
import numpy as np

def identify_inside_bars(df):
    """
    Identify inside bars in the given dataframe.
    An inside bar is a bar where the high is lower than the previous bar's high
    and the low is higher than the previous bar's low.
    """
    df = df.copy()
    
    # Calculate previous bar's high and low
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Identify inside bars
    df['is_inside_bar'] = (df['high'] < df['prev_high']) & (df['low'] > df['prev_low'])
    
    # Calculate inside bar size
    df['ib_size'] = (df['high'] - df['low']) / df['low'] * 100
    
    # Calculate relative position within previous bar
    df['ib_position'] = ((df['high'] + df['low']) / 2 - df['prev_low']) / (df['prev_high'] - df['prev_low'])
    
    return df

def analyze_inside_bars(df):
    """
    Analyze inside bars and return statistics.
    """
    df = identify_inside_bars(df)
    
    # Basic statistics
    total_bars = len(df)
    inside_bars = df['is_inside_bar'].sum()
    ib_percentage = (inside_bars / total_bars) * 100
    
    # Inside bar size analysis
    ib_sizes = df[df['is_inside_bar']]['ib_size']
    size_stats = {
        'mean': ib_sizes.mean(),
        'median': ib_sizes.median(),
        'min': ib_sizes.min(),
        'max': ib_sizes.max(),
        'std': ib_sizes.std()
    }
    
    # Position analysis
    ib_positions = df[df['is_inside_bar']]['ib_position']
    position_stats = {
        'mean': ib_positions.mean(),
        'median': ib_positions.median(),
        'std': ib_positions.std()
    }
    
    # Breakout analysis
    df['next_bar_high'] = df['high'].shift(-1)
    df['next_bar_low'] = df['low'].shift(-1)
    df['breakout_up'] = df['is_inside_bar'] & (df['next_bar_high'] > df['prev_high'])
    df['breakout_down'] = df['is_inside_bar'] & (df['next_bar_low'] < df['prev_low'])
    
    breakout_stats = {
        'up_breakouts': df['breakout_up'].sum(),
        'down_breakouts': df['breakout_down'].sum(),
        'up_breakout_rate': (df['breakout_up'].sum() / inside_bars) * 100 if inside_bars > 0 else 0,
        'down_breakout_rate': (df['breakout_down'].sum() / inside_bars) * 100 if inside_bars > 0 else 0
    }
    
    return {
        'total_bars': total_bars,
        'inside_bars': inside_bars,
        'ib_percentage': ib_percentage,
        'size_stats': size_stats,
        'position_stats': position_stats,
        'breakout_stats': breakout_stats
    }

def get_ib_summary(df):
    """
    Create a summary dataframe of inside bars.
    """
    df = identify_inside_bars(df)
    ib_df = df[df['is_inside_bar']].copy()
    
    if ib_df.empty:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame({
        'datetime': ib_df['datetime'],
        'open': ib_df['open'],
        'high': ib_df['high'],
        'low': ib_df['low'],
        'close': ib_df['close'],
        'ib_size': ib_df['ib_size'],
        'ib_position': ib_df['ib_position'],
        'breakout_up': ib_df['breakout_up'],
        'breakout_down': ib_df['breakout_down']
    })
    
    return summary_df 