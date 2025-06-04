import pandas as pd
import numpy as np

def analyze_volume(df):
    """
    Analyze volume patterns in the given dataframe.
    """
    df = df.copy()
    
    # Calculate volume metrics
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Identify high volume bars
    df['is_high_volume'] = df['volume_ratio'] > 1.5
    
    # Calculate volume statistics
    volume_stats = {
        'avg_volume': df['volume'].mean(),
        'max_volume': df['volume'].max(),
        'min_volume': df['volume'].min(),
        'high_volume_bars': df['is_high_volume'].sum(),
        'high_volume_percentage': (df['is_high_volume'].sum() / len(df)) * 100
    }
    
    return df, volume_stats

def find_support_resistance(df, window=20, threshold=0.02):
    """
    Find potential support and resistance levels.
    """
    df = df.copy()
    
    # Calculate local minima and maxima
    df['is_min'] = df['low'].rolling(window=window, center=True).min() == df['low']
    df['is_max'] = df['high'].rolling(window=window, center=True).max() == df['high']
    
    # Get potential levels
    support_levels = df[df['is_min']]['low'].unique()
    resistance_levels = df[df['is_max']]['high'].unique()
    
    # Cluster nearby levels
    def cluster_levels(levels, threshold):
        if len(levels) == 0:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        
        for level in levels[1:]:
            if (level - clusters[-1][-1]) / clusters[-1][-1] <= threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    support_levels = cluster_levels(support_levels, threshold)
    resistance_levels = cluster_levels(resistance_levels, threshold)
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels
    } 