import pandas as pd
import numpy as np

def analyze_gaps(df, date_range):
    """
    Analyze gaps in the given dataframe.
    """
    df = df.copy()
    
    # Calculate gaps
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = ((df['open'] - df['prev_close']) / df['prev_close']) * 100
    
    # Identify gap direction
    df['gap_direction'] = np.where(df['gap'] > 0, 'Up', 'Down')
    
    # Categorize gaps
    df['gap_category'] = pd.cut(
        abs(df['gap']),
        bins=[0, 0.3, 0.7, float('inf')],
        labels=['Flat', 'Medium', 'High']
    )
    
    # Check if gaps are filled
    df['gap_filled'] = False
    df['days_to_fill'] = np.nan
    
    for i in range(1, len(df)):
        if abs(df['gap'].iloc[i]) > 0.3:  # Only consider significant gaps
            gap_direction = df['gap_direction'].iloc[i]
            gap_size = df['gap'].iloc[i]
            gap_high = df['high'].iloc[i]
            gap_low = df['low'].iloc[i]
            
            # Look ahead to find if gap is filled
            for j in range(i + 1, len(df)):
                if gap_direction == 'Up':
                    if df['low'].iloc[j] <= gap_low:
                        df.loc[df.index[i], 'gap_filled'] = True
                        df.loc[df.index[i], 'days_to_fill'] = j - i
                        break
                else:  # Down gap
                    if df['high'].iloc[j] >= gap_high:
                        df.loc[df.index[i], 'gap_filled'] = True
                        df.loc[df.index[i], 'days_to_fill'] = j - i
                        break
    
    # Calculate gap statistics
    gap_stats = {
        'total_gaps': len(df[abs(df['gap']) > 0.3]),
        'filled_gaps': df['gap_filled'].sum(),
        'gap_fill_rate': (df['gap_filled'].sum() / len(df[abs(df['gap']) > 0.3])) * 100 if len(df[abs(df['gap']) > 0.3]) > 0 else 0
    }
    
    return df, gap_stats

def get_gap_summary(df):
    """
    Create a summary dataframe of gaps.
    """
    df = df.copy()
    
    # Filter for significant gaps
    gap_df = df[abs(df['gap']) > 0.3].copy()
    
    if gap_df.empty:
        return pd.DataFrame()
    
    # Calculate additional metrics
    gap_df['fill_return'] = ((gap_df['close'] - gap_df['open']) / gap_df['open']) * 100
    gap_df['risk_reward_ratio'] = abs(gap_df['fill_return'] / gap_df['gap'])
    
    # Calculate max adverse and favorable excursion
    gap_df['max_adverse_excursion'] = 0.0
    gap_df['max_favorable_excursion'] = 0.0
    
    for i in range(len(gap_df)):
        idx = gap_df.index[i]
        if gap_df.loc[idx, 'gap_filled']:
            days_to_fill = int(gap_df.loc[idx, 'days_to_fill'])
            if days_to_fill > 0:
                future_data = df.loc[idx+1:idx+days_to_fill]
                if not future_data.empty:
                    if gap_df.loc[idx, 'gap_direction'] == 'Up':
                        gap_df.loc[idx, 'max_adverse_excursion'] = ((future_data['low'].min() - gap_df.loc[idx, 'open']) / gap_df.loc[idx, 'open']) * 100
                        gap_df.loc[idx, 'max_favorable_excursion'] = ((future_data['high'].max() - gap_df.loc[idx, 'open']) / gap_df.loc[idx, 'open']) * 100
                    else:
                        gap_df.loc[idx, 'max_adverse_excursion'] = ((gap_df.loc[idx, 'open'] - future_data['high'].max()) / gap_df.loc[idx, 'open']) * 100
                        gap_df.loc[idx, 'max_favorable_excursion'] = ((gap_df.loc[idx, 'open'] - future_data['low'].min()) / gap_df.loc[idx, 'open']) * 100
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'datetime': gap_df['datetime'],
        'gap': gap_df['gap'],
        'gap_category': gap_df['gap_category'],
        'gap_direction': gap_df['gap_direction'],
        'gap_filled': gap_df['gap_filled'],
        'days_to_fill': gap_df['days_to_fill'],
        'fill_return': gap_df['fill_return'],
        'risk_reward_ratio': gap_df['risk_reward_ratio'],
        'max_adverse_excursion': gap_df['max_adverse_excursion'],
        'max_favorable_excursion': gap_df['max_favorable_excursion']
    })
    
    return summary_df 