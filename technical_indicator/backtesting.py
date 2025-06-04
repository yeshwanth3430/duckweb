import pandas as pd
import numpy as np

# Helper to calculate max drawdown
def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return drawdown.min()

# Helper to calculate expectancy
def calculate_expectancy(trade_profits):
    wins = trade_profits[trade_profits > 0]
    losses = trade_profits[trade_profits <= 0]
    win_rate = len(wins) / len(trade_profits) if len(trade_profits) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    return win_rate, expectancy

# EMA Crossover Backtest
def backtest_ema(df, fast=12, slow=26):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
    df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = df['position'].diff().abs().sum() / 2
    trade_profits = df.loc[df['position'].diff() != 0, 'strategy']
    win_rate, expectancy = calculate_expectancy(trade_profits)
    max_dd = max_drawdown(df['equity'])
    return {
        'Strategy': f'EMA {fast}:{slow}',
        'Total P&L': df['equity'].iloc[-1] - 1,
        'Trades': int(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_dd
    }

# SMA Crossover Backtest
def backtest_sma(df, fast=20, slow=50):
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(fast).mean()
    df['sma_slow'] = df['close'].rolling(slow).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = df['position'].diff().abs().sum() / 2
    trade_profits = df.loc[df['position'].diff() != 0, 'strategy']
    win_rate, expectancy = calculate_expectancy(trade_profits)
    max_dd = max_drawdown(df['equity'])
    return {
        'Strategy': f'SMA {fast}:{slow}',
        'Total P&L': df['equity'].iloc[-1] - 1,
        'Trades': int(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_dd
    }

# SuperTrend Backtest (assumes 'supertrend' column exists)
def backtest_supertrend(df, supertrend_col='supertrend'):
    df = df.copy()
    df['signal'] = np.where(df[supertrend_col] < df['close'], 1, -1)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = df['position'].diff().abs().sum() / 2
    trade_profits = df.loc[df['position'].diff() != 0, 'strategy']
    win_rate, expectancy = calculate_expectancy(trade_profits)
    max_dd = max_drawdown(df['equity'])
    return {
        'Strategy': f'SuperTrend',
        'Total P&L': df['equity'].iloc[-1] - 1,
        'Trades': int(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_dd
    }

# MACD Backtest (assumes 'macd_line' and 'macd_signal' columns exist)
def backtest_macd(df, macd_line_col='macd_line', macd_signal_col='macd_signal'):
    df = df.copy()
    df['signal'] = 0
    df.loc[df[macd_line_col] > df[macd_signal_col], 'signal'] = 1
    df.loc[df[macd_line_col] < df[macd_signal_col], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = df['position'].diff().abs().sum() / 2
    trade_profits = df.loc[df['position'].diff() != 0, 'strategy']
    win_rate, expectancy = calculate_expectancy(trade_profits)
    max_dd = max_drawdown(df['equity'])
    return {
        'Strategy': f'MACD',
        'Total P&L': df['equity'].iloc[-1] - 1,
        'Trades': int(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_dd
    }

def backtest_ema_with_trades(df, fast=12, slow=26):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
    df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = []
    current_pos = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    for i, row in df.iterrows():
        if current_pos == 0 and row['position'] != 0:
            # Entry
            current_pos = row['position']
            entry_idx = i
            entry_price = row['close']
            entry_time = row['datetime'] if 'datetime' in row else i
        elif current_pos != 0 and row['position'] != current_pos:
            # Exit
            exit_price = row['close']
            exit_time = row['datetime'] if 'datetime' in row else i
            direction = 'Long' if current_pos == 1 else 'Short'
            pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
            trades.append({
                'Strategy': f'EMA {fast}:{slow}',
                'Entry Time': entry_time,
                'Entry Price': entry_price,
                'Exit Time': exit_time,
                'Exit Price': exit_price,
                'Direction': direction,
                'P&L': pnl
            })
            current_pos = row['position']
            if current_pos != 0:
                entry_idx = i
                entry_price = row['close']
                entry_time = row['datetime'] if 'datetime' in row else i
            else:
                entry_idx = None
                entry_price = None
                entry_time = None
    # If still in a trade at the end, close at last price
    if current_pos != 0 and entry_idx is not None:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime'] if 'datetime' in df.columns else df.index[-1]
        direction = 'Long' if current_pos == 1 else 'Short'
        pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
        trades.append({
            'Strategy': f'EMA {fast}:{slow}',
            'Entry Time': entry_time,
            'Entry Price': entry_price,
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'Direction': direction,
            'P&L': pnl
        })
    # Summary
    trade_profits = pd.Series([t['P&L'] for t in trades])
    win_rate, expectancy = calculate_expectancy(trade_profits)
    summary = {
        'Strategy': f'EMA {fast}:{slow}',
        'Total P&L': trade_profits.sum(),
        'Trades': len(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_drawdown(df['equity'])
    }
    trades_df = pd.DataFrame(trades)
    return summary, trades_df

def backtest_single_ema_with_trades(df, period=20):
    df = df.copy()
    df['ema'] = df['close'].ewm(span=period, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['close'] > df['ema'], 'signal'] = 1
    df.loc[df['close'] < df['ema'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    trades = []
    current_pos = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    for i, row in df.iterrows():
        if current_pos == 0 and row['position'] != 0:
            # Entry
            current_pos = row['position']
            entry_idx = i
            entry_price = row['close']
            entry_time = row['datetime'] if 'datetime' in row else i
        elif current_pos != 0 and row['position'] != current_pos:
            # Exit
            exit_price = row['close']
            exit_time = row['datetime'] if 'datetime' in row else i
            direction = 'Long' if current_pos == 1 else 'Short'
            pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
            trades.append({
                'Strategy': f'EMA {period}',
                'Entry Time': entry_time,
                'Entry Price': entry_price,
                'Exit Time': exit_time,
                'Exit Price': exit_price,
                'Direction': direction,
                'P&L': pnl
            })
            current_pos = row['position']
            if current_pos != 0:
                entry_idx = i
                entry_price = row['close']
                entry_time = row['datetime'] if 'datetime' in row else i
            else:
                entry_idx = None
                entry_price = None
                entry_time = None
    # If still in a trade at the end, close at last price
    if current_pos != 0 and entry_idx is not None:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime'] if 'datetime' in df.columns else df.index[-1]
        direction = 'Long' if current_pos == 1 else 'Short'
        pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
        trades.append({
            'Strategy': f'EMA {period}',
            'Entry Time': entry_time,
            'Entry Price': entry_price,
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'Direction': direction,
            'P&L': pnl
        })
    # Summary
    trade_profits = pd.Series([t['P&L'] for t in trades])
    win_rate, expectancy = calculate_expectancy(trade_profits)
    summary = {
        'Strategy': f'EMA {period}',
        'Total Points': trade_profits.sum(),
        'Trades': len(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_drawdown(df['close'])
    }
    trades_df = pd.DataFrame(trades)
    return summary, trades_df

def backtest_supertrend_with_trades(df, supertrend_col='supertrend'):
    df = df.copy()
    df['signal'] = np.where(df[supertrend_col] < df['close'], 1, -1)
    df['position'] = df['signal'].shift(1).fillna(0)
    trades = []
    current_pos = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    for i, row in df.iterrows():
        if current_pos == 0 and row['position'] != 0:
            # Entry
            current_pos = row['position']
            entry_idx = i
            entry_price = row['close']
            entry_time = row['datetime'] if 'datetime' in row else i
        elif current_pos != 0 and row['position'] != current_pos:
            # Exit
            exit_price = row['close']
            exit_time = row['datetime'] if 'datetime' in row else i
            direction = 'Long' if current_pos == 1 else 'Short'
            pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
            trades.append({
                'Strategy': f'SuperTrend {supertrend_col}',
                'Entry Time': entry_time,
                'Entry Price': entry_price,
                'Exit Time': exit_time,
                'Exit Price': exit_price,
                'Direction': direction,
                'P&L': pnl
            })
            current_pos = row['position']
            if current_pos != 0:
                entry_idx = i
                entry_price = row['close']
                entry_time = row['datetime'] if 'datetime' in row else i
            else:
                entry_idx = None
                entry_price = None
                entry_time = None
    # If still in a trade at the end, close at last price
    if current_pos != 0 and entry_idx is not None:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime'] if 'datetime' in df.columns else df.index[-1]
        direction = 'Long' if current_pos == 1 else 'Short'
        pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
        trades.append({
            'Strategy': f'SuperTrend {supertrend_col}',
            'Entry Time': entry_time,
            'Entry Price': entry_price,
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'Direction': direction,
            'P&L': pnl
        })
    # Summary
    trade_profits = pd.Series([t['P&L'] for t in trades])
    win_rate, expectancy = calculate_expectancy(trade_profits)
    summary = {
        'Strategy': f'SuperTrend {supertrend_col}',
        'Total Points': trade_profits.sum(),
        'Trades': len(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_drawdown(df['close'])
    }
    trades_df = pd.DataFrame(trades)
    return summary, trades_df

def backtest_supertrend_rr_with_trades(df, supertrend_col='supertrend', rr_list=[1,2,3,4,5]):
    results = {}
    # First pass: identify all entry points
    df = df.copy()
    df['direction'] = np.where(df[supertrend_col] < df['close'], 1, -1)
    entry_points = []
    for i in range(1, len(df)):
        if df['direction'].iloc[i] != df['direction'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'st': df[supertrend_col].iloc[i],
                'direction': df['direction'].iloc[i]
            })
    
    # Second pass: backtest each R:R ratio using the same entry points
    for rr in rr_list:
        trades = []
        for entry in entry_points:
            entry_idx = entry['index']
            entry_price = entry['price']
            entry_time = entry['time']
            entry_st = entry['st']
            direction = entry['direction']
            
            # Calculate risk
            risk = abs(entry_price - entry_st)
            if risk == 0:
                risk = 1e-6  # avoid zero division
                
            # Find exit point
            exit_found = False
            for i in range(entry_idx + 1, len(df)):
                if direction == 1:  # Long
                    target = entry_price + rr * risk
                    stop = entry_st
                    # Check for target or stop or reversal
                    if df['low'].iloc[i] <= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price
                        })
                        exit_found = True
                        break
                    elif df['high'].iloc[i] >= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price
                        })
                        exit_found = True
                        break
                    elif df['direction'].iloc[i] != direction:
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price
                        })
                        exit_found = True
                        break
                else:  # Short
                    target = entry_price - rr * risk
                    stop = entry_st
                    # Check for target or stop or reversal
                    if df['high'].iloc[i] >= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price
                        })
                        exit_found = True
                        break
                    elif df['low'].iloc[i] <= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price
                        })
                        exit_found = True
                        break
                    elif df['direction'].iloc[i] != direction:
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price
                        })
                        exit_found = True
                        break
            
            # If no exit found by the end, close at last price
            if not exit_found:
                exit_price = df['close'].iloc[-1]
                exit_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else df.index[-1]
                trades.append({
                    'R:R': f'1:{rr}',
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'P&L': (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
                })
        
        # Summary
        trade_profits = pd.Series([t['P&L'] for t in trades])
        win_rate, expectancy = calculate_expectancy(trade_profits)
        summary = {
            'Strategy': f'SuperTrend {supertrend_col} 1:{rr}',
            'R:R': f'1:{rr}',
            'Total Points': trade_profits.sum(),
            'Trades': len(trades),
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Max DD': max_drawdown(trade_profits.cumsum())
        }
        trades_df = pd.DataFrame(trades)
        results[rr] = (summary, trades_df)
    
    return results 

def backtest_bollinger_bands_rr_with_trades(df, period=20, std_dev=2.0, rr_list=[1,2,3,4,5]):
    """
    Backtest Bollinger Bands strategy with R:R ratios.
    Entry: Price crosses above/below bands
    Stop: Middle band
    Target: R:R * (Entry - Stop) distance
    """
    results = {}
    
    # Calculate Bollinger Bands
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
    
    # First pass: identify all entry points
    entry_points = []
    for i in range(1, len(df)):
        # Long entry: price crosses above upper band
        if df['close'].iloc[i] > df['bb_upper'].iloc[i] and df['close'].iloc[i-1] <= df['bb_upper'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['bb_middle'].iloc[i],
                'direction': 1  # Long
            })
        # Short entry: price crosses below lower band
        elif df['close'].iloc[i] < df['bb_lower'].iloc[i] and df['close'].iloc[i-1] >= df['bb_lower'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['bb_middle'].iloc[i],
                'direction': -1  # Short
            })
    
    # Second pass: backtest each R:R ratio using the same entry points
    for rr in rr_list:
        trades = []
        for entry in entry_points:
            entry_idx = entry['index']
            entry_price = entry['price']
            entry_time = entry['time']
            stop_price = entry['stop']
            direction = entry['direction']
            
            # Calculate risk
            risk = abs(entry_price - stop_price)
            if risk == 0:
                risk = 1e-6  # avoid zero division
                
            # Find exit point
            exit_found = False
            for i in range(entry_idx + 1, len(df)):
                if direction == 1:  # Long
                    target = entry_price + rr * risk
                    stop = stop_price
                    # Check for target or stop
                    if df['low'].iloc[i] <= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price
                        })
                        exit_found = True
                        break
                    elif df['high'].iloc[i] >= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price
                        })
                        exit_found = True
                        break
                else:  # Short
                    target = entry_price - rr * risk
                    stop = stop_price
                    # Check for target or stop
                    if df['high'].iloc[i] >= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price
                        })
                        exit_found = True
                        break
                    elif df['low'].iloc[i] <= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price
                        })
                        exit_found = True
                        break
            
            # If no exit found by the end, close at last price
            if not exit_found:
                exit_price = df['close'].iloc[-1]
                exit_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else df.index[-1]
                trades.append({
                    'R:R': f'1:{rr}',
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'P&L': (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
                })
        
        # Summary
        trade_profits = pd.Series([t['P&L'] for t in trades])
        win_rate, expectancy = calculate_expectancy(trade_profits)
        summary = {
            'Strategy': f'Bollinger Bands {period},{std_dev} 1:{rr}',
            'R:R': f'1:{rr}',
            'Total Points': trade_profits.sum(),
            'Trades': len(trades),
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Max DD': max_drawdown(trade_profits.cumsum())
        }
        trades_df = pd.DataFrame(trades)
        results[rr] = (summary, trades_df)
    
    return results 

def backtest_ema_rr_with_trades(df, fast=12, slow=26, rr_list=[1,2,3,4,5]):
    """
    Backtest EMA crossover strategy with R:R ratios.
    Entry: EMA crossover
    Stop: Opposite EMA
    Target: R:R * (Entry - Stop) distance
    """
    results = {}
    
    # Calculate EMAs
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    # First pass: identify all entry points
    entry_points = []
    for i in range(1, len(df)):
        # Long entry: fast EMA crosses above slow EMA
        if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['ema_slow'].iloc[i],  # Use slow EMA as stop
                'direction': 1,  # Long
                'risk': abs(df['close'].iloc[i] - df['ema_slow'].iloc[i])  # Calculate risk at entry
            })
        # Short entry: fast EMA crosses below slow EMA
        elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['ema_slow'].iloc[i],  # Use slow EMA as stop
                'direction': -1,  # Short
                'risk': abs(df['close'].iloc[i] - df['ema_slow'].iloc[i])  # Calculate risk at entry
            })
    
    # Second pass: backtest each R:R ratio using the same entry points
    for rr in rr_list:
        trades = []
        for entry in entry_points:
            entry_idx = entry['index']
            entry_price = entry['price']
            entry_time = entry['time']
            stop_price = entry['stop']
            direction = entry['direction']
            risk = entry['risk']
            
            if risk == 0:
                risk = 1e-6  # avoid zero division
                
            # Find exit point
            exit_found = False
            exit_reason = None
            for i in range(entry_idx + 1, len(df)):
                if direction == 1:  # Long
                    target = entry_price + rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['low'].iloc[i] <= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['high'].iloc[i] >= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]:  # EMA crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                else:  # Short
                    target = entry_price - rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['high'].iloc[i] >= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['low'].iloc[i] <= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:  # EMA crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
            
            # If no exit found by the end, close at last price
            if not exit_found:
                exit_price = df['close'].iloc[-1]
                exit_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else df.index[-1]
                exit_reason = 'End of Data'
                trades.append({
                    'R:R': f'1:{rr}',
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'P&L': (exit_price - entry_price) if direction == 1 else (entry_price - exit_price),
                    'Risk': risk,
                    'Target': target if direction == 1 else target,
                    'Stop': stop,
                    'Exit Reason': exit_reason
                })
        
        # Summary
        trade_profits = pd.Series([t['P&L'] for t in trades])
        win_rate, expectancy = calculate_expectancy(trade_profits)
        
        # Calculate additional metrics
        winning_trades = [t for t in trades if t['P&L'] > 0]
        losing_trades = [t for t in trades if t['P&L'] <= 0]
        avg_win = sum(t['P&L'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['P&L'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Count exit reasons
        exit_reasons = {}
        for t in trades:
            reason = t['Exit Reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        summary = {
            'Strategy': f'EMA {fast}:{slow} 1:{rr}',
            'R:R': f'1:{rr}',
            'Total Points': trade_profits.sum(),
            'Trades': len(trades),
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Max DD': max_drawdown(trade_profits.cumsum()),
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Exit Reasons': exit_reasons
        }
        trades_df = pd.DataFrame(trades)
        results[rr] = (summary, trades_df)
    
    return results 

def backtest_macd_rr_with_trades(df, macd_line_col='macd_line', macd_signal_col='macd_signal', rr_list=[1,2,3,4,5]):
    """
    Backtest MACD strategy with R:R ratios.
    Entry: MACD line crosses signal line
    Stop: Signal line
    Target: R:R * (Entry - Stop) distance
    """
    results = {}
    
    # First pass: identify all entry points
    entry_points = []
    for i in range(1, len(df)):
        # Long entry: MACD line crosses above signal line
        if df[macd_line_col].iloc[i] > df[macd_signal_col].iloc[i] and df[macd_line_col].iloc[i-1] <= df[macd_signal_col].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df[macd_signal_col].iloc[i],  # Use signal line as stop
                'direction': 1,  # Long
                'risk': abs(df['close'].iloc[i] - df[macd_signal_col].iloc[i])  # Calculate risk at entry
            })
        # Short entry: MACD line crosses below signal line
        elif df[macd_line_col].iloc[i] < df[macd_signal_col].iloc[i] and df[macd_line_col].iloc[i-1] >= df[macd_signal_col].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df[macd_signal_col].iloc[i],  # Use signal line as stop
                'direction': -1,  # Short
                'risk': abs(df['close'].iloc[i] - df[macd_signal_col].iloc[i])  # Calculate risk at entry
            })
    
    # Second pass: backtest each R:R ratio using the same entry points
    for rr in rr_list:
        trades = []
        for entry in entry_points:
            entry_idx = entry['index']
            entry_price = entry['price']
            entry_time = entry['time']
            stop_price = entry['stop']
            direction = entry['direction']
            risk = entry['risk']
            
            if risk == 0:
                risk = 1e-6  # avoid zero division
                
            # Find exit point
            exit_found = False
            exit_reason = None
            for i in range(entry_idx + 1, len(df)):
                if direction == 1:  # Long
                    target = entry_price + rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['low'].iloc[i] <= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['high'].iloc[i] >= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df[macd_line_col].iloc[i] < df[macd_signal_col].iloc[i]:  # MACD crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                else:  # Short
                    target = entry_price - rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['high'].iloc[i] >= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['low'].iloc[i] <= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df[macd_line_col].iloc[i] > df[macd_signal_col].iloc[i]:  # MACD crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
            
            # If no exit found by the end, close at last price
            if not exit_found:
                exit_price = df['close'].iloc[-1]
                exit_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else df.index[-1]
                exit_reason = 'End of Data'
                trades.append({
                    'R:R': f'1:{rr}',
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'P&L': (exit_price - entry_price) if direction == 1 else (entry_price - exit_price),
                    'Risk': risk,
                    'Target': target if direction == 1 else target,
                    'Stop': stop,
                    'Exit Reason': exit_reason
                })
        
        # Summary
        trade_profits = pd.Series([t['P&L'] for t in trades])
        win_rate, expectancy = calculate_expectancy(trade_profits)
        
        # Calculate additional metrics
        winning_trades = [t for t in trades if t['P&L'] > 0]
        losing_trades = [t for t in trades if t['P&L'] <= 0]
        avg_win = sum(t['P&L'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['P&L'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Count exit reasons
        exit_reasons = {}
        for t in trades:
            reason = t['Exit Reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        summary = {
            'Strategy': f'MACD 1:{rr}',
            'R:R': f'1:{rr}',
            'Total Points': trade_profits.sum(),
            'Trades': len(trades),
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Max DD': max_drawdown(trade_profits.cumsum()),
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Exit Reasons': exit_reasons
        }
        trades_df = pd.DataFrame(trades)
        results[rr] = (summary, trades_df)
    
    return results 

def backtest_sma_with_trades(df, fast=20, slow=50):
    """
    Backtest SMA crossover strategy with detailed trade tracking.
    Entry: SMA crossover
    Exit: Opposite crossover or end of data
    """
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(window=fast).mean()
    df['sma_slow'] = df['close'].rolling(window=slow).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    
    trades = []
    current_pos = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    
    for i, row in df.iterrows():
        if current_pos == 0 and row['position'] != 0:
            # Entry
            current_pos = row['position']
            entry_idx = i
            entry_price = row['close']
            entry_time = row['datetime'] if 'datetime' in row else i
        elif current_pos != 0 and row['position'] != current_pos:
            # Exit
            exit_price = row['close']
            exit_time = row['datetime'] if 'datetime' in row else i
            direction = 'Long' if current_pos == 1 else 'Short'
            pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
            trades.append({
                'Strategy': f'SMA {fast}:{slow}',
                'Entry Time': entry_time,
                'Entry Price': entry_price,
                'Exit Time': exit_time,
                'Exit Price': exit_price,
                'Direction': direction,
                'P&L': pnl
            })
            current_pos = row['position']
            if current_pos != 0:
                entry_idx = i
                entry_price = row['close']
                entry_time = row['datetime'] if 'datetime' in row else i
            else:
                entry_idx = None
                entry_price = None
                entry_time = None
    
    # If still in a trade at the end, close at last price
    if current_pos != 0 and entry_idx is not None:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime'] if 'datetime' in df.columns else df.index[-1]
        direction = 'Long' if current_pos == 1 else 'Short'
        pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
        trades.append({
            'Strategy': f'SMA {fast}:{slow}',
            'Entry Time': entry_time,
            'Entry Price': entry_price,
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'Direction': direction,
            'P&L': pnl
        })
    
    # Summary
    trade_profits = pd.Series([t['P&L'] for t in trades])
    win_rate, expectancy = calculate_expectancy(trade_profits)
    summary = {
        'Strategy': f'SMA {fast}:{slow}',
        'Total Points': trade_profits.sum(),
        'Trades': len(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_drawdown(df['close'])
    }
    trades_df = pd.DataFrame(trades)
    return summary, trades_df

def backtest_sma_rr_with_trades(df, fast=20, slow=50, rr_list=[1,2,3,4,5]):
    """
    Backtest SMA crossover strategy with R:R ratios.
    Entry: SMA crossover
    Stop: Opposite SMA
    Target: R:R * (Entry - Stop) distance
    """
    results = {}
    
    # Calculate SMAs
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(window=fast).mean()
    df['sma_slow'] = df['close'].rolling(window=slow).mean()
    
    # First pass: identify all entry points
    entry_points = []
    for i in range(1, len(df)):
        # Long entry: fast SMA crosses above slow SMA
        if df['sma_fast'].iloc[i] > df['sma_slow'].iloc[i] and df['sma_fast'].iloc[i-1] <= df['sma_slow'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['sma_slow'].iloc[i],  # Use slow SMA as stop
                'direction': 1,  # Long
                'risk': abs(df['close'].iloc[i] - df['sma_slow'].iloc[i])  # Calculate risk at entry
            })
        # Short entry: fast SMA crosses below slow SMA
        elif df['sma_fast'].iloc[i] < df['sma_slow'].iloc[i] and df['sma_fast'].iloc[i-1] >= df['sma_slow'].iloc[i-1]:
            entry_points.append({
                'index': i,
                'price': df['close'].iloc[i],
                'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                'stop': df['sma_slow'].iloc[i],  # Use slow SMA as stop
                'direction': -1,  # Short
                'risk': abs(df['close'].iloc[i] - df['sma_slow'].iloc[i])  # Calculate risk at entry
            })
    
    # Second pass: backtest each R:R ratio using the same entry points
    for rr in rr_list:
        trades = []
        for entry in entry_points:
            entry_idx = entry['index']
            entry_price = entry['price']
            entry_time = entry['time']
            stop_price = entry['stop']
            direction = entry['direction']
            risk = entry['risk']
            
            if risk == 0:
                risk = 1e-6  # avoid zero division
                
            # Find exit point
            exit_found = False
            exit_reason = None
            for i in range(entry_idx + 1, len(df)):
                if direction == 1:  # Long
                    target = entry_price + rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['low'].iloc[i] <= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['high'].iloc[i] >= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['sma_fast'].iloc[i] < df['sma_slow'].iloc[i]:  # SMA crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Long',
                            'P&L': exit_price - entry_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                else:  # Short
                    target = entry_price - rr * risk
                    stop = stop_price
                    # Check for target or stop or reversal
                    if df['high'].iloc[i] >= stop:
                        exit_price = stop
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Stop Loss'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['low'].iloc[i] <= target:
                        exit_price = target
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Target'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
                    elif df['sma_fast'].iloc[i] > df['sma_slow'].iloc[i]:  # SMA crossover reversal
                        exit_price = df['close'].iloc[i]
                        exit_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i
                        exit_reason = 'Reversal'
                        trades.append({
                            'R:R': f'1:{rr}',
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Direction': 'Short',
                            'P&L': entry_price - exit_price,
                            'Risk': risk,
                            'Target': target,
                            'Stop': stop,
                            'Exit Reason': exit_reason
                        })
                        exit_found = True
                        break
            
            # If no exit found by the end, close at last price
            if not exit_found:
                exit_price = df['close'].iloc[-1]
                exit_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else df.index[-1]
                exit_reason = 'End of Data'
                trades.append({
                    'R:R': f'1:{rr}',
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'P&L': (exit_price - entry_price) if direction == 1 else (entry_price - exit_price),
                    'Risk': risk,
                    'Target': target if direction == 1 else target,
                    'Stop': stop,
                    'Exit Reason': exit_reason
                })
        
        # Summary
        trade_profits = pd.Series([t['P&L'] for t in trades])
        win_rate, expectancy = calculate_expectancy(trade_profits)
        
        # Calculate additional metrics
        winning_trades = [t for t in trades if t['P&L'] > 0]
        losing_trades = [t for t in trades if t['P&L'] <= 0]
        avg_win = sum(t['P&L'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['P&L'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Count exit reasons
        exit_reasons = {}
        for t in trades:
            reason = t['Exit Reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        summary = {
            'Strategy': f'SMA {fast}:{slow} 1:{rr}',
            'R:R': f'1:{rr}',
            'Total Points': trade_profits.sum(),
            'Trades': len(trades),
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Max DD': max_drawdown(trade_profits.cumsum()),
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Exit Reasons': exit_reasons
        }
        trades_df = pd.DataFrame(trades)
        results[rr] = (summary, trades_df)
    
    return results