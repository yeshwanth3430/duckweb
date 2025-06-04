import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from db_operations import DatabaseManager
from spot import show_analysis_options
from visualization.charts import plot_price_chart, plot_volume_chart, plot_technical_indicators
from technical_indicator.moving_average import calculate_moving_averages
from technical_indicator.macd import calculate_macd
from technical_indicator.super_trend import calculate_supertrend, get_supertrend_signals
from technical_indicator.bollinger_bands import calculate_bollinger_bands, get_bollinger_signals
import numpy as np
from analysis_tools.gap_analysis import analyze_gaps, get_gap_summary
from analysis_tools.ib_analysis import analyze_inside_bars, get_ib_summary
from analysis_tools.orb_analysis import analyze_orb, get_orb_summary
from visualization.gap_charts import plot_gap_analysis, plot_gap_fill_analysis, plot_gap_direction_analysis, plot_gap_patterns, plot_gap_risk_reward
from visualization.ib_charts import plot_ib_analysis, plot_ib_size_distribution, plot_ib_position_analysis, plot_ib_breakout_analysis
from visualization.orb_charts import plot_orb_analysis, plot_orb_success_analysis, plot_orb_time_analysis
import plotly.graph_objects as go
import itertools
from datetime import datetime as dt
from analysis import show_analysis_page
# Import backtesting functions
from technical_indicator.backtesting import (
    backtest_ema, backtest_sma, backtest_supertrend, backtest_macd, backtest_ema_with_trades, backtest_single_ema_with_trades, backtest_supertrend_with_trades, backtest_supertrend_rr_with_trades, backtest_ema_rr_with_trades, backtest_macd_rr_with_trades
)

def backtest_bollinger_bands(df, period=20, std_dev=2.0, bb_middle_col=None, bb_upper_col=None, bb_lower_col=None):
    # Simple BBands backtest: Buy when close crosses above lower band, sell when close crosses below upper band
    df = df.copy()
    if not (bb_middle_col and bb_upper_col and bb_lower_col):
        return None
    if not all(col in df.columns for col in [bb_middle_col, bb_upper_col, bb_lower_col]):
        return None
    df['signal'] = 0
    df.loc[df['close'] > df[bb_lower_col], 'signal'] = 1
    df.loc[df['close'] < df[bb_upper_col], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']
    df['equity'] = (1 + df['strategy']).cumprod()
    trades = df['position'].diff().abs().sum() / 2
    trade_profits = df.loc[df['position'].diff() != 0, 'strategy']
    from technical_indicator.backtesting import calculate_expectancy, max_drawdown
    win_rate, expectancy = calculate_expectancy(trade_profits)
    max_dd = max_drawdown(df['equity'])
    return {
        'Strategy': f'BBands {period},{std_dev}',
        'Total P&L': df['equity'].iloc[-1] - 1,
        'Trades': int(trades),
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Max DD': max_dd
    }

# Set page config
st.set_page_config(
    page_title="Indian Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .stSelectbox {
        background-color: white;
    }
    .stDateInput {
        background-color: white;
    }
    .analysis-option {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .analysis-option:hover {
        background-color: #e9ecef;
        cursor: pointer;
    }
    .card {
        background: #23272f;
        border-radius: 12px;
        padding: 1.2rem 1rem 1rem 1rem;
        margin: 0.5rem 0.5rem 1.2rem 0.5rem;
        box-shadow: 0 2px 12px #0003;
        color: #fff;
        text-align: center;
        transition: transform 0.15s;
    }
    .card:hover {
        transform: scale(1.04);
        box-shadow: 0 4px 24px #0005;
    }
    .card .icon {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .card .highlight {
        color: #4CAF50;
        font-weight: bold;
    }
    .card .negative {
        color: #e53935;
        font-weight: bold;
    }
    .card .neutral {
        color: #fbc02d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #1E88E5;'>Indian Market Analysis Dashboard</h1>
        <p style='color: #666;'>Analyze Nifty, Bank Nifty, and Sensex data with advanced tools</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def init_db():
    return DatabaseManager()

# Add this function at the top-level (after imports, before main)
def show_inside_bar_analysis(db, index_type, date_range):
    st.markdown("## Inside Bar Analysis Dashboard")
    # Get data for analysis
    df = db.get_index_data(index_type, date_range, "daily")
    if df is not None and not df.empty:
        df = df.sort_values('datetime')
        # Identify inside bars
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['is_ib'] = (df['high'] <= df['prev_high']) & (df['low'] >= df['prev_low'])
        total_bars = len(df)
        inside_bars = df['is_ib'].sum()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bars", total_bars)
        with col2:
            st.metric("Inside Bars", inside_bars)
        # Only use inside bar days for the table
        ib_df = df[df['is_ib']].copy()
        # Get previous day's open and close
        ib_df['prev_open'] = ib_df['open'].shift(1)
        ib_df['prev_close'] = ib_df['close'].shift(1)
        def get_pattern(row):
            if pd.isna(row['prev_open']) or pd.isna(row['prev_close']):
                return None
            today_ud = 'U' if row['open'] < row['close'] else 'D'
            prev_ud = 'U' if row['prev_open'] < row['prev_close'] else 'D'
            return prev_ud + today_ud
        ib_df['pattern'] = ib_df.apply(get_pattern, axis=1)
        # Next day broke up/down
        ib_df['next_high'] = ib_df['high'].shift(-1)
        ib_df['next_low'] = ib_df['low'].shift(-1)
        ib_df['broke_up'] = ib_df['next_high'] > ib_df['high']
        ib_df['broke_down'] = ib_df['next_low'] < ib_df['low']
        # Build summary table
        patterns = ['DD', 'UU', 'DU', 'UD']
        table_data = []
        for pat in patterns:
            pat_df = ib_df[ib_df['pattern'] == pat]
            count = len(pat_df)
            broke_up = pat_df['broke_up'].sum()
            broke_down = pat_df['broke_down'].sum()
            table_data.append({
                'Pattern': pat,
                'Count': count,
                'Next Day Broke Up': broke_up,
                'Next Day Broke Down': broke_down
            })
        pattern_table = pd.DataFrame(table_data)
        st.markdown("### 2-Day Pattern Table for Inside Bar Days with Next Day Breakout")
        st.dataframe(pattern_table, use_container_width=True)
        # Add detailed data table below
        st.markdown("### Inside Bar Pattern Details")
        detail_cols = [
            'datetime', 'open', 'high', 'low', 'close',
            'prev_open', 'prev_close', 'pattern', 'broke_up', 'broke_down'
        ]
        detail_df = ib_df[detail_cols].copy()
        detail_df['datetime'] = pd.to_datetime(detail_df['datetime']).dt.strftime('%Y-%m-%d')
        st.dataframe(detail_df, use_container_width=True)
    else:
        st.error("No data available for the selected period.")

def calculate_indicator(df, indicator_type, params):
    """Calculate technical indicators based on type and parameters"""
    if indicator_type == "EMA":
        fast_period, slow_period = params
        df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['signal'] = np.where(df[f'ema_{fast_period}'] > df[f'ema_{slow_period}'], 1, -1)
        return df
    
    elif indicator_type == "SuperTrend":
        period, multiplier = params
        # Calculate ATR
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate SuperTrend
        df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
        df['in_uptrend'] = True
        
        for i in range(1, len(df)):
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-1]
            
            if df['in_uptrend'].iloc[i-1]:
                if current_close < df['lowerband'].iloc[i-1]:
                    df.loc[df.index[i], 'in_uptrend'] = False
                else:
                    df.loc[df.index[i], 'in_uptrend'] = True
            else:
                if current_close > df['upperband'].iloc[i-1]:
                    df.loc[df.index[i], 'in_uptrend'] = True
                else:
                    df.loc[df.index[i], 'in_uptrend'] = False
        
        df['signal'] = np.where(df['in_uptrend'], 1, -1)
        return df
    
    elif indicator_type == "MACD":
        fast_period, slow_period, signal_period = params
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['signal'] = np.where(df['macd'] > df['signal_line'], 1, -1)
        return df
    
    elif indicator_type == "Bollinger Bands":
        period, std_dev = params
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        df['signal'] = np.where(df['close'] < df['bb_lower'], 1, np.where(df['close'] > df['bb_upper'], -1, 0))
        return df

def analyze_orb_with_indicator(db, index_type, date_range, df, indicator_type, indicator_params, orb_timeframe, orb_range, filters=None, rr_analysis=False, rr_ratios=None):
    """Analyze ORB with indicator signals"""
    # Calculate indicator signals
    df = calculate_indicator(df, indicator_type, indicator_params)
    
    # Get intraday data for ORB analysis
    intraday_df = db.get_index_data(index_type, date_range, orb_timeframe)
    if intraday_df is None or intraday_df.empty:
        return None, "No intraday data available"
    
    # Calculate ORB levels for each day
    intraday_df['date'] = pd.to_datetime(intraday_df['datetime']).dt.date
    orb_levels = []
    
    for date, group in intraday_df.groupby('date'):
        # Get first N minutes of data
        group = group.sort_values('datetime')
        orb_data = group.head(orb_range)
        
        if len(orb_data) == orb_range:
            orb_high = orb_data['high'].max()
            orb_low = orb_data['low'].min()
            orb_levels.append({
                'date': date,
                'orb_high': orb_high,
                'orb_low': orb_low
            })
    
    orb_df = pd.DataFrame(orb_levels)
    
    # Merge ORB levels with daily data
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df = df.merge(orb_df, on='date', how='left')
    
    # Apply filters if specified
    if filters:
        if filters.get('gap_filter'):
            gap_type = filters['gap_type']
            if gap_type == "Gap Up":
                df = df[df['gap'] > 0.3]
            elif gap_type == "Gap Down":
                df = df[df['gap'] < -0.3]
            else:  # Flat Open
                df = df[(df['gap'] >= -0.3) & (df['gap'] <= 0.3)]
        
        if filters.get('volume_filter'):
            volume_threshold = filters['volume_threshold']
            avg_volume = df['volume'].mean()
            df = df[df['volume'] > (avg_volume * volume_threshold)]
        
        if filters.get('trend_filter'):
            trend_period = 20  # Fixed period
            trend_direction = filters.get('trend_direction', 'Uptrend')
            df['trend'] = df['close'].rolling(window=trend_period).mean()
            if trend_direction == 'Uptrend':
                df = df[df['close'] > df['trend']]
            else:
                df = df[df['close'] < df['trend']]
    
    # Initialize results
    trades = []
    current_position = None
    
    # Analyze each day
    for i in range(1, len(df)):
        if pd.isna(df['orb_high'].iloc[i]) or pd.isna(df['orb_low'].iloc[i]):
            continue
        
        current_signal = df['signal'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        orb_high = df['orb_high'].iloc[i]
        orb_low = df['orb_low'].iloc[i]
        
        # Check for entry signals
        if current_position is None:
            # Long entry: Price breaks above ORB high and indicator is bullish
            if current_high > orb_high and current_signal == 1:
                entry_price = orb_high
                stop_loss = orb_low
                risk = entry_price - stop_loss
                
                if rr_analysis:
                    for rr in rr_ratios:
                        target = entry_price + (risk * rr)
                        trades.append({
                            'date': df['datetime'].iloc[i],
                            'type': 'LONG',
                            'entry': entry_price,
                            'stop': stop_loss,
                            'target': target,
                            'rr': rr,
                            'risk': risk,
                            'reward': risk * rr
                        })
                else:
                    trades.append({
                        'date': df['datetime'].iloc[i],
                        'type': 'LONG',
                        'entry': entry_price,
                        'stop': stop_loss,
                        'risk': risk
                    })
                current_position = 'LONG'
            
            # Short entry: Price breaks below ORB low and indicator is bearish
            elif current_low < orb_low and current_signal == -1:
                entry_price = orb_low
                stop_loss = orb_high
                risk = stop_loss - entry_price
                
                if rr_analysis:
                    for rr in rr_ratios:
                        target = entry_price - (risk * rr)
                        trades.append({
                            'date': df['datetime'].iloc[i],
                            'type': 'SHORT',
                            'entry': entry_price,
                            'stop': stop_loss,
                            'target': target,
                            'rr': rr,
                            'risk': risk,
                            'reward': risk * rr
                        })
                else:
                    trades.append({
                        'date': df['datetime'].iloc[i],
                        'type': 'SHORT',
                        'entry': entry_price,
                        'stop': stop_loss,
                        'risk': risk
                    })
                current_position = 'SHORT'
        
        # Check for exit signals
        elif current_position == 'LONG':
            if rr_analysis:
                if current_low < stop_loss or current_high >= target:
                    current_position = None
            else:
                if current_low < stop_loss:
                    current_position = None
        elif current_position == 'SHORT':
            if rr_analysis:
                if current_high > stop_loss or current_low <= target:
                    current_position = None
            else:
                if current_high > stop_loss:
                    current_position = None
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['type'] == 'LONG'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        if 'reward' in trades_df.columns:
            gross_profit = trades_df[trades_df['type'] == 'LONG']['reward'].sum()
            gross_loss = trades_df[trades_df['type'] == 'SHORT']['risk'].sum()
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0.0
        
        # Calculate max drawdown
        if 'reward' in trades_df.columns:
            cumulative_returns = trades_df['reward'].cumsum()
        else:
            cumulative_returns = trades_df['risk'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        metrics = {
            'Total Trades': total_trades,
            'Win Rate': f"{win_rate*100:.1f}%",
            'Profit Factor': f"{profit_factor:.2f}",
            'Max Drawdown': f"{max_drawdown:.1f}%"
        }
    else:
        metrics = {
            'Total Trades': 0,
            'Win Rate': "0%",
            'Profit Factor': "0.00",
            'Max Drawdown': "0%"
        }
    
    return trades_df, metrics

def main():
    # Set pandas option to increase maximum number of cells for styling
    pd.set_option("styler.render.max_elements", 800000)
    
    # Initialize database
    db = init_db()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #1E88E5;'>Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        # Add 'Indicators Analysis' to the navigation
        page = st.radio(
            "Select Analysis Type",
            ["Spot", "Analysis", "Analysis Reports", "Indicators Analysis", "Options"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        # Add data import buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Sample Data"):
                with st.spinner("Inserting sample data..."):
                    if db.insert_sample_data():
                        st.success("Sample data inserted successfully!")
                    else:
                        st.error("Failed to insert sample data.")
        with col2:
            if st.button("📥 Import DuckDB"):
                with st.spinner("Importing data from DuckDB..."):
                    if db.import_from_duckdb():
                        st.success("Data imported successfully!")
                    else:
                        st.error("Failed to import data.")
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <p style='color: #666;'>Select an index and date range to begin analysis</p>
            </div>
        """, unsafe_allow_html=True)

    # Add a placeholder for the new section
    if page == "Indicators Analysis":
        from indicators_analysis import show_indicators_analysis_page
        show_indicators_analysis_page(db)

    if page == "Spot":
        # Only spot-specific logic here, remove Actions section and related logic
        tables = db.get_table_names()
        base_tables = [table.replace('_1min', '') for table in tables if table.endswith('_1min')]
        base_tables = sorted(list(set(base_tables)))
        if not base_tables:
            st.error("No data tables found in the database. Please check your database setup.")
            return
        st.markdown("### Analysis Controls")
        st.markdown("#### Select Index")
        index_type = st.selectbox("Select Index", base_tables, key="spot_index")
        st.markdown("#### Select Timeframe")
        timeframe = st.selectbox(
            "Select Timeframe",
            ["Daily", "1 Hour", "15 Minute", "5 Minute", "1 Minute"],
            key="spot_timeframe_main"
        )
        # Convert timeframe selection to database format
        timeframe_map = {
            "Daily": "daily",
            "1 Hour": "1hour",
            "15 Minute": "15min",
            "5 Minute": "5min",
            "1 Minute": "1min"
        }
        selected_timeframe = timeframe_map[timeframe]
        
        # Add date range selection
        st.markdown("#### Select Date Range")
        table_name = f"{index_type}_daily"
        start_date, end_date = db.get_available_dates(table_name)
        if start_date and end_date:
            date_range = st.date_input(
                "Select Date Range",
                value=(end_date - timedelta(days=30), end_date),
                min_value=start_date,
                max_value=end_date,
                key="spot_date_range"
            )
            
            # Get data for the selected timeframe
            df = db.get_index_data(index_type, date_range, selected_timeframe)
            if df is not None and not df.empty:
                # Display price chart
                st.markdown("### Price Chart")
                st.plotly_chart(plot_price_chart(df), use_container_width=True)
                
                # Add indicator selection
                st.markdown("### Select Indicators")
                indicators = st.multiselect(
                    "Choose Technical Indicators",
                    ["Moving Averages", "MACD", "RSI", "Bollinger Bands", "SuperTrend"],
                    default=["Moving Averages"],
                    key="spot_indicators"
                )
                
                # Calculate and display selected indicators
                if indicators:
                    st.markdown("### Technical Indicators")
                    
                    # Calculate selected indicators
                    if "Moving Averages" in indicators:
                        st.markdown("#### Moving Average Settings")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### EMA Periods")
                            ema_periods = st.multiselect(
                                "Select EMA Periods",
                                [5, 9, 13, 20, 50, 100, 200],
                                default=[20, 50, 200],
                                key="ema_periods"
                            )
                        with col2:
                            st.markdown("##### SMA Periods")
                            sma_periods = st.multiselect(
                                "Select SMA Periods",
                                [5, 9, 13, 20, 50, 100, 200],
                                default=[20, 50, 200],
                                key="sma_periods"
                            )
                        
                        # Get full data for calculation
                        full_df = db.get_index_data(index_type, (start_date, end_date), selected_timeframe)
                        if full_df is not None and not full_df.empty:
                            # Calculate moving averages on full data
                            full_df = calculate_moving_averages(full_df, periods=ema_periods + sma_periods)
                            
                            # Filter for selected date range
                            mask = (full_df['datetime'].dt.date >= date_range[0]) & (full_df['datetime'].dt.date <= date_range[1])
                            df = full_df[mask].copy()
                            
                            # Plot the moving averages
                            st.markdown("#### Moving Average Chart")
                            fig = plot_technical_indicators(
                                df,
                                indicators=[f'sma_{p}' for p in sma_periods] + [f'ema_{p}' for p in ema_periods],
                                title="Price with Moving Averages"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if "MACD" in indicators:
                        st.markdown("#### MACD Settings")
                        
                        # Initialize session state for MACD parameters if not exists
                        if 'macd_params' not in st.session_state:
                            st.session_state.macd_params = [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}]
                        
                        # Display existing MACD parameters
                        for i, params in enumerate(st.session_state.macd_params):
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                            with col1:
                                st.session_state.macd_params[i]['fast_period'] = st.number_input(
                                    f"Fast Period {i+1}",
                                    min_value=1,
                                    max_value=100,
                                    value=params['fast_period'],
                                    key=f"macd_fast_{i}"
                                )
                            with col2:
                                st.session_state.macd_params[i]['slow_period'] = st.number_input(
                                    f"Slow Period {i+1}",
                                    min_value=1,
                                    max_value=100,
                                    value=params['slow_period'],
                                    key=f"macd_slow_{i}"
                                )
                            with col3:
                                st.session_state.macd_params[i]['signal_period'] = st.number_input(
                                    f"Signal Period {i+1}",
                                    min_value=1,
                                    max_value=100,
                                    value=params['signal_period'],
                                    key=f"macd_signal_{i}"
                                )
                            with col4:
                                if st.button("Remove", key=f"remove_macd_{i}"):
                                    st.session_state.macd_params.pop(i)
                                    st.rerun()
                        
                        # Add new MACD button
                        if st.button("Add MACD"):
                            st.session_state.macd_params.append({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
                            st.rerun()
                        
                        # Get full data for calculation
                        full_df = db.get_index_data(index_type, (start_date, end_date), selected_timeframe)
                        if full_df is not None and not full_df.empty:
                            # Calculate MACD for each parameter set
                            for i, params in enumerate(st.session_state.macd_params):
                                macd_data = calculate_macd(
                                    full_df,
                                    fast_period=params['fast_period'],
                                    slow_period=params['slow_period'],
                                    signal_period=params['signal_period']
                                )
                                full_df[f'macd_line_{i+1}'] = macd_data['macd_line']
                                full_df[f'macd_signal_{i+1}'] = macd_data['signal_line']
                                full_df[f'macd_hist_{i+1}'] = macd_data['histogram']
                            
                            # Filter for selected date range
                            mask = (full_df['datetime'].dt.date >= date_range[0]) & (full_df['datetime'].dt.date <= date_range[1])
                            df = full_df[mask].copy()
                            
                            # Plot the MACDs
                            st.markdown("#### MACD Chart")
                            fig = plot_technical_indicators(
                                df,
                                indicators=[f'macd_line_{i+1}' for i in range(len(st.session_state.macd_params))] +
                                         [f'macd_signal_{i+1}' for i in range(len(st.session_state.macd_params))] +
                                         [f'macd_hist_{i+1}' for i in range(len(st.session_state.macd_params))],
                                title="MACD Indicators"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if "Bollinger Bands" in indicators:
                        st.markdown("#### Bollinger Bands Settings")
                        
                        # Initialize session state for Bollinger Bands parameters if not exists
                        if 'bb_params' not in st.session_state:
                            st.session_state.bb_params = [{'period': 20, 'std_dev': 2.0}]
                        
                        # Display existing Bollinger Bands parameters
                        for i, params in enumerate(st.session_state.bb_params):
                            col1, col2, col3 = st.columns([3, 3, 1])
                            with col1:
                                st.session_state.bb_params[i]['period'] = st.number_input(
                                    f"Period {i+1}",
                                    min_value=1,
                                    max_value=100,
                                    value=params['period'],
                                    key=f"bb_period_{i}"
                                )
                            with col2:
                                st.session_state.bb_params[i]['std_dev'] = st.number_input(
                                    f"Standard Deviation {i+1}",
                                    min_value=0.1,
                                    max_value=5.0,
                                    value=params['std_dev'],
                                    step=0.1,
                                    key=f"bb_std_{i}"
                                )
                            with col3:
                                if st.button("Remove", key=f"remove_bb_{i}"):
                                    st.session_state.bb_params.pop(i)
                                    st.rerun()
                        
                        # Add new Bollinger Bands button
                        if st.button("Add Bollinger Bands"):
                            st.session_state.bb_params.append({'period': 20, 'std_dev': 2.0})
                            st.rerun()
                        
                        # Get full data for calculation
                        full_df = db.get_index_data(index_type, (start_date, end_date), selected_timeframe)
                        if full_df is not None and not full_df.empty:
                            # Calculate Bollinger Bands for each parameter set
                            for i, params in enumerate(st.session_state.bb_params):
                                bb_data = calculate_bollinger_bands(
                                    full_df,
                                    period=params['period'],
                                    std_dev=params['std_dev']
                                )
                                full_df[f'bb_middle_{i+1}'] = bb_data['bb_middle']
                                full_df[f'bb_upper_{i+1}'] = bb_data['bb_upper']
                                full_df[f'bb_lower_{i+1}'] = bb_data['bb_lower']
                                full_df[f'bb_bandwidth_{i+1}'] = bb_data['bb_bandwidth']
                                full_df[f'bb_percent_b_{i+1}'] = bb_data['bb_percent_b']
                            
                            # Filter for selected date range
                            mask = (full_df['datetime'].dt.date >= date_range[0]) & (full_df['datetime'].dt.date <= date_range[1])
                            df = full_df[mask].copy()
                            
                            # Plot the SuperTrends
                            st.markdown("#### SuperTrend Chart")
                            fig = plot_technical_indicators(
                                df,
                                indicators=[f'supertrend_{i+1}' for i in range(len(st.session_state.supertrend_params))],
                                title="Price with SuperTrends"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if "SuperTrend" in indicators:
                        st.markdown("#### SuperTrend Settings")
                        # Initialize session state for SuperTrend parameters if not exists
                        if 'supertrend_params' not in st.session_state:
                            st.session_state.supertrend_params = [{'period': 10, 'multiplier': 3.0}]
                        # Display existing SuperTrend parameters
                        for i, params in enumerate(st.session_state.supertrend_params):
                            col1, col2, col3 = st.columns([3, 3, 1])
                            with col1:
                                st.session_state.supertrend_params[i]['period'] = st.number_input(
                                    f"Period {i+1}",
                                    min_value=1,
                                    max_value=100,
                                    value=params['period'],
                                    key=f"supertrend_period_{i}"
                                )
                            with col2:
                                st.session_state.supertrend_params[i]['multiplier'] = st.number_input(
                                    f"Multiplier {i+1}",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=params['multiplier'],
                                    step=0.1,
                                    key=f"supertrend_multiplier_{i}"
                                )
                            with col3:
                                if st.button("Remove", key=f"remove_supertrend_{i}"):
                                    st.session_state.supertrend_params.pop(i)
                                    st.rerun()
                        # Add new SuperTrend button
                        if st.button("Add SuperTrend"):
                            st.session_state.supertrend_params.append({'period': 10, 'multiplier': 3.0})
                            st.rerun()
                        # Get full data for calculation
                        full_df = db.get_index_data(index_type, (start_date, end_date), selected_timeframe)
                        if full_df is not None and not full_df.empty:
                            # Calculate SuperTrend for each parameter set
                            from technical_indicator.super_trend import calculate_supertrend
                            for i, params in enumerate(st.session_state.supertrend_params):
                                st_data = calculate_supertrend(
                                    full_df,
                                    period=params['period'],
                                    multiplier=params['multiplier']
                                )
                                full_df[f'supertrend_{i+1}'] = st_data['supertrend']
                                full_df[f'supertrend_direction_{i+1}'] = st_data['direction']
                            # Filter for selected date range
                            mask = (full_df['datetime'].dt.date >= date_range[0]) & (full_df['datetime'].dt.date <= date_range[1])
                            df = full_df[mask].copy()
                            # Plot the SuperTrends
                            st.markdown("#### SuperTrend Chart")
                            fig = plot_technical_indicators(
                                df,
                                indicators=[f'supertrend_{i+1}' for i in range(len(st.session_state.supertrend_params))],
                                title="Price with SuperTrends"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw data
                    st.markdown("### Raw Data")
                    display_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Create format dictionary for numeric columns
                    format_dict = {
                        'open': '{:.2f}',
                        'high': '{:.2f}',
                        'low': '{:.2f}',
                        'close': '{:.2f}',
                        'volume': '{:,.0f}'
                    }
                    
                    # Add indicator columns if they exist
                    if "Moving Averages" in indicators:
                        # Add only the selected EMA and SMA columns
                        for period in ema_periods:
                            display_columns.append(f'ema_{period}')
                            format_dict[f'ema_{period}'] = '{:.2f}'
                        for period in sma_periods:
                            display_columns.append(f'sma_{period}')
                            format_dict[f'sma_{period}'] = '{:.2f}'
                    
                    if "MACD" in indicators:
                        # Add all MACD columns
                        for i in range(len(st.session_state.macd_params)):
                            display_columns.extend([
                                f'macd_line_{i+1}',
                                f'macd_signal_{i+1}',
                                f'macd_hist_{i+1}'
                            ])
                            format_dict.update({
                                f'macd_line_{i+1}': '{:.2f}',
                                f'macd_signal_{i+1}': '{:.2f}',
                                f'macd_hist_{i+1}': '{:.2f}'
                            })
                    
                    if "SuperTrend" in indicators:
                        # Add all SuperTrend columns
                        for i in range(len(st.session_state.supertrend_params)):
                            display_columns.extend([f'supertrend_{i+1}', f'supertrend_direction_{i+1}'])
                            format_dict.update({
                                f'supertrend_{i+1}': '{:.2f}',
                                f'supertrend_direction_{i+1}': '{:.0f}'
                            })
                    
                    # Filter columns that exist in the dataframe
                    display_columns = [col for col in display_columns if col in df.columns]
                    
                    # Display the dataframe with formatted values
                    st.dataframe(
                        df[display_columns].style.format(format_dict),
                        use_container_width=True,
                        height=400
                    )

                    # --- Backtesting Summary Section ---
                    st.markdown('## Backtesting Summary')

                    backtest_results = []
                    ema_trade_logs = {}
                    ema_rr_results = {}  # Add this line
                    macd_trade_logs = {}
                    macd_rr_results = {}  # Add this line
                    supertrend_trade_logs = {}
                    supertrend_rr_results = {}

                    # Use the filtered data for backtesting (selected date range)
                    # df is already filtered for the selected date range
                    if df is not None and not df.empty:
                        # EMA Backtests (single or crossover)
                        if "Moving Averages" in indicators:
                            if len(ema_periods) == 1:
                                period = ema_periods[0]
                                summary, trades_df = backtest_single_ema_with_trades(df, period=period)
                                summary['Strategy'] = f'EMA {period}'
                                backtest_results.append(summary)
                                ema_trade_logs[f'EMA {period}'] = trades_df
                                # Add R:R backtest for single EMA
                                rr_results = backtest_ema_rr_with_trades(df, fast=period, slow=period, rr_list=[1,2,3,4,5])
                                ema_rr_results[f'EMA {period}'] = rr_results
                            elif len(ema_periods) > 1:
                                for i, fast in enumerate(ema_periods):
                                    for j, slow in enumerate(ema_periods):
                                        if fast < slow:
                                            summary, trades_df = backtest_ema_with_trades(df, fast=fast, slow=slow)
                                            summary['Strategy'] = f'EMA {fast}:{slow}'
                                            backtest_results.append(summary)
                                            ema_trade_logs[f'EMA {fast}:{slow}'] = trades_df
                                            # Add R:R backtest for this EMA pair
                                            rr_results = backtest_ema_rr_with_trades(df, fast=fast, slow=slow, rr_list=[1,2,3,4,5])
                                            ema_rr_results[f'EMA {fast}:{slow}'] = rr_results
                        # SMA Backtests (only within SMA periods)
                        if "Moving Averages" in indicators:
                            for i, fast in enumerate(sma_periods):
                                for j, slow in enumerate(sma_periods):
                                    if fast < slow:
                                        result = backtest_sma(df, fast=fast, slow=slow)
                                        result['Strategy'] = f'SMA {fast}:{slow}'
                                        backtest_results.append(result)
                        # SuperTrend Backtests
                        if "SuperTrend" in indicators:
                            for i in range(len(st.session_state.supertrend_params)):
                                col = f'supertrend_{i+1}'
                                if col in df.columns:
                                    summary, trades_df = backtest_supertrend_with_trades(df, supertrend_col=col)
                                    summary['Strategy'] = f'SuperTrend {i+1}'
                                    backtest_results.append(summary)
                                    supertrend_trade_logs[f'SuperTrend {i+1}'] = trades_df
                                    # R:R Backtest for this SuperTrend
                                    rr_results = backtest_supertrend_rr_with_trades(df, supertrend_col=col, rr_list=[1,2,3,4,5])
                                    supertrend_rr_results[f'SuperTrend {i+1}'] = rr_results
                        # MACD Backtests
                        if "MACD" in indicators:
                            for i in range(len(st.session_state.macd_params)):
                                macd_line_col = f'macd_line_{i+1}'
                                macd_signal_col = f'macd_signal_{i+1}'
                                if macd_line_col in df.columns and macd_signal_col in df.columns:
                                    result = backtest_macd(df, macd_line_col=macd_line_col, macd_signal_col=macd_signal_col)
                                    result['Strategy'] = f'MACD {i+1}'
                                    backtest_results.append(result)
                                    # Add R:R backtest for MACD
                                    rr_results = backtest_macd_rr_with_trades(df, macd_line_col=macd_line_col, macd_signal_col=macd_signal_col, rr_list=[1,2,3,4,5])
                                    macd_rr_results[f'MACD {i+1}'] = rr_results
                        # Bollinger Bands Backtests
                        if "Bollinger Bands" in indicators:
                            for i, params in enumerate(st.session_state.bb_params):
                                bb_middle_col = f'bb_middle_{i+1}'
                                bb_upper_col = f'bb_upper_{i+1}'
                                bb_lower_col = f'bb_lower_{i+1}'
                                result = backtest_bollinger_bands(
                                    df,
                                    period=params['period'],
                                    std_dev=params['std_dev'],
                                    bb_middle_col=bb_middle_col,
                                    bb_upper_col=bb_upper_col,
                                    bb_lower_col=bb_lower_col
                                )
                                if result:
                                    backtest_results.append(result)

                    if backtest_results:
                        summary_df = pd.DataFrame(backtest_results)
                        # Format columns for display
                        summary_df = summary_df.rename(columns={"Total P&L": "Total Points"})
                        summary_df['Total Points'] = summary_df['Total Points'].apply(lambda x: f"{x:.2f}")
                        summary_df['Win Rate'] = summary_df['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
                        summary_df['Expectancy'] = summary_df['Expectancy'].apply(lambda x: f"{x:.2f}")
                        summary_df['Max DD'] = summary_df['Max DD'].apply(lambda x: f"{x:.2f}")
                        summary_df['Trades'] = summary_df['Trades'].apply(lambda x: f"{x}")
                        st.dataframe(summary_df, use_container_width=True, height=400)

                        # --- EMA Trade Log Section ---
                        if ema_trade_logs:
                            st.markdown('### EMA Trade Log')
                            selected_ema = st.selectbox(
                                'Select EMA Strategy for Trade Log',
                                list(ema_trade_logs.keys()),
                                key='ema_trade_log_select'
                            )
                            trades_df = ema_trade_logs[selected_ema]
                            if not trades_df.empty:
                                trades_df_disp = trades_df.copy()
                                trades_df_disp['Entry Time'] = pd.to_datetime(trades_df_disp['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Exit Time'] = pd.to_datetime(trades_df_disp['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Entry Price'] = trades_df_disp['Entry Price'].round(2)
                                trades_df_disp['Exit Price'] = trades_df_disp['Exit Price'].round(2)
                                trades_df_disp['P&L'] = trades_df_disp['P&L'].round(2)
                                st.dataframe(trades_df_disp, use_container_width=True, height=400)
                            else:
                                st.info('No trades for this EMA strategy.')

                        # --- EMA R:R Backtest Section ---
                        if ema_rr_results:
                            st.markdown('### EMA R:R Backtest Summary')
                            selected_ema_rr = st.selectbox(
                                'Select EMA Strategy for R:R Table',
                                list(ema_rr_results.keys()),
                                key='ema_rr_select'
                            )
                            rr_results = ema_rr_results[selected_ema_rr]
                            
                            # Build summary table
                            rr_summary_list = []
                            for rr in sorted(rr_results.keys()):
                                summary = rr_results[rr][0]
                                rr_summary_list.append({
                                    'R:R': f'1:{rr}',
                                    'Total Points': summary['Total Points'],
                                    'Trades': summary['Trades'],
                                    'Win Rate': summary['Win Rate'],
                                    'Expectancy': summary['Expectancy'],
                                    'Max DD': summary['Max DD'],
                                    'Avg Win': summary['Avg Win'],
                                    'Avg Loss': summary['Avg Loss']
                                })
                            
                            rr_summary_df = pd.DataFrame(rr_summary_list)
                            
                            # Format columns
                            rr_summary_df['Total Points'] = rr_summary_df['Total Points'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Win Rate'] = rr_summary_df['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
                            rr_summary_df['Expectancy'] = rr_summary_df['Expectancy'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Max DD'] = rr_summary_df['Max DD'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Trades'] = rr_summary_df['Trades'].apply(lambda x: f"{x}")
                            rr_summary_df['Avg Win'] = rr_summary_df['Avg Win'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Avg Loss'] = rr_summary_df['Avg Loss'].apply(lambda x: f"{x:.2f}")
                            
                            # Display summary table
                            st.dataframe(rr_summary_df, use_container_width=True, height=300)
                            
                            # Trade log for selected R:R
                            st.markdown('#### EMA R:R Trade Log')
                            selected_rr = st.selectbox(
                                'Select R:R for Trade Log',
                                [f'1:{rr}' for rr in sorted(rr_results.keys())],
                                key='ema_rr_trade_log_select'
                            )
                            rr_num = int(selected_rr.split(':')[1])
                            trades_df = rr_results[rr_num][1]
                            
                            if not trades_df.empty:
                                trades_df_disp = trades_df.copy()
                                trades_df_disp['Entry Time'] = pd.to_datetime(trades_df_disp['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Exit Time'] = pd.to_datetime(trades_df_disp['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Entry Price'] = trades_df_disp['Entry Price'].round(2)
                                trades_df_disp['Exit Price'] = trades_df_disp['Exit Price'].round(2)
                                trades_df_disp['P&L'] = trades_df_disp['P&L'].round(2)
                                trades_df_disp['Risk'] = trades_df_disp['Risk'].round(2)
                                trades_df_disp['Target'] = trades_df_disp['Target'].round(2)
                                trades_df_disp['Stop'] = trades_df_disp['Stop'].round(2)
                                
                                # Display trade log
                                st.dataframe(trades_df_disp, use_container_width=True, height=400)
                                
                                # Display exit reason distribution
                                st.markdown('#### Exit Reason Distribution')
                                exit_reasons = trades_df_disp['Exit Reason'].value_counts()
                                st.bar_chart(exit_reasons)
                            else:
                                st.info('No trades for this R:R.')
                        # --- SuperTrend Trade Log Section ---
                        if supertrend_trade_logs:
                            st.markdown('### SuperTrend Trade Log')
                            selected_st = st.selectbox(
                                'Select SuperTrend Strategy for Trade Log',
                                list(supertrend_trade_logs.keys()),
                                key='supertrend_trade_log_select'
                            )
                            trades_df = supertrend_trade_logs[selected_st]
                            if not trades_df.empty:
                                trades_df_disp = trades_df.copy()
                                trades_df_disp['Entry Time'] = pd.to_datetime(trades_df_disp['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Exit Time'] = pd.to_datetime(trades_df_disp['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Entry Price'] = trades_df_disp['Entry Price'].round(2)
                                trades_df_disp['Exit Price'] = trades_df_disp['Exit Price'].round(2)
                                trades_df_disp['P&L'] = trades_df_disp['P&L'].round(2)
                                st.dataframe(trades_df_disp, use_container_width=True, height=400)
                            else:
                                st.info('No trades for this SuperTrend strategy.')
                        # --- SuperTrend R:R Backtest Section ---
                        if supertrend_rr_results:
                            st.markdown('### SuperTrend R:R Backtest Summary')
                            selected_st_rr = st.selectbox(
                                'Select SuperTrend for R:R Table',
                                list(supertrend_rr_results.keys()),
                                key='supertrend_rr_select'
                            )
                            rr_results = supertrend_rr_results[selected_st_rr]
                            # Build summary table
                            rr_summary_list = [rr_results[rr][0] for rr in sorted(rr_results.keys())]
                            rr_summary_df = pd.DataFrame(rr_summary_list)
                            rr_summary_df['Total Points'] = rr_summary_df['Total Points'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Win Rate'] = rr_summary_df['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
                            rr_summary_df['Expectancy'] = rr_summary_df['Expectancy'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Max DD'] = rr_summary_df['Max DD'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Trades'] = rr_summary_df['Trades'].apply(lambda x: f"{x}")
                            st.dataframe(rr_summary_df[['R:R','Total Points','Trades','Win Rate','Expectancy','Max DD']], use_container_width=True, height=300)
                            # Trade log for selected R:R
                            st.markdown('#### SuperTrend R:R Trade Log')
                            selected_rr = st.selectbox(
                                'Select R:R for Trade Log',
                                [f'1:{rr}' for rr in sorted(rr_results.keys())],
                                key='supertrend_rr_trade_log_select'
                            )
                            rr_num = int(selected_rr.split(':')[1])
                            trades_df = rr_results[rr_num][1]
                            if not trades_df.empty:
                                trades_df_disp = trades_df.copy()
                                trades_df_disp['Entry Time'] = pd.to_datetime(trades_df_disp['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Exit Time'] = pd.to_datetime(trades_df_disp['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Entry Price'] = trades_df_disp['Entry Price'].round(2)
                                trades_df_disp['Exit Price'] = trades_df_disp['Exit Price'].round(2)
                                trades_df_disp['P&L'] = trades_df_disp['P&L'].round(2)
                                st.dataframe(trades_df_disp, use_container_width=True, height=400)
                            else:
                                st.info('No trades for this R:R.')
                        # --- MACD R:R Backtest Section ---
                        if macd_rr_results:
                            st.markdown('### MACD R:R Backtest Summary')
                            selected_macd_rr = st.selectbox(
                                'Select MACD for R:R Table',
                                list(macd_rr_results.keys()),
                                key='macd_rr_select'
                            )
                            rr_results = macd_rr_results[selected_macd_rr]
                            
                            # Build summary table
                            rr_summary_list = []
                            for rr in sorted(rr_results.keys()):
                                summary = rr_results[rr][0]
                                rr_summary_list.append({
                                    'R:R': f'1:{rr}',
                                    'Total Points': summary['Total Points'],
                                    'Trades': summary['Trades'],
                                    'Win Rate': summary['Win Rate'],
                                    'Expectancy': summary['Expectancy'],
                                    'Max DD': summary['Max DD'],
                                    'Avg Win': summary['Avg Win'],
                                    'Avg Loss': summary['Avg Loss']
                                })
                            
                            rr_summary_df = pd.DataFrame(rr_summary_list)
                            
                            # Format columns
                            rr_summary_df['Total Points'] = rr_summary_df['Total Points'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Win Rate'] = rr_summary_df['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
                            rr_summary_df['Expectancy'] = rr_summary_df['Expectancy'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Max DD'] = rr_summary_df['Max DD'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Trades'] = rr_summary_df['Trades'].apply(lambda x: f"{x}")
                            rr_summary_df['Avg Win'] = rr_summary_df['Avg Win'].apply(lambda x: f"{x:.2f}")
                            rr_summary_df['Avg Loss'] = rr_summary_df['Avg Loss'].apply(lambda x: f"{x:.2f}")
                            
                            # Display summary table
                            st.dataframe(rr_summary_df, use_container_width=True, height=300)
                            
                            # Trade log for selected R:R
                            st.markdown('#### MACD R:R Trade Log')
                            selected_rr = st.selectbox(
                                'Select R:R for Trade Log',
                                [f'1:{rr}' for rr in sorted(rr_results.keys())],
                                key='macd_rr_trade_log_select'
                            )
                            rr_num = int(selected_rr.split(':')[1])
                            trades_df = rr_results[rr_num][1]
                            
                            if not trades_df.empty:
                                trades_df_disp = trades_df.copy()
                                trades_df_disp['Entry Time'] = pd.to_datetime(trades_df_disp['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Exit Time'] = pd.to_datetime(trades_df_disp['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
                                trades_df_disp['Entry Price'] = trades_df_disp['Entry Price'].round(2)
                                trades_df_disp['Exit Price'] = trades_df_disp['Exit Price'].round(2)
                                trades_df_disp['P&L'] = trades_df_disp['P&L'].round(2)
                                trades_df_disp['Risk'] = trades_df_disp['Risk'].round(2)
                                trades_df_disp['Target'] = trades_df_disp['Target'].round(2)
                                trades_df_disp['Stop'] = trades_df_disp['Stop'].round(2)
                                
                                # Display trade log
                                st.dataframe(trades_df_disp, use_container_width=True, height=400)
                                
                                # Display exit reason distribution
                                st.markdown('#### Exit Reason Distribution')
                                exit_reasons = trades_df_disp['Exit Reason'].value_counts()
                                st.bar_chart(exit_reasons)
                            else:
                                st.info('No trades for this R:R.')
                    else:
                        st.info('No backtest results to display. Please select indicators and parameters.')
            else:
                st.error("No data available for the selected period.")
    elif page == "Analysis":
        show_analysis_page(db)
    elif page == "Analysis Reports":
        # Analysis page: allow user to select index, timeframe, date range, and show data/analysis
        st.markdown("# Analysis Section")
        tables = db.get_table_names()
        base_tables = [table.replace('_1min', '') for table in tables if table.endswith('_1min')]
        base_tables = sorted(list(set(base_tables)))
        if not base_tables:
            st.error("No data tables found in the database. Please check your database setup.")
        else:
            st.markdown("#### Select Index and Date Range")
            # Use columns to align Select Index, Select Date Range, and Reports horizontally
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown("<label style='font-size: 1.1rem; font-weight: 600; color: #23272f;'><span style='font-size: 1.5rem; vertical-align: middle;'>📈</span> Select Index</label>", unsafe_allow_html=True)
                index_type = st.selectbox("Select Index", base_tables, key="analysis_index")
            with col2:
                st.markdown("<label style='font-size: 1.1rem; font-weight: 600; color: #23272f;'><span style='font-size: 1.5rem; vertical-align: middle;'>📅</span> Select Date Range</label>", unsafe_allow_html=True)
                table_name = f"{index_type}_daily"
                start_date, end_date = db.get_available_dates(table_name)
                if start_date and end_date:
                    start_date = start_date.date()
                    end_date = end_date.date()
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(end_date - timedelta(days=30), end_date),
                        min_value=start_date,
                        max_value=end_date,
                        key="analysis_date_range"
                    )
            with col3:
                st.markdown("<label style='font-size: 1.1rem; font-weight: 600; color: #23272f;'><span style='font-size: 1.5rem; vertical-align: middle;'>📝</span> Reports</label>", unsafe_allow_html=True)
                selected_report = st.selectbox("Select Report", ["gap", "Inside Bar", "orb-gap", "daily-ohlc", "monthly-ohlc", "pattern", "new-high-low", "nr-ws", "nr-ws-pattern", "candle-pattern"], key="analysis_report")
            
            # Add weekday filter for all analysis
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            selected_weekdays = st.multiselect("Filter by Weekday", weekdays, default=weekdays, key="analysis_weekday_filter")
            
            # Show gap analysis section if 'gap' is selected in Reports
            if selected_report == "gap":
                st.markdown("## Gap Analysis Dashboard")
                
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # --- Ensure only daily data is used for gap analysis ---
                    if '1min' in df.columns or 'time' in df.columns or (len(df) > 0 and hasattr(df['datetime'].iloc[0], 'time') and df['datetime'].iloc[0].hour != 0):
                        st.error('Gap analysis should only be performed on daily data. Please select the daily timeframe.')
                    else:
                        # Print first few rows before gap calculation
                        st.write('Daily data preview before gap calculation:', df.head(10))
                        df = df.sort_values('datetime')
                        df['prev_close'] = df['close'].shift(1)
                        df['gap'] = ((df['open'] - df['prev_close']) / df['prev_close']) * 100
                        # Print first few rows after gap calculation
                        st.write('Daily data preview after gap calculation:', df[['datetime', 'open', 'close', 'prev_close', 'gap']].head(10))
                    
                    # Analyze gaps
                    df, gap_stats = analyze_gaps(df, date_range)
                    
                    # Display key metrics
                    total_gaps = gap_stats['total_gaps']
                    filled_gaps = gap_stats['filled_gaps']
                    unfilled_gaps = total_gaps - filled_gaps
                    up_gaps = df[df['gap_direction'] == 'Up']['gap_category'].isin(['Medium', 'High']).sum()
                    down_gaps = df[df['gap_direction'] == 'Down']['gap_category'].isin(['Medium', 'High']).sum()
                    fill_rate = gap_stats['gap_fill_rate']
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Gaps", total_gaps, f"Fill Rate: {fill_rate:.1f}%")
                    with col2:
                        st.metric("Gap Up Count", up_gaps)
                    with col3:
                        st.metric("Gap Down Count", down_gaps)
                    with col4:
                        st.metric("Total Gaps Filled", filled_gaps)
                    with col5:
                        st.metric("Total Gaps Unfilled", unfilled_gaps)
                    
                    # Gap Range Analysis Table
                    st.markdown("""
                    ### Gap Range Analysis Table
                    **Gap Ranges:**  
                    - Flat: ±0.3%  
                    - Medium: ±0.3% to ±0.7%  
                    - High: >±0.7%  
                    
                    **Table format:** `count (filled/unfilled)`
                    """)
                    gap_ranges = ['Flat', 'Medium', 'High']
                    directions = ['Up', 'Down']
                    table_data = []
                    overall = {'Direction': 'Overall'}
                    total_sum = 0
                    for rng in gap_ranges:
                        overall[rng] = 0
                    overall_filled = 0
                    overall_unfilled = 0
                    for direction in directions:
                        row = {'Direction': f'Gap {direction}'}
                        dir_gaps = df[df['gap_direction'] == direction]
                        total = 0
                        for rng in gap_ranges:
                            gaps = dir_gaps[dir_gaps['gap_category'] == rng]
                            count = len(gaps)
                            filled = gaps['gap_filled'].sum()
                            unfilled = count - filled
                            row[rng] = f"{count} ({filled}/{unfilled})"
                            # For overall row
                            if overall[rng] == 0:
                                overall[rng] = [0, 0, 0]  # [count, filled, unfilled]
                            overall[rng][0] += count
                            overall[rng][1] += filled
                            overall[rng][2] += unfilled
                            total += count
                        row['Total'] = total
                        table_data.append(row)
                        total_sum += total
                    # Build overall row
                    overall_total = 0
                    for rng in gap_ranges:
                        count, filled, unfilled = overall[rng]
                        overall[rng] = f"{count} ({filled}/{unfilled})"
                        overall_total += count
                        overall_filled += filled
                        overall_unfilled += unfilled
                    overall['Total'] = f"{overall_total} ({overall_filled}/{overall_unfilled})"
                    table_data.append(overall)
                    gap_range_df = pd.DataFrame(table_data)
                    st.dataframe(gap_range_df, use_container_width=True)
                    
                    # Display gap analysis charts
                    st.plotly_chart(plot_gap_analysis(df), use_container_width=True)
                    st.plotly_chart(plot_gap_fill_analysis(df), use_container_width=True)
                    
                    # Add new detailed gap analysis visualizations
                    st.markdown("### Detailed Gap Analysis")
                    tab1, tab2, tab3 = st.tabs(["Direction Analysis", "Pattern Analysis", "Risk-Reward Analysis"])
                    
                    with tab1:
                        direction_fig = plot_gap_direction_analysis(df)
                        if direction_fig:
                            st.plotly_chart(direction_fig, use_container_width=True)
                        
                        # Separate detailed analysis tables for gap up and gap down
                        st.markdown("### Detailed Gap Up Analysis")
                        up_gaps = df[df['gap_direction'] == 'Up'].copy()
                        if not up_gaps.empty:
                            # Calculate additional metrics for gap up
                            up_gaps['gap_size_category'] = pd.qcut(up_gaps['gap_size'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
                            up_gaps['fill_time_category'] = pd.qcut(up_gaps['days_to_fill'].fillna(999), q=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
                            
                            # Create detailed metrics table
                            up_metrics = pd.DataFrame({
                                'Metric': [
                                    'Total Gaps',
                                    'Filled Gaps',
                                    'Fill Rate',
                                    'Average Gap Size',
                                    'Average Days to Fill',
                                    'Average Return',
                                    'Win Rate',
                                    'Average Risk-Reward',
                                    'Max Adverse Excursion',
                                    'Max Favorable Excursion'
                                ],
                                'Value': [
                                    len(up_gaps),
                                    up_gaps['gap_filled'].sum(),
                                    f"{up_gaps['gap_filled'].mean()*100:.1f}%",
                                    f"{up_gaps['gap_size'].mean():.2f}%",
                                    f"{up_gaps['days_to_fill'].mean():.1f}",
                                    f"{up_gaps['fill_return'].mean():.2f}%",
                                    f"{(up_gaps['fill_return'] > 0).mean()*100:.1f}%",
                                    f"{up_gaps['risk_reward_ratio'].mean():.2f}",
                                    f"{up_gaps['max_adverse_excursion'].mean():.2f}%",
                                    f"{up_gaps['max_favorable_excursion'].mean():.2f}%"
                                ]
                            })
                            st.dataframe(up_metrics, use_container_width=True)
                            
                            # Size-based analysis
                            st.markdown("#### Gap Up Analysis by Size")
                            size_analysis = up_gaps.groupby('gap_size_category').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'days_to_fill': 'mean',
                                'fill_return': 'mean',
                                'risk_reward_ratio': 'mean'
                            }).round(3)
                            st.dataframe(size_analysis, use_container_width=True)
                            
                            # Fill time analysis
                            st.markdown("#### Gap Up Analysis by Fill Time")
                            time_analysis = up_gaps.groupby('fill_time_category').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'fill_return': 'mean',
                                'risk_reward_ratio': 'mean'
                            }).round(3)
                            st.dataframe(time_analysis, use_container_width=True)
                        
                        st.markdown("### Detailed Gap Down Analysis")
                        down_gaps = df[df['gap_direction'] == 'Down'].copy()
                        if not down_gaps.empty:
                            # Calculate additional metrics for gap down
                            down_gaps['gap_size_category'] = pd.qcut(down_gaps['gap_size'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
                            down_gaps['fill_time_category'] = pd.qcut(down_gaps['days_to_fill'].fillna(999), q=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
                            
                            # Create detailed metrics table
                            down_metrics = pd.DataFrame({
                                'Metric': [
                                    'Total Gaps',
                                    'Filled Gaps',
                                    'Fill Rate',
                                    'Average Gap Size',
                                    'Average Days to Fill',
                                    'Average Return',
                                    'Win Rate',
                                    'Average Risk-Reward',
                                    'Max Adverse Excursion',
                                    'Max Favorable Excursion'
                                ],
                                'Value': [
                                    len(down_gaps),
                                    down_gaps['gap_filled'].sum(),
                                    f"{down_gaps['gap_filled'].mean()*100:.1f}%",
                                    f"{down_gaps['gap_size'].mean():.2f}%",
                                    f"{down_gaps['days_to_fill'].mean():.1f}",
                                    f"{down_gaps['fill_return'].mean():.2f}%",
                                    f"{(down_gaps['fill_return'] > 0).mean()*100:.1f}%",
                                    f"{down_gaps['risk_reward_ratio'].mean():.2f}",
                                    f"{down_gaps['max_adverse_excursion'].mean():.2f}%",
                                    f"{down_gaps['max_favorable_excursion'].mean():.2f}%"
                                ]
                            })
                            st.dataframe(down_metrics, use_container_width=True)
                            
                            # Size-based analysis
                            st.markdown("#### Gap Down Analysis by Size")
                            size_analysis = down_gaps.groupby('gap_size_category').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'days_to_fill': 'mean',
                                'fill_return': 'mean',
                                'risk_reward_ratio': 'mean'
                            }).round(3)
                            st.dataframe(size_analysis, use_container_width=True)
                            
                            # Fill time analysis
                            st.markdown("#### Gap Down Analysis by Fill Time")
                            time_analysis = down_gaps.groupby('fill_time_category').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'fill_return': 'mean',
                                'risk_reward_ratio': 'mean'
                            }).round(3)
                            st.dataframe(time_analysis, use_container_width=True)
                        
                        # Add comparison table
                        st.markdown("### Gap Up vs Gap Down Comparison")
                        comparison_data = {
                            'Metric': [
                                'Total Gaps',
                                'Fill Rate',
                                'Average Gap Size',
                                'Average Days to Fill',
                                'Average Return',
                                'Win Rate',
                                'Risk-Reward Ratio',
                                'Max Adverse Excursion',
                                'Max Favorable Excursion'
                            ],
                            'Gap Up': [
                                len(up_gaps),
                                f"{up_gaps['gap_filled'].mean()*100:.1f}%",
                                f"{up_gaps['gap_size'].mean():.2f}%",
                                f"{up_gaps['days_to_fill'].mean():.1f}",
                                f"{up_gaps['fill_return'].mean():.2f}%",
                                f"{(up_gaps['fill_return'] > 0).mean()*100:.1f}%",
                                f"{up_gaps['risk_reward_ratio'].mean():.2f}",
                                f"{up_gaps['max_adverse_excursion'].mean():.2f}%",
                                f"{up_gaps['max_favorable_excursion'].mean():.2f}%"
                            ],
                            'Gap Down': [
                                len(down_gaps),
                                f"{down_gaps['gap_filled'].mean()*100:.1f}%",
                                f"{down_gaps['gap_size'].mean():.2f}%",
                                f"{down_gaps['days_to_fill'].mean():.1f}",
                                f"{down_gaps['fill_return'].mean():.2f}%",
                                f"{(down_gaps['fill_return'] > 0).mean()*100:.1f}%",
                                f"{down_gaps['risk_reward_ratio'].mean():.2f}",
                                f"{down_gaps['max_adverse_excursion'].mean():.2f}%",
                                f"{down_gaps['max_favorable_excursion'].mean():.2f}%"
                            ]
                        }
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    with tab2:
                        pattern_fig = plot_gap_patterns(df)
                        if pattern_fig:
                            st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # Pattern statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Temporal Patterns")
                            weekday_stats = df.groupby('weekday').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'fill_return': 'mean'
                            }).round(3)
                            st.dataframe(weekday_stats)
                        
                        with col2:
                            st.subheader("Consecutive Gaps")
                            consecutive_stats = df.groupby('consecutive').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'fill_return': 'mean'
                            }).round(3)
                            st.dataframe(consecutive_stats)
                    
                    with tab3:
                        risk_fig = plot_gap_risk_reward(df)
                        if risk_fig:
                            st.plotly_chart(risk_fig, use_container_width=True)
                        
                        # Risk metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Risk Metrics")
                            st.metric("Avg Max Adverse Excursion", f"{df['max_adverse_excursion'].mean():.2f}%")
                            st.metric("Avg Max Favorable Excursion", f"{df['max_favorable_excursion'].mean():.2f}%")
                            st.metric("Win Rate", f"{(df['fill_return'] > 0).mean()*100:.1f}%")
                        
                        with col2:
                            st.subheader("Gap Size Analysis")
                            size_stats = df.groupby('gap_size_bin').agg({
                                'gap': 'count',
                                'gap_filled': 'mean',
                                'fill_return': 'mean',
                                'risk_reward_ratio': 'mean'
                            }).round(3)
                            st.dataframe(size_stats)
                    
                    # Display gap summary table
                    st.markdown("### Gap Summary")
                    summary_df = get_gap_summary(df)
                    if not summary_df.empty:
                        st.dataframe(
                            summary_df.style.background_gradient(subset=['fill_rate'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                    
                    # Display detailed gap history
                    st.markdown("### Gap History")
                    gap_history = df[df['gap_category'].isin(['Medium', 'High'])][
                        ['datetime', 'open', 'close', 'gap', 'gap_category', 'gap_direction', 'gap_filled', 'days_to_fill']
                    ].copy()
                    
                    if not gap_history.empty:
                        gap_history['datetime'] = gap_history['datetime'].dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            gap_history.style.format({
                                'gap': '{:.2f}%',
                                'open': '{:.2f}',
                                'close': '{:.2f}',
                                'days_to_fill': lambda x: f"{x:.0f}" if pd.notnull(x) else '-',
                                'gap_filled': lambda x: 'Yes' if x is True else ('No' if x is False else '-')
                            }).background_gradient(subset=['gap'], cmap='RdYlGn'),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("No significant gaps found in the selected period.")
            elif selected_report == "Inside Bar":
                show_inside_bar_analysis(db, index_type, date_range)
            elif selected_report == "orb-gap":
                st.markdown("## ORB-Gap Analysis Setup")
                
                # 1. Select ORB minutes range first
                selected_timeframe = st.selectbox(
                    "Select Timeframe",
                    ["1 Minute", "5 Minute", "15 Minute"],
                    key="orb_gap_timeframe"
                )
                if selected_timeframe == "1 Minute":
                    range_minutes = st.slider("Select ORB Range Minutes", min_value=5, max_value=60, value=5, step=5)
                elif selected_timeframe == "5 Minute":
                    range_minutes = st.slider("Select ORB Range Minutes", min_value=5, max_value=60, value=15, step=5)
                else:  # 15 Minute
                    range_minutes = st.slider("Select ORB Range Minutes", min_value=15, max_value=60, value=30, step=15)
                
                # 2. Select Gap Type
                gap_type = st.radio(
                    "Select Gap Type",
                    ["Gap Up", "Gap Down", "Flat Open"],
                    index=0,
                    key="orb_gap_type"
                )
                
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df[(df['datetime'].dt.date >= date_range[0]) & (df['datetime'].dt.date <= date_range[1])]
                    df['date'] = df['datetime'].dt.date
                    df['time'] = df['datetime'].dt.time
                    df['weekday'] = df['datetime'].dt.day_name()
                    df = df[df['weekday'].isin(selected_weekdays)]
                    # Calculate gaps
                    df = df.sort_values('datetime')
                    df['prev_close'] = df['close'].shift(1)
                    df['gap'] = ((df['open'] - df['prev_close']) / df['prev_close']) * 100
                else:
                    df = pd.DataFrame()
                
                # 3. Select Gap Range (%)
                # Custom bins for gap up and gap down
                gap_up_bins = [
                    (0.0, 0.3, '0.0% to 0.3%'),
                    (0.3, 0.5, '0.3% to 0.5%'),
                    (0.5, 0.7, '0.5% to 0.7%'),
                    (1.0, float('inf'), 'Above 1%')
                ]
                gap_down_bins = [
                    (-0.0, -0.3, '-0.0% to -0.3%'),
                    (-0.3, -0.5, '-0.3% to -0.5%'),
                    (-0.5, -0.7, '-0.5% to -0.7%'),
                    (-1.0, -float('inf'), 'Below -1%')
                ]
                gap_range_label = None
                if gap_type == "Gap Up":
                    gap_range_options = [b[2] for b in gap_up_bins]
                    gap_range_label = st.selectbox("Select Gap Range (%)", gap_range_options, key="orb_gap_range")
                    # Map label to bin
                    selected_bin = next(b for b in gap_up_bins if b[2] == gap_range_label)
                    gap_filter = (df['gap'] >= selected_bin[0]) & (df['gap'] < selected_bin[1]) if selected_bin[1] != float('inf') else (df['gap'] >= selected_bin[0])
                elif gap_type == "Gap Down":
                    gap_range_options = [b[2] for b in gap_down_bins]
                    gap_range_label = st.selectbox("Select Gap Range (%)", gap_range_options, key="orb_gap_range")
                    # Map label to bin
                    selected_bin = next(b for b in gap_down_bins if b[2] == gap_range_label)
                    gap_filter = (df['gap'] <= selected_bin[0]) & (df['gap'] > selected_bin[1]) if selected_bin[1] != -float('inf') else (df['gap'] <= selected_bin[0])
                else:  # Flat Open
                    gap_range_label = 'Flat (|gap| < 0.3%)'
                    gap_filter = (df['gap'] > -0.3) & (df['gap'] < 0.3)
                # Filter by gap type and range
                filtered_df = df[gap_filter]
                # Show filtered days below selection
                st.markdown(f"**Filtered Days: {len(filtered_df)}**")
                if not filtered_df.empty:
                    st.dataframe(filtered_df[['date', 'open', 'high', 'low', 'close', 'gap']].reset_index(drop=True), use_container_width=True, height=200)

                    # Compute orb_results_df (ORB analysis for filtered days)
                    timeframe_map = {
                        "Daily": "daily",
                        "1 Hour": "1hour",
                        "15 Minute": "15min",
                        "5 Minute": "5min",
                        "1 Minute": "1min"
                    }
                    db_timeframe = timeframe_map[selected_timeframe]
                    orb_df = db.get_index_data(index_type, date_range, db_timeframe)
                    orb_results_df = pd.DataFrame()
                    if orb_df is not None and not orb_df.empty:
                        orb_df['datetime'] = pd.to_datetime(orb_df['datetime'])
                        orb_df['date'] = orb_df['datetime'].dt.date
                        orb_df['time'] = orb_df['datetime'].dt.time
                        filtered_dates = set(filtered_df['date'])
                        orb_df = orb_df[orb_df['date'].isin(filtered_dates)]
                        market_start = time(9, 15)
                        orb_results = []
                        for day, day_df in orb_df.groupby('date'):
                            day_df = day_df.sort_values('datetime')
                            orb_window = day_df[day_df['time'] >= market_start].head(range_minutes)
                            if orb_window.empty or len(orb_window) < range_minutes:
                                continue
                            orb_high = orb_window['high'].max()
                            orb_low = orb_window['low'].min()
                            after_orb = day_df[day_df['datetime'] > orb_window['datetime'].max()]
                            first_break = None
                            first_break_time = None
                            for _, row in after_orb.iterrows():
                                if row['high'] > orb_high:
                                    first_break = 'Up'
                                    first_break_time = row['datetime']
                                    break
                                elif row['low'] < orb_low:
                                    first_break = 'Down'
                                    first_break_time = row['datetime']
                                    break
                            close_price = day_df['close'].iloc[-1]
                            if first_break is None:
                                orb_results.append({
                                    'date': day,
                                    'orb_high': orb_high,
                                    'orb_low': orb_low,
                                    'first_break': 'No Break',
                                    'first_break_time': None,
                                    'close_price': close_price
                                })
                            else:
                                orb_results.append({
                                    'date': day,
                                    'orb_high': orb_high,
                                    'orb_low': orb_low,
                                    'first_break': first_break,
                                    'first_break_time': first_break_time,
                                    'close_price': close_price
                                })
                        orb_results_df = pd.DataFrame(orb_results)

                        # --- ORB Range % Summary Table (First Break Up) ---
                        orb_range_bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
                        orb_range_bin_labels = ['0-0.2%', '0.2-0.4%', '0.4-0.6%', '0.6-0.8%', '>0.8%']
                        # --- High Break Pivot Table ---
                        high_breaks = orb_results_df[orb_results_df['first_break'] == 'Up']
                        high_made_bins = [0, 0.2, 0.5, 1.0, float('inf')]
                        high_made_bin_labels = ['0-0.2%', '0.2-0.5%', '0.5-1%', '>1%']
                        high_pivot_data = {rng: {hmb: 0 for hmb in high_made_bin_labels} for rng in orb_range_bin_labels}
                        for _, row in high_breaks.iterrows():
                            day_df = orb_df[orb_df['date'] == row['date']]
                            after_break = day_df[day_df['datetime'] > row['first_break_time']] if pd.notnull(row['first_break_time']) else pd.DataFrame()
                            if not after_break.empty:
                                max_high = after_break['high'].max()
                                high_made = ((max_high - row['orb_high']) / row['orb_high'] * 100) if row['orb_high'] else 0
                            else:
                                high_made = 0
                            orb_range_pct = abs((row['orb_high'] - row['orb_low']) / row['orb_high'] * 100) if row['orb_high'] else 0
                            orb_range_label = None
                            for i in range(len(orb_range_bins)-1):
                                if orb_range_bins[i] <= orb_range_pct < orb_range_bins[i+1]:
                                    orb_range_label = orb_range_bin_labels[i]
                                    break
                            high_made_label = None
                            for i in range(len(high_made_bins)-1):
                                if high_made_bins[i] <= high_made < high_made_bins[i+1]:
                                    high_made_label = high_made_bin_labels[i]
                                    break
                            if orb_range_label and high_made_label:
                                high_pivot_data[orb_range_label][high_made_label] += 1
                        high_pivot_df = pd.DataFrame(high_pivot_data).T
                        high_pivot_df.index.name = 'ORB Candle Range %'

                        # --- Low Break Pivot Table ---
                        low_breaks = orb_results_df[orb_results_df['first_break'] == 'Down']
                        low_made_bins = [0, -0.2, -0.4, -0.6, -0.8, -1.0, float('-inf')]
                        low_made_bin_labels = ['0 to -0.2%', '-0.2 to -0.4%', '-0.4 to -0.6%', '-0.6 to -0.8%', '-0.8 to -1%', 'Below -1%']
                        low_pivot_data = {rng: {lmb: 0 for lmb in low_made_bin_labels} for rng in orb_range_bin_labels}
                        for _, row in low_breaks.iterrows():
                            day_df = orb_df[orb_df['date'] == row['date']]
                            after_break = day_df[day_df['datetime'] > row['first_break_time']] if pd.notnull(row['first_break_time']) else pd.DataFrame()
                            if not after_break.empty:
                                min_low = after_break['low'].min()
                                low_made = ((min_low - row['orb_low']) / row['orb_low'] * 100) if row['orb_low'] else None
                                orb_range_pct = abs((row['orb_high'] - row['orb_low']) / row['orb_high'] * 100) if row['orb_high'] else 0
                                orb_range_label = None
                                for i in range(len(orb_range_bins)-1):
                                    if orb_range_bins[i] <= orb_range_pct < orb_range_bins[i+1]:
                                        orb_range_label = orb_range_bin_labels[i]
                                        break
                                low_made_label = None
                                for i in range(len(low_made_bins)-1):
                                    if low_made_bins[i] >= low_made > low_made_bins[i+1]:
                                        low_made_label = low_made_bin_labels[i]
                                        break
                                if orb_range_label and low_made_label:
                                    low_pivot_data[orb_range_label][low_made_label] += 1
                        low_pivot_df = pd.DataFrame(low_pivot_data).T
                        low_pivot_df.index.name = 'ORB Candle Range %'

                    # 1. ORB Analysis Results (gauge charts)
                    total_days = len(orb_results_df)
                    up_only = (orb_results_df['first_break'] == 'Up').sum()
                    down_only = (orb_results_df['first_break'] == 'Down').sum()
                    none = (orb_results_df['first_break'] == 'No Break').sum()
                    up_pct = (up_only / total_days * 100) if total_days > 0 else 0
                    down_pct = (down_only / total_days * 100) if total_days > 0 else 0
                    none_pct = (none / total_days * 100) if total_days > 0 else 0
                    import plotly.graph_objects as go
                    st.markdown('### ORB Analysis Results')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_up = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = up_pct,
                            title = {'text': "Up First %"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': 'green'}}
                        ))
                        st.plotly_chart(fig_up, use_container_width=True)
                    with col2:
                        fig_down = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = down_pct,
                            title = {'text': "Down First %"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': 'green'}}
                        ))
                        st.plotly_chart(fig_down, use_container_width=True)
                    with col3:
                        fig_none = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = none_pct,
                            title = {'text': "No Break %"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': 'green'}}
                        ))
                        st.plotly_chart(fig_none, use_container_width=True)

                    # 2. ORB Analysis for Filtered Days (table)
                    st.markdown("### ORB Analysis for Filtered Days")
                    st.dataframe(orb_results_df, use_container_width=True)

                    # Add pie charts for Up First and Down First analysis
                    st.markdown("### Close Price Distribution Relative to ORB")
                    
                    # Create two columns for the pie charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Up First Break: Close Distribution")
                        # Filter for Up First breaks
                        up_breaks = orb_results_df[orb_results_df['first_break'] == 'Up']
                        if not up_breaks.empty:
                            import numpy as np  # Ensure np is available in this scope
                            # Calculate close positions relative to ORB
                            up_breaks['close_position'] = np.where(
                                up_breaks['close_price'] > up_breaks['orb_high'],
                                'Above ORB',
                                np.where(
                                    up_breaks['close_price'] < up_breaks['orb_low'],
                                    'Below ORB',
                                    'In ORB'
                                )
                            )
                            
                            # Calculate percentages
                            up_close_dist = up_breaks['close_position'].value_counts(normalize=True) * 100
                            
                            # Create pie chart
                            fig_up = go.Figure(data=[go.Pie(
                                labels=up_close_dist.index,
                                values=up_close_dist.values,
                                hole=.3,
                                title='Up First Break Close Distribution'
                            )])
                            fig_up.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_up, use_container_width=True)
                        else:
                            st.info("No Up First breaks found in the selected period.")
                    
                    with col2:
                        st.subheader("Down First Break: Close Distribution")
                        # Filter for Down First breaks
                        down_breaks = orb_results_df[orb_results_df['first_break'] == 'Down']
                        if not down_breaks.empty:
                            import numpy as np  # Ensure np is available in this scope
                            # Calculate close positions relative to ORB
                            down_breaks['close_position'] = np.where(
                                down_breaks['close_price'] > down_breaks['orb_high'],
                                'Above ORB',
                                np.where(
                                    down_breaks['close_price'] < down_breaks['orb_low'],
                                    'Below ORB',
                                    'In ORB'
                                )
                            )
                            
                            # Calculate percentages
                            down_close_dist = down_breaks['close_position'].value_counts(normalize=True) * 100
                            
                            # Create pie chart
                            fig_down = go.Figure(data=[go.Pie(
                                labels=down_close_dist.index,
                                values=down_close_dist.values,
                                hole=.3,
                                title='Down First Break Close Distribution'
                            )])
                            fig_down.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_down, use_container_width=True)
                        else:
                            st.info("No Down First breaks found in the selected period.")

                    # 3. ORB Range % Summary Table (First Break Up)
                    st.markdown('#### ORB Range % Summary Table (First Break Up)')
                    # Build new summary table for Up First
                    orb_range_bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
                    orb_range_bin_labels = ['0-0.2%', '0.2-0.4%', '0.4-0.6%', '0.6-0.8%', '>0.8%']
                    up_breaks = orb_results_df[orb_results_df['first_break'] == 'Up']
                    summary_data_up = []
                    for i in range(len(orb_range_bins)-1):
                        label = orb_range_bin_labels[i]
                        def orb_range_pct(row):
                            if row['orb_high'] and row['orb_low']:
                                return abs((row['orb_high'] - row['orb_low']) / row['orb_high'] * 100)
                            return 0
                        bin_df = up_breaks[up_breaks.apply(lambda row: orb_range_bins[i] <= orb_range_pct(row) < orb_range_bins[i+1], axis=1)]
                        total = len(bin_df)
                        close_above = (bin_df['close_price'] > bin_df['orb_high']).sum()
                        close_below = (bin_df['close_price'] < bin_df['orb_low']).sum()
                        in_between = ((bin_df['close_price'] <= bin_df['orb_high']) & (bin_df['close_price'] >= bin_df['orb_low'])).sum()
                        def fmt(val):
                            pct = (val / total * 100) if total > 0 else 0
                            return f"{val} ({pct:.1f}%)"
                        summary_data_up.append({
                            'ORB Candle Range %': label,
                            'Close Above ORB': fmt(close_above),
                            'Close Below ORB': fmt(close_below),
                            'In Between ORB': fmt(in_between),
                            'Total': total
                        })
                    summary_table_up = pd.DataFrame(summary_data_up)
                    st.dataframe(summary_table_up, use_container_width=True)

                    # 5. ORB Range % Summary Table (First Break Down)
                    st.markdown('#### ORB Range % Summary Table (First Break Down)')
                    # Build new summary table for Down First
                    down_breaks = orb_results_df[orb_results_df['first_break'] == 'Down']
                    summary_data_down = []
                    for i in range(len(orb_range_bins)-1):
                        label = orb_range_bin_labels[i]
                        bin_df = down_breaks[down_breaks.apply(lambda row: orb_range_bins[i] <= orb_range_pct(row) < orb_range_bins[i+1], axis=1)]
                        total = len(bin_df)
                        close_above = (bin_df['close_price'] > bin_df['orb_high']).sum()
                        close_below = (bin_df['close_price'] < bin_df['orb_low']).sum()
                        in_between = ((bin_df['close_price'] <= bin_df['orb_high']) & (bin_df['close_price'] >= bin_df['orb_low'])).sum()
                        def fmt(val):
                            pct = (val / total * 100) if total > 0 else 0
                            return f"{val} ({pct:.1f}%)"
                        summary_data_down.append({
                            'ORB Candle Range %': label,
                            'Close Above ORB': fmt(close_above),
                            'Close Below ORB': fmt(close_below),
                            'In Between ORB': fmt(in_between),
                            'Total': total
                        })
                    summary_table_down = pd.DataFrame(summary_data_down)
                    st.dataframe(summary_table_down, use_container_width=True)

                    # 7. High Break Pivot Table
                    st.markdown('#### High Break Pivot Table (Rows: ORB Candle Range %, Columns: High Made After Break %)')
                    st.dataframe(high_pivot_df, use_container_width=True)

                    # 8. Low Break Pivot Table
                    st.markdown('#### Low Break Pivot Table (Rows: ORB Candle Range %, Columns: Low Made After Break %)')
                    st.dataframe(low_pivot_df, use_container_width=True)

                    # 9. Low Break Coil Analysis Table
                    st.markdown('#### Low Break Close Analysis (Where Price Closed After Making Low)')
                    low_breaks = orb_results_df[orb_results_df['first_break'] == 'Down']
                    low_made_bins = [0, -0.2, -0.4, -0.6, -0.8, -1.0, float('-inf')]
                    low_made_bin_labels = ['0 to -0.2%', '-0.2 to -0.4%', '-0.4 to -0.6%', '-0.6 to -0.8%', '-0.8 to -1%', 'Below -1%']
                    coil_positions = ['Above ORB', 'Below ORB', 'In ORB']
                    # Initialize the coil analysis data structure
                    coil_data = {rng: {pos: {'count': 0, 'total': 0} for pos in coil_positions} for rng in low_made_bin_labels}
                    for _, row in low_breaks.iterrows():
                        day_df = orb_df[orb_df['date'] == row['date']]
                        after_break = day_df[day_df['datetime'] > row['first_break_time']] if pd.notnull(row['first_break_time']) else pd.DataFrame()
                        if not after_break.empty:
                            min_low = after_break['low'].min()
                            # FIX: Corrected calculation for low_made
                            low_made = ((min_low - row['orb_low']) / row['orb_low'] * 100) if row['orb_low'] else None
                            last_price = after_break['close'].iloc[-1]
                            coil_position = 'Above ORB' if last_price > row['orb_high'] else ('Below ORB' if last_price < row['orb_low'] else 'In ORB')
                            for i in range(len(low_made_bins)-1):
                                if low_made_bins[i] >= low_made > low_made_bins[i+1]:
                                    bin_label = low_made_bin_labels[i]
                                    coil_data[bin_label][coil_position]['count'] += 1
                                    coil_data[bin_label][coil_position]['total'] += 1
                                    break
                    coil_table_data = []
                    for rng in low_made_bin_labels:
                        row_data = {'Low Made Range': rng}
                        total_cases = sum(coil_data[rng][pos]['total'] for pos in coil_positions)
                        for pos in coil_positions:
                            count = coil_data[rng][pos]['count']
                            percentage = (count / total_cases * 100) if total_cases > 0 else 0
                            row_data[pos] = f"{count} ({percentage:.1f}%)"
                        row_data['Total Cases'] = total_cases
                        coil_table_data.append(row_data)
                    coil_table_df = pd.DataFrame(coil_table_data)
                    st.dataframe(coil_table_df, use_container_width=True)

                    # 10. High Break Coil Analysis Table
                    st.markdown('#### High Break Close Analysis (Where Price Closed After Making High)')
                    high_breaks = orb_results_df[orb_results_df['first_break'] == 'Up']
                    high_made_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
                    high_made_bin_labels = ['0 to 0.2%', '0.2 to 0.4%', '0.4 to 0.6%', '0.6 to 0.8%', '0.8 to 1%', 'Above 1%']
                    
                    # Initialize the coil analysis data structure
                    coil_data_high = {rng: {pos: {'count': 0, 'total': 0} for pos in coil_positions} for rng in high_made_bin_labels}
                    
                    for _, row in high_breaks.iterrows():
                        day_df = orb_df[orb_df['date'] == row['date']]
                        after_break = day_df[day_df['datetime'] > row['first_break_time']] if pd.notnull(row['first_break_time']) else pd.DataFrame()
                        
                        if not after_break.empty:
                            max_high = after_break['high'].max()
                            high_made = ((max_high - row['orb_high']) / row['orb_high'] * 100) if row['orb_high'] else 0
                            
                            # Determine where price coiled (last position)
                            last_price = after_break['close'].iloc[-1]
                            coil_position = 'Above ORB' if last_price > row['orb_high'] else ('Below ORB' if last_price < row['orb_low'] else 'In ORB')
                            
                            # Find the appropriate bin for the high made
                            for i in range(len(high_made_bins)-1):
                                if high_made_bins[i] <= high_made < high_made_bins[i+1]:
                                    bin_label = high_made_bin_labels[i]
                                    coil_data_high[bin_label][coil_position]['count'] += 1
                                    coil_data_high[bin_label][coil_position]['total'] += 1
                                    break
                    
                    # Create the final table
                    coil_table_data_high = []
                    for rng in high_made_bin_labels:
                        row_data = {'High Made Range': rng}
                        total_cases = sum(coil_data_high[rng][pos]['total'] for pos in coil_positions)
                        
                        for pos in coil_positions:
                            count = coil_data_high[rng][pos]['count']
                            percentage = (count / total_cases * 100) if total_cases > 0 else 0
                            row_data[pos] = f"{count} ({percentage:.1f}%)"
                        
                        row_data['Total Cases'] = total_cases
                        coil_table_data_high.append(row_data)
                    
                    coil_table_df_high = pd.DataFrame(coil_table_data_high)
                    st.dataframe(coil_table_df_high, use_container_width=True)

                    # --- Combined Matrix Table for Transition, Correlation, Covariance ---
                    st.markdown("### Day-to-Day Transition, Correlation, and Covariance Matrix")
                    matrix_data = []
                    for i, from_day in enumerate(weekdays):
                        row = {'From/To': from_day}
                        for j, to_day in enumerate(weekdays):
                            if j == i + 1 or (from_day == 'Monday' and to_day == 'Friday'):  # Consecutive or Monday to Friday
                                # Prepare data for consecutive days or Monday to Friday
                                df_from = df[df['weekday'] == from_day].copy()
                                df_to = df[df['weekday'] == to_day].copy()
                                df_from = df_from.sort_values('datetime').reset_index(drop=True)
                                df_to = df_to.sort_values('datetime').reset_index(drop=True)
                                df_from['week'] = pd.to_datetime(df_from['datetime']).dt.isocalendar().week
                                df_from['year'] = pd.to_datetime(df_from['datetime']).dt.isocalendar().year
                                df_to['week'] = pd.to_datetime(df_to['datetime']).dt.isocalendar().week
                                df_to['year'] = pd.to_datetime(df_to['datetime']).dt.isocalendar().year
                                merged = pd.merge(df_from, df_to, on=['week', 'year'], suffixes=(f'_{from_day}', f'_{to_day}'))
                                # Transition probability
                                cond = merged[f'return_{from_day}'] > 0
                                if cond.sum() > 0:
                                    prob = (merged.loc[cond, f'return_{to_day}'] > 0).mean()
                                else:
                                    prob = float('nan')
                                # Correlation and covariance
                                if len(merged) > 1:
                                    corr = merged[f'return_{from_day}'].corr(merged[f'return_{to_day}'])
                                    covar = merged[f'return_{from_day}'].cov(merged[f'return_{to_day}'])
                                else:
                                    corr = float('nan')
                                    covar = float('nan')
                                # Format cell
                                cell = f"P(+|+): {prob*100:.1f}%\nCorr: {corr:.2f}\nCov: {covar:.4f}" if not pd.isna(prob) else "N/A"
                            else:
                                cell = "N/A"
                            row[to_day] = cell
                        matrix_data.append(row)
                    matrix_df = pd.DataFrame(matrix_data)
                    st.dataframe(matrix_df, use_container_width=True)

                    # Monday to Friday
                    day1 = 'Monday'
                    day2 = 'Friday'
                    df_day1 = df[df['weekday'] == day1].copy()
                    df_day2 = df[df['weekday'] == day2].copy()
                    df_day1 = df_day1.sort_values('datetime').reset_index(drop=True)
                    df_day2 = df_day2.sort_values('datetime').reset_index(drop=True)
                    df_day1['week'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().week
                    df_day1['year'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().year
                    df_day2['week'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().week
                    df_day2['year'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().year
                    merged = pd.merge(df_day1, df_day2, on=['week', 'year'], suffixes=(f'_{day1}', f'_{day2}'))
                    if len(merged) > 1:
                        corr = merged[f'return_{day1}'].corr(merged[f'return_{day2}'])
                        covar = merged[f'return_{day1}'].cov(merged[f'return_{day2}'])
                    else:
                        corr = float('nan')
                        covar = float('nan')
                    corr_data.append({
                        'From': day1,
                        'To': day2,
                        'Correlation': f"{corr:.2f}" if not pd.isna(corr) else '-',
                        'Covariance': f"{covar:.4f}" if not pd.isna(covar) else '-'
                    })
                    # Friday to next Monday
                    day1 = 'Friday'
                    day2 = 'Monday'
                    df_friday = df[df['weekday'] == day1].copy()
                    df_monday = df[df['weekday'] == day2].copy()
                    df_friday = df_friday.sort_values('datetime').reset_index(drop=True)
                    df_monday = df_monday.sort_values('datetime').reset_index(drop=True)
                    # For each Friday, find the next Monday
                    friday_dates = pd.to_datetime(df_friday['datetime']).dt.date.values
                    monday_dates = pd.to_datetime(df_monday['datetime']).dt.date.values
                    friday_to_next_monday = []
                    for f_date in friday_dates:
                        next_monday = min([m for m in monday_dates if m > f_date], default=None)
                        if next_monday:
                            friday_row = df_friday[pd.to_datetime(df_friday['datetime']).dt.date == f_date]
                            monday_row = df_monday[pd.to_datetime(df_monday['datetime']).dt.date == next_monday]
                            if not friday_row.empty and not monday_row.empty:
                                friday_to_next_monday.append({
                                    'friday_return': friday_row['return'].values[0],
                                    'monday_return': monday_row['return'].values[0]
                                })
                    if len(friday_to_next_monday) > 1:
                        friday_returns = [x['friday_return'] for x in friday_to_next_monday]
                        monday_returns = [x['monday_return'] for x in friday_to_next_monday]
                        corr = np.corrcoef(friday_returns, monday_returns)[0, 1]
                        covar = np.cov(friday_returns, monday_returns)[0, 1]
                    else:
                        corr = float('nan')
                        covar = float('nan')
                    corr_data.append({
                        'From': 'Friday',
                        'To': 'Monday (next week)',
                        'Correlation': f"{corr:.2f}" if not pd.isna(corr) else '-',
                        'Covariance': f"{covar:.4f}" if not pd.isna(covar) else '-'
                    })
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)

                    # --- Day Pair Sign Table ---
                    st.markdown("""
                    ### Day-to-Day Sign Combination Table (A/B/C/D)
                    **Legend:**
                    - **A**: Both days +ve (close > open)
                    - **B**: Both days -ve (close < open) "
                    - **C**: Row day +ve, Col day -ve
                    - **D**: Row day -ve, Col day +ve
                    
                    Each cell shows: `A/B/C/D`
                    """)
                    try:
                        sign_matrix = []
                        for row_day in weekdays:
                            row = {'Day': row_day}
                            for col_day in weekdays:
                                if row_day == col_day:
                                    row[col_day] = ''
                                else:
                                    # Align by week and year
                                    df_row = df[df['weekday'] == row_day].copy()
                                    df_col = df[df['weekday'] == col_day].copy()
                                    df_row = df_row.sort_values('datetime').reset_index(drop=True)
                                    df_col = df_col.sort_values('datetime').reset_index(drop=True)
                                    df_row['week'] = pd.to_datetime(df_row['datetime']).dt.isocalendar().week
                                    df_row['year'] = pd.to_datetime(df_row['datetime']).dt.isocalendar().year
                                    df_col['week'] = pd.to_datetime(df_col['datetime']).dt.isocalendar().week
                                    df_col['year'] = pd.to_datetime(df_col['datetime']).dt.isocalendar().year
                                    merged = pd.merge(df_row, df_col, on=['week', 'year'], suffixes=(f'_{row_day}', f'_{col_day}'))
                                    # Only keep pairs where col_day is after row_day in the week
                                    merged = merged[pd.to_datetime(merged[f'datetime_{col_day}']) > pd.to_datetime(merged[f'datetime_{row_day}'])]
                                    total_pairs = len(merged)
                                    # A: both +ve
                                    both_pos = ((merged[f'close_{row_day}'] > merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] > merged[f'open_{col_day}'])).sum()
                                    # B: both -ve
                                    both_neg = ((merged[f'close_{row_day}'] < merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] < merged[f'open_{col_day}'])).sum()
                                    # C: row +ve, col -ve
                                    row_pos_col_neg = ((merged[f'close_{row_day}'] > merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] < merged[f'open_{col_day}'])).sum()
                                    # D: row -ve, col +ve
                                    row_neg_col_pos = ((merged[f'close_{row_day}'] < merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] > merged[f'open_{col_day}'])).sum()
                                    # Calculate percentages
                                    both_pos_pct = round(both_pos / total_pairs * 100) if total_pairs > 0 else 0
                                    both_neg_pct = round(both_neg / total_pairs * 100) if total_pairs > 0 else 0
                                    row_pos_col_neg_pct = round(row_pos_col_neg / total_pairs * 100) if total_pairs > 0 else 0
                                    row_neg_col_pos_pct = round(row_neg_col_pos / total_pairs * 100) if total_pairs > 0 else 0
                                    row[col_day] = f"{both_pos_pct}%/{both_neg_pct}%/{row_pos_col_neg_pct}%/{row_neg_col_pos_pct}%"
                            sign_matrix.append(row)
                        sign_matrix_df = pd.DataFrame(sign_matrix)
                        if sign_matrix_df.empty:
                            st.warning("No data available for the sign combination table.")
                        else:
                            st.dataframe(
                                sign_matrix_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                                    {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}]),
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error in sign combination table: {e}")

                    # --- Month Filter ---
                    st.markdown("#### Filter by Month")
                    if 'datetime' in df.columns:
                        df['month'] = pd.to_datetime(df['datetime']).dt.strftime('%B')
                        unique_months = df['month'].unique().tolist()
                        unique_months = sorted(unique_months, key=lambda m: pd.to_datetime(m, format='%B').month)
                        selected_months = st.multiselect(
                            "Select Months",
                            options=unique_months,
                            default=unique_months,
                            key="analysis_month_filter",
                            help="Filter analysis by selected months"
                        )
                        df = df[df['month'].isin(selected_months)]
                    else:
                        st.info("No datetime column found for month filtering.")
            elif selected_report == "daily-ohlc":
                st.write("ENTERED DAILY OHLC BLOCK")
                st.markdown("## Daily OHLC Analysis")
                
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # Add month filter before any table or stats
                    st.markdown("#### Filter by Month")
                    if 'datetime' in df.columns:
                        df['month'] = pd.to_datetime(df['datetime']).dt.strftime('%B')
                        unique_months = df['month'].unique().tolist()
                        unique_months = sorted(unique_months, key=lambda m: pd.to_datetime(m, format='%B').month)
                        selected_months = st.multiselect(
                            "Select Months",
                            options=unique_months,
                            default=unique_months,
                            key="daily_ohlc_month_filter",
                            help="Filter analysis by selected months"
                        )
                        df = df[df['month'].isin(selected_months)]
                    else:
                        st.info("No datetime column found for month filtering.")
                    
                    # Calculate daily returns
                    df['return'] = ((df['close'] - df['open']) / df['open']) * 100
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    
                    # Create price movement bins
                    bins = [float('-inf'), -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, float('inf')]
                    labels = [
                        'Below -2%', '-2% to -1.5%', '-1.5% to -1%', '-1% to -0.5%', '-0.5% to 0%',
                        '0% to 0.5%', '0.5% to 1%', '1% to 1.5%', '1.5% to 2%', 'Above 2%'
                    ]
                    
                    # Create the distribution table
                    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    table_data = []
                    
                    # Total positive days
                    pos_days = df[df['return'] > 0].groupby('weekday').size()
                    pos_row = {'Range': 'Total Positive Days'}
                    for day in weekdays:
                        pos_row[day] = pos_days.get(day, 0)
                    table_data.append(pos_row)
                    
                    # Total negative days
                    neg_days = df[df['return'] < 0].groupby('weekday').size()
                    neg_row = {'Range': 'Total Negative Days'}
                    for day in weekdays:
                        neg_row[day] = neg_days.get(day, 0)
                    table_data.append(neg_row)
                    
                    # Price movement distribution
                    df['movement_range'] = pd.cut(df['return'], bins=bins, labels=labels)
                    for label in labels:
                        range_data = df[df['movement_range'] == label].groupby('weekday').size()
                        row = {'Range': label}
                        for day in weekdays:
                            row[day] = range_data.get(day, 0)
                        table_data.append(row)
                    
                    # Convert to DataFrame and display
                    distribution_df = pd.DataFrame(table_data)
                    
                    # Calculate total days for each weekday
                    weekday_totals = {day: df[df['weekday'] == day].shape[0] for day in weekdays}
                    # Update each cell to show count and percentage
                    for idx, row in distribution_df.iterrows():
                        for day in weekdays:
                            count = row[day]
                            total = weekday_totals[day]
                            pct = (count / total * 100) if total > 0 else 0
                            distribution_df.at[idx, day] = f"{count} ({pct:.1f}%)"
                    # Add total column (just count, not percent)
                    distribution_df['Total'] = distribution_df[weekdays].apply(lambda x: sum(int(str(val).split(' ')[0]) for val in x), axis=1)
                    
                    # Style the DataFrame
                    def style_distribution(val):
                        if val == 0:
                            return 'color: #999999'
                        return 'color: #000000'
                    
                    st.markdown("### Daily Price Movement Distribution")
                    st.markdown("""
                    This table shows the distribution of price movements by weekday.
                    - Positive days: Days where close > open
                    - Negative days: Days where close < open
                    - Each range shows the number of days that fell within that price movement range
                    """)
                    
                    st.dataframe(
                        distribution_df.style
                        .applymap(style_distribution)
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                    
                    # Add new table for positive/negative closing days by weekday
                    st.markdown("### Positive/Negative Closing Days by Weekday")
                    weekday_stats = []
                    for day in weekdays:
                        day_data = df[df['weekday'] == day]
                        positive_days = len(day_data[day_data['close'] > day_data['open']])
                        negative_days = len(day_data[day_data['close'] < day_data['open']])
                        total_days = len(day_data)
                        positive_pct = (positive_days / total_days * 100) if total_days > 0 else 0
                        negative_pct = (negative_days / total_days * 100) if total_days > 0 else 0
                        weekday_stats.append({
                            'Weekday': day,
                            'Positive Days': f"{positive_days} ({positive_pct:.1f}%)",
                            'Negative Days': f"{negative_days} ({negative_pct:.1f}%)",
                            'Total Days': total_days
                        })
                    
                    weekday_stats_df = pd.DataFrame(weekday_stats)
                    st.dataframe(
                        weekday_stats_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                    
                    # Add new section for minute-to-day candle pattern analysis
                    st.markdown("### Minute-to-Day Candle Pattern Analysis")
                    st.markdown("""
                    This analysis shows how the first candle of a selected timeframe (5min, 15min, 1hour) aligns with the daily candle pattern.
                    - Green candle: Close > Open
                    - Red candle: Close < Open
                    """)
                    
                    # User input: select timeframe
                    timeframe_options = ["5min", "15min", "1hour"]
                    selected_tf = st.selectbox(
                        "Select timeframe to compare (first candle of the day)",
                        timeframe_options,
                        key="minute_to_day_tf"
                    )
                    
                    # Get data for the selected timeframe
                    tf_df = db.get_index_data(index_type, date_range, selected_tf)
                    pattern_analysis = []
                    if tf_df is not None and not tf_df.empty:
                        tf_df['datetime'] = pd.to_datetime(tf_df['datetime'])
                        tf_df['date'] = tf_df['datetime'].dt.date
                        tf_df['weekday'] = tf_df['datetime'].dt.day_name()
                        tf_df['return'] = ((tf_df['close'] - tf_df['open']) / tf_df['open']) * 100
                        for day in weekdays:
                            day_data = tf_df[tf_df['weekday'] == day]
                            if not day_data.empty:
                                dates = day_data['date'].unique()
                                matches = 0
                                total = 0
                                green_matched = 0
                                red_matched = 0
                                for date in dates:
                                    date_data = day_data[day_data['date'] == date]
                                    if not date_data.empty:
                                        # Get the FIRST candle of the day
                                        first_candle = date_data.iloc[0]
                                        first_candle_direction = 'Green' if first_candle['close'] > first_candle['open'] else 'Red'
                                        # Get the daily candle for this date
                                        daily_candle = df[df['datetime'].dt.date == date]
                                        if not daily_candle.empty:
                                            daily_candle = daily_candle.iloc[0]
                                            daily_direction = 'Green' if daily_candle['close'] > daily_candle['open'] else 'Red'
                                            if first_candle_direction == daily_direction:
                                                matches += 1
                                            if first_candle_direction == 'Green' and daily_direction == 'Green':
                                                green_matched += 1
                                            if first_candle_direction == 'Red' and daily_direction == 'Red':
                                                red_matched += 1
                                            total += 1
                                match_pct = (matches / total * 100) if total > 0 else 0
                                green_pct = (green_matched / total * 100) if total > 0 else 0
                                red_pct = (red_matched / total * 100) if total > 0 else 0
                                pattern_analysis.append({
                                    'Timeframe': selected_tf,
                                    'Weekday': day,
                                    'Matches': matches,
                                    'Total': total,
                                    'Match %': f"{match_pct:.1f}%",
                                    'Green % Matched': f"{green_pct:.1f}%",
                                    'Red % Matched': f"{red_pct:.1f}%"
                                })
                        pattern_df = pd.DataFrame(pattern_analysis)
                        st.dataframe(
                            pattern_df.style
                            .set_properties(**{'text-align': 'center'})
                            .set_table_styles([
                                {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                                {'selector': 'td', 'props': [('text-align', 'center')]}]),
                            use_container_width=True
                        )
                    else:
                        st.info("No data available for minute-to-day pattern analysis.")
                    
                    # Add summary statistics
                    st.markdown("### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_days = len(df)
                        positive_days = len(df[df['return'] > 0])
                        negative_days = len(df[df['return'] < 0])
                        st.metric("Total Days", total_days)
                        st.metric("Positive Days", positive_days, f"{positive_days/total_days*100:.1f}%")
                        st.metric("Negative Days", negative_days, f"{negative_days/total_days*100:.1f}%")
                    
                    with col2:
                        avg_return = df['return'].mean()
                        max_return = df['return'].max()
                        min_return = df['return'].min()
                        st.metric("Average Return", f"{avg_return:.2f}%")
                        st.metric("Maximum Return", f"{max_return:.2f}%")
                        st.metric("Minimum Return", f"{min_return:.2f}%")
                    
                    with col3:
                        std_return = df['return'].std()
                        median_return = df['return'].median()
                        st.metric("Return Std Dev", f"{std_return:.2f}%")
                        st.metric("Median Return", f"{median_return:.2f}%")

                    # --- Correlation and Covariance Table ---
                    st.markdown("### Day-to-Day Correlation and Covariance of Returns")
                    import numpy as np
                    corr_data = []
                    # Consecutive days
                    for i in range(len(weekdays)-1):
                        day1 = weekdays[i]
                        day2 = weekdays[i+1]
                        df_day1 = df[df['weekday'] == day1].copy()
                        df_day2 = df[df['weekday'] == day2].copy()
                        df_day1 = df_day1.sort_values('datetime').reset_index(drop=True)
                        df_day2 = df_day2.sort_values('datetime').reset_index(drop=True)
                        df_day1['week'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().week
                        df_day1['year'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().year
                        df_day2['week'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().week
                        df_day2['year'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().year
                        merged = pd.merge(df_day1, df_day2, on=['week', 'year'], suffixes=(f'_{day1}', f'_{day2}'))
                        n = len(merged)
                        if n > 1:
                            corr = merged[f'return_{day1}'].corr(merged[f'return_{day2}'])
                            covar = merged[f'return_{day1}'].cov(merged[f'return_{day2}'])
                        else:
                            corr = float('nan')
                            covar = float('nan')
                        corr_data.append({
                            'From': day1,
                            'To': day2,
                            'Correlation': f"{corr:.2f}" if not pd.isna(corr) else '-',
                            'Covariance': f"{covar:.4f}" if not pd.isna(covar) else '-',
                            'N': n
                        })
                    # Monday to Friday
                    day1 = 'Monday'
                    day2 = 'Friday'
                    df_day1 = df[df['weekday'] == day1].copy()
                    df_day2 = df[df['weekday'] == day2].copy()
                    df_day1 = df_day1.sort_values('datetime').reset_index(drop=True)
                    df_day2 = df_day2.sort_values('datetime').reset_index(drop=True)
                    df_day1['week'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().week
                    df_day1['year'] = pd.to_datetime(df_day1['datetime']).dt.isocalendar().year
                    df_day2['week'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().week
                    df_day2['year'] = pd.to_datetime(df_day2['datetime']).dt.isocalendar().year
                    merged = pd.merge(df_day1, df_day2, on=['week', 'year'], suffixes=(f'_{day1}', f'_{day2}'))
                    n = len(merged)
                    if n > 1:
                        corr = merged[f'return_{day1}'].corr(merged[f'return_{day2}'])
                        covar = merged[f'return_{day1}'].cov(merged[f'return_{day2}'])
                    else:
                        corr = float('nan')
                        covar = float('nan')
                    corr_data.append({
                        'From': day1,
                        'To': day2,
                        'Correlation': f"{corr:.2f}" if not pd.isna(corr) else '-',
                        'Covariance': f"{covar:.4f}" if not pd.isna(covar) else '-',
                        'N': n
                    })
                    # Friday to next Monday
                    day1 = 'Friday'
                    day2 = 'Monday'
                    df_friday = df[df['weekday'] == day1].copy()
                    df_monday = df[df['weekday'] == day2].copy()
                    df_friday = df_friday.sort_values('datetime').reset_index(drop=True)
                    df_monday = df_monday.sort_values('datetime').reset_index(drop=True)
                    friday_dates = pd.to_datetime(df_friday['datetime']).dt.date.values
                    monday_dates = pd.to_datetime(df_monday['datetime']).dt.date.values
                    friday_to_next_monday = []
                    for f_date in friday_dates:
                        next_monday = min([m for m in monday_dates if m > f_date], default=None)
                        if next_monday:
                            friday_row = df_friday[pd.to_datetime(df_friday['datetime']).dt.date == f_date]
                            monday_row = df_monday[pd.to_datetime(df_monday['datetime']).dt.date == next_monday]
                            if not friday_row.empty and not monday_row.empty:
                                friday_to_next_monday.append({
                                    'friday_return': friday_row['return'].values[0],
                                    'monday_return': monday_row['return'].values[0]
                                })
                    n = len(friday_to_next_monday)
                    if n > 1:
                        friday_returns = [x['friday_return'] for x in friday_to_next_monday]
                        monday_returns = [x['monday_return'] for x in friday_to_next_monday]
                        corr = np.corrcoef(friday_returns, monday_returns)[0, 1]
                        covar = np.cov(friday_returns, monday_returns)[0, 1]
                    else:
                        corr = float('nan')
                        covar = float('nan')
                    corr_data.append({
                        'From': 'Friday',
                        'To': 'Monday (next week)',
                        'Correlation': f"{corr:.2f}" if not pd.isna(corr) else '-',
                        'Covariance': f"{covar:.4f}" if not pd.isna(covar) else '-',
                        'N': n
                    })
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)
                    # --- Day Pair Sign Table ---
                    st.markdown("""
                    ### Day-to-Day Sign Combination Table (A/B/C/D)
                    **Legend:**
                    - **A**: Both days +ve (close > open)
                    - **B**: Both days -ve (close < open)
                    - **C**: Row day +ve, Col day -ve
                    - **D**: Row day -ve, Col day +ve
                    
                    Each cell shows: `A/B/C/D`
                    """)
                    try:
                        sign_matrix = []
                        for row_day in weekdays:
                            row = {'Day': row_day}
                            for col_day in weekdays:
                                if row_day == col_day:
                                    row[col_day] = ''
                                else:
                                    # Align by week and year
                                    df_row = df[df['weekday'] == row_day].copy()
                                    df_col = df[df['weekday'] == col_day].copy()
                                    df_row = df_row.sort_values('datetime').reset_index(drop=True)
                                    df_col = df_col.sort_values('datetime').reset_index(drop=True)
                                    df_row['week'] = pd.to_datetime(df_row['datetime']).dt.isocalendar().week
                                    df_row['year'] = pd.to_datetime(df_row['datetime']).dt.isocalendar().year
                                    df_col['week'] = pd.to_datetime(df_col['datetime']).dt.isocalendar().week
                                    df_col['year'] = pd.to_datetime(df_col['datetime']).dt.isocalendar().year
                                    merged = pd.merge(df_row, df_col, on=['week', 'year'], suffixes=(f'_{row_day}', f'_{col_day}'))
                                    # Only keep pairs where col_day is after row_day in the week
                                    merged = merged[pd.to_datetime(merged[f'datetime_{col_day}']) > pd.to_datetime(merged[f'datetime_{row_day}'])]
                                    total_pairs = len(merged)
                                    # A: both +ve
                                    both_pos = ((merged[f'close_{row_day}'] > merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] > merged[f'open_{col_day}'])).sum()
                                    # B: both -ve
                                    both_neg = ((merged[f'close_{row_day}'] < merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] < merged[f'open_{col_day}'])).sum()
                                    # C: row +ve, col -ve
                                    row_pos_col_neg = ((merged[f'close_{row_day}'] > merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] < merged[f'open_{col_day}'])).sum()
                                    # D: row -ve, col +ve
                                    row_neg_col_pos = ((merged[f'close_{row_day}'] < merged[f'open_{row_day}']) & (merged[f'close_{col_day}'] > merged[f'open_{col_day}'])).sum()
                                    # Calculate percentages
                                    both_pos_pct = round(both_pos / total_pairs * 100) if total_pairs > 0 else 0
                                    both_neg_pct = round(both_neg / total_pairs * 100) if total_pairs > 0 else 0
                                    row_pos_col_neg_pct = round(row_pos_col_neg / total_pairs * 100) if total_pairs > 0 else 0
                                    row_neg_col_pos_pct = round(row_neg_col_pos / total_pairs * 100) if total_pairs > 0 else 0
                                    row[col_day] = f"{both_pos_pct}%/{both_neg_pct}%/{row_pos_col_neg_pct}%/{row_neg_col_pos_pct}%"
                            sign_matrix.append(row)
                        sign_matrix_df = pd.DataFrame(sign_matrix)
                        if sign_matrix_df.empty:
                            st.warning("No data available for the sign combination table.")
                        else:
                            st.dataframe(
                                sign_matrix_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                                    {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}
                                ]),
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error in sign combination table: {e}")
                    
                    # Add new table for open-to-close return ranges by weekday
                    st.markdown("### Open-to-Close Return Ranges by Weekday")
                    st.markdown("""
                    This table shows the distribution of open-to-close returns by weekday, categorized into percentage ranges.
                    Each cell shows: `count (percentage)`
                    """)
                    
                    # Define return ranges with detailed categories between -2% and +2%
                    return_ranges = [
                        (float('-inf'), -2.0, 'Below -2%'),
                        (-2.0, -1.75, '-2% to -1.75%'),
                        (-1.75, -1.5, '-1.75% to -1.5%'),
                        (-1.5, -1.25, '-1.5% to -1.25%'),
                        (-1.25, -1.0, '-1.25% to -1%'),
                        (-1.0, -0.75, '-1% to -0.75%'),
                        (-0.75, -0.5, '-0.75% to -0.5%'),
                        (-0.5, -0.25, '-0.5% to -0.25%'),
                        (-0.25, 0, '-0.25% to 0%'),
                        (0, 0.25, '0% to 0.25%'),
                        (0.25, 0.5, '0.25% to 0.5%'),
                        (0.5, 0.75, '0.5% to 0.75%'),
                        (0.75, 1.0, '0.75% to 1%'),
                        (1.0, 1.25, '1% to 1.25%'),
                        (1.25, 1.5, '1.25% to 1.5%'),
                        (1.5, 1.75, '1.5% to 1.75%'),
                        (1.75, 2.0, '1.75% to 2%'),
                        (2.0, float('inf'), 'Above 2%')
                    ]
                    
                    # Create the distribution table
                    weekday_ranges_data = []
                    for day in weekdays:
                        day_data = df[df['weekday'] == day]
                        row = {'Weekday': day}
                        total_days = len(day_data)
                        
                        for _, (lower, upper, label) in enumerate(return_ranges):
                            if lower == float('-inf'):
                                count = len(day_data[day_data['return'] < upper])
                            elif upper == float('inf'):
                                count = len(day_data[day_data['return'] >= lower])
                            else:
                                count = len(day_data[(day_data['return'] >= lower) & (day_data['return'] < upper)])
                            
                            percentage = (count / total_days * 100) if total_days > 0 else 0
                            row[label] = f"{count} ({percentage:.1f}%)"
                        
                        row['Total Days'] = total_days
                        weekday_ranges_data.append(row)
                    
                    weekday_ranges_df = pd.DataFrame(weekday_ranges_data)
                    
                    # Display the table
                    st.dataframe(
                        weekday_ranges_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                    
                    # Add summary statistics
                    st.markdown("### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_days = len(df)
                        positive_days = len(df[df['return'] > 0])
                        negative_days = len(df[df['return'] < 0])
                        st.metric("Total Days", total_days)
                        st.metric("Positive Days", positive_days, f"{positive_days/total_days*100:.1f}%")
                        st.metric("Negative Days", negative_days, f"{negative_days/total_days*100:.1f}%")
                    
                    with col2:
                        avg_return = df['return'].mean()
                        max_return = df['return'].max()
                        min_return = df['return'].min()
                        st.metric("Average Return", f"{avg_return:.2f}%")
                        st.metric("Maximum Return", f"{max_return:.2f}%")
                        st.metric("Minimum Return", f"{min_return:.2f}%")
                    
                    with col3:
                        std_return = df['return'].std()
                        median_return = df['return'].median()
                        st.metric("Return Std Dev", f"{std_return:.2f}%")
                        st.metric("Median Return", f"{median_return:.2f}%")
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "monthly-ohlc":
                st.markdown("## Monthly OHLC Analysis")
                # Get daily data for the selected index and date range
                df_daily = db.get_index_data(index_type, date_range, "daily")
                if df_daily is not None and not df_daily.empty:
                    # Aggregate to monthly OHLC
                    df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
                    df_daily['month'] = df_daily['datetime'].dt.to_period('M')
                    df_monthly = df_daily.groupby('month').agg(
                        open=('open', 'first'),
                        high=('high', 'max'),
                        low=('low', 'min'),
                        close=('close', 'last'),
                        volume=('volume', 'sum')
                    ).reset_index()
                    df_monthly['month'] = df_monthly['month'].dt.to_timestamp()
                    df_monthly['return'] = ((df_monthly['close'] - df_monthly['open']) / df_monthly['open']) * 100
                    df_monthly['month_name'] = df_monthly['month'].dt.strftime('%B')

                    # Month filter
                    unique_months = df_monthly['month_name'].unique().tolist()
                    unique_months = sorted(unique_months, key=lambda m: pd.to_datetime(m, format='%B').month)
                    selected_months = st.multiselect(
                        "Filter by Month",
                        options=unique_months,
                        default=unique_months,
                        key="monthly_ohlc_month_filter",
                        help="Filter analysis by selected months"
                    )
                    df_monthly = df_monthly[df_monthly['month_name'].isin(selected_months)]

                    # Price movement bins (same as daily, but for monthly returns)
                    bins = [float('-inf'), -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, float('inf')]
                    labels = [
                        'Below -2%', '-2% to -1.5%', '-1.5% to -1%', '-1% to -0.5%', '-0.5% to 0%',
                        '0% to 0.5%', '0.5% to 1%', '1% to 1.5%', '1.5% to 2%', 'Above 2%'
                    ]
                    df_monthly['movement_range'] = pd.cut(df_monthly['return'], bins=bins, labels=labels)

                    # Price movement distribution table
                    st.markdown("### Monthly Price Movement Distribution")
                    table_data = []
                    # Total positive months
                    pos_months = (df_monthly['return'] > 0).sum()
                    table_data.append({'Range': 'Total Positive Months', 'Count': pos_months})
                    # Total negative months
                    neg_months = (df_monthly['return'] < 0).sum()
                    table_data.append({'Range': 'Total Negative Months', 'Count': neg_months})
                    # Movement ranges
                    for label in labels:
                        count = (df_monthly['movement_range'] == label).sum()
                        table_data.append({'Range': label, 'Count': count})
                    movement_df = pd.DataFrame(table_data)
                    st.dataframe(movement_df, use_container_width=True)

                    # Summary statistics
                    st.markdown("### Monthly Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Months", len(df_monthly))
                        st.metric("Positive Months", pos_months, f"{pos_months/len(df_monthly)*100:.1f}%")
                        st.metric("Negative Months", neg_months, f"{neg_months/len(df_monthly)*100:.1f}%")
                    with col2:
                        st.metric("Average Return", f"{df_monthly['return'].mean():.2f}%")
                        st.metric("Maximum Return", f"{df_monthly['return'].max():.2f}%")
                        st.metric("Minimum Return", f"{df_monthly['return'].min():.2f}%")
                    with col3:
                        st.metric("Return Std Dev", f"{df_monthly['return'].std():.2f}%")
                        st.metric("Median Return", f"{df_monthly['return'].median():.2f}%")

                    # Show raw monthly data
                    st.markdown("### Raw Monthly OHLC Data")
                    st.dataframe(df_monthly[['month', 'open', 'high', 'low', 'close', 'volume', 'return']], use_container_width=True)
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "pattern":
                st.markdown("## Pattern Analysis")
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # Filter for weekdays only
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    df = df[df['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
                    
                    # Calculate daily returns
                    df['return'] = ((df['close'] - df['open']) / df['open']) * 100
                    
                    # Sort by date
                    df = df.sort_values('datetime')
                    
                    # Create 2-day pattern column
                    df['prev_return'] = df['return'].shift(1)
                    df['next_return'] = df['return'].shift(-1)  # Get next day's return
                    df['pattern'] = 'None'
                    
                    # Define patterns
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0), 'pattern'] = 'DD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0), 'pattern'] = 'UU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0), 'pattern'] = 'DU'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0), 'pattern'] = 'UD'
                    
                    # Create pattern analysis table
                    st.markdown("### 2-Day Close Pattern Analysis")
                    st.markdown("""
                    **Pattern Definitions:**
                    - DD: Two consecutive down closes (negative)
                    - UU: Two consecutive up closes (positive)
                    - DU: First day down, second day up
                    - UD: First day up, second day down
                    
                    **Example:**
                    If we have 4 days: Up → Down → Down → Up
                    - Days 1-2 form a "UD" pattern (Up → Down)
                    - Days 2-3 form a "DD" pattern (Down → Down)
                    
                    **Next Day Analysis:**
                    Shows how the market closed the next day after each pattern
                    """)
                    
                    # Calculate pattern counts and next day analysis
                    pattern_data = []
                    for pattern in ['DD', 'UU', 'DU', 'UD']:
                        pattern_df = df[df['pattern'] == pattern]
                        total = len(pattern_df)
                        if total > 0:
                            # Basic next day direction
                            next_up = len(pattern_df[pattern_df['next_return'] > 0])
                            next_down = len(pattern_df[pattern_df['next_return'] < 0])
                            next_up_pct = (next_up / total * 100) if total > 0 else 0
                            next_down_pct = (next_down / total * 100) if total > 0 else 0
                            
                            # Return metrics - separate for positive and negative returns
                            positive_returns = pattern_df[pattern_df['next_return'] > 0]['next_return']
                            negative_returns = pattern_df[pattern_df['next_return'] < 0]['next_return']
                            
                            avg_positive_return = positive_returns.mean() if not positive_returns.empty else 0
                            avg_negative_return = negative_returns.mean() if not negative_returns.empty else 0
                            max_positive_return = positive_returns.max() if not positive_returns.empty else 0
                            max_negative_return = negative_returns.min() if not negative_returns.empty else 0
                            
                            pattern_data.append({
                                'Pattern': pattern,
                                'Count': total,
                                'Next Day Up': f"{next_up} ({next_up_pct:.1f}%)",
                                'Next Day Down': f"{next_down} ({next_down_pct:.1f}%)",
                                'Avg +ve Return': f"{avg_positive_return:.2f}%",
                                'Avg -ve Return': f"{avg_negative_return:.2f}%",
                                'Max +ve Return': f"{max_positive_return:.2f}%",
                                'Max -ve Return': f"{max_negative_return:.2f}%"
                            })
                    
                    pattern_df = pd.DataFrame(pattern_data)
                    
                    # Display the main table
                    st.markdown("### 2-Day Close Pattern Analysis")
                    st.dataframe(
                        pattern_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                    
                    # Add ranges analysis
                    st.markdown("### Return Ranges Analysis")
                    range_bins = [
                        (-float('inf'), -2.0, 'Below -2%'),
                        (-2.0, -1.0, '-2% to -1%'),
                        (-1.0, -0.5, '-1% to -0.5%'),
                        (-0.5, 0, '-0.5% to 0%'),
                        (0, 0.5, '0% to 0.5%'),
                        (0.5, 1.0, '0.5% to 1%'),
                        (1.0, 2.0, '1% to 2%'),
                        (2.0, float('inf'), 'Above 2%')
                    ]
                    
                    ranges_data = []
                    for pattern in ['DD', 'UU', 'DU', 'UD']:
                        pattern_df = df[df['pattern'] == pattern]
                        row = {'Pattern': pattern}
                        
                        for _, (lower, upper, label) in enumerate(range_bins):
                            if lower == -float('inf'):
                                count = len(pattern_df[pattern_df['next_return'] < upper])
                            elif upper == float('inf'):
                                count = len(pattern_df[pattern_df['next_return'] >= lower])
                            else:
                                count = len(pattern_df[(pattern_df['next_return'] >= lower) & (pattern_df['next_return'] < upper)])
                            
                            total = len(pattern_df)
                            percentage = (count / total * 100) if total > 0 else 0
                            row[label] = f"{count} ({percentage:.1f}%)"
                        
                        ranges_data.append(row)
                    
                    ranges_df = pd.DataFrame(ranges_data)
                    st.dataframe(
                        ranges_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )

                    # Add 3-day, 4-day, and 5-day pattern analysis
                    st.markdown("### 3-Day, 4-Day, and 5-Day Close Pattern Analysis")
                    
                    # Calculate 3-day, 4-day, and 5-day patterns
                    df['prev_return_2'] = df['return'].shift(2)  # Get return from 2 days ago
                    df['prev_return_3'] = df['return'].shift(3)  # Get return from 3 days ago
                    df['prev_return_4'] = df['return'].shift(4)  # Get return from 4 days ago
                    df['pattern_3_4_5day'] = 'None'
                    
                    # Define 3-day patterns
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0), 'pattern_3_4_5day'] = 'DDD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0), 'pattern_3_4_5day'] = 'UUU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0), 'pattern_3_4_5day'] = 'DDU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] < 0), 'pattern_3_4_5day'] = 'DUU'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0), 'pattern_3_4_5day'] = 'UUD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] > 0), 'pattern_3_4_5day'] = 'UDU'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] < 0), 'pattern_3_4_5day'] = 'DUD'
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] > 0), 'pattern_3_4_5day'] = 'UDD'
                    
                    # Define 4-day patterns
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0), 'pattern_3_4_5day'] = 'DDDD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0), 'pattern_3_4_5day'] = 'UUUU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0), 'pattern_3_4_5day'] = 'DDDU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] < 0), 'pattern_3_4_5day'] = 'DUUU'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0), 'pattern_3_4_5day'] = 'UUUD'
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] > 0), 'pattern_3_4_5day'] = 'UDDD'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0), 'pattern_3_4_5day'] = 'DDUD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0), 'pattern_3_4_5day'] = 'UUDU'

                    # Define 5-day patterns
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0) & (df['prev_return_4'] < 0), 'pattern_3_4_5day'] = 'DDDDD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0) & (df['prev_return_4'] > 0), 'pattern_3_4_5day'] = 'UUUUU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0) & (df['prev_return_4'] < 0), 'pattern_3_4_5day'] = 'DDDDU'
                    df.loc[(df['return'] > 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0) & (df['prev_return_4'] < 0), 'pattern_3_4_5day'] = 'DUUUU'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0) & (df['prev_return_4'] > 0), 'pattern_3_4_5day'] = 'UUUUD'
                    df.loc[(df['return'] < 0) & (df['prev_return'] < 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0) & (df['prev_return_4'] > 0), 'pattern_3_4_5day'] = 'UDDDD'
                    df.loc[(df['return'] < 0) & (df['prev_return'] > 0) & (df['prev_return_2'] < 0) & (df['prev_return_3'] < 0) & (df['prev_return_4'] < 0), 'pattern_3_4_5day'] = 'DDDUD'
                    df.loc[(df['return'] > 0) & (df['prev_return'] < 0) & (df['prev_return_2'] > 0) & (df['prev_return_3'] > 0) & (df['prev_return_4'] > 0), 'pattern_3_4_5day'] = 'UUUDU'
                    
                    # Calculate pattern counts and next day analysis
                    pattern_3_4_5day_data = []
                    patterns = ['DDD', 'UUU', 'DDU', 'DUU', 'UUD', 'UDU', 'DUD', 'UDD',
                              'DDDD', 'UUUU', 'DDDU', 'DUUU', 'UUUD', 'UDDD', 'DDUD', 'UUDU',
                              'DUDU', 'UDUD', 'DUUD', 'UDDU', 'DDUU', 'UUDD', 'DUDD', 'UDUU',
                              'DDDDD', 'UUUUU', 'DDDDU', 'DUUUU', 'UUUUD', 'UDDDD', 'DDDUD', 'UUUDU',
                              'DUDUD', 'UDUDU', 'DUUDU', 'UDUDD', 'DDUUU', 'UUDDD', 'DUDDD', 'UDUUU',
                              'DDUUD', 'UUDDU', 'DUUUD', 'UDDUD', 'DUDUU', 'UDUDD', 'DUUDD', 'UDDUU',
                              'DUDDU', 'UDUUD', 'DDUDU', 'UUDUD', 'DUDUD', 'UDUDU', 'DDUUD', 'UUDDU']
                    
                    for pattern in patterns:
                        pattern_df = df[df['pattern_3_4_5day'] == pattern]
                        total = len(pattern_df)
                        if total > 0:
                            # Basic next day direction
                            next_up = len(pattern_df[pattern_df['next_return'] > 0])
                            next_down = len(pattern_df[pattern_df['next_return'] < 0])
                            next_up_pct = (next_up / total * 100) if total > 0 else 0
                            next_down_pct = (next_down / total * 100) if total > 0 else 0
                            
                            # Return metrics - separate for positive and negative returns
                            positive_returns = pattern_df[pattern_df['next_return'] > 0]['next_return']
                            negative_returns = pattern_df[pattern_df['next_return'] < 0]['next_return']
                            
                            avg_positive_return = positive_returns.mean() if not positive_returns.empty else 0
                            avg_negative_return = negative_returns.mean() if not negative_returns.empty else 0
                            max_positive_return = positive_returns.max() if not positive_returns.empty else 0
                            max_negative_return = negative_returns.min() if not negative_returns.empty else 0
                            
                            pattern_3_4_5day_data.append({
                                'Pattern': pattern,
                                'Count': total,
                                'Next Day Up': f"{next_up} ({next_up_pct:.1f}%)",
                                'Next Day Down': f"{next_down} ({next_down_pct:.1f}%)",
                                'Avg +ve Return': f"{avg_positive_return:.2f}%",
                                'Avg -ve Return': f"{avg_negative_return:.2f}%",
                                'Max +ve Return': f"{max_positive_return:.2f}%",
                                'Max -ve Return': f"{max_negative_return:.2f}%"
                            })
                    
                    pattern_3_4_5day_df = pd.DataFrame(pattern_3_4_5day_data)
                    
                    # Display the combined pattern table
                    st.dataframe(
                        pattern_3_4_5day_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                    
                    # Add ranges analysis for 3-day, 4-day, and 5-day patterns
                    st.markdown("### 3-Day, 4-Day, and 5-Day Pattern Return Ranges Analysis")
                    ranges_3_4_5day_data = []
                    for pattern in patterns:
                        pattern_df = df[df['pattern_3_4_5day'] == pattern]
                        row = {'Pattern': pattern}
                        
                        for _, (lower, upper, label) in enumerate(range_bins):
                            if lower == -float('inf'):
                                count = len(pattern_df[pattern_df['next_return'] < upper])
                            elif upper == float('inf'):
                                count = len(pattern_df[pattern_df['next_return'] >= lower])
                            else:
                                count = len(pattern_df[(pattern_df['next_return'] >= lower) & (pattern_df['next_return'] < upper)])
                            
                            total = len(pattern_df)
                            percentage = (count / total * 100) if total > 0 else 0
                            row[label] = f"{count} ({percentage:.1f}%)"
                        
                        ranges_3_4_5day_data.append(row)
                    
                    ranges_3_4_5day_df = pd.DataFrame(ranges_3_4_5day_data)
                    st.dataframe(
                        ranges_3_4_5day_df.style
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ]),
                        use_container_width=True
                    )
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "new-high-low":
                st.markdown("## New Highs and Lows Analysis")
                lookback_days = st.number_input("Enter number of days to look back", min_value=1, max_value=100, value=5, step=1)
                
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # Sort by date
                    df = df.sort_values('datetime')
                    
                    # Calculate rolling high and low based on user input
                    df['rolling_high'] = df['high'].rolling(window=lookback_days).max()
                    df['rolling_low'] = df['low'].rolling(window=lookback_days).min()
                    
                    # Identify new highs and lows
                    df['new_high'] = (df['high'] > df['rolling_high'].shift(1)) & (df['high'] > df['high'].shift(1))
                    df['new_low'] = (df['low'] < df['rolling_low'].shift(1)) & (df['low'] < df['low'].shift(1))
                    
                    # Calculate next day returns
                    df['next_day_return'] = df['close'].pct_change(1).shift(-1) * 100
                    
                    # Filter for weekdays
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    df = df[df['weekday'].isin(selected_weekdays)]
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_days = len(df)
                        st.metric("Total Days", total_days)
                    with col2:
                        new_highs = df['new_high'].sum()
                        st.metric("New Highs", new_highs, f"{new_highs/total_days*100:.1f}%")
                    with col3:
                        new_lows = df['new_low'].sum()
                        st.metric("New Lows", new_lows, f"{new_lows/total_days*100:.1f}%")
                    with col4:
                        both = ((df['new_high'] & df['new_low'])).sum()
                        st.metric("Both High & Low", both, f"{both/total_days*100:.1f}%")
                    
                    # New Highs Analysis
                    st.markdown(f"### New Highs Analysis (Breaking {lookback_days}-day High)")
                    high_df = df[df['new_high']].copy()
                    if not high_df.empty:
                        # Calculate days stayed above high
                        high_df['days_above_high'] = 0
                        for idx, row in high_df.iterrows():
                            # Get the broken high from the previous day
                            broken_high = df.loc[df.index[df.index.get_loc(idx)-1], 'rolling_high']
                            days_above = 0
                            for i in range(1, len(df) - df.index.get_loc(idx)):
                                if df.iloc[df.index.get_loc(idx) + i]['low'] > broken_high:
                                    days_above += 1
                                else:
                                    break
                            high_df.at[idx, 'days_above_high'] = days_above
                        
                        # Calculate next day statistics
                        high_df['next_day_close_higher'] = high_df['next_day_return'] > 0
                        
                        # Calculate statistics
                        same_day_stats = {
                            'Total New Highs': len(high_df),
                            'Next Day Close Higher': f"{high_df['next_day_close_higher'].sum()} ({high_df['next_day_close_higher'].mean()*100:.1f}%)",
                            'Average Next Day Return': f"{high_df['next_day_return'].mean():.2f}%",
                            'Max Next Day Return': f"{high_df['next_day_return'].max():.2f}%",
                            'Min Next Day Return': f"{high_df['next_day_return'].min():.2f}%",
                            'Avg Days Above High': f"{high_df['days_above_high'].mean():.1f}",
                            'Max Days Above High': f"{high_df['days_above_high'].max()}",
                            'Days Above High Distribution': f"0: {len(high_df[high_df['days_above_high'] == 0])}, 1: {len(high_df[high_df['days_above_high'] == 1])}, 2+: {len(high_df[high_df['days_above_high'] >= 2])}"
                        }
                        st.dataframe(pd.DataFrame([same_day_stats]).T, use_container_width=True)
                        # Calculate detailed distribution
                        import numpy as np
                        # Prepare bins 0-9 and '10+'
                        bins = [str(i) for i in range(0, 10)] + ['10+']
                        high_df['days_above_high_capped'] = np.where(high_df['days_above_high'] >= 10, '10+', high_df['days_above_high'].astype(str))
                        distribution = high_df['days_above_high_capped'].value_counts().reindex(bins, fill_value=0)
                        distribution_pct = (distribution / len(high_df) * 100).round(1)
                        detailed_table = pd.DataFrame({
                            'Days Above High': distribution.index,
                            'Count': distribution.values,
                            'Percentage': distribution_pct.values
                        })
                        st.markdown("#### Detailed Days Above High Distribution")
                        st.dataframe(detailed_table, use_container_width=True)
                        # Bar chart
                        st.bar_chart(distribution)
                    
                    # New Lows Analysis
                    st.markdown(f"### New Lows Analysis (Breaking {lookback_days}-day Low)")
                    low_df = df[df['new_low']].copy()
                    if not low_df.empty:
                        # Calculate days stayed below low
                        low_df['days_below_low'] = 0
                        for idx, row in low_df.iterrows():
                            # Get the broken low from the previous day
                            broken_low = df.loc[df.index[df.index.get_loc(idx)-1], 'rolling_low']
                            days_below = 0
                            for i in range(1, len(df) - df.index.get_loc(idx)):
                                if df.iloc[df.index.get_loc(idx) + i]['high'] < broken_low:
                                    days_below += 1
                                else:
                                    break
                            low_df.at[idx, 'days_below_low'] = days_below
                        
                        # Calculate next day statistics
                        low_df['next_day_close_lower'] = low_df['next_day_return'] < 0
                        
                        # Calculate statistics
                        same_day_stats = {
                            'Total New Lows': len(low_df),
                            'Next Day Close Lower': f"{low_df['next_day_close_lower'].sum()} ({low_df['next_day_close_lower'].mean()*100:.1f}%)",
                            'Average Next Day Return': f"{low_df['next_day_return'].mean():.2f}%",
                            'Max Next Day Return': f"{low_df['next_day_return'].max():.2f}%",
                            'Min Next Day Return': f"{low_df['next_day_return'].min():.2f}%",
                            'Avg Days Below Low': f"{low_df['days_below_low'].mean():.1f}",
                            'Max Days Below Low': f"{low_df['days_below_low'].max()}",
                            'Days Below Low Distribution': f"0: {len(low_df[low_df['days_below_low'] == 0])}, 1: {len(low_df[low_df['days_below_low'] == 1])}, 2+: {len(low_df[low_df['days_below_low'] >= 2])}"
                        }
                        st.dataframe(pd.DataFrame([same_day_stats]).T, use_container_width=True)
                        # Calculate detailed distribution
                        bins = [str(i) for i in range(0, 10)] + ['10+']
                        low_df['days_below_low_capped'] = np.where(low_df['days_below_low'] >= 10, '10+', low_df['days_below_low'].astype(str))
                        distribution_low = low_df['days_below_low_capped'].value_counts().reindex(bins, fill_value=0)
                        distribution_low_pct = (distribution_low / len(low_df) * 100).round(1)
                        detailed_table_low = pd.DataFrame({
                            'Days Below Low': distribution_low.index,
                            'Count': distribution_low.values,
                            'Percentage': distribution_low_pct.values
                        })
                        st.markdown("#### Detailed Days Below Low Distribution")
                        st.dataframe(detailed_table_low, use_container_width=True)
                        # Bar chart
                        st.bar_chart(distribution_low)
                    
                    # Show raw data
                    st.markdown("### Raw Data")
                    display_df = df[df['new_high'] | df['new_low']].copy()
                    display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d')
                    display_df['next_day_close_higher'] = display_df['next_day_return'] > 0
                    display_df['next_day_close_lower'] = display_df['next_day_return'] < 0
                    
                    # Calculate days above/below for display
                    display_df['days_above_high'] = 0
                    display_df['days_below_low'] = 0
                    
                    for idx, row in display_df.iterrows():
                        if row['new_high']:
                            # Get the broken high from the previous day
                            broken_high = df.loc[df.index[df.index.get_loc(idx)-1], 'rolling_high']
                            days_above = 0
                            for i in range(1, len(df) - df.index.get_loc(idx)):
                                if df.iloc[df.index.get_loc(idx) + i]['low'] > broken_high:
                                    days_above += 1
                                else:
                                    break
                            display_df.at[idx, 'days_above_high'] = days_above
                        
                        if row['new_low']:
                            # Get the broken low from the previous day
                            broken_low = df.loc[df.index[df.index.get_loc(idx)-1], 'rolling_low']
                            days_below = 0
                            for i in range(1, len(df) - df.index.get_loc(idx)):
                                if df.iloc[df.index.get_loc(idx) + i]['high'] < broken_low:
                                    days_below += 1
                                else:
                                    break
                            display_df.at[idx, 'days_below_low'] = days_below
                    
                    st.dataframe(
                        display_df[['datetime', 'open', 'high', 'low', 'close', 'new_high', 'new_low', 
                                  'next_day_close_higher', 'next_day_close_lower', 'next_day_return',
                                  'days_above_high', 'days_below_low']].style.format({
                            'open': '{:.2f}',
                            'high': '{:.2f}',
                            'low': '{:.2f}',
                            'close': '{:.2f}',
                            'next_day_return': '{:.2f}%',
                            'days_above_high': '{:.0f}',
                            'days_below_low': '{:.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "nr-ws":
                st.markdown("## Narrow Range (NR) and Wide Spread (WS) Analysis")
                
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # Sort by date
                    df = df.sort_values('datetime')
                    
                    # Calculate daily range in points
                    df['daily_range'] = df['high'] - df['low']
                    
                    # Get number of days for pattern
                    pattern_days = st.number_input("Enter number of days for pattern", min_value=2, max_value=10, value=4, step=1)
                    
                    # Function to check NR pattern
                    def is_nr_pattern(group):
                        if len(group) < pattern_days:
                            return False
                        ranges = group['daily_range'].values
                        for i in range(1, len(ranges)):
                            if ranges[i] >= ranges[i-1]:
                                return False
                        return True
                    
                    # Function to check WS pattern
                    def is_ws_pattern(group):
                        if len(group) < pattern_days:
                            return False
                        ranges = group['daily_range'].values
                        for i in range(1, len(ranges)):
                            if ranges[i] <= ranges[i-1]:
                                return False
                        return True
                    
                    # Create rolling windows for pattern detection
                    df['is_nr'] = False
                    df['is_ws'] = False
                    
                    for i in range(len(df) - pattern_days + 1):
                        window = df.iloc[i:i+pattern_days]
                        if is_nr_pattern(window):
                            df.loc[window.index[-1], 'is_nr'] = True
                        if is_ws_pattern(window):
                            df.loc[window.index[-1], 'is_ws'] = True
                    
                    # Calculate next day returns
                    df['next_day_return'] = df['close'].pct_change(1).shift(-1) * 100
                    
                    # Filter for weekdays
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    df = df[df['weekday'].isin(selected_weekdays)]
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_days = len(df)
                        st.metric("Total Days", total_days)
                    with col2:
                        nr_patterns = df['is_nr'].sum()
                        st.metric(f"NR ({pattern_days}-day) Patterns", nr_patterns, f"{nr_patterns/total_days*100:.1f}%")
                    with col3:
                        ws_patterns = df['is_ws'].sum()
                        st.metric(f"WS ({pattern_days}-day) Patterns", ws_patterns, f"{ws_patterns/total_days*100:.1f}%")
                    
                    # NR Pattern Analysis
                    st.markdown(f"### {pattern_days}-Day Narrow Range (NR) Analysis")
                    nr_df = df[df['is_nr']].copy()
                    if not nr_df.empty:
                        # Calculate statistics
                        nr_stats = {
                            'Total NR Patterns': len(nr_df),
                            'Average Range on Last Day': f"{nr_df['daily_range'].mean():.2f} points",
                            'Min Range on Last Day': f"{nr_df['daily_range'].min():.2f} points",
                            'Next Day Up': f"{len(nr_df[nr_df['next_day_return'] > 0])} ({len(nr_df[nr_df['next_day_return'] > 0])/len(nr_df)*100:.1f}%)",
                            'Next Day Down': f"{len(nr_df[nr_df['next_day_return'] < 0])} ({len(nr_df[nr_df['next_day_return'] < 0])/len(nr_df)*100:.1f}%)",
                            'Average Next Day Return': f"{nr_df['next_day_return'].mean():.2f}%",
                            'Max Next Day Return': f"{nr_df['next_day_return'].max():.2f}%",
                            'Min Next Day Return': f"{nr_df['next_day_return'].min():.2f}%"
                        }
                        st.dataframe(pd.DataFrame([nr_stats]).T, use_container_width=True)
                        
                        # Show example patterns
                        st.markdown("#### Example NR Pattern")
                        if not nr_df.empty:
                            idx = nr_df.index[0]  # Get only the first pattern
                            pattern_days_df = df.loc[idx-pattern_days+1:idx]
                            st.markdown(f"**Pattern ending on {pd.to_datetime(pattern_days_df['datetime'].iloc[-1]).strftime('%Y-%m-%d')}**")
                            st.dataframe(
                                pattern_days_df[['datetime', 'open', 'high', 'low', 'close', 'daily_range']].style.format({
                                    'open': '{:.2f}',
                                    'high': '{:.2f}',
                                    'low': '{:.2f}',
                                    'close': '{:.2f}',
                                    'daily_range': '{:.2f}'
                                }),
                                use_container_width=True
                            )
                    
                    # WS Pattern Analysis
                    st.markdown(f"### {pattern_days}-Day Wide Spread (WS) Analysis")
                    ws_df = df[df['is_ws']].copy()
                    if not ws_df.empty:
                        # Calculate statistics
                        ws_stats = {
                            'Total WS Patterns': len(ws_df),
                            'Average Range on Last Day': f"{ws_df['daily_range'].mean():.2f} points",
                            'Max Range on Last Day': f"{ws_df['daily_range'].max():.2f} points",
                            'Next Day Up': f"{len(ws_df[ws_df['next_day_return'] > 0])} ({len(ws_df[ws_df['next_day_return'] > 0])/len(ws_df)*100:.1f}%)",
                            'Next Day Down': f"{len(ws_df[ws_df['next_day_return'] < 0])} ({len(ws_df[ws_df['next_day_return'] < 0])/len(ws_df)*100:.1f}%)",
                            'Average Next Day Return': f"{ws_df['next_day_return'].mean():.2f}%",
                            'Max Next Day Return': f"{ws_df['next_day_return'].max():.2f}%",
                            'Min Next Day Return': f"{ws_df['next_day_return'].min():.2f}%"
                        }
                        st.dataframe(pd.DataFrame([ws_stats]).T, use_container_width=True)
                        
                        # Show example patterns
                        st.markdown("#### Example WS Pattern")
                        if not ws_df.empty:
                            idx = ws_df.index[0]  # Get only the first pattern
                            pattern_days_df = df.loc[idx-pattern_days+1:idx]
                            st.markdown(f"**Pattern ending on {pd.to_datetime(pattern_days_df['datetime'].iloc[-1]).strftime('%Y-%m-%d')}**")
                            st.dataframe(
                                pattern_days_df[['datetime', 'open', 'high', 'low', 'close', 'daily_range']].style.format({
                                    'open': '{:.2f}',
                                    'high': '{:.2f}',
                                    'low': '{:.2f}',
                                    'close': '{:.2f}',
                                    'daily_range': '{:.2f}'
                                }),
                                use_container_width=True
                            )
                    
                    # Show all patterns
                    st.markdown("### All Patterns")
                    display_df = df[df['is_nr'] | df['is_ws']].copy()
                    display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d')
                    display_df['next_day_close_higher'] = display_df['next_day_return'] > 0
                    display_df['next_day_close_lower'] = display_df['next_day_return'] < 0
                    
                    st.dataframe(
                        display_df[['datetime', 'open', 'high', 'low', 'close', 'daily_range', 
                                  'is_nr', 'is_ws', 'next_day_close_higher', 'next_day_close_lower', 
                                  'next_day_return']].style.format({
                            'open': '{:.2f}',
                            'high': '{:.2f}',
                            'low': '{:.2f}',
                            'close': '{:.2f}',
                            'daily_range': '{:.2f}',
                            'next_day_return': '{:.2f}%'
                        }),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "nr-ws-pattern":
                st.markdown("## NR-WS-Pattern Report")
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    df = df.sort_values('datetime')
                    df['daily_range'] = df['high'] - df['low']
                    df['daily_return'] = df['close'].pct_change() * 100
                    df['next_day_return'] = df['daily_return'].shift(-1)
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    df = df[df['weekday'].isin(selected_weekdays)]
                    def is_nr_pattern(ranges):
                        return all(ranges.iloc[i] < ranges.iloc[i-1] for i in range(1, len(ranges)))
                    def is_ws_pattern(ranges):
                        return all(ranges.iloc[i] > ranges.iloc[i-1] for i in range(1, len(ranges)))
                    def pattern_summary_auto(df, mask, label, N):
                        pat_df = df[mask].copy()
                        count = len(pat_df)
                        next_up = (pat_df['next_day_return'] > 0).sum()
                        next_down = (pat_df['next_day_return'] < 0).sum()
                        avg_pos = pat_df[pat_df['next_day_return'] > 0]['next_day_return'].mean() if count else 0
                        avg_neg = pat_df[pat_df['next_day_return'] < 0]['next_day_return'].mean() if count else 0
                        max_pos = pat_df['next_day_return'].max() if count else 0
                        max_neg = pat_df['next_day_return'].min() if count else 0
                        return {
                            'Pattern': label,
                            'N': N,
                            'Count': count,
                            'Next Day Up': f"{next_up} ({(next_up/count*100) if count else 0:.1f}%)",
                            'Next Day Down': f"{next_down} ({(next_down/count*100) if count else 0:.1f}%)",
                            'Avg +ve Return': f"{avg_pos:.2f}%",
                            'Avg -ve Return': f"{avg_neg:.2f}%",
                            'Max +ve Return': f"{max_pos:.2f}%",
                            'Max -ve Return': f"{max_neg:.2f}%"
                        }
                    import itertools
                    nr_rows = []
                    ws_rows = []
                    for N in range(2, 6):
                        # Generate all up/down patterns for N
                        patterns = list(itertools.product(['D', 'U'], repeat=N))
                        for pat in patterns:
                            pat_str = ''.join(pat)
                            # Build mask for this pattern
                            def match_pattern(returns):
                                return all((r < 0 if p == 'D' else r > 0) for r, p in zip(returns, pat))
                            pat_mask = df['daily_return'].rolling(N).apply(lambda x: match_pattern(x), raw=True) == 1
                            nr_mask = df['daily_range'].rolling(N).apply(is_nr_pattern, raw=False) == 1
                            ws_mask = df['daily_range'].rolling(N).apply(is_ws_pattern, raw=False) == 1
                            # Only consider the last day of each sequence
                            nr_final_mask = pat_mask & nr_mask
                            ws_final_mask = pat_mask & ws_mask
                            nr_rows.append(pattern_summary_auto(df, nr_final_mask, pat_str, N))
                            ws_rows.append(pattern_summary_auto(df, ws_final_mask, pat_str, N))
                    st.markdown("### NR + Pattern Table (All N)")
                    st.dataframe(pd.DataFrame([row for row in nr_rows if row['Count'] > 0]), use_container_width=True)
                    st.markdown("### WS + Pattern Table (All N)")
                    st.dataframe(pd.DataFrame([row for row in ws_rows if row['Count'] > 0]), use_container_width=True)

                    # NR4/WS4 and NR7/WS7 tables
                    for N in [4, 7]:
                        patterns = list(itertools.product(['D', 'U'], repeat=N))
                        nr_rows_n = []
                        ws_rows_n = []
                        for pat in patterns:
                            pat_str = ''.join(pat)
                            def match_pattern(returns):
                                return all((r < 0 if p == 'D' else r > 0) for r, p in zip(returns, pat))
                            pat_mask = df['daily_return'].rolling(N).apply(lambda x: match_pattern(x), raw=True) == 1
                            nr_mask = df['daily_range'].rolling(N).apply(is_nr_pattern, raw=False) == 1
                            ws_mask = df['daily_range'].rolling(N).apply(is_ws_pattern, raw=False) == 1
                            nr_final_mask = pat_mask & nr_mask
                            ws_final_mask = pat_mask & ws_mask
                            nr_rows_n.append(pattern_summary_auto(df, nr_final_mask, pat_str, N))
                            ws_rows_n.append(pattern_summary_auto(df, ws_final_mask, pat_str, N))
                        st.markdown(f"### NR{N} + Pattern Table")
                        st.dataframe(pd.DataFrame([row for row in nr_rows_n if row['Count'] > 0]), use_container_width=True)
                        st.markdown(f"### WS{N} + Pattern Table")
                        st.dataframe(pd.DataFrame([row for row in ws_rows_n if row['Count'] > 0]), use_container_width=True)
                else:
                    st.error("No data available for the selected period.")
            elif selected_report == "Inside Bar":
                show_inside_bar_analysis(db, index_type, date_range)
            elif selected_report == "candle-pattern":
                st.markdown("## Pattern Analysis")
                # Get data for analysis
                df = db.get_index_data(index_type, date_range, "daily")
                if df is not None and not df.empty:
                    # Sort by date
                    df = df.sort_values('datetime')
                    
                    # Calculate candle properties
                    df['body_size'] = abs(df['close'] - df['open'])
                    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
                    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
                    df['total_size'] = df['high'] - df['low']
                    df['body_ratio'] = df['body_size'] / df['total_size']
                    df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_size']
                    df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_size']
                    
                    # Define pattern
                    df['pattern'] = 'Not Shooting Star/Doji'
                    # Doji: small body, body near high
                    doji_mask = (df['body_ratio'] < 0.1) & (df[['open', 'close']].max(axis=1) > df['high'] - 0.3 * (df['high'] - df['low']))
                    df.loc[doji_mask, 'pattern'] = 'Doji'
                    # Shooting Star: small body, long upper shadow, small lower shadow, body near low
                    shooting_star_mask = (
                        (df['body_ratio'] < 0.3) &
                        (df['upper_shadow_ratio'] > 0.6) &
                        (df['lower_shadow_ratio'] < 0.1) &
                        (df[['open', 'close']].min(axis=1) < df['low'] + 0.3 * (df['high'] - df['low']))
                    )
                    df.loc[shooting_star_mask, 'pattern'] = 'Shooting Star'
                    
                    # Calculate next day returns
                    df['next_day_return'] = df['close'].pct_change(1).shift(-1) * 100
                    df['next_day_close'] = df['close'].shift(-1)
                    
                    # Next day close vs pattern close
                    df['next_day_close_vs_pattern'] = '-'
                    pattern_list = ['Shooting Star', 'Doji', 'Not Shooting Star/Doji']
                    for pat in ['Shooting Star', 'Doji']:
                        mask = df['pattern'] == pat
                        df.loc[mask & (df['next_day_close'] > df['close']), 'next_day_close_vs_pattern'] = 'Above'
                        df.loc[mask & (df['next_day_close'] < df['close']), 'next_day_close_vs_pattern'] = 'Below'
                        df.loc[mask & (df['next_day_close'] == df['close']), 'next_day_close_vs_pattern'] = 'Equal'
                    
                    # Filter for weekdays
                    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
                    df = df[df['weekday'].isin(selected_weekdays)]
                    
                    # Display pattern statistics
                    st.markdown("### Pattern Statistics")
                    pattern_stats = df.groupby('pattern').agg(
                        count=('pattern', 'count'),
                        avg_next_return=('next_day_return', 'mean'),
                        success_rate=('next_day_return', lambda x: (x > 0).mean() * 100)
                    )
                    # Reindex to ensure all patterns are present
                    pattern_stats = pattern_stats.reindex(pattern_list, fill_value=0).reset_index()
                    # Add only two columns: next_day_close_above_pattern and next_day_close_below_pattern
                    above_counts = df[df['next_day_close_vs_pattern'] == 'Above'].groupby('pattern').size()
                    below_counts = df[df['next_day_close_vs_pattern'] == 'Below'].groupby('pattern').size()
                    pattern_stats['next_day_close_above_pattern'] = pattern_stats['pattern'].map(above_counts).fillna(0).astype(int)
                    pattern_stats['next_day_close_below_pattern'] = pattern_stats['pattern'].map(below_counts).fillna(0).astype(int)
                    # Reorder columns for clarity
                    stat_cols = ['pattern', 'count', 'avg_next_return', 'success_rate',
                        'next_day_close_above_pattern', 'next_day_close_below_pattern']
                    pattern_stats = pattern_stats[[col for col in stat_cols if col in pattern_stats.columns]]
                    
                    # Display pattern statistics
                    st.dataframe(
                        pattern_stats.style.format({
                            'avg_next_return': '{:.2f}%',
                            'success_rate': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Display raw data
                    st.markdown("### Raw Pattern Data")
                    display_df = df[['datetime', 'open', 'high', 'low', 'close', 'pattern', 'next_day_return', 'next_day_close_vs_pattern']].copy()
                    display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d')
                    st.dataframe(
                        display_df.style.format({
                            'next_day_return': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("No data available for the selected period.")

if __name__ == "__main__":
    main() 