# indicators_analysis.py
# This module contains the Indicators Analysis page logic, including orb-gap analysis.
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

def calculate_supertrend(df, period=10, multiplier=3.0):
    df = df.copy()
    hl2 = (df['high'] + df['low']) / 2
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df.at[df.index[i], 'in_uptrend'] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df.at[df.index[i], 'in_uptrend'] = False
        else:
            df.at[df.index[i], 'in_uptrend'] = df['in_uptrend'].iloc[i-1]
    return df

def show_indicators_analysis_page(db):
    st.markdown("# Indicators Analysis")
    
    # Check if database is initialized
    if db is None:
        st.error("Database connection failed. Please try refreshing the page.")
        return
        
    try:
        tables = db.get_table_names()
        if not tables:
            st.error("No tables found in the database. Please import data first.")
            return
            
        base_tables = [table.replace('_1min', '') for table in tables if table.endswith('_1min')]
        base_tables = sorted(list(set(base_tables)))
        
        if not base_tables:
            st.error("No index data found. Please import data first.")
            return
        
        # --- Place all filters side by side ---
        col1, col2, col3, col4, col5 = st.columns([1.2, 2, 1.2, 1.2, 1.2])

        with col1:
            index_type = st.selectbox("Index", base_tables, key="indicators_analysis_index")
        with col2:
            table_name = f"{index_type}_daily"
            start_date, end_date = db.get_available_dates(table_name)
            
            if not start_date or not end_date:
                st.error(f"No data available for {index_type}. Please import data first.")
                return
            
            date_range = st.date_input(
                "Date Range",
                value=(end_date - pd.Timedelta(days=30), end_date),
                min_value=start_date,
                max_value=end_date,
                key="indicators_analysis_date_range"
            )
        with col3:
            timeframe = st.selectbox(
                "Timeframe",
                ["1 Minute", "5 Minute", "15 Minute", "1 Hour", "Daily"],
                key="indicators_timeframe"
            )
        with col4:
            indicator = st.selectbox(
                "Indicator",
                ["EMA", "SMA", "SuperTrend", "MACD", "Bollinger Bands"],
                key="selected_indicator"
            )
        with col5:
            trend_direction = st.radio(
                "Trend",
                ["Uptrend", "Downtrend"],
                key="trend_direction"
            )
        
        # Add ORB minutes input
        orb_minutes = st.number_input(
            "ORB Minutes",
            min_value=1,
            max_value=60,
            value=15,
            help="Number of minutes to consider for Open Range Breakout calculation"
        )
        
        # SuperTrend parameter inputs
        if indicator == "SuperTrend":
            st.markdown("#### SuperTrend Parameters")
            period = st.number_input("Period", min_value=1, max_value=100, value=10, step=1, key="supertrend_period")
            multiplier = st.number_input("Multiplier", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="supertrend_multiplier")
        else:
            period = None
            multiplier = None
        
        # Convert timeframe to database format
        timeframe_map = {
            "1 Minute": "1min",
            "5 Minute": "5min",
            "15 Minute": "15min",
            "1 Hour": "1hour",
            "Daily": "daily"
        }
        db_timeframe = timeframe_map[timeframe]
        
        # If timeframe is daily, disable ORB analysis
        if db_timeframe == "daily":
            st.warning("ORB analysis requires intraday data (1min, 5min, 15min). Please select a lower timeframe.")
            return
        
        # --- 1. Get full data for the index and timeframe (for all available dates) ---
        full_df = db.get_index_data(index_type, (start_date, end_date), db_timeframe)
        if full_df is None or full_df.empty:
            st.error("No data available for the selected criteria. Please try different parameters.")
            return
        
        # --- 2. Calculate indicator on the full data ---
        if indicator == "SuperTrend" and period and multiplier:
            full_df = calculate_supertrend(full_df, period=period, multiplier=multiplier)
            # Filter by trend direction
            if trend_direction == "Uptrend":
                full_df = full_df[full_df['in_uptrend']]
            else:
                full_df = full_df[~full_df['in_uptrend']]
        else:
            # Fallback: filter by close/trend as before
            full_df['trend'] = full_df['close'].rolling(window=20).mean()
            if trend_direction == "Uptrend":
                full_df = full_df[full_df['close'] > full_df['trend']]
            else:
                full_df = full_df[full_df['close'] < full_df['trend']]
        
        # --- 3. Filter to selected date range for display and analysis ---
        mask = (full_df['datetime'].dt.date >= date_range[0]) & (full_df['datetime'].dt.date <= date_range[1])
        df = full_df[mask].copy()
        
        if df.empty:
            st.warning("No data matches the selected trend direction. Please try different parameters.")
            return
        
        # --- 4. For the filtered days, get 1min data and perform ORB analysis ---
        filtered_dates = df['datetime'].dt.date.unique()
        if len(filtered_dates) == 0:
            st.warning("No days available for ORB analysis after filtering.")
            return
        
        # Get 1min data for these days
        one_min_df = db.get_index_data(index_type, (min(filtered_dates), max(filtered_dates)), "1min")
        if one_min_df is None or one_min_df.empty:
            st.warning("No 1min data available for ORB analysis on the filtered days.")
            return
        one_min_df = one_min_df[one_min_df['datetime'].dt.date.isin(filtered_dates)].copy()
        if one_min_df.empty:
            st.warning("No 1min data available for the filtered days.")
            return
        
        # Calculate ORB for each filtered day using 1min data
        def calculate_orb_for_day(day_data):
            start_time = day_data['datetime'].iloc[0].time()
            orb_end_time = (datetime.combine(datetime.today(), start_time) + timedelta(minutes=orb_minutes)).time()
            first_n_minutes = day_data[day_data['datetime'].dt.time <= orb_end_time]
            if first_n_minutes.empty:
                return pd.Series({
                    'orb_high': day_data['high'].iloc[0],
                    'orb_low': day_data['low'].iloc[0],
                    'first_break': 'No Break',
                    'close_price': day_data['close'].iloc[-1]
                })
            orb_high = first_n_minutes['high'].max()
            orb_low = first_n_minutes['low'].min()
            remaining_data = day_data[day_data['datetime'].dt.time > orb_end_time]
            for _, row in remaining_data.iterrows():
                if row['high'] > orb_high:
                    return pd.Series({'orb_high': orb_high, 'orb_low': orb_low, 'first_break': 'Up', 'close_price': day_data['close'].iloc[-1]})
                elif row['low'] < orb_low:
                    return pd.Series({'orb_high': orb_high, 'orb_low': orb_low, 'first_break': 'Down', 'close_price': day_data['close'].iloc[-1]})
            return pd.Series({'orb_high': orb_high, 'orb_low': orb_low, 'first_break': 'No Break', 'close_price': day_data['close'].iloc[-1]})
        one_min_df['date'] = one_min_df['datetime'].dt.date
        orb_results = one_min_df.groupby('date').apply(calculate_orb_for_day).reset_index()
        
        # Merge ORB results back to the filtered indicator df (by date)
        df['date'] = df['datetime'].dt.date
        df = df.merge(orb_results, on='date', how='left')
        
        # Display raw data
        st.markdown("## Raw Data")
        st.dataframe(
            df[['datetime', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True),
            use_container_width=True,
            height=400
        )
        
        # --- ORB-style Gauge Charts and Table ---
        # Placeholder logic: create a 'first_break' column for demonstration
        # In a real scenario, replace this with your actual event logic
        if 'first_break' not in df.columns:
            np.random.seed(0)
            choices = ['Up', 'Down', 'No Break']
            df['first_break'] = np.random.choice(choices, size=len(df), p=[0.4, 0.4, 0.2])

        total_days = len(df)
        up_first = (df['first_break'] == 'Up').sum()
        down_first = (df['first_break'] == 'Down').sum()
        no_break = (df['first_break'] == 'No Break').sum()

        up_pct = (up_first / total_days * 100) if total_days > 0 else 0
        down_pct = (down_first / total_days * 100) if total_days > 0 else 0
        no_break_pct = (no_break / total_days * 100) if total_days > 0 else 0

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
                value = no_break_pct,
                title = {'text': "No Break %"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': 'green'}}
            ))
            st.plotly_chart(fig_none, use_container_width=True)

        # --- Close Price Distribution Relative to ORB Pie Charts ---
        st.markdown("## Close Price Distribution Relative to ORB")
        # Placeholder: create 'close_price', 'orb_high', 'orb_low' if not present
        if 'close_price' not in df.columns:
            df['close_price'] = df['close']
        if 'orb_high' not in df.columns:
            df['orb_high'] = df['high']
        if 'orb_low' not in df.columns:
            df['orb_low'] = df['low']

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Up First Break: Close Distribution")
            up_breaks = df[df['first_break'] == 'Up']
            if not up_breaks.empty and 'close_price' in up_breaks.columns and 'orb_high' in up_breaks.columns and 'orb_low' in up_breaks.columns:
                up_breaks['close_position'] = np.where(
                    up_breaks['close_price'] > up_breaks['orb_high'],
                    'Above ORB',
                    np.where(
                        up_breaks['close_price'] < up_breaks['orb_low'],
                        'Below ORB',
                        'In ORB'
                    )
                )
                up_close_dist = up_breaks['close_position'].value_counts(normalize=True) * 100
                fig_up = go.Figure(data=[go.Pie(
                    labels=up_close_dist.index,
                    values=up_close_dist.values,
                    hole=.3,
                    title='Up First Break Close Distribution'
                )])
                fig_up.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_up, use_container_width=True)
            else:
                st.info("No Up First breaks found or required columns missing.")
        with col2:
            st.subheader("Down First Break: Close Distribution")
            down_breaks = df[df['first_break'] == 'Down']
            if not down_breaks.empty and 'close_price' in down_breaks.columns and 'orb_high' in down_breaks.columns and 'orb_low' in down_breaks.columns:
                down_breaks['close_position'] = np.where(
                    down_breaks['close_price'] > down_breaks['orb_high'],
                    'Above ORB',
                    np.where(
                        down_breaks['close_price'] < down_breaks['orb_low'],
                        'Below ORB',
                        'In ORB'
                    )
                )
                down_close_dist = down_breaks['close_position'].value_counts(normalize=True) * 100
                fig_down = go.Figure(data=[go.Pie(
                    labels=down_close_dist.index,
                    values=down_close_dist.values,
                    hole=.3,
                    title='Down First Break Close Distribution'
                )])
                fig_down.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_down, use_container_width=True)
            else:
                st.info("No Down First breaks found or required columns missing.")

        st.markdown("### ORB Analysis for Filtered Days")
        st.dataframe(df, use_container_width=True)

        # Display basic statistics
        st.markdown("## Basic Statistics")
        stats = {
            'Total Bars': len(df),
            'Average Volume': f"{df['volume'].mean():,.0f}",
            'Price Range': f"{df['high'].max() - df['low'].min():.2f}",
            'Average Daily Range': f"{((df['high'] - df['low']) / df['low'] * 100).mean():.2f}%"
        }
        
        col1, col2, col3, col4 = st.columns(4)
        for (stat_name, stat_value), col in zip(stats.items(), [col1, col2, col3, col4]):
            with col:
                st.metric(stat_name, stat_value)
                
        # Debug information below filters
        if st.checkbox("Show Debug Information"):
            st.write("Database Connection Info:")
            st.write(f"Index Type: {index_type}")
            st.write(f"Timeframe: {db_timeframe}")
            st.write(f"Date Range: {date_range}")
            st.write(f"Data Shape: {df.shape}")
            st.write("First few rows of data:")
            st.write(df.head())
                
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check your database connection and try again.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or importing data first.") 