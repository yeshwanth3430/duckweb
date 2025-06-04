import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from analysis_tools.gap_analysis import analyze_gaps, get_gap_summary
from analysis_tools.volume_analysis import analyze_volume, find_support_resistance
from visualization.gap_charts import plot_gap_analysis, plot_gap_fill_analysis
from db_operations import DatabaseManager

def show_analysis_options(db, index_type, date_range, timeframe):
    """Display available analysis options"""
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Gap Analysis", "Volume Analysis", "Trend Analysis", "Support/Resistance"],
        label_visibility="collapsed"
    )
    
    if analysis_type:
        show_analysis_results(analysis_type, db, index_type, date_range, timeframe)

def show_analysis_results(analysis_type, db, index_type, date_range, timeframe):
    """Show analysis results based on selected type"""
    df = db.get_index_data(index_type, date_range, timeframe)
    
    if df is None or df.empty:
        st.error("No data available for analysis.")
        return
    
    if analysis_type == "Gap Analysis":
        df = analyze_gaps(df, date_range)
        display_gap_analysis(df, date_range)
    elif analysis_type == "Volume Analysis":
        df = analyze_volume(df)
        display_volume_analysis(df)
    elif analysis_type == "Support/Resistance":
        support_levels, resistance_levels = find_support_resistance(df)
        display_support_resistance(df, support_levels, resistance_levels)
    else:
        st.info(f"{analysis_type} is under development.")

def display_gap_analysis(df, date_range):
    """Display gap analysis results"""
    st.markdown("### Gap Analysis Results")
    st.markdown(f"**Date Range:** {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Gap", f"{df['gap'].iloc[-1]:.2f}%")
    with col2:
        st.metric("Gap Category", df['gap_category'].iloc[-1])
    with col3:
        total_days = len(df)
        st.metric("Total Days Analyzed", total_days)
    with col4:
        med_high = df[df['gap_category'].isin(["Medium", "High"])]
        total_med_high = len(med_high)
        filled_med_high = med_high['gap_filled'].sum() if total_med_high > 0 else 0
        percent_filled = (filled_med_high / total_med_high * 100) if total_med_high > 0 else 0
        st.metric("% Med/High Gaps Filled", f"{percent_filled:.1f}%")
    
    # Show gap distribution
    st.markdown("#### Gap Distribution")
    gap_counts = df['gap_category'].value_counts()
    st.bar_chart(gap_counts, use_container_width=True)
    
    # Show detailed gap history
    st.markdown("#### Gap History")
    gap_history = df[['datetime', 'open', 'close', 'gap', 'gap_category', 'gap_filled']]
    st.dataframe(
        gap_history.style.format({
            'gap': '{:.2f}%',
            'open': '{:.2f}',
            'close': '{:.2f}',
            'gap_filled': lambda x: 'Yes' if x is True else ('No' if x is False else '-')
        }).background_gradient(subset=['gap'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )

def display_volume_analysis(df):
    """Display volume analysis results"""
    st.markdown("### Volume Analysis Results")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Volume", f"{df['volume'].mean():,.0f}")
    with col2:
        high_volume_days = df['high_volume'].sum()
        st.metric("High Volume Days", f"{high_volume_days}")
    
    # Show volume chart
    st.markdown("#### Volume Trend")
    volume_data = df[['datetime', 'volume', 'volume_ma5', 'volume_ma20']]
    st.line_chart(volume_data.set_index('datetime'), use_container_width=True)
    
    # Show high volume days
    st.markdown("#### High Volume Days")
    high_volume_df = df[df['high_volume']][['datetime', 'open', 'close', 'volume']]
    st.dataframe(
        high_volume_df.style.format({
            'volume': '{:,.0f}',
            'open': '{:.2f}',
            'close': '{:.2f}'
        }),
        use_container_width=True
    )

def display_support_resistance(df, support_levels, resistance_levels):
    """Display support and resistance levels"""
    st.markdown("### Support and Resistance Levels")
    
    # Display levels
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Support Levels")
        for level in support_levels:
            st.markdown(f"- {level:.2f}")
    with col2:
        st.markdown("#### Resistance Levels")
        for level in resistance_levels:
            st.markdown(f"- {level:.2f}")
    
    # Show price chart with levels
    st.markdown("#### Price Chart with Levels")
    chart_data = df[['datetime', 'close']].set_index('datetime')
    st.line_chart(chart_data, use_container_width=True)
