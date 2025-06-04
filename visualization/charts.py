import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_price_chart(df, title="Price Chart"):
    """Create an interactive price chart using Plotly (no volume subplot)"""
    # Create figure with only one row for price
    fig = go.Figure()

    # Format hover text for candlestick
    hover_text = []
    for i in range(len(df)):
        hover_text.append(
            f"Time: {df['datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')}<br>" +
            f"Open: {df['open'].iloc[i]:.2f}<br>" +
            f"High: {df['high'].iloc[i]:.2f}<br>" +
            f"Low: {df['low'].iloc[i]:.2f}<br>" +
            f"Close: {df['close'].iloc[i]:.2f}"
        )

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a',  # Green for up candles
        decreasing_line_color='#ef5350',  # Red for down candles
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        hovertext=hover_text,
        hoverlabel=dict(
            bgcolor='white',
            font=dict(size=12)
        )
    ))

    # Update layout for large chart and better zoom/scroll
    fig.update_layout(
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        height=900,
        width=1800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        dragmode='pan',
    )

    # Update y-axis
    fig.update_yaxes(
        title_text="Price",
        gridcolor='lightgray',
        zerolinecolor='lightgray',
        showgrid=True,
        zeroline=True,
        tickformat='.2f',
        hoverformat='.2f',
        fixedrange=False  # Allow vertical zoom/scroll
    )

    # Update x-axis
    fig.update_xaxes(
        gridcolor='lightgray',
        zerolinecolor='lightgray',
        showgrid=True,
        zeroline=True,
        rangeslider=dict(visible=False),
        type="date",
        tickformat='%H:%M:%S',
        hoverformat='%Y-%m-%d %H:%M:%S',
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=4, label="4h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(255, 255, 255, 0.8)',
            activecolor='#26a69a'
        ),
        fixedrange=False  # Allow horizontal zoom/scroll
    )

    return fig

def plot_volume_chart(df, title="Volume Chart"):
    """Create an interactive volume chart using Plotly"""
    fig = go.Figure()
    
    # Add volume bars
    colors = ['red' if row['close'] < row['open'] else 'green' 
              for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df['datetime'],
        y=df['volume'],
        name='Volume',
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

def plot_technical_indicators(df, indicators, title="Technical Analysis", show_price=True, trades=None):
    """Create a chart with price and technical indicators (no volume), and optionally plot trade entries/exits."""
    fig = go.Figure()
    if show_price:
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
    for indicator in indicators:
        if indicator == 'supertrend' and 'supertrend' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['supertrend'],
                name='SuperTrend',
                line=dict(width=2, color='green'),
                mode='lines'
            ))
        elif indicator in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[indicator],
                name=indicator,
                line=dict(width=1)
            ))
    # Plot trade entries and exits if provided
    if trades is not None and len(trades) > 0:
        entry_x, entry_y, entry_color, entry_text = [], [], [], []
        exit_x, exit_y, exit_text = [], [], []
        for t in trades:
            # Entry
            entry_x.append(t['entry_time'])
            entry_y.append(t['entry_price'])
            entry_color.append('green' if t['direction'] == 'long' else 'red')
            entry_text.append(f"Entry ({t['direction']})<br>{t['entry_time']}<br>{t['entry_price']:.2f}")
            # Exit
            exit_x.append(t['exit_time'])
            exit_y.append(t['exit_price'])
            exit_text.append(f"Exit<br>{t['exit_time']}<br>{t['exit_price']:.2f}")
        # Entry markers
        fig.add_trace(go.Scatter(
            x=entry_x,
            y=entry_y,
            mode='markers',
            marker=dict(symbol='triangle-up', size=14, color=entry_color, line=dict(width=2, color='black')),
            name='Trade Entry',
            text=entry_text,
            hoverinfo='text',
            showlegend=True
        ))
        # Exit markers
        fig.add_trace(go.Scatter(
            x=exit_x,
            y=exit_y,
            mode='markers',
            marker=dict(symbol='x', size=12, color='blue', line=dict(width=2, color='black')),
            name='Trade Exit',
            text=exit_text,
            hoverinfo='text',
            showlegend=True
        ))
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )
    return fig

def plot_support_resistance(df, support_levels, resistance_levels, title="Support and Resistance"):
    """Create a chart with price and support/resistance levels"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add support levels
    for level in support_levels:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=[level] * len(df),
            name=f'Support {level:.2f}',
            line=dict(dash='dash', color='green')
        ))
    
    # Add resistance levels
    for level in resistance_levels:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=[level] * len(df),
            name=f'Resistance {level:.2f}',
            line=dict(dash='dash', color='red')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig
