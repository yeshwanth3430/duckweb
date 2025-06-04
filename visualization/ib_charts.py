import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_ib_analysis(df):
    """
    Create a visualization for inside bar analysis.
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price and Inside Bars', 'Inside Bar Distribution'),
        row_heights=[0.7, 0.3]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Highlight inside bars
    ib_df = df[df['is_inside_bar']]
    fig.add_trace(
        go.Scatter(
            x=ib_df['datetime'],
            y=ib_df['high'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='blue',
                line=dict(width=1, color='blue')
            ),
            name='Inside Bar'
        ),
        row=1, col=1
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.3)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title='Inside Bar Analysis',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def plot_ib_size_distribution(df):
    """
    Create a visualization for inside bar size distribution.
    """
    ib_df = df[df['is_inside_bar']]
    
    fig = go.Figure()
    
    # Add histogram for IB size
    fig.add_trace(
        go.Histogram(
            x=ib_df['ib_size'],
            nbinsx=50,
            name='IB Size Distribution',
            marker_color='blue'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Inside Bar Size Distribution',
        xaxis_title='Inside Bar Size (%)',
        yaxis_title='Count',
        showlegend=True
    )
    
    return fig

def plot_ib_position_analysis(df):
    """
    Create a visualization for inside bar position analysis.
    """
    ib_df = df[df['is_inside_bar']]
    
    fig = go.Figure()
    
    # Add scatter plot for IB position vs size
    fig.add_trace(
        go.Scatter(
            x=ib_df['ib_position'],
            y=ib_df['ib_size'],
            mode='markers',
            marker=dict(
                size=10,
                color=ib_df['breakout_up'].astype(int),
                colorscale='RdYlGn',
                showscale=True
            ),
            name='IB Position vs Size'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Inside Bar Position Analysis',
        xaxis_title='Relative Position in Previous Bar',
        yaxis_title='Inside Bar Size (%)',
        showlegend=True
    )
    
    return fig

def plot_ib_breakout_analysis(df):
    """
    Create a visualization for inside bar breakout analysis.
    """
    ib_df = df[df['is_inside_bar']]
    
    # Calculate breakout statistics
    up_breakouts = ib_df['breakout_up'].sum()
    down_breakouts = ib_df['breakout_down'].sum()
    no_breakouts = len(ib_df) - up_breakouts - down_breakouts
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Up Breakout', 'Down Breakout', 'No Breakout'],
        values=[up_breakouts, down_breakouts, no_breakouts],
        hole=.3
    )])
    
    # Update layout
    fig.update_layout(
        title='Inside Bar Breakout Distribution',
        showlegend=True
    )
    
    return fig 