import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_orb_analysis(df):
    """
    Create a comprehensive ORB analysis visualization.
    
    Parameters:
    df (pd.DataFrame): DataFrame with ORB signals and statistics
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price and ORB Signals', 'Range Size Distribution', 'Breakout Strength'),
        row_heights=[0.5, 0.25, 0.25]
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

    # Add ORB signals
    up_signals = df[df['orb_up']]
    down_signals = df[df['orb_down']]

    # Plot up breakouts
    fig.add_trace(
        go.Scatter(
            x=up_signals['datetime'],
            y=up_signals['close'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green'
            ),
            name='Up Breakout'
        ),
        row=1, col=1
    )

    # Plot down breakouts
    fig.add_trace(
        go.Scatter(
            x=down_signals['datetime'],
            y=down_signals['close'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red'
            ),
            name='Down Breakout'
        ),
        row=1, col=1
    )

    # Add range size distribution
    fig.add_trace(
        go.Histogram(
            x=df[df['orb_up'] | df['orb_down']]['range_size'],
            name='Range Size',
            marker_color='blue',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Add breakout strength
    fig.add_trace(
        go.Scatter(
            x=df[df['orb_up'] | df['orb_down']]['datetime'],
            y=df[df['orb_up'] | df['orb_down']]['breakout_strength'],
            mode='markers',
            marker=dict(
                size=8,
                color=df[df['orb_up'] | df['orb_down']]['breakout_strength'],
                colorscale='RdYlGn',
                showscale=True
            ),
            name='Breakout Strength'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title='Open Range Break (ORB) Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        height=800,
        showlegend=True
    )

    return fig

def plot_orb_success_analysis(df):
    """
    Create a visualization for ORB success rate analysis.
    
    Parameters:
    df (pd.DataFrame): DataFrame with ORB signals and statistics
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate success rates by range size
    df['range_size_bin'] = pd.qcut(df['range_size'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    success_by_size = df.groupby('range_size_bin').agg({
        'successful': 'mean',
        'orb_up': 'sum',
        'orb_down': 'sum'
    }).reset_index()

    # Create figure
    fig = go.Figure()

    # Add success rate bars
    fig.add_trace(
        go.Bar(
            x=success_by_size['range_size_bin'],
            y=success_by_size['successful'] * 100,
            name='Success Rate',
            marker_color='green',
            opacity=0.7
        )
    )

    # Add up/down breakout counts
    fig.add_trace(
        go.Bar(
            x=success_by_size['range_size_bin'],
            y=success_by_size['orb_up'],
            name='Up Breakouts',
            marker_color='lightgreen',
            opacity=0.5
        )
    )

    fig.add_trace(
        go.Bar(
            x=success_by_size['range_size_bin'],
            y=success_by_size['orb_down'],
            name='Down Breakouts',
            marker_color='lightcoral',
            opacity=0.5
        )
    )

    # Update layout
    fig.update_layout(
        title='ORB Success Rate by Range Size',
        xaxis_title='Range Size',
        yaxis_title='Count / Success Rate (%)',
        barmode='group',
        showlegend=True
    )

    return fig

def plot_orb_time_analysis(df):
    """
    Create a visualization for ORB time-based analysis.
    
    Parameters:
    df (pd.DataFrame): DataFrame with ORB signals and statistics
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object
    """
    # Extract hour from datetime
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Calculate success rates by hour
    success_by_hour = df.groupby('hour').agg({
        'successful': 'mean',
        'orb_up': 'sum',
        'orb_down': 'sum'
    }).reset_index()

    # Create figure
    fig = go.Figure()

    # Add success rate line
    fig.add_trace(
        go.Scatter(
            x=success_by_hour['hour'],
            y=success_by_hour['successful'] * 100,
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='green', width=2)
        )
    )

    # Add up/down breakout bars
    fig.add_trace(
        go.Bar(
            x=success_by_hour['hour'],
            y=success_by_hour['orb_up'],
            name='Up Breakouts',
            marker_color='lightgreen',
            opacity=0.5
        )
    )

    fig.add_trace(
        go.Bar(
            x=success_by_hour['hour'],
            y=success_by_hour['orb_down'],
            name='Down Breakouts',
            marker_color='lightcoral',
            opacity=0.5
        )
    )

    # Update layout
    fig.update_layout(
        title='ORB Analysis by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Count / Success Rate (%)',
        barmode='group',
        showlegend=True
    )

    return fig 