import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_gap_analysis(df):
    """Create a comprehensive gap analysis visualization"""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price and Gaps', 'Gap Distribution', 'Gap Fill Timeline'),
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

    # Add gap annotations
    for i in range(1, len(df)):
        gap = df['gap'].iloc[i]
        if abs(gap) > 0.3:  # Only show medium and high gaps
            color = 'green' if gap > 0 else 'red'
            fig.add_annotation(
                x=df['datetime'].iloc[i],
                y=df['open'].iloc[i],
                text=f"{gap:.2f}%",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                ax=0,
                ay=-40 if gap > 0 else 40,
                font=dict(color=color, size=10),
                row=1, col=1
            )

    # Add gap distribution bar chart
    gap_categories = ['Flat', 'Medium', 'High', 'N/A']
    gap_counts = df['gap_category'].value_counts().reindex(gap_categories, fill_value=0)
    total_gaps = gap_counts.sum()
    colors = {'Flat': 'gray', 'Medium': 'orange', 'High': 'red', 'N/A': 'lightgray'}
    
    bar_x = []
    bar_y = []
    bar_text = []
    bar_colors = []
    for category in gap_categories:
        count = gap_counts[category]
        percent = (count / total_gaps) * 100 if total_gaps > 0 else 0
        bar_x.append(category)
        bar_y.append(count)
        bar_text.append(f"{count} ({percent:.1f}%)")
        bar_colors.append(colors.get(category, 'gray'))
    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=bar_y,
            marker_color=bar_colors,
            text=bar_text,
            textposition='auto',
            textfont=dict(size=13),
            width=0.5,
            name='Gap Category',
        ),
        row=2, col=1
    )

    # Add gap fill timeline
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    if not gap_df.empty:
        fig.add_trace(
            go.Scatter(
                x=gap_df['datetime'],
                y=gap_df['days_to_fill'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=gap_df['gap_filled'].map({True: 'green', False: 'red'}),
                    symbol='circle'
                ),
                name='Days to Fill'
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        title='Gap Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        template='plotly_dark',
        barmode='group',
        bargap=0.35,
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Days to Fill", row=3, col=1)
    # Add improved title for gap distribution
    fig.layout.annotations[1].text = "Gap Distribution (Count and % of Total)"
    return fig

def plot_gap_fill_analysis(df):
    """Create a visualization for gap fill analysis"""
    # Filter for medium and high gaps
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    
    # Calculate fill statistics
    total_gaps = len(gap_df)
    filled_gaps = gap_df['gap_filled'].sum()
    fill_rate = (filled_gaps / total_gaps * 100) if total_gaps > 0 else 0
    
    # Gap up and gap down
    up_gaps = gap_df[gap_df['gap_direction'] == 'Up']
    down_gaps = gap_df[gap_df['gap_direction'] == 'Down']
    up_filled = up_gaps['gap_filled'].sum()
    up_unfilled = len(up_gaps) - up_filled
    down_filled = down_gaps['gap_filled'].sum()
    down_unfilled = len(down_gaps) - down_filled
    
    # Create figure with subplots (3 donuts)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Overall Gap Fill Status', 'Gap Up Fill Status', 'Gap Down Fill Status'),
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
    )
    
    # Overall donut
    fig.add_trace(
        go.Pie(
            labels=['Filled', 'Unfilled'],
            values=[filled_gaps, total_gaps - filled_gaps],
            marker_colors=['green', 'red'],
            hole=0.5,
            name='Overall',
            textinfo='percent+value',
            showlegend=False
        ),
        row=1, col=1
    )
    # Gap Up donut
    fig.add_trace(
        go.Pie(
            labels=['Filled', 'Unfilled'],
            values=[up_filled, up_unfilled],
            marker_colors=['green', 'red'],
            hole=0.5,
            name='Gap Up',
            textinfo='percent+value',
            showlegend=False
        ),
        row=1, col=2
    )
    # Gap Down donut
    fig.add_trace(
        go.Pie(
            labels=['Filled', 'Unfilled'],
            values=[down_filled, down_unfilled],
            marker_colors=['green', 'red'],
            hole=0.5,
            name='Gap Down',
            textinfo='percent+value',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=f'Gap Fill Analysis (Overall Fill Rate: {fill_rate:.1f}%)',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    return fig

def plot_gap_size_distribution(df):
    """Create a visualization for gap size distribution"""
    # Filter for medium and high gaps
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram for gap sizes
    fig.add_trace(
        go.Histogram(
            x=gap_df['gap'],
            nbinsx=20,
            name='Gap Size Distribution',
            marker_color='blue',
            opacity=0.7
        )
    )
    
    # Add vertical lines for category boundaries
    fig.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Medium Gap")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Gap")
    fig.add_vline(x=-0.3, line_dash="dash", line_color="orange")
    fig.add_vline(x=-0.7, line_dash="dash", line_color="red")
    
    # Update layout
    fig.update_layout(
        title='Gap Size Distribution',
        xaxis_title='Gap Size (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_gap_fill_timing(df):
    """Create a visualization for gap fill timing analysis"""
    # Filter for filled gaps
    filled_gaps = df[
        (df['gap_category'].isin(['Medium', 'High'])) & 
        (df['gap_filled'] == True)
    ].copy()
    
    if filled_gaps.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for fill timing
    fig.add_trace(
        go.Scatter(
            x=filled_gaps['gap_size'],
            y=filled_gaps['days_to_fill'],
            mode='markers',
            marker=dict(
                size=10,
                color=filled_gaps['gap_direction'].map({'Up': 'green', 'Down': 'red'}),
                symbol='circle'
            ),
            name='Fill Timing'
        )
    )
    
    # Add trend line
    z = np.polyfit(filled_gaps['gap_size'], filled_gaps['days_to_fill'], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=filled_gaps['gap_size'],
            y=p(filled_gaps['gap_size']),
            mode='lines',
            line=dict(color='yellow', dash='dash'),
            name='Trend Line'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Gap Fill Timing Analysis',
        xaxis_title='Gap Size (%)',
        yaxis_title='Days to Fill',
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_gap_direction_analysis(df):
    """Create detailed analysis of gap up and gap down patterns"""
    # Filter for medium and high gaps
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    
    if gap_df.empty:
        return None
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Gap Size Distribution by Direction',
            'Fill Rate by Direction',
            'Days to Fill by Direction',
            'Return Distribution by Direction'
        ),
        specs=[[{"type": "box"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # 1. Box plot of gap sizes by direction
    for direction in ['Up', 'Down']:
        fig.add_trace(
            go.Box(
                y=gap_df[gap_df['gap_direction'] == direction]['gap'],
                name=direction,
                marker_color='green' if direction == 'Up' else 'red'
            ),
            row=1, col=1
        )
    
    # 2. Pie chart of fill rates by direction
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        filled = dir_gaps['gap_filled'].sum()
        unfilled = len(dir_gaps) - filled
        fig.add_trace(
            go.Pie(
                labels=['Filled', 'Unfilled'],
                values=[filled, unfilled],
                name=direction,
                marker_colors=['green', 'red'],
                hole=0.4
            ),
            row=1, col=2
        )
    
    # 3. Bar chart of average days to fill by direction
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        avg_days = dir_gaps['days_to_fill'].mean()
        fig.add_trace(
            go.Bar(
                x=[direction],
                y=[avg_days],
                name=direction,
                marker_color='green' if direction == 'Up' else 'red'
            ),
            row=2, col=1
        )
    
    # 4. Histogram of returns by direction
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        fig.add_trace(
            go.Histogram(
                x=dir_gaps['fill_return'],
                name=direction,
                marker_color='green' if direction == 'Up' else 'red',
                opacity=0.7
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Gap Direction Analysis',
        height=800,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_gap_patterns(df):
    """Create visualizations for gap patterns"""
    # Filter for medium and high gaps
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    
    if gap_df.empty:
        return None
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Gaps by Weekday',
            'Gaps by Month',
            'Consecutive Gaps',
            'Gap Clusters'
        )
    )
    
    # 1. Gaps by weekday
    weekday_counts = gap_df['weekday'].value_counts()
    fig.add_trace(
        go.Bar(
            x=weekday_counts.index,
            y=weekday_counts.values,
            name='Gaps by Weekday'
        ),
        row=1, col=1
    )
    
    # 2. Gaps by month
    monthly_counts = gap_df['month'].value_counts()
    fig.add_trace(
        go.Bar(
            x=monthly_counts.index,
            y=monthly_counts.values,
            name='Gaps by Month'
        ),
        row=1, col=2
    )
    
    # 3. Consecutive gaps
    gap_df['consecutive'] = (gap_df['gap_direction'] == gap_df['gap_direction'].shift(1))
    consecutive_counts = gap_df['consecutive'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=['Consecutive', 'Non-consecutive'],
            values=consecutive_counts.values,
            name='Consecutive Gaps'
        ),
        row=2, col=1
    )
    
    # 4. Gap clusters
    gap_df['cluster'] = (gap_df['datetime'].diff() > pd.Timedelta(days=5)).cumsum()
    cluster_sizes = gap_df.groupby('cluster').size()
    fig.add_trace(
        go.Histogram(
            x=cluster_sizes,
            name='Gap Cluster Sizes'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Gap Pattern Analysis',
        height=800,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_gap_risk_reward(df):
    """Create visualizations for gap risk-reward analysis"""
    # Filter for medium and high gaps
    gap_df = df[df['gap_category'].isin(['Medium', 'High'])].copy()
    
    if gap_df.empty:
        return None
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk-Reward by Direction',
            'Maximum Excursions',
            'Win Rate by Gap Size',
            'Return Distribution'
        )
    )
    
    # 1. Risk-Reward by direction
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        risk_reward = abs(dir_gaps['max_favorable_excursion'] / dir_gaps['max_adverse_excursion'])
        fig.add_trace(
            go.Box(
                y=risk_reward,
                name=direction,
                marker_color='green' if direction == 'Up' else 'red'
            ),
            row=1, col=1
        )
    
    # 2. Maximum excursions
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        fig.add_trace(
            go.Scatter(
                x=dir_gaps['max_adverse_excursion'],
                y=dir_gaps['max_favorable_excursion'],
                mode='markers',
                name=direction,
                marker=dict(
                    color='green' if direction == 'Up' else 'red',
                    size=10
                )
            ),
            row=1, col=2
        )
    
    # 3. Win rate by gap size
    gap_df['gap_size_bin'] = pd.qcut(gap_df['gap_size'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    win_rates = gap_df.groupby('gap_size_bin')['fill_return'].apply(lambda x: (x > 0).mean() * 100)
    fig.add_trace(
        go.Bar(
            x=win_rates.index,
            y=win_rates.values,
            name='Win Rate by Gap Size'
        ),
        row=2, col=1
    )
    
    # 4. Return distribution
    for direction in ['Up', 'Down']:
        dir_gaps = gap_df[gap_df['gap_direction'] == direction]
        fig.add_trace(
            go.Histogram(
                x=dir_gaps['fill_return'],
                name=direction,
                marker_color='green' if direction == 'Up' else 'red',
                opacity=0.7
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Gap Risk-Reward Analysis',
        height=800,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig 