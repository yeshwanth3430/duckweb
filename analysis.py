import streamlit as st
import pandas as pd
from datetime import timedelta, time
import plotly.graph_objects as go
import numpy as np

@st.cache_data(show_spinner=False)
def load_all_1min_data(_db, index_type, start_date, end_date):
    df = _db.get_index_data(index_type, (start_date, end_date), "1min")
    if df is not None and not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
    return df

def precompute_prev_close_map(df_all):
    # For each date, find the previous trading day and its close (at 15:30 or last available)
    prev_close_map = {}
    sorted_dates = sorted(df_all['date'].unique())
    last_close = None
    last_date = None
    for i, d in enumerate(sorted_dates):
        day_df = df_all[df_all['date'] == d]
        prev_close = None
        prev_date = None
        # Find previous trading day with data
        for j in range(i-1, -1, -1):
            prev_d = sorted_dates[j]
            prev_df = df_all[df_all['date'] == prev_d]
            if not prev_df.empty:
                prev_close_row = prev_df[prev_df['datetime'].dt.time == time(15, 30)]
                if not prev_close_row.empty:
                    prev_close = prev_close_row['close'].iloc[0]
                else:
                    prev_close = prev_df['close'].iloc[-1]
                prev_date = prev_d
                break
        prev_close_map[d] = (prev_close, prev_date)
    return prev_close_map

def show_analysis_page(db):
    st.markdown("# 1-Minute Data Viewer")
    # Get available indices from DuckDB
    tables = db.get_table_names()
    base_tables = [table.replace('_1min', '') for table in tables if table.endswith('_1min')]
    base_tables = sorted(list(set(base_tables)))
    if not base_tables:
        st.error("No data tables found in the database. Please check your database setup.")
        return
    # Side by side selectors
    col1, col2 = st.columns(2)
    with col1:
        index_type = st.selectbox("Select Index", base_tables, key="analysis_index")
    table_name = f"{index_type}_1min"
    start_date, end_date = db.get_available_dates(table_name)
    if not (start_date and end_date):
        st.error("No available dates for the selected index.")
        return
    with col2:
        selected_day = st.date_input(
            "Select Day",
            value=end_date,
            min_value=start_date,
            max_value=end_date,
            key="analysis_single_day"
        )
    # Time range selection (09:15 to 15:30)
    allowed_times = []
    h, m = 9, 15
    while True:
        allowed_times.append(time(h, m))
        m += 1
        if m == 60:
            m = 0
            h += 1
        if (h > 15) or (h == 15 and m > 30):
            break
    col_time1, col_time2 = st.columns(2)
    with col_time1:
        start_time = st.selectbox(
            "Start Time",
            options=allowed_times,
            index=0,
            format_func=lambda t: t.strftime('%H:%M'),
            key="analysis_time_start"
        )
    with col_time2:
        end_time = st.selectbox(
            "End Time",
            options=allowed_times,
            index=len(allowed_times)-1,
            format_func=lambda t: t.strftime('%H:%M'),
            key="analysis_time_end"
        )
    # Query for the full day (00:00:00 to 23:59:59)
    start_dt = pd.to_datetime(f"{selected_day} 00:00:00")
    end_dt = pd.to_datetime(f"{selected_day} 23:59:59")
    df = db.get_index_data(index_type, (start_dt, end_dt), "1min")

    # --- Load all 1-min data for this index (cached) ---
    df_all = load_all_1min_data(db, index_type, start_date, end_date)
    prev_close_map = precompute_prev_close_map(df_all)

    # --- Gap Analysis at the top ---
    gap_info = None
    warning_msg = None
    if df is not None and not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        open_915_row = df[df['datetime'].dt.time == time(9, 15)]
        open_915 = open_915_row['open'].iloc[0] if not open_915_row.empty else None
        if open_915 is None:
            warning_msg = "No 09:15 open found for the selected day. Gap cannot be calculated."
        prev_close, prev_close_day = prev_close_map.get(selected_day, (None, None))
        if open_915 is not None and prev_close is not None:
            gap_pct = ((open_915 - prev_close) / prev_close) * 100
            gap_type = "Gap Up" if gap_pct > 0 else ("Gap Down" if gap_pct < 0 else "Flat")
            gap_info = (open_915, prev_close, gap_type, gap_pct, prev_close_day)
    if warning_msg:
        st.warning(warning_msg)
    if gap_info:
        open_915, prev_close, gap_type, gap_pct, prev_close_day = gap_info
        st.markdown(f"### Gap Analysis for {selected_day}")
        colg1, colg2, colg3, colg4, colg5 = st.columns(5)
        with colg1:
            st.metric("09:15 Open", f"{open_915:.2f}")
        with colg2:
            st.metric("Prev Close", f"{prev_close:.2f}")
        with colg3:
            st.metric("Prev Day", str(prev_close_day))
        with colg4:
            st.metric("Gap Type", gap_type)
        with colg5:
            st.metric("Gap %", f"{gap_pct:.2f}%")

        # --- 2, 3, 4, 5-Day Close Pattern Analysis (arranged, with probabilities) ---
        st.markdown("### 2, 3, 4, 5-Day Close Pattern Summary")
        all_dates_sorted = sorted([d for d in df_all['date'].unique() if d < selected_day])
        def get_pattern(last_n):
            if len(all_dates_sorted) >= last_n:
                last_n_dates = all_dates_sorted[-last_n:]
                pattern_str = ""
                for d in last_n_dates:
                    d_df = df_all[df_all['date'] == d]
                    if d_df.empty:
                        pattern_str += "-"
                        continue
                    day_open = d_df.iloc[0]['open']
                    day_close = d_df.iloc[-1]['close']
                    if day_close > day_open:
                        pattern_str += "U"
                    else:
                        pattern_str += "D"
                return pattern_str, last_n_dates
            else:
                return None, None
        def pattern_probability(pattern_len, pattern_str):
            # For all days in history, find where the previous N days had this pattern, and compute next day up/down probability
            if not pattern_str or '-' in pattern_str:
                return None, None
            up_count = 0
            down_count = 0
            total = 0
            for i in range(pattern_len, len(all_dates_sorted)):
                dates_window = all_dates_sorted[i-pattern_len:i]
                # Build pattern for this window
                window_pattern = ""
                for d in dates_window:
                    d_df = df_all[df_all['date'] == d]
                    if d_df.empty:
                        window_pattern += "-"
                        continue
                    day_open = d_df.iloc[0]['open']
                    day_close = d_df.iloc[-1]['close']
                    if day_close > day_open:
                        window_pattern += "U"
                    else:
                        window_pattern += "D"
                if window_pattern == pattern_str:
                    # Check next day up/down
                    next_day = all_dates_sorted[i]
                    next_df = df_all[df_all['date'] == next_day]
                    if not next_df.empty:
                        next_open = next_df.iloc[0]['open']
                        next_close = next_df.iloc[-1]['close']
                        if next_close > next_open:
                            up_count += 1
                        else:
                            down_count += 1
                        total += 1
            if total > 0:
                return up_count / total * 100, down_count / total * 100
            else:
                return None, None
        # Top row: 2, 3 day
        col2, col3 = st.columns(2)
        for n, col in zip([2, 3], [col2, col3]):
            with col:
                pattern_str, pattern_dates = get_pattern(n)
                if pattern_str:
                    up_prob, down_prob = pattern_probability(n, pattern_str)
                    st.markdown(f"**Last {n}-Day Pattern:** <span style='font-size:2rem;font-weight:bold'>{pattern_str}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:0.9rem'>({', '.join(str(d) for d in pattern_dates)})</span>", unsafe_allow_html=True)
                    if up_prob is not None:
                        st.markdown(f"<span style='color:green;font-weight:bold'>Next Day Up: {up_prob:.1f}%</span> | <span style='color:#e53935;font-weight:bold'>Down: {down_prob:.1f}%</span>", unsafe_allow_html=True)
                    else:
                        st.info("No historical probability for this pattern.")
                else:
                    st.info(f"Not enough previous days for {n}-day pattern.")
        # Bottom row: 4, 5 day
        col4, col5 = st.columns(2)
        for n, col in zip([4, 5], [col4, col5]):
            with col:
                pattern_str, pattern_dates = get_pattern(n)
                if pattern_str:
                    up_prob, down_prob = pattern_probability(n, pattern_str)
                    st.markdown(f"**Last {n}-Day Pattern:** <span style='font-size:2rem;font-weight:bold'>{pattern_str}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:0.9rem'>({', '.join(str(d) for d in pattern_dates)})</span>", unsafe_allow_html=True)
                    if up_prob is not None:
                        st.markdown(f"<span style='color:green;font-weight:bold'>Next Day Up: {up_prob:.1f}%</span> | <span style='color:#e53935;font-weight:bold'>Down: {down_prob:.1f}%</span>", unsafe_allow_html=True)
                    else:
                        st.info("No historical probability for this pattern.")
                else:
                    st.info(f"Not enough previous days for {n}-day pattern.")

        # --- Gap Fill Probability Section (Optimized) ---
        st.markdown("---")
        st.markdown("#### Gap Fill Probability (Historical, Similar Gap Range)")
        gap_filled_count = 0
        total_similar_gaps = 0
        gap_days = []
        gap_lower = gap_pct - 0.3
        gap_upper = gap_pct + 0.3
        all_dates = sorted(df_all['date'].unique())
        for d in all_dates:
            if d == selected_day:
                continue  # skip current day
            d_df = df_all[df_all['date'] == d]
            if d_df.empty:
                continue
            d_open_915_row = d_df[d_df['datetime'].dt.time == time(9, 15)]
            d_open_915 = d_open_915_row['open'].iloc[0] if not d_open_915_row.empty else None
            prev_close, _ = prev_close_map.get(d, (None, None))
            if d_open_915 is None or prev_close is None:
                continue
            d_gap_pct = ((d_open_915 - prev_close) / prev_close) * 100
            if gap_lower <= d_gap_pct <= gap_upper:
                total_similar_gaps += 1
                filled = ((d_df['high'] >= prev_close) & (d_df['low'] <= prev_close)).any()
                gap_days.append({
                    'date': d,
                    'gap_pct': d_gap_pct,
                    'filled': filled
                })
                if filled:
                    gap_filled_count += 1
        if total_similar_gaps > 0:
            prob = gap_filled_count / total_similar_gaps * 100
            st.success(f"Out of {total_similar_gaps} days with a gap in [{gap_lower:.2f}%, {gap_upper:.2f}%], {gap_filled_count} were filled. Probability of fill: {prob:.1f}%. Not filled: {100-prob:.1f}%.")
            # Donut chart for fill vs not fill
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Filled", "Not Filled"],
                    values=[gap_filled_count, total_similar_gaps-gap_filled_count],
                    hole=0.5,
                    marker_colors=["#388e3c", "#e53935"],
                    textinfo='value+percent',
                )
            ])
            fig.update_layout(
                title_text=f"Gap Fill Analysis (Overall Fill Rate: {prob:.1f}%)",
                annotations=[dict(text="Overall Gap Fill Status", x=0.5, y=0.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
            show_table = st.checkbox("Show historical gap fill table", value=False)
            if show_table:
                gap_days_df = pd.DataFrame(gap_days)
                gap_days_df['date'] = gap_days_df['date'].astype(str)
                gap_days_df['gap_pct'] = gap_days_df['gap_pct'].map(lambda x: f"{x:.2f}%")
                gap_days_df['filled'] = gap_days_df['filled'].map({True: 'Yes', False: 'No'})
                st.dataframe(gap_days_df, use_container_width=True)

    # --- ORB Window Historical Close Stats Table for Selected Day ---
    st.markdown("---")
    st.markdown("### ORB Window Historical Close Stats (Selected Day's Break Direction)")
    orb_windows = [5, 15, 60]
    table_rows = []
    move_rows = []
    for orb_win in orb_windows:
        # For selected day, determine first break direction and time
        sel_df = df_all[df_all['date'] == selected_day].sort_values('datetime')
        orb_window = sel_df[(sel_df['datetime'].dt.time >= time(9, 15)) & (sel_df['datetime'].dt.time < (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time())]
        if orb_window.empty or len(orb_window) < orb_win:
            first_break = 'No Break'
            break_time = 'No Break'
            where_closed = 'No Data'
            today_move = ''
        else:
            orb_high = orb_window['high'].max()
            orb_low = orb_window['low'].min()
            orb_end_time = (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time()
            after_orb = sel_df[sel_df['datetime'].dt.time >= orb_end_time]
            first_break = None
            break_time = 'No Break'
            break_price = None
            for _, row in after_orb.iterrows():
                if row['high'] > orb_high:
                    first_break = 'Up'
                    break_time = row['datetime'].strftime('%H:%M')
                    break_price = orb_high
                    break
                elif row['low'] < orb_low:
                    first_break = 'Down'
                    break_time = row['datetime'].strftime('%H:%M')
                    break_price = orb_low
                    break
            if not first_break:
                first_break = 'No Break'
                break_time = 'No Break'
            # Where closed for selected day
            close_1530_row = sel_df[sel_df['datetime'].dt.time == time(15, 30)]
            if not close_1530_row.empty:
                close_1530 = close_1530_row['close'].iloc[0]
            else:
                close_1530 = sel_df.iloc[-1]['close']
            if close_1530 > orb_high:
                where_closed = 'Above ORB'
            elif close_1530 < orb_low:
                where_closed = 'Below ORB'
            else:
                where_closed = 'In ORB'
            # Today's move after break
            if break_price is not None:
                today_move = (close_1530 - break_price) / break_price * 100
            else:
                today_move = ''
        # For all days, compute close stats for this break direction
        above, below, in_orb, total = 0, 0, 0, 0
        moves = []
        if first_break in ['Up', 'Down']:
            for d in sorted(df_all['date'].unique()):
                d_df = df_all[df_all['date'] == d].sort_values('datetime')
                orb_window = d_df[(d_df['datetime'].dt.time >= time(9, 15)) & (d_df['datetime'].dt.time < (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time())]
                if orb_window.empty or len(orb_window) < orb_win:
                    continue
                orb_high = orb_window['high'].max()
                orb_low = orb_window['low'].min()
                orb_end_time = (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time()
                after_orb = d_df[d_df['datetime'].dt.time >= orb_end_time]
                this_break = None
                break_price_hist = None
                for _, row in after_orb.iterrows():
                    if row['high'] > orb_high:
                        this_break = 'Up'
                        break_price_hist = orb_high
                        break
                    elif row['low'] < orb_low:
                        this_break = 'Down'
                        break_price_hist = orb_low
                        break
                if this_break != first_break:
                    continue
                close_1530_row = d_df[d_df['datetime'].dt.time == time(15, 30)]
                if not close_1530_row.empty:
                    close_1530 = close_1530_row['close'].iloc[0]
                else:
                    close_1530 = d_df.iloc[-1]['close']
                if close_1530 > orb_high:
                    above += 1
                elif close_1530 < orb_low:
                    below += 1
                else:
                    in_orb += 1
                total += 1
                # For move stats, only if close is In ORB
                if where_closed == 'In ORB' and close_1530 <= orb_high and close_1530 >= orb_low and break_price_hist is not None:
                    move = (close_1530 - break_price_hist) / break_price_hist * 100
                    moves.append(move)
        # Calculate percentages
        if total > 0:
            pct_above = above / total * 100
            pct_below = below / total * 100
            pct_in = in_orb / total * 100
        else:
            pct_above = pct_below = pct_in = 0
        table_rows.append({
            'ORB Window': f"{orb_win} min",
            'First Break (Selected Day)': first_break,
            'Time of Break': break_time,
            'Where Closed (Selected Day)': where_closed,
            '% Close Above ORB': f"{pct_above:.1f}%",
            '% Close Below ORB': f"{pct_below:.1f}%",
            '% Close In ORB': f"{pct_in:.1f}%"
        })
        # Move stats table row
        if where_closed == 'In ORB' and first_break in ['Up', 'Down'] and today_move != '':
            if moves:
                mean_move = np.mean(moves)
                min_move = np.min(moves)
                max_move = np.max(moves)
                std_move = np.std(moves)
            else:
                mean_move = min_move = max_move = std_move = 0
            move_rows.append({
                'ORB Window': f"{orb_win} min",
                'First Break (Selected Day)': first_break,
                'Where Closed (Selected Day)': where_closed,
                "Today's Move After Break (%)": f"{today_move:.2f}%",
                'Historical Mean Move (%)': f"{mean_move:.2f}%",
                'Min Move (%)': f"{min_move:.2f}%",
                'Max Move (%)': f"{max_move:.2f}%",
                'Std Dev (%)': f"{std_move:.2f}%"
            })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    # New move stats table
    if move_rows:
        st.markdown('#### Move After Break (Selected Day In ORB)')
        st.dataframe(pd.DataFrame(move_rows), use_container_width=True)

    # New table: Max Move After Break (Selected Day vs Historical)
    move_max_rows = []
    for orb_win in orb_windows:
        sel_df = df_all[df_all['date'] == selected_day].sort_values('datetime')
        orb_window = sel_df[(sel_df['datetime'].dt.time >= time(9, 15)) & (sel_df['datetime'].dt.time < (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time())]
        if orb_window.empty or len(orb_window) < orb_win:
            first_break = 'No Break'
            break_time = 'No Break'
            today_max_move = ''
            today_reach_max = ''
            today_reach_avg = ''
            today_reach_min = ''
        else:
            orb_high = orb_window['high'].max()
            orb_low = orb_window['low'].min()
            orb_end_time = (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time()
            after_orb = sel_df[sel_df['datetime'].dt.time >= orb_end_time]
            first_break = None
            break_time = 'No Break'
            break_idx = None
            for idx, row in after_orb.iterrows():
                if row['high'] > orb_high:
                    first_break = 'Up'
                    break_time = row['datetime'].strftime('%H:%M')
                    break_idx = idx
                    break
                elif row['low'] < orb_low:
                    first_break = 'Down'
                    break_time = row['datetime'].strftime('%H:%M')
                    break_idx = idx
                    break
            if not first_break:
                first_break = 'No Break'
                break_time = 'No Break'
            # Today's max move after break
            if first_break == 'Up' and break_idx is not None:
                after_break = after_orb.loc[break_idx:]
                max_high = after_break['high'].max() if not after_break.empty else np.nan
                today_max_move = ((max_high - orb_high) / orb_high * 100) if max_high and orb_high else np.nan
            elif first_break == 'Down' and break_idx is not None:
                after_break = after_orb.loc[break_idx:]
                min_low = after_break['low'].min() if not after_break.empty else np.nan
                today_max_move = ((min_low - orb_low) / orb_low * 100) if min_low and orb_low else np.nan
            else:
                today_max_move = ''
        # Historical max move stats for same break direction
        hist_moves = []
        if first_break in ['Up', 'Down']:
            for d in sorted(df_all['date'].unique()):
                d_df = df_all[df_all['date'] == d].sort_values('datetime')
                orb_window = d_df[(d_df['datetime'].dt.time >= time(9, 15)) & (d_df['datetime'].dt.time < (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time())]
                if orb_window.empty or len(orb_window) < orb_win:
                    continue
                orb_high = orb_window['high'].max()
                orb_low = orb_window['low'].min()
                orb_end_time = (pd.Timestamp('09:15') + pd.Timedelta(minutes=orb_win)).time()
                after_orb = d_df[d_df['datetime'].dt.time >= orb_end_time]
                this_break = None
                break_idx = None
                for idx, row in after_orb.iterrows():
                    if row['high'] > orb_high:
                        this_break = 'Up'
                        break_idx = idx
                        break
                    elif row['low'] < orb_low:
                        this_break = 'Down'
                        break_idx = idx
                        break
                if this_break != first_break or break_idx is None:
                    continue
                after_break = after_orb.loc[break_idx:]
                if first_break == 'Up':
                    max_high = after_break['high'].max() if not after_break.empty else np.nan
                    move = ((max_high - orb_high) / orb_high * 100) if max_high and orb_high else np.nan
                else:
                    min_low = after_break['low'].min() if not after_break.empty else np.nan
                    move = ((min_low - orb_low) / orb_low * 100) if min_low and orb_low else np.nan
                if not np.isnan(move):
                    hist_moves.append(move)
        if hist_moves:
            hist_max = np.max(hist_moves)
            hist_min = np.min(hist_moves)
            hist_avg = np.mean(hist_moves)
        else:
            hist_max = hist_min = hist_avg = np.nan
        # Did today reach historical max/avg/min?
        if today_max_move != '' and not np.isnan(today_max_move):
            reach_max = 'Yes' if today_max_move >= hist_max else 'No'
            reach_avg = 'Yes' if today_max_move >= hist_avg else 'No'
            reach_min = 'Yes' if today_max_move >= hist_min else 'No'
        else:
            reach_max = reach_avg = reach_min = ''
        move_max_rows.append({
            'ORB Window': f"{orb_win} min",
            'First Break (Selected Day)': first_break,
            'Time of Break': break_time,
            "Today's Max Move (%)": f"{today_max_move:.2f}%" if today_max_move != '' and not np.isnan(today_max_move) else '',
            'Historical Max Move (%)': f"{hist_max:.2f}% ({reach_max})" if not np.isnan(hist_max) else '',
            'Historical Min Move (%)': f"{hist_min:.2f}% ({reach_min})" if not np.isnan(hist_min) else '',
            'Historical Avg Move (%)': f"{hist_avg:.2f}% ({reach_avg})" if not np.isnan(hist_avg) else ''
        })
    st.markdown('#### Max Move After Break (Selected Day vs Historical)')
    st.dataframe(pd.DataFrame(move_max_rows), use_container_width=True)

    # --- End Gap Analysis ---
    if df is not None and not df.empty:
        # Filter by selected time range
        df = df[(df['datetime'].dt.time >= start_time) & (df['datetime'].dt.time <= end_time)]
        st.markdown(f"### 1-Minute Data Table for {selected_day} ({start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')})")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No 1-minute data available for the selected index and day.") 