import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def get_mature_weeks(df, window_days, week_col='week_dt', max_weeks=4, target_col=None):
    today = datetime.today()
    mature_rows = []
    df_sorted = df.sort_values(by=week_col, ascending=False)

    for _, row in df_sorted.iterrows():
        week_start = row[week_col]
        week_dates = [week_start + timedelta(days=i) for i in range(7)]

        if all((today - d).days >= window_days for d in week_dates):
            # Include the row only if target_col is not zero or NaN (if specified)
            if target_col is None or (pd.notna(row[target_col]) and row[target_col] != 0):
                mature_rows.append(row)
            
        if len(mature_rows) == max_weeks:
            break

    if not mature_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(mature_rows)

st.set_page_config(layout="wide")
st.title("⚡ GD ROAS Radar")

#File uploader for CSV
#uploaded_file = st.file_uploader("📂 Upload your Adjust CSV file", type=["csv"])
st.markdown(
'📂 Upload your[ Adjust CSV file](https://suite.adjust.com/datascape/report?app_token__in=%223krzf2i9ugcg%22&utc_offset=%2B00%3A00&reattributed=all&attribution_source=first&attribution_type=all&ad_spend_mode=network&date_period=-127d%3A-5d&cohort_maturity=mature&sandbox=false&channel_id__in=%22partner_-300%22%2C%22partner_254%22%2C%22partner_257%22%2C%22partner_7%22%2C%22partner_2725%22%2C%22partner_34%22%2C%22partner_2127%22%2C%22partner_2360%22%2C%22partner_182%22%2C%22partner_100%22%2C%22partner_369%7Cnetwork_Mintegral%22%2C%22partner_56%22%2C%22partner_490%7Cnetwork_NativeX%22%2C%22partner_1770%22%2C%22partner_2682%22%2C%22partner_217%22%2C%22partner_1718%22%2C%22partner_560%22%2C%22network_SpinnerBattle+in+Ragdoll%22%2C%22network_Unknown+Devices%22%2C%22network_Untrusted+Devices%22%2C%22network_Twisted_in_Ragdoll%22%2C%22network_Unknown+Devices+%28restored+1ekeu2wz%29%22%2C%22network_Ragdoll_CP%22%2C%22network_Disabled+Third+Party+Sharing+Before+Install%22%2C%22network_Ragdoll_X_Twisted%22%2C%22network_Unknown+Devices+%28restored+1esqjar0%29%22%2C%22network_Meta%22%2C%22network_Measurement+Consent+Updated+Before+Install%22%2C%22network_Unknown+Devices+%28restored+1770rwkr%29%22%2C%22network_Unknown%22&applovin_mode=probabilistic&ironsource_mode=ironsource&digital_turbine_mode=digital_turbine&dimensions=app%2Cchannel%2Ccampaign_network%2Cweek&metrics=installs%2Cinstalls_per_mile%2Ccost%2Call_revenue%2Cgross_profit%2Ccustom_roas%2Cattribution_clicks%2Cattribution_impressions%2Cnetwork_ctr%2Cecpi%2Cretention_rate_d1%2Cretention_rate_d3%2Cretention_rate_d7%2Cretention_rate_d14%2Cretention_rate_d30%2Croas_d0%2Croas_d3%2Croas_d7%2Croas_d14%2Croas_d28%2Croas_d30%2Croas_d45%2Croas_d60%2Croas_d75%2Croas_d90%2Croas_d120%2Croas_ad_d0%2Croas_ad_d7%2Croas_ad_d14%2Croas_ad_d28%2Croas_ad_d30%2Croas_ad_d45%2Croas_ad_d60%2Croas_ad_d75%2Croas_ad_d90%2Croas_ad_d120%2Croas_iap_d0%2Croas_iap_d7%2Croas_iap_d14%2Croas_iap_d28%2Croas_iap_d30%2Croas_iap_d45%2Croas_iap_d60%2Croas_iap_d75%2Croas_iap_d90%2Croas_iap_d120&sort=app&table_view=pivot&parent_report_id=290817)',
unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    try:
        # Check if the file is a CSV based on extension
        if not uploaded_file.name.endswith('.csv'):
            raise ValueError("Invalid file type. Please upload a CSV file.")

        # Try reading the CSV
        df = pd.read_csv(uploaded_file)
        df = df.rename(columns=lambda x: x.strip())

        # Check for required columns
        required_cols = [
            'app', 'channel', 'campaign_network', 'week', 'installs', 'installs_per_mile', 
            'cost', 'all_revenue', 'gross_profit', 'custom_roas', 'attribution_clicks', 
            'attribution_impressions', 'network_ctr', 'ecpi', 'retention_rate_d1', 
            'retention_rate_d3', 'retention_rate_d7', 'retention_rate_d14', 'retention_rate_d30', 
            'roas_d0', 'roas_d3', 'roas_d7', 'roas_d14', 'roas_d28', 'roas_d30', 'roas_d45', 
            'roas_d60', 'roas_d75', 'roas_d90', 'roas_d120', 'roas_ad_d0', 'roas_ad_d7', 
            'roas_ad_d14', 'roas_ad_d28', 'roas_ad_d30', 'roas_ad_d45', 'roas_ad_d60', 
            'roas_ad_d75', 'roas_ad_d90', 'roas_ad_d120', 'roas_iap_d0', 'roas_iap_d7', 
            'roas_iap_d14', 'roas_iap_d28', 'roas_iap_d30', 'roas_iap_d45', 'roas_iap_d60', 
            'roas_iap_d75', 'roas_iap_d90', 'roas_iap_d120'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        # If columns are missing, show a warning but continue
        if missing_cols:
            st.error(f"⚠️ CSV is missing required columns: {', '.join(missing_cols)}.")
            st.stop()

    except ValueError as ve:
        st.error(f"❌ {str(ve)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load CSV file. Error: {str(e)}")
        st.stop()

    # Channel and campaign selection
    selected_channel = st.selectbox("🎯 Select Channel", df["channel"].dropna().unique())
    df_filtered = df[df["channel"] == selected_channel]
    selected_network = st.selectbox("📡 Select Campaign", df_filtered["campaign_network"].dropna().unique())
    df_filtered = df_filtered[df_filtered["campaign_network"] == selected_network]
    
    # Convert 'week' to string and datetime
    if 'week' in df_filtered.columns:
        df_filtered["week_str"] = df_filtered["week"].astype(str)  # Ensure string for charts
        df_filtered["week_dt"] = pd.to_datetime(df_filtered["week"], errors='coerce')
        df_filtered = df_filtered.dropna(subset=["week_dt"])
    
    # Check if df_filtered is empty after datetime conversion
    if df_filtered.empty:
        st.error("No valid data available after datetime conversion. Please check the 'week' column format.")
    else:
        # Debug: Display week range
        min_week = df_filtered["week_dt"].min()
        max_week = df_filtered["week_dt"].max()
        st.write("📅 Week Range in data:", 
                min_week.strftime('%Y-%m-%d') if pd.notnull(min_week) else "NaT",
                "→",
                max_week.strftime('%Y-%m-%d') if pd.notnull(max_week) else "NaT")

        # Define date ranges
        date_ranges = {
            "D7 Range": (max_week - pd.Timedelta(days=7), max_week - pd.Timedelta(days=35)),
            "D14 Range": (max_week - pd.Timedelta(days=14), max_week - pd.Timedelta(days=42)),
            "D28 Range": (max_week - pd.Timedelta(days=28), max_week - pd.Timedelta(days=56)),
            "D45 Range": (max_week - pd.Timedelta(days=45), max_week - pd.Timedelta(days=73)),
            "D60 Range": (max_week - pd.Timedelta(days=60), max_week - pd.Timedelta(days=88)),
            "D75 Range": (max_week - pd.Timedelta(days=75), max_week - pd.Timedelta(days=103)),
            "D90 Range": (max_week - pd.Timedelta(days=90), max_week - pd.Timedelta(days=118)),
        }

        # Create date range table
        date_table = pd.DataFrame(date_ranges, index=["Start", "End"]).map(
            lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "NaT"
        )

        # 🚀 Step 1: Tabs to reduce scroll clutter
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Performance Table",
            "📈 Growth Rates",
            "📦 Channels Campaigns Comparison",
            "🎯 Break-Even Analysis",
            "📊 Visualizations",
            "💡 User Manual"
        ])

        with tab1:
            st.markdown("### 📊 Current Campaign Performance")
            st.write(f"Network: {selected_channel}")
            st.write(f"Campaign: {selected_network}")

            if not df_filtered.empty:
                display_columns = [col for col in df_filtered.columns if col not in [
                    'week_str', 'week_dt', 'channel', 'app', 'campaign_network',
                    'attribution_clicks', 'attribution_impressions', 'ecpi', 'retention_rate_d3',
                    'roas_ad_d0', 'roas_ad_d7', 'roas_ad_d14', 'roas_ad_d28', 'roas_ad_d45',
                    'roas_ad_d60', 'roas_ad_d75', 'roas_ad_d90', 'roas_ad_d120',
                    'roas_iap_d0', 'roas_iap_d7', 'roas_iap_d14', 'roas_iap_d28', 'roas_iap_d30',
                    'roas_ad_d30', 'roas_iap_d45', 'roas_iap_d60', 'roas_iap_d75',
                    'roas_iap_d90', 'roas_iap_d120'
                ]]
                st.dataframe(df_filtered[display_columns], use_container_width=True)

                # 📈 Weekly Trend Visualizer (Dual Y-Axis)
                st.markdown("### 📈 Weekly Trend Visualizer")

                numeric_columns = df_filtered.select_dtypes(include='number').columns.tolist()
                default_selection = [col for col in ['all_revenue', 'custom_roas', 'cost'] if col in numeric_columns]


                selected_y_cols = st.multiselect(
                    "Select up to 5 columns to plot (Y-axis)", 
                    options=numeric_columns,
                    default=default_selection,
                    max_selections=5
                )

                right_axis_keywords = ['roas', 'retention', 'rate', 'ctr']
                use_right_axis = lambda col: any(key in col.lower() for key in right_axis_keywords)

                if selected_y_cols:
                    fig = go.Figure()

                    for col in selected_y_cols:
                        yaxis = 'y2' if use_right_axis(col) else 'y1'
                        fig.add_trace(go.Scatter(
                            x=df_filtered['week_dt'],
                            y=df_filtered[col],
                            mode='lines+markers',
                            name=col,
                            yaxis=yaxis,
                            hovertemplate=f'%{{y:.2f}}<extra>{col}</extra>'
                        ))

                    fig.update_layout(
                        title="📈 Weekly Trend for Selected Metrics",
                        xaxis=dict(title="Week", tickformat="%Y-%m-%d"),
                        yaxis=dict(
                            title="Primary Axis",  
                            side="left"
                        ),
                        yaxis2=dict(
                            title="Secondary Axis", 
                            overlaying="y", 
                            side="right", 
                            showgrid=False
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("⚠️ No data available for the selected campaign.")



        # ROAS Ratios Calculation
        day_intervals = [(0, 7), (7, 14), (14, 28), (28, 45), (45, 60), (60, 75), (75, 90)]
        prefix_map = {
            'roas': 'Overall ROAS',
            'roas_ad': 'Ad ROAS',
            'roas_iap': 'IAP ROAS'
        }

        def compute_roas_ratios(df, prefix, label):
            for start, end in day_intervals:
                col_name = f'{label} D{start} to D{end}'
                start_col = f'{prefix}_d{start}'
                end_col = f'{prefix}_d{end}'
                if start_col in df.columns and end_col in df.columns:
                    df[col_name] = df[end_col] / df[start_col]
                    df[col_name] = df[col_name].fillna(0)
            return df

        updated_df = df_filtered.copy()
        df_filtered["installs"] = df_filtered["installs"].replace(0, pd.NA)
        df_filtered["cpi"] = df_filtered["cost"] / df_filtered["installs"]

        for prefix, label in prefix_map.items():
            updated_df = compute_roas_ratios(updated_df, prefix, label)

        with tab2:
            st.subheader("📈 Calculated ROAS Growth - From Uploaded CSV")
            for prefix, label in prefix_map.items():
                with st.expander(f"{label}"):
                    if not updated_df.empty:
                        output_cols = ['app', 'channel', 'campaign_network', 'week'] + [col for col in updated_df.columns if col.startswith(label)]
                        st.dataframe(updated_df[output_cols], use_container_width=True)
                    else:
                        st.warning("No data available for the selected campaign.")

        with tab4:
            st.subheader("📈 Break Even Points Analysis")

            roas_types = {
                "Overall ROAS": ["Overall ROAS D0 to D7", "Overall ROAS D7 to D14", "Overall ROAS D14 to D28", 
                                "Overall ROAS D28 to D45", "Overall ROAS D45 to D60", "Overall ROAS D60 to D75", 
                                "Overall ROAS D75 to D90"],
                "Ad ROAS": ["Ad ROAS D0 to D7", "Ad ROAS D7 to D14", "Ad ROAS D14 to D28", "Ad ROAS D28 to D45", 
                            "Ad ROAS D45 to D60", "Ad ROAS D60 to D75", "Ad ROAS D75 to D90"],
                "IAP ROAS": ["IAP ROAS D0 to D7", "IAP ROAS D7 to D14", "IAP ROAS D14 to D28", "IAP ROAS D28 to D45", 
                            "IAP ROAS D45 to D60", "IAP ROAS D60 to D75", "IAP ROAS D75 to D90"],
            }
            optimization_window = st.selectbox(
                "📌 Select Optimization Window",
                options=["D0", "D7", "D28"],
                index=0
            )

            for roas_type, growth_columns in roas_types.items():
                with st.expander(f"📦 {roas_type} Growth Analysis", expanded=True):
                    st.subheader(f"📈 {roas_type} Average Growth Rates")
                    growth_rates = {}

                    for col in growth_columns:
                        try:
                            # Extract window_days from column name, e.g., "D28 to D45" → 45
                            window_days = int(col.split("D")[-1])
                            mature_weeks = get_mature_weeks(updated_df, window_days=window_days, week_col="week_dt", max_weeks=4, target_col=col)
                            
                            if not mature_weeks.empty and col in mature_weeks.columns:
                                growth_rates[col] = mature_weeks[col].mean()
                            else:
                                growth_rates[col] = 0
                        except Exception as e:
                            growth_rates[col] = 0

                    growth_df = pd.DataFrame([growth_rates], index=["Avg Growth Rate"])
                    st.table(growth_df)

                    # Optimization window logic
                    if optimization_window == "D0":
                        d0_goal = st.number_input(f"🎯 Enter {roas_type} D0 Goal (%)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)
                        roas_performance = {}
                        prev_value = d0_goal
                        for col in growth_columns:
                            roas_performance[col] = prev_value * growth_rates[col]
                            prev_value = roas_performance[col]
                        roas_df = pd.DataFrame([roas_performance], index=["D0 Goal-Based ROAS"])
                        st.subheader(f"📊 {roas_type} Performance")
                        st.table(roas_df)

                    elif optimization_window == "D7":
                        d7_goal = st.number_input(f"🎯 Enter {roas_type} D7 Goal (%)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)
                        roas_d7_performance = {}
                        prev_value = d7_goal
                        for col in growth_columns[1:]:  # Start from D7 to D14 onward
                            roas_d7_performance[col] = prev_value * growth_rates[col]
                            prev_value = roas_d7_performance[col]
                        roas_d7_df = pd.DataFrame([roas_d7_performance], index=["D7 Goal-Based ROAS"])
                        st.subheader(f"📊 {roas_type} Performance")
                        st.table(roas_d7_df)

                    elif optimization_window == "D28":
                        d28_goal = st.number_input(f"🎯 Enter {roas_type} D28 Goal (%)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)
                        roas_d28_performance = {}
                        prev_value = d28_goal
                        for col in growth_columns[3:]:  # Start from D28 to D45 onward
                            roas_d28_performance[col] = prev_value * growth_rates[col]
                            prev_value = roas_d28_performance[col]
                        roas_d28_df = pd.DataFrame([roas_d28_performance], index=["D28 Goal-Based ROAS"])
                        st.subheader(f"📊 {roas_type} Performance")
                        st.table(roas_d28_df)


        with tab5:
            st.subheader("📊 Campaign Performance Visualizations")
            with st.expander("Visualizations", expanded=True):
                # Global Spend Pie Chart
                if 'channel' in df.columns and 'cost' in df.columns:
                    spend_by_channel = df.groupby('channel')['cost'].sum().reset_index()
                    spend_by_channel = spend_by_channel[spend_by_channel['cost'] > 0]
                    spend_by_channel = spend_by_channel.sort_values(by="cost", ascending=False)

                    pull_values = [0.1 if ch == selected_channel else 0 for ch in spend_by_channel['channel']]

                    fig_pie = px.pie(
                        spend_by_channel,
                        names='channel',
                        values='cost',
                        title='📊 Channel-Wise Spend Distribution - Global',
                        hole=0.4
                    )
                    fig_pie.update_traces(textinfo='percent+label', pull=pull_values)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("⚠️ 'channel' or 'cost' column not found in the data.")

                # Spend vs Revenue vs Gross Profit
                if all(col in df_filtered.columns for col in ['cost', 'all_revenue', 'gross_profit']):
                    fig_mix = go.Figure()

                    fig_mix.add_trace(go.Bar(
                        x=df_filtered['week_dt'],
                        y=df_filtered['all_revenue'],
                        name='Revenue',
                        marker=dict(color='#1f77b4'),
                        hovertemplate='%{y:.2f}<extra>Revenue</extra>'
                    ))

                    fig_mix.add_trace(go.Bar(
                        x=df_filtered['week_dt'],
                        y=df_filtered['cost'],
                        name='Spend',
                        marker=dict(color='#aec7e8'),
                        hovertemplate='%{y:.2f}<extra>Spend</extra>'
                    ))

                    fig_mix.add_trace(go.Scatter(
                        x=df_filtered['week_dt'],
                        y=df_filtered['gross_profit'],
                        name='Gross Profit',
                        yaxis='y2',
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=6),
                        hovertemplate='%{y:.2f}<extra>Gross Profit</extra>'
                    ))

                    fig_mix.update_layout(
                        title="Spend vs Revenue (Bar) and Gross Profit (Line)",
                        xaxis=dict(
                            title="WoW",
                            tickmode="linear",
                            dtick=604800000,
                            tickformat="%b %d",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="Spend & Revenue",
                            side='left'
                        ),
                        yaxis2=dict(
                            title="Gross Profit",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        barmode='group',
                        bargap=0.4,
                        bargroupgap=0.3,
                        legend_title_text=None
                    )

                    st.plotly_chart(fig_mix, use_container_width=True)
                else:
                    st.warning("⚠️ Missing one or more required columns: 'cost', 'all_revenue', or 'gross_profit'.")

                # ROAS Over Time
                roas_columns = ['roas_d0', 'roas_d7', 'roas_d14', 'roas_d28', 'roas_d45', 'roas_d60']
                available_roas = [col for col in roas_columns if col in df_filtered.columns]

                if available_roas:
                    df_filtered[available_roas] = df_filtered[available_roas].replace(0, pd.NA)
                    for col in available_roas:
                        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                    legend_map = {
                        'roas_d0': 'ROAS D0',
                        'roas_d7': 'ROAS D7',
                        'roas_d14': 'ROAS D14',
                        'roas_d28': 'ROAS D28',
                        'roas_d45': 'ROAS D45',
                        'roas_d60': 'ROAS D60'
                    }

                    roas_df_renamed = df_filtered[['week_dt'] + available_roas].rename(columns=legend_map)

                    fig_roas = px.line(
                        roas_df_renamed,
                        x='week_dt',
                        y=list(legend_map.values()),
                        title="ROAS - D0, D7, D14, D28, D45, D60",
                        labels={"week_dt": "WoW"}
                    )
                    fig_roas.update_traces(
                        mode="lines+markers",
                        connectgaps=False,
                        marker=dict(size=6)
                    )
                    fig_roas.update_layout(
                        xaxis=dict(
                            tickmode="linear",
                            dtick=604800000,
                            tickformat="%b %d",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="ROAS Value",
                            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
                            range=[0.2, 1.5],
                            showgrid=True
                        ),
                        legend_title_text=None
                    )

                    st.plotly_chart(fig_roas, use_container_width=True)
                else:
                    st.warning("ROAS columns not found!")

                # CPI Over Time
                try:
                    if 'cpi' in df_filtered.columns:
                        df_filtered['cpi'] = pd.to_numeric(df_filtered['cpi'], errors='coerce')
                        df_filtered = df_filtered.dropna(subset=['cpi', 'week_dt'])

                        fig_cpi = px.line(
                            df_filtered,
                            x='week_dt',
                            y='cpi',
                            title="CPI",
                            labels={"week_dt": "WoW"}
                        )

                        fig_cpi.update_traces(
                            mode="lines+markers",
                            marker=dict(size=6)
                        )

                        fig_cpi.update_layout(
                            xaxis=dict(
                                tickmode="linear",
                                dtick=604800000,
                                tickformat="%b %d",
                                tickangle=-45
                            )
                        )

                        st.plotly_chart(fig_cpi, use_container_width=True)
                    else:
                        st.warning("⚠️ 'cpi' column not found.")
                except Exception as e:
                    st.error(f"❌ Could not generate CPI chart: {str(e)}")

                # Weekly Trend of ROAS Growth Intervals
                growth_cols = [col for col in updated_df.columns if col.startswith("Overall ROAS D")]

                if growth_cols:
                    updated_df[growth_cols] = updated_df[growth_cols].replace(0, pd.NA)
                    for col in growth_cols:
                        updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')

                    growth_trend_df = updated_df.groupby("week_dt")[growth_cols].mean().reset_index()

                    pretty_labels = {
                        "Overall ROAS D0 to D7": "D0 → D7",
                        "Overall ROAS D7 to D14": "D7 → D14",
                        "Overall ROAS D14 to D28": "D14 → D28",
                        "Overall ROAS D28 to D45": "D28 → D45",
                        "Overall ROAS D45 to D60": "D45 → D60",
                        "Overall ROAS D60 to D75": "D60 → D75",
                        "Overall ROAS D75 to D90": "D75 → D90"
                    }
                    growth_trend_df = growth_trend_df.rename(columns=pretty_labels)

                    fig_growth = px.line(
                        growth_trend_df,
                        x="week_dt",
                        y=list(pretty_labels.values()),
                        title="📊 ROAS Growth Rates - WoW",
                        labels={"value": "Average Growth Rate", "week_dt": "WoW"}
                    )

                    fig_growth.update_traces(
                        mode="lines+markers",
                        connectgaps=False,
                        marker=dict(size=6)
                    )

                    fig_growth.update_layout(
                        xaxis=dict(
                            tickmode="linear",
                            dtick=604800000,
                            tickformat="%b %d",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="Average Growth Rate",
                            tickvals=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
                            range=[0.8, 2.2]
                        ),
                        legend_title_text=None
                    )

                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.warning("⚠️ No Overall ROAS growth columns found to plot weekly trend.")
                
                # Average ROAS Growth Rates Line Graph
                growth_cols = [col for col in updated_df.columns if col.startswith("Overall ROAS D")]
                if growth_cols:
                    updated_df[growth_cols] = updated_df[growth_cols].replace(0, pd.NA)
                    for col in growth_cols:
                        updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
                    
                    # Calculate mean for each growth column
                    avg_growth = updated_df[growth_cols].mean().reset_index()
                    avg_growth.columns = ['Interval', 'Average Growth Rate']
                    
                    # Pretty labels for intervals
                    pretty_labels = {
                        "Overall ROAS D0 to D7": "D0 → D7",
                        "Overall ROAS D7 to D14": "D7 → D14",
                        "Overall ROAS D14 to D28": "D14 → D28",
                        "Overall ROAS D28 to D45": "D28 → D45",
                        "Overall ROAS D45 to D60": "D45 → D60",
                        "Overall ROAS D60 to D75": "D60 → D75",
                        "Overall ROAS D75 to D90": "D75 → D90"
                    }
                    avg_growth['Interval'] = avg_growth['Interval'].map(pretty_labels)
                    
                    fig_avg_growth = px.line(
                        avg_growth,
                        x='Interval',
                        y='Average Growth Rate',
                        title="📊 Average ROAS Growth Rates Across Intervals",
                        labels={"Interval": "ROAS Interval", "Average Growth Rate": "Average Growth Rate"},
                        markers=True
                    )
                    fig_avg_growth.update_traces(
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(color='red')
                    )
                    fig_avg_growth.update_layout(
                        xaxis=dict(
                            title="ROAS Interval",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="Average Growth Rate",
                            tickvals=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                            range=[0.8, 1.8]
                        ),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_avg_growth, use_container_width=True)
                else:
                    st.warning("⚠️ No Overall ROAS growth columns found to plot average growth rates.")

        with tab3:
            if selected_channel.lower() == "applovin" and uploaded_file:
                st.subheader("📈 ROAS Average Growth Rates for All AppLovin Campaigns")
                st.markdown("⚠️ This table reflects data from the most recent 4 fully matured weeks only.")

                df_applovin = df[df["channel"].str.lower() == "applovin"].copy()
                
                if 'week' in df_applovin.columns:
                    df_applovin["week_dt"] = pd.to_datetime(df_applovin["week"], errors='coerce')
                    df_applovin = df_applovin.dropna(subset=["week_dt"])

                df_applovin["campaign_network"] = df_applovin["campaign_network"].astype(str).str.strip()

                df_applovin = df_applovin[
                    df_applovin["campaign_network"].notna() &
                    (df_applovin["campaign_network"] != "") &
                    ~df_applovin["campaign_network"].str.lower().str.contains(
                        "unknown|bracket|^\(\)$|\{.*\}", regex=True
                    )
                ]
                
                df_applovin_ratios = df_applovin.copy()
                for prefix, label in prefix_map.items():
                    df_applovin_ratios = compute_roas_ratios(df_applovin_ratios, prefix, label)
                
                all_campaigns = df_applovin_ratios["campaign_network"].unique()
                growth_rates_by_campaign = {}
                
                roas_types = {
                    "Overall ROAS": ["Overall ROAS D0 to D7", "Overall ROAS D7 to D14", "Overall ROAS D14 to D28", 
                                    "Overall ROAS D28 to D45", "Overall ROAS D45 to D60", "Overall ROAS D60 to D75", 
                                    "Overall ROAS D75 to D90"],
                }
                
                for campaign in all_campaigns:
                    full_campaign_data = df_applovin_ratios[df_applovin_ratios["campaign_network"] == campaign]
                    growth_rates = {}
                    
                    for roas_type, growth_columns in roas_types.items():
                        for col in growth_columns:
                            try:
                                # Extract window_days from column name, e.g., "D28 to D45" → 45
                                window_days = int(col.split("D")[-1])
                                mature_data = get_mature_weeks(full_campaign_data, window_days=window_days, week_col="week_dt", max_weeks=4, target_col=col)
                                
                                if not mature_data.empty and col in mature_data.columns:
                                    avg_growth = mature_data[col].mean()
                                    growth_rates[f"{roas_type}: {col}"] = avg_growth if pd.notna(avg_growth) else 0
                                else:
                                    growth_rates[f"{roas_type}: {col}"] = 0
                            except Exception as e:
                                growth_rates[f"{roas_type}: {col}"] = 0
                    
                    growth_rates_by_campaign[campaign] = growth_rates 
                
                growth_df = pd.DataFrame.from_dict(growth_rates_by_campaign, orient="index")
                
                ordered_columns = []
                for roas_type in roas_types.keys():
                    ordered_columns += [col for col in growth_df.columns if col.startswith(roas_type)]
                growth_df = growth_df[ordered_columns]
                
                growth_df = growth_df.round(4)
                
                if not growth_df.empty:
                    st.dataframe(
                        growth_df,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            col: st.column_config.NumberColumn(
                                format="%.4f"
                            ) for col in growth_df.columns
                        }
                    )
                else:
                    st.warning("⚠️ No valid data available for AppLovin campaigns.")

                # Prepare data for the line graph
                growth_cols = ["Overall ROAS D0 to D7", "Overall ROAS D7 to D14", "Overall ROAS D14 to D28", 
                            "Overall ROAS D28 to D45", "Overall ROAS D45 to D60", "Overall ROAS D60 to D75", 
                            "Overall ROAS D75 to D90"]
                pretty_labels = {
                    "Overall ROAS D0 to D7": "D0 → D7",
                    "Overall ROAS D7 to D14": "D7 → D14",
                    "Overall ROAS D14 to D28": "D14 → D28",
                    "Overall ROAS D28 to D45": "D28 → D45",
                    "Overall ROAS D45 to D60": "D45 → D60",
                    "Overall ROAS D60 to D75": "D60 → D75",
                    "Overall ROAS D75 to D90": "D75 → D90"
                }
                
                plot_data = []
                
                for campaign in all_campaigns:
                    full_campaign_data = df_applovin_ratios[df_applovin_ratios["campaign_network"] == campaign]
                    campaign_growth = {}
                    for col in growth_cols:
                        try:
                            window_days = int(col.split("D")[-1])
                            mature_data = get_mature_weeks(full_campaign_data, window_days=window_days, week_col="week_dt", max_weeks=4, target_col=col)
                            if not mature_data.empty and col in mature_data.columns:
                                avg_growth = mature_data[col].mean()
                                campaign_growth[col] = avg_growth if pd.notna(avg_growth) else 0
                            else:
                                campaign_growth[col] = 0
                        except Exception:
                            campaign_growth[col] = 0
                    # Append data for this campaign
                    for col in growth_cols:
                        plot_data.append({
                            "Campaign": campaign,
                            "Interval": pretty_labels[col],
                            "Average Growth Rate": campaign_growth[col]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                if not plot_df.empty:
                    fig_applovin_growth = px.line(
                        plot_df,
                        x="Interval",
                        y="Average Growth Rate",
                        color="Campaign",
                        title="📊 AppLovin Campaigns ROAS Growth Rates",
                        labels={"Interval": "ROAS Interval", "Average Growth Rate": "Average Growth Rate"},
                        markers=True
                    )
                    fig_applovin_growth.update_traces(
                        mode="lines+markers",
                        marker=dict(size=8)
                    )
                    fig_applovin_growth.update_layout(
                        xaxis=dict(
                            title="ROAS Interval",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="Average Growth Rate",
                            tickvals=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                            range=[0.8, 2.0]
                        ),
                        legend_title_text="Campaign",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_applovin_growth, use_container_width=True)
                else:
                    st.warning("⚠️ No valid data available for AppLovin campaigns line graph.")
            if selected_channel.lower() == "mintegral" and uploaded_file:
                st.subheader("📈 ROAS Average Growth Rates for All Mintegral Campaigns")
                st.markdown("⚠️ This table reflects data from the most recent 4 fully matured weeks only.")

                df_mintegral = df[df["channel"].str.lower() == "mintegral"].copy()
                
                if 'week' in df_mintegral.columns:
                    df_mintegral["week_dt"] = pd.to_datetime(df_mintegral["week"], errors='coerce')
                    df_mintegral = df_mintegral.dropna(subset=["week_dt"])

                df_mintegral["campaign_network"] = df_mintegral["campaign_network"].astype(str).str.strip()

                df_mintegral = df_mintegral[
                    df_mintegral["campaign_network"].notna() &
                    (df_mintegral["campaign_network"] != "") &
                    ~df_mintegral["campaign_network"].str.lower().str.contains(
                        "unknown|bracket|^\(\)$|\{.*\}", regex=True
                    )
                ]
                
                df_mintegral_ratios = df_mintegral.copy()
                for prefix, label in prefix_map.items():
                    df_mintegral_ratios = compute_roas_ratios(df_mintegral_ratios, prefix, label)
                
                all_campaigns = df_mintegral_ratios["campaign_network"].unique()
                growth_rates_by_campaign = {}
                
                roas_types = {
                    "Overall ROAS": ["Overall ROAS D0 to D7", "Overall ROAS D7 to D14", "Overall ROAS D14 to D28", 
                                    "Overall ROAS D28 to D45", "Overall ROAS D45 to D60", "Overall ROAS D60 to D75", 
                                    "Overall ROAS D75 to D90"],
                }
                
                for campaign in all_campaigns:
                    full_campaign_data = df_mintegral_ratios[df_mintegral_ratios["campaign_network"] == campaign]
                    growth_rates = {}
                    
                    for roas_type, growth_columns in roas_types.items():
                        for col in growth_columns:
                            try:
                                # Extract window_days from column name, e.g., "D28 to D45" → 45
                                window_days = int(col.split("D")[-1])
                                mature_data = get_mature_weeks(full_campaign_data, window_days=window_days, week_col="week_dt", max_weeks=4, target_col=col)
                                
                                if not mature_data.empty and col in mature_data.columns:
                                    avg_growth = mature_data[col].mean()
                                    growth_rates[f"{roas_type}: {col}"] = avg_growth if pd.notna(avg_growth) else 0
                                else:
                                    growth_rates[f"{roas_type}: {col}"] = 0
                            except Exception as e:
                                growth_rates[f"{roas_type}: {col}"] = 0
                    
                    growth_rates_by_campaign[campaign] = growth_rates
                
                growth_df = pd.DataFrame.from_dict(growth_rates_by_campaign, orient="index")
                
                ordered_columns = []
                for roas_type in roas_types.keys():
                    ordered_columns += [col for col in growth_df.columns if col.startswith(roas_type)]
                growth_df = growth_df[ordered_columns]
                
                growth_df = growth_df.round(4)
                
                if not growth_df.empty:
                    st.dataframe(
                        growth_df,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            col: st.column_config.NumberColumn(
                                format="%.4f"
                            ) for col in growth_df.columns
                        }
                    )
                else:
                    st.warning("⚠️ No valid data available for Mintegral campaigns.")

                # Prepare data for the line graph
                growth_cols = ["Overall ROAS D0 to D7", "Overall ROAS D7 to D14", "Overall ROAS D14 to D28", 
                            "Overall ROAS D28 to D45", "Overall ROAS D45 to D60", "Overall ROAS D60 to D75", 
                            "Overall ROAS D75 to D90"]
                pretty_labels = {
                    "Overall ROAS D0 to D7": "D0 → D7",
                    "Overall ROAS D7 to D14": "D7 → D14",
                    "Overall ROAS D14 to D28": "D14 → D28",
                    "Overall ROAS D28 to D45": "D28 → D45",
                    "Overall ROAS D45 to D60": "D45 → D60",
                    "Overall ROAS D60 to D75": "D60 → D75",
                    "Overall ROAS D75 to D90": "D75 → D90"
                }
                
                plot_data = []
                
                for campaign in all_campaigns:
                    full_campaign_data = df_mintegral_ratios[df_mintegral_ratios["campaign_network"] == campaign]
                    campaign_growth = {}
                    for col in growth_cols:
                        try:
                            window_days = int(col.split("D")[-1])
                            mature_data = get_mature_weeks(full_campaign_data, window_days=window_days, week_col="week_dt", max_weeks=4, target_col=col)
                            if not mature_data.empty and col in mature_data.columns:
                                avg_growth = mature_data[col].mean()
                                campaign_growth[col] = avg_growth if pd.notna(avg_growth) else 0
                            else:
                                campaign_growth[col] = 0
                        except Exception:
                            campaign_growth[col] = 0
                    # Append data for this campaign
                    for col in growth_cols:
                        plot_data.append({
                            "Campaign": campaign,
                            "Interval": pretty_labels[col],
                            "Average Growth Rate": campaign_growth[col]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                if not plot_df.empty:
                    fig_mintegral_growth = px.line(
                        plot_df,
                        x="Interval",
                        y="Average Growth Rate",
                        color="Campaign",
                        title="📊 Mintegral Campaigns ROAS Growth Rates",
                        labels={"Interval": "ROAS Interval", "Average Growth Rate": "Average Growth Rate"},
                        markers=True
                    )
                    fig_mintegral_growth.update_traces(
                        mode="lines+markers",
                        marker=dict(size=8)
                    )
                    fig_mintegral_growth.update_layout(
                        xaxis=dict(
                            title="ROAS Interval",
                            tickangle=-45
                        ),
                        yaxis=dict(
                            title="Average Growth Rate",
                            tickvals=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                            range=[0.8, 2.0]
                        ),
                        legend_title_text="Campaign",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_mintegral_growth, use_container_width=True)
                else:
                    st.warning("⚠️ No valid data available for Mintegral campaigns line graph.") 



        with tab6:
            st.subheader("💡 User Manual – How to Use GD ROAS Radar")

            st.markdown(
            "Welcome to **GD ROAS Radar**!⚡ This dashboard helps you evaluate your campaigns' ROAS performance over different maturity windows and identify break-even points with confidence.  \n"
            "Here’s how to use it:"
            )

            with st.expander("📂 1. Upload Your CSV File"):
                st.markdown("""
                - "📂 Upload your [Adjust CSV file](https://suite.adjust.com/datascape/report?app_token__in=n1dbmx5fs8ow&utc_offset=%2B00%3A00&reattributed=all&attribution_source=first&attribution_type=all&ad_spend_mode=network&date_period=-127d%3A-5d&cohort_maturity=mature&sandbox=false&channel_id__in=partner_-300%2Cpartner_254%2Cpartner_257%2Cpartner_7%2Cpartner_2725%2Cpartner_34%2Cpartner_2127%2Cpartner_2360%2Cpartner_182%2Cpartner_100%2Cpartner_369%7Cnetwork_Mintegral%2Cpartner_56%2Cpartner_490%7Cnetwork_NativeX%2Cpartner_1770%2Cpartner_2682%2Cpartner_217%2Cpartner_1718%2Cpartner_560%2Cnetwork_SpinnerBattle+in+Ragdoll%2Cnetwork_Unknown+Devices%2Cnetwork_Untrusted+Devices%2Cnetwork_Twisted_in_Ragdoll%2Cnetwork_Unknown+Devices+%28restored+1ekeu2wz%29%2Cnetwork_Ragdoll_CP%2Cnetwork_Disabled+Third+Party+Sharing+Before+Install%2Cnetwork_Ragdoll_X_Twisted%2Cnetwork_Unknown+Devices+%28restored+1esqjar0%29%2Cnetwork_Meta%2Cnetwork_Measurement+Consent+Updated+Before+Install%2Cnetwork_Unknown+Devices+%28restored+1770rwkr%29%2Cnetwork_Unknown&applovin_mode=probabilistic&ironsource_mode=ironsource&digital_turbine_mode=digital_turbine&dimensions=app%2Cchannel%2Ccampaign_network%2Cweek&metrics=installs%2Cinstalls_per_mile%2Ccost%2Call_revenue%2Cgross_profit%2Ccustom_roas%2Cattribution_clicks%2Cattribution_impressions%2Cnetwork_ctr%2Cecpi%2Cretention_rate_d1%2Cretention_rate_d3%2Cretention_rate_d7%2Cretention_rate_d14%2Cretention_rate_d30%2Croas_d0%2Croas_d3%2Croas_d7%2Croas_d14%2Croas_d30%2Croas_d45%2Croas_d60%2Croas_d75%2Croas_d90%2Croas_d120%2Croas_ad_d0%2Croas_ad_d7%2Croas_ad_d14%2Croas_ad_d30%2Croas_ad_d45%2Croas_ad_d60%2Croas_ad_d75%2Croas_ad_d90%2Croas_ad_d120%2Croas_iap_d0%2Croas_iap_d7%2Croas_iap_d14%2Croas_iap_d30%2Croas_iap_d45%2Croas_iap_d60%2Croas_iap_d75%2Croas_iap_d90%2Croas_iap_d120&sort=app&table_view=pivot&parent_report_id=243954)" – this link will take you directly to the Adjust report you need to export.
                - Upload your Adjust report using the file uploader at the top of the dashboard.
                - The file must include these required columns:
                `app`, `channel`, `campaign_network`, `week`, `cost`, `installs`, `roas_d0`, `roas_d7`, `roas_d14`, ..., `roas_d120`.
                """)

            with st.expander("🎯 2. Select Channel and Campaign"):
                st.markdown("""
                - Use the **Channel** dropdown to pick an ad network (e.g., Applovin, Mintegral).
                - Then select a **Campaign** under that network to see detailed performance data.
                """)

            with st.expander("🧮 3. Overview of Dashboard Tabs"):
                st.markdown("""
            <ul>
                <li><b style="color:#FF6F61">📊 Performance Table</b>: View installs, cost, CPI, and ROAS values — the current performance of your campaign.</li>
                <li><b style="color:#FF6F61">📈 Growth Rates</b>: Shows ROAS growth intervals (e.g., D0→D7, D7→D14)</li>
                <li><b style="color:#FF6F61">📦 Channels Campaigns Comparison</b>: Compare campaigns within Applovin and Mintegral to evaluate which performs better. - (⚠️ The table reflects data from the most recent 4 fully matured weeks only)</li>
                <li><b style="color:#FF6F61">🎯 Break-Even Analysis</b>: Enter a goal (D0/D7/D28) and see future ROAS projections. - (⚠️ The table reflects data from the most recent 4 fully matured weeks only)</li>
                <li><b style="color:#FF6F61">📊 Visualizations</b>: Interactive charts showing spend, revenue, ROAS trends, and growth.</li>
                </ul>
                    """, unsafe_allow_html=True)


            with st.expander("🔍 4. ROAS Growth Calculation"):
                st.markdown("""
                - ROAS Growth is calculated as:
                ```
                ROAS at end of interval / ROAS at start of interval
                ```
                - Example:  
                If ROAS D14 = 1.0 and ROAS D7 = 0.8 →  
                `1.0 / 0.8 = 1.25` → A 25% improvement between D7 and D14.
                """)

            with st.expander("⚠️ 5. Troubleshooting Guide"):
                st.markdown("""
                | Issue                        | Solution                                                 |
                |-----------------------------|----------------------------------------------------------|
                | Missing required columns    | Check the error message and verify all necessary fields. |
                | Date parsing fails          | Make sure the `week` column uses `YYYY-MM-DD` format.    |
                | CPI shows as NaN or inf     | Ensure `installs` are not zero.                          |
                | ROAS Growth = 0             | Check for missing or zero ROAS values in those columns.  |
                | “No valid data” warning     | Ensure valid dates exist and are correctly parsed.       |
                """)

