"""
Comprehensive Visualization Components
Sector graphs, Channel graphs, Weekly price charts
Based on WMS-LLM Agent
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any


def render_sector_performance_charts(holdings_df: pd.DataFrame):
    """
    Render sector performance analysis with pie chart and bar chart
    
    Args:
        holdings_df: DataFrame with holdings data (must include sector, invested_amount, current_value columns)
    """
    st.subheader("📊 Sector Performance Analysis")
    
    # Prepare sector data
    sector_data = holdings_df.groupby('sector').agg({
        'invested_amount': 'sum',
        'current_value': 'sum'
    }).reset_index()
    
    sector_data['unrealized_pnl'] = sector_data['current_value'] - sector_data['invested_amount']
    sector_data['pnl_percentage'] = (sector_data['unrealized_pnl'] / sector_data['invested_amount'] * 100)
    sector_data = sector_data.sort_values('pnl_percentage', ascending=False)
    
    # Display table
    st.dataframe(
        sector_data.style.format({
            'invested_amount': '₹{:,.0f}',
            'current_value': '₹{:,.0f}',
            'unrealized_pnl': '₹{:,.0f}',
            'pnl_percentage': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart - Allocation
        fig_pie = px.pie(
            sector_data,
            values='current_value',
            names='sector',
            title='Sector Allocation by Value',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart - Performance
        fig_bar = px.bar(
            sector_data,
            x='sector',
            y='pnl_percentage',
            title='Sector P&L Performance (%)',
            color='pnl_percentage',
            color_continuous_scale='RdYlGn',
            text='pnl_percentage'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)


def render_channel_performance_charts(holdings_df: pd.DataFrame):
    """
    Render channel performance analysis with ratings
    
    Args:
        holdings_df: DataFrame with holdings data (must include channel, invested_amount, current_value columns)
    """
    st.subheader("📡 Channel Performance Analysis")
    
    # Check if channel data exists
    if 'channel' not in holdings_df.columns or holdings_df['channel'].isna().all():
        st.warning("Channel information not available in data")
        return
    
    # Prepare channel data
    channel_data = holdings_df.groupby('channel').agg({
        'invested_amount': 'sum',
        'current_value': 'sum'
    }).reset_index()
    
    channel_data['unrealized_pnl'] = channel_data['current_value'] - channel_data['invested_amount']
    channel_data['pnl_percentage'] = (channel_data['unrealized_pnl'] / channel_data['invested_amount'] * 100)
    
    # Add rating based on performance
    def get_rating(return_pct):
        if return_pct >= 20:
            return "🟢 Excellent"
        elif return_pct >= 10:
            return "🟡 Good"
        elif return_pct >= 0:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    channel_data['rating'] = channel_data['pnl_percentage'].apply(get_rating)
    channel_data = channel_data.sort_values('pnl_percentage', ascending=False)
    
    # Display table with color coding
    st.dataframe(
        channel_data.style.format({
            'invested_amount': '₹{:,.0f}',
            'current_value': '₹{:,.0f}',
            'unrealized_pnl': '₹{:,.0f}',
            'pnl_percentage': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart - Channel allocation
        fig_pie = px.pie(
            channel_data,
            values='current_value',
            names='channel',
            title='Channel Allocation by Value',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart - Channel performance
        fig_bar = px.bar(
            channel_data,
            x='channel',
            y='pnl_percentage',
            title='Channel P&L Performance (%)',
            color='pnl_percentage',
            color_continuous_scale='RdYlGn',
            text='pnl_percentage'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)


def render_weekly_price_comparison(tickers_data: Dict[str, pd.DataFrame], avg_prices: Dict[str, float]):
    """
    Render weekly price comparison chart for multiple tickers
    Shows 52 weeks (1 year) of price movements
    
    Args:
        tickers_data: Dict mapping ticker to DataFrame with price_date and price columns
        avg_prices: Dict mapping ticker to average purchase price
    """
    st.subheader("📈 Weekly Price Comparison (1 Year)")
    
    if not tickers_data:
        st.info("No weekly data available")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each ticker
    for ticker, df in tickers_data.items():
        if df.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=df['price_date'],
            y=df['price'],
            mode='lines+markers',
            name=ticker,
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Price: ₹%{{y:.2f}}<extra></extra>"
        ))
        
        # Add average price line if available
        if ticker in avg_prices:
            avg_price = avg_prices[ticker]
            fig.add_hline(
                y=avg_price,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"{ticker} Avg: ₹{avg_price:.2f}",
                annotation_position="right"
            )
    
    fig.update_layout(
        title='Weekly Price Movement (52 Weeks)',
        xaxis_title='Week',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_individual_stock_weekly_chart(ticker: str, weekly_df: pd.DataFrame, avg_price: float, stock_name: str = None):
    """
    Render individual stock weekly price chart with P&L subplot
    Shows 52 weeks of data
    
    Args:
        ticker: Ticker symbol
        weekly_df: DataFrame with price_date and price columns
        avg_price: Average purchase price
        stock_name: Optional stock name for display
    """
    if weekly_df.empty:
        st.warning(f"No weekly data available for {ticker}")
        return
    
    display_name = f"{stock_name} ({ticker})" if stock_name else ticker
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{display_name} - Weekly Price Movement',
            f'{display_name} - Weekly P&L Progression'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Calculate P&L
    weekly_df = weekly_df.copy()
    weekly_df['pnl'] = (weekly_df['price'] - avg_price) * 1  # P&L per unit
    weekly_df['pnl_pct'] = ((weekly_df['price'] - avg_price) / avg_price * 100)
    
    # Row 1: Price chart
    fig.add_trace(
        go.Scatter(
            x=weekly_df['price_date'],
            y=weekly_df['price'],
            mode='lines+markers',
            name='Price',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8),
            hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add purchase price line
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text=f"Avg Price: ₹{avg_price:.2f}",
        row=1, col=1
    )
    
    # Row 2: P&L chart
    fig.add_trace(
        go.Scatter(
            x=weekly_df['price_date'],
            y=weekly_df['pnl_pct'],
            mode='lines+markers',
            name='P&L %',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            hovertemplate='Date: %{x}<br>P&L: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text=f"{display_name} - 52 Weeks Analysis"
    )
    
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="P&L (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    latest = weekly_df.iloc[-1]
    first = weekly_df.iloc[0]
    
    with col1:
        st.metric("Purchase Price", f"₹{avg_price:.2f}")
    
    with col2:
        change = latest['price'] - avg_price
        change_color = "normal" if change >= 0 else "inverse"
        st.metric(
            "Current Price",
            f"₹{latest['price']:.2f}",
            f"₹{change:+.2f}",
            delta_color=change_color
        )
    
    with col3:
        st.metric("Highest (52W)", f"₹{weekly_df['price'].max():.2f}")
    
    with col4:
        st.metric("Lowest (52W)", f"₹{weekly_df['price'].min():.2f}")
    
    # Weekly changes table
    st.markdown("### 📊 Weekly Changes (Last 12 Weeks)")
    
    recent_weeks = weekly_df.tail(12).copy()
    recent_weeks['week_change'] = recent_weeks['price'].diff()
    recent_weeks['week_change_pct'] = recent_weeks['price'].pct_change() * 100
    recent_weeks = recent_weeks.iloc[::-1].reset_index(drop=True)  # Reverse to show latest first
    
    st.dataframe(
        recent_weeks[['price_date', 'price', 'week_change', 'week_change_pct', 'pnl_pct']].style.format({
            'price': '₹{:.2f}',
            'week_change': '₹{:.2f}',
            'week_change_pct': '{:.2f}%',
            'pnl_pct': '{:.2f}%'
        }).applymap(
            lambda x: 'color: #00c853' if isinstance(x, (int, float)) and x > 0 else ('color: #ff1744' if isinstance(x, (int, float)) and x < 0 else ''),
            subset=['week_change', 'week_change_pct', 'pnl_pct']
        ),
        use_container_width=True
    )


def render_sector_drill_down(holdings_df: pd.DataFrame):
    """
    Interactive sector drill-down with multi-select
    
    Args:
        holdings_df: DataFrame with holdings
    """
    st.markdown("---")
    st.subheader("🔍 Sector Drill-Down (Multi-Select)")
    
    # Get unique sectors
    sectors = sorted(holdings_df['sector'].dropna().unique().tolist())
    
    selected_sectors = st.multiselect(
        "Select sector(s) to analyze:",
        options=sectors,
        default=[],
        help="Select one or more sectors to view detailed holdings"
    )
    
    if selected_sectors:
        # Filter holdings
        sector_holdings = holdings_df[holdings_df['sector'].isin(selected_sectors)]
        
        st.info(f"📊 Showing {len(sector_holdings)} holdings in: {', '.join(selected_sectors)}")
        
        # If multiple sectors, show comparison
        if len(selected_sectors) > 1:
            st.markdown("### 📊 Sector Comparison")
            
            sector_comparison = sector_holdings.groupby('sector').agg({
                'invested_amount': 'sum',
                'current_value': 'sum'
            }).reset_index()
            
            sector_comparison['unrealized_pnl'] = sector_comparison['current_value'] - sector_comparison['invested_amount']
            sector_comparison['pnl_percentage'] = (sector_comparison['unrealized_pnl'] / sector_comparison['invested_amount'] * 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    sector_comparison.style.format({
                        'invested_amount': '₹{:,.0f}',
                        'current_value': '₹{:,.0f}',
                        'unrealized_pnl': '₹{:,.0f}',
                        'pnl_percentage': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                fig = px.bar(
                    sector_comparison,
                    x='sector',
                    y='pnl_percentage',
                    title='Sector P&L Comparison',
                    color='pnl_percentage',
                    color_continuous_scale='RdYlGn',
                    text='pnl_percentage'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        # Show holdings table
        st.markdown("### 📋 Holdings Details")
        
        display_cols = ['asset_name', 'asset_symbol', 'sector', 'quantity', 'invested_amount', 'current_value', 'unrealized_pnl', 'pnl_percentage']
        display_df = sector_holdings[display_cols].copy()
        
        st.dataframe(
            display_df.style.format({
                'quantity': '{:.4f}',
                'invested_amount': '₹{:,.0f}',
                'current_value': '₹{:,.0f}',
                'unrealized_pnl': '₹{:,.0f}',
                'pnl_percentage': '{:.2f}%'
            }).applymap(
                lambda x: 'color: #00c853' if isinstance(x, (int, float)) and x > 0 else ('color: #ff1744' if isinstance(x, (int, float)) and x < 0 else ''),
                subset=['unrealized_pnl', 'pnl_percentage']
            ),
            use_container_width=True
        )


def render_channel_drill_down(holdings_df: pd.DataFrame):
    """
    Interactive channel drill-down with ratings
    
    Args:
        holdings_df: DataFrame with holdings
    """
    st.markdown("---")
    st.subheader("🔍 Channel Drill-Down")
    
    if 'channel' not in holdings_df.columns:
        st.warning("Channel information not available")
        return
    
    # Get unique channels
    channels = sorted(holdings_df['channel'].dropna().unique().tolist())
    
    selected_channels = st.multiselect(
        "Select channel(s) to analyze:",
        options=channels,
        default=[],
        help="Select one or more channels to view detailed holdings"
    )
    
    if selected_channels:
        channel_holdings = holdings_df[holdings_df['channel'].isin(selected_channels)]
        
        st.info(f"📊 Showing {len(channel_holdings)} holdings from: {', '.join(selected_channels)}")
        
        # Show holdings
        display_cols = ['asset_name', 'asset_symbol', 'channel', 'quantity', 'invested_amount', 'current_value', 'unrealized_pnl', 'pnl_percentage']
        
        st.dataframe(
            channel_holdings[display_cols].style.format({
                'quantity': '{:.4f}',
                'invested_amount': '₹{:,.0f}',
                'current_value': '₹{:,.0f}',
                'unrealized_pnl': '₹{:,.0f}',
                'pnl_percentage': '{:.2f}%'
            }),
            use_container_width=True
        )


def render_52_week_trends(weekly_manager, tickers_with_types: List[tuple], avg_prices: Dict[str, float]):
    """
    Render 52-week price trends for selected holdings
    
    Args:
        weekly_manager: WeeklyPriceManager instance
        tickers_with_types: List of (ticker, asset_type) tuples
        avg_prices: Dict of average purchase prices
    """
    st.subheader("📉 52-Week Price Trends")
    
    if not tickers_with_types:
        st.info("Select holdings to view weekly trends")
        return
    
    # Get weekly data for all tickers
    weekly_data = {}
    
    for ticker, asset_type in tickers_with_types:
        df = weekly_manager.get_weekly_data_for_ticker(ticker, asset_type, weeks=52)
        if not df.empty:
            weekly_data[ticker] = df
    
    if not weekly_data:
        st.warning("No weekly data available. Click 'Refresh Prices' to populate cache.")
        return
    
    # Render comparison chart
    render_weekly_price_comparison(weekly_data, avg_prices)
    
    # Individual stock selection
    st.markdown("---")
    st.subheader("📊 Individual Stock Analysis")
    
    ticker_options = list(weekly_data.keys())
    selected_ticker = st.selectbox(
        "Select a holding for detailed weekly analysis:",
        options=ticker_options
    )
    
    if selected_ticker and selected_ticker in weekly_data:
        weekly_df = weekly_data[selected_ticker]
        avg_price = avg_prices.get(selected_ticker, weekly_df['price'].iloc[0])
        
        render_individual_stock_weekly_chart(
            selected_ticker,
            weekly_df,
            avg_price
        )


def render_monthly_performance_heatmap(holdings_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Render monthly performance heatmap
    
    Args:
        holdings_df: DataFrame with holdings
        transactions_df: DataFrame with transactions
    """
    st.subheader("📅 Monthly Performance Heatmap")
    
    try:
        # Group transactions by month
        transactions_df['month'] = pd.to_datetime(transactions_df['transaction_date']).dt.to_period('M')
        
        monthly_data = transactions_df.groupby(['month', 'transaction_type']).agg({
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        # Calculate monthly net investment
        monthly_net = monthly_data.pivot_table(
            index='month',
            columns='transaction_type',
            values='quantity',
            fill_value=0
        )
        
        if 'buy' in monthly_net.columns and 'sell' in monthly_net.columns:
            monthly_net['net_investment'] = (monthly_net['buy'] - monthly_net['sell'])
        elif 'buy' in monthly_net.columns:
            monthly_net['net_investment'] = monthly_net['buy']
        else:
            st.info("No monthly data available")
            return
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[monthly_net['net_investment'].values],
            x=monthly_net.index.astype(str),
            y=['Net Investment'],
            colorscale='RdYlGn',
            text=[monthly_net['net_investment'].values],
            texttemplate='%{text:.0f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Monthly Investment Activity',
            xaxis_title='Month',
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Error rendering monthly heatmap: {str(e)}")

