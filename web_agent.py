"""
Streamlined Wealth Manager - Matches Your Image Requirements
- Register/Login with file upload
- Store files to DB and calculate historical from date in file based on week of year
- Fetch missing weeks till current week based on week of year
- Calculate P&L based on current week price
- Portfolio analysis based on sector/channel
- PMS CAGR when price mentioned, 52-week NAVs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
import functools
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Performance optimization decorators
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_holdings(user_id: str):
    """Cache holdings data to avoid repeated database calls"""
    from database_shared import SharedDatabaseManager
    db = SharedDatabaseManager()
    return db.get_user_holdings_silent(user_id)

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_portfolio_summary(holdings: List[Dict]) -> str:
    """Cache comprehensive portfolio summary calculation"""
    if not holdings:
        return "No holdings found"
    
    # Safe conversion to handle None values
    def safe_float(value, default=0):
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    total_investment = 0
    total_current = 0
    
    # Asset type breakdown
    asset_types = {}
    channels = {}
    sectors = {}
    
    for holding in holdings:
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        # Use safe_float to handle None values
        current_value = safe_float(current_price, 0) * safe_float(holding.get('total_quantity'), 0)
        investment = safe_float(holding.get('total_quantity'), 0) * safe_float(holding.get('average_price'), 0)
        total_investment += investment
        total_current += current_value
        
        # Track asset types
        asset_type = holding.get('asset_type', 'Unknown')
        if asset_type not in asset_types:
            asset_types[asset_type] = {'investment': 0, 'current': 0, 'count': 0}
        asset_types[asset_type]['investment'] += investment
        asset_types[asset_type]['current'] += current_value
        asset_types[asset_type]['count'] += 1
        
        # Track channels
        channel = holding.get('channel', 'Unknown')
        if channel not in channels:
            channels[channel] = {'investment': 0, 'current': 0, 'count': 0}
        channels[channel]['investment'] += investment
        channels[channel]['current'] += current_value
        channels[channel]['count'] += 1
        
        # Track sectors
        sector = holding.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = {'investment': 0, 'current': 0, 'count': 0}
        sectors[sector]['investment'] += investment
        sectors[sector]['current'] += current_value
        sectors[sector]['count'] += 1
    
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    
    portfolio_summary = f"""📊 COMPREHENSIVE PORTFOLIO OVERVIEW:

💰 FINANCIAL SUMMARY:
• Total Holdings: {len(holdings)} assets
• Total Investment: ₹{total_investment:,.0f}
• Current Value: ₹{total_current:,.0f}
• Total P&L: ₹{total_pnl:,.0f} ({total_pnl_pct:+.1f}%)

📈 ASSET TYPE BREAKDOWN:"""
    
    for asset_type, data in sorted(asset_types.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {asset_type.title()}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    portfolio_summary += f"\n\n🏢 CHANNEL BREAKDOWN:"
    for channel, data in sorted(channels.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {channel}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    portfolio_summary += f"\n\n🏭 SECTOR BREAKDOWN:"
    for sector, data in sorted(sectors.items(), key=lambda x: x[1]['current'], reverse=True):
        pnl = data['current'] - data['investment']
        pnl_pct = (pnl / data['investment'] * 100) if data['investment'] > 0 else 0
        portfolio_summary += f"\n• {sector}: {data['count']} holdings, ₹{data['current']:,.0f} ({pnl_pct:+.1f}%)"
    
    # Calculate P&L for all holdings for top gainers/losers
    holdings_with_pnl = []
    for holding in holdings:
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        current_value = safe_float(current_price, 0) * safe_float(holding.get('total_quantity'), 0)
        investment = safe_float(holding.get('total_quantity'), 0) * safe_float(holding.get('average_price'), 0)
        pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0
        
        holdings_with_pnl.append({
            'ticker': holding.get('ticker', 'N/A'),
            'stock_name': holding.get('stock_name', 'N/A'),
            'asset_type': holding.get('asset_type', 'Unknown'),
            'channel': holding.get('channel', 'Unknown'),
            'sector': holding.get('sector', 'Unknown'),
            'current_value': current_value,
            'pnl_pct': pnl_pct
        })
    
    portfolio_summary += f"\n\n🏆 TOP 5 GAINERS (by P&L %):"
    top_gainers = sorted(holdings_with_pnl, key=lambda h: h['pnl_pct'], reverse=True)[:5]
    for holding in top_gainers:
        emoji = "🚀" if holding['pnl_pct'] > 10 else "📈"
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    portfolio_summary += f"\n\n📉 TOP 5 LOSERS (by P&L %):"
    top_losers = sorted(holdings_with_pnl, key=lambda h: h['pnl_pct'])[:5]
    for holding in top_losers:
        emoji = "📉" if holding['pnl_pct'] < 0 else "➡️"
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    portfolio_summary += f"\n\n🏆 TOP 10 HOLDINGS (by current value):"
    
    # Sort holdings by current value and take top 10
    sorted_holdings = sorted(
        holdings_with_pnl, 
        key=lambda h: h['current_value'], 
        reverse=True
    )
    
    for holding in sorted_holdings[:10]:
        emoji = "🚀" if holding['pnl_pct'] > 10 else "📈" if holding['pnl_pct'] > 0 else "📉" if holding['pnl_pct'] < -5 else "➡️"
        
        portfolio_summary += f"\n{emoji} {holding['ticker']} ({holding['asset_type']}) - {holding['stock_name'][:30]}"
        portfolio_summary += f"\n   Channel: {holding['channel']} | Sector: {holding['sector']} | P&L: {holding['pnl_pct']:+.1f}% | Value: ₹{holding['current_value']:,.0f}"
    
    return portfolio_summary

# Import modules
from database_shared import SharedDatabaseManager
from enhanced_price_fetcher import EnhancedPriceFetcher
from bulk_ai_fetcher import BulkAIFetcher
from weekly_manager_streamlined import StreamlinedWeeklyManager
from smart_ticker_detector import detect_ticker_type, normalize_ticker

# Page configuration
st.set_page_config(
    page_title="Wealth Manager - Streamlined",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'db' not in st.session_state:
    st.session_state.db = SharedDatabaseManager()
if 'price_fetcher' not in st.session_state:
    st.session_state.price_fetcher = EnhancedPriceFetcher()

# Function to update all live prices
def update_all_live_prices():
    """Update live prices for all holdings"""
    if 'user_id' in st.session_state:
        holdings = db.get_user_holdings(st.session_state.user_id)
        if holdings:
            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings, db)
            st.success("✅ Live prices updated successfully!")
            st.rerun()
        else:
            st.warning("No holdings found to update.")

def should_update_prices_today(holdings):
    """Check if prices need to be updated today"""
    from datetime import datetime, date
    
    today = date.today()
    needs_update = []
    
    for holding in holdings:
        stock_id = holding.get('stock_id')
        if stock_id:
            # Get last updated date from database
            try:
                # Check if method exists
                if hasattr(db, 'get_stock_last_updated'):
                    last_updated = db.get_stock_last_updated(stock_id)
                    if last_updated:
                        last_updated_date = datetime.fromisoformat(last_updated).date()
                        if last_updated_date < today:
                            needs_update.append(holding)
                    else:
                        # No last_updated record, needs update
                        needs_update.append(holding)
                else:
                    # Method not available, assume needs update
                    needs_update.append(holding)
            except Exception as e:
                # Any error, assume needs update
                needs_update.append(holding)
    
    return needs_update
if 'bulk_ai_fetcher' not in st.session_state:
    st.session_state.bulk_ai_fetcher = BulkAIFetcher()
if 'weekly_manager' not in st.session_state:
    st.session_state.weekly_manager = StreamlinedWeeklyManager(
        st.session_state.db,
        st.session_state.price_fetcher
    )

db = st.session_state.db
price_fetcher = st.session_state.price_fetcher
bulk_ai_fetcher = st.session_state.bulk_ai_fetcher
weekly_manager = st.session_state.weekly_manager

# ============================================================================
# AUTHENTICATION
# ============================================================================

def login_page():
    """Login page"""
    st.title("💰 Wealth Manager - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            # Convert username to lowercase for case-insensitive login
            user = db.login_user(username.lower(), password)
            if user:
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        full_name = st.text_input("Full Name", key="register_name")
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email (Optional)", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")
        
        # File upload during registration (as per your image)
        st.subheader("Upload Transaction Files")
        uploaded_files = st.file_uploader(
            "Upload CSV files with transactions",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload your transaction files during registration"
        )
        
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords don't match")
            elif not username:
                st.error("Username is required")
            else:
                # Convert username to lowercase for case-insensitive registration
                result = db.register_user(username.lower(), password, full_name, email)
                if result['success']:
                    user = result['user']
                    st.session_state.user = user
                    
                    # Create default portfolio
                    portfolio_result = db.create_portfolio(user['id'], "Main Portfolio")
                    if portfolio_result['success']:
                        portfolio_id = portfolio_result['portfolio']['id']
                        
                        # Process uploaded files (as per your image)
                        if uploaded_files:
                            st.info("📁 Processing uploaded files...")
                            process_uploaded_files(uploaded_files, user['id'], portfolio_id)
                            st.success("✅ Registration and file processing complete!")
                        else:
                            st.success("✅ Registration successful!")
                        
                        st.info("🔄 Redirecting to dashboard...")
                        time.sleep(2)  # Brief pause to show success message
                        st.rerun()
                else:
                    st.error(f"Registration failed: {result['error']}")

def process_uploaded_files(uploaded_files, user_id, portfolio_id):
    """
    Process uploaded files and store to DB
    Matches your image: "store the files to db and calculate historical from date in file and store based on week of year"
    """
    total_imported = 0
    processing_log = []
    
    st.info(f"🚀 Starting to process {len(uploaded_files)} file(s)...")
    
    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        st.caption(f"📁 [{file_idx}/{len(uploaded_files)}] Processing {uploaded_file.name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.caption(f"   📊 Found {len(df)} rows in {uploaded_file.name}")
            
            imported = 0
            skipped = 0
            errors = 0
            
            # Progress bar for large files
            if len(df) > 10:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            for idx, row in df.iterrows():
                try:
                    # Detect ticker type
                    ticker = str(row.get('ticker', '')).strip()
                    if not ticker:
                        skipped += 1
                        continue
                    
                    # Log ticker detection
                    asset_type = detect_ticker_type(ticker)
                    normalized_ticker = normalize_ticker(ticker, asset_type)
                    #st.caption(f"   🔍 Row {idx+1}: {ticker} → {normalized_ticker} ({asset_type})")
                    
                    # Parse date
                    try:
                        trans_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                        #st.caption(f"   📅 Date: {trans_date}")
                    except Exception as date_error:
                        trans_date = datetime.now().strftime('%Y-%m-%d')
                        #st.caption(f"   ⚠️ Date parsing failed, using today: {trans_date}")
                    
                    # Get price - if not provided, fetch historical price for that date
                    price = row.get('price', 0)
                    if pd.isna(price) or price == '' or price == 0:
                        ##st.caption(f"   💰 Price: Not provided, fetching historical price for {trans_date}...")
                        
                        # Fetch historical price for the transaction date
                        try:
                            # Get fund name for enhanced AI fallback
                            fund_name = row.get('stock_name', ticker) if asset_type == 'mutual_fund' else None
                            historical_price = price_fetcher.get_historical_price(
                                normalized_ticker, 
                                asset_type, 
                                trans_date,
                                fund_name
                            )
                            if historical_price and historical_price > 0:
                                price = historical_price
                                #st.caption(f"   ✅ Historical price: ₹{price:,.2f}")
                            else:
                                price = 0
                                #st.caption(f"   ⚠️ Could not fetch historical price (using 0)")
                        except Exception as e:
                            price = 0
                            #st.caption(f"   ❌ Error fetching price: {str(e)[:50]}")
                    else:
                        price = float(price)
                        ##st.caption(f"   💰 Price: ₹{price:,.2f}")
                    
                    # Determine channel: check CSV column first, then use filename
                    channel = None
                    if 'channel' in row and pd.notna(row['channel']) and str(row['channel']).strip():
                        channel = str(row['channel']).strip()
                    else:
                        channel = uploaded_file.name.replace('.csv', '')
                    
                    # Create transaction (week tracking auto-calculated in database_shared.py)
                    transaction_data = {
                        'user_id': user_id,
                        'portfolio_id': portfolio_id,
                        'ticker': normalized_ticker,
                        'stock_name': row.get('stock_name', ticker),
                        'asset_type': asset_type,
                        'sector': row.get('sector', 'Unknown'),
                        'transaction_type': 'buy' if 'buy' in str(row['transaction_type']).lower() else 'sell',
                        'quantity': float(row['quantity']),
                        'price': price,
                        'transaction_date': trans_date,
                        'channel': channel,  # Channel from CSV column or filename
                        'notes': f"Imported from {uploaded_file.name}"
                    }
                    
                    st.caption(f"   💾 Saving transaction to database...")
                    result = db.add_transaction(transaction_data)
                    
                    if result['success']:
                        imported += 1
                        st.caption(f"   ✅ Transaction saved successfully")
                        
                        # Log week calculation
                        if 'week_label' in result.get('transaction', {}):
                            week_label = result['transaction']['week_label']
                            st.caption(f"   📅 Week calculated: {week_label}")
                        else:
                            pass
#st.caption(f"   ⚠️ Week calculation may have failed")
                    else:
                        errors += 1
                        #st.caption(f"   ❌ Database error: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    errors += 1
                    #st.caption(f"   ❌ Error in row {idx+1}: {str(e)}")
                
                # Update progress bar
                if len(df) > 10:
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing row {idx+1}/{len(df)}")
            
            # Clear progress bar
            if len(df) > 10:
                progress_bar.empty()
                progress_text.empty()
            
            total_imported += imported
            
            # File summary
            file_summary = {
                'file': uploaded_file.name,
                'total_rows': len(df),
                'imported': imported,
                'skipped': skipped,
                'errors': errors
            }
            processing_log.append(file_summary)
            
            st.caption(f"✅ {uploaded_file.name}: {imported}/{len(df)} transactions imported ({skipped} skipped, {errors} errors)")
        
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            processing_log.append({
                'file': uploaded_file.name,
                'error': str(e)
            })
    
    # Final summary
    st.success(f"🎉 Processing Complete!")
    st.info(f"📊 **Final Summary:**")
    st.info(f"   • Total files processed: {len(uploaded_files)}")
    st.info(f"   • Total transactions imported: {total_imported}")
    
    # Detailed log
    with st.expander("📋 Detailed Processing Log"):
        for log in processing_log:
            if 'error' in log:
                st.error(f"❌ {log['file']}: {log['error']}")
            else:
                st.success(f"✅ {log['file']}: {log['imported']}/{log['total_rows']} imported ({log['skipped']} skipped, {log['errors']} errors)")
    
    if total_imported > 0:
        st.info("🔄 Next: Fetching missing weekly prices for your holdings...")
        
        # Automatically fetch missing weeks after import
        st.subheader("📅 Fetching Historical Prices")
        
        with st.spinner("Fetching missing weekly prices..."):
            try:
                # Import weekly manager if not already available
                from weekly_manager_streamlined import StreamlinedWeeklyManager
                
                # Initialize if needed
                if 'weekly_manager' not in st.session_state:
                    st.session_state.weekly_manager = StreamlinedWeeklyManager(
                        st.session_state.db,
                        st.session_state.price_fetcher
                    )
                
                weekly_manager = st.session_state.weekly_manager
                
                # Fetch missing weeks (silent)
                result = weekly_manager.fetch_missing_weeks_till_current(user_id)
                
                if result['success']:
                    fetched_count = result.get('fetched', 0)
                    if fetched_count > 0:
                        st.success(f"✅ Fetched {fetched_count} missing week prices!")
                    else:
                        st.info("✅ All weeks already up-to-date")
                else:
                    st.warning(f"⚠️ Some prices could not be fetched: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"❌ Error fetching prices: {str(e)}")
                st.caption("You can manually refresh prices from the sidebar after login.")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main_dashboard():
    """Main dashboard after login"""
    user = st.session_state.user
    
    st.title(f"💰 Welcome, {user['full_name']}")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    
    # Auto-fetch missing weeks on login (as per your image)
    # Only fetch once per session to avoid duplicate fetching
    if 'missing_weeks_fetched' not in st.session_state:
        # Show loading animation during initial setup
        with st.spinner("🔄 Initializing your portfolio..."):
            # Create a nice loading container
            loading_container = st.container()
            with loading_container:
                st.markdown("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #f0f2f6, #e1e5e9); border-radius: 10px; margin: 10px 0;">
                    <div style="font-size: 18px; color: #1f2937; margin-bottom: 10px;">
                        🚀 Setting up your wealth management dashboard...
                    </div>
                    <div style="font-size: 14px; color: #6b7280;">
                        • Analyzing holdings and transactions<br>
                        • Fetching latest market prices<br>
                        • Calculating portfolio performance<br>
                        • Preparing insights and analytics
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress steps
                steps = [
                    "🔍 Analyzing user holdings and transaction weeks...",
                    "📊 Processing portfolio data...",
                    "💰 Fetching latest market prices...",
                    "📈 Calculating performance metrics...",
                    "🎯 Preparing personalized insights...",
                    "✅ Dashboard ready!"
                ]
                
                for i, step in enumerate(steps):
                    progress_bar.progress((i + 1) / len(steps))
                    status_text.text(step)
                    time.sleep(0.3)  # Small delay for visual effect
                
        # Auto-fetch missing weeks and update prices (silent background process)
            holdings = db.get_user_holdings(user['id'])
            if holdings:
            # Auto-fetch missing weeks (silent)
                result = weekly_manager.fetch_missing_weeks_till_current(user['id'])
                
            # Smart price update - only if needed (silent)
                try:
                    holdings_needing_update = should_update_prices_today(holdings)
                    if holdings_needing_update:
                        # Only update holdings that haven't been updated today
                        st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
                    # If no holdings need update, prices are already current
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Price update: {str(e)[:50]}")
            else:
                result = {'success': True, 'fetched': 0}
        
        # Mark as fetched to prevent re-fetching on page navigation
        st.session_state.missing_weeks_fetched = True
        st.session_state.last_fetch_time = datetime.now()
        st.rerun()
            
    else:
        # Already fetched this session
        if 'last_fetch_time' in st.session_state:
            time_since_fetch = (datetime.now() - st.session_state.last_fetch_time).total_seconds() / 60
            st.sidebar.caption(f"✅ Prices checked {int(time_since_fetch)} min ago")
            
            # Add refresh button (will re-fetch both historical and current prices)
            if st.sidebar.button("🔄 Refresh All Prices"):
                st.session_state.missing_weeks_fetched = False
                st.rerun()
    
    # Navigation
    page = st.sidebar.radio(
        "Choose a page:",
        [
            "🏠 Portfolio Overview",
            "📊 P&L Analysis",
            "📈 Charts & Analytics",
            "🤖 AI Assistant",
            "📁 Upload More Files"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Simplified sidebar - AI Assistant now has its own page
    with st.sidebar:
        st.markdown("**💡 Quick Access**")
        st.caption("🤖 AI Assistant now has its own dedicated page!")
        st.caption("📊 Access all portfolio insights and chat features")
        st.caption("📚 Upload and analyze PDF documents")
        st.caption("💬 Chat with AI about your portfolio")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**👤 {user['full_name']}**")
    st.sidebar.markdown(f"📧 {user['email']}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.missing_weeks_fetched = False
        st.rerun()
    
    # Route to pages
    if page == "🏠 Portfolio Overview":
        portfolio_overview_page()
    elif page == "📊 P&L Analysis":
        pnl_analysis_page()
    elif page == "📈 Charts & Analytics":
        charts_page()
    elif page == "🤖 AI Assistant":
        ai_assistant_page()
    elif page == "📁 Upload More Files":
        upload_files_page()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_portfolio_metrics(holdings: List[Dict]) -> Dict[str, Any]:
    """Cache expensive portfolio calculations"""
    if not holdings:
        return {}
    
    # Calculate all metrics in one pass
    total_investment = 0
    total_current = 0
    gainers = 0
    momentum_stocks = 0
    returns = []
    
    for holding in holdings:
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        current_value = float(current_price) * float(holding['total_quantity'])
        investment = float(holding['total_quantity']) * float(holding['average_price'])
        
        total_investment += investment
        total_current += current_value
        
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100) if investment > 0 else 0
        returns.append(pnl_pct)
        
        if pnl > 0:
            gainers += 1
        if pnl_pct > 10:
            momentum_stocks += 1
    
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    market_breadth = (gainers / len(holdings) * 100) if holdings else 0
    momentum_score = (momentum_stocks / len(holdings) * 100) if holdings else 0
    volatility = np.std(returns) if returns else 0
    
    return {
        'total_investment': total_investment,
        'total_current': total_current,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'market_breadth': market_breadth,
        'momentum_score': momentum_score,
        'volatility': volatility,
        'gainers': gainers,
        'momentum_stocks': momentum_stocks,
        'total_holdings': len(holdings)
    }

def portfolio_overview_page():
    """Portfolio overview with current week prices"""
    st.header("🏠 Portfolio Overview")
    
    user = st.session_state.user
    
    # Add smart manual update prices button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        # Check how many holdings need updating
        holdings = get_cached_holdings(user['id'])
        holdings_needing_update = should_update_prices_today(holdings) if holdings else []
        
        if holdings_needing_update:
            button_text = f"🔄 Update {len(holdings_needing_update)}"
            help_text = f"Update prices for {len(holdings_needing_update)} holdings that haven't been updated today"
        else:
            button_text = "✅ All Current"
            help_text = "All prices are up-to-date for today"
        
        if st.button(button_text, help=help_text, disabled=(len(holdings_needing_update) == 0)):
            with st.spinner(f"Updating {len(holdings_needing_update)} holdings..."):
                if holdings_needing_update:
                    st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
                    st.success(f"✅ Updated {len(holdings_needing_update)} holdings!")
                    st.rerun()
                else:
                    st.info("All prices are already up-to-date!")
    
    # Use cached holdings data
    holdings = get_cached_holdings(user['id'])
    
    # Get cached metrics
    metrics = get_portfolio_metrics(holdings)
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see your portfolio.")
        return
    
    st.success(f"📊 Loaded {len(holdings)} holdings")
    
    # Calculate portfolio metrics (clean, no logs)
    total_investment = 0
    total_current = 0
    total_pnl = 0
    
    for holding in holdings:
        # Calculate investment value
        investment_value = float(holding['total_quantity']) * float(holding['average_price'])
        total_investment += investment_value
        
        # Get current price - handle None values
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        
        if current_price and current_price != holding['average_price']:
            # Price was fetched successfully
            current_value = float(holding['total_quantity']) * float(current_price)
            total_current += current_value
            
            pnl = current_value - investment_value
            total_pnl += pnl
        else:
            # Using average price as fallback
            current_value = investment_value
            total_current += current_value
            total_pnl += 0  # No P&L if using average price
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.0f}")
    
    with col2:
        st.metric("Current Value", f"₹{total_current:,.0f}")
    
    with col3:
        st.metric("Total P&L", f"₹{total_pnl:,.0f}")
    
    with col4:
        pnl_percent = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        st.metric("P&L %", f"{pnl_percent:+.1f}%")
    
    # Holdings table
    st.subheader("📊 Your Holdings")
    
    holdings_data = []
    for holding in holdings:
        # Handle None current_price
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = float(holding['total_quantity']) * float(current_price) if current_price else 0
        investment_value = float(holding['total_quantity']) * float(holding['average_price'])
        pnl = current_value - investment_value
        pnl_percent = (pnl / investment_value * 100) if investment_value > 0 else 0
        
        # Get rating for this stock
        stars, grade, rating = get_performance_rating(pnl_percent)
        
        holdings_data.append({
            'Ticker': holding['ticker'],
            'Name': holding['stock_name'],
            'Rating': stars,
            'Grade': grade,
            'Type': holding['asset_type'],
            'Quantity': f"{holding['total_quantity']:,.0f}",
            'Avg Price': f"₹{holding['average_price']:,.2f}",
            'Current Price': f"₹{current_price:,.2f}" if current_price else "N/A",
            'Current Value': f"₹{current_value:,.0f}",
            'P&L': f"₹{pnl:,.0f}",
            'P&L %': f"{pnl_percent:+.1f}%",
            'Performance': rating
        })
    
    df_holdings = pd.DataFrame(holdings_data)
    st.dataframe(df_holdings, use_container_width=True)
    
    # Market Sentiment Section
    st.markdown("---")
    st.subheader("📈 Market Sentiment & Trends")
    
    # Market Sentiment Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Market Breadth (using cached metrics)
        market_breadth = metrics.get('market_breadth', 0)
        gainers = metrics.get('gainers', 0)
        total_holdings = metrics.get('total_holdings', 0)
        
        st.metric(
            "Market Breadth",
            f"{market_breadth:.1f}%",
            delta=f"{gainers}/{total_holdings}",
            help="Percentage of holdings in positive territory"
        )
        
        if market_breadth > 70:
            st.success("Strong Bullish Sentiment")
        elif market_breadth < 30:
            st.error("Bearish Sentiment")
        else:
            st.info("Mixed Sentiment")
    
    with col2:
        # Average P&L (using cached metrics)
        total_pnl_pct = metrics.get('total_pnl_pct', 0)
        
        st.metric(
            "Avg Portfolio Return",
            f"{total_pnl_pct:.2f}%",
            help="Average return across all holdings"
        )
        
        if total_pnl_pct > 5:
            st.success("Strong Performance")
        elif total_pnl_pct < -5:
            st.error("Underperforming")
        else:
            st.info("Moderate Performance")
    
    with col3:
        # Volatility Index (using cached metrics)
        volatility = metrics.get('volatility', 0)
        
        st.metric(
            "Portfolio Volatility",
            f"{volatility:.2f}%",
            help="Standard deviation of returns"
        )
        
        if volatility > 20:
            st.error("High Volatility")
        elif volatility < 10:
            st.success("Low Volatility")
        else:
            st.warning("Moderate Volatility")
    
    with col4:
        # Momentum Score (using cached metrics)
        momentum_score = metrics.get('momentum_score', 0)
        momentum_stocks = metrics.get('momentum_stocks', 0)
        
        st.metric(
            "Momentum Score",
            f"{momentum_score:.1f}%",
            delta=f"{momentum_stocks} stocks >10%",
            help="Percentage of holdings with strong momentum"
        )
        
        if momentum_score > 50:
            st.success("Strong Momentum")
        elif momentum_score < 20:
            st.error("Weak Momentum")
        else:
            st.info("Moderate Momentum")
    
    # Market Sentiment Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector Performance Heatmap
        sector_data = {}
        for holding in holdings:
            sector = holding.get('sector', holding.get('asset_type', 'Unknown'))
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            
            pnl_pct = ((current_price - holding.get('average_price', 0)) / holding.get('average_price', 1) * 100) if holding.get('average_price', 0) > 0 else 0
            
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(pnl_pct)
        
        # Calculate average sector performance
        sector_performance = {sector: np.mean(returns) for sector, returns in sector_data.items()}
        
        if sector_performance:
            fig_sector_heatmap = go.Figure(data=go.Heatmap(
                z=[[list(sector_performance.values())[i]] for i in range(len(sector_performance))],
                x=['Performance'],
                y=list(sector_performance.keys()),
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return %")
            ))
            
            fig_sector_heatmap.update_layout(
                title="Sector Performance Heatmap",
                height=300
            )
            
            st.plotly_chart(fig_sector_heatmap, use_container_width=True)
    
    with col2:
        # Market Cap Distribution
        market_cap_data = []
        for holding in holdings:
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            
            market_value = float(current_price) * float(holding['total_quantity'])
            
            # Categorize by market cap (simplified)
            if market_value > 1000000:  # > 10L
                cap_category = "Large Cap"
            elif market_value > 100000:  # > 1L
                cap_category = "Mid Cap"
            else:
                cap_category = "Small Cap"
            
            market_cap_data.append({
                'ticker': holding['ticker'],
                'market_value': market_value,
                'cap_category': cap_category
            })
        
        if market_cap_data:
            df_market_cap = pd.DataFrame(market_cap_data)
            cap_distribution = df_market_cap.groupby('cap_category')['market_value'].sum()
            
            fig_market_cap = px.pie(
                values=cap_distribution.values,
                names=cap_distribution.index,
                title="Market Cap Distribution",
                hole=0.4
            )
            fig_market_cap.update_layout(height=300)
            st.plotly_chart(fig_market_cap, use_container_width=True)
    
    # Advanced Charts Section
    st.markdown("---")
    st.subheader("📊 Advanced Portfolio Analytics")
    
    # Top Performers and Underperformers
    col1, col2 = st.columns(2)
    
    # Prepare performance data
    perf_data = []
    for holding in holdings:
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = float(holding['total_quantity']) * float(current_price) if current_price else 0
        investment_value = float(holding['total_quantity']) * float(holding['average_price'])
        pnl = current_value - investment_value
        pnl_percent = (pnl / investment_value * 100) if investment_value > 0 else 0
        
        perf_data.append({
            'ticker': holding['ticker'],
            'stock_name': holding['stock_name'],
            'invested_amount': investment_value,
            'unrealized_pnl': pnl,
            'pnl_percentage': pnl_percent
        })
    
    df_perf = pd.DataFrame(perf_data)
    
    with col1:
        st.markdown("### 📈 Top 5 Performers")
        top_performers = df_perf.nlargest(5, 'pnl_percentage')
        
        if not top_performers.empty:
            top_performers = top_performers.sort_values('pnl_percentage', ascending=True)
            top_performers['display_label'] = top_performers['ticker'] + ' - ' + top_performers['stock_name'].str[:20]
            
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                x=top_performers['pnl_percentage'],
                y=top_performers['display_label'],
                orientation='h',
                marker=dict(
                    color=top_performers['pnl_percentage'],
                    colorscale='Greens',
                    showscale=False
                ),
                text=top_performers['pnl_percentage'].round(2),
                texttemplate='%{text:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<br>P&L: ₹%{customdata:,.0f}<extra></extra>',
                customdata=top_performers['unrealized_pnl']
            ))
            
            fig_top.update_layout(
                title="",
                xaxis_title="Return %",
                yaxis_title="",
                height=300,
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No gainers yet")
    
    with col2:
        st.markdown("### 📉 Bottom 5 Performers")
        underperformers = df_perf.nsmallest(5, 'pnl_percentage')
        
        if not underperformers.empty:
            underperformers = underperformers.sort_values('pnl_percentage', ascending=False)
            underperformers['display_label'] = underperformers['ticker'] + ' - ' + underperformers['stock_name'].str[:20]
            
            fig_bottom = go.Figure()
            fig_bottom.add_trace(go.Bar(
                x=underperformers['pnl_percentage'],
                y=underperformers['display_label'],
                orientation='h',
                marker=dict(
                    color=underperformers['pnl_percentage'],
                    colorscale='Reds',
                    showscale=False,
                    reversescale=True
                ),
                text=underperformers['pnl_percentage'].round(2),
                texttemplate='%{text:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<br>P&L: ₹%{customdata:,.0f}<extra></extra>',
                customdata=underperformers['unrealized_pnl']
            ))
            
            fig_bottom.update_layout(
                title="",
                xaxis_title="Return %",
                yaxis_title="",
                height=300,
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No underperformers - Great portfolio!")
    
    # Investment Distribution Treemap
    st.markdown("---")
    st.subheader("🗺️ Portfolio Allocation Treemap")
    
    # Create treemap data
    treemap_data = []
    for holding in holdings:
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = float(holding['total_quantity']) * float(current_price) if current_price else 0
        
        treemap_data.append({
            'labels': f"{holding['ticker']}<br>{holding['stock_name'][:20]}",
            'parents': holding.get('asset_type', 'Unknown'),
            'values': current_value,
            'text': f"₹{current_value:,.0f}"
        })
    
    # Add parent categories
    asset_types = set([holding.get('asset_type', 'Unknown') for holding in holdings])
    for asset_type in asset_types:
        treemap_data.append({
            'labels': asset_type,
            'parents': '',
            'values': 0,
            'text': asset_type
        })
    
    df_treemap = pd.DataFrame(treemap_data)
    
    fig_treemap = go.Figure(go.Treemap(
        labels=df_treemap['labels'],
        parents=df_treemap['parents'],
        values=df_treemap['values'],
        text=df_treemap['text'],
        textposition='middle center',
        marker=dict(
            colorscale='Viridis',
            cmid=df_treemap['values'].mean()
        )
    ))
    
    fig_treemap.update_layout(
        title="Portfolio Allocation by Asset Type and Holdings",
        height=500
    )
    
    st.plotly_chart(fig_treemap, use_container_width=True)

def get_performance_rating(pnl_percent):
    """Get star rating and grade based on P&L percentage"""
    if pnl_percent >= 50:
        return "⭐⭐⭐⭐⭐", "A+", "Excellent"
    elif pnl_percent >= 30:
        return "⭐⭐⭐⭐", "A", "Very Good"
    elif pnl_percent >= 15:
        return "⭐⭐⭐", "B+", "Good"
    elif pnl_percent >= 5:
        return "⭐⭐", "B", "Average"
    elif pnl_percent >= 0:
        return "⭐", "C", "Below Average"
    elif pnl_percent >= -10:
        return "❌", "D", "Poor"
    else:
        return "❌❌", "F", "Very Poor"

def get_risk_score(volatility):
    """Get risk rating based on volatility"""
    if volatility < 10:
        return "🟢 Low Risk", "Conservative"
    elif volatility < 20:
        return "🟡 Moderate Risk", "Balanced"
    elif volatility < 30:
        return "🟠 High Risk", "Aggressive"
    else:
        return "🔴 Very High Risk", "Speculative"

def pnl_analysis_page():
    """P&L Analysis based on current week price (as per your image)"""
    st.header("📊 P&L Analysis")
    st.caption("Calculated based on current week prices")
    
    user = st.session_state.user
    holdings = db.get_user_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found.")
        return
    
    # Group by sector and channel
    sector_data = {}
    channel_data = {}
    
    for holding in holdings:
        sector = holding.get('sector', 'Unknown')
        channel = holding.get('channel', 'Direct')
        
        # Handle None current_price
        current_price = holding.get('current_price')
        if current_price is None or current_price == 0:
            current_price = holding.get('average_price', 0)
        current_value = float(holding['total_quantity']) * float(current_price) if current_price else 0
        investment_value = float(holding['total_quantity']) * float(holding['average_price'])
        pnl = current_value - investment_value
        
        # Sector analysis
        if sector not in sector_data:
            sector_data[sector] = {'investment': 0, 'current': 0, 'pnl': 0}
        sector_data[sector]['investment'] += investment_value
        sector_data[sector]['current'] += current_value
        sector_data[sector]['pnl'] += pnl
        
        # Channel analysis
        if channel not in channel_data:
            channel_data[channel] = {'investment': 0, 'current': 0, 'pnl': 0}
        channel_data[channel]['investment'] += investment_value
        channel_data[channel]['current'] += current_value
        channel_data[channel]['pnl'] += pnl
    
    # Sector analysis
    st.subheader("📊 Sector Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector pie chart
        sectors = list(sector_data.keys())
        values = [sector_data[s]['current'] for s in sectors]
        
        fig = px.pie(values=values, names=sectors, title="Portfolio by Sector")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector table with ratings
        sector_table = []
        for sector, data in sector_data.items():
            pnl_percent = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
            stars, grade, rating = get_performance_rating(pnl_percent)
            
            sector_table.append({
                'Sector': sector,
                'Rating': stars,
                'Grade': grade,
                'Investment': f"₹{data['investment']:,.0f}",
                'Current': f"₹{data['current']:,.0f}",
                'P&L': f"₹{data['pnl']:,.0f}",
                'P&L %': f"{pnl_percent:+.1f}%",
                'Performance': rating
            })
        
        df_sectors = pd.DataFrame(sector_table)
        st.dataframe(df_sectors, use_container_width=True)
    
    # Channel analysis
    st.subheader("📊 Channel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel pie chart
        channels = list(channel_data.keys())
        values = [channel_data[c]['current'] for c in channels]
        
        fig = px.pie(values=values, names=channels, title="Portfolio by Channel")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Channel table with ratings
        channel_table = []
        for channel, data in channel_data.items():
            pnl_percent = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
            stars, grade, rating = get_performance_rating(pnl_percent)
            
            channel_table.append({
                'Channel': channel,
                'Rating': stars,
                'Grade': grade,
                'Investment': f"₹{data['investment']:,.0f}",
                'Current': f"₹{data['current']:,.0f}",
                'P&L': f"₹{data['pnl']:,.0f}",
                'P&L %': f"{pnl_percent:+.1f}%",
                'Performance': rating
            })
        
        df_channels = pd.DataFrame(channel_table)
        st.dataframe(df_channels, use_container_width=True)

def charts_page():
    """Comprehensive charts and analytics page"""
    st.header("📈 Charts & Analytics")
    
    user = st.session_state.user
    
    # Use cached holdings data (same as portfolio overview)
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see charts.")
        return
    
    # Add a small loading indicator for better UX
    with st.spinner("📊 Loading charts..."):
        pass  # This creates a brief loading state
    
    # Add info about page behavior
    with st.expander("ℹ️ About Page Refreshing", expanded=False):
        st.info("""
        **Why does the page refresh when changing comparisons?**
        
        This is normal Streamlit behavior - when you change dropdown selections, the entire page reruns to update the charts with new data. 
        
        **Tips for smoother experience:**
        - ✅ Your selections are preserved between refreshes
        - ✅ Data is cached for faster loading
        - ✅ Only the comparison section refreshes, not the entire app
        
        **What's optimized:**
        - Session state maintains your selections
        - Cached data reduces loading time
        - Smart defaults for common selections
        """)
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Portfolio Allocation", 
        "💰 Performance", 
        "📅 52-Week NAVs",
        "🔍 Advanced Analytics",
        "📈 Technical Analysis",
        "⚡ Risk Metrics",
        "🔄 Compare Holdings"
    ])
    
    with tab1:
        st.subheader("📊 Portfolio Allocation")
        
        # Asset Type Distribution
        asset_types = {}
        for holding in holdings:
            asset_type = holding.get('asset_type', 'Unknown')
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding.get('total_quantity', 0))
            asset_types[asset_type] = asset_types.get(asset_type, 0) + current_value
        
        if asset_types:
            fig_pie = px.pie(
                values=list(asset_types.values()),
                names=list(asset_types.keys()),
                title="Asset Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top Holdings
        holdings_data = []
        for holding in holdings:
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding.get('total_quantity', 0))
            holdings_data.append({
                'Ticker': holding['ticker'],
                'Name': holding['stock_name'],
                'Current Value': current_value,
                'Quantity': holding['total_quantity'],
                'Avg Price': holding['average_price']
            })
        
        if holdings_data:
            df_holdings = pd.DataFrame(holdings_data)
            df_holdings = df_holdings.sort_values('Current Value', ascending=False).head(10)
            
            fig_bar = px.bar(
                df_holdings, 
                x='Ticker', 
                y='Current Value',
                title="Top 10 Holdings by Value"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Performance Analysis")
        
        # Calculate P&L for each holding
        performance_data = []
        for holding in holdings:
            investment = float(holding['total_quantity']) * float(holding['average_price'])
            # Handle None current_price
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding['total_quantity'])
            pnl = current_value - investment
            pnl_pct = (pnl / investment * 100) if investment > 0 else 0
            
            performance_data.append({
                'Ticker': holding['ticker'],
                'Name': holding['stock_name'],
                'Investment': investment,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_pct,
                'Asset Type': holding.get('asset_type', 'Unknown'),
                'Channel': holding.get('channel', 'Direct')
            })
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            
            # Top Gainers and Losers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Top 10 Gainers")
                gainers = df_performance[df_performance['P&L'] > 0].nlargest(10, 'P&L %')
                if not gainers.empty:
                    for idx, row in gainers.iterrows():
                        with st.container():
                            st.markdown(f"**{row['Name']}** ({row['Ticker']})")
                            st.markdown(f"💰 P&L: ₹{row['P&L']:,.0f} | 📊 {row['P&L %']:.2f}%")
                            st.progress(min(row['P&L %'] / 100, 1.0))
                            st.markdown("---")
                else:
                    st.info("No gainers yet")
            
            with col2:
                st.subheader("📉 Top 10 Losers")
                losers = df_performance[df_performance['P&L'] < 0].nsmallest(10, 'P&L %')
                if not losers.empty:
                    for idx, row in losers.iterrows():
                        with st.container():
                            st.markdown(f"**{row['Name']}** ({row['Ticker']})")
                            st.markdown(f"💸 Loss: ₹{row['P&L']:,.0f} | 📊 {row['P&L %']:.2f}%")
                            st.progress(min(abs(row['P&L %']) / 100, 1.0))
                            st.markdown("---")
                else:
                    st.info("No losers - Excellent performance!")
            
            st.markdown("---")
            
            # Performance by Asset Type and Channel
            st.subheader("📊 Performance Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏢 By Asset Type")
                asset_perf = df_performance.groupby('Asset Type').agg({
                    'Investment': 'sum',
                    'Current Value': 'sum',
                    'P&L': 'sum'
                }).reset_index()
                asset_perf['P&L %'] = (asset_perf['P&L'] / asset_perf['Investment'] * 100)
                asset_perf = asset_perf.sort_values('P&L %', ascending=False)
                
                fig_asset_bar = go.Figure()
                fig_asset_bar.add_trace(go.Bar(
                    x=asset_perf['Asset Type'],
                    y=asset_perf['P&L %'],
                    marker_color=['green' if x >= 0 else 'red' for x in asset_perf['P&L %']],
                    text=asset_perf['P&L %'].round(2),
                    textposition='auto'
                ))
                fig_asset_bar.update_layout(
                    title="Asset Type Performance %",
                    xaxis_title="Asset Type",
                    yaxis_title="P&L %",
                    height=400
                )
                st.plotly_chart(fig_asset_bar, use_container_width=True)
            
            with col2:
                st.markdown("### 📡 By Channel")
                channel_perf = df_performance.groupby('Channel').agg({
                    'Investment': 'sum',
                    'Current Value': 'sum',
                    'P&L': 'sum'
                }).reset_index()
                channel_perf['P&L %'] = (channel_perf['P&L'] / channel_perf['Investment'] * 100)
                channel_perf = channel_perf.sort_values('P&L %', ascending=False)
                
                fig_channel_bar = go.Figure()
                fig_channel_bar.add_trace(go.Bar(
                    x=channel_perf['Channel'],
                    y=channel_perf['P&L %'],
                    marker_color=['green' if x >= 0 else 'red' for x in channel_perf['P&L %']],
                    text=channel_perf['P&L %'].round(2),
                    textposition='auto'
                ))
                fig_channel_bar.update_layout(
                    title="Channel Performance %",
                    xaxis_title="Channel",
                    yaxis_title="P&L %",
                    height=400
                )
                st.plotly_chart(fig_channel_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Portfolio Statistics
            st.subheader("📊 Portfolio Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_investment = df_performance['Investment'].sum()
                st.metric("Total Investment", f"₹{total_investment:,.0f}")
            
            with col2:
                total_current = df_performance['Current Value'].sum()
                st.metric("Current Value", f"₹{total_current:,.0f}")
            
            with col3:
                total_pnl = df_performance['P&L'].sum()
                st.metric("Total P&L", f"₹{total_pnl:,.0f}")
            
            with col4:
                total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                st.metric("Total P&L %", f"{total_pnl_pct:+.1f}%")
            
            # Performance Table
            st.subheader("📋 Detailed Performance")
            df_display = df_performance.copy()
            df_display['Investment'] = df_display['Investment'].apply(lambda x: f"₹{x:,.0f}")
            df_display['Current Value'] = df_display['Current Value'].apply(lambda x: f"₹{x:,.0f}")
            df_display['P&L'] = df_display['P&L'].apply(lambda x: f"₹{x:,.0f}")
            df_display['P&L %'] = df_display['P&L %'].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(df_display, use_container_width=True)
    
    with tab3:
        st.subheader("📅 52-Week NAVs")
        
        # Get ticker options
        ticker_options = [h['ticker'] for h in holdings]
        
        # Select ticker for NAV view
        selected_ticker_nav = st.selectbox("Select Ticker for NAV History", ticker_options, key="nav_ticker")
        
        if selected_ticker_nav:
            stock_id = next(h['stock_id'] for h in holdings if h['ticker'] == selected_ticker_nav)
            prices = db.get_historical_prices_for_stock_silent(stock_id)
            
            if prices:
                df_navs = pd.DataFrame(prices)
                df_navs['date'] = pd.to_datetime(df_navs['price_date'])
                df_navs = df_navs.sort_values('date')
                
                # NAV Chart
                fig_nav = px.line(
                    df_navs, 
                    x='date', 
                    y='price',
                    title=f"{selected_ticker_nav} - 52-Week NAVs"
                )
                fig_nav.update_layout(xaxis_title="Date", yaxis_title="NAV (₹)")
                st.plotly_chart(fig_nav, use_container_width=True)
                
                # NAV Table
                st.subheader(f"{selected_ticker_nav} - NAV History")
                df_display = df_navs[['date', 'price', 'iso_week', 'iso_year']].copy()
                df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
                df_display.columns = ['Date', 'NAV', 'Week', 'Year']
                st.dataframe(df_display.tail(20), use_container_width=True)  # Show last 20 entries
            else:
                st.info(f"No NAV data available for {selected_ticker_nav}")
    
    with tab4:
        st.subheader("🔍 Advanced Analytics")
        
        # Prepare comprehensive analytics data
        analytics_data = []
        for holding in holdings:
            current_price = holding.get('current_price')
            if current_price is None or current_price == 0:
                current_price = holding.get('average_price', 0)
            current_value = float(current_price) * float(holding['total_quantity'])
            investment = float(holding['total_quantity']) * float(holding['average_price'])
            pnl = current_value - investment
            pnl_pct = (pnl / investment * 100) if investment > 0 else 0
            
            analytics_data.append({
                'ticker': holding['ticker'],
                'stock_name': holding['stock_name'],
                'asset_type': holding.get('asset_type', 'Unknown'),
                'channel': holding.get('channel', 'Direct'),
                'investment': investment,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'quantity': float(holding['total_quantity']),
                'avg_price': float(holding['average_price']),
                'current_price': float(current_price) if current_price else 0
            })
        
        df_analytics = pd.DataFrame(analytics_data)
        
        # Create sub-tabs for different analytics
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "📊 Risk Metrics",
            "💰 Value Analysis", 
            "📈 Performance Distribution",
            "🎯 Portfolio Insights"
        ])
        
        with analytics_tab1:
            st.markdown("### 📊 Risk & Return Metrics")
            
            # Calculate portfolio metrics
            total_investment = df_analytics['investment'].sum()
            total_current = df_analytics['current_value'].sum()
            portfolio_return = (total_current - total_investment) / total_investment * 100 if total_investment > 0 else 0
            individual_returns = df_analytics['pnl_pct'].tolist()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Return", f"{portfolio_return:+.2f}%")
            
            with col2:
                if individual_returns:
                    volatility = np.std(individual_returns)
                    risk_label, risk_strategy = get_risk_score(volatility)
                    st.metric("Volatility", f"{volatility:.2f}%", delta=risk_label)
                else:
                    st.metric("Volatility", "N/A")
            
            with col3:
                if individual_returns and volatility > 0:
                    sharpe_ratio = portfolio_return / volatility
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            
            with col4:
                max_drawdown = min(individual_returns) if individual_returns else 0
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            st.markdown("---")
            
            # Additional risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Win rate
                winners = len(df_analytics[df_analytics['pnl'] > 0])
                total_holdings = len(df_analytics)
                win_rate = (winners / total_holdings * 100) if total_holdings > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{winners}/{total_holdings} holdings")
            
            with col2:
                # Average gain/loss
                avg_gain = df_analytics[df_analytics['pnl'] > 0]['pnl_pct'].mean() if winners > 0 else 0
                st.metric("Avg Gain", f"{avg_gain:+.2f}%")
            
            with col3:
                losers = len(df_analytics[df_analytics['pnl'] < 0])
                avg_loss = df_analytics[df_analytics['pnl'] < 0]['pnl_pct'].mean() if losers > 0 else 0
                st.metric("Avg Loss", f"{avg_loss:.2f}%")
            
            # Risk-Return Scatter Plot
            st.markdown("#### 🎯 Risk-Return Profile")
            
            fig_risk_return = px.scatter(
                df_analytics,
                x='investment',
                y='pnl_pct',
                size='current_value',
                color='asset_type',
                hover_data=['ticker', 'stock_name'],
                title="Risk-Return Analysis (Size = Current Value)",
                labels={'investment': 'Investment Amount (₹)', 'pnl_pct': 'Return (%)'}
            )
            
            # Add quadrant lines
            fig_risk_return.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_risk_return.add_vline(x=df_analytics['investment'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_risk_return, use_container_width=True)
        
        with analytics_tab2:
            st.markdown("### 💰 Value & Allocation Analysis")
            
            # Portfolio composition by value
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Top 10 Holdings by Value")
                top_10_value = df_analytics.nlargest(10, 'current_value')
                
                fig_top_value = px.bar(
                    top_10_value,
                    x='current_value',
                    y='ticker',
                    orientation='h',
                    title="Top 10 Holdings by Current Value",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={'current_value': 'Current Value (₹)', 'ticker': 'Ticker'}
                )
                st.plotly_chart(fig_top_value, use_container_width=True)
            
            with col2:
                st.markdown("#### 💸 Top 10 Holdings by Investment")
                top_10_inv = df_analytics.nlargest(10, 'investment')
                
                fig_top_inv = px.bar(
                    top_10_inv,
                    x='investment',
                    y='ticker',
                    orientation='h',
                    title="Top 10 Holdings by Investment",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={'investment': 'Investment (₹)', 'ticker': 'Ticker'}
                )
                st.plotly_chart(fig_top_inv, use_container_width=True)
            
            # Concentration metrics
            st.markdown("#### 🎯 Concentration Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Top holding concentration
                top_holding_pct = (df_analytics['current_value'].max() / total_current * 100) if total_current > 0 else 0
                st.metric("Top Holding", f"{top_holding_pct:.1f}%")
            
            with col2:
                # Top 5 concentration
                top_5_value = df_analytics.nlargest(5, 'current_value')['current_value'].sum()
                top_5_pct = (top_5_value / total_current * 100) if total_current > 0 else 0
                st.metric("Top 5 Holdings", f"{top_5_pct:.1f}%")
            
            with col3:
                # Top 10 concentration
                top_10_value_sum = df_analytics.nlargest(10, 'current_value')['current_value'].sum()
                top_10_pct = (top_10_value_sum / total_current * 100) if total_current > 0 else 0
                st.metric("Top 10 Holdings", f"{top_10_pct:.1f}%")
            
            with col4:
                # Number of holdings
                st.metric("Total Holdings", len(df_analytics))
            
            # Herfindahl Index (concentration measure)
            holdings_shares = (df_analytics['current_value'] / total_current * 100) ** 2
            herfindahl_index = holdings_shares.sum()
            
            st.markdown(f"**Herfindahl Index:** {herfindahl_index:.2f}")
            if herfindahl_index > 2500:
                st.warning("⚠️ Highly concentrated portfolio (HHI > 2500)")
            elif herfindahl_index > 1500:
                st.info("ℹ️ Moderately concentrated portfolio (HHI 1500-2500)")
            else:
                st.success("✅ Well diversified portfolio (HHI < 1500)")
        
        with analytics_tab3:
            st.markdown("### 📈 Performance Distribution Analysis")
            
            # Performance histogram
            fig_hist = px.histogram(
                df_analytics,
                x='pnl_pct',
                nbins=20,
                title="Distribution of Returns Across Holdings",
                labels={'pnl_pct': 'Return (%)', 'count': 'Number of Holdings'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig_hist.add_vline(x=df_analytics['pnl_pct'].median(), line_dash="dash", line_color="green", annotation_text="Median")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Performance quartiles
            st.markdown("#### 📊 Performance Quartiles")
            
            q1 = df_analytics['pnl_pct'].quantile(0.25)
            q2 = df_analytics['pnl_pct'].quantile(0.50)  # Median
            q3 = df_analytics['pnl_pct'].quantile(0.75)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Q1 (25th %ile)", f"{q1:+.2f}%")
            with col2:
                st.metric("Q2 (Median)", f"{q2:+.2f}%")
            with col3:
                st.metric("Q3 (75th %ile)", f"{q3:+.2f}%")
            with col4:
                iqr = q3 - q1
                st.metric("IQR", f"{iqr:.2f}%")
            
            # Box plot by asset type
            st.markdown("#### 📦 Performance by Asset Type (Box Plot)")
            
            fig_box = px.box(
                df_analytics,
                x='asset_type',
                y='pnl_pct',
                title="Return Distribution by Asset Type",
                labels={'asset_type': 'Asset Type', 'pnl_pct': 'Return (%)'},
                color='asset_type'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### 📊 Statistical Summary")
            
            summary_stats = df_analytics['pnl_pct'].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{summary_stats['mean']:+.2f}%",
                        f"{summary_stats['std']:.2f}%",
                        f"{summary_stats['min']:+.2f}%",
                        f"{summary_stats['max']:+.2f}%"
                    ]
                }), use_container_width=True, hide_index=True)
            
            with col2:
                # Skewness and kurtosis
                from scipy import stats
                skewness = stats.skew(df_analytics['pnl_pct'])
                kurtosis = stats.kurtosis(df_analytics['pnl_pct'])
                
                st.dataframe(pd.DataFrame({
                    'Metric': ['Skewness', 'Kurtosis', 'Range', 'CV'],
                    'Value': [
                        f"{skewness:.2f}",
                        f"{kurtosis:.2f}",
                        f"{summary_stats['max'] - summary_stats['min']:.2f}%",
                        f"{(summary_stats['std'] / abs(summary_stats['mean']) * 100):.2f}%" if summary_stats['mean'] != 0 else "N/A"
                    ]
                }), use_container_width=True, hide_index=True)
        
        with analytics_tab4:
            st.markdown("### 🎯 Portfolio Insights & Recommendations")
            
            # Key insights
            st.markdown("#### 💡 Key Insights")
            
            # 1. Best and worst performers
            best_performer = df_analytics.nlargest(1, 'pnl_pct').iloc[0]
            worst_performer = df_analytics.nsmallest(1, 'pnl_pct').iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🏆 Best Performer:**
                - {best_performer['ticker']} - {best_performer['stock_name']}
                - Return: {best_performer['pnl_pct']:+.2f}%
                - P&L: ₹{best_performer['pnl']:,.0f}
                """)
            
            with col2:
                st.error(f"""
                **📉 Worst Performer:**
                - {worst_performer['ticker']} - {worst_performer['stock_name']}
                - Return: {worst_performer['pnl_pct']:+.2f}%
                - P&L: ₹{worst_performer['pnl']:,.0f}
                """)
            
            # 2. Asset allocation insights
            st.markdown("#### 📊 Asset Allocation Insights")
            
            asset_allocation = df_analytics.groupby('asset_type').agg({
                'current_value': 'sum',
                'pnl_pct': 'mean'
            }).reset_index()
            asset_allocation['allocation_pct'] = (asset_allocation['current_value'] / total_current * 100)
            
            for _, row in asset_allocation.iterrows():
                asset_type = row['asset_type']
                allocation = row['allocation_pct']
                avg_return = row['pnl_pct']
                
                if allocation > 50:
                    st.warning(f"⚠️ {asset_type.upper()}: {allocation:.1f}% allocation (High concentration)")
                elif allocation > 30:
                    st.info(f"ℹ️ {asset_type.upper()}: {allocation:.1f}% allocation | Avg Return: {avg_return:+.1f}%")
                else:
                    st.success(f"✅ {asset_type.upper()}: {allocation:.1f}% allocation | Avg Return: {avg_return:+.1f}%")
            
            # 3. Channel performance insights
            st.markdown("#### 📡 Channel Performance Insights")
            
            channel_performance = df_analytics.groupby('channel').agg({
                'current_value': 'sum',
                'pnl': 'sum',
                'pnl_pct': 'mean'
            }).reset_index()
            channel_performance = channel_performance.sort_values('pnl_pct', ascending=False)
            
            best_channel = channel_performance.iloc[0]
            st.info(f"""
            **🏆 Best Performing Channel:** {best_channel['channel']}
            - Average Return: {best_channel['pnl_pct']:+.2f}%
            - Total P&L: ₹{best_channel['pnl']:,.0f}
            - Portfolio Value: ₹{best_channel['current_value']:,.0f}
            """)
            
            # 4. Recommendations
            st.markdown("#### 🎯 Recommendations")
            
            recommendations = []
            
            # Check for underperformers
            underperformers = df_analytics[df_analytics['pnl_pct'] < -10]
            if len(underperformers) > 0:
                recommendations.append(f"⚠️ Review {len(underperformers)} holdings with losses > 10%")
            
            # Check for concentration
            if top_holding_pct > 25:
                recommendations.append(f"⚠️ Top holding represents {top_holding_pct:.1f}% of portfolio - consider diversification")
            
            # Check for winners
            big_winners = df_analytics[df_analytics['pnl_pct'] > 50]
            if len(big_winners) > 0:
                recommendations.append(f"✅ {len(big_winners)} holdings with returns > 50% - consider profit booking")
            
            # Check volatility
            if volatility > 30:
                recommendations.append(f"⚠️ High portfolio volatility ({volatility:.1f}%) - consider adding stable assets")
            
            # Check win rate
            if win_rate < 50:
                recommendations.append(f"⚠️ Win rate below 50% ({win_rate:.1f}%) - review investment strategy")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ Portfolio is well balanced with no immediate concerns!")
    
    with tab5:
        st.subheader("📈 Technical Analysis")
        st.caption("Professional-grade technical indicators and analysis")
        
        # Technical Analysis Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_ticker_tech = st.selectbox(
                "Select Stock for Technical Analysis:",
                [h['ticker'] for h in holdings],
                key="tech_ticker_select"
            )
        
        with col2:
            time_period = st.selectbox(
                "Time Period:",
                ["1M", "3M", "6M", "1Y", "2Y"],
                index=3,
                key="tech_time_period"
            )
        
        with col3:
            show_indicators = st.multiselect(
                "Technical Indicators:",
                ["RSI", "MACD", "Moving Averages", "Bollinger Bands", "Volume"],
                default=["RSI", "MACD", "Moving Averages"],
                key="tech_indicators"
            )
        
        # Only load technical analysis if user has made selections
        if selected_ticker_tech and show_indicators:
            st.markdown("---")
            
            # Get historical data for technical analysis
            try:
                import yfinance as yf
                
                # Convert time period to days
                period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
                period = period_map.get(time_period, "1y")
                
                # Try multiple ticker formats
                hist = pd.DataFrame()
                ticker_formats = [
                    selected_ticker_tech,  # Original (e.g., IDFCFIRSTB.NS)
                    selected_ticker_tech.replace('.NS', '').replace('.BO', ''),  # Without suffix
                    selected_ticker_tech + '.NS' if '.NS' not in selected_ticker_tech and '.BO' not in selected_ticker_tech else selected_ticker_tech,  # Add .NS
                    selected_ticker_tech.replace('.NS', '.BO') if '.NS' in selected_ticker_tech else selected_ticker_tech.replace('.BO', '.NS')  # Switch exchange
                ]
                
                for ticker_format in ticker_formats:
                    try:
                        stock = yf.Ticker(ticker_format)
                        hist = stock.history(period=period)
                        if not hist.empty and len(hist) > 20:  # Need at least 20 days for indicators
                            st.caption(f"✅ Data fetched using ticker: {ticker_format}")
                            break
                    except:
                        continue
                
                if not hist.empty and len(hist) > 20:
                    # Price Chart with Technical Indicators
                    fig_tech = go.Figure()
                    
                    # Add candlestick chart
                    fig_tech.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name="Price",
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    ))
                    
                    # Add technical indicators
                    if "Moving Averages" in show_indicators:
                        # 20-day MA
                        hist['MA20'] = hist['Close'].rolling(window=20).mean()
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['MA20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='orange', width=2)
                        ))
                        
                        # 50-day MA
                        hist['MA50'] = hist['Close'].rolling(window=50).mean()
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['MA50'],
                            mode='lines',
                            name='MA 50',
                            line=dict(color='blue', width=2)
                        ))
                    
                    if "Bollinger Bands" in show_indicators:
                        # Bollinger Bands
                        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
                        bb_std = hist['Close'].rolling(window=20).std()
                        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
                        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
                        
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')
                        ))
                        fig_tech.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty'
                        ))
                    
                    fig_tech.update_layout(
                        title=f"{selected_ticker_tech} - Technical Analysis ({time_period})",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_tech, use_container_width=True)
                    
                    # Technical Indicators Panel
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if "RSI" in show_indicators:
                            # Calculate RSI
                            delta = hist['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
                            
                            st.metric(
                                "RSI (14)",
                                f"{current_rsi:.1f}",
                                delta=f"{rsi.iloc[-1] - rsi.iloc[-2]:.1f}" if len(rsi) > 1 else "0.0"
                            )
                            
                            # RSI interpretation
                            if current_rsi > 70:
                                st.error("Overbought")
                            elif current_rsi < 30:
                                st.success("Oversold")
                            else:
                                st.info("Neutral")
                    
                    with col2:
                        if "MACD" in show_indicators:
                            # Calculate MACD
                            exp1 = hist['Close'].ewm(span=12).mean()
                            exp2 = hist['Close'].ewm(span=26).mean()
                            macd = exp1 - exp2
                            signal = macd.ewm(span=9).mean()
                            histogram = macd - signal
                            
                            current_macd = macd.iloc[-1] if not macd.empty else 0
                            current_signal = signal.iloc[-1] if not signal.empty else 0
                            
                            st.metric(
                                "MACD",
                                f"{current_macd:.2f}",
                                delta=f"{current_macd - current_signal:.2f}"
                            )
                            
                            # MACD interpretation
                            if current_macd > current_signal:
                                st.success("Bullish")
                            else:
                                st.error("Bearish")
                    
                    with col3:
                        # Volume Analysis
                        if "Volume" in show_indicators:
                            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
                            current_volume = hist['Volume'].iloc[-1]
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            
                            st.metric(
                                "Volume Ratio",
                                f"{volume_ratio:.1f}x",
                                delta=f"{((current_volume - avg_volume) / avg_volume * 100):.1f}%" if avg_volume > 0 else "0%"
                            )
                            
                            if volume_ratio > 1.5:
                                st.success("High Volume")
                            elif volume_ratio < 0.5:
                                st.warning("Low Volume")
                            else:
                                st.info("Normal Volume")
                    
                    with col4:
                        # Price Change
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
                        price_change_pct = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 and hist['Close'].iloc[-2] > 0 else 0
                        
                        st.metric(
                            "Price Change",
                            f"{price_change:.2f}",
                            delta=f"{price_change_pct:.2f}%"
                        )
                        
                        if price_change > 0:
                            st.success("Gaining")
                        else:
                            st.error("Declining")
                    
                    # Technical Analysis Summary (only if we have data)
                    if show_indicators and len(hist) > 0:
                        st.markdown("---")
                        st.subheader("📊 Technical Analysis Summary")
                        
                        # Build summary based on available indicators
                        tech_summary_parts = [
                            f"Stock: {selected_ticker_tech}",
                            f"Current Price: ₹{hist['Close'].iloc[-1]:.2f}",
                            f"Time Period: {time_period}"
                        ]
                        
                        # Add indicator-specific data only if calculated
                        if "RSI" in show_indicators:
                            delta = hist['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
                            tech_summary_parts.append(f"RSI (14): {current_rsi:.1f}")
                        
                        if "MACD" in show_indicators:
                            exp1 = hist['Close'].ewm(span=12).mean()
                            exp2 = hist['Close'].ewm(span=26).mean()
                            macd = exp1 - exp2
                            signal = macd.ewm(span=9).mean()
                            current_macd = macd.iloc[-1] if not macd.empty else 0
                            current_signal = signal.iloc[-1] if not signal.empty else 0
                            tech_summary_parts.append(f"MACD: {current_macd:.2f} (Signal: {current_signal:.2f})")
                        
                        if "Volume" in show_indicators and 'Volume' in hist.columns:
                            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
                            current_volume = hist['Volume'].iloc[-1]
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            tech_summary_parts.append(f"Volume Ratio: {volume_ratio:.1f}x")
                        
                        # Price change
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                        price_change_pct = (price_change / hist['Close'].iloc[0] * 100) if hist['Close'].iloc[0] > 0 else 0
                        tech_summary_parts.append(f"Price Change: {price_change_pct:+.2f}%")
                        
                        tech_summary = "\n".join(tech_summary_parts)
                        
                        # Generate AI-powered technical analysis
                        try:
                            import openai
                            openai.api_key = st.secrets["api_keys"]["open_ai"]
                            
                            response = openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a professional technical analyst. Provide a brief technical analysis summary based on the indicators provided. Focus on key signals and trading implications. Use emojis and be concise."},
                                    {"role": "user", "content": tech_summary}
                                ],
                                temperature=0.7,
                                max_tokens=300
                            )
                            
                            ai_tech_analysis = response.choices[0].message.content
                            st.markdown(f'<div class="ai-response-box"><strong>🤖 Technical Analysis:</strong><br><br>{ai_tech_analysis}</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate AI analysis: {str(e)[:100]}")
                
                else:
                    st.warning(f"⚠️ No sufficient data available for {selected_ticker_tech}")
                    st.info("""
                    **Possible reasons:**
                    - Stock may be delisted or suspended
                    - Ticker format may be incorrect
                    - Insufficient trading history
                    
                    **Suggestions:**
                    - Try a different stock from your portfolio
                    - Check if the stock is actively trading
                    - Use a shorter time period (1M or 3M)
                    """)
                    
            except Exception as e:
                st.error(f"⚠️ Error fetching technical data: {str(e)[:100]}")
                st.info("Try selecting a different stock or time period.")
    
    with tab6:
        st.subheader("⚡ Risk Metrics")
        st.caption("Advanced risk analysis and portfolio risk management")
        
        # Risk Analysis Controls
        col1, col2 = st.columns(2)
        
        with col1:
            risk_timeframe = st.selectbox(
                "Risk Analysis Timeframe:",
                ["1M", "3M", "6M", "1Y"],
                index=2,
                key="risk_timeframe"
            )
        
        with col2:
            confidence_level = st.selectbox(
                "VaR Confidence Level:",
                ["90%", "95%", "99%"],
                index=1,
                key="var_confidence"
            )
        
        # Calculate Risk Metrics
        try:
            # Prepare risk data with robust null handling
            risk_data = []
            total_portfolio_value = 0
            
            # First pass: calculate total portfolio value
            for holding in holdings:
                try:
                    current_price = holding.get('current_price')
                    if current_price is None or current_price == 0:
                        current_price = holding.get('average_price', 0)
                    
                    if current_price is None or current_price == 0:
                        continue  # Skip holdings with no price data
                    
                    quantity = holding.get('total_quantity', 0)
                    if quantity is None or quantity == 0:
                        continue
                    
                    current_value = float(current_price) * float(quantity)
                    total_portfolio_value += current_value
                except (TypeError, ValueError):
                    continue
            
            # Second pass: build risk data
            for holding in holdings:
                try:
                    current_price = holding.get('current_price')
                    if current_price is None or current_price == 0:
                        current_price = holding.get('average_price', 0)
                    
                    if current_price is None or current_price == 0:
                        continue
                    
                    quantity = holding.get('total_quantity', 0)
                    avg_price = holding.get('average_price', 0)
                    
                    if quantity is None or quantity == 0 or avg_price is None or avg_price == 0:
                        continue
                    
                    current_value = float(current_price) * float(quantity)
                    investment = float(quantity) * float(avg_price)
                    pnl = current_value - investment
                    pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                    weight = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                    
                    risk_data.append({
                        'ticker': holding.get('ticker', 'Unknown'),
                        'stock_name': holding.get('stock_name', 'Unknown'),
                        'current_value': current_value,
                        'pnl_pct': pnl_pct,
                        'weight': weight
                    })
                except (TypeError, ValueError, KeyError) as e:
                    continue
            
            df_risk = pd.DataFrame(risk_data)
            
            if not df_risk.empty:
                # Risk Metrics Dashboard
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Portfolio Volatility (simplified)
                    portfolio_volatility = df_risk['pnl_pct'].std()
                    st.metric(
                        "Portfolio Volatility",
                        f"{portfolio_volatility:.2f}%",
                        help="Standard deviation of returns"
                    )
                
                with col2:
                    # Value at Risk (VaR) - Simplified calculation
                    var_95 = np.percentile(df_risk['pnl_pct'], 5)  # 5th percentile for 95% VaR
                    st.metric(
                        f"VaR ({confidence_level})",
                        f"{var_95:.2f}%",
                        help="Maximum expected loss with given confidence"
                    )
                
                with col3:
                    # Maximum Drawdown
                    max_drawdown = df_risk['pnl_pct'].min()
                    st.metric(
                        "Max Drawdown",
                        f"{max_drawdown:.2f}%",
                        help="Maximum peak-to-trough decline"
                    )
                
                with col4:
                    # Sharpe Ratio (simplified)
                    risk_free_rate = 6.0  # Assume 6% risk-free rate
                    excess_return = df_risk['pnl_pct'].mean() - risk_free_rate
                    sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
                    st.metric(
                        "Sharpe Ratio",
                        f"{sharpe_ratio:.2f}",
                        help="Risk-adjusted return measure"
                    )
                
                # Risk Analysis Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk-Return Scatter Plot
                    fig_risk_return = px.scatter(
                        df_risk,
                        x='pnl_pct',
                        y='current_value',
                        size='weight',
                        color='pnl_pct',
                        hover_data=['ticker', 'stock_name'],
                        title="Risk-Return Analysis",
                        labels={'pnl_pct': 'Return (%)', 'current_value': 'Current Value (₹)'},
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_risk_return.update_layout(height=400)
                    st.plotly_chart(fig_risk_return, use_container_width=True)
                
                with col2:
                    # Portfolio Concentration
                    fig_concentration = px.pie(
                        df_risk,
                        values='current_value',
                        names='ticker',
                        title="Portfolio Concentration",
                        hole=0.4
                    )
                    fig_concentration.update_layout(height=400)
                    st.plotly_chart(fig_concentration, use_container_width=True)
                
                # Risk Summary Table
                st.markdown("---")
                st.subheader("📊 Individual Stock Risk Analysis")
                
                risk_summary = df_risk.copy()
                risk_summary['Risk Level'] = risk_summary['pnl_pct'].apply(
                    lambda x: 'High' if abs(x) > 20 else 'Medium' if abs(x) > 10 else 'Low'
                )
                risk_summary['Recommendation'] = risk_summary['pnl_pct'].apply(
                    lambda x: 'Hold' if x > 5 else 'Review' if x < -10 else 'Monitor'
                )
                
                st.dataframe(
                    risk_summary[['ticker', 'stock_name', 'current_value', 'pnl_pct', 'weight', 'Risk Level', 'Recommendation']].style.format({
                        'current_value': '₹{:,.0f}',
                        'pnl_pct': '{:+.2f}%',
                        'weight': '{:.1%}'
                    }),
                    use_container_width=True
                )
                
                # AI Risk Analysis
                try:
                    import openai
                    openai.api_key = st.secrets["api_keys"]["open_ai"]
                    
                    risk_summary_text = f"""
                    Portfolio Risk Analysis:
                    - Portfolio Volatility: {portfolio_volatility:.2f}%
                    - VaR (95%): {var_95:.2f}%
                    - Max Drawdown: {max_drawdown:.2f}%
                    - Sharpe Ratio: {sharpe_ratio:.2f}
                    - Number of Holdings: {len(holdings)}
                    - Top Risk Holdings: {', '.join(df_risk.nsmallest(3, 'pnl_pct')['ticker'].tolist())}
                    """
                    
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a professional risk analyst. Analyze the portfolio risk metrics and provide actionable risk management recommendations. Focus on diversification, position sizing, and risk mitigation strategies. Use emojis and be practical."},
                            {"role": "user", "content": risk_summary_text}
                        ],
                        temperature=0.7,
                        max_tokens=400
                    )
                    
                    ai_risk_analysis = response.choices[0].message.content
                    st.markdown(f'<div class="ai-response-box"><strong>🤖 Risk Analysis:</strong><br><br>{ai_risk_analysis}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate AI risk analysis: {str(e)[:50]}")
            
        except Exception as e:
            st.error(f"Error calculating risk metrics: {str(e)[:100]}")
    
    with tab7:
        st.subheader("🔄 Compare Holdings")
        st.caption("Compare performance across different dimensions")
        
        # Prepare comparison data
        comparison_data = []
        if not holdings:
            st.warning("⚠️ No holdings found. Please upload transaction files first.")
            return
        
        for holding in holdings:
            try:
                current_price = holding.get('current_price')
                if current_price is None or current_price == 0:
                    current_price = holding.get('average_price', 0)
                
                # Ensure we have valid numeric values
                if current_price is None or current_price == 0:
                    continue  # Skip holdings with no price data
                
                current_value = float(current_price) * float(holding['total_quantity'])
                investment = float(holding['total_quantity']) * float(holding['average_price'])
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                stars, grade, rating = get_performance_rating(pnl_pct)
                
                comparison_data.append({
                    'ticker': holding['ticker'],
                    'stock_name': holding['stock_name'],
                    'asset_type': holding.get('asset_type', 'Unknown'),
                    'channel': holding.get('channel', 'Unknown'),
                    'sector': holding.get('sector', 'Unknown'),
                    'investment': investment,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'rating': stars,
                    'grade': grade
                })
            except (ValueError, TypeError) as e:
                st.warning(f"⚠️ Skipping holding {holding.get('ticker', 'Unknown')}: Invalid data")
                continue
        
        df_compare = pd.DataFrame(comparison_data)
        
        if df_compare.empty:
            st.warning("⚠️ No valid holdings data available for comparison. Please check if prices are being fetched correctly.")
            # Debug information
            #st.caption(f"Debug: Found {len(holdings)} holdings, but none had valid price data")
            if holdings:
                #st.caption("Sample holding data:")
                sample_holding = holdings[0]
                #st.caption(f"Ticker: {sample_holding.get('ticker')}, Current Price: {sample_holding.get('current_price')}, Average Price: {sample_holding.get('average_price')}")
            return
        
        # Debug information
        #st.caption(f"✅ Loaded {len(df_compare)} holdings for comparison")
        
        # Enhanced Comparison Options with Better UI
        st.markdown("### 📊 Select Comparison Type")
        
        comparison_type = st.radio(
            "Compare by:",
            ["By Channel", "By Sector", "By Asset Type", "By Individual Holdings", "Multi-Comparison"],
            horizontal=True,
            help="Choose how you want to compare your holdings"
        )
        
        # Add some spacing
        st.markdown("---")
        
        if comparison_type == "By Channel":
            st.markdown("#### 📡 Channel Comparison")
            
            # Get unique channels with counts
            channels = df_compare['channel'].unique().tolist()
            channel_counts = df_compare['channel'].value_counts().to_dict()
            
            # Enhanced multi-select for channels with search
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Initialize session state for channels
                if 'selected_channels_state' not in st.session_state:
                    st.session_state.selected_channels_state = channels[:min(3, len(channels))]
                
                selected_channels = st.multiselect(
                    "🔍 Select channels to compare:",
                    channels,
                    default=st.session_state.selected_channels_state,
                    help="Search and select multiple channels. Use Ctrl+Click for multiple selections.",
                    placeholder="Type to search channels...",
                    key="channel_comparison_multiselect"
                )
                
                # Update session state
                st.session_state.selected_channels_state = selected_channels
            
            with col2:
                st.markdown("**📊 Available Channels:**")
                for channel in channels:
                    count = channel_counts.get(channel, 0)
                    st.caption(f"• {channel}: {count} holdings")
            
            # Quick select buttons
            if len(channels) > 1:
                st.markdown("**⚡ Quick Select:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Select All", key="select_all_channels"):
                        st.session_state.selected_channels_state = channels
                        st.rerun()
                with col2:
                    if st.button("Select Top 3", key="select_top3_channels"):
                        top_channels = df_compare.groupby('channel')['current_value'].sum().nlargest(3).index.tolist()
                        st.session_state.selected_channels_state = top_channels
                        st.rerun()
                with col3:
                    if st.button("Clear All", key="clear_channels"):
                        st.session_state.selected_channels_state = []
                        st.rerun()
            
            if selected_channels:
                # Filter data
                channel_comparison = df_compare[df_compare['channel'].isin(selected_channels)].groupby('channel').agg({
                    'investment': 'sum',
                    'current_value': 'sum',
                    'pnl': 'sum'
                }).reset_index()
                
                channel_comparison['pnl_pct'] = (channel_comparison['pnl'] / channel_comparison['investment'] * 100)
                channel_comparison['rating'] = channel_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[0])
                channel_comparison['grade'] = channel_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[1])
                
                # Comparison Chart
                fig_channel_compare = go.Figure()
                
                fig_channel_compare.add_trace(go.Bar(
                    name='Investment',
                    x=channel_comparison['channel'],
                    y=channel_comparison['investment'],
                    marker_color='lightblue'
                ))
                
                fig_channel_compare.add_trace(go.Bar(
                    name='Current Value',
                    x=channel_comparison['channel'],
                    y=channel_comparison['current_value'],
                    marker_color='lightgreen'
                ))
                
                fig_channel_compare.update_layout(
                    title="Channel Investment vs Current Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_channel_compare, use_container_width=True)
                
                # Performance comparison
                fig_perf = px.bar(
                    channel_comparison,
                    x='channel',
                    y='pnl_pct',
                    title="Channel Performance Comparison (%)",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='pnl_pct'
                )
                fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Detailed table
                st.dataframe(channel_comparison.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        elif comparison_type == "By Sector":
            st.markdown("#### 🏢 Sector Comparison")
            
            # Get unique sectors (from holdings data we need to fetch sector info)
            # For now, we'll use asset_type as a proxy, but you can enhance this by adding sector to holdings
            st.info("📊 Sector analysis based on stock data")
            
            # Get holdings with sector information
            sector_holdings = {}
            for holding in holdings:
                sector = holding.get('sector', holding.get('asset_type', 'Unknown'))
                current_price = holding.get('current_price')
                if current_price is None or current_price == 0:
                    current_price = holding.get('average_price', 0)
                current_value = float(current_price) * float(holding['total_quantity'])
                investment = float(holding['total_quantity']) * float(holding['average_price'])
                pnl = current_value - investment
                
                if sector not in sector_holdings:
                    sector_holdings[sector] = {'investment': 0, 'current_value': 0, 'pnl': 0, 'count': 0}
                
                sector_holdings[sector]['investment'] += investment
                sector_holdings[sector]['current_value'] += current_value
                sector_holdings[sector]['pnl'] += pnl
                sector_holdings[sector]['count'] += 1
            
            # Create sector comparison dataframe
            sector_data = []
            for sector, data in sector_holdings.items():
                pnl_pct = (data['pnl'] / data['investment'] * 100) if data['investment'] > 0 else 0
                stars, grade, rating = get_performance_rating(pnl_pct)
                
                sector_data.append({
                    'Sector': sector,
                    'Holdings': data['count'],
                    'Rating': stars,
                    'Grade': grade,
                    'Investment': data['investment'],
                    'Current Value': data['current_value'],
                    'P&L': data['pnl'],
                    'P&L %': pnl_pct
                })
            
            df_sectors = pd.DataFrame(sector_data).sort_values('P&L %', ascending=False)
            
            # Sector comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sector_pie = px.pie(
                    df_sectors,
                    values='Current Value',
                    names='Sector',
                    title="Portfolio Allocation by Sector"
                )
                st.plotly_chart(fig_sector_pie, use_container_width=True)
            
            with col2:
                fig_sector_perf = px.bar(
                    df_sectors,
                    x='Sector',
                    y='P&L %',
                    title="Sector Performance (%)",
                    color='P&L %',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='P&L %'
                )
                fig_sector_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_sector_perf, use_container_width=True)
            
            # Detailed sector table
            st.dataframe(df_sectors.style.format({
                'Investment': '₹{:,.0f}',
                'Current Value': '₹{:,.0f}',
                'P&L': '₹{:,.0f}',
                'P&L %': '{:+.2f}%'
            }), use_container_width=True)
        
        elif comparison_type == "By Asset Type":
            st.markdown("#### 💼 Asset Type Comparison")
            
            # Get unique asset types with counts and values
            asset_types = df_compare['asset_type'].unique().tolist()
            type_counts = df_compare['asset_type'].value_counts().to_dict()
            type_values = df_compare.groupby('asset_type')['current_value'].sum().to_dict()
            
            # Enhanced multi-select for asset types
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Initialize session state for asset types
                if 'selected_types_state' not in st.session_state:
                    st.session_state.selected_types_state = asset_types[:min(3, len(asset_types))]
                
                selected_types = st.multiselect(
                    "🔍 Select asset types to compare:",
                    asset_types,
                    default=st.session_state.selected_types_state,
                    help="Search and select multiple asset types. Use Ctrl+Click for multiple selections.",
                    placeholder="Type to search asset types...",
                    key="asset_type_comparison_multiselect"
                )
                
                # Update session state
                st.session_state.selected_types_state = selected_types
            
            with col2:
                st.markdown("**📊 Available Asset Types:**")
                for asset_type in asset_types:
                    count = type_counts.get(asset_type, 0)
                    value = type_values.get(asset_type, 0)
                    st.caption(f"• {asset_type}: {count} holdings (₹{value:,.0f})")
            
            # Quick select buttons for asset types
            if len(asset_types) > 1:
                st.markdown("**⚡ Quick Select:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Select All", key="select_all_types"):
                        st.session_state.selected_types = asset_types
                        st.rerun()
                with col2:
                    if st.button("Select Top 3", key="select_top3_types"):
                        top_types = df_compare.groupby('asset_type')['current_value'].sum().nlargest(3).index.tolist()
                        st.session_state.selected_types = top_types
                        st.rerun()
                with col3:
                    if st.button("Clear All", key="clear_types"):
                        st.session_state.selected_types = []
                        st.rerun()
            
            if selected_types:
                # Filter data
                type_comparison = df_compare[df_compare['asset_type'].isin(selected_types)].groupby('asset_type').agg({
                    'investment': 'sum',
                    'current_value': 'sum',
                    'pnl': 'sum'
                }).reset_index()
                
                type_comparison['pnl_pct'] = (type_comparison['pnl'] / type_comparison['investment'] * 100)
                type_comparison['rating'] = type_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[0])
                type_comparison['grade'] = type_comparison['pnl_pct'].apply(lambda x: get_performance_rating(x)[1])
                
                # Comparison Chart
                fig_type_compare = go.Figure()
                
                fig_type_compare.add_trace(go.Bar(
                    name='Investment',
                    x=type_comparison['asset_type'],
                    y=type_comparison['investment'],
                    marker_color='lightblue'
                ))
                
                fig_type_compare.add_trace(go.Bar(
                    name='Current Value',
                    x=type_comparison['asset_type'],
                    y=type_comparison['current_value'],
                    marker_color='lightgreen'
                ))
                
                fig_type_compare.update_layout(
                    title="Asset Type Investment vs Current Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_type_compare, use_container_width=True)
                
                # Performance comparison
                fig_perf = px.bar(
                    type_comparison,
                    x='asset_type',
                    y='pnl_pct',
                    title="Asset Type Performance Comparison (%)",
                    color='pnl_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text='pnl_pct'
                )
                fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Detailed table
                st.dataframe(type_comparison.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        elif comparison_type == "By Individual Holdings":
            st.markdown("#### 📈 Individual Holdings Comparison")
            
            # Create enhanced holdings list with performance info
            holdings_list = []
            for _, row in df_compare.iterrows():
                pnl_pct = row['pnl_pct']
                emoji = "🚀" if pnl_pct > 10 else "📈" if pnl_pct > 0 else "📉" if pnl_pct < -5 else "➡️"
                holdings_list.append(f"{emoji} {row['ticker']} - {row['stock_name']} ({pnl_pct:+.1f}%)")
            
            # Enhanced multi-select for holdings
            # Initialize session state for holdings
            if 'selected_holdings_state' not in st.session_state:
                st.session_state.selected_holdings_state = holdings_list[:min(3, len(holdings_list))]
            
            selected_holdings = st.multiselect(
                "🔍 Select holdings to compare (up to 10):",
                holdings_list,
                default=st.session_state.selected_holdings_state,
                help="Search and select multiple holdings. Use Ctrl+Click for multiple selections.",
                placeholder="Type to search holdings...",
                key="holdings_comparison_multiselect"
            )
            
            # Update session state
            st.session_state.selected_holdings_state = selected_holdings
            
            if selected_holdings:
                # Extract tickers from selection (handle emoji and formatting)
                selected_tickers = []
                for h in selected_holdings:
                    # Split by space and get the part after emoji (ticker)
                    parts = h.split(' ')
                    if len(parts) >= 2:
                        ticker = parts[1]  # Get ticker (e.g., "HCLTECH.NS")
                        selected_tickers.append(ticker)
                
                holding_comparison = df_compare[df_compare['ticker'].isin(selected_tickers)]
                
                # Historical Performance Line Chart
                fig_holdings = go.Figure()
                
                # Get historical data for each selected holding
                for _, holding in holding_comparison.iterrows():
                    ticker = holding['ticker']
                    stock_name = holding['stock_name']
                    
                    # Get stock_id from the original holdings data
                    stock_id = None
                    for h in holdings:
                        if h['ticker'] == ticker:
                            stock_id = h['stock_id']
                            break
                    
                    if not stock_id:
                        continue  # Skip if stock_id not found
                    
                    # Get historical prices for this stock_id
                    historical_prices = db.get_historical_prices_for_stock_silent(stock_id)
                    
                    # Debug information
                    #st.caption(f"Debug: {ticker} - Found {len(historical_prices) if historical_prices else 0} historical prices")
                    
                    if historical_prices and len(historical_prices) > 0:
                        # Sort by date
                        historical_prices.sort(key=lambda x: x['price_date'])
                        
                        # Prepare data for line chart
                        dates = [price['price_date'] for price in historical_prices]
                        prices = [price['price'] for price in historical_prices]
                        
                        # Calculate percentage change from first price
                        if len(prices) > 0:
                            first_price = prices[0]
                            pct_changes = [((price - first_price) / first_price) * 100 for price in prices]
                            
                            # Add line trace
                            fig_holdings.add_trace(go.Scatter(
                                x=dates,
                                y=pct_changes,
                                mode='lines+markers',
                                name=f"{ticker} - {stock_name[:20]}{'...' if len(stock_name) > 20 else ''}",
                                line=dict(width=2),
                                marker=dict(size=4),
                                hovertemplate=f"<b>{ticker}</b><br>" +
                                            "Date: %{x}<br>" +
                                            "Price: ₹%{customdata:.2f}<br>" +
                                            "Change: %{y:.2f}%<br>" +
                                            "<extra></extra>",
                                customdata=prices
                            ))
                
                # Update layout for line chart
                fig_holdings.update_layout(
                    title="📈 Historical Performance Comparison - Selected Holdings",
                    xaxis_title="Date",
                    yaxis_title="Price Change (%)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)'
                    )
                )
                
                if fig_holdings.data:
                    st.plotly_chart(fig_holdings, use_container_width=True)
                else:
                    st.warning("⚠️ No historical data available for selected holdings. This could be because:")
                    st.caption("• Historical prices haven't been fetched yet")
                    st.caption("• The holdings don't have price history in the database")
                    st.caption("• Try running 'Update Prices' to fetch historical data")
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Holdings Selected", len(selected_holdings))
                with col2:
                    avg_performance = holding_comparison['pnl_pct'].mean()
                    st.metric("📈 Average Performance", f"{avg_performance:+.1f}%")
                with col3:
                    best_performer = holding_comparison.loc[holding_comparison['pnl_pct'].idxmax()]
                    st.metric("🏆 Best Performer", f"{best_performer['ticker']} ({best_performer['pnl_pct']:+.1f}%)")
                
                # Detailed comparison table
                comparison_table = holding_comparison[['ticker', 'stock_name', 'rating', 'grade', 'investment', 'current_value', 'pnl', 'pnl_pct']]
                st.dataframe(comparison_table.style.format({
                    'investment': '₹{:,.0f}',
                    'current_value': '₹{:,.0f}',
                    'pnl': '₹{:,.0f}',
                    'pnl_pct': '{:+.2f}%'
                }), use_container_width=True)
        
        else:  # Multi-Comparison
            st.markdown("#### 🔀 Multi-Dimensional Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select channel
                channels = ['All'] + df_compare['channel'].unique().tolist()
                selected_channel = st.selectbox("Filter by Channel:", channels)
            
            with col2:
                # Select asset type
                asset_types = ['All'] + df_compare['asset_type'].unique().tolist()
                selected_type = st.selectbox("Filter by Asset Type:", asset_types)
            
            # Apply filters
            filtered_df = df_compare.copy()
            if selected_channel != 'All':
                filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
            if selected_type != 'All':
                filtered_df = filtered_df[filtered_df['asset_type'] == selected_type]
            
            if not filtered_df.empty:
                # Summary metrics
                total_inv = filtered_df['investment'].sum()
                total_curr = filtered_df['current_value'].sum()
                total_pnl = filtered_df['pnl'].sum()
                total_pnl_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Investment", f"₹{total_inv:,.0f}")
                with col2:
                    st.metric("Current Value", f"₹{total_curr:,.0f}")
                with col3:
                    st.metric("P&L", f"₹{total_pnl:,.0f}")
                with col4:
                    st.metric("P&L %", f"{total_pnl_pct:+.1f}%")
                
                # Holdings in this filter
                st.markdown(f"**{len(filtered_df)} holdings match your filters**")
                
                # Top performers in this filter
                top_3 = filtered_df.nlargest(3, 'pnl_pct')
                st.markdown("**🏆 Top 3 Performers:**")
                for _, row in top_3.iterrows():
                    st.write(f"{row['rating']} {row['ticker']} - {row['stock_name'][:30]}: {row['pnl_pct']:+.1f}%")
                
                # Chart
                fig_multi = px.scatter(
                    filtered_df,
                    x='investment',
                    y='pnl_pct',
                    size='current_value',
                    color='asset_type',
                    hover_data=['ticker', 'stock_name', 'channel'],
                    title="Investment vs Performance (bubble size = current value)"
                )
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("No holdings match the selected filters")

def ai_assistant_page():
    """Dedicated AI Assistant page"""
    st.header("🤖 AI Assistant")
    st.caption("Your intelligent portfolio advisor with access to all your data")
    
    user = st.session_state.user
    
    # Initialize chat history and PDF context
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Always load PDF context from database to ensure older session PDFs are included
    st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
    
    # Show PDF context status and refresh option
    pdf_count = len(db.get_user_pdfs(user['id']))
    if pdf_count > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📚 Loaded {pdf_count} PDFs from database for AI context")
        with col2:
            if st.button("🔄", key="refresh_pdf_context_main", help="Refresh PDF context from database"):
                st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                st.success("PDF context refreshed!")
                st.rerun()
    
    # Get portfolio context (cached)
    holdings = get_cached_holdings(user['id'])
    
    # Get cached portfolio summary
    portfolio_summary = get_cached_portfolio_summary(holdings)
    
    # Get user PDFs for the session
    user_pdfs = db.get_user_pdfs(user['id'])
    
    # Enhanced chat interface
    st.markdown("---")
    st.markdown("**💬 Chat with AI**")
    
    # Chat input
    user_question = st.text_input(
        "Ask me anything about your portfolio:",
        placeholder="e.g., 'How is my technology sector performing?' or 'Should I rebalance my portfolio?'",
        key="ai_chat_input"
    )
    
    if user_question:
        with st.spinner("🤖 AI is thinking..."):
            try:
                import openai
                openai.api_key = st.secrets["api_keys"]["open_ai"]
                
                # Safe float conversion helper
                def safe_float(value, default=0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                # Create comprehensive raw data context for AI
                raw_data_context = "\n\n📊 RAW PORTFOLIO DATA (All Holdings with Details):\n"
                raw_data_context += "=" * 80 + "\n"
                
                for idx, holding in enumerate(holdings, 1):
                    current_price = holding.get('current_price') or holding.get('average_price', 0)
                    current_value = safe_float(current_price, 0) * safe_float(holding.get('total_quantity'), 0)
                    investment = safe_float(holding.get('total_quantity'), 0) * safe_float(holding.get('average_price'), 0)
                    pnl = current_value - investment
                    pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0
                    
                    raw_data_context += f"\n[{idx}] {holding.get('ticker', 'N/A')} - {holding.get('stock_name', 'N/A')}\n"
                    raw_data_context += f"  • Asset Type: {holding.get('asset_type', 'Unknown')}\n"
                    raw_data_context += f"  • Channel: {holding.get('channel', 'Unknown')}\n"
                    raw_data_context += f"  • Sector: {holding.get('sector', 'Unknown')}\n"
                    raw_data_context += f"  • Quantity: {holding.get('total_quantity', 0):,.2f}\n"
                    raw_data_context += f"  • Avg Buy Price: ₹{holding.get('average_price', 0):,.2f}\n"
                    raw_data_context += f"  • Current Price: ₹{current_price:,.2f}\n"
                    raw_data_context += f"  • Investment: ₹{investment:,.2f}\n"
                    raw_data_context += f"  • Current Value: ₹{current_value:,.2f}\n"
                    raw_data_context += f"  • P&L: ₹{pnl:,.2f} ({pnl_pct:+.1f}%)\n"
                    raw_data_context += "-" * 80 + "\n"
                
                # Get transactions data for detailed analysis
                transactions_context = "\n\n📝 TRANSACTION HISTORY (Recent Activity):\n"
                transactions_context += "=" * 80 + "\n"
                try:
                    # Get all transactions for the user (increased limit for comprehensive analysis)
                    all_transactions = db.supabase.table('user_transactions').select('*').eq('user_id', user['id']).order('transaction_date', desc=True).limit(100).execute()
                    
                    if all_transactions.data:
                        transactions_context += f"Total Transactions Available: {len(all_transactions.data)}\n\n"
                        # Show more transactions for better context (50 instead of 20)
                        for trans in all_transactions.data[:50]:
                            transactions_context += f"Date: {trans.get('transaction_date', 'N/A')} | "
                            transactions_context += f"Stock: {trans.get('ticker', 'N/A')} | "
                            transactions_context += f"Type: {trans.get('transaction_type', 'N/A')} | "
                            transactions_context += f"Qty: {trans.get('quantity', 0)} | "
                            transactions_context += f"Price: ₹{trans.get('price', 0):,.2f} | "
                            transactions_context += f"Channel: {trans.get('channel', 'N/A')}\n"
                    else:
                        transactions_context += "No transactions found.\n"
                except Exception as e:
                    transactions_context += f"Error loading transactions: {str(e)[:100]}\n"
                
                # Include PDF context
                pdf_context_text = ""
                
                # Include ALL PDF summaries (more comprehensive)
                recent_pdfs = db.get_user_pdfs(user['id'])
                if recent_pdfs:
                    pdf_context_text += f"\n\n📚 UPLOADED DOCUMENTS ({len(recent_pdfs)} total):"
                    for pdf in recent_pdfs:
                        pdf_context_text += f"\n\n📄 {pdf['filename']}:"
                        if pdf.get('ai_summary'):
                            # Include full AI summary (not truncated)
                            pdf_context_text += f"\n{pdf.get('ai_summary', 'No summary')}"
                        else:
                            # If no summary, include first part of PDF text
                            pdf_text = pdf.get('pdf_text', '')
                            if pdf_text:
                                pdf_context_text += f"\n{pdf_text[:1000]}..."
                
                # If no PDFs, note that
                if not recent_pdfs:
                    pdf_context_text = "\n\n📄 No documents uploaded yet."
                
                full_context = f"""🎯 COMPREHENSIVE DATA SOURCES (COMBINE BOTH for complete analysis):

1️⃣ CURRENT PORTFOLIO DATA (Financial metrics - prices, P&L, values):
{raw_data_context}

2️⃣ PORTFOLIO SUMMARY (Aggregated performance view):
{portfolio_summary}

3️⃣ TRANSACTION HISTORY (Recent activity and patterns):
{transactions_context}

4️⃣ RESEARCH DOCUMENTS (Market analysis, insights, and outlook):
{pdf_context_text}

📋 USER QUESTION: {user_question}

💡 ANALYSIS APPROACH: 
- Use portfolio data for current financial performance
- Use PDF research for market insights and future outlook  
- COMBINE both sources for comprehensive analysis
- Show how research insights align with current portfolio performance"""
                
                # Add debug expander to show what's being sent to AI (only if needed)
                with st.expander("🔍 Debug: Full context sent to AI (click to expand)", expanded=False):
                    approx_tokens = len(full_context) // 4
                    st.caption(f"Total characters: {len(full_context):,} | Estimated tokens: {approx_tokens:,}")
                    st.text_area("Full context:", full_context, height=400)
                
                # Use GPT-4 for larger context window (128K tokens) - better for comprehensive data
                model_to_use = "gpt-4o"  # 128K context window, better reasoning
                
                response = openai.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": """You are an expert portfolio analyst with COMPLETE ACCESS to:

📊 DATA SOURCES (Use BOTH for comprehensive analysis):
1. RAW PORTFOLIO DATA - Current prices, P&L, values, holdings details
2. Portfolio Summary - Aggregated by asset type, channel, sector, top gainers/losers
3. TRANSACTION HISTORY - Recent buy/sell transactions with dates, prices, quantities
4. UPLOADED PDF DOCUMENTS - Research reports, analysis, and market insights

🎯 ANALYSIS APPROACH:
- COMBINE portfolio data with PDF research for comprehensive insights
- Use portfolio data for current financial metrics (prices, P&L, values)
- Use PDF data for research insights, market analysis, and future outlook
- Create a unified analysis that leverages BOTH sources

📈 ANALYSIS REQUIREMENTS:
- ALWAYS cite specific tickers and stock names from RAW PORTFOLIO DATA
- Use EXACT numbers from portfolio data (P&L %, prices, quantities, values)
- INTEGRATE PDF research insights with current portfolio performance
- Compare holdings within same channel/sector when relevant
- Reference transaction history to understand buying patterns
- Provide data-driven recommendations combining both sources

💡 ANSWER STRUCTURE:
1. Current performance using portfolio data
2. Research insights from PDFs (market analysis, outlook, recommendations)
3. Combined analysis showing how research aligns with current performance
4. Actionable recommendations based on BOTH portfolio data AND research

✅ COMBINE BOTH SOURCES:
- "Current Price: ₹1,165.80 (portfolio) + Research shows positive momentum..."
- "P&L: +38.3% (portfolio) + PDF analysis suggests continued growth potential..."
- "Portfolio shows strong performance + Research confirms bullish outlook..."

🚫 AVOID:
- Ignoring either portfolio data OR PDF insights
- Generic advice without specific data
- Not connecting research insights to current performance

Use emojis appropriately and be encouraging but data-focused."""},
                        {"role": "user", "content": full_context}
                    ],
                    temperature=0.7,
                    max_tokens=4000,  # Increased significantly for comprehensive responses
                    timeout=60  # Increased timeout for better model
                )
                
                if not response or not response.choices:
                    st.error("❌ No response received from AI. Please try again.")
                    st.stop()
                
                ai_response = response.choices[0].message.content
                
                if not ai_response:
                    st.error("❌ Empty response from AI. Please try again.")
                    st.stop()
                
                # Display the response immediately
                st.markdown("---")
                st.markdown("### 💬 AI Response:")
                st.success(ai_response)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "q": user_question,
                    "a": ai_response
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")
                st.error(f"Full error: {str(e)}")  # Show full error for debugging
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💭 Chat History**")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['q'][:50]}..."):
                st.markdown(f"**Question:** {chat['q']}")
                st.markdown(f"**Answer:** {chat['a']}")
    
    # PDF Library Section
    st.markdown("---")
    st.markdown("**📚 Your PDF Library**")
    
    if user_pdfs and len(user_pdfs) > 0:
        for pdf in user_pdfs:
            with st.expander(f"📄 {pdf['filename']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📅 Uploaded: {pdf['uploaded_at'][:10]}")
                    
                    if pdf.get('ai_summary'):
                        st.markdown("**🤖 AI Summary:**")
                        st.info(pdf['ai_summary'])
                    
                    # Add button to use this PDF for analysis
                    if st.button(f"🔍 Analyze {pdf['filename'][:20]}...", key=f"analyze_{pdf['id']}", help="Use this PDF for AI analysis"):
                        try:
                            import openai
                            openai.api_key = st.secrets["api_keys"]["open_ai"]
                            
                            # Get portfolio context
                            portfolio_summary = get_cached_portfolio_summary(holdings)
                            
                            # Analyze the stored PDF
                            analysis_prompt = f"""
                            Analyze this stored PDF document for portfolio management insights.
                            
                            📄 DOCUMENT INFO:
                            - Filename: {pdf['filename']}
                            - Uploaded: {pdf['uploaded_at'][:10]}
                            
                            💼 USER'S PORTFOLIO:
                            {portfolio_summary}
                            
                            📝 PDF CONTENT:
                            {pdf.get('pdf_text', '')[:5000]}...
                            
                            🤖 PREVIOUS AI SUMMARY:
                            {pdf.get('ai_summary', 'No previous summary')}
                            
                            Please provide a fresh analysis focusing on:
                            1. Key insights from the document
                            2. How it relates to the user's current portfolio
                            3. Actionable recommendations
                            
                            Be specific and actionable. Use emojis and clear formatting.
                            """
                            
                            response = openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": analysis_prompt}],
                                temperature=0.7,
                                max_tokens=800
                            )
                            
                            fresh_analysis = response.choices[0].message.content
                            
                            # Display the fresh analysis
                            st.markdown("### 🔍 Fresh Analysis")
                            st.markdown(fresh_analysis)
                            
                            # Store in chat history
                            st.session_state.chat_history.append({
                                "q": f"Analyze PDF: {pdf['filename']}", 
                                "a": fresh_analysis
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing PDF: {str(e)[:100]}")
                
                with col2:
                    if st.button("🗑️", key=f"del_{pdf['id']}", help="Delete this PDF"):
                        if db.delete_pdf(pdf['id']):
                            st.success("Deleted!")
                            st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                            st.rerun()
    else:
        st.caption("No PDFs uploaded yet")
    
    # PDF Upload for AI Analysis
    st.markdown("---")
    st.markdown("**📤 Upload PDF for AI Analysis**")
    
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file to analyze",
        type=['pdf'],
        help="Upload research reports, financial statements, or any document for AI analysis"
    )
    
    if uploaded_pdf:
        if st.button("🔍 Analyze PDF", type="primary"):
            with st.spinner("🔍 Analyzing PDF..."):
                try:
                    import PyPDF2
                    import pdfplumber
                    import openai
                    openai.api_key = st.secrets["api_keys"]["open_ai"]
                    
                    # Enhanced PDF extraction with tables and structure
                    pdf_text = ""
                    tables_found = []
                    page_count = 0
                    
                    st.info("🔍 Extracting content from PDF (text, tables, structure)...")
                    
                    # Try pdfplumber first (better for tables)
                    with pdfplumber.open(uploaded_pdf) as pdf:
                        for page_num, page in enumerate(pdf.pages, 1):
                            page_count += 1
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                            
                            # Extract tables
                            tables = page.extract_tables()
                            if tables:
                                for table_idx, table in enumerate(tables, 1):
                                    tables_found.append({
                                        'page': page_num,
                                        'table': table_idx,
                                        'data': table
                                    })
                    
                    if pdf_text:
                        st.success(f"✅ Extracted {len(pdf_text)} characters from {page_count} pages, found {len(tables_found)} tables")
                        
                        # Truncate if too long (OpenAI has token limits)
                        pdf_text_for_ai = pdf_text[:10000] + "..." if len(pdf_text) > 10000 else pdf_text
                        
                        # Prepare tables summary
                        tables_summary = ""
                        if tables_found:
                            tables_summary = f"\n📊 Tables Found: {len(tables_found)}\n"
                            for table in tables_found[:3]:  # Show first 3 tables
                                tables_summary += f"• Page {table['page']}, Table {table['table']}: {len(table['data'])} rows\n"
                        
                        # Get portfolio context
                        portfolio_summary = get_cached_portfolio_summary(holdings)
                        
                        # Enhanced analysis prompt with structured output
                        analysis_prompt = f"""
                        Analyze this PDF document comprehensively for portfolio management insights.
                        
                        📄 DOCUMENT INFO:
                        - Filename: {uploaded_pdf.name}
                        - Pages: {page_count}
                        {tables_summary}
                        
                        💼 USER'S PORTFOLIO:
                        {portfolio_summary}
                        
                        📝 PDF CONTENT:
                        {pdf_text_for_ai}
                        
                        Please provide a STRUCTURED analysis using this format:
                        
                        📋 **DOCUMENT SUMMARY**
                        [2-3 sentences describing what this PDF contains]
                        
                        📊 **KEY METRICS & DATA**
                        [Extract specific numbers, percentages, dates, returns mentioned]
                        • Metric: Value
                        • Metric: Value
                        
                        📈 **INSIGHTS FROM CHARTS/TABLES**
                        [Describe trends, patterns, or comparisons shown in tables/graphs]
                        
                        💡 **MAIN FINDINGS**
                        [3-5 key takeaways from the document]
                        
                        🎯 **PORTFOLIO RELEVANCE**
                        [How this information relates to the user's current holdings]
                        
                        ⚡ **RECOMMENDED ACTIONS**
                        [Specific, actionable next steps - if applicable]
                        
                        Use actual numbers and be specific. Focus on actionable insights.
                        """
                        
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": analysis_prompt}],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        ai_analysis = response.choices[0].message.content
                        
                        # Display the analysis
                        st.markdown("### 🤖 AI Analysis")
                        st.markdown(ai_analysis)
                        
                        # Store in chat history
                        st.session_state.chat_history.append({
                            "q": f"Analyze PDF: {uploaded_pdf.name}", 
                            "a": ai_analysis
                        })
                        
                        # Clean PDF text before saving (remove null bytes and control characters)
                        import re
                        cleaned_pdf_text = pdf_text.replace('\x00', ' ').replace('\u0000', ' ')
                        cleaned_pdf_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', ' ', cleaned_pdf_text)
                        cleaned_pdf_text = ''.join(char if char.isprintable() or char in '\n\t\r ' else ' ' for char in cleaned_pdf_text)
                        cleaned_pdf_text = re.sub(r'\s+', ' ', cleaned_pdf_text).strip()
                        
                        # Save PDF to database
                        save_result = db.save_pdf(
                            user_id=user['id'],
                            filename=uploaded_pdf.name,
                            pdf_text=cleaned_pdf_text,  # Store cleaned text
                            ai_summary=ai_analysis
                        )
                        
                        if save_result['success']:
                            st.success(f"✅ PDF '{uploaded_pdf.name}' saved to database!")
                            # Refresh the page to update the PDF library count
                            st.rerun()
                            
                            # Reload PDF context from database
                            st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                            st.info("📄 PDF content is now stored and available for all future sessions!")
                            
                        else:
                            st.error(f"❌ Could not save PDF: {save_result.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Could not extract text from PDF. Please ensure the PDF contains readable text.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)[:100]}")
    
    # Quick Tips Section
    st.markdown("---")
    st.markdown("**💡 Quick Tips**")
    st.caption("Try asking me:")
    st.caption("• 'How is my portfolio performing overall?'")
    st.caption("• 'Which sectors are my best performers?'")
    st.caption("• 'How can I reduce portfolio risk?'")
    st.caption("• 'Which channels are giving me the best returns?'")
    st.caption("• 'Should I rebalance my portfolio?'")
    st.caption("• 'Upload a research report for analysis'")

def upload_files_page():
    """Enhanced upload more files page"""
    st.header("📁 Upload More Files")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        color: white;
    }
    .upload-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .file-preview {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    user = st.session_state.user
    portfolios = db.get_user_portfolios(user['id'])
    
    if not portfolios:
        st.error("No portfolios found.")
        return
    
    # Portfolio selection with enhanced UI
    st.markdown("### 📊 Select Target Portfolio")
    portfolio = st.selectbox(
        "Choose portfolio:", 
        portfolios, 
        format_func=lambda x: f"{x['portfolio_name']} (ID: {x['id'][:8]}...)",
        help="Select which portfolio to add the new transactions to"
    )
    
    st.markdown("---")
    
    # Enhanced file upload section
    st.markdown("### 📤 Upload Transaction Files")
    
    # Show sample format
    with st.expander("📋 Sample CSV Format (Click to expand)", expanded=False):
        st.code("""date,ticker,quantity,transaction_type,price,stock_name,sector,channel
2024-01-15,RELIANCE,100,buy,2500,Reliance Industries,Oil & Gas,Zerodha
2024-02-01,120760,50,buy,250.75,Quant Flexi Cap Fund,Flexi Cap,Groww
2024-03-10,TCS,25,buy,3600,Tata Consultancy Services,IT Services,Zerodha""", language="csv")
        
        st.markdown("""
        **📝 Required Columns:**
        - `date`: Transaction date (YYYY-MM-DD)
        - `ticker`: Stock/MF ticker symbol
        - `quantity`: Number of shares/units
        - `transaction_type`: buy/sell
        - `price`: Price per share/unit (optional - will be fetched if missing)
        - `stock_name`: Name of the security (optional)
        - `sector`: Sector classification (optional)
        - `channel`: Channel/platform name (optional - will use filename if missing)
        """)
    
    # File uploader with enhanced styling
    uploaded_files = st.file_uploader(
        "📁 Choose CSV files to upload",
        type=['csv'],
        accept_multiple_files=True,
        help="Select one or more CSV files containing your transaction data. Files will be processed automatically.",
        key="upload_files_main"
    )
    
    # Show file preview
    if uploaded_files:
        st.markdown("### 📋 File Preview")
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.container():
                st.markdown(f"""
                <div class="file-preview">
                    <strong>📄 File {i+1}:</strong> {uploaded_file.name}<br>
                    <strong>📊 Size:</strong> {uploaded_file.size:,} bytes<br>
                    <strong>📅 Type:</strong> {uploaded_file.type}
                </div>
                """, unsafe_allow_html=True)
        
        # Processing options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**⚙️ Processing Options:**")
            auto_fetch_prices = st.checkbox(
                "🔍 Auto-fetch missing prices", 
                value=True, 
                help="Automatically fetch historical prices for transactions with missing or zero prices"
            )
            fetch_weekly_prices = st.checkbox(
                "📅 Fetch weekly historical prices", 
                value=True, 
                help="Automatically fetch 52 weeks of historical price data for all holdings"
            )
        
        with col2:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                st.info(f"🚀 Processing {len(uploaded_files)} file(s) for portfolio: {portfolio['portfolio_name']}")
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    process_uploaded_files(uploaded_files, user['id'], portfolio['id'])
                    progress_bar.progress(1.0)
                    status_text.success("✅ All files processed successfully!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.error(f"❌ Error processing files: {str(e)}")
        
        with col3:
            if st.button("🗑️ Clear Files", use_container_width=True):
                st.session_state.upload_files_main = []
        st.rerun()
    
    # Show transactions with week info
    st.subheader("📊 Your Transactions (with Week Info)")
    
    try:
        with st.spinner("🔄 Loading transactions..."):
            transactions = db.get_user_transactions(user['id'])
        
        if transactions:
            st.success(f"📊 Loaded {len(transactions)} transactions")
            
            # Analyze transaction data
            st.caption("🔍 Analyzing transaction data...")
            
            # Count by asset type
            asset_types = {}
            channels = {}
            missing_weeks = []
            
            for trans in transactions:
                # Asset type count
                asset_type = trans.get('asset_type', 'Unknown')
                asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
                
                # Channel count
                channel = trans.get('channel', 'Direct')
                channels[channel] = channels.get(channel, 0) + 1
                
                # Check week info
                if not trans.get('week_label'):
                    missing_weeks.append(trans)
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Transactions", len(transactions))
            with col2:
                st.metric("📅 Missing Week Info", len(missing_weeks))
            with col3:
                st.metric("🎯 Unique Channels", len(channels))
            
            # Asset type breakdown
            st.caption("📈 Asset Type Breakdown:")
            for asset_type, count in asset_types.items():
                st.caption(f"   • {asset_type}: {count} transactions")
            # Channel breakdown
            st.caption("📁 Channel Breakdown:")
            for channel, count in channels.items():
                st.caption(f"   • {channel}: {count} transactions")
            
            # Create display table
            st.caption("📋 Creating transaction table...")
            trans_data = []
            for trans in transactions:
                week_status = "✅" if trans.get('week_label') else "❌"
                trans_data.append({
                    'Ticker': trans['ticker'],
                    'Name': trans['stock_name'],
                    'Date': trans['transaction_date'],
                    'Week': trans.get('week_label', 'Not calculated'),
                    'Status': week_status,
                    'Type': trans['transaction_type'],
                    'Quantity': f"{trans['quantity']:,.0f}",
                    'Price': f"₹{trans['price']:,.2f}",
                    'Channel': trans.get('channel', 'Direct')
                })
            
            df_transactions = pd.DataFrame(trans_data)
            st.dataframe(df_transactions, use_container_width=True)
            
            # Week calculation status
            if missing_weeks:
                st.warning(f"⚠️ {len(missing_weeks)} transactions missing week information")
                #st.caption("🔧 To fix this, run: `streamlit run fix_week_calculation.py`")
                
                # Show which transactions are missing week info
                with st.expander("🔍 Transactions Missing Week Info"):
                    missing_data = []
                    for trans in missing_weeks:
                        missing_data.append({
                            'Ticker': trans['ticker'],
                            'Date': trans['transaction_date'],
                            'Type': trans['transaction_type'],
                            'Channel': trans.get('channel', 'Direct')
                        })
                    df_missing = pd.DataFrame(missing_data)
                    st.dataframe(df_missing, use_container_width=True)
            else:
                st.success("✅ All transactions have week information calculated!")
        else:
            st.info("No transactions found. Upload CSV files to see your transaction history.")
    
    except Exception as e:
        st.error(f"❌ Error loading transactions: {str(e)}")
        #st.caption("🔧 This might be a database connection issue. Check your Supabase connection.")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app function"""
    if st.session_state.user is None:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
