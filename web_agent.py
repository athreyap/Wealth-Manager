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
import csv
import importlib
import difflib
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

import requests
from dateutil import parser as dateutil_parser
warnings.filterwarnings('ignore')

# Add current directory to Python path for AI agents import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add the current working directory as a fallback
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Import AI agents for insights and recommendations
AI_AGENTS_AVAILABLE = False
AI_FILE_EXTRACTION_ENABLED = False  # DISABLED to avoid API quota issues - use direct CSV/Excel processing
AI_TICKER_RESOLUTION_ENABLED = True  # ENABLED - uses Gemini fallback (free/cheap)

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

try:
    _assistant_helper = importlib.import_module("assistant_helper")
    run_gpt5_completion = getattr(_assistant_helper, "run_gpt5_completion")
except Exception as exc:  # pragma: no cover - runtime dependency
    run_gpt5_completion = None  # type: ignore[assignment]
    AI_COMPLETION_IMPORT_ERROR = exc
else:
    AI_COMPLETION_IMPORT_ERROR = None

try:
    from ai_agents.agent_manager import get_agent_manager, run_ai_analysis, get_ai_recommendations, get_ai_alerts
    from ai_agents.ai_file_processor import AIFileProcessor
    AI_AGENTS_AVAILABLE = True
    import logging
    logging.info("✅ AI Agents imported successfully")
except ImportError as e:
    import logging
    logging.error(f"❌ AI Agents import failed: {e}")
    # Try to show more helpful error in Streamlit
    if 'ai_agents' in str(e):
        logging.error(f"   Current directory: {os.getcwd()}")
        logging.error(f"   ai_agents exists: {os.path.exists('ai_agents')}")
        logging.error(f"   ai_agents/__init__.py exists: {os.path.exists('ai_agents/__init__.py')}")
        logging.error(f"   Python path: {sys.path[:5]}")
        logging.error(f"   File directory: {os.path.dirname(os.path.abspath(__file__))}")
except Exception as e:
    import logging
    logging.error(f"❌ AI Agents error: {e}")

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

# AI Ticker Resolver not needed - using ai_resolve_tickers_from_names() function instead
TICKER_RESOLVER_AVAILABLE = False
AITickerResolver = None

# from weekly_manager_streamlined import StreamlinedWeeklyManager  # Removed - file deleted

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

# Function to detect corporate actions (splits/bonus)
def detect_corporate_actions(user_id, db):
    """
    Detect stock splits and bonus shares by comparing CSV prices with current prices
    Returns list of stocks with likely corporate actions
    """
    try:
        import yfinance as yf

        holdings = db.get_user_holdings(user_id)
        corporate_actions: List[Dict[str, Any]] = []

        @functools.lru_cache(maxsize=128)
        def _latest_split_info(symbol: str) -> Optional[Tuple[str, float]]:
            try:
                split_series = yf.Ticker(symbol).splits
                if split_series is not None and not split_series.empty:
                    last_ratio = float(split_series.iloc[-1])
                    if last_ratio > 0:
                        # yfinance reports 0.25 for 4:1 split etc.
                        resolved_ratio = round(1.0 / last_ratio)
                        return split_series.index[-1], resolved_ratio
            except Exception:
                pass
            return None

        def _find_split_confirmation(ticker_code: str) -> Optional[Tuple[str, float]]:
            candidates = []
            if ticker_code.endswith(('.NS', '.BO')):
                candidates.append(ticker_code)
            elif ticker_code.isdigit():
                candidates.append(f"{ticker_code}.BO")
            else:
                candidates.extend([f"{ticker_code}.NS", f"{ticker_code}.BO", ticker_code])

            for candidate in candidates:
                info = _latest_split_info(candidate)
                if info:
                    return info
            return None

        for holding in holdings:
            if holding.get('asset_type') != 'stock':
                continue

            ticker = str(holding.get('ticker') or '').strip()
            avg_price = float(holding.get('average_price') or 0)
            current_price = holding.get('current_price')
            quantity = holding.get('total_quantity', 0)

            if avg_price == 0 or not current_price:
                continue

            current_price = float(current_price)
            price_ratio = avg_price / current_price if current_price else 0

            if price_ratio < 1.5:
                continue

            confirmation = _find_split_confirmation(ticker)
            if not confirmation:
                continue

            split_date, confirmed_ratio = confirmation
            if confirmed_ratio <= 1:
                continue

            corporate_actions.append({
                'ticker': ticker,
                'stock_name': holding.get('stock_name'),
                'stock_id': holding.get('stock_id'),
                'avg_price': avg_price,
                'current_price': current_price,
                'quantity': quantity,
                'ratio': price_ratio,
                'split_ratio': confirmed_ratio,
                'split_date': str(split_date),
                'action_type': 'split',
            })
        
        if corporate_actions:
            print(f"[CORPORATE_ACTIONS] Detected {len(corporate_actions)} stocks with splits/bonus")
            for action in corporate_actions:
                print(f"  - {action['ticker']}: 1:{action['split_ratio']} {action['action_type']} on {action.get('split_date')}")
        
        return corporate_actions

    except Exception as e:
        print(f"[CORPORATE_ACTIONS] Error detecting: {e}")
        return []

def adjust_for_corporate_action(user_id, stock_id, split_ratio, db):
    """
    Adjust transaction quantities and prices for a stock split/bonus
    
    Args:
        user_id: User ID
        stock_id: Stock master ID
        split_ratio: Split ratio (e.g., 20 for 1:20 split)
        db: Database manager
    """
    try:
        # Get all transactions for this stock and user
        transactions = db.supabase.table('user_transactions').select('*').eq(
            'user_id', user_id
        ).eq('stock_id', stock_id).execute()
        
        if not transactions.data:
            return 0
        
        updated_count = 0
        for txn in transactions.data:
            old_quantity = float(txn['quantity'])
            old_price = float(txn['price'])
            
            # Adjust for split
            new_quantity = old_quantity * split_ratio
            new_price = old_price / split_ratio
            
            # Update transaction
            db.supabase.table('user_transactions').update({
                'quantity': new_quantity,
                'price': new_price,
                'notes': f"Auto-adjusted for 1:{split_ratio} stock split"
            }).eq('id', txn['id']).execute()
            
            updated_count += 1
        
        print(f"[CORPORATE_ACTIONS] Adjusted {updated_count} transactions for 1:{split_ratio} split")
        return updated_count
        
    except Exception as e:
        print(f"[CORPORATE_ACTIONS] Error adjusting: {e}")
        return 0

# Function to update bond prices using AI
def update_bond_prices_with_ai(user_id, db):
    """Update bond prices using AI (called automatically on login)"""
    try:
        # Get all bond holdings for this user
        all_holdings = db.get_user_holdings(user_id)
        bonds = [h for h in all_holdings if h.get('asset_type') == 'bond']
        
        if not bonds:
            return  # No bonds to update
        
        print(f"[BOND_UPDATE] Found {len(bonds)} bonds, fetching current prices...")
        
        from enhanced_price_fetcher import EnhancedPriceFetcher
        price_fetcher = EnhancedPriceFetcher()
        
        updated_count = 0
        for bond in bonds:
            ticker = bond.get('ticker')
            stock_name = bond.get('stock_name')
            
            if ticker and stock_name:
                print(f"[BOND_UPDATE] Fetching price for {stock_name} ({ticker})...")
                
                # Try to get bond price from AI
                price, source = price_fetcher._get_bond_price(ticker, stock_name)
                
                if price and price > 0:
                    # Update the price in database
                    db._store_current_price(ticker, price, 'bond')
                    print(f"[BOND_UPDATE] ✅ Updated {ticker}: ₹{price:.2f} (from {source})")
                    updated_count += 1
                else:
                    print(f"[BOND_UPDATE] ❌ Failed to get price for {ticker}")
        
        if updated_count > 0:
            print(f"[BOND_UPDATE] Successfully updated {updated_count}/{len(bonds)} bond prices")
        else:
            print(f"[BOND_UPDATE] No bond prices updated (all failed)")
    except Exception as e:
        print(f"[BOND_UPDATE] Error updating bond prices: {e}")
        pass  # Silent failure - don't break login

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

def should_update_prices_today(holdings, db_manager):
    """Check if prices need to be updated today"""
    from datetime import datetime, date
    
    today = date.today()
    needs_update = []
    
    if not db_manager:
        # No database manager, assume all need update
        return holdings
    
    for holding in holdings:
        stock_id = holding.get('stock_id')
        if stock_id:
            # Get last updated date from database
            try:
                if hasattr(db_manager, 'get_stock_last_updated'):
                    last_updated = db_manager.get_stock_last_updated(stock_id)
                    if last_updated:
                        try:
                            last_updated_date = datetime.fromisoformat(last_updated).date()
                            if last_updated_date < today:
                                needs_update.append(holding)
                        except (ValueError, TypeError):
                            # Invalid date format, needs update
                            needs_update.append(holding)
                    else:
                        # No last_updated record, needs update
                        needs_update.append(holding)
                else:
                    # Method not available, assume needs update
                    needs_update.append(holding)
            except Exception as e:
                # Any error, assume needs update (conservative approach)
                print(f"[PRICE_CHECK] Error checking last_updated for stock_id {stock_id}: {str(e)}")
                needs_update.append(holding)
        else:
            # No stock_id, needs update
            needs_update.append(holding)
    
    return needs_update

if 'bulk_ai_fetcher' not in st.session_state:
    st.session_state.bulk_ai_fetcher = BulkAIFetcher()
if 'weekly_manager' not in st.session_state:
    st.session_state.weekly_manager = None  # Simplified - removed StreamlinedWeeklyManager

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
                
                # Update bond prices automatically on login (AI-powered)
                try:
                    update_bond_prices_with_ai(user['id'], db)
                except:
                    pass  # Silent failure - don't break login
                
                # Recalculate holdings on login (ensures MFs show up)
                try:
                    db.recalculate_holdings(user['id'])
                except:
                    pass  # Silent failure
                
        # Fetch current prices for all holdings on login (only if not updated today OR if prices are stale)
                try:
                    holdings = db.get_user_holdings(user['id'])
                    if holdings and 'price_fetcher' in st.session_state:
                        # Check which holdings need price updates (not updated today)
                        holdings_needing_update = should_update_prices_today(holdings, db)
                        
                        # Also check for stale prices (current_price == average_price or missing)
                        stale_holdings = []
                        for h in holdings:
                            cp = h.get('current_price') or h.get('live_price') or 0
                            ap = h.get('average_price', 0)
                            if cp is None or cp == 0 or (ap > 0 and abs(cp - ap) < 0.01):
                                stale_holdings.append(h)
                        
                        # Combine both lists (remove duplicates)
                        all_needing_update = {h.get('stock_id'): h for h in holdings_needing_update}
                        for h in stale_holdings:
                            all_needing_update[h.get('stock_id')] = h
                        
                        holdings_to_fetch = list(all_needing_update.values())
                        
                        if holdings_to_fetch:
                            print(f"[LOGIN] Fetching current prices for {len(holdings_to_fetch)} holdings ({len(holdings_needing_update)} not updated today, {len(stale_holdings)} stale)...")
                            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_to_fetch, db)
                            print(f"[LOGIN] ✅ Price fetching complete")
                        else:
                            print(f"[LOGIN] ✅ All prices already updated today, skipping fetch")
                except Exception as e:
                    print(f"[LOGIN] ⚠️ Price fetching failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    pass  # Silent failure - don't break login

                # Fetch weekly historical prices on login if missing
                if holdings:
                    try:
                        missing_weeks = db.get_missing_weeks_for_user(user['id'])
                        if missing_weeks:
                            st.caption("📅 Fetching weekly historical prices in background...")
                            remaining = db.fetch_and_store_missing_weekly_prices(user['id'], missing_weeks)
                            fetched_count = len(missing_weeks) - len(remaining)
                            if fetched_count > 0:
                                st.caption(f"✅ Cached {fetched_count} weekly price records")
                    except Exception as exc:
                        print(f"[LOGIN] ⚠️ Weekly price fetch failed: {exc}")
                        pass
                
                # Detect corporate actions (splits/bonus) on login
                try:
                    corporate_actions = detect_corporate_actions(user['id'], db)
                    if corporate_actions:
                        st.session_state.corporate_actions_detected = corporate_actions
                    else:
                        st.session_state.corporate_actions_detected = None
                except:
                    st.session_state.corporate_actions_detected = None
                
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
        
        # File upload during registration
        st.subheader("Upload Transaction Files (Optional)")
        if AI_FILE_EXTRACTION_ENABLED:
            uploaded_files = st.file_uploader(
                "Upload transaction files (CSV, PDF, Excel, Images)",
                type=['csv', 'pdf', 'xlsx', 'xls', 'txt', 'jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload your transaction files in any format - AI will extract the data automatically"
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload CSV or Excel files",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload CSV or Excel files with transaction data. Format: date,ticker,quantity,transaction_type,price,stock_name,sector,channel"
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
                            imported_count = process_uploaded_files(uploaded_files, user['id'], portfolio_id)
                            
                            if imported_count > 0:
                                # Auto-fetch comprehensive data (info + prices + weekly) in bulk
                                st.info("🔍 Auto-fetching comprehensive data (info + prices + historical)...")
                                
                                try:
                                    holdings = db.get_user_holdings(user['id'])
                                    if holdings:
                                        # Get unique tickers and asset types
                                        unique_tickers = list(set([h['ticker'] for h in holdings if h.get('ticker')]))
                                        asset_types = {h['ticker']: h.get('asset_type', 'stock') for h in holdings if h.get('ticker')}
                                        
                                        if unique_tickers and st.session_state.bulk_ai_fetcher.available:
                                            st.caption("📊 Bulk fetching all data in one AI call...")
                                            # Fetch everything (stock info + current price + 52-week data) in ONE AI call
                                            stock_ids = db.bulk_process_new_stocks_with_comprehensive_data(
                                                tickers=unique_tickers,
                                                asset_types=asset_types
                                            )
                                            st.caption(f"✅ Fetched comprehensive data for {len(stock_ids)} tickers")
                                        else:
                                            # Fallback to individual updates
                                            st.caption("📊 Fetching prices individually...")
                                            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings, db)
                                            st.caption(f"✅ Updated {len(holdings)} holdings")
                                    
                                    st.success("✅ Registration, file processing, and comprehensive data fetching complete!")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Registration successful, but data fetching had issues: {str(e)[:100]}")
                                    st.success("✅ Registration and file processing complete!")
                            else:
                                st.success("✅ Registration and file processing complete!")
                        else:
                            st.success("✅ Registration successful!")
                        
                        st.info("🔄 Redirecting to dashboard...")
                        time.sleep(2)  # Brief pause to show success message
                        st.rerun()
                else:
                    st.error(f"Registration failed: {result['error']}")


def normalize_scheme_name(name: str) -> str:
    """Normalize mutual fund scheme names for comparison."""
    cleaned = name.lower().strip()
    replacements = [
        ("- regular plan - growth", ""),
        ("- regular plan growth", ""),
        ("regular plan - growth", ""),
        ("regular plan growth", ""),
        ("- growth", ""),
        ("growth", ""),
        (" plan", ""),
        (" (regular)", ""),
        (" direct plan", ""),
        (" direct growth", ""),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return "".join(ch for ch in cleaned if ch.isalnum())


@st.cache_data(ttl=3600)  # Cache AMFI download for 1 hour
def get_amfi_dataset() -> Dict[str, Any]:
    """Download AMFI NAV dataset and build lookup tables."""
    try:
        response = requests.get(AMFI_NAV_URL, timeout=60)
        response.raise_for_status()

        data = response.text.splitlines()
        reader = csv.DictReader(data, delimiter=';')

        schemes: List[Dict[str, str]] = []
        code_lookup: Dict[str, Dict[str, str]] = {}
        name_lookup: Dict[str, List[Dict[str, str]]] = {}

        for row in reader:
            scheme = {
                "code": (row.get("Scheme Code") or "").strip(),
                "name": (row.get("Scheme Name") or "").strip(),
                "nav": (row.get("Net Asset Value") or "").strip(),
                "date": (row.get("Date") or "").strip(),
            }
            if not scheme["code"] or not scheme["name"]:
                continue

            schemes.append(scheme)
            code_lookup[scheme["code"]] = scheme

            normalized = normalize_scheme_name(scheme["name"])
            if normalized:
                name_lookup.setdefault(normalized, []).append(scheme)

        return {
            "schemes": schemes,
            "code_lookup": code_lookup,
            "name_lookup": name_lookup,
        }
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AMFI dataset unavailable: {str(exc)[:80]}")
        return {"schemes": [], "code_lookup": {}, "name_lookup": {}}


def match_scheme_by_name(
    scheme_name: str,
    name_lookup: Dict[str, List[Dict[str, str]]],
    *,
    max_matches: int = 5,
) -> List[Dict[str, Any]]:
    """Return candidate AMFI schemes ranked by similarity."""
    normalized = normalize_scheme_name(scheme_name)
    if not normalized or not name_lookup:
        return []

    matches: List[Dict[str, Any]] = []
    if normalized in name_lookup:
        matches = [
            {
                "code": scheme["code"],
                "name": scheme["name"],
                "nav": scheme["nav"],
                "date": scheme["date"],
                "score": 1.0,
            }
            for scheme in name_lookup[normalized]
        ]
    else:
        candidates = difflib.get_close_matches(
            normalized,
            name_lookup.keys(),
            n=max_matches * 5,
            cutoff=0.6,
        )

        for candidate in candidates:
            base_score = difflib.SequenceMatcher(a=normalized, b=candidate).ratio()
            for scheme in name_lookup.get(candidate, []):
                matches.append(
                    {
                        "code": scheme["code"],
                        "name": scheme["name"],
                        "nav": scheme["nav"],
                        "date": scheme["date"],
                        "score": base_score,
                    }
                )

    if not matches:
        return []

    target_upper = (scheme_name or "").upper()

    def adjusted_score(entry: Dict[str, Any]) -> float:
        weight = entry.get("score", 0.0)
        name_upper = entry.get("name", "").upper()

        def tweak(keyword: str, bonus: float) -> None:
            nonlocal weight
            if keyword in target_upper:
                if keyword in name_upper:
                    weight += bonus
                else:
                    weight -= bonus

        tweak("DIRECT", 0.08)
        tweak("REGULAR", 0.05)
        tweak("GROWTH", 0.04)
        tweak("DIVIDEND", 0.04)
        tweak("IDCW", 0.04)

        # Keep score within a sensible band
        return max(0.0, min(weight, 1.2))

    for entry in matches:
        entry["adjusted_score"] = adjusted_score(entry)

    matches.sort(
        key=lambda item: (item.get("adjusted_score", 0.0), item.get("score", 0.0)),
        reverse=True,
    )

    return matches[:max_matches]


def resolve_mutual_fund_with_amfi(
    scheme_name: str,
    current_code: str,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve mutual fund against AMFI dataset using code and name heuristics."""
    code_lookup = dataset.get("code_lookup", {})
    name_lookup = dataset.get("name_lookup", {})

    result: Dict[str, Any] = {
        "status": "unresolved",
        "direct_scheme": None,
        "matches": [],
    }

    cleaned_code = str(current_code).strip()
    direct_scheme = code_lookup.get(cleaned_code)
    matches = match_scheme_by_name(scheme_name, name_lookup)

    result["matches"] = matches

    if direct_scheme:
        normalized_target = normalize_scheme_name(scheme_name) if scheme_name else ""
        normalized_direct = normalize_scheme_name(direct_scheme["name"]) if direct_scheme["name"] else ""
        similarity = (
            difflib.SequenceMatcher(a=normalized_target, b=normalized_direct).ratio()
            if normalized_target and normalized_direct
            else 0.0
        )
        result["direct_scheme"] = {**direct_scheme, "similarity": similarity}
        result["status"] = "direct" if similarity >= 0.9 else "direct_mismatch"
        return result

    if matches:
        result["status"] = "name_matches"

    return result


def _format_amfi_matches_for_prompt(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "None"

    lines = []
    for candidate in matches[:5]:
        lines.append(
            f"- code: {candidate['code']} | name: {candidate['name']} | NAV: {candidate['nav']} ({candidate['date']}) | score: {candidate['score']:.3f}"
        )
    return "\n".join(lines)


def _extract_json_block(text: str) -> Optional[str]:
    """Extract first JSON object or array from text."""
    if not text:
        return None
    pattern = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


def _parse_ai_amfi_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse AI response for AMFI code suggestions."""
    if not raw:
        return None

    candidates = []
    try:
        candidates.append(json.loads(raw))
    except json.JSONDecodeError:
        block = _extract_json_block(raw)
        if block:
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                pass

    for payload in candidates:
        if isinstance(payload, dict):
            if "code" in payload:
                return payload
            suggestions = payload.get("suggestions")
            if isinstance(suggestions, list) and suggestions:
                first = suggestions[0]
                if isinstance(first, dict) and "code" in first:
                    return first
        elif isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "code" in first:
                return first

    return None


def ai_select_amfi_code(
    scheme_name: str,
    user_code: str,
    matches: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask GPT to choose the best AMFI code from candidates."""
    if run_gpt5_completion is None:
        return None

    matches_prompt = _format_amfi_matches_for_prompt(matches)
    user_content = (
        "Determine the most likely AMFI mutual fund scheme code based on the user-provided name.\n"
        "Respond with JSON containing at minimum 'code' and 'confidence' (0-1). Include 'reason' if helpful.\n"
        "If none of the candidates are suitable, return an empty JSON object.\n\n"
        f"User scheme name: {scheme_name}\n"
        f"User provided code: {user_code}\n"
        f"Candidate matches:\n{matches_prompt}"
    )

    try:
        response = run_gpt5_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Indian mutual funds. Choose the correct AMFI scheme code "
                        "given candidate matches. Only output JSON."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=200,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AI AMFI suggestion failed: {str(exc)[:80]}")
        return None

    suggestion = _parse_ai_amfi_response(response)
    return suggestion


def ai_suggest_market_identifiers(
    name: str,
    user_ticker: str,
    *,
    asset_hint: Optional[str] = None,
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """Use GPT to propose market identifiers (ticker/AMFI/PMS/AIF) with type classification."""
    if run_gpt5_completion is None:
        return []

    hint_text = asset_hint or "unknown"
    system_message = {
        "role": "system",
        "content": (
            "You map Indian financial instruments to tickers/codes that work with finance APIs. "
            "Return ONLY JSON array. Each item must include ticker, instrument_type "
            "(stock, mutual_fund, bond, pms, aif), confidence (0-1), and source "
            "indicating which API/database to use (yfinance_nse, yfinance_bse, mftool, manual, isin). "
            "Ensure tickers are directly usable with the stated API."
        ),
    }
    user_message = {
        "role": "user",
        "content": (
            f"Instrument name: {name}\n"
            f"User ticker/code: {user_ticker}\n"
            f"Instrument type hint: {hint_text}\n"
            "Provide up to {max_suggestions} high-confidence identifiers.\n"
            "Rules:\n"
            "- Stocks/BSE listings: return NSE (.NS) or BSE (.BO) symbols that resolve on Yahoo Finance.\n"
            "- Mutual funds: return the 6-digit AMFI scheme code that works with the mftool library.\n"
            "- Bonds (incl. SGB): return the exchange symbol or ISIN that works with yfinance (e.g., SGBFEB32IV).\n"
            "- PMS: return SEBI registration code (INP...).\n"
            "- AIF: return the AIF registration code.\n"
            "- Exclude guesses that fail these rules.\n"
            "Respond with a JSON array, e.g.:\n"
            '[{\"ticker\": \"TIMEX.BO\", \"instrument_type\": \"stock\", \"source\": \"yfinance_bse\", \"confidence\": 0.82, \"notes\": \"BSE symbol\"}]'
        ).format(max_suggestions=max_suggestions),
    }

    try:
        response = run_gpt5_completion(
            messages=[system_message, user_message],
            temperature=0,
            max_tokens=400,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        st.caption(f"   ⚠️ AI identifier suggestion failed: {str(exc)[:80]}")
        return []

    suggestions: List[Dict[str, Any]] = []
    candidates: List[Any] = []
    try:
        candidates.append(json.loads(response))
    except json.JSONDecodeError:
        block = _extract_json_block(response)
        if block:
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                pass

    for payload in candidates:
        if isinstance(payload, dict):
            raw = payload.get("suggestions")
            if isinstance(raw, list):
                suggestions = [s for s in raw if isinstance(s, dict)]
                break
        elif isinstance(payload, list):
            suggestions = [item for item in payload if isinstance(item, dict)]
            break

    if suggestions:
        return suggestions[:max_suggestions]

    # Deterministic fallbacks when AI cannot help
    fallback_suggestions: List[Dict[str, Any]] = []
    ticker_clean = (user_ticker or "").strip()
    name_lower = (name or "").lower()
    hint_lower = hint_text.lower()

    if ticker_clean.isdigit() and len(ticker_clean) <= 6:
        fallback_suggestions.append({
            "ticker": f"{ticker_clean}.BO" if not ticker_clean.endswith(".BO") else ticker_clean,
            "instrument_type": "stock",
            "confidence": 0.6,
            "source": "yfinance_bse",
            "notes": "Numeric ticker mapped to BSE code",
        })

    if ("mutual_fund" in hint_lower) or ("fund" in name_lower):
        try:
            dataset = get_amfi_dataset()
            amfi_resolution = resolve_mutual_fund_with_amfi(name or ticker_clean, ticker_clean, dataset)

            candidate_code = None
            confidence = 0.0

            if amfi_resolution.get("direct_scheme"):
                candidate_code = amfi_resolution["direct_scheme"]["code"]
                confidence = 0.9 if amfi_resolution["status"] == "direct" else 0.75
            elif amfi_resolution.get("matches"):
                candidate_code = amfi_resolution["matches"][0]["code"]
                confidence = amfi_resolution["matches"][0].get("score", 0.7)

            if candidate_code:
                fallback_suggestions.append({
                    "ticker": candidate_code,
                    "instrument_type": "mutual_fund",
                    "confidence": min(1.0, confidence),
                    "source": "amfi_lookup",
                    "notes": "Resolved using AMFI dataset",
                })
        except Exception:
            pass

    if fallback_suggestions:
        return fallback_suggestions[:max_suggestions]

    return []


def _should_refine_identifier(
    original_ticker: str,
    resolved_data: Dict[str, Any],
) -> bool:
    verified_ticker = (resolved_data.get('ticker') or "").strip()
    asset_type = (resolved_data.get('asset_type') or "").lower()
    if not verified_ticker:
        return True
    if verified_ticker == original_ticker:
        return True
    if asset_type == 'stock' and not (verified_ticker.endswith('.NS') or verified_ticker.endswith('.BO')):
        return True
    if asset_type == 'mutual_fund' and not verified_ticker.isdigit():
        return True
    if asset_type == 'bond' and verified_ticker.isdigit():
        return True
    if asset_type in {'pms', 'aif'} and verified_ticker == original_ticker:
        return True
    return False


def refine_resolved_identifier_with_ai(
    original_ticker: str,
    resolved_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Refine ticker/type using GPT suggestions when deterministic mapping is weak."""
    if run_gpt5_completion is None:
        return resolved_data

    if not _should_refine_identifier(original_ticker, resolved_data):
        return resolved_data

    name = resolved_data.get('name') or original_ticker
    asset_hint = resolved_data.get('asset_type')
    suggestions = ai_suggest_market_identifiers(name, original_ticker, asset_hint=asset_hint)
    if not suggestions:
        return resolved_data

    for candidate in suggestions:
        ticker_candidate = (candidate.get('ticker') or "").strip()
        instrument_type = (candidate.get('instrument_type') or "").strip().lower()
        if not ticker_candidate or not instrument_type:
            continue

        resolved_data['ticker'] = ticker_candidate
        resolved_data['asset_type'] = instrument_type
        resolved_data['source'] = candidate.get('source', resolved_data.get('source', 'ai'))
        resolved_data['confidence'] = candidate.get('confidence')
        resolved_data['notes'] = candidate.get('notes')
        break

    return resolved_data


def search_mftool_for_amfi_code(scheme_name):
    """
    Search mftool database for AMFI code by scheme name
    More reliable than AI! Uses intelligent keyword matching.
    """
    try:
        from mftool import Mftool
        mf = Mftool()
        
        # Get all schemes
        schemes = mf.get_scheme_codes()
        
        # Extract important keywords (filter out common words)
        name_lower = scheme_name.lower()
        skip_words = {'fund', 'plan', 'option', 'scheme', 'the', 'and', 'of', '-'}
        keywords = [word for word in name_lower.split() if len(word) > 2 and word not in skip_words]
        
        # Search for best match
        best_matches = []
        for code, name in schemes.items():
            scheme_lower = name.lower()
            
            # Count matching keywords
            match_count = sum(1 for kw in keywords if kw in scheme_lower)
            
            # Calculate match percentage
            match_pct = match_count / len(keywords) if keywords else 0
            
            # Bonus points for exact company/fund house match
            company_match = 0
            if 'sbi' in name_lower and 'sbi' in scheme_lower:
                company_match = 1
            elif 'hdfc' in name_lower and 'hdfc' in scheme_lower:
                company_match = 1
            elif 'tata' in name_lower and 'tata' in scheme_lower:
                company_match = 1
            elif 'quant' in name_lower and 'quant' in scheme_lower:
                company_match = 1
            elif 'iifl' in name_lower and 'iifl' in scheme_lower:
                company_match = 1
            elif '360' in name_lower and '360' in scheme_lower:
                company_match = 1
            
            # Only consider if at least 70% keywords match OR company matches with 50%+ keywords
            if match_pct >= 0.7 or (company_match and match_pct >= 0.5):
                best_matches.append((code, name, match_count, match_pct, company_match))
        
        # Sort by: company match, then keyword count, then percentage
        best_matches.sort(key=lambda x: (x[4], x[2], x[3]), reverse=True)
        
        if best_matches:
            # Return best match
            code, name, count, pct, company = best_matches[0]
            return {
                'ticker': code,
                'name': name,
                'sector': 'Mutual Fund',
                'source': 'mftool',
                'match_confidence': pct
            }
        
        return None
        
    except Exception as e:
        return None

def ai_resolve_tickers_from_names(ticker_name_pairs):
    """
    Use AI to resolve tickers from stock/fund names
    Returns verified tickers that work with yfinance/mftool
    """
    try:
        import openai
        
        # Try OpenAI first
        api_key = st.secrets["api_keys"].get("openai") or st.secrets["api_keys"].get("open_ai")
        use_gemini = False
        
        if api_key:
            client = openai.OpenAI(api_key=api_key)
        else:
            use_gemini = True
        
        # Build prompt
        ticker_list = []
        for ticker, info in ticker_name_pairs.items():
            asset_type = info.get('asset_type', 'stock')
            name = info.get('name', ticker)
            ticker_list.append(f"- Ticker: {ticker}, Name: {name}, Type: {asset_type}")
        
        prompt = f"""For each holding below, provide the VERIFIED ticker/code that works with yfinance (for stocks) or mftool (for mutual funds).

HOLDINGS:
{chr(10).join(ticker_list)}

YOUR TASK:
1. For STOCKS: 
   - Use the STOCK NAME (not ticker) as primary identifier
   - Search for the correct NSE/BSE ticker based on the company NAME
   - Try NSE first (.NS suffix): Verify it works with yfinance
   - If NSE fails, try BSE (.BO suffix): Verify it works with yfinance
   - Return whichever exchange ticker actually works
   - Example: "ITC LTD" → Search online → Find "ITC.NS" → Verify with yfinance → Return ITC.NS
   - IMPORTANT: Some stocks are ONLY on BSE, not NSE!
   - CRITICAL: Use the NAME to find correct ticker, don't just add .NS to existing ticker
   
2. For MUTUAL_FUND:
   - **EACH FUND MUST HAVE A UNIQUE AMFI CODE** - DO NOT reuse codes!
   - Search AMFI website or Value Research for the EXACT scheme code
   - Find the exact AMFI scheme code (6-digit number like 120760, 101305, etc.)
   - Different funds ALWAYS have different codes
   - Verify it works with mftool in Python
   - CRITICAL: Return numeric AMFI code, not scheme name
   - DOUBLE CHECK: Make sure you're not returning the same code for different funds!

3. For PMS (Portfolio Management Service):
   - Use the PMS registration code (format: INP000001234)
   - If not found, create a unique identifier based on PMS name
   - Return as-is (no API to verify)

4. For AIF (Alternative Investment Fund):
   - Use the AIF registration code (format: AIF-CAT1-12345)
   - If not found, create identifier from AIF name
   - Return as-is (no API to verify)

5. For BONDS:
   - For Sovereign Gold Bonds (SGB): Use NSE ticker (e.g., SGBFEB32IV)
   - For other bonds: Use ISIN code or exchange ticker
   - Try yfinance verification if possible
   
6. Provide sector information based on the stock/fund/bond name

7. Indicate source: yfinance_nse, yfinance_bse, mftool, manual, or isin

Return ONLY this JSON format:
{{
  "ORIGINAL_TICKER_OR_NAME": {{
    "ticker": "VERIFIED_TICKER_OR_AMFI_CODE",
    "name": "Full Name",
    "sector": "Sector Name",
    "source": "yfinance_nse|yfinance_bse|mftool",
    "verified": true
  }}
}}

EXAMPLES:
{{
  "RELIANCE": {{
    "ticker": "RELIANCE.NS",
    "name": "Reliance Industries Limited",
    "sector": "Oil & Gas",
    "source": "yfinance_nse",
    "verified": true
  }},
  "IDEA": {{
    "ticker": "IDEA.NS",
    "name": "Vodafone Idea Limited",
    "sector": "Telecom",
    "source": "yfinance_nse",
    "verified": true
  }},
  "BEDMUTHA": {{
    "ticker": "BEDMUTHA.BO",
    "name": "Bedmutha Industries Limited",
    "sector": "Chemicals",
    "source": "yfinance_bse",
    "verified": true
  }},
  "SBI Gold Direct Plan Growth": {{
    "ticker": "101305",
    "name": "SBI Gold Direct Plan Growth",
    "sector": "Gold",
    "source": "mftool",
    "verified": true
  }},
  "HDFC ELSS Tax Saver Direct Plan Growth": {{
    "ticker": "100104",
    "name": "HDFC ELSS Tax Saver Direct Plan Growth", 
    "sector": "ELSS",
    "source": "mftool",
    "verified": true
  }},
  "Quant Tax Plan Direct Growth": {{
    "ticker": "120760",
    "name": "Quant Tax Plan Direct Growth", 
    "sector": "ELSS",
    "source": "mftool",
    "verified": true
  }},
  "IDFC Nifty 50 Index Direct Plan Growth": {{
    "ticker": "134997",
    "name": "IDFC Nifty 50 Index Direct Plan Growth", 
    "sector": "Index Fund",
    "source": "mftool",
    "verified": true
  }},
  "Tata Small Cap Fund Direct Growth": {{
    "ticker": "125497",
    "name": "Tata Small Cap Fund Direct Growth", 
    "sector": "Small Cap",
    "source": "mftool",
    "verified": true
  }},
  "2.50% Gold Bonds 2032 SR-IV": {{
    "ticker": "SGBFEB32IV",
    "name": "Sovereign Gold Bond 2032 Series IV",
    "sector": "Government Securities",
    "source": "yfinance_nse",
    "verified": true
  }},
  "Buoyant Opportunities Fund": {{
    "ticker": "INP000012345",
    "name": "Buoyant Opportunities Fund",
    "sector": "PMS",
    "source": "manual",
    "verified": true
  }},
  "Private Equity Fund XYZ": {{
    "ticker": "AIF-CAT1-12345",
    "name": "Private Equity Fund XYZ",
    "sector": "AIF",
    "source": "manual",
    "verified": true
  }}
}}

CRITICAL RULES:
- **USE THE STOCK NAME** as primary identifier, not the ticker! Search for the correct ticker based on company name.
- For stocks: Search company name → Find NSE/BSE ticker → Verify with yfinance → Return working ticker
- For mutual funds: 
  * **EVERY MUTUAL FUND HAS A UNIQUE AMFI CODE** - NEVER use the same code for different funds!
  * Search AMFI website, Value Research, or MoneyControl for the EXACT scheme code
  * Each fund name gets its OWN unique 6-digit code (like 101305, 100104, 120760, 134997, 125497)
  * Return numeric AMFI code, NOT scheme name
  * DOUBLE-CHECK: If you're returning the same code twice, YOU ARE WRONG!
- For PMS/AIF: Search name → Find registration code (INP/AIF format) → Return code
- For Bonds: Search name → Find ISIN or NSE ticker → Verify with yfinance
- Always verify ticker actually works with yfinance/mftool before returning
- If ticker already has .NS or .BO, verify it still works (don't assume it's correct)
- If you're unsure, prefer NSE for major stocks, BSE for smaller/regional stocks

IMPORTANT: Don't just add .NS to existing ticker! Use the NAME to search and find the correct working ticker!

**MUTUAL FUND WARNING**: Each mutual fund scheme has its own unique AMFI code. NEVER return duplicate codes. If you don't know the exact code, search for it online before responding!

**JSON FORMAT REQUIREMENTS:**
- Return ONLY valid JSON - no comments, no trailing commas, no extra text
- Each object must have ALL fields: ticker, name, sector, source, verified
- Ensure proper comma separation between objects
- Do not include any explanatory text before or after the JSON
- Test your JSON is valid before returning

Return ONLY the JSON object, nothing else."""

        # Try OpenAI first, fallback to Gemini
        ai_response = None
        
        if not use_gemini:
            try:
                st.caption(f"   🔄 Using OpenAI (gpt-5) for ticker resolution...")
                response = client.chat.completions.create(
                    model="gpt-5",  # GPT-5 for better accuracy and reasoning
                    messages=[
                        {"role": "system", "content": "You are a financial ticker verification expert with access to real-time data. For each ticker, search online databases and verify it works with yfinance or mftool APIs. Return ONLY valid JSON with unique tickers for each holding."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=2000
                    # Note: GPT-5 only supports default temperature (1)
                )
                ai_response = response.choices[0].message.content
                st.caption(f"   ✅ OpenAI response received")
            except Exception as e:
                error_msg = str(e)
                st.caption(f"   ❌ OpenAI error: {error_msg[:100]}")
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.caption(f"   ⚠️ OpenAI quota exceeded, trying Gemini...")
                    use_gemini = True
                else:
                    st.caption(f"   ⚠️ OpenAI failed, trying Gemini...")
                    use_gemini = True
        
        # Fallback to Gemini
        if use_gemini and ai_response is None:
            try:
                import google.generativeai as genai
                gemini_key = st.secrets["api_keys"].get("gemini_api_key")
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    full_prompt = f"""You are a financial ticker verification expert. Verify tickers for yfinance and mftool APIs. Return ONLY valid JSON.

{prompt}"""
                    
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    st.caption(f"   ✅ Using Gemini for ticker resolution")
            except Exception as e:
                st.caption(f"   ❌ Gemini also failed: {str(e)[:50]}")
                return {}

        if not ai_response:
            return {}
        
        # Extract JSON with better error handling
        import json
        import re
        
        response_text = ai_response.strip()
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            # Remove ```json or ``` at start
            response_text = re.sub(r'^```(?:json)?\s*\n', '', response_text)
            # Remove ``` at end
            response_text = re.sub(r'\n```\s*$', '', response_text)
            response_text = response_text.strip()
        
        # Find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as je:
                # Try to fix common JSON errors
                st.caption(f"   ⚠️ JSON parse error, attempting auto-fix...")
                
                # Fix 1: Remove trailing commas before } or ]
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                # Fix 2: Remove comments (// or /* */)
                fixed_json = re.sub(r'//.*$', '', fixed_json, flags=re.MULTILINE)
                fixed_json = re.sub(r'/\*.*?\*/', '', fixed_json, flags=re.DOTALL)
                
                # Fix 3: Fix unquoted keys
                fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                
                try:
                    parsed = json.loads(fixed_json)
                    st.caption(f"   ✅ Auto-fixed JSON successfully")
                    return parsed
                except:
                    # Show details about original error
                    st.caption(f"   ⚠️ JSON parse error at line {je.lineno}, column {je.colno}")
                    st.caption(f"   Error: {je.msg}")
                    # Show snippet of problematic JSON
                    lines = json_str.split('\n')
                    if je.lineno <= len(lines):
                        problem_line = lines[je.lineno - 1] if je.lineno > 0 else ""
                        st.caption(f"   Problem: {problem_line[:100]}")
                    return {}
        
        st.caption(f"   ⚠️ Could not find valid JSON in AI response")
        return {}
        
    except Exception as e:
        st.caption(f"   ⚠️ AI ticker resolution error: {str(e)[:100]}")
        import traceback
        st.caption(f"   Traceback: {traceback.format_exc()[:200]}")
        return {}

def process_uploaded_files(uploaded_files, user_id, portfolio_id):
    """
    Process uploaded files and store to DB using AI
    Supports CSV, PDF, Excel, Images, and any file format
    """
    total_imported = 0
    processing_log = []
    
    # Always show CSV processing message (AI is used only for ticker resolution)
    st.info(f"📊 Processing {len(uploaded_files)} file(s)...")
    
    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        st.caption(f"📁 [{file_idx}/{len(uploaded_files)}] Processing {uploaded_file.name}...")
        
        # Initialize error tracking variables at the start
        imported = 0
        skipped = 0
        errors = 0
        tickers_in_this_file = set()  # Track tickers from this file to avoid re-fetching old data
        
        try:
            # Direct CSV/Excel processing (AI file extraction is disabled)
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext in ['csv', 'xlsx', 'xls']:
                st.caption(f"   📊 Processing {file_ext.upper()} file directly (no AI needed)...")
                
                # Read file with pandas
                if file_ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:  # xlsx or xls
                    df = pd.read_excel(uploaded_file)
                
                st.caption(f"   ✅ Read {len(df)} rows from {file_ext.upper()}")
                
                # Helper functions to handle NaN/None values
                def safe_value(val, default=''):
                    """Convert pandas NaN to safe value"""
                    if pd.isna(val) or val is None:
                        return default
                    return val
                
                def safe_float(val, default=0.0):
                    """Convert to float, handling NaN/None"""
                    try:
                        if pd.isna(val) or val is None or val == '':
                            return default
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                # Detect dominant date format in CSV by sampling dates
                def detect_date_format(df, date_column='date'):
                    """
                    Detect the dominant date format in the CSV by sampling dates.
                    Returns 'dayfirst' (DD-MM-YYYY) or 'monthfirst' (MM-DD-YYYY) or None (ambiguous)
                    """
                    if date_column not in df.columns:
                        return None
                    
                    # Sample up to 50 non-null dates from the CSV
                    sample_dates = df[date_column].dropna().head(50).tolist()
                    if not sample_dates:
                        return None
                    
                    dayfirst_count = 0
                    monthfirst_count = 0
                    unambiguous_count = 0
                    
                    for date_str in sample_dates:
                        try:
                            date_str = str(date_str).strip()
                            # Remove time part if present
                            if ' AM' in date_str.upper() or ' PM' in date_str.upper():
                                date_str = date_str.split()[0]
                            
                            # Check if date has numeric parts separated by - or /
                            parts = date_str.replace('/', '-').replace('.', '-').split('-')
                            if len(parts) == 3:
                                try:
                                    first_part = int(parts[0])
                                    second_part = int(parts[1])
                                    
                                    # If first part > 12, it MUST be DD-MM-YYYY (dayfirst)
                                    if first_part > 12:
                                        dayfirst_count += 1
                                        unambiguous_count += 1
                                    # If second part > 12, it MUST be MM-DD-YYYY (monthfirst)
                                    elif second_part > 12:
                                        monthfirst_count += 1
                                        unambiguous_count += 1
                                    # If both <= 12, it's ambiguous - try both and see which makes sense
                                    else:
                                        # Try dayfirst
                                        dt1 = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                                        # Try monthfirst
                                        dt2 = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                                        
                                        if pd.notna(dt1) and pd.notna(dt2):
                                            # Both parse successfully - check which is more reasonable
                                            # Prefer dates that are not too far in the future or past
                                            from datetime import datetime
                                            today = datetime.now()
                                            diff1 = abs((dt1 - today).days)
                                            diff2 = abs((dt2 - today).days)
                                            
                                            # If one is clearly more reasonable (closer to today), use that
                                            if diff1 < diff2:
                                                dayfirst_count += 0.5  # Half weight for ambiguous
                                            elif diff2 < diff1:
                                                monthfirst_count += 0.5
                                except:
                                    pass
                        except:
                            pass
                    
                    # Determine dominant format
                    if unambiguous_count > 0:
                        # If we have unambiguous dates, trust them
                        if dayfirst_count > monthfirst_count:
                            return 'dayfirst'
                        elif monthfirst_count > dayfirst_count:
                            return 'monthfirst'
                    
                    # If no clear winner, default to dayfirst for Indian CSVs
                    return 'dayfirst'
                
                # Detect format once for this CSV
                detected_format = detect_date_format(df)
                if detected_format:
                    st.caption(f"   📅 Detected date format: {'DD-MM-YYYY' if detected_format == 'dayfirst' else 'MM-DD-YYYY'}")
                
                def normalize_date(date_str, preferred_format=None):
                    """Convert ANY date format to YYYY-MM-DD with robust format detection"""
                    try:
                        if pd.isna(date_str) or not date_str:
                            return ''
                        
                        date_str = str(date_str).strip()
                        
                        # Remove time part ONLY if AM/PM is present (e.g., "18-03-2021 09:34 AM" -> "18-03-2021")
                        # Preserve formats like "10 Oct 2025" (don't split these!)
                        if ' AM' in date_str.upper() or ' PM' in date_str.upper():
                            date_str = date_str.split()[0]
                        
                        # Try multiple parsing strategies for maximum compatibility
                        dt = None
                        
                        # First, check if date is unambiguous (can determine format from values)
                        parts = date_str.replace('/', '-').replace('.', '-').split('-')
                        is_ambiguous = False
                        if len(parts) == 3:
                            try:
                                first_part = int(parts[0])
                                second_part = int(parts[1])
                                # If first part > 12, it MUST be DD-MM-YYYY (unambiguous)
                                # If second part > 12, it MUST be MM-DD-YYYY (unambiguous)
                                # If both <= 12, it's ambiguous
                                is_ambiguous = (first_part <= 12 and second_part <= 12)
                            except:
                                pass
                        
                        # For unambiguous dates, use the correct format
                        # For ambiguous dates, use preferred_format (detected from CSV)
                        use_dayfirst = preferred_format == 'dayfirst' if preferred_format else True
                        
                        # Strategy 1: Try preferred format first (for ambiguous dates) or dayfirst (for unambiguous)
                        if not is_ambiguous or use_dayfirst:
                            dt = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                            if pd.notna(dt):
                                # Validate unambiguous dates
                                if len(parts) == 3:
                                    try:
                                        first_part = int(parts[0])
                                        if first_part > 12:
                                            return dt.strftime('%Y-%m-%d')  # Definitely DD-MM-YYYY
                                    except:
                                        pass
                                # For ambiguous dates, trust preferred format
                                if is_ambiguous and use_dayfirst:
                                    return dt.strftime('%Y-%m-%d')
                                elif not is_ambiguous:
                                    return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 2: Try opposite format (for ambiguous dates with different preference)
                        if is_ambiguous and not use_dayfirst:
                            dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            if pd.notna(dt):
                                return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 3: Try both formats and validate (fallback for ambiguous dates)
                        if is_ambiguous:
                            dt1 = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                            dt2 = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            
                            if pd.notna(dt1) and pd.notna(dt2):
                                # Both parse successfully - use preferred format
                                if use_dayfirst:
                                    return dt1.strftime('%Y-%m-%d')
                                else:
                                    return dt2.strftime('%Y-%m-%d')
                            elif pd.notna(dt1):
                                return dt1.strftime('%Y-%m-%d')
                            elif pd.notna(dt2):
                                return dt2.strftime('%Y-%m-%d')
                        
                        # Strategy 4: Try dayfirst=False for unambiguous dates (if Strategy 1 failed)
                        if not is_ambiguous:
                            dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
                            if pd.notna(dt):
                                # Validate
                                if len(parts) == 3:
                                    try:
                                        second_part = int(parts[1])
                                        if second_part > 12:
                                            return dt.strftime('%Y-%m-%d')  # Definitely MM-DD-YYYY
                                    except:
                                        pass
                                return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 5: Try infer_datetime_format (auto-detect)
                        dt = pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
                        if pd.notna(dt):
                            return dt.strftime('%Y-%m-%d')
                        
                        # Strategy 6: Try common explicit formats
                        common_formats = [
                            '%d-%m-%Y',      # DD-MM-YYYY
                            '%d/%m/%Y',      # DD/MM/YYYY
                            '%m-%d-%Y',      # MM-DD-YYYY
                            '%m/%d/%Y',      # MM/DD/YYYY
                            '%Y-%m-%d',      # YYYY-MM-DD
                            '%Y/%m/%d',      # YYYY/MM/DD
                            '%d-%m-%y',      # DD-MM-YY
                            '%d/%m/%y',      # DD/MM/YY
                            '%m-%d-%y',      # MM-DD-YY
                            '%m/%d/%y',      # MM/DD/YY
                            '%d %b %Y',      # DD Mon YYYY (e.g., "10 Oct 2025")
                            '%d %B %Y',      # DD Month YYYY (e.g., "10 October 2025")
                            '%b %d, %Y',     # Mon DD, YYYY (e.g., "Oct 10, 2025")
                            '%B %d, %Y',     # Month DD, YYYY (e.g., "October 10, 2025")
                            '%Y-%m-%d %H:%M:%S',  # With timestamp
                            '%d-%m-%Y %H:%M:%S',  # With timestamp
                        ]
                        
                        for fmt in common_formats:
                            try:
                                dt = pd.to_datetime(date_str, format=fmt, errors='coerce')
                                if pd.notna(dt):
                                    return dt.strftime('%Y-%m-%d')
                            except:
                                continue
                        
                        # Final fallback: try python-dateutil parser with different assumptions
                        for dayfirst_flag in (
                            preferred_format != 'monthfirst',
                            True,
                            False,
                        ):
                            try:
                                dt = dateutil_parser.parse(
                                    date_str,
                                    dayfirst=dayfirst_flag,
                                    fuzzy=True,
                                )
                                return dt.strftime('%Y-%m-%d')
                            except Exception:
                                continue

                        # Final resort: strip non-date characters and retry
                        cleaned = re.sub(r'[^0-9A-Za-z:/\\ -]', ' ', date_str)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                        for dayfirst_flag in (True, False):
                            try:
                                dt = dateutil_parser.parse(
                                    cleaned,
                                    dayfirst=dayfirst_flag,
                                    fuzzy=True,
                                )
                                return dt.strftime('%Y-%m-%d')
                            except Exception:
                                continue

                        # If all strategies failed, return empty
                        return ''
                    except Exception as e:
                        # Log error for debugging (but don't break the import)
                        print(f"[DATE_PARSE_ERROR] Failed to parse date '{date_str}': {str(e)}")
                        return ''
                    finally:
                        pass
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_rows = len(df)
                
                for idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        # Update progress every 50 rows
                        if idx % 50 == 0:
                            progress_bar.progress((idx + 1) / total_rows)
                            status_text.text(f"📝 Processing row {idx + 1}/{total_rows}... ({imported} imported, {skipped} skipped)")
                        
                        # Extract channel from filename if not provided
                        channel = safe_value(row.get('channel'), None)
                        if not channel:
                            # Use filename without extension
                            import os
                            channel = os.path.splitext(uploaded_file.name)[0]
                        
                        # Smart mapping - ticker can be NaN (AI will resolve later)
                        ticker = safe_value(row.get('ticker'), '')
                        stock_name = safe_value(row.get('stock_name'), '')
                        
                        # Clean ticker - remove $ and other special characters
                        ticker = str(ticker).replace('$', '').strip()
                        
                        # If ticker is NaN/empty, use stock_name as placeholder
                        # AI will resolve this to proper ticker later
                        if not ticker:
                            ticker = stock_name  # Temporary - AI will fix
                        
                        # Determine asset type
                        asset_type = safe_value(row.get('asset_type'), None)
                        if not asset_type:
                            # Auto-detect from stock name
                            name_lower = stock_name.lower()
                            if 'pms' in name_lower or 'portfolio management' in name_lower or ticker.startswith('INP'):
                                asset_type = 'pms'
                            elif 'aif' in name_lower or 'alternative investment' in name_lower or ticker.startswith('AIF'):
                                asset_type = 'aif'
                            elif 'bond' in name_lower or 'debenture' in name_lower or 'sgb' in name_lower:
                                asset_type = 'bond'
                            elif 'fund' in name_lower or 'scheme' in name_lower or 'growth' in name_lower:
                                asset_type = 'mutual_fund'
                            else:
                                asset_type = 'stock'
                        else:
                            asset_type = str(asset_type)
                        
                        # For PMS, AIF, Bonds - ignore CSV ticker, use stock_name as placeholder
                        # AI will fetch proper ticker/code later
                        if asset_type in ['pms', 'aif', 'bond']:
                            ticker = stock_name  # Force AI to resolve
                        
                        # SMART PRICE DETECTION (Priority Order):
                        # 1. If CSV has 'price' column → use it directly
                        # 2. If CSV has 'amount' column → calculate price = amount ÷ quantity
                        # 3. If CSV has only 'date' → fetch historical price for that date
                        # 4. If nothing works → price = 0 (will show as missing)
                        
                        csv_price = safe_float(row.get('price', 0), 0)
                        
                        # Try calculating from amount if price is missing (check both 'amount' and 'Amount')
                        if csv_price == 0:
                            # Check both lowercase and capital A (Groww uses different cases)
                            csv_amount = safe_float(row.get('amount', 0), 0)
                            if csv_amount == 0:
                                csv_amount = safe_float(row.get('Amount', 0), 0)
                            
                            csv_quantity = safe_float(row.get('quantity', 0), 0)
                            
                            if csv_amount > 0 and csv_quantity > 0:
                                csv_price = csv_amount / csv_quantity
                        
                        transaction_date = normalize_date(row['date'], preferred_format=detected_format)
                        
                        # Last resort: Fetch historical price for transaction date if price still missing
                        if csv_price == 0 and transaction_date:
                            try:
                                from enhanced_price_fetcher import EnhancedPriceFetcher
                                price_fetcher = EnhancedPriceFetcher()
                                
                                # Fetch historical price for transaction date
                                hist_price = price_fetcher.get_historical_price(
                                    ticker=ticker,
                                    asset_type=asset_type,
                                    date=transaction_date,
                                    fund_name=stock_name
                                )
                                
                                if hist_price and hist_price > 0:
                                    csv_price = hist_price
                                    st.caption(f"      📅 Fetched price for {ticker} on {transaction_date}: ₹{hist_price:.2f}")
                            except Exception as e:
                                pass  # Keep price as 0, will show in logs
                        
                        # Track ticker from this file
                        if ticker:
                            tickers_in_this_file.add(ticker)
                        
                        # Import with placeholder ticker (AI will resolve later)
                        transaction_data = {
                            'user_id': user_id,
                            'portfolio_id': portfolio_id,
                            'ticker': str(ticker),  # May be stock_name if ticker was NaN
                            'stock_name': str(stock_name),
                            'scheme_name': str(stock_name) if asset_type == 'mutual_fund' else None,
                            'quantity': safe_float(row['quantity'], 0),
                            'price': csv_price,  # Use fetched historical price if CSV price was missing
                            'transaction_date': transaction_date,
                            'transaction_type': str(safe_value(row['transaction_type'], 'buy')).lower(),
                            'asset_type': str(asset_type),
                            'channel': str(safe_value(channel, 'Direct')),
                            # For bonds, always set sector to "bond"
                            'sector': 'bond' if asset_type == 'bond' else str(safe_value(row.get('sector', 'Unknown'), 'Unknown')),
                            'filename': uploaded_file.name
                        }

                        result = db.add_transaction(transaction_data)
                        if result.get('success'):
                            imported += 1

                            # Add small delay every 50 transactions to avoid overwhelming database
                            if imported % 50 == 0:
                                import time
                                time.sleep(0.5)  # 500ms pause
                        else:
                            # Show why it failed
                            error_msg = result.get('error', 'Unknown error')
                            if 'duplicate' in error_msg.lower():
                                skipped += 1  # Count duplicates as skipped
                            elif '502' in error_msg or 'bad gateway' in error_msg.lower():
                                # 502 error - Supabase server issue
                                st.warning("   ⚠️ Database server error (502). Retrying in 2 seconds...")
                                import time
                                time.sleep(2)
                                # Retry once
                                retry_result = db.add_transaction(transaction_data)
                                if retry_result.get('success'):
                                    imported += 1
                                else:
                                    st.caption(f"   ⚠️ Retry failed: {retry_result.get('error', 'Unknown')[:100]}")
                                    errors += 1
                            else:
                                st.caption(f"   ⚠️ Failed: {error_msg[:100]}")
                                errors += 1
                    except Exception as e:
                        errors += 1
                        st.caption(f"   ❌ Error processing row {idx + 1}: {str(e)[:100]}")
                        continue
                
                # Clear progress bar
                progress_bar.empty()
                status_text.empty()
                
                # Show detailed import summary
                if file_ext in ['csv', 'xlsx', 'xls']:
                    if imported > 0:
                        st.success(f"   ✅ Imported {imported} transactions from {uploaded_file.name}")
                    if skipped > 0:
                        st.info(f"   ⏭️  Skipped {skipped} duplicate transactions")
                    if errors > 0:
                        st.warning(f"   ⚠️  {errors} errors encountered")
                    if imported == 0 and skipped == 0 and errors == 0:
                        st.error(f"   ❌ No transactions imported from {uploaded_file.name} - check CSV format!")
                
                # Log summary
                print(f"[CSV_IMPORT] File: {uploaded_file.name} | Rows read: {total_rows} | Imported: {imported} | Skipped: {skipped} | Errors: {errors}")
                
                # Use AI to resolve tickers based on stock names
                # Run even if imported=0 to fix existing data!
                if AI_TICKER_RESOLUTION_ENABLED:
                    st.caption(f"   🤖 Using AI to resolve tickers from stock names...")
                    
                    try:
                        # Get ALL transactions to fix tickers
                        all_transactions = db.get_user_transactions(user_id)
                        
                        # Filter: Get transactions that need ticker resolution
                        # Resolve ALL tickers to ensure they have correct .NS/.BO suffix and AMFI codes
                        file_transactions = []
                        for t in all_transactions:
                            ticker = t.get('ticker', '')
                            asset_type = t.get('asset_type', 'stock')
                            
                            needs_resolution = False
                            
                            # STOCKS: Resolve ALL to get proper .NS/.BO suffix
                            if asset_type == 'stock':
                                # Skip if already has .NS or .BO suffix
                                if not (ticker.endswith('.NS') or ticker.endswith('.BO')):
                                    needs_resolution = True
                            
                            # If ticker contains spaces, $, or is very long (scheme names)
                            if ' ' in ticker or '$' in ticker or len(ticker) > 50:
                                needs_resolution = True
                            
                            # PMS, AIF, Bonds ALWAYS need AI resolution to find proper codes
                            if asset_type in ['pms', 'aif', 'bond']:
                                needs_resolution = True
                            
                            # Mutual funds: SKIP AI resolution (AI doesn't have accurate AMFI codes)
                            # Will fetch prices using scheme_name with mftool later
                            if asset_type == 'mutual_fund':
                                needs_resolution = False  # Don't resolve MF with AI
                            
                            if needs_resolution:
                                file_transactions.append(t)
                        
                        st.caption(f"   Found {len(file_transactions)} transactions needing ticker resolution...")
                        
                        if file_transactions:
                            # Separate by asset type for different processing strategies
                            stocks_to_resolve = {}
                            others_to_resolve = {}
                            
                            for trans in file_transactions:
                                ticker = trans.get('ticker')
                                stock_name = trans.get('stock_name')
                                asset_type = trans.get('asset_type', 'stock')
                                
                                if ticker and stock_name:
                                    if asset_type == 'stock':
                                        # Stocks - will batch process
                                        stocks_to_resolve[ticker] = {
                                            'name': stock_name,
                                            'asset_type': asset_type
                                        }
                                    else:
                                        # PMS/AIF/Bonds - process one by one
                                        others_to_resolve[ticker] = {
                                            'name': stock_name,
                                            'asset_type': asset_type
                                        }
                            
                            all_resolved = {}
                            
                            # Process STOCKS in larger batches (10 at a time - AI is very accurate for stocks)
                            if stocks_to_resolve:
                                st.caption(f"   🔍 Resolving {len(stocks_to_resolve)} stock tickers (batch size: 10)...")
                                
                                batch_size = 10  # Larger batch for stocks
                                stock_items = list(stocks_to_resolve.items())
                                
                                for batch_start in range(0, len(stock_items), batch_size):
                                    batch_end = min(batch_start + batch_size, len(stock_items))
                                    batch = dict(stock_items[batch_start:batch_end])
                                    
                                    st.caption(f"      Stock batch {batch_start//batch_size + 1} ({len(batch)} tickers)...")
                                    
                                    # Use AI to get verified tickers and sectors
                                    batch_resolved = ai_resolve_tickers_from_names(batch)
                                    if batch_resolved:
                                        for original_key, info in batch.items():
                                            resolved_entry = batch_resolved.get(original_key)
                                            if not resolved_entry:
                                                continue
                                            resolved_entry.setdefault('name', info.get('name'))
                                            resolved_entry['asset_type'] = info.get('asset_type', resolved_entry.get('asset_type', 'stock'))
                                            batch_resolved[original_key] = refine_resolved_identifier_with_ai(original_key, resolved_entry)
                                    if batch_resolved:
                                        all_resolved.update(batch_resolved)
                            
                            # Process PMS/AIF/BONDS one by one (need more careful verification)
                            if others_to_resolve:
                                st.caption(f"   🔍 Resolving {len(others_to_resolve)} PMS/AIF/Bond tickers (one-by-one)...")
                                
                                for ticker, info in others_to_resolve.items():
                                    single_pair = {ticker: info}
                                    
                                    # Use AI to get verified ticker
                                    single_resolved = ai_resolve_tickers_from_names(single_pair)
                                    if single_resolved:
                                        resolved_entry = single_resolved.get(ticker)
                                        if resolved_entry:
                                            resolved_entry.setdefault('name', info.get('name'))
                                            resolved_entry['asset_type'] = info.get('asset_type', resolved_entry.get('asset_type'))
                                            single_resolved[ticker] = refine_resolved_identifier_with_ai(ticker, resolved_entry)
                                    if single_resolved:
                                        all_resolved.update(single_resolved)
                            
                            resolved_tickers = all_resolved
                            
                            if resolved_tickers:
                                st.caption(f"   ✅ AI resolved {len(resolved_tickers)} tickers with verified sources")
                                
                                # CRITICAL: Check for duplicate tickers (AI bug detection)
                                verified_ticker_list = [data.get('ticker') for data in resolved_tickers.values() if data.get('ticker')]
                                unique_verified = set(verified_ticker_list)
                                
                                if len(verified_ticker_list) != len(unique_verified):
                                    # DUPLICATES DETECTED - This is EXPECTED for same stocks with different CSV ticker variations
                                    st.info(f"   ℹ️ Found {len(verified_ticker_list) - len(unique_verified)} ticker merges (same stocks, different CSV tickers)")
                                    
                                    # Find which tickers are duplicated
                                    from collections import Counter
                                    ticker_counts = Counter(verified_ticker_list)
                                    duplicates = {ticker: count for ticker, count in ticker_counts.items() if count > 1}
                                    
                                    for dup_ticker, count in duplicates.items():
                                        # Show which CSV tickers will be merged
                                        affected = [orig for orig, data in resolved_tickers.items() if data.get('ticker') == dup_ticker]
                                        st.caption(f"      ✓ Merging: {', '.join(affected[:5])} → {dup_ticker}")
                                    
                                    st.caption("   ℹ️ These are legitimate merges (e.g., rights issues, bonus shares, name variations)")
                                else:
                                    # All tickers are unique
                                    st.caption(f"   ✅ All {len(unique_verified)} tickers are unique")
                                
                                # Proceed with updates (duplicates are valid merges!)
                                updated_count = 0
                                for original_ticker, resolved_data in resolved_tickers.items():
                                    resolved_data = refine_resolved_identifier_with_ai(original_ticker, resolved_data)
                                    verified_ticker = resolved_data.get('ticker')
                                    sector = resolved_data.get('sector', 'Unknown')
                                    source = resolved_data.get('source', 'ai')
                                    
                                    if verified_ticker:
                                        # Update stock_master record with verified ticker
                                        # Handle merges (when multiple CSV tickers resolve to same verified ticker)
                                        
                                        if verified_ticker != original_ticker:
                                            # Find old stock_master record with original ticker
                                            old_stock = db.supabase.table('stock_master').select('id, ticker, stock_name').eq(
                                                'ticker', original_ticker
                                            ).execute()
                                            
                                            if old_stock.data:
                                                old_stock_id = old_stock.data[0]['id']
                                                stock_name = resolved_data.get('name', old_stock.data[0].get('stock_name'))
                                                
                                                # Check if target ticker already exists
                                                existing_stock = db.supabase.table('stock_master').select('id').eq(
                                                    'ticker', verified_ticker
                                                ).eq('stock_name', stock_name).execute()
                                                
                                                if existing_stock.data:
                                                    # Target ticker already exists - MERGE!
                                                    # Point transactions to existing stock and delete old record
                                                    target_stock_id = existing_stock.data[0]['id']
                                                    
                                                    # Update all transactions to point to existing stock
                                                    db.supabase.table('user_transactions').update({
                                                        'stock_id': target_stock_id
                                                    }).eq('stock_id', old_stock_id).execute()
                                                    
                                                    # Delete old stock_master record
                                                    db.supabase.table('stock_master').delete().eq('id', old_stock_id).execute()
                                                    
                                                    st.caption(f"      ✓ Merged: {original_ticker} → {verified_ticker}")
                                                    updated_count += 1
                                                else:
                                                    # Target doesn't exist - safe to update
                                                    db.supabase.table('stock_master').update({
                                                        'ticker': verified_ticker,
                                                        'sector': sector
                                                    }).eq('id', old_stock_id).execute()
                                                    
                                                    st.caption(f"      ✓ Updated: {original_ticker} → {verified_ticker}")
                                                    updated_count += 1
                                            else:
                                                # Create new stock_master record with verified ticker
                                                # Get asset_type from original groups
                                                asset_type = resolved_data.get('asset_type') or \
                                                    stocks_to_resolve.get(original_ticker, {}).get('asset_type') or \
                                                    others_to_resolve.get(original_ticker, {}).get('asset_type', 'stock')
                                                
                                                # For bonds, always set sector to "bond"
                                                if asset_type == 'bond':
                                                    sector = 'bond'
                                                
                                                new_stock_id = db.get_or_create_stock(
                                                    ticker=verified_ticker,
                                                    stock_name=resolved_data.get('name', original_ticker),
                                                    asset_type=asset_type,
                                                    sector=sector
                                                )
                                                
                                                st.caption(f"      ✓ Created: {verified_ticker}")
                                                updated_count += 1
                                        else:
                                            # Just update sector if ticker is same
                                            update_payload = {'sector': sector}
                                            if resolved_data.get('asset_type'):
                                                update_payload['asset_type'] = resolved_data['asset_type']
                                            db.supabase.table('stock_master').update(update_payload).eq('ticker', verified_ticker).execute()
                                            updated_count += 1
                                
                                if updated_count > 0:
                                    st.caption(f"   ✅ Updated {updated_count} transactions with verified tickers")
                    
                    except Exception as e:
                        st.caption(f"   ⚠️ AI ticker resolution skipped: {str(e)[:100]}")
                
                # Resolve MUTUAL FUNDS using mftool search (more reliable than AI)
                st.caption(f"   🔍 Resolving mutual fund AMFI codes using mftool search...")
                try:
                    # Get all mutual fund transactions
                    mf_transactions = [t for t in db.get_user_transactions(user_id) if t.get('asset_type') == 'mutual_fund']
                    
                    if mf_transactions:
                        mf_updated = 0
                        amfi_dataset = get_amfi_dataset()
                        code_lookup = amfi_dataset.get('code_lookup', {})
                        for trans in mf_transactions:
                            scheme_name = trans.get('stock_name', '') or ''
                            current_ticker = str(trans.get('ticker', '') or '').strip()

                            final_code = None
                            final_name = None
                            final_source = None
                            ai_confidence = None

                            resolution = resolve_mutual_fund_with_amfi(scheme_name, current_ticker, amfi_dataset)
                            status = resolution.get('status')
                            direct_scheme = resolution.get('direct_scheme')
                            matches = resolution.get('matches', [])

                            if status == 'direct' and direct_scheme:
                                final_code = direct_scheme['code']
                                final_name = direct_scheme['name']
                                final_source = 'amfi_direct'
                            elif status in {'direct_mismatch', 'name_matches'}:
                                top_match = matches[0] if matches else None
                                if top_match and top_match.get('score', 0) >= 0.92:
                                    final_code = top_match['code']
                                    final_name = top_match['name']
                                    final_source = 'amfi_name_match'
                                else:
                                    ai_choice = ai_select_amfi_code(scheme_name, current_ticker, matches)
                                    if ai_choice and ai_choice.get('code'):
                                        candidate_code = str(ai_choice['code']).strip()
                                        scheme = code_lookup.get(candidate_code)
                                        if scheme:
                                            final_code = candidate_code
                                            final_name = scheme['name']
                                            final_source = 'ai_amfi'
                                            ai_confidence = ai_choice.get('confidence')
                                    elif status == 'direct_mismatch' and direct_scheme:
                                        final_code = direct_scheme['code']
                                        final_name = direct_scheme['name']
                                        final_source = 'amfi_direct_mismatch'
                            else:
                                if matches:
                                    ai_choice = ai_select_amfi_code(scheme_name, current_ticker, matches)
                                    if ai_choice and ai_choice.get('code'):
                                        candidate_code = str(ai_choice['code']).strip()
                                        scheme = code_lookup.get(candidate_code)
                                        if scheme:
                                            final_code = candidate_code
                                            final_name = scheme['name']
                                            final_source = 'ai_amfi'
                                            ai_confidence = ai_choice.get('confidence')

                            if not final_code:
                                result = search_mftool_for_amfi_code(scheme_name)
                                if result:
                                    final_code = result['ticker']
                                    final_name = result['name']
                                    final_source = 'mftool'
                                    ai_confidence = result.get('match_confidence')

                            if not final_code:
                                continue

                            if final_code == current_ticker and final_source != 'amfi_direct_mismatch':
                                continue

                            old_stock = db.supabase.table('stock_master').select('id').eq(
                                'ticker', current_ticker
                            ).execute()

                            confidence_note = ""
                            if isinstance(ai_confidence, (int, float)):
                                confidence_note = f" ({ai_confidence:.0%})"

                            if final_code == current_ticker:
                                if old_stock.data and final_name:
                                    db.supabase.table('stock_master').update({
                                        'stock_name': final_name,
                                        'sector': 'Mutual Fund'
                                    }).eq('id', old_stock.data[0]['id']).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} name harmonized [{final_source or 'amfi'}{confidence_note}]"
                                    )
                                    mf_updated += 1
                                continue

                            existing_stock = db.supabase.table('stock_master').select('id').eq(
                                'ticker', final_code
                            ).execute()

                            if existing_stock.data:
                                target_stock_id = existing_stock.data[0]['id']
                                if old_stock.data:
                                    old_stock_id = old_stock.data[0]['id']
                                    db.supabase.table('user_transactions').update({
                                        'stock_id': target_stock_id
                                    }).eq('stock_id', old_stock_id).execute()
                                    db.supabase.table('stock_master').delete().eq('id', old_stock_id).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} → {final_code} [{final_source or 'amfi'}{confidence_note}] (merged)"
                                    )
                                    mf_updated += 1
                            else:
                                if old_stock.data:
                                    db.supabase.table('stock_master').update({
                                        'ticker': final_code,
                                        'stock_name': final_name or scheme_name,
                                        'sector': 'Mutual Fund'
                                    }).eq('id', old_stock.data[0]['id']).execute()
                                    st.caption(
                                        f"      ✓ MF: {scheme_name[:40]} → {final_code} [{final_source or 'amfi'}{confidence_note}]"
                                    )
                                    mf_updated += 1

                        if mf_updated > 0:
                            st.caption(f"   ✅ Updated {mf_updated} mutual fund AMFI codes")
                    
                except Exception as e:
                    st.caption(f"   ⚠️ Mutual fund resolution skipped: {str(e)[:100]}")
                
                # Recalculate holdings from transactions (CRITICAL for MFs to show up!)
                if imported > 0:
                    st.caption(f"   📊 Recalculating holdings from transactions...")
                    try:
                        holdings_count = db.recalculate_holdings(user_id, portfolio_id)
                        st.caption(f"   ✅ Calculated {holdings_count} holdings")
                    except Exception as e:
                        st.caption(f"   ⚠️ Holdings recalculation skipped: {str(e)[:100]}")
                
                # Auto-fetch prices AND 52-week historical data (only for tickers in this file)
                if imported > 0 and tickers_in_this_file:
                    st.caption(f"   📊 Fetching current prices and 52-week historical data for {len(tickers_in_this_file)} ticker(s) from this file...")
                    
                    try:
                        # Get asset types for tickers in this file
                        new_asset_types = {}
                        holdings = db.get_user_holdings(user_id)
                        if holdings:
                            # Map tickers to asset types from holdings
                            for h in holdings:
                                ticker = h.get('ticker')
                                if ticker and ticker in tickers_in_this_file:
                                    new_asset_types[ticker] = h.get('asset_type', 'stock')
                        
                        # If some tickers don't have asset types yet, default to 'stock'
                        for ticker in tickers_in_this_file:
                            if ticker not in new_asset_types:
                                new_asset_types[ticker] = 'stock'
                        
                        if tickers_in_this_file:
                            new_tickers_list = list(tickers_in_this_file)
                            st.caption(f"      📈 Fetching comprehensive data for {len(new_tickers_list)} ticker(s)...")
                            
                            # Fetch current prices + 52-week historical data (only for tickers in this file)
                            db.bulk_process_new_stocks_with_comprehensive_data(
                                tickers=new_tickers_list,
                                asset_types=new_asset_types
                            )
                            
                            st.caption(f"   ✅ Updated current prices and 52-week data for {len(new_tickers_list)} ticker(s)")
                        else:
                            st.caption(f"   ℹ️  No tickers to fetch")
                            
                    except Exception as e:
                        st.caption(f"   ⚠️ Price/historical fetching skipped: {str(e)[:50]}")
                elif imported > 0:
                    st.caption(f"   ℹ️  No tickers identified in this file (skipping price fetch)")
                
                    processing_log.append({
                'file': uploaded_file.name,
                'imported': imported,
                    'skipped': 0,
                    'errors': len(df) - imported
                    })
                    
                    total_imported += imported
                    
                else:
                # Non-CSV/Excel files require AI
                    st.caption(f"   ⚠️ Only CSV/Excel files supported in direct mode. {uploaded_file.name} requires AI extraction.")
                processing_log.append({
                    'file': uploaded_file.name,
                    'imported': 0,
                    'skipped': 0,
                    'errors': 1
                })
        
        except Exception as e:
            st.caption(f"   ❌ Error processing {uploaded_file.name}: {e}")
            processing_log.append({
                'file': uploaded_file.name,
                'imported': 0,
                'skipped': 0,
                'errors': 1
            })
    
    # Final holdings recalculation AFTER all files processed (ensures MFs + stocks are combined)
    if total_imported > 0:
        st.caption(f"📊 Final holdings recalculation (combining all assets)...")
        try:
            final_holdings_count = db.recalculate_holdings(user_id, portfolio_id)
            st.caption(f"✅ Total portfolio: {final_holdings_count} unique holdings")
        except Exception as e:
            st.caption(f"⚠️ Final recalculation skipped: {str(e)[:100]}")
    
    # Show summary
    if processing_log:
        st.success(f"✅ Processing complete! Imported {total_imported} transactions from {len(uploaded_files)} files.")
        
        # Show detailed log
        with st.expander("📋 Processing Details", expanded=False):
            for log in processing_log:
                st.caption(f"📄 {log['file']}: {log['imported']} imported, {log['skipped']} skipped, {log['errors']} errors")
    
    return total_imported

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
                
                # Update bond prices silently (AI-powered)
                try:
                    update_bond_prices_with_ai(user['id'], db)
                except:
                    pass  # Silent failure
                
                # Simulate progress steps
                steps = [
                    "🔍 Analyzing user holdings and transaction weeks...",
                    "💰 Updating bond prices via AI...",
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
                
        # Auto-fetch comprehensive data for all holdings (only if needed - not on every login)
            holdings = db.get_user_holdings(user['id'])
            if holdings:
                try:
                    # Check which holdings need price updates (not updated today)
                    holdings_needing_update = should_update_prices_today(holdings, db)
                    
                    if holdings_needing_update and bulk_ai_fetcher.available:
                        # Get unique tickers that need updating
                        unique_tickers = list(set([h['ticker'] for h in holdings_needing_update if h.get('ticker')]))
                        
                        if unique_tickers:
                            # Use bulk comprehensive fetching for holdings that need update
                            asset_types = {h['ticker']: h.get('asset_type', 'stock') for h in holdings_needing_update if h.get('ticker')}
                            stock_ids = db.bulk_process_new_stocks_with_comprehensive_data(
                                tickers=unique_tickers,
                                asset_types=asset_types
                            )
                            # All data (stock info, current prices, weekly prices) fetched in bulk!
                    elif holdings_needing_update:
                        # Fallback to individual price updates
                        st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
                    # else: All prices are current, skip fetching
                    
                except Exception as e:
                    # Silent fallback
                    try:
                        holdings_needing_update = should_update_prices_today(holdings, db)
                        if holdings_needing_update:
                            st.session_state.price_fetcher.update_live_prices_for_holdings(holdings_needing_update, db)
                    except:
                        pass
            
                result = {'success': True, 'fetched': len(holdings_needing_update) if holdings_needing_update else 0}
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
    navigation_options = [
        "🏠 Portfolio Overview",
        "📊 P&L Analysis",
        "📡 Channel Analytics",
        "📈 Charts & Analytics",
        "🤖 AI Assistant",
        "📁 Upload More Files"
    ]
    
    # Add AI Insights page if agents are available
    if AI_AGENTS_AVAILABLE:
        navigation_options.insert(-1, "🧠 AI Insights")
    
    # Add User Profile page
    navigation_options.insert(-1, "👤 Profile Settings")
    
    page = st.sidebar.radio(
        "Choose a page:",
        navigation_options
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
    elif page == "📡 Channel Analytics":
        channel_analytics_page()
    elif page == "📈 Charts & Analytics":
        charts_page()
    elif page == "🤖 AI Assistant":
        ai_assistant_page()
    elif page == "🧠 AI Insights" and AI_AGENTS_AVAILABLE:
        ai_insights_page()
    elif page == "👤 Profile Settings":
        user_profile_page()
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
    
    # Show Corporate Actions Alert (Stock Splits/Bonus)
    if 'corporate_actions_detected' in st.session_state and st.session_state.corporate_actions_detected:
        corporate_actions = st.session_state.corporate_actions_detected
        
        st.warning(f"📊 **{len(corporate_actions)} Stock Splits/Bonus Shares Detected!**")
        
        with st.expander(f"🔧 View and Fix Corporate Actions ({len(corporate_actions)} stocks)", expanded=True):
            st.markdown("""
            **Corporate actions detected!** Your portfolio has stocks that underwent splits or bonus issues.  
            Click the "Fix" button to automatically adjust quantities and prices.
            """)
            
            # Create a table of detected corporate actions
            for idx, action in enumerate(corporate_actions):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"**{action['stock_name']}** (`{action['ticker']}`)")
                
                with col2:
                    st.caption(f"Your Avg: ₹{action['avg_price']:,.2f}")
                    st.caption(f"Current: ₹{action['current_price']:,.2f}")
                
                with col3:
                    st.info(f"**1:{action['split_ratio']} Split**")
                    st.caption(f"({action['ratio']:.1f}x difference)")
                
                with col4:
                    if st.button(f"✅ Fix", key=f"fix_split_{idx}_{action['ticker']}"):
                        with st.spinner(f"Adjusting {action['ticker']}..."):
                            adjusted = adjust_for_corporate_action(
                                user['id'], 
                                action['stock_id'], 
                                action['split_ratio'],
                                db
                            )
                            
                            if adjusted > 0:
                                st.success(f"✅ Adjusted {adjusted} transactions for {action['ticker']}")
                                # Clear from session state
                                st.session_state.corporate_actions_detected = [
                                    a for a in corporate_actions if a['ticker'] != action['ticker']
                                ]
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to adjust {action['ticker']}")
            
            # Add "Fix All" button
            st.markdown("---")
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button("✅ Fix All Splits", type="primary", use_container_width=True):
                    with st.spinner("Adjusting all stocks..."):
                        total_adjusted = 0
                        for action in corporate_actions:
                            adjusted = adjust_for_corporate_action(
                                user['id'],
                                action['stock_id'],
                                action['split_ratio'],
                                db
                            )
                            total_adjusted += adjusted
                        
                        if total_adjusted > 0:
                            st.success(f"✅ Adjusted {total_adjusted} total transactions across {len(corporate_actions)} stocks!")
                            st.session_state.corporate_actions_detected = None
                            time.sleep(2)
                            st.rerun()
            
            with col_b:
                if st.button("❌ Dismiss", use_container_width=True):
                    st.session_state.corporate_actions_detected = None
                    st.rerun()
        
        st.markdown("---")
    
    # Add AI-powered proactive alerts if available
    if AI_AGENTS_AVAILABLE:
        try:
            alerts = get_ai_alerts()
            if alerts:
                st.markdown("### 🚨 AI Alerts")
                for alert in alerts[:3]:  # Show top 3 alerts
                    severity_emoji = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢"
                    }.get(alert.get("severity", "low"), "🟢")
                    
                    with st.expander(f"{severity_emoji} {alert.get('title', 'Alert')}", expanded=(alert.get("severity") == "high")):
                        st.markdown(f"**{alert.get('description', 'No description')}**")
                        st.markdown(f"*Recommendation: {alert.get('recommendation', 'No recommendation')}*")
                st.markdown("---")
        except Exception as e:
            # Silently handle errors to not disrupt the main page
            pass
    
    # Add smart manual update prices button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        # Check how many holdings need updating
        holdings = get_cached_holdings(user['id'])
        
        # Find ALL holdings with stale prices (current_price == avg_price, 0% return, or missing)
        stale_prices = []
        bonds_to_update = []
        for h in holdings:
            # Check both current_price and live_price fields
            cp = h.get('current_price') or h.get('live_price') or 0
            ap = h.get('average_price', 0)
            asset_type = h.get('asset_type', '')
            
            # If current_price is None, 0, or exactly equals avg_price → likely stale (for ALL asset types)
            is_stale = False
            
            if cp is None or cp == 0:
                # Missing price - definitely stale
                is_stale = True
            elif ap > 0 and abs(cp - ap) < 0.01:
                # Current price equals average (within 1 paisa) = stale (price wasn't fetched)
                is_stale = True
            
            # Special handling for bonds - check if price seems wrong
            if asset_type == 'bond':
                ticker = (h.get('ticker') or '').lower()
                name = (h.get('stock_name') or '').lower()
                is_sgb = any(k in ticker or k in name for k in ['sgb', 'gold bond', 'goldbond', 'sovereign', 'sr-'])
                
                # SGBs should be around ₹12,000-15,000, not ₹6,000
                if is_sgb and cp < 10000:
                    bonds_to_update.append(h)
                    is_stale = True
                elif cp is None or cp == 0 or (ap > 0 and cp < ap * 0.5):  # Price is < 50% of avg = likely wrong
                    bonds_to_update.append(h)
                    is_stale = True
                elif ap > 0 and cp < ap * 0.7:  # Current price < 70% of average = likely wrong
                    bonds_to_update.append(h)
                    is_stale = True
            
            # Add to stale_prices if stale (for ALL asset types)
            if is_stale:
                stale_prices.append(h)
    
        # Show "Refresh All Prices" button if there are stale prices
        if stale_prices:
            if st.button(f"🔄 Refresh {len(stale_prices)} Price(s)", help=f"Refresh prices for holdings with missing or stale prices"):
                with st.spinner(f"Refreshing prices for {len(stale_prices)} holdings..."):
                    from enhanced_price_fetcher import EnhancedPriceFetcher
                    price_fetcher = EnhancedPriceFetcher()
                    price_fetcher.update_live_prices_for_holdings(stale_prices, db)
                    st.success(f"✅ Refreshed prices for {len(stale_prices)} holdings!")
                    st.rerun()
        
        # Show bond update button if bonds need updating
        if bonds_to_update:
            if st.button(f"💰 Update {len(bonds_to_update)} Bond Price(s)", help=f"Update bond prices using AI (SGBs need market prices)"):
                with st.spinner(f"Fetching bond prices for {len(bonds_to_update)} bonds..."):
                    from enhanced_price_fetcher import EnhancedPriceFetcher
                    price_fetcher = EnhancedPriceFetcher()
                    
                    updated = 0
                    for bond in bonds_to_update:
                        ticker = bond.get('ticker')
                        stock_name = bond.get('stock_name')
                        
                        print(f"[BOND_UPDATE] Manual update: {stock_name} ({ticker})")
                        price, source = price_fetcher._get_bond_price(ticker, stock_name)
                        
                        if price and price > 0:
                            db._store_current_price(ticker, price, 'bond')
                            print(f"[BOND_UPDATE] ✅ Updated {ticker}: ₹{price:.2f} (from {source})")
                            updated += 1
                        else:
                            print(f"[BOND_UPDATE] ❌ Failed to get price for {ticker}")
                    
                    if updated > 0:
                        st.success(f"✅ Updated {updated}/{len(bonds_to_update)} bond price(s)!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No bond prices updated (AI may be unavailable)")
                        # Don't rerun if update failed - prevent infinite loop
    
    # Show general "Refresh All Prices" button
    with col3:
            if st.button("🔄 Refresh All Prices", help="Refresh current prices for all holdings"):
                with st.spinner("Refreshing all prices (this may take a minute)..."):
                    holdings = get_cached_holdings(user['id'])
                    if holdings:
                        from enhanced_price_fetcher import EnhancedPriceFetcher
                        price_fetcher = EnhancedPriceFetcher()
                        price_fetcher.update_live_prices_for_holdings(holdings, db)
                        st.success(f"✅ Refreshed prices for {len(holdings)} holdings!")
                        st.rerun()
                    else:
                        st.warning("No holdings found to update.")
    
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
        
        # Get current price - handle None values (check both current_price and live_price)
        current_price = holding.get('current_price') or holding.get('live_price')
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
        # Handle None current_price - check both current_price and live_price fields
        current_price = holding.get('current_price') or holding.get('live_price')
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

                if len(df_navs) > 52:
                    df_navs = df_navs.tail(52)
                
                fig_nav = px.line(
                    df_navs, 
                    x='date', 
                    y='price',
                    title=f"{selected_ticker_nav} - 52-Week NAVs"
                )
                fig_nav.update_layout(xaxis_title="Date", yaxis_title="NAV (₹)")
                st.plotly_chart(fig_nav, use_container_width=True)
                
                st.subheader(f"{selected_ticker_nav} - NAV History")
                df_display = df_navs[['date', 'price', 'iso_week', 'iso_year']].copy()
                df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
                df_display.columns = ['Date', 'NAV', 'Week', 'Year']
                st.dataframe(df_display.tail(20), use_container_width=True)
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
                                model="gpt-5",  # Upgraded to GPT-5 for better results
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a professional technical analyst. Provide a brief technical analysis "
                                            "summary based on the indicators provided. Focus on key signals and trading "
                                            "implications. Use emojis and be concise."
                                        )
                                    },
                                    {"role": "user", "content": tech_summary},
                                ]
                                # Note: GPT-5 only supports default temperature (1)
                                # Removed max_completion_tokens to allow full response
                            )
                            
                            if response and response.choices and len(response.choices) > 0:
                                ai_tech_analysis = response.choices[0].message.content
                            if ai_tech_analysis and ai_tech_analysis.strip():
                                st.markdown(
                                    f'<div class="ai-response-box"><strong>🤖 Technical Analysis:</strong><br><br>{ai_tech_analysis}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # Show fallback if AI response is empty
                                st.info("📊 **Technical Indicators Summary:**\n\n" + tech_summary.replace("\n", "\n- "))
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate AI analysis: {str(e)[:100]}")
                            # Show fallback summary even if AI fails
                            st.info("📊 **Technical Indicators Summary:**\n\n" + tech_summary.replace("\n", "\n- "))
                
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
                        model="gpt-5-mini",  # GPT-5-mini for faster risk analysis
                        messages=[
                            {"role": "system", "content": "You are a professional risk analyst. Analyze the portfolio risk metrics and provide actionable risk management recommendations. Focus on diversification, position sizing, and risk mitigation strategies. Use emojis and be practical."},
                            {"role": "user", "content": risk_summary_text}
                        ],
                        max_completion_tokens=400,
                        # Note: GPT-5-mini only supports default temperature (1)
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

def channel_analytics_page():
    """Dedicated channel analytics dashboard"""
    st.header("📡 Channel Analytics")

    user = st.session_state.user
    holdings = get_cached_holdings(user['id'])

    if not holdings:
        st.info("No holdings available. Upload transactions to view channel analytics.")
        return

    df = pd.DataFrame(holdings)
    if df.empty:
        st.info("Channel analytics unavailable because holdings data could not be processed.")
        return

    for col in ['channel', 'total_quantity', 'average_price', 'current_price', 'investment', 'current_value', 'pnl']:
        if col not in df.columns:
            df[col] = 0

    df['channel'] = df['channel'].fillna('Unknown').replace('', 'Unknown').astype(str)
    df['total_quantity'] = pd.to_numeric(df['total_quantity'], errors='coerce').fillna(0.0)
    df['average_price'] = pd.to_numeric(df['average_price'], errors='coerce').fillna(0.0)
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0.0)
    df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0.0)
    df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce').fillna(0.0)
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

    df['effective_current_price'] = df['current_price'].where(df['current_price'] > 0, df['average_price'])
    df.loc[df['investment'] == 0, 'investment'] = df['total_quantity'] * df['average_price']
    df.loc[df['current_value'] == 0, 'current_value'] = df['total_quantity'] * df['effective_current_price']
    df.loc[df['pnl'] == 0, 'pnl'] = df['current_value'] - df['investment']

    df['pnl_pct'] = df.apply(
        lambda row: (row['pnl'] / row['investment'] * 100) if row['investment'] > 0 else 0.0,
        axis=1
    )

    channel_summary = df.groupby('channel').agg(
        total_positions=('ticker', 'count') if 'ticker' in df.columns else ('channel', 'count'),
        unique_assets=('ticker', 'nunique') if 'ticker' in df.columns else ('channel', 'count'),
        total_investment=('investment', 'sum'),
        current_value=('current_value', 'sum'),
        total_pnl=('pnl', 'sum')
    ).reset_index()

    channel_summary['pnl_pct'] = channel_summary.apply(
        lambda row: (row['total_pnl'] / row['total_investment'] * 100) if row['total_investment'] > 0 else 0.0,
        axis=1
    )
    total_current_value = channel_summary['current_value'].sum()
    if total_current_value > 0:
        channel_summary['allocation_pct'] = channel_summary['current_value'] / total_current_value * 100
    else:
        channel_summary['allocation_pct'] = 0.0

    total_channels = channel_summary.shape[0]
    total_value = channel_summary['current_value'].sum()
    total_investment = channel_summary['total_investment'].sum()
    total_pnl = channel_summary['total_pnl'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Channels", total_channels)
    col2.metric("Total Investment", f"₹{total_investment:,.0f}")
    col3.metric("Current Value", f"₹{total_value:,.0f}")
    col4.metric("Aggregate P&L", f"₹{total_pnl:,.0f}")

    if total_investment > 0:
        st.metric("Portfolio P&L %", f"{(total_pnl / total_investment) * 100:,.2f}%")

    st.markdown("### Channel Performance Overview")
    st.dataframe(
        channel_summary.assign(
            total_investment=lambda df_: df_['total_investment'].map(lambda v: f"₹{v:,.0f}"),
            current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
            total_pnl=lambda df_: df_['total_pnl'].map(lambda v: f"₹{v:,.0f}"),
            pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            allocation_pct=lambda df_: df_['allocation_pct'].map(lambda v: f"{v:.1f}%")
        ),
        use_container_width=True
    )

    st.markdown("### Allocation by Channel")
    try:
        fig_allocation = px.pie(
            channel_summary,
            values='current_value',
            names='channel',
            title="Current Value Distribution by Channel"
        )
        st.plotly_chart(fig_allocation, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render allocation chart: {exc}")

    st.markdown("### Channel P&L % Comparison")
    try:
        channel_summary_sorted = channel_summary.sort_values('pnl_pct', ascending=False).reset_index(drop=True)
        fig_pnl = px.bar(
            channel_summary_sorted,
            x='channel',
            y='pnl_pct',
            text=channel_summary_sorted['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
            title="Channel Performance (Total P&L %)",
        )
        fig_pnl.update_traces(textposition='outside', texttemplate='%{text}')
        fig_pnl.update_layout(
            yaxis_title="Total P&L %",
            yaxis=dict(zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render total performance chart: {exc}")

    try:
        weekly_history = db.get_channel_weekly_history(user['id'])
        if weekly_history:
            tab_weeks = st.tabs(["All Channels"] + sorted(list({entry['channel'] for entry in weekly_history})))

            with tab_weeks[0]:
                all_df = pd.DataFrame([entry for entry in weekly_history])
                fig_all = px.line(
                    all_df,
                    x='date',
                    y='pnl_pct',
                    color='channel',
                    title="Channel Performance (Last 52 Weeks)",
                )
                fig_all.update_layout(
                    yaxis_title="Weekly P&L %",
                    xaxis_title="Date",
                    hovermode='x unified',
                )
                st.plotly_chart(fig_all, use_container_width=True)

            channels = sorted(list({entry['channel'] for entry in weekly_history}))
            for idx, channel_name in enumerate(channels, start=1):
                with tab_weeks[idx]:
                    channel_df = pd.DataFrame(
                        [entry for entry in weekly_history if entry['channel'] == channel_name]
                    )
                    if not channel_df.empty:
                        st.write(f"**{channel_name}** — 52-Week Trend (Select weeks as needed)")
                        channel_df['date'] = pd.to_datetime(channel_df['date'])
                        channel_df = channel_df.sort_values('date')
                        fig_channel = px.line(
                            channel_df,
                            x='date',
                            y='pnl_pct',
                            markers=True,
                            title=f"{channel_name} — Weekly P&L %",
                        )
                        fig_channel.update_layout(
                            yaxis_title="Weekly P&L %",
                            xaxis_title="Date",
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_channel, use_container_width=True)
                        st.dataframe(
                            channel_df[['date', 'pnl_pct', 'investment', 'current_value']].assign(
                                date=lambda df_: df_['date'].dt.strftime('%Y-%m-%d'),
                                pnl_pct=lambda df_: df_['pnl_pct'].map(lambda v: f"{v:+.2f}%"),
                                investment=lambda df_: df_['investment'].map(lambda v: f"₹{v:,.0f}"),
                                current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info(f"No weekly history available for {channel_name}.")
        else:
            st.info("52-week channel performance not available yet—weekly NAV history is still being gathered.")
    except Exception as exc:
        st.warning(f"Could not render 52-week performance chart: {exc}")

    st.markdown("### Channel Details")
    for _, row in channel_summary.sort_values('current_value', ascending=False).iterrows():
        channel_name = row['channel']
        with st.expander(f"{channel_name} • Current Value ₹{row['current_value']:,.0f}", expanded=False):
            channel_df = df[df['channel'] == channel_name].copy()
            channel_df_display = channel_df[
                [col for col in ['ticker', 'stock_name', 'asset_type', 'total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
                 if col in channel_df.columns]
            ].copy()

            numeric_columns = ['total_quantity', 'investment', 'current_value', 'pnl', 'pnl_pct']
            for col in numeric_columns:
                if col in channel_df_display.columns:
                    if col == 'pnl_pct':
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"{v:+.2f}%")
                    elif col == 'total_quantity':
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"{v:,.2f}")
                    else:
                        channel_df_display[col] = channel_df_display[col].map(lambda v: f"₹{v:,.2f}")

            st.dataframe(channel_df_display, use_container_width=True)

            asset_breakdown = channel_df.groupby('asset_type').agg(
                current_value=('current_value', 'sum'),
                investment=('investment', 'sum')
            ).reset_index()
            if not asset_breakdown.empty:
                total_channel_value = asset_breakdown['current_value'].sum()
                if total_channel_value > 0:
                    asset_breakdown['allocation_pct'] = asset_breakdown['current_value'] / total_channel_value * 100
                else:
                    asset_breakdown['allocation_pct'] = 0.0
                st.markdown("**Asset Allocation within Channel**")
                st.dataframe(
                    asset_breakdown.assign(
                        current_value=lambda df_: df_['current_value'].map(lambda v: f"₹{v:,.0f}"),
                        investment=lambda df_: df_['investment'].map(lambda v: f"₹{v:,.0f}"),
                        allocation_pct=lambda df_: df_['allocation_pct'].map(lambda v: f"{v:.1f}%")
                    ),
                    use_container_width=True
                )

    st.markdown("---")
    st.caption("Tip: Keep channel metadata (e.g., broker/platform names) consistent while uploading transactions to unlock richer analytics.")


def ai_assistant_page():
    """Dedicated AI Assistant page"""
    st.header("🤖 AI Assistant")
    st.caption("Your intelligent portfolio advisor with access to all your data")
    
    user = st.session_state.user
    
    # Initialize chat history and PDF context
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load chat history from database (user-specific)
    try:
        # Check if method exists (for backward compatibility)
        if hasattr(db, 'get_user_chat_history'):
            db_chat_history = db.get_user_chat_history(user['id'], limit=50)
            if db_chat_history:
                # Convert database format to session state format
                st.session_state.chat_history = [
                    {"q": chat['question'], "a": chat['answer']}
                    for chat in reversed(db_chat_history)  # Reverse to show oldest first
                ]
        else:
            # Method not available, use empty list
            st.session_state.chat_history = []
    except Exception as e:
        # If table doesn't exist yet, just use empty list
        st.session_state.chat_history = []
    
    # Always load PDF context from database to ensure older session PDFs are included
    st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
    
    # Show PDF context status and refresh option
    pdf_count = len(db.get_user_pdfs(user['id']))
    if pdf_count > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📚 Loaded {pdf_count} PDFs from shared library (all users) for AI context")
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
                from datetime import datetime
                import json
                openai.api_key = st.secrets["api_keys"]["open_ai"]
                
                # Safe float conversion helper
                def safe_float(value, default=0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                # ===== DATABASE QUERY FUNCTIONS FOR AI =====
                # These functions give the AI direct access to query the database
                
                def get_holdings(user_id: str, asset_type: str = None, sector: str = None, limit: int = None) -> str:
                    """Get user holdings from database. Returns JSON string of holdings data."""
                    try:
                        query = db.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
                        if asset_type:
                            query = query.eq('asset_type', asset_type)
                        if sector:
                            query = query.eq('sector', sector)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_transactions(user_id: str, date_from: str = None, date_to: str = None, transaction_type: str = None, ticker: str = None, limit: int = 200) -> str:
                    """Get user transactions from database. Returns JSON string of transactions data."""
                    try:
                        query = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user_id)
                        if date_from:
                            query = query.gte('transaction_date', date_from)
                        if date_to:
                            query = query.lte('transaction_date', date_to)
                        if transaction_type:
                            query = query.eq('transaction_type', transaction_type.lower())
                        if ticker:
                            query = query.eq('ticker', ticker)
                        if limit:
                            query = query.limit(limit)
                        response = query.order('transaction_date', desc=True).execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_historical_prices(ticker: str, date_from: str = None, date_to: str = None, limit: int = 52) -> str:
                    """Get historical prices for a ticker. Returns JSON string of historical price data."""
                    try:
                        # Get stock_id first
                        stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                        if not stock_response.data:
                            return json.dumps({"error": f"Ticker {ticker} not found"})
                        stock_id = stock_response.data[0]['id']
                        
                        # Get historical prices
                        query = db.supabase.table('historical_prices').select('*').eq('stock_id', stock_id)
                        if date_from:
                            query = query.gte('price_date', date_from)
                        if date_to:
                            query = query.lte('price_date', date_to)
                        query = query.order('price_date', desc=True)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_stock_master(ticker: str = None, asset_type: str = None, limit: int = 100) -> str:
                    """Get stock master data. Returns JSON string of stock master records."""
                    try:
                        query = db.supabase.table('stock_master').select('*')
                        if ticker:
                            query = query.eq('ticker', ticker)
                        if asset_type:
                            query = query.eq('asset_type', asset_type)
                        if limit:
                            query = query.limit(limit)
                        response = query.execute()
                        return json.dumps(response.data if response.data else [], indent=2)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_pdfs(user_id: str = None, search_term: str = None, limit: int = 10) -> str:
                    """Get PDF documents and their summaries. Returns JSON string of PDF data."""
                    try:
                        pdfs = db.get_user_pdfs(user_id) if user_id else db.get_user_pdfs()
                        if search_term:
                            search_lower = search_term.lower()
                            pdfs = [p for p in pdfs if search_lower in (p.get('filename', '') + p.get('ai_summary', '')).lower()]
                        if limit:
                            pdfs = pdfs[:limit]
                        return json.dumps(pdfs, indent=2, default=str)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                def get_financial_news(ticker: str = None, company_name: str = None, sector: str = None, limit: int = 10) -> str:
                    """Get latest financial news from Moneycontrol, Economic Times, and other sources. Returns JSON string of news articles."""
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        from datetime import datetime, timedelta
                        
                        news_articles = []
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        # 1. Moneycontrol - Try multiple approaches
                        try:
                            moneycontrol_urls = []
                            
                            # Approach 1: Ticker-specific news (if ticker provided)
                            if ticker:
                                clean_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()
                                # Try different Moneycontrol URL patterns
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{clean_ticker.lower()}.html")
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{clean_ticker}.html")
                                # Try company page news
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/india/stockpricequote/{clean_ticker.lower()}")
                            
                            # Approach 2: General market news (always try)
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/")
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/markets/")
                            moneycontrol_urls.append("https://www.moneycontrol.com/news/business/stocks/")
                            
                            # Approach 3: Sector-specific news
                            if sector:
                                sector_lower = sector.lower().replace(' ', '-')
                                moneycontrol_urls.append(f"https://www.moneycontrol.com/news/tags/{sector_lower}.html")
                            
                            # Try each URL until we get results
                            for url in moneycontrol_urls[:3]:  # Limit to 3 URLs to avoid too many requests
                                try:
                                    response = requests.get(url, headers=headers, timeout=10)
                                    if response.status_code == 200:
                                        soup = BeautifulSoup(response.content, 'html.parser')
                                        
                                        # Try multiple selectors for Moneycontrol's article structure
                                        articles = []
                                        
                                        # Selector 1: Standard news list
                                        articles.extend(soup.find_all('li', class_='clearfix', limit=limit))
                                        
                                        # Selector 2: Article cards
                                        if not articles:
                                            articles.extend(soup.find_all('div', class_='newslist', limit=limit))
                                        
                                        # Selector 3: Generic article links
                                        if not articles:
                                            articles.extend(soup.find_all('a', href=lambda x: x and '/news/' in x, limit=limit))
                                        
                                        for article in articles[:limit]:
                                            try:
                                                # Try to find title
                                                title_elem = (article.find('h2') or 
                                                            article.find('h3') or 
                                                            article.find('a', class_=lambda x: x and 'title' in str(x).lower()) or
                                                            article.find('a'))
                                                
                                                # Try to find link
                                                link_elem = article.find('a') if not isinstance(article, type(soup.find('a'))) else article
                                                
                                                if title_elem:
                                                    title = title_elem.get_text(strip=True)
                                                    if title and len(title) > 10:  # Valid title
                                                        url_link = ""
                                                        if link_elem and hasattr(link_elem, 'get'):
                                                            url_link = link_elem.get('href', '')
                                                            # Make absolute URL if relative
                                                            if url_link and not url_link.startswith('http'):
                                                                url_link = f"https://www.moneycontrol.com{url_link}" if url_link.startswith('/') else f"https://www.moneycontrol.com/{url_link}"
                                                        
                                                        # Try to find date
                                                        date_elem = (article.find('span', class_='date') or 
                                                                   article.find('span', class_=lambda x: x and 'date' in str(x).lower()) or
                                                                   article.find('time'))
                                                        date_str = date_elem.get_text(strip=True) if date_elem else datetime.now().strftime("%Y-%m-%d")
                                                        
                                                        news_articles.append({
                                                            "title": title,
                                                            "url": url_link,
                                                            "source": "Moneycontrol",
                                                            "date": date_str,
                                                            "ticker": ticker if ticker else None,
                                                            "sector": sector if sector else None
                                                        })
                                                        
                                                        if len(news_articles) >= limit:
                                                            break
                                            except Exception:
                                                continue
                                        
                                        if len(news_articles) >= limit:
                                            break
                                except Exception:
                                    continue
                                    
                        except Exception as e:
                            pass  # Moneycontrol failed, try other sources
                        
                        # 2. Use AI to fetch news if web scraping fails or for general queries
                        if len(news_articles) < limit:
                            try:
                                # Use OpenAI to get recent financial news (GPT-5 has knowledge up to its training date)
                                news_query = f"Latest financial news"
                                if ticker:
                                    news_query += f" about {ticker}"
                                if company_name:
                                    news_query += f" ({company_name})"
                                if sector:
                                    news_query += f" in {sector} sector"

                                # Note: This uses AI's training knowledge, not real-time web access
                                # For true real-time news, you'd need a news API like NewsAPI, Alpha Vantage, etc.
                                ai_news_response = openai.chat.completions.create(
                                    model="gpt-5",
                                    messages=[{
                                        "role": "user",
                                        "content": f"Provide the latest financial news and market updates {news_query}. Include recent developments, market trends, and any significant events. Format as a list of news items with titles and brief summaries."
                                    }],
                                    max_completion_tokens=1000
                                )
                                
                                if ai_news_response.choices:
                                    ai_news_text = ai_news_response.choices[0].message.content
                                    # Parse AI response into structured format
                                news_articles.append({
                                    "title": "AI-Generated Market Update",
                                    "content": ai_news_text,
                                        "source": "AI Analysis (Training Data)",
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                        "note": "Based on AI training data - may not include very recent news"
                                })
                            except Exception as e:
                                pass
                        
                        # If no news found, return a helpful message
                        if not news_articles:
                            return json.dumps({
                                "message": "No recent news found. For real-time financial news, consider integrating NewsAPI, Alpha Vantage News API, or other financial news services.",
                                "suggestions": [
                                    "Use NewsAPI.org for real-time news",
                                    "Use Alpha Vantage News & Sentiment API",
                                    "Scrape Moneycontrol, Economic Times, or Business Standard",
                                    "Use RSS feeds from financial news sources"
                                ]
                            }, indent=2)
                        
                        return json.dumps(news_articles[:limit], indent=2, default=str)
                    except Exception as e:
                        return json.dumps({"error": str(e), "message": "Financial news fetching failed. Consider using a dedicated news API for real-time updates."})
                
                # Define OpenAI function tools
                functions = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_holdings",
                            "description": "Get user holdings from the database. Use this to query portfolio holdings, filter by asset_type (stock, mutual_fund, bond, pms, aif) or sector, and limit results.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (required)"},
                                    "asset_type": {"type": "string", "description": "Filter by asset type: stock, mutual_fund, bond, pms, aif"},
                                    "sector": {"type": "string", "description": "Filter by sector"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_transactions",
                            "description": "Get user transactions from the database. Use this to query transaction history, filter by date range, transaction type (buy/sell), or ticker.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (required)"},
                                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD format)"},
                                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD format)"},
                                    "transaction_type": {"type": "string", "description": "Filter by transaction type: buy or sell"},
                                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return (default 200)"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_historical_prices",
                            "description": "Get historical price data for a ticker. Use this to analyze price trends, 52-week highs/lows, and price movements over time.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Ticker symbol (required)"},
                                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD format)"},
                                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD format)"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return (default 52 for 52 weeks)"}
                                },
                                "required": ["ticker"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock_master",
                            "description": "Get stock master data including ticker information, asset types, sectors, and current prices. Use this to get metadata about stocks, mutual funds, bonds, etc.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Filter by specific ticker"},
                                    "asset_type": {"type": "string", "description": "Filter by asset type: stock, mutual_fund, bond, pms, aif"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_pdfs",
                            "description": "Get PDF documents and their AI summaries from the shared library. Use this to access research documents, reports, and analysis that can inform recommendations.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "description": "User ID (optional, if not provided returns all shared PDFs)"},
                                    "search_term": {"type": "string", "description": "Search term to filter PDFs by filename or content"},
                                    "limit": {"type": "integer", "description": "Maximum number of records to return"}
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_financial_news",
                            "description": "Get latest financial news and market updates from Moneycontrol, Economic Times, and other sources. Use this to access real-time financial news, market trends, company updates, and sector news. Note: Currently uses web scraping and AI knowledge - for true real-time news, consider integrating NewsAPI or Alpha Vantage News API.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'RELIANCE.NS', 'TCS.NS') to get news about a specific stock"},
                                    "company_name": {"type": "string", "description": "Company name to search for news"},
                                    "sector": {"type": "string", "description": "Sector name (e.g., 'Technology', 'Banking') to get sector-specific news"},
                                    "limit": {"type": "integer", "description": "Maximum number of news articles to return (default 10)"}
                                }
                            }
                        }
                    }
                ]
                
                # Get current date for context
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_year = datetime.now().year
                
                # Prepare session dataframes summary (available in memory)
                session_data_summary = f"""
📊 SESSION DATA AVAILABLE (Already loaded in memory):
- Holdings DataFrame: {len(holdings)} holdings loaded
- Portfolio Summary: Available
- PDF Context: {len(user_pdfs)} PDFs loaded
- User ID: {user['id']}

💡 INSTRUCTIONS:
- For data-related questions, use the database query functions (get_holdings, get_transactions, etc.) to fetch the exact data you need
- The session dataframes are already loaded, but you can query the database for more specific or filtered data
- Always query the database when you need:
  * Filtered transactions (by date, type, ticker)
  * Historical prices for specific tickers
  * Stock master metadata
  * PDF documents matching search terms
- Use the functions based on what the user is asking - don't query everything, only what's needed
"""
                
                # Build system prompt with full database access instructions
                system_prompt = f"""You are an expert portfolio analyst and financial advisor with DIRECT ACCESS to the complete database for user_id: {user['id']}.

📅 CURRENT DATE: {current_date} (Year: {current_year})
⚠️ CRITICAL: Today's date is {current_date}. Always use this date when:
- Calculating time periods (e.g., "1 year ago" means {current_year - 1}-{datetime.now().strftime('%m-%d')})
- Referencing current market conditions
- Making time-based predictions
- Analyzing transaction dates and holding periods
Do NOT use 2024 or any other year - use {current_year}.

🔑 DATABASE ACCESS:
You have DIRECT ACCESS to query the database using these functions:
1. get_holdings(user_id, asset_type, sector, limit) - Query holdings
2. get_transactions(user_id, date_from, date_to, transaction_type, ticker, limit) - Query transactions
3. get_historical_prices(ticker, date_from, date_to, limit) - Query historical prices
4. get_stock_master(ticker, asset_type, limit) - Query stock metadata
5. get_pdfs(user_id, search_term, limit) - Query PDF documents
6. get_financial_news(ticker, company_name, sector, limit) - Get latest financial news from Moneycontrol, Economic Times, and other sources

📰 FINANCIAL NEWS ACCESS:
- Use get_financial_news() to fetch latest market news, company updates, and sector trends
- You can search by ticker, company name, or sector
- This helps you provide recommendations based on current market conditions and news
- Note: Currently uses web scraping and AI knowledge - for true real-time news, consider integrating NewsAPI or Alpha Vantage News API

📊 SESSION DATA:
{session_data_summary}

🎯 HOW TO USE:
1. Analyze the user's question to determine what data you need
2. Use the appropriate function(s) to query the database for the exact data needed
3. Don't query everything - only query what's relevant to the question
4. For example:
   - "Show my tech stocks" → get_holdings(user_id="{user['id']}", asset_type="stock", sector="Technology")
   - "1 year buy transactions" → get_transactions(user_id="{user['id']}", date_from="{datetime.now().replace(year=current_year-1).strftime('%Y-%m-%d')}", transaction_type="buy")
   - "Price history of RELIANCE" → get_historical_prices(ticker="RELIANCE.NS")
   - "PDFs about banking" → get_pdfs(search_term="banking")

💡 CAPABILITIES:
- ✅ Suggest BUY recommendations based on PDF research, market analysis, and portfolio gaps
- ✅ Suggest SELL recommendations based on overvaluation, poor performance, or risk concerns
- ✅ Analyze when transactions would have been more profitable using historical price data
- ✅ Provide actionable investment recommendations with specific tickers and reasoning
- ✅ Make PREDICTIONS and FORECASTS about stock prices, market trends, and economic indicators
- ✅ Calculate P&L, returns, and other metrics from transaction data
- ✅ Compare filtered results with overall portfolio when relevant

Always:
- Use the database functions to get the exact data you need based on the question
- Cite specific tickers, dates, and amounts from the queried data
- Reference PDF research documents when making recommendations
- Provide data-driven recommendations based on actual numbers from the database"""
                
                # Start conversation with chat history (if available) and user question
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add chat history to context (last 10 conversations)
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history[-10:]:  # Last 10 conversations
                        messages.append({"role": "user", "content": chat.get("q", "")})
                        messages.append({"role": "assistant", "content": chat.get("a", "")})
                
                # Add current user question
                messages.append({"role": "user", "content": user_question})
                
                # Function calling loop - allow AI to query database multiple times
                max_iterations = 5
                for iteration in range(max_iterations):
                    response = openai.chat.completions.create(
                        model="gpt-5",
                        messages=messages,
                        tools=functions,
                        tool_choice="auto"  # Let AI decide when to use functions
                    )
                    
                    choice = response.choices[0]
                    # Convert message object to dict format for consistency
                    message_dict = {
                        "role": choice.message.role,
                        "content": choice.message.content if choice.message.content else None
                    }
                    if choice.message.tool_calls:
                        message_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in choice.message.tool_calls
                        ]
                    messages.append(message_dict)
                    
                    # Check if AI wants to call a function
                    if choice.message.tool_calls:
                        # Execute function calls
                        for tool_call in choice.message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            # Add user_id to function calls that need it
                            if function_name in ['get_holdings', 'get_transactions'] and 'user_id' not in function_args:
                                function_args['user_id'] = user['id']
                            
                            # Execute the function
                            if function_name == 'get_holdings':
                                function_result = get_holdings(**function_args)
                            elif function_name == 'get_transactions':
                                function_result = get_transactions(**function_args)
                            elif function_name == 'get_historical_prices':
                                function_result = get_historical_prices(**function_args)
                            elif function_name == 'get_stock_master':
                                function_result = get_stock_master(**function_args)
                            elif function_name == 'get_pdfs':
                                function_result = get_pdfs(**function_args)
                            elif function_name == 'get_financial_news':
                                function_result = get_financial_news(**function_args)
                            else:
                                function_result = json.dumps({"error": f"Unknown function: {function_name}"})
                            
                            # Add function result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": function_result
                            })
                    else:
                        # AI provided final answer, break loop
                        break
                
                # Get final AI response
                # Handle both dict and object formats
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    ai_response = last_message.get("content", "") if last_message.get("content") else "I apologize, but I couldn't generate a response."
                else:
                    ai_response = last_message.content if hasattr(last_message, 'content') and last_message.content else "I apologize, but I couldn't generate a response."
                
                # Check if response was truncated
                if not ai_response or ai_response.strip() == "":
                    st.error("❌ Empty response from AI. Please try again.")
                    st.stop()
                
                # Display the response immediately
                st.markdown("---")
                st.markdown("### 💬 AI Response:")
                st.success(ai_response)
                
                # Store in chat history (session state)
                st.session_state.chat_history.append({
                    "q": user_question,
                    "a": ai_response
                })
                
                # Save to database (user-specific, persistent)
                # Check if method exists before calling
                if hasattr(db, 'save_chat_history'):
                    try:
                        db.save_chat_history(user['id'], user_question, ai_response)
                    except Exception as e:
                        # If table doesn't exist, just continue without saving
                        pass
                
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
    
    # PDF Library Section (Shared across all users)
    st.markdown("---")
    st.markdown("**📚 Shared PDF Library (Available to All Users)**")
    st.caption("💡 PDFs uploaded by any user are visible to everyone")
    
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
                                model="gpt-5",  # Upgraded to GPT-5 for better results
                                messages=[{"role": "user", "content": analysis_prompt}],
                                max_completion_tokens=800,
                                # Note: GPT-5 only supports default temperature (1)
                            )
                            
                            fresh_analysis = response.choices[0].message.content
                            
                            # Display the fresh analysis
                            st.markdown("### 🔍 Fresh Analysis")
                            st.markdown(fresh_analysis)
                            
                            # Store in chat history (session state)
                            st.session_state.chat_history.append({
                                "q": f"Analyze PDF: {pdf['filename']}", 
                                "a": fresh_analysis
                            })
                            # Save to database (user-specific, persistent)
                            if hasattr(db, 'save_chat_history'):
                                try:
                                    db.save_chat_history(user['id'], f"Analyze PDF: {pdf['filename']}", fresh_analysis)
                                except Exception:
                                    pass
                            
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
    
    # PDF Upload for AI Analysis (Multiple Files Supported)
    st.markdown("---")
    st.markdown("**📤 Upload PDFs for AI Analysis**")
    st.caption("💡 You can upload multiple PDFs at once!")
    
    uploaded_pdfs = st.file_uploader(
        "Choose PDF file(s) to analyze",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload research reports, financial statements, or any document for AI analysis. Select multiple files to upload them all at once!"
    )
    
    if uploaded_pdfs:
        # Handle both single file and multiple files
        if not isinstance(uploaded_pdfs, list):
            uploaded_pdfs = [uploaded_pdfs]
        
        if len(uploaded_pdfs) == 1:
            button_text = f"🔍 Analyze 1 PDF"
        else:
            button_text = f"🔍 Analyze & Upload {len(uploaded_pdfs)} PDFs"
        
        if st.button(button_text, type="primary"):
            success_count = 0
            failed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_pdf in enumerate(uploaded_pdfs):
                try:
                    status_text.text(f"📄 Processing {idx + 1}/{len(uploaded_pdfs)}: {uploaded_pdf.name}...")
                    progress_bar.progress((idx + 1) / len(uploaded_pdfs))
                    
                    import PyPDF2
                    import pdfplumber
                    import openai
                    openai.api_key = st.secrets["api_keys"]["open_ai"]
                    
                    # Enhanced PDF extraction with tables and structure
                    pdf_text = ""
                    tables_found = []
                    page_count = 0
                    
                    # Try pdfplumber first (better for tables)
                    # Reset file pointer for each file
                    uploaded_pdf.seek(0)
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
                            model="gpt-5",  # Upgraded to GPT-5 for better PDF analysis
                            messages=[{"role": "user", "content": analysis_prompt}],
                            max_completion_tokens=1000,
                            # Note: GPT-5 only supports default temperature (1)
                        )
                        
                        ai_analysis = response.choices[0].message.content
                        
                        # Display the analysis for this PDF
                        with st.expander(f"📄 {uploaded_pdf.name} - Analysis", expanded=(idx == 0)):
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_analysis)
                            st.info(f"✅ Extracted {len(pdf_text)} characters from {page_count} pages, found {len(tables_found)} tables")
                        
                        # Store in chat history (session state)
                        st.session_state.chat_history.append({
                            "q": f"Analyze PDF: {uploaded_pdf.name}", 
                            "a": ai_analysis
                        })
                        # Save to database (user-specific, persistent)
                        if hasattr(db, 'save_chat_history'):
                            try:
                                db.save_chat_history(user['id'], f"Analyze PDF: {uploaded_pdf.name}", ai_analysis)
                            except Exception:
                                pass
                        
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
                            success_count += 1
                        else:
                            failed_count += 1
                            st.error(f"❌ Failed to save '{uploaded_pdf.name}': {save_result.get('error', 'Unknown error')}")
                    else:
                        failed_count += 1
                        st.error(f"❌ Could not extract text from '{uploaded_pdf.name}'. Please ensure the PDF contains readable text.")
                        
                except Exception as e:
                    failed_count += 1
                    st.error(f"❌ Error processing '{uploaded_pdf.name}': {str(e)[:100]}")
            
            # Final summary
            progress_bar.empty()
            status_text.empty()
            
            if success_count > 0:
                st.success(f"✅ Successfully processed and saved {success_count} PDF(s) to shared library!")
                if failed_count > 0:
                    st.warning(f"⚠️ {failed_count} PDF(s) failed to process")
                
                # Refresh the page to update the PDF library count
                st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                st.info("📄 PDF content is now stored and available for all future sessions!")
                st.rerun()
            else:
                st.error(f"❌ All {len(uploaded_pdfs)} PDF(s) failed to process. Please check the files and try again.")
    
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

def ai_insights_page():
    """AI Insights page with agent analysis and recommendations"""
    st.header("🧠 AI Insights")
    st.caption("Powered by specialized AI agents for portfolio analysis")
    
    if not AI_AGENTS_AVAILABLE:
        st.error("AI agents are not available. Please check the installation.")
        return
    
    # Get user data
    user = st.session_state.user
    db = st.session_state.db
    
    # Get holdings data
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see AI insights.")
        return
    
    # Get comprehensive context for AI analysis (same as AI Assistant has)
    pdf_context = db.get_all_pdfs_text(user['id'])
    pdf_count = len(db.get_user_pdfs(user['id']))
    
    # Get all transactions for the user
    try:
        all_transactions_response = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user['id']).order('transaction_date', desc=False).limit(1000).execute()
        all_transactions = all_transactions_response.data if all_transactions_response.data else []
    except:
        all_transactions = []
    
    # Get historical prices for all tickers in holdings (increase limit)
    historical_prices = {}
    historical_prices_fetched = 0
    historical_prices_missing = 0
    try:
        # Get unique tickers from all holdings
        unique_tickers = set()
        for holding in holdings:
            ticker = holding.get('ticker')
            if ticker:
                unique_tickers.add(ticker)
        
        # Fetch historical prices for all unique tickers (not just top 30)
        for ticker in unique_tickers:
            try:
                # Get stock_id
                stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                if stock_response.data:
                    stock_id = stock_response.data[0]['id']
                    # Get 52 weeks of historical prices
                    hist_response = db.supabase.table('historical_prices').select('price_date, price').eq('stock_id', stock_id).order('price_date', desc=True).limit(52).execute()
                    if hist_response.data and len(hist_response.data) > 0:
                        historical_prices[ticker] = hist_response.data
                        historical_prices_fetched += 1
                    else:
                        historical_prices_missing += 1
            except:
                historical_prices_missing += 1
    except Exception as e:
        pass
    
    # Get stock master data for all holdings
    try:
        # Get all stock master records for holdings (get all unique tickers)
        tickers = list(set([h.get('ticker') for h in holdings if h.get('ticker')]))
        if tickers:
            # Fetch in batches if needed (Supabase .in_() has limits)
            stock_master = []
            for i in range(0, len(tickers), 100):  # Process 100 at a time
                batch = tickers[i:i+100]
                try:
                    stock_master_response = db.supabase.table('stock_master').select('*').in_('ticker', batch).execute()
                    if stock_master_response.data:
                        stock_master.extend(stock_master_response.data)
                except:
                    pass
        else:
            stock_master = []
    except:
        stock_master = []
    
    # Show context status with detailed information
    context_info = []
    if pdf_count > 0:
        context_info.append(f"📚 {pdf_count} PDF(s)")
    if all_transactions:
        context_info.append(f"📝 {len(all_transactions)} transactions")
    if historical_prices:
        context_info.append(f"📊 Historical prices: {len(historical_prices)} tickers (52 weeks each)")
    if stock_master:
        context_info.append(f"🏢 {len(stock_master)} stock master records")
    
    if context_info:
        st.info(f"**AI Context Active:** {' | '.join(context_info)}")
        if historical_prices_missing > 0:
            st.caption(f"ℹ️ Note: {historical_prices_missing} ticker(s) don't have historical price data yet. They will be automatically fetched during next login. Click below to fetch now if needed.")
            if st.button("📊 Fetch Historical Prices Now", key="fetch_missing_historical", help="Manually fetch 52 weeks of historical prices for all tickers missing data (normally done automatically during login)"):
                with st.spinner(f"Fetching historical prices for {historical_prices_missing} ticker(s)..."):
                    try:
                        # Get tickers missing historical data
                        missing_tickers = []
                        for holding in holdings:
                            ticker = holding.get('ticker')
                            if ticker and ticker not in historical_prices:
                                missing_tickers.append(ticker)
                        
                        if missing_tickers:
                            # Get asset types
                            asset_types = {h.get('ticker'): h.get('asset_type', 'stock') for h in holdings if h.get('ticker')}
                            
                            # Fetch comprehensive data (includes historical prices)
                            if 'bulk_ai_fetcher' in st.session_state and st.session_state.bulk_ai_fetcher.available:
                                db.bulk_process_new_stocks_with_comprehensive_data(
                                    tickers=list(set(missing_tickers)),
                                    asset_types=asset_types
                                )
                                st.success(f"✅ Fetched historical prices for {len(set(missing_tickers))} ticker(s)!")
                                st.rerun()
                            else:
                                st.warning("⚠️ Bulk AI fetcher not available. Historical prices will be fetched automatically during next login.")
                                st.rerun()
                        else:
                            st.info("All tickers already have historical price data.")
                    except Exception as e:
                        st.error(f"❌ Error fetching historical prices: {str(e)[:100]}")
    else:
        st.caption("💡 Upload PDFs and transactions to enhance insights")
    
    # Create tabs for different AI insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Smart Recommendations",
        "📊 Portfolio Analysis",
        "🔍 Market Insights",
        "🔮 Scenario Analysis",
        "💡 Investment Recommendations",
        "⚙️ Agent Status"
    ])

    analysis_cache_key = f"ai_analysis_{user['id']}_{len(holdings)}_{len(all_transactions)}"
    analysis_result = st.session_state.get(analysis_cache_key)

    if analysis_result is None or not st.session_state.get('ai_analysis_complete', False):
        with st.spinner("🤖 AI agents analyzing your portfolio (this may take 30-60 seconds)..."):
            try:
                user_profile_data = db.get_user_profile(user['id'])
                user_profile = {
                    "user_id": user['id'],
                    "risk_tolerance": user_profile_data.get('risk_tolerance', 'moderate'),
                    "goals": user_profile_data.get('investment_goals', []),
                    "rebalancing_frequency": user_profile_data.get('rebalancing_frequency', 'quarterly'),
                    "tax_optimization": user_profile_data.get('tax_optimization', True),
                    "esg_investing": user_profile_data.get('esg_investing', False),
                    "international_exposure": user_profile_data.get('international_exposure', 20)
                }

                from ai_agents.agent_manager import run_ai_analysis
                analysis_result = run_ai_analysis(
                    holdings,
                    user_profile,
                    pdf_context,
                    all_transactions,
                    historical_prices,
                    stock_master
                )
                st.session_state[analysis_cache_key] = analysis_result
                st.session_state['ai_analysis_complete'] = True
            except Exception as e:
                st.error(f"Error running AI analysis: {str(e)}")
                analysis_result = {"error": str(e)}
                st.session_state[analysis_cache_key] = analysis_result
    else:
        analysis_result = st.session_state[analysis_cache_key]

    with tab1:
        st.subheader("🎯 Smart Recommendations")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                recommendations = analysis_result.get("investment_recommendations", [])
                if not recommendations:
                    try:
                        from ai_agents.agent_manager import get_ai_recommendations
                        recommendations = get_ai_recommendations(5)
                    except Exception:
                        recommendations = []

                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} AI recommendations")
                    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    for rec in recommendations:
                        icon = severity_icons.get(rec.get("severity", "low"), "🟢")
                        with st.expander(f"{icon} {rec.get('title', 'Recommendation')}", expanded=rec.get("severity") == "high"):
                            st.markdown(f"**Description:** {rec.get('description', 'No description')}")
                            st.markdown(f"**Recommendation:** {rec.get('recommendation', 'No recommendation')}")
                            if rec.get("data"):
                                st.json(rec["data"])
                else:
                    st.info("🎉 No urgent recommendations found. Your portfolio looks well-balanced!")
        except Exception as e:
            st.error(f"Error displaying recommendations: {str(e)}")

    with tab2:
        st.subheader("📊 Portfolio Analysis")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                insights = analysis_result.get("portfolio_insights", [])
                if insights:
                    st.success(f"📈 Portfolio analysis complete - {len(insights)} insights found")
                    for insight in insights:
                        with st.expander(f"💡 {insight.get('title', 'Insight')}", expanded=False):
                            st.markdown(insight.get('description', 'No description'))
                            if insight.get('data'):
                                st.json(insight['data'])
                else:
                    st.info("No portfolio insights available yet.")
        except Exception as e:
            st.error(f"Error displaying portfolio analysis: {str(e)}")

    with tab3:
        st.subheader("🔍 Market Insights")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                market_insights = analysis_result.get("market_insights", [])
                if market_insights:
                    st.success(f"🌍 Market analysis complete - {len(market_insights)} insights found")
                    for insight in market_insights:
                        with st.expander(f"📊 {insight.get('title', 'Market Insight')}", expanded=False):
                            st.markdown(insight.get('description', 'No description'))
                            if insight.get('data'):
                                st.json(insight['data'])
                else:
                    st.info("No market insights available yet.")
        except Exception as e:
            st.error(f"Error displaying market insights: {str(e)}")

    with tab4:
        st.subheader("🔮 Scenario Analysis")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                scenario_insights = analysis_result.get("scenario_insights", [])
                if scenario_insights:
                    st.success(f"🔮 Scenario analysis complete - {len(scenario_insights)} scenarios analyzed")
                    for scenario in scenario_insights:
                        with st.expander(f"🎯 {scenario.get('title', 'Scenario')}", expanded=False):
                            st.markdown(scenario.get('description', 'No description'))
                            if scenario.get('data'):
                                st.json(scenario['data'])
                else:
                    st.info("No scenario insights available yet.")
        except Exception as e:
            st.error(f"Error displaying scenario analysis: {str(e)}")

    with tab5:
        st.subheader("💡 Investment Recommendations")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                recommendations = analysis_result.get("investment_recommendations", [])
                if recommendations:
                    st.success(f"💡 Investment recommendations complete - {len(recommendations)} recommendations found")
                    for rec in recommendations:
                        with st.expander(f"💼 {rec.get('title', 'Recommendation')}", expanded=False):
                            st.markdown(rec.get('description', 'No description'))
                            st.markdown(f"**Action:** {rec.get('recommendation', 'No specific action')}")
                            if rec.get('data'):
                                st.json(rec['data'])
                else:
                    st.info("No investment recommendations available yet.")
        except Exception as e:
            st.error(f"Error displaying investment recommendations: {str(e)}")

    with tab6:
        st.subheader("⚙️ Agent Status")
        try:
            if not analysis_result or "error" in analysis_result:
                st.error(analysis_result.get('error', 'No analysis result') if analysis_result else "No analysis result")
            else:
                st.info("✅ All AI agents have completed their analysis")
                st.json(analysis_result)
        except Exception as e:
            st.error(f"Error displaying agent status: {str(e)}")

    # PDF Upload for AI Analysis (Multiple Files Supported)
    st.markdown("---")
    st.markdown("**📤 Upload PDFs for AI Analysis**")
    st.caption("💡 You can upload multiple PDFs at once!")
    
    uploaded_pdfs = st.file_uploader(
        "Choose PDF file(s) to analyze",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload research reports, financial statements, or any document for AI analysis. Select multiple files to upload them all at once!"
    )
    
    if uploaded_pdfs:
        # Handle both single file and multiple files
        if not isinstance(uploaded_pdfs, list):
            uploaded_pdfs = [uploaded_pdfs]
        
        if len(uploaded_pdfs) == 1:
            button_text = f"🔍 Analyze 1 PDF"
        else:
            button_text = f"🔍 Analyze & Upload {len(uploaded_pdfs)} PDFs"
        
        if st.button(button_text, type="primary"):
            success_count = 0
            failed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_pdf in enumerate(uploaded_pdfs):
                try:
                    status_text.text(f"📄 Processing {idx + 1}/{len(uploaded_pdfs)}: {uploaded_pdf.name}...")
                    progress_bar.progress((idx + 1) / len(uploaded_pdfs))
                    
                    import PyPDF2
                    import pdfplumber
                    import openai
                    openai.api_key = st.secrets["api_keys"]["open_ai"]
                    
                    # Enhanced PDF extraction with tables and structure
                    pdf_text = ""
                    tables_found = []
                    page_count = 0
                    
                    # Try pdfplumber first (better for tables)
                    # Reset file pointer for each file
                    uploaded_pdf.seek(0)
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
                            model="gpt-5",  # Upgraded to GPT-5 for better PDF analysis
                            messages=[{"role": "user", "content": analysis_prompt}],
                            max_completion_tokens=1000,
                            # Note: GPT-5 only supports default temperature (1)
                        )
                        
                        ai_analysis = response.choices[0].message.content
                        
                        # Display the analysis for this PDF
                        with st.expander(f"📄 {uploaded_pdf.name} - Analysis", expanded=(idx == 0)):
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_analysis)
                        st.info(f"✅ Extracted {len(pdf_text)} characters from {page_count} pages, found {len(tables_found)} tables")
                        
                        # Store in chat history (session state)
                        st.session_state.chat_history.append({
                            "q": f"Analyze PDF: {uploaded_pdf.name}", 
                            "a": ai_analysis
                        })
                        # Save to database (user-specific, persistent)
                        if hasattr(db, 'save_chat_history'):
                            try:
                                db.save_chat_history(user['id'], f"Analyze PDF: {uploaded_pdf.name}", ai_analysis)
                            except Exception:
                                pass
                        
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
                            success_count += 1
                        else:
                            failed_count += 1
                            st.error(f"❌ Failed to save '{uploaded_pdf.name}': {save_result.get('error', 'Unknown error')}")
                    else:
                        failed_count += 1
                        st.error(f"❌ Could not extract text from '{uploaded_pdf.name}'. Please ensure the PDF contains readable text.")
                        
                except Exception as e:
                    failed_count += 1
                    st.error(f"❌ Error processing '{uploaded_pdf.name}': {str(e)[:100]}")
            
            # Final summary
            progress_bar.empty()
            status_text.empty()
            
            if success_count > 0:
                st.success(f"✅ Successfully processed and saved {success_count} PDF(s) to shared library!")
                if failed_count > 0:
                    st.warning(f"⚠️ {failed_count} PDF(s) failed to process")
                
                # Refresh the page to update the PDF library count
                st.session_state.pdf_context = db.get_all_pdfs_text(user['id'])
                st.info("📄 PDF content is now stored and available for all future sessions!")
                st.rerun()
            else:
                st.error(f"❌ All {len(uploaded_pdfs)} PDF(s) failed to process. Please check the files and try again.")
    
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

def ai_insights_page():
    """AI Insights page with agent analysis and recommendations"""
    st.header("🧠 AI Insights")
    st.caption("Powered by specialized AI agents for portfolio analysis")
    
    if not AI_AGENTS_AVAILABLE:
        st.error("AI agents are not available. Please check the installation.")
        return
    
    # Get user data
    user = st.session_state.user
    db = st.session_state.db
    
    # Get holdings data
    holdings = get_cached_holdings(user['id'])
    
    if not holdings:
        st.info("No holdings found. Upload transaction files to see AI insights.")
        return
    
    # Get comprehensive context for AI analysis (same as AI Assistant has)
    pdf_context = db.get_all_pdfs_text(user['id'])
    pdf_count = len(db.get_user_pdfs(user['id']))
    
    # Get all transactions for the user
    try:
        all_transactions_response = db.supabase.table('user_transactions_detailed').select('*').eq('user_id', user['id']).order('transaction_date', desc=False).limit(1000).execute()
        all_transactions = all_transactions_response.data if all_transactions_response.data else []
    except:
        all_transactions = []
    
    # Get historical prices for all tickers in holdings (increase limit)
    historical_prices = {}
    historical_prices_fetched = 0
    historical_prices_missing = 0
    try:
        # Get unique tickers from all holdings
        unique_tickers = set()
        for holding in holdings:
            ticker = holding.get('ticker')
            if ticker:
                unique_tickers.add(ticker)
        
        # Fetch historical prices for all unique tickers (not just top 30)
        for ticker in unique_tickers:
            try:
                # Get stock_id
                stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
                if stock_response.data:
                    stock_id = stock_response.data[0]['id']
                    # Get 52 weeks of historical prices
                    hist_response = db.supabase.table('historical_prices').select('price_date, price').eq('stock_id', stock_id).order('price_date', desc=True).limit(52).execute()
                    if hist_response.data and len(hist_response.data) > 0:
                        historical_prices[ticker] = hist_response.data
                        historical_prices_fetched += 1
                    else:
                        historical_prices_missing += 1
            except:
                historical_prices_missing += 1
    except Exception as e:
        pass
    
    # Get stock master data for all holdings
    try:
        # Get all stock master records for holdings (get all unique tickers)
        tickers = list(set([h.get('ticker') for h in holdings if h.get('ticker')]))
        if tickers:
            # Fetch in batches if needed (Supabase .in_() has limits)
            stock_master = []
            for i in range(0, len(tickers), 100):  # Process 100 at a time
                batch = tickers[i:i+100]
                try:
                    stock_master_response = db.supabase.table('stock_master').select('*').in_('ticker', batch).execute()
                    if stock_master_response.data:
                        stock_master.extend(stock_master_response.data)
                except:
                    pass
        else:
            stock_master = []
    except:
        stock_master = []
    
    # Show context status with detailed information
    context_info = []
    if pdf_count > 0:
        context_info.append(f"📚 {pdf_count} PDF(s)")
    if all_transactions:
        context_info.append(f"📝 {len(all_transactions)} transactions")
    if historical_prices:
        context_info.append(f"📊 Historical prices: {len(historical_prices)} tickers (52 weeks each)")
    if stock_master:
        context_info.append(f"🏢 {len(stock_master)} stock master records")
    
    if context_info:
        st.info(f"**AI Context Active:** {' | '.join(context_info)}")
        if historical_prices_missing > 0:
            st.caption(f"ℹ️ Note: {historical_prices_missing} ticker(s) don't have historical price data yet. Use 'Fetch Historical Prices' button to add them.")
    else:
        st.caption("💡 Upload PDFs and transactions to enhance insights")
    
    # Create tabs for different AI insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Smart Recommendations", 
        "📊 Portfolio Analysis", 
        "🔍 Market Insights",
        "🔮 Scenario Analysis",
        "💡 Investment Recommendations",
        "⚙️ Agent Status"
    ])
    
    # Run AI analysis ONCE and cache it (all tabs will use the same result)
    # This prevents running 5 separate analyses (one per tab)
    analysis_cache_key = f"ai_analysis_{user['id']}_{len(holdings)}_{len(all_transactions)}"
    
    if analysis_cache_key not in st.session_state or not st.session_state.get('ai_analysis_complete', False):
        # Run analysis once for all tabs
        with st.spinner("🤖 AI agents analyzing your portfolio (this may take 30-60 seconds)..."):
            try:
                # Get user profile from database
                user_profile_data = db.get_user_profile(user['id'])
                user_profile = {
                    "user_id": user['id'],
                    "risk_tolerance": user_profile_data.get('risk_tolerance', 'moderate'),
                    "goals": user_profile_data.get('investment_goals', []),
                    "rebalancing_frequency": user_profile_data.get('rebalancing_frequency', 'quarterly'),
                    "tax_optimization": user_profile_data.get('tax_optimization', True),
                    "esg_investing": user_profile_data.get('esg_investing', False),
                    "international_exposure": user_profile_data.get('international_exposure', 20)
                }
                
                # Run comprehensive AI analysis with all context data (agents run in parallel)
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context, all_transactions, historical_prices, stock_master)
                
                # Cache the result
                st.session_state[analysis_cache_key] = analysis_result
                st.session_state['ai_analysis_complete'] = True
            except Exception as e:
                st.error(f"Error running AI analysis: {str(e)}")
                st.session_state[analysis_cache_key] = {"error": str(e)}
    else:
        # Use cached result
        analysis_result = st.session_state[analysis_cache_key]
    
    with tab1:
        st.subheader("🎯 Smart Recommendations")
        
        try:
            # Use cached analysis result
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                # Get recommendations from analysis result
                recommendations = analysis_result.get("investment_recommendations", [])
                
                # If no recommendations in result, try getting from agent manager cache
                if not recommendations:
                    try:
                        recommendations = get_ai_recommendations(5)
                    except:
                        recommendations = []
                
                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} AI recommendations")
                    
                    for i, rec in enumerate(recommendations, 1):
                        severity_color = {
                            "high": "🔴",
                            "medium": "🟡", 
                            "low": "🟢"
                        }.get(rec.get("severity", "low"), "🟢")
                        
                        with st.expander(f"{severity_color} {rec.get('title', 'Recommendation')}", expanded=(rec.get("severity") == "high")):
                            st.markdown(f"**Description:** {rec.get('description', 'No description')}")
                            st.markdown(f"**Recommendation:** {rec.get('recommendation', 'No recommendation')}")
                            
                            if rec.get("data"):
                                st.json(rec["data"])
                else:
                    st.info("🎉 No urgent recommendations found. Your portfolio looks well-balanced!")
        except Exception as e:
            st.error(f"Error running AI analysis: {str(e)}")
    
    with tab2:
        st.subheader("📊 Portfolio Analysis")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "portfolio_insights" in analysis_result:
                    # Get portfolio insights directly
                    portfolio_insights = analysis_result["portfolio_insights"]
                    
                    if portfolio_insights:
                        st.success(f"📈 Portfolio analysis complete - {len(portfolio_insights)} insights found")
                        
                        for insight in portfolio_insights:
                            severity_emoji = {
                                "high": "🚨",
                                "medium": "⚠️",
                                "low": "ℹ️"
                            }.get(insight.get("severity", "low"), "ℹ️")
                            
                            st.markdown(f"**{severity_emoji} {insight.get('title', 'Insight')}**")
                            st.markdown(f"{insight.get('description', 'No description')}")
                            st.markdown(f"*Recommendation: {insight.get('recommendation', 'No recommendation')}*")
                            
                            # Show additional data if available
                            if insight.get("data"):
                                with st.expander("📊 Detailed Analysis", expanded=False):
                                    st.json(insight["data"])
                            
                            st.markdown("---")
                    else:
                        st.info("No specific portfolio insights at this time.")
                else:
                    st.info("No specific portfolio insights at this time.")
                
        except Exception as e:
            st.error(f"Error getting portfolio insights: {str(e)}")
    
    with tab3:
        st.subheader("🔍 Market Insights")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "market_insights" in analysis_result:
                    # Get market insights directly
                    market_insights = analysis_result["market_insights"]
                    
                    if market_insights:
                        st.success(f"📊 Market analysis complete - {len(market_insights)} insights found")
                        
                        for insight in market_insights:
                            severity_emoji = {
                                "high": "🚨",
                                "medium": "⚠️",
                                "low": "ℹ️"
                            }.get(insight.get("severity", "low"), "ℹ️")
                            
                            st.markdown(f"**{severity_emoji} {insight.get('title', 'Market Insight')}**")
                            st.markdown(f"{insight.get('description', 'No description')}")
                            st.markdown(f"*Recommendation: {insight.get('recommendation', 'No recommendation')}*")
                            
                            # Show additional data if available
                            if insight.get("data"):
                                with st.expander("📊 Market Data", expanded=False):
                                    st.json(insight["data"])
                            
                            st.markdown("---")
                    else:
                        st.info("No specific market insights at this time.")
                else:
                    st.info("No specific market insights at this time.")
                
        except Exception as e:
            st.error(f"Error getting market insights: {str(e)}")
    
    with tab4:
        st.subheader("🔮 Scenario Analysis")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "scenario_insights" in analysis_result:
                    # Get scenario insights directly
                    scenario_insights = analysis_result["scenario_insights"]
                    
                    if scenario_insights:
                        st.success(f"🔮 Scenario analysis complete - {len(scenario_insights)} scenarios analyzed")
                        
                        # Group scenarios by type
                        scenario_types = {}
                        for insight in scenario_insights:
                            scenario_type = insight.get("type", "unknown")
                            if scenario_type not in scenario_types:
                                scenario_types[scenario_type] = []
                            scenario_types[scenario_type].append(insight)
                        
                        # Display scenarios by type
                        for scenario_type, scenarios in scenario_types.items():
                            st.markdown(f"**{scenario_type.replace('_', ' ').title()} Scenarios:**")
                            
                            for scenario in scenarios:
                                severity_emoji = {
                                    "high": "🚨",
                                    "medium": "⚠️",
                                    "low": "ℹ️"
                                }.get(scenario.get("severity", "low"), "ℹ️")
                                
                                with st.expander(f"{severity_emoji} {scenario.get('title', 'Scenario')}", expanded=(scenario.get("severity") == "high")):
                                    st.markdown(f"**{scenario.get('description', 'No description')}**")
                                    st.markdown(f"*Recommendation: {scenario.get('recommendation', 'No recommendation')}*")
                                    
                                    if scenario.get("data"):
                                        st.json(scenario["data"])
                                st.markdown("---")
                    else:
                        st.info("No scenario analysis available at this time.")
                else:
                    st.info("No scenario analysis available at this time.")
                
        except Exception as e:
            st.error(f"Error getting scenario insights: {str(e)}")
    
    with tab5:
        st.subheader("💡 Investment Recommendations")
        st.caption("AI-powered suggestions for new holdings to complement your portfolio")
        
        try:
            # Use cached analysis result (no need to run again)
            if not analysis_result or "error" in analysis_result:
                st.error(f"Analysis error: {analysis_result.get('error', 'Unknown error') if analysis_result else 'No analysis result'}")
            else:
                if "investment_recommendations" in analysis_result:
                    # Get investment recommendations directly
                    investment_recommendations = analysis_result["investment_recommendations"]
                    
                    if investment_recommendations:
                        st.success(f"💼 Found {len(investment_recommendations)} investment opportunities for you")
                        
                        # Group recommendations by type
                        recommendation_types = {
                            "sell_recommendation": {"title": "🔴 SELL Recommendations", "icon": "📉", "recommendations": [], "color": "red"},
                            "stock_recommendation": {"title": "📈 BUY: Stock Recommendations", "icon": "🏢", "recommendations": [], "color": "green"},
                            "mutual_fund_recommendation": {"title": "📊 BUY: Mutual Fund Recommendations", "icon": "💰", "recommendations": [], "color": "green"},
                            "pms_recommendation": {"title": "🎯 BUY: PMS Recommendations", "icon": "💼", "recommendations": [], "color": "green"},
                            "bond_recommendation": {"title": "🔐 BUY: Bond Recommendations", "icon": "📜", "recommendations": [], "color": "green"},
                            "diversification_opportunity": {"title": "🌈 Diversification Opportunities", "icon": "🎨", "recommendations": [], "color": "blue"},
                            "investment_recommendation": {"title": "💡 General Investment Opportunities", "icon": "✨", "recommendations": [], "color": "blue"}
                        }
                        
                        for rec in investment_recommendations:
                            rec_type = rec.get("type", "investment_recommendation")
                            if rec_type in recommendation_types:
                                recommendation_types[rec_type]["recommendations"].append(rec)
                        
                        # Display recommendations by type
                        for rec_type, rec_group in recommendation_types.items():
                            if rec_group["recommendations"]:
                                st.markdown(f"### {rec_group['icon']} {rec_group['title']}")
                                
                                for rec in rec_group["recommendations"]:
                                    # Create expandable card for each recommendation
                                    with st.expander(f"**{rec.get('title', 'Investment Opportunity')}**", expanded=True):
                                        # Severity indicator
                                        severity_emoji = {
                                            "high": "🔥 High Priority",
                                            "medium": "⚡ Medium Priority", 
                                            "low": "💡 Consider"
                                        }.get(rec.get("severity", "medium"), "💡 Consider")
                                        
                                        st.markdown(f"**Priority:** {severity_emoji}")
                                        
                                        # Description
                                        st.markdown("**📝 Why This Investment:**")
                                        st.markdown(rec.get('description', 'No description available'))
                                        
                                        # Recommendation/Action
                                        st.markdown("**🎯 Action Plan:**")
                                        st.markdown(rec.get('recommendation', 'No recommendation available'))
                                        
                                        # Investment details
                                        if rec.get('data'):
                                            data = rec['data']
                                            
                                            # Check if this is a SELL recommendation
                                            if data.get('action') == 'SELL':
                                                st.markdown("**🔴 SELL Details:**")
                                                
                                                col1, col2, col3 = st.columns(3)
                                                
                                                with col1:
                                                    if data.get('ticker'):
                                                        st.metric("Ticker to Sell", data['ticker'])
                                                    if data.get('current_holding_quantity'):
                                                        st.metric("Current Holding", f"{data['current_holding_quantity']:,} shares")
                                                
                                                with col2:
                                                    if data.get('suggested_sell_quantity'):
                                                        st.metric("Sell Quantity", f"{data['suggested_sell_quantity']:,} shares", 
                                                                delta=f"-{data.get('percentage_to_sell', 0)}%", delta_color="inverse")
                                                    if data.get('funds_freed'):
                                                        st.metric("Funds Freed", f"₹{data['funds_freed']:,.0f}")
                                                
                                                with col3:
                                                    if data.get('value_after_sale') is not None:
                                                        st.metric("Value After Sale", f"₹{data['value_after_sale']:,.0f}")
                                                    if data.get('current_loss'):
                                                        st.metric("Current Loss", f"₹{abs(data['current_loss']):,.0f}", 
                                                                delta=f"{data.get('loss_percentage', 0):.1f}%", delta_color="inverse")
                                                
                                                # Reason for sell
                                                if data.get('reason'):
                                                    st.error(f"⚠️ **Reason to Sell:** {data['reason']}")
                                                
                                                # Rebalancing strategy
                                                if data.get('rebalancing_strategy'):
                                                    st.success(f"♻️ **Rebalancing Plan:** {data['rebalancing_strategy']}")
                                                
                                                # Tax consideration
                                                if data.get('tax_consideration'):
                                                    st.info(f"💰 **Tax Impact:** {data['tax_consideration']}")
                                                
                                                # Why now
                                                if data.get('why_now'):
                                                    st.warning(f"⏰ **Why Sell Now:** {data['why_now']}")
                                            
                                            else:
                                                # BUY recommendation details
                                                st.markdown("**📊 Investment Details:**")
                                                
                                                col1, col2, col3 = st.columns(3)
                                                
                                                with col1:
                                                    if data.get('ticker'):
                                                        st.metric("Ticker", data['ticker'])
                                                    if data.get('asset_type'):
                                                        st.markdown(f"**Asset Type:** {data['asset_type']}")
                                                
                                                with col2:
                                                    if data.get('sector'):
                                                        st.markdown(f"**Sector:** {data['sector']}")
                                                    if data.get('risk_level'):
                                                        st.markdown(f"**Risk Level:** {data['risk_level']}")
                                                
                                                with col3:
                                                    if data.get('suggested_allocation_percentage'):
                                                        st.metric("Suggested Allocation", f"{data['suggested_allocation_percentage']}%")
                                                    if data.get('expected_return'):
                                                        st.markdown(f"**Expected Return:** {data['expected_return']}")
                                                
                                                # Investment thesis
                                                if data.get('investment_thesis'):
                                                    st.info(f"💡 **Investment Thesis:** {data['investment_thesis']}")
                                                
                                                # Why now
                                                if data.get('why_now'):
                                                    st.success(f"⏰ **Why Now:** {data['why_now']}")
                                                
                                                # Suggested amount
                                                if data.get('suggested_amount'):
                                                    st.markdown(f"**💵 Suggested Investment:** ₹{data['suggested_amount']:,.0f}")
                                        
                                        st.markdown("---")
                                
                                st.markdown("")  # Add spacing between groups
                    else:
                        st.info("No specific investment recommendations at this time. Your portfolio appears well-balanced!")
                else:
                    st.info("No investment recommendations available. Run analysis to generate recommendations.")
                
        except Exception as e:
            st.error(f"Error getting investment recommendations: {str(e)}")
    
    with tab6:
        st.subheader("⚙️ Agent Status")
        
        try:
            # Get agent status
            agent_manager = get_agent_manager()
            agent_status = agent_manager.get_agent_status()
            
            st.markdown("**🤖 AI Agent Status:**")
            
            for agent_id, status in agent_status.items():
                status_emoji = {
                    "active": "🟢",
                    "analyzing": "🟡",
                    "error": "🔴",
                    "initialized": "🔵"
                }.get(status.get("status", "unknown"), "⚪")
                
                st.markdown(f"**{status_emoji} {status.get('agent_name', agent_id)}**")
                st.markdown(f"Status: {status.get('status', 'unknown')}")
                st.markdown(f"Last Update: {status.get('last_update', 'never')}")
                
                if status.get("capabilities"):
                    st.markdown(f"Capabilities: {', '.join(status['capabilities'])}")
                
                st.markdown("---")
            
            # Show analysis summary
            summary = agent_manager.get_recommendations_summary()
            if summary and "summary" in summary:
                st.markdown("**📊 Analysis Summary:**")
                st.json(summary["summary"])
            
            # Show performance metrics
            try:
                from ai_agents.performance_optimizer import performance_optimizer
                perf_report = performance_optimizer.get_performance_report()
                
                st.markdown("**⚡ Performance Metrics:**")
                cache_stats = perf_report["cache_statistics"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
                with col2:
                    st.metric("Avg Response Time", f"{cache_stats['avg_response_time']:.2f}s")
                with col3:
                    st.metric("Cache Size", cache_stats['cache_size'])
                with col4:
                    st.metric("Total Requests", cache_stats['total_requests'])
                
                # Show optimization recommendations
                recommendations = perf_report.get("recommendations", [])
                if recommendations:
                    st.markdown("**💡 Optimization Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)
                        
            except ImportError:
                pass
                
        except Exception as e:
            st.error(f"Error getting agent status: {str(e)}")
    
    # Add refresh button
    if st.button("🔄 Refresh AI Analysis"):
        # Clear any cached analysis
        if 'agent_manager' in st.session_state:
            del st.session_state.agent_manager
        st.rerun()

def user_profile_page():
    """User Profile Settings page"""
    st.header("👤 Profile Settings")
    st.caption("Manage your investment preferences and goals")
    
    user = st.session_state.user
    db = st.session_state.db
    
    # Get current user profile
    user_profile = db.get_user_profile(user['id'])
    if not user_profile:
        st.error("Could not load user profile")
        return
    
    # Add info box at the top explaining personalization
    st.info("""
    💡 **Your profile directly impacts AI recommendations!** 
    
    The AI analyzes your risk tolerance, goals, and preferences to suggest investments that match YOUR needs.
    Update your settings below to get more personalized recommendations.
    """)
    
    # Create tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Investment Goals",
        "⚖️ Risk Preferences", 
        "📊 Profile Summary",
        "🤖 How AI Uses Your Profile"
    ])
    
    with tab1:
        st.subheader("🎯 Investment Goals")
        st.caption("Set and manage your financial goals")
        
        # Display current goals
        current_goals = user_profile.get('investment_goals', [])
        
        if current_goals:
            st.markdown("**Current Goals:**")
            for i, goal in enumerate(current_goals):
                with st.expander(f"Goal {i+1}: {goal.get('type', 'Unknown').title()}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Type:** {goal.get('type', 'Unknown')}")
                        st.markdown(f"**Target Amount:** ₹{goal.get('target_amount', 0):,.0f}")
                        st.markdown(f"**Timeline:** {goal.get('timeline_years', 0)} years")
                        st.markdown(f"**Current Progress:** ₹{goal.get('current_progress', 0):,.0f}")
                        
                        if goal.get('description'):
                            st.markdown(f"**Description:** {goal.get('description')}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_goal_{i}"):
                            result = db.delete_investment_goal(user['id'], goal.get('id'))
                            if result['success']:
                                st.success("Goal deleted!")
                                st.rerun()
                            else:
                                st.error(f"Error: {result['error']}")
        else:
            st.info("No investment goals set yet. Add your first goal below!")
        
        st.markdown("---")
        
        # Add new goal form
        st.markdown("**Add New Goal:**")
        
        with st.form("add_goal_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                goal_type = st.selectbox(
                    "Goal Type:",
                    ["retirement", "education", "emergency_fund", "home_purchase", "vacation", "other"],
                    key="new_goal_type"
                )
                
                target_amount = st.number_input(
                    "Target Amount (₹):",
                    min_value=10000,
                    max_value=100000000,
                    value=1000000,
                    step=10000,
                    key="new_target_amount"
                )
            
            with col2:
                timeline_years = st.number_input(
                    "Timeline (Years):",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="new_timeline"
                )
                
                current_progress = st.number_input(
                    "Current Progress (₹):",
                    min_value=0,
                    max_value=target_amount,
                    value=0,
                    step=10000,
                    key="new_progress"
                )
            
            description = st.text_area(
                "Description (Optional):",
                placeholder="Describe your goal...",
                key="new_description"
            )
            
            if st.form_submit_button("➕ Add Goal"):
                new_goal = {
                    "type": goal_type,
                    "target_amount": target_amount,
                    "timeline_years": timeline_years,
                    "current_progress": current_progress,
                    "description": description
                }
                
                result = db.add_investment_goal(user['id'], new_goal)
                if result['success']:
                    st.success("Goal added successfully!")
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    with tab2:
        st.subheader("⚖️ Risk Preferences")
        st.caption("Configure your risk tolerance and investment preferences")
        
        # Risk tolerance settings
        current_risk = user_profile.get('risk_tolerance', 'moderate')
        
        with st.form("risk_settings_form"):
            st.markdown("**Risk Tolerance:**")
            
            risk_options = {
                "conservative": {
                    "name": "Conservative",
                    "description": "Low risk, stable returns. Focus on capital preservation.",
                    "allocation": "40% Stocks, 50% Bonds, 10% Alternatives"
                },
                "moderate": {
                    "name": "Moderate", 
                    "description": "Balanced risk and return. Growth with some stability.",
                    "allocation": "60% Stocks, 30% Bonds, 10% Alternatives"
                },
                "aggressive": {
                    "name": "Aggressive",
                    "description": "High risk, high potential returns. Growth-focused.",
                    "allocation": "80% Stocks, 15% Bonds, 5% Alternatives"
                }
            }
            
            selected_risk = st.radio(
                "Select your risk tolerance:",
                list(risk_options.keys()),
                format_func=lambda x: f"{risk_options[x]['name']}: {risk_options[x]['description']}",
                index=list(risk_options.keys()).index(current_risk)
            )
            
            # Show allocation preview
            st.markdown("**Recommended Allocation:**")
            st.info(risk_options[selected_risk]['allocation'])
            
            # Additional preferences
            st.markdown("**Additional Preferences:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rebalancing_frequency = st.selectbox(
                    "Rebalancing Frequency:",
                    ["monthly", "quarterly", "semi_annually", "annually"],
                    index=1  # Default to quarterly
                )
                
                tax_optimization = st.checkbox(
                    "Enable Tax Optimization",
                    value=True,
                    help="AI will suggest tax-efficient strategies"
                )
            
            with col2:
                esg_investing = st.checkbox(
                    "ESG Investing Preference",
                    value=False,
                    help="Consider Environmental, Social, and Governance factors"
                )
                
                international_exposure = st.slider(
                    "International Exposure (%)",
                    min_value=0,
                    max_value=50,
                    value=20,
                    help="Percentage of portfolio in international markets"
                )
            
            if st.form_submit_button("💾 Save Risk Preferences"):
                # Update user profile
                profile_data = {
                    "risk_tolerance": selected_risk,
                    "rebalancing_frequency": rebalancing_frequency,
                    "tax_optimization": tax_optimization,
                    "esg_investing": esg_investing,
                    "international_exposure": international_exposure
                }
                
                result = db.update_user_profile(user['id'], profile_data)
                if result['success']:
                    st.success("Risk preferences updated successfully!")
                    # Update session state
                    st.session_state.user = result['user']
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    with tab3:
        st.subheader("📊 Profile Summary")
        st.caption("Overview of your investment profile")
        
        # Display current profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information:**")
            st.markdown(f"**Name:** {user_profile.get('full_name', 'N/A')}")
            st.markdown(f"**Username:** {user_profile.get('username', 'N/A')}")
            st.markdown(f"**Email:** {user_profile.get('email', 'N/A')}")
            st.markdown(f"**Member Since:** {user_profile.get('created_at', 'N/A')[:10] if user_profile.get('created_at') else 'N/A'}")
        
        with col2:
            st.markdown("**Investment Preferences:**")
            st.markdown(f"**Risk Tolerance:** {user_profile.get('risk_tolerance', 'moderate').title()}")
            st.markdown(f"**Rebalancing:** {user_profile.get('rebalancing_frequency', 'quarterly').title()}")
            st.markdown(f"**Tax Optimization:** {'Yes' if user_profile.get('tax_optimization') else 'No'}")
            st.markdown(f"**ESG Investing:** {'Yes' if user_profile.get('esg_investing') else 'No'}")
            st.markdown(f"**International Exposure:** {user_profile.get('international_exposure', 20)}%")
        
        # Goals summary
        goals = user_profile.get('investment_goals', [])
        if goals:
            st.markdown("**Investment Goals Summary:**")
            
            total_target = sum(goal.get('target_amount', 0) for goal in goals)
            total_progress = sum(goal.get('current_progress', 0) for goal in goals)
            progress_pct = (total_progress / total_target * 100) if total_target > 0 else 0
            
            st.markdown(f"**Total Goals:** {len(goals)}")
            st.markdown(f"**Total Target:** ₹{total_target:,.0f}")
            st.markdown(f"**Total Progress:** ₹{total_progress:,.0f} ({progress_pct:.1f}%)")
            
            # Progress bar
            st.progress(progress_pct / 100)
        else:
            st.info("No investment goals set yet.")
        
        # AI Agent Integration
        if AI_AGENTS_AVAILABLE:
            st.markdown("**AI Agent Integration:**")
            st.success("✅ AI agents are using your profile for personalized recommendations")
            
            # Show how profile affects AI recommendations
            st.markdown("**How your profile affects AI recommendations:**")
            st.markdown(f"• **Risk-based allocation:** AI uses your {user_profile.get('risk_tolerance', 'moderate')} risk tolerance")
            st.markdown(f"• **Goal-based analysis:** AI considers your {len(goals)} investment goals")
            st.markdown(f"• **Tax optimization:** {'Enabled' if user_profile.get('tax_optimization') else 'Disabled'}")
            st.markdown(f"• **ESG preferences:** {'Considered' if user_profile.get('esg_investing') else 'Not considered'}")
        else:
            st.warning("⚠️ AI agents not available - profile settings won't affect recommendations")
            
            with st.expander("ℹ️ How to enable AI agents"):
                st.markdown("""
                **To enable AI agents, ensure:**
                
                1. ✅ All files in `ai_agents/` directory exist:
                   - `__init__.py`
                   - `base_agent.py`
                   - `communication.py`
                   - `portfolio_agent.py`
                   - `market_agent.py`
                   - `strategy_agent.py`
                   - `scenario_agent.py`
                   - `agent_manager.py`
                   - `performance_optimizer.py`
                
                2. ✅ Required packages installed:
                   - `pandas`, `numpy` (already in requirements.txt)
                
                3. ✅ Restart the Streamlit application:
                   ```bash
                   streamlit run web_agent.py
                   ```
                
                4. ✅ Check the terminal for any import errors
                
                **Note**: If you're on Streamlit Cloud, the app will auto-restart after deployment.
                """)

def process_file_with_ai(uploaded_file, filename, user_id):
    """
    Universal AI-powered file processor
    Handles CSV, PDF, Excel, and any other file format
    Returns extracted transactions ready for database storage
    """
    if not AI_AGENTS_AVAILABLE:
        st.error("🚫 AI agents not available. Please check your configuration.")
        return None
    
    try:
        # Initialize AI File Processor
        file_processor = AIFileProcessor()
        
        # Process file with AI
        with st.spinner(f"🤖 AI is analyzing {filename} and extracting transactions..."):
            transactions = file_processor.process_file(uploaded_file, filename)
        
        if not transactions:
            st.warning(f"⚠️ No transactions found in {filename}")
            return None
        
        # Enhance transactions with metadata
        for trans in transactions:
            # If price is 0 or missing, it will be fetched automatically later
            if not trans.get('price') or trans['price'] == 0:
                trans['price'] = 0  # Will trigger automatic price fetching
        
        return transactions
        
    except Exception as e:
        st.error(f"❌ Error processing file {filename}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []  # Return empty list instead of None

def upload_files_page():
    """Enhanced upload more files page with AI PDF extraction"""
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
    
    # Initialize uploaded_files_ai to None
    uploaded_files_ai = None
    
    # AI-powered file extraction section
    if AI_FILE_EXTRACTION_ENABLED:
        st.markdown("""
        <div class="upload-section">
            <h3>🤖 AI-Powered Transaction Extraction</h3>
            <p>Upload ANY file type (PDF, CSV, Excel, Images) and let AI automatically extract transaction data!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **AI processes all file types**: CSV, PDF, Excel (XLSX/XLS), Text files, and Images (JPG, PNG, JPEG). Just upload and let AI handle the rest!")
        
        # Universal file uploader (AI extraction)
        uploaded_files_ai = st.file_uploader(
            "📁 Choose files to upload (CSV, PDF, Excel, Images, etc.)",
            type=['csv', 'pdf', 'xlsx', 'xls', 'txt', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="ai_file_uploader",
            help="Upload transaction files in any format - AI will extract the data automatically. Supports images too!"
        )
        
    # Process AI-extracted files if any were uploaded
    if uploaded_files_ai:
            for uploaded_file in uploaded_files_ai:
                file_type = uploaded_file.name.split('.')[-1].upper()
                st.markdown(f"**🤖 AI Processing: {uploaded_file.name}** ({file_type})")
                
                # Extract transactions using AI (works for ALL file types)
                transactions = process_file_with_ai(uploaded_file, uploaded_file.name, user['id'])
                
                if transactions:
                    st.success(f"✅ AI extracted {len(transactions)} transactions from {uploaded_file.name}")
                    
                    # Display extracted transactions
                    with st.expander(f"View {len(transactions)} extracted transactions", expanded=True):
                        # Create DataFrame for display
                        df = pd.DataFrame(transactions)
                        
                        # Display summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", len(transactions))
                        with col2:
                            total_value = sum(t.get('quantity', 0) * t.get('price', 0) for t in transactions)
                            st.metric("Total Value", f"₹{total_value:,.0f}" if total_value > 0 else "Price to be fetched")
                        with col3:
                            asset_types = set(t.get('asset_type', 'unknown') for t in transactions)
                            st.metric("Asset Types", len(asset_types))
                        with col4:
                            channels = set(t.get('channel', 'unknown') for t in transactions)
                            st.metric("Channels", len(channels))
                        
                        # Display transactions table
                        st.dataframe(df, use_container_width=True)
                        
                        # Upload to database button
                        if st.button(f"📥 Upload {len(transactions)} transactions to portfolio", key=f"upload_{uploaded_file.name}"):
                            # Process and upload transactions
                            success_count = 0
                            for transaction in transactions:
                                try:
                                    # Convert AI extracted data to database format
                                    transaction_data = {
                                        'user_id': user['id'],
                                        'portfolio_id': portfolio['id'],  # Add portfolio_id
                                        'ticker': transaction.get('ticker'),
                                        'stock_name': transaction.get('stock_name') or transaction.get('ticker', 'Unknown'),  # Fix null stock_name
                                        'scheme_name': transaction.get('scheme_name'),
                                        'quantity': transaction.get('quantity', 0),
                                        'price': transaction.get('price', 0),  # 0 triggers auto-fetch
                                        'transaction_date': transaction.get('date'),
                                        'transaction_type': transaction.get('transaction_type', 'buy'),
                                        'asset_type': transaction.get('asset_type', 'stock'),
                                        'channel': transaction.get('channel', 'Direct'),
                                        'sector': transaction.get('sector', 'Unknown'),  # Fix null sector
                                        'filename': uploaded_file.name
                                    }
                                    result = db.add_transaction(transaction_data)
                                    if result.get('success'):
                                        success_count += 1
                                except Exception as e:
                                    st.error(f"Error uploading transaction: {e}")
                            
                            if success_count > 0:
                                st.success(f"✅ Successfully uploaded {success_count} transactions to your portfolio!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to upload transactions")
    else:
        # File uploader for CSV/Excel (no AI needed)
        uploaded_files = st.file_uploader(
            "📁 Choose CSV or Excel files to upload",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Select CSV or Excel files with required columns: date, ticker, quantity, transaction_type, price, stock_name, sector, channel",
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
