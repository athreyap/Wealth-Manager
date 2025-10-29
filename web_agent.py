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
import sys
import os
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
        from enhanced_price_fetcher import EnhancedPriceFetcher
        price_fetcher = EnhancedPriceFetcher()
        
        holdings = db.get_user_holdings(user_id)
        corporate_actions = []
        
        for holding in holdings:
            asset_type = holding.get('asset_type')
            
            # Only check stocks (not MF, bonds, etc.)
            if asset_type != 'stock':
                continue
            
            ticker = holding.get('ticker')
            avg_price = holding.get('average_price', 0)
            current_price = holding.get('current_price')
            quantity = holding.get('total_quantity', 0)
            
            # Skip if current_price is None or 0 (can't calculate ratio)
            if avg_price == 0 or current_price is None or current_price == 0:
                continue
            
            # Convert to float to handle any type issues
            avg_price = float(avg_price)
            current_price = float(current_price)
            
            # Calculate ratio
            ratio = avg_price / current_price
            
            # If ratio >= 1.5, likely a corporate action
            # (stock split, bonus shares, or reverse split)
            if ratio >= 1.5:
                split_ratio = round(ratio)
                
                corporate_actions.append({
                    'ticker': ticker,
                    'stock_name': holding.get('stock_name'),
                    'stock_id': holding.get('stock_id'),
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'quantity': quantity,
                    'ratio': ratio,
                    'split_ratio': split_ratio,
                    'action_type': 'split' if ratio > 1 else 'reverse_split'
                })
        
        if corporate_actions:
            print(f"[CORPORATE_ACTIONS] Detected {len(corporate_actions)} stocks with splits/bonus")
            for action in corporate_actions:
                print(f"  - {action['ticker']}: 1:{action['split_ratio']} {action['action_type']}")
        
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
                st.caption(f"   🔄 Using OpenAI (gpt-4o) for ticker resolution...")
                response = client.chat.completions.create(
                    model="gpt-4o",  # Better model for accurate verification
                    messages=[
                        {"role": "system", "content": "You are a financial ticker verification expert with access to real-time data. For each ticker, search online databases and verify it works with yfinance or mftool APIs. Return ONLY valid JSON with unique tickers for each holding."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.0  # Zero temperature for deterministic results
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
                
                def normalize_date(date_str):
                    """Convert ANY date format to YYYY-MM-DD"""
                    try:
                        if pd.isna(date_str) or not date_str:
                            return ''
                        
                        date_str = str(date_str).strip()
                        
                        # Remove time part ONLY if AM/PM is present (e.g., "18-03-2021 09:34 AM" -> "18-03-2021")
                        # Preserve formats like "10 Oct 2025" (don't split these!)
                        if ' AM' in date_str.upper() or ' PM' in date_str.upper():
                            date_str = date_str.split()[0]
                        
                        # Use pandas to_datetime - handles most formats automatically!
                        dt = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                        
                        if pd.notna(dt):
                            return dt.strftime('%Y-%m-%d')
                        
                        # If pandas failed, return empty
                        return ''
                    except:
                        return ''
                
                imported = 0
                skipped = 0
                errors = 0
                
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
                        
                        transaction_date = normalize_date(row['date'])
                        
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

                        elif not result.get('success'):
                            # Show why it failed
                            error_msg = result.get('error', 'Unknown error')
                            if 'duplicate' in error_msg.lower():
                                skipped += 1  # Count duplicates as skipped
                                pass  # Silent for duplicates
                            elif '502' in error_msg or 'bad gateway' in error_msg.lower():
                                # 502 error - Supabase server issue
                                st.warning(f"   ⚠️ Database server error (502). Retrying in 2 seconds...")
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
                                        all_resolved.update(batch_resolved)
                            
                            # Process PMS/AIF/BONDS one by one (need more careful verification)
                            if others_to_resolve:
                                st.caption(f"   🔍 Resolving {len(others_to_resolve)} PMS/AIF/Bond tickers (one-by-one)...")
                                
                                for ticker, info in others_to_resolve.items():
                                    single_pair = {ticker: info}
                                    
                                    # Use AI to get verified ticker
                                    single_resolved = ai_resolve_tickers_from_names(single_pair)
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
                                                asset_type = stocks_to_resolve.get(original_ticker, {}).get('asset_type') or \
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
                                            db.supabase.table('stock_master').update({
                                                'sector': sector
                                            }).eq('ticker', verified_ticker).execute()
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
                        for trans in mf_transactions:
                            scheme_name = trans.get('stock_name', '')
                            current_ticker = trans.get('ticker', '')
                            
                            # Skip if already has numeric AMFI code
                            if current_ticker.isdigit() and len(current_ticker) == 6:
                                continue
                            
                            # Search mftool for AMFI code
                            result = search_mftool_for_amfi_code(scheme_name)
                            
                            if result:
                                amfi_code = result['ticker']
                                matched_name = result['name']
                                confidence = result.get('match_confidence', 0)
                                
                                # Check if target ticker already exists in stock_master
                                existing_stock = db.supabase.table('stock_master').select('id').eq(
                                    'ticker', amfi_code
                                ).execute()
                                
                                # Get the old stock_master record
                                old_stock = db.supabase.table('stock_master').select('id').eq(
                                    'ticker', current_ticker
                                ).execute()
                                
                                if existing_stock.data:
                                    # Target AMFI code already exists, point transactions to it
                                    target_stock_id = existing_stock.data[0]['id']
                                    
                                    if old_stock.data:
                                        old_stock_id = old_stock.data[0]['id']
                                        
                                        # Update all transactions to point to existing stock
                                        db.supabase.table('user_transactions').update({
                                            'stock_id': target_stock_id
                                        }).eq('stock_id', old_stock_id).execute()
                                        
                                        # Delete old stock_master record
                                        db.supabase.table('stock_master').delete().eq('id', old_stock_id).execute()
                                        
                                        st.caption(f"      ✓ MF: {scheme_name[:40]} → {amfi_code} (merged)")
                                        mf_updated += 1
                                else:
                                    # Target doesn't exist, safe to update
                                    if old_stock.data:
                                        db.supabase.table('stock_master').update({
                                            'ticker': amfi_code,
                                            'stock_name': matched_name,
                                            'sector': 'Mutual Fund'
                                        }).eq('id', old_stock.data[0]['id']).execute()
                                        
                                        st.caption(f"      ✓ MF: {scheme_name[:40]} → {amfi_code} ({confidence:.0%} match)")
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
                
                # Auto-fetch prices AND 52-week historical data
                if imported > 0:
                    st.caption(f"   📊 Fetching current prices and 52-week historical data...")
                    
                    try:
                        holdings = db.get_user_holdings(user_id)
                        
                        if holdings:
                            # Get unique tickers and asset types
                            unique_tickers = list(set([h['ticker'] for h in holdings if h.get('ticker')]))
                            asset_types = {h['ticker']: h.get('asset_type', 'stock') for h in holdings if h.get('ticker')}
                            
                            st.caption(f"      📈 Fetching comprehensive data for {len(unique_tickers)} holdings...")
                            
                            # Fetch current prices + 52-week historical data
                            db.bulk_process_new_stocks_with_comprehensive_data(
                                tickers=unique_tickers,
                                asset_types=asset_types
                            )
                            
                            st.caption(f"   ✅ Updated current prices and 52-week data for {len(unique_tickers)} holdings")
                            
                    except Exception as e:
                        st.caption(f"   ⚠️ Price/historical fetching skipped: {str(e)[:50]}")
                
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
                    holdings_needing_update = should_update_prices_today(holdings)
                    
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
                        holdings_needing_update = should_update_prices_today(holdings)
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
        
        # Find stocks with stale prices (current_price == avg_price or 0% return)
        stale_prices = []
        bonds_to_update = []
        for h in holdings:
            cp = h.get('current_price', 0)
            ap = h.get('average_price', 0)
            asset_type = h.get('asset_type', '')
            
            # Special handling for bonds - check if price seems wrong
            if asset_type == 'bond':
                # Bonds often show wrong prices, check if current price is suspiciously low
                ticker = (h.get('ticker') or '').lower()
                name = (h.get('stock_name') or '').lower()
                is_sgb = any(k in ticker or k in name for k in ['sgb', 'gold bond', 'goldbond', 'sovereign', 'sr-'])
                
                # SGBs should be around ₹14,000-15,000, not ₹2,000
                if is_sgb and cp < 9000:
                    bonds_to_update.append(h)
                elif cp is None or cp == 0 or cp < ap * 0.5:  # Price is < 50% of avg = likely wrong
                    bonds_to_update.append(h)
            
            # If current_price is None, 0, or exactly equals avg_price → likely stale
            if cp is None or cp == 0 or abs(cp - ap) < 0.01:
                stale_prices.append(h)
        
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
                    else:
                        st.warning(f"⚠️ No bond prices updated (AI may be unavailable)")
                    st.rerun()
        
        if stale_prices:
            button_text = f"🔄 Update {len(stale_prices)} Prices"
            help_text = f"{len(stale_prices)} holdings have stale/missing prices (showing 0% return)"
        else:
            button_text = "✅ All Current"
            help_text = "All prices are up-to-date"
        
        if st.button(button_text, help=help_text, disabled=(len(stale_prices) == 0)):
            with st.spinner(f"Fetching prices for {len(stale_prices)} holdings..."):
                if stale_prices:
                    # Try to update stale prices using enhanced fetcher with AI fallback
                    from enhanced_price_fetcher import EnhancedPriceFetcher
                    price_fetcher = EnhancedPriceFetcher()
                    
                    updated = 0
                    for holding in stale_prices:
                        ticker = holding.get('ticker')
                        asset_type = holding.get('asset_type')
                        stock_name = holding.get('stock_name')
                        
                        # Fetch current price with AI fallback
                        price, source = price_fetcher.get_current_price(ticker, asset_type, stock_name)
                        
                        if price and price > 0:
                            db._store_current_price(ticker, price, asset_type)
                            updated += 1
                    
                    if updated > 0:
                        st.success(f"✅ Updated {updated}/{len(stale_prices)} prices (some may be delisted)")
                    else:
                        st.warning(f"⚠️ No prices updated (stocks may be delisted or AI unavailable)")
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
    
    # Load chat history from database (user-specific)
    try:
        db_chat_history = db.get_user_chat_history(user['id'], limit=50)
        if db_chat_history:
            # Convert database format to session state format
            st.session_state.chat_history = [
                {"q": chat['question'], "a": chat['answer']}
                for chat in reversed(db_chat_history)  # Reverse to show oldest first
            ]
    except Exception as e:
        st.caption(f"⚠️ Could not load chat history: {str(e)[:100]}")
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
                openai.api_key = st.secrets["api_keys"]["open_ai"]
                
                # Safe float conversion helper
                def safe_float(value, default=0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                # Create streamlined portfolio data context (limit to reduce tokens)
                raw_data_context = "\n\n📊 PORTFOLIO DATA (Top Holdings):\n"
                
                # Limit to top 30 holdings to reduce token usage
                top_holdings = holdings[:30]
                for idx, holding in enumerate(top_holdings, 1):
                    current_price = holding.get('current_price') or holding.get('average_price', 0)
                    current_value = safe_float(current_price, 0) * safe_float(holding.get('total_quantity'), 0)
                    investment = safe_float(holding.get('total_quantity'), 0) * safe_float(holding.get('average_price'), 0)
                    pnl = current_value - investment
                    pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0
                    
                    # Compact format to reduce tokens
                    raw_data_context += f"{idx}. {holding.get('ticker', 'N/A')} ({holding.get('stock_name', 'N/A')[:30]}) | "
                    raw_data_context += f"Qty: {holding.get('total_quantity', 0):,.0f} | "
                    raw_data_context += f"Avg: ₹{holding.get('average_price', 0):,.0f} | "
                    raw_data_context += f"Curr: ₹{current_price:,.0f} | "
                    raw_data_context += f"P&L: {pnl_pct:+.1f}%\n"
                
                if len(holdings) > 30:
                    raw_data_context += f"\n... and {len(holdings) - 30} more holdings\n"
                
                # Get transactions data (limited to reduce tokens)
                transactions_context = "\n\n📝 RECENT TRANSACTIONS:\n"
                try:
                    # Limit to 20 most recent transactions
                    all_transactions = db.supabase.table('user_transactions').select('*').eq('user_id', user['id']).order('transaction_date', desc=True).limit(20).execute()
                    
                    if all_transactions.data:
                        for trans in all_transactions.data:
                            # Compact format
                            transactions_context += f"{trans.get('transaction_date', 'N/A')} | "
                            transactions_context += f"{trans.get('ticker', 'N/A')} | "
                            transactions_context += f"{trans.get('transaction_type', 'N/A')} | "
                            transactions_context += f"Qty: {trans.get('quantity', 0)} | "
                            transactions_context += f"₹{trans.get('price', 0):,.0f}\n"
                    else:
                        transactions_context += "No transactions found.\n"
                except Exception as e:
                    transactions_context += f"Error loading transactions.\n"
                
                # Include PDF context
                pdf_context_text = ""
                
                # Include PDF summaries (limited to reduce tokens)
                recent_pdfs = db.get_user_pdfs(user['id'])
                if recent_pdfs:
                    # Limit to last 5 PDFs and truncate summaries
                    pdf_context_text += f"\n\n📚 PDF DOCUMENTS ({len(recent_pdfs)} total):"
                    for pdf in recent_pdfs[:5]:  # Only last 5 PDFs
                        pdf_context_text += f"\n\n📄 {pdf['filename']}:"
                        if pdf.get('ai_summary'):
                            # Truncate summary to 500 chars to save tokens
                            summary = pdf.get('ai_summary', 'No summary')
                            pdf_context_text += f"\n{summary[:500]}{'...' if len(summary) > 500 else ''}"
                        else:
                            # Truncate PDF text to 500 chars
                            pdf_text = pdf.get('pdf_text', '')
                            if pdf_text:
                                pdf_context_text += f"\n{pdf_text[:500]}..."
                    
                    if len(recent_pdfs) > 5:
                        pdf_context_text += f"\n\n... and {len(recent_pdfs) - 5} more PDFs"
                
                # If no PDFs, note that
                if not recent_pdfs:
                    pdf_context_text = "\n\n📄 No documents uploaded yet."
                
                # Truncate portfolio summary if too long
                summary_text = portfolio_summary[:2000] + "..." if len(portfolio_summary) > 2000 else portfolio_summary
                
                full_context = f"""📊 PORTFOLIO DATA:
{raw_data_context}

📈 SUMMARY:
{summary_text}

📝 RECENT TRANSACTIONS:
{transactions_context}

📚 RESEARCH DOCUMENTS:
{pdf_context_text}

❓ QUESTION: {user_question}

💡 Analyze using portfolio data, transactions, and PDF insights."""
                
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
                
                # Store in chat history (session state)
                st.session_state.chat_history.append({
                    "q": user_question,
                    "a": ai_response
                })
                
                # Save to database (user-specific, persistent)
                db.save_chat_history(user['id'], user_question, ai_response)
                
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
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": analysis_prompt}],
                                temperature=0.7,
                                max_tokens=800
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
                            db.save_chat_history(user['id'], f"Analyze PDF: {pdf['filename']}", fresh_analysis)
                            
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
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": analysis_prompt}],
                            temperature=0.7,
                            max_tokens=1000
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
                        db.save_chat_history(user['id'], f"Analyze PDF: {uploaded_pdf.name}", ai_analysis)
                        
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
    
    # Get PDF context for enhanced AI analysis
    pdf_context = db.get_all_pdfs_text(user['id'])
    pdf_count = len(db.get_user_pdfs(user['id']))
    
    # Show PDF context status
    if pdf_count > 0:
        st.info(f"📚 **PDF Context Active:** {pdf_count} PDF(s) from shared library will be included in AI analysis")
    else:
        st.caption("💡 Upload PDFs in the AI Assistant page to enhance insights with research documents")
    
    # Create tabs for different AI insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Smart Recommendations", 
        "📊 Portfolio Analysis", 
        "🔍 Market Insights",
        "🔮 Scenario Analysis",
        "💡 Investment Recommendations",
        "⚙️ Agent Status"
    ])
    
    with tab1:
        st.subheader("🎯 Smart Recommendations")
        
        # Run AI analysis
        with st.spinner("🤖 AI agents analyzing your portfolio..."):
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
                
                # Run comprehensive AI analysis with PDF insights
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context)
                
                if "error" in analysis_result:
                    st.error(f"Analysis error: {analysis_result['error']}")
                else:
                    # Display top recommendations
                    recommendations = get_ai_recommendations(5)
                    
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
            # Run fresh portfolio analysis
            with st.spinner("🤖 AI analyzing your portfolio..."):
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
                
                # Get portfolio-specific insights from the analysis with PDF context
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context)
                
                if analysis_result and "portfolio_insights" in analysis_result:
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
            # Run fresh market analysis
            with st.spinner("🤖 AI analyzing market conditions..."):
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
                
                # Get market-specific insights from the analysis with PDF context
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context)
                
                if analysis_result and "market_insights" in analysis_result:
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
            # Run fresh scenario analysis
            with st.spinner("🤖 AI running scenario analysis..."):
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
                
                # Get scenario-specific insights from the analysis with PDF context
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context)
                
                if analysis_result and "scenario_insights" in analysis_result:
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
            # Run fresh analysis to get recommendations
            with st.spinner("🤖 AI analyzing investment opportunities..."):
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
                
                # Get investment recommendations from analysis with PDF insights  
                analysis_result = run_ai_analysis(holdings, user_profile, pdf_context)
                
                if analysis_result and "investment_recommendations" in analysis_result:
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
    
    # AI-powered file extraction section
    if AI_FILE_EXTRACTION_ENABLED:
        st.markdown("""
        <div class="upload-section">
            <h3>🤖 AI-Powered Transaction Extraction</h3>
            <p>Upload ANY file type (PDF, CSV, Excel, Images) and let AI automatically extract transaction data!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **AI processes all file types**: CSV, PDF, Excel (XLSX/XLS), Text files, and Images (JPG, PNG, JPEG). Just upload and let AI handle the rest!")
        
        # Universal file uploader
    uploaded_files = st.file_uploader(
            "📁 Choose files to upload (CSV, PDF, Excel, Images, etc.)",
            type=['csv', 'pdf', 'xlsx', 'xls', 'txt', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="ai_file_uploader",
            help="Upload transaction files in any format - AI will extract the data automatically. Supports images too!"
        )
        
    if uploaded_files:
            for uploaded_file in uploaded_files:
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
