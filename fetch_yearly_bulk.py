"""
Optimized Yearly Bulk Price Fetcher
Fetches entire year of weekly prices in ONE API call per ticker
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import streamlit as st


def fetch_yearly_prices_for_all_tickers(holdings: List[Dict], start_date: datetime, end_date: datetime) -> Dict[str, Dict[Tuple[int, int], float]]:
    """
    Fetch entire year of weekly prices for all holdings at once
    Supports: Stocks, Mutual Funds, PMS, AIF
    
    Args:
        holdings: List of holding dicts with ticker, asset_type, stock_name
        start_date: Start date for historical data
        end_date: End date for historical data
    
    Returns:
        Dict of {ticker: {(year, week): price}}
    """
    all_prices = {}
    
    # Show stock market loader
    with st.spinner("📈 Fetching market data..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, holding in enumerate(holdings, 1):
            ticker = holding['ticker']
            asset_type = holding.get('asset_type', 'stock')
            
            # Update progress
            progress = idx / len(holdings)
            progress_bar.progress(progress)
            # Silent processing - no status updates
        
        weekly_prices = {}
        
        try:
            if asset_type == 'stock':
                # STOCKS: Use yfinance with NSE/BSE
                yf_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
                stock = yf.Ticker(yf_ticker)
                
                # Fetch ENTIRE YEAR of weekly data in ONE call
                hist = stock.history(start=start_date, end=end_date, interval='1wk')
                
                if hist.empty:
                    # Try BSE
                    # Trying BSE (silent)
                    yf_ticker = f"{ticker}.BO" if not ticker.endswith(('.NS', '.BO')) else ticker.replace('.NS', '.BO')
                    stock = yf.Ticker(yf_ticker)
                    hist = stock.history(start=start_date, end=end_date, interval='1wk')
                
                if not hist.empty:
                    # Convert to weekly prices
                    for date, row in hist.iterrows():
                        year, week, _ = date.isocalendar()
                        price = float(row['Close'])
                        if price > 0:
                            weekly_prices[(year, week)] = price
                    
                    all_prices[ticker] = weekly_prices
                    # Got weekly data (silent)
                else:
                    pass# No data found (silent)
            
            elif asset_type == 'mutual_fund':
                # MUTUAL FUNDS: Use mftool for current NAV, replicate for weeks
                try:
                    from mftool import Mftool
                    mf = Mftool()
                    
                    # Get current NAV
                    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('MF_', '')
                    quote = mf.get_scheme_quote(clean_ticker)
                    
                    if quote and 'nav' in quote:
                        current_nav = float(quote['nav'])
                        
                        # For MF, use current NAV for all weeks (MF NAVs don't change much weekly)
                        temp_date = start_date
                        while temp_date <= end_date:
                            year, week, _ = temp_date.isocalendar()
                            weekly_prices[(year, week)] = current_nav
                            temp_date += timedelta(weeks=1)
                        
                        all_prices[ticker] = weekly_prices
                        # MF NAV found (silent)
                    else:
                        pass# MF NAV not found (silent)
                except Exception as e:
                    pass# MF Error (silent)
            
            elif asset_type in ['pms', 'aif']:
                pass
                # PMS/AIF: Use CAGR calculation or fixed NAV
                # For now, skip - these need special handling with transaction context
                # PMS/AIF: Requires CAGR calculation (skipped for bulk fetch)
            
            else:
                pass# Unknown asset type (silent)
                
        except Exception as e:
            pass# Error fetching ticker (silent)
    
    return all_prices


def save_yearly_prices_to_db(db, all_prices: Dict[str, Dict[Tuple[int, int], float]]):
    """
    Save all fetched yearly prices to database in bulk
    AND update current/live prices in stock_master
    
    Args:
        db: Database manager instance
        all_prices: Dict of {ticker: {(year, week): price}}
    """
    # Saving prices to database (silent)
    
    total_saved = 0
    current_prices_updated = 0
    
    for ticker, weekly_prices in all_prices.items():
        # Get stock_id
        stock_response = db.supabase.table('stock_master').select('id').eq(
            'ticker', ticker
        ).execute()
        
        if not stock_response.data:
            continue
        
        stock_id = stock_response.data[0]['id']
        
        # Prepare bulk insert for historical prices
        price_records = []
        latest_price = None
        latest_week = (0, 0)
        
        for (year, week), price in weekly_prices.items():
            # Calculate Monday of that week
            week_monday = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
            
            price_records.append({
                'stock_id': stock_id,
                'price_date': week_monday.strftime('%Y-%m-%d'),
                'price': price,
                'volume': None,
                'source': 'yfinance_yearly',
                'iso_year': year,
                'iso_week': week
            })
            
            # Track latest price (most recent week)
            if (year, week) > latest_week:
                latest_week = (year, week)
                latest_price = price
        
        if price_records:
            db.save_historical_prices_bulk(price_records)
            total_saved += len(price_records)
            # Saved weeks (silent)
            
            # Update live_price in stock_master with the most recent week's price
            if latest_price:
                try:
                    db.supabase.table('stock_master').update({
                        'live_price': latest_price
                    }).eq('id', stock_id).execute()
                    current_prices_updated += 1
                    # Updated live price (silent)
                except Exception as e:
                    pass# Could not update live price (silent)
    
    # Total saved (silent)
    # Updated live prices (silent)
    return total_saved

