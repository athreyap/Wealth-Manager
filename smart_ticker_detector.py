"""
Smart Ticker Detection System
Auto-detects if ticker is NSE/BSE stock, Mutual Fund, PMS, or AIF
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any

def detect_ticker_type(ticker: str) -> str:
    """
    Detect what type of asset the ticker represents
    
    Args:
        ticker: The ticker symbol
    
    Returns:
        'stock', 'mutual_fund', 'pms', 'aif', or 'unknown'
    """
    ticker_str = str(ticker).strip().upper()
    
    # PMS detection
    if (ticker_str.startswith('INP') or 
        ticker_str.endswith('_PMS') or 
        ticker_str.endswith('PMS') or
        any(keyword in ticker_str for keyword in ['BUOYANT', 'CARNELIAN', 'JULIUS', 'VALENTIS', 'UNIFI'])):
        return 'pms'
    
    # AIF detection
    if ticker_str.startswith('AIF') or 'AIF' in ticker_str:
        return 'aif'
    
    # For numeric tickers, try to determine if it's MF or BSE stock
    if ticker_str.isdigit():
        # Most 6-digit numbers starting with 1 are mutual funds
        if len(ticker_str) == 6 and ticker_str.startswith('1'):
            return 'mutual_fund'
        # Other numeric codes are likely BSE stocks
        return 'stock'
    
    # Text tickers are likely stocks
    return 'stock'


def smart_price_fetch(ticker: str, date: str = None) -> Dict[str, Any]:
    """
    Smart detection and price fetching for any ticker
    Tries NSE → BSE → Mutual Fund → Returns best match
    
    Args:
        ticker: Ticker symbol
        date: Date in YYYY-MM-DD format (None for current price)
    
    Returns:
        dict with price, source, type, and exchange
    """
    ticker_str = str(ticker).strip()
    
    # Check type first
    ticker_type = detect_ticker_type(ticker_str)
    
    if ticker_type == 'pms':
        return {'ticker': ticker_str, 'price': None, 'source': 'manual', 'type': 'pms', 'message': 'PMS requires manual entry'}
    
    if ticker_type == 'aif':
        return {'ticker': ticker_str, 'price': None, 'source': 'manual', 'type': 'aif', 'message': 'AIF requires manual entry'}
    
    # For stocks and numeric tickers, try multiple sources
    if ticker_type in ['stock', 'numeric_unknown']:
        result = try_stock_sources(ticker_str, date)
        if result['price']:
            return result
        
        # If numeric, also try mutual fund
        if ticker_str.isdigit():
            result = try_mutual_fund(ticker_str, date)
            if result['price']:
                return result
    
    return {'ticker': ticker_str, 'price': None, 'source': 'unknown', 'type': 'unknown'}


def try_stock_sources(ticker: str, date: str = None) -> Dict[str, Any]:
    """Try to fetch stock price from NSE and BSE"""
    
    # Try NSE first
    try:
        nse_ticker = f"{ticker}.NS"
        stock = yf.Ticker(nse_ticker)
        
        if date:
            hist = stock.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=1))
        else:
            hist = stock.history(period='1d')
        
        if not hist.empty:
            price = float(hist['Close'].iloc[0])
            return {
                'ticker': ticker,
                'price': price,
                'source': 'yfinance_nse',
                'type': 'stock',
                'exchange': 'NSE'
            }
    except Exception:
        pass
    
    # Try BSE
    try:
        bse_ticker = f"{ticker}.BO"
        stock = yf.Ticker(bse_ticker)
        
        if date:
            hist = stock.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=1))
        else:
            hist = stock.history(period='1d')
        
        if not hist.empty:
            price = float(hist['Close'].iloc[0])
            return {
                'ticker': ticker,
                'price': price,
                'source': 'yfinance_bse',
                'type': 'stock',
                'exchange': 'BSE'
            }
    except Exception:
        pass
    
    return {'ticker': ticker, 'price': None, 'source': 'not_found', 'type': 'stock'}


def try_mutual_fund(ticker: str, date: str = None) -> Dict[str, Any]:
    """Try to fetch mutual fund NAV"""
    try:
        from mftool import Mftool
        mf = Mftool()
        
        # For current NAV
        if not date:
            quote = mf.get_scheme_quote(ticker)
            if quote and 'nav' in quote:
                return {
                    'ticker': ticker,
                    'price': float(quote['nav']),
                    'source': 'mftool',
                    'type': 'mutual_fund',
                    'scheme_name': quote.get('scheme_name', 'Unknown Fund')
                }
        else:
            # For historical NAV
            hist_data = mf.get_scheme_historical_nav(ticker, as_Dataframe=True)
            if hist_data is not None and not hist_data.empty:
                hist_data['date'] = pd.to_datetime(hist_data.index, format='%d-%m-%Y', dayfirst=True)
                target_date = pd.to_datetime(date)
                
                # Find closest date
                hist_data['date_diff'] = abs(hist_data['date'] - target_date)
                closest = hist_data.loc[hist_data['date_diff'].idxmin()]
                
                if closest['date_diff'].days <= 7:  # Within a week
                    return {
                        'ticker': ticker,
                        'price': float(closest['nav']),
                        'source': 'mftool',
                        'type': 'mutual_fund',
                        'date_match': 'approximate'
                    }
    except Exception:
        pass
    
    return {'ticker': ticker, 'price': None, 'source': 'not_found', 'type': 'mutual_fund'}


def normalize_ticker(ticker: str, asset_type: str = None) -> str:
    """
    Normalize ticker based on detected type
    FIXED: Prevents double suffix (e.g., RELIANCE.NS.NS)
    
    Args:
        ticker: Original ticker
        asset_type: Optional pre-determined asset type
    
    Returns:
        Normalized ticker
    """
    ticker_str = str(ticker).strip().upper()
    
    # If type not provided, detect it
    if not asset_type:
        asset_type = detect_ticker_type(ticker_str)
    
    # For stocks, add exchange suffix if not already present
    if asset_type == 'stock':
        # Check if suffix already exists
        if ticker_str.endswith('.NS') or ticker_str.endswith('.BO'):
            return ticker_str  # Already normalized, return as-is
        
        # For numeric tickers (BSE codes)
        if ticker_str.isdigit():
            return ticker_str  # Keep as-is, will try both .NS and .BO in fetcher
        
        # For text tickers, add .NS suffix
        return f"{ticker_str}.NS"
    
    # For MF, PMS, AIF, keep as-is
    return ticker_str

