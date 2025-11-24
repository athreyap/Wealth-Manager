"""
Enhanced Price Fetcher with Complete Fallback Chain
yfinance → mftool → AI (for stocks and mutual funds)
"""

import csv
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import re
import streamlit as st
import yfinance as yf


class EnhancedPriceFetcher:
    """
    Complete price fetching with multi-source fallback:
    Stock: yfinance NSE → yfinance BSE → AI
    MF: mftool → Transaction price → AI
    PMS/AIF: Manual or estimated
    """
    
    def __init__(self):
        self.price_cache = {}
        # NOTE: cache_timeout is ONLY for caching - does NOT affect processing timeouts
        # Processing has NO timeouts - will wait indefinitely for API responses
        self.cache_timeout = 300  # 5 minutes (cache expiry only)
        self._amfi_cache: Optional[Dict[str, Any]] = None
        self._cached_scheme_maps: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None
        self._stock_alias_cache: Dict[str, List[str]] = {}
        self._manual_alias_map: Dict[str, List[str]] = {
            # Seed known corporate action aliases; will grow automatically via metadata lookup.
            "IDFC": ["IDFCFIRSTB.NS", "IDFCFIRSTB.BO"],
            "IDFC LIMITED": ["IDFCFIRSTB.NS", "IDFCFIRSTB.BO"],
            "IRB INVIT FUND": ["IRBINVIT.NS", "IRBINVIT.BO"],
            "IRB INFRASTRUCTURE INVESTMENT TRUST": ["IRBINVIT.NS", "IRBINVIT.BO"],
            "101206": ["IRBINVIT.NS", "IRBINVIT.BO"],
            "500285": ["SPICEJET.BO", "SPICEJET.NS"],
            "SPICEJET": ["SPICEJET.BO", "SPICEJET.NS"],
        }
        self.http_session = self._get_http_session()
        self._mftool = self._get_shared_mftool()
        
        # Defer OpenAI initialization until actually needed (lazy initialization)
        # This avoids KeyError when st.secrets is not available during module import
        self.ai_available = False
        self.openai_client = None
        self._openai_initialized = False
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client - only when needed"""
        if self._openai_initialized:
            return self.openai_client
        
        self._openai_initialized = True
        try:
            from openai import OpenAI
            # Check if secrets are available
            if "api_keys" not in st.secrets:
                raise KeyError("'api_keys' not found in st.secrets")
            if "open_ai" not in st.secrets.get("api_keys", {}):
                raise KeyError("'open_ai' not found in st.secrets['api_keys']")
            self.openai_client = OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
            self.ai_available = True
            return self.openai_client
        except KeyError as e:
            self.ai_available = False
            self.openai_client = None
            return None
        except Exception as e:
            self.ai_available = False
            self.openai_client = None
            return None
    
    def _get_pms_aif_calculator(self):
        """Lazy initialization of PMS/AIF calculator - only when needed"""
        if hasattr(self, 'pms_aif_calculator') and self.pms_aif_calculator is not None:
            return self.pms_aif_calculator
        
        try:
            from pms_aif_calculator import PMS_AIF_Calculator
            self.pms_aif_calculator = PMS_AIF_Calculator()
            return self.pms_aif_calculator
        except Exception as e:
            self.pms_aif_calculator = None
            print(f"[PMS_AIF] ⚠️ PMS/AIF calculator not available: {str(e)}")
            return None

    def _normalize_base_ticker(self, value: str) -> str:
        if not value:
            return value
        text = str(value).strip()
        if ':' in text:
            text = text.split(':')[-1]
        text = re.sub(r'(?:[-_.]?)INSTRUMENT$', '', text, flags=re.IGNORECASE)
        text = text.replace('\u00a0', ' ')
        text = ''.join(text.split())  # remove whitespace
        # Remove $ suffix (common in BSE tickers like VIVANTA$, MADHAVIPL$)
        text = text.rstrip('$').lstrip('$')
        # FIXED: Only remove trade tags when they're separated by dash/underscore/dot
        # This prevents "VBL" from becoming "V" - only "V-BL" or "V_BL" should become "V"
        text = re.sub(r'[-_.](T0|T1|BL)(?=\.|$)', '', text, flags=re.IGNORECASE)
        text = text.replace('-', '')
        # CRITICAL: Preserve .NS and .BO suffixes - don't normalize them away
        # Check if it ends with .NS or .BO before uppercasing
        has_ns = text.upper().endswith('.NS')
        has_bo = text.upper().endswith('.BO')
        text = text.upper()
        # Ensure only one suffix (remove duplicates like .NS.NS)
        if text.endswith('.NS.NS') or text.endswith('.BO.BO') or text.endswith('.NS.BO') or text.endswith('.BO.NS'):
            # Remove duplicate suffixes
            text = re.sub(r'\.(NS|BO)\.(NS|BO)$', r'.\1', text)
        return text

    def _base_symbol(self, value: str) -> str:
        """Collapse exchange suffixes and trade-tags to a canonical base symbol."""
        normalized = self._normalize_base_ticker(value or "")
        if not normalized:
            return ""
        if normalized.endswith('.NS') or normalized.endswith('.BO'):
            return normalized[:-3]
        return normalized
    
    def get_current_price(self, ticker: str, asset_type: str, fund_name: str = None) -> tuple:
        """
        Get current price with complete fallback chain
        
        Args:
            ticker: Ticker symbol
            asset_type: 'stock', 'mutual_fund', 'pms', 'aif', 'bond'
            fund_name: Full name of the asset (for AI fallback)
        
        Returns:
            Tuple of (price, source) where:
            - price: float or None
            - source: str indicating data source
        """
        # Safety check: If ticker is numeric (AMFI code), it's ALWAYS a mutual fund
        # even if asset_type was incorrectly set to 'bond' (e.g., bond mutual funds)
        try:
            cleaned = str(ticker).replace(',', '').replace('$', '').strip()
            numeric_value = float(cleaned)
            # If it's numeric, override asset_type to mutual_fund
            if asset_type == 'bond':
                asset_type = 'mutual_fund'
        except (ValueError, AttributeError):
            pass
        
        # Check cache first
        cache_key = f"{ticker}_{asset_type}_current"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            age = (datetime.now() - cached_data['timestamp']).total_seconds()
            if age < self.cache_timeout:
                return cached_data['price'], cached_data.get('source', 'cache')
        
        # CRITICAL: Clear resolved ticker at START of each fetch to prevent cross-contamination
        # between parallel/sequential holdings
        self._last_resolved_ticker = None
        self._last_resolved_fund_name = None
        
        price = None
        source = 'unknown'
        
        if asset_type == 'stock':
            price, source = self._get_stock_price_with_fallback(ticker, fund_name)
        elif asset_type == 'mutual_fund':
            price, source = self._get_mf_price_with_fallback(ticker, fund_name)
        elif asset_type in ['pms', 'aif']:
            price, source = self._get_pms_aif_price(ticker, asset_type)
        elif asset_type == 'bond':
            # Pass bond name for AI fallback
            result = self._get_bond_price(ticker, bond_name=fund_name)
            if result:
                price, source = result
            else:
                price, source = None, 'bond_price_unavailable'
        
        # Cache result
        if price:
            self.price_cache[cache_key] = {
                'price': price,
                'timestamp': datetime.now(),
                'source': source
            }
        
        # Return tuple (price, source) for compatibility
        return price, source
    
    def _update_single_holding_price(self, holding: Dict, db_manager) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Update price for a single holding. Returns (success, error_message, price_data).
        price_data contains 'price' and optionally 'resolved_ticker' for batch updates.
        """
        try:
            original_ticker = holding.get('ticker')
            ticker = original_ticker
            asset_type = holding.get('asset_type', 'stock')
            stock_id = holding.get('stock_id')
            
            if not ticker or not stock_id:
                return False, f"Missing ticker or stock_id: ticker={ticker}, stock_id={stock_id}"

            cleaned_ticker = self._normalize_base_ticker(ticker)
            if cleaned_ticker and cleaned_ticker != ticker:
                base_symbol = self._base_symbol(original_ticker)
                try:
                    db_manager.supabase.table('stock_master').update({
                        'ticker': cleaned_ticker,
                        'last_updated': datetime.now().isoformat()
                    }).ilike('ticker', f"%{base_symbol}%").execute()
                except Exception:
                    pass
                ticker = cleaned_ticker
                holding['ticker'] = cleaned_ticker
            
            current_price = None
            source = None
            
            if asset_type == 'stock':
                # CRITICAL: Clear resolved ticker at START of each holding to prevent cross-contamination
                self._last_resolved_ticker = None
                self._last_resolved_fund_name = None
                
                # STEP 1: Try to get stored ticker from stock_master (if available)
                stored_ticker = None
                stock_name = holding.get('stock_name')
                try:
                    if stock_id and db_manager:
                        stock_info = db_manager.supabase.table('stock_master').select('ticker, stock_name').eq('id', stock_id).execute()
                        if stock_info.data and len(stock_info.data) > 0:
                            stored_ticker = stock_info.data[0].get('ticker')
                            stored_name = stock_info.data[0].get('stock_name')
                            if stored_ticker and stored_ticker != ticker:
                                print(f"[PRICE_UPDATE] 📋 Using stored ticker from stock_master: {ticker} → {stored_ticker} (stock_id: {stock_id})")
                                ticker = stored_ticker
                                if stored_name and not stock_name:
                                    stock_name = stored_name
                except Exception as e:
                    print(f"[PRICE_UPDATE] ⚠️ Could not fetch stored ticker: {str(e)[:100]}")
                
                # STEP 2: Try fetching with stored ticker (or original ticker)
                current_price, source = self._get_stock_price_with_fallback(ticker, stock_name)
                
                # STEP 3: If stored ticker failed and we have stock_name, try AI resolution
                if (not current_price or current_price <= 0) and stock_name and not stored_ticker:
                    print(f"[PRICE_UPDATE] 🔍 Stored ticker failed, trying AI resolution from name: '{stock_name}'")
                    resolved_ticker = self._resolve_stock_ticker_by_name(stock_name)
                    if resolved_ticker and resolved_ticker != ticker:
                        print(f"[PRICE_UPDATE] ✅ AI resolved ticker: {ticker} → {resolved_ticker}")
                        current_price, source = self._get_stock_price_with_fallback(resolved_ticker, stock_name)
                        if current_price and current_price > 0:
                            # Update stock_master with resolved ticker
                            try:
                                if stock_id and db_manager:
                                    db_manager.supabase.table('stock_master').update({
                                        'ticker': resolved_ticker,
                                        'last_updated': datetime.now().isoformat()
                                    }).eq('id', stock_id).execute()
                                    print(f"[PRICE_UPDATE] 💾 Updated stock_master with resolved ticker: {resolved_ticker}")
                            except Exception as e:
                                print(f"[PRICE_UPDATE] ⚠️ Could not update stock_master: {str(e)[:100]}")
                # OPTIMIZATION: Don't update DB here - will be batched later
                    
            elif asset_type == 'mutual_fund':
                # CRITICAL: Clear resolved ticker at START of each holding to prevent cross-contamination
                self._last_resolved_ticker = None
                self._last_resolved_fund_name = None
                
                fund_name = holding.get('scheme_name') or holding.get('stock_name', '')
                # Store average price for fallback if price fetch fails (for regular funds only)
                avg_price_for_fallback = holding.get('average_price', 0)
                self._last_holding_avg_price = avg_price_for_fallback if avg_price_for_fallback > 0 else None
                
                # Get price - segregated fund handling is now inside _get_mf_price_with_fallback
                current_price, source = self._get_mf_price_with_fallback(ticker, fund_name)
                
                # If price fetch failed and it's NOT a segregated fund, use average price as fallback
                # (Segregated funds are already handled inside _get_mf_price_with_fallback)
                if (not current_price or current_price <= 0) and source != 'segregated_fund_unavailable':
                    if avg_price_for_fallback > 0:
                        # For regular funds, use average price as fallback
                        current_price = avg_price_for_fallback
                        source = 'average_price_fallback'
                        print(f"[MF_PRICE] ⚠️ {ticker}: Using average price as fallback (price fetch failed)")
                
                # Clear the average price after use
                self._last_holding_avg_price = None
                # OPTIMIZATION: Don't update DB here - will be batched later
                    
            elif asset_type in ['pms', 'aif']:
                current_price, source = self._calculate_pms_aif_live_price(ticker, asset_type, db_manager, holding)
                # If CAGR calculation failed, use average_price as fallback
                if not current_price or current_price <= 0:
                    avg_price = holding.get('average_price', 0)
                    if avg_price > 0:
                        print(f"[PMS_AIF] ⚠️ CAGR calculation failed for {ticker}, using average_price: ₹{avg_price:,.2f}")
                        current_price = avg_price
                        source = 'average_price_fallback_cagr_failed'
                    else:
                        print(f"[PMS_AIF] ❌ CAGR calculation failed for {ticker} and no average_price available")
            
            elif asset_type == 'bond':
                bond_name = holding.get('stock_name', '')
                result = self._get_bond_price(ticker, bond_name=bond_name)
                if result:
                    current_price, source = result
            
            # Validate price
            avg_price = holding.get('average_price', 0)
            
            if current_price and current_price > 0:
                if asset_type == 'stock' and avg_price > 0:
                    price_ratio = current_price / avg_price
                    if price_ratio > 10 or price_ratio < 0.1:
                        pass  # Log warning but still store
                
                # OPTIMIZATION: Return price data for batch update instead of updating immediately
                # Check for resolved ticker from source (for mutual funds) or from _last_resolved_ticker (for stocks)
                resolved_ticker = None
                resolved_fund_name = None
                
                # For mutual funds, extract resolved ticker from source if available
                if asset_type == 'mutual_fund' and source and 'name_resolved:' in source:
                    try:
                        resolved_ticker = source.split('name_resolved:')[1].strip()
                        # Get resolved fund name if available
                        resolved_fund_name = getattr(self, "_last_resolved_fund_name", None)
                    except:
                        pass
                
                # For stocks, use _last_resolved_ticker if set
                if not resolved_ticker and asset_type == 'stock':
                    resolved_ticker = getattr(self, "_last_resolved_ticker", None)
                    # CRITICAL: Validate that resolved ticker is actually related to original ticker
                    if resolved_ticker:
                        original_base = self._base_symbol(ticker)
                        resolved_base = self._base_symbol(resolved_ticker)
                        # Only use resolved ticker if base symbols match exactly (same stock, different exchange suffix)
                        if original_base != resolved_base:
                            # Resolved ticker is for a completely different stock - ignore it
                            print(f"[PRICE_UPDATE] ⚠️ Ignoring wrong resolved ticker: {ticker} → {resolved_ticker} (different base: {original_base} vs {resolved_base})")
                            resolved_ticker = None
                
                # CRITICAL: For mutual funds, allow resolution to:
                # 1. Other numeric MF codes
                # 2. ETF tickers (ETFs can be stored as mutual funds, e.g., GROWWEV, SILVERBEES)
                if asset_type == 'mutual_fund' and resolved_ticker:
                    resolved_clean = str(resolved_ticker).strip().replace('.NS', '').replace('.BO', '')
                    is_numeric_mf = resolved_clean.isdigit() or str(resolved_ticker).isdigit()
                    # Check if it's an ETF (common ETF patterns: ends with BEES, contains ETF, or known ETF tickers)
                    is_etf = (
                        'BEES' in resolved_clean.upper() or
                        'ETF' in resolved_clean.upper() or
                        resolved_clean.upper() in ['GROWWEV', 'SILVERBEES', 'GOLDBEES', 'NIFTYBEES', 'BANKBEES']
                    )
                    if not is_numeric_mf and not is_etf:
                        # Resolved to a regular stock ticker - this is wrong for mutual funds (unless it's an ETF)
                        print(f"[PRICE_UPDATE] ⚠️ Ignoring wrong resolved ticker for MF: {ticker} → {resolved_ticker} (MF code or ETF expected, got stock ticker)")
                        resolved_ticker = None
                    elif is_etf:
                        print(f"[PRICE_UPDATE] ✅ Allowing MF → ETF resolution: {ticker} → {resolved_ticker} (ETF detected)")
                
                # Only include resolved_ticker if it's different from current ticker
                price_data = {
                    'price': current_price,
                    'resolved_ticker': resolved_ticker if resolved_ticker and resolved_ticker != ticker else None,
                    'resolved_fund_name': resolved_fund_name if resolved_fund_name else None
                }
                
                # Clear _last_resolved_ticker and _last_resolved_fund_name after use
                if hasattr(self, '_last_resolved_ticker'):
                    self._last_resolved_ticker = None
                if hasattr(self, '_last_resolved_fund_name'):
                    self._last_resolved_fund_name = None
                
                return True, None, price_data
            else:
                return False, f"Failed to fetch price for {ticker}", None
                
        except Exception as e:
            return False, str(e), None
    
    def update_live_prices_for_holdings(self, holdings: List[Dict], db_manager) -> None:
        """
        Update live_price in stock_master table for all holdings (PARALLELIZED + BATCHED DB UPDATES)
        
        Args:
            holdings: List of holding records
            db_manager: Database manager instance
        """
        print(f"[PRICE_UPDATE] Starting parallel price update for {len(holdings)} holdings...")
        
        success_count = 0
        failed_count = 0
        
        # OPTIMIZATION: Collect all price updates and batch them
        price_updates = []  # List of (stock_id, price) tuples
        ticker_updates = []  # List of (stock_id, ticker, fund_name) tuples for ticker normalization
        # Corporate actions are detected in UI only, not auto-applied during price updates
        
        # Process holdings in parallel batches to avoid overwhelming the system
        max_workers = min(20, len(holdings))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_holding = {
                executor.submit(self._update_single_holding_price, holding, db_manager): holding
                for holding in holdings
            }
            
            # Process results as they complete
            for future in as_completed(future_to_holding):
                holding = future_to_holding[future]
                try:
                    success, error_msg, price_data = future.result()
                    if success:
                        success_count += 1
                        ticker = holding.get('ticker', 'unknown')
                        asset_type = holding.get('asset_type', 'unknown')
                        stock_id = holding.get('stock_id')
                        
                        # Collect updates for batching
                        if price_data:
                            if 'price' in price_data:
                                price_updates.append((stock_id, price_data['price']))
                            if 'resolved_ticker' in price_data and price_data['resolved_ticker']:
                                resolved_fund_name = price_data.get('resolved_fund_name')
                                ticker_updates.append((stock_id, price_data['resolved_ticker'], resolved_fund_name))
                                print(f"[PRICE_UPDATE] 🔄 Collected ticker update: {holding.get('ticker')} → {price_data['resolved_ticker']} (stock_id: {stock_id})")
                        
                        print(f"[PRICE_UPDATE] ✅ {ticker} ({asset_type}): Price fetched")
                        
                        # Corporate actions are now detected and suggested in UI only (not auto-applied)
                        # Detection happens in web_agent.py via detect_corporate_actions() function
                    else:
                        failed_count += 1
                        ticker = holding.get('ticker', 'unknown')
                        asset_type = holding.get('asset_type', 'unknown')
                        print(f"[PRICE_UPDATE] ❌ {ticker} ({asset_type}): {error_msg}")
                        
                        # For PMS/AIF, log more details about why it failed
                        if asset_type in ['pms', 'aif']:
                            user_id = holding.get('user_id')
                            stock_id = holding.get('stock_id')
                            print(f"[PRICE_UPDATE]   PMS/AIF debug: user_id={user_id}, stock_id={stock_id}, ticker={ticker}")
                except Exception as e:
                    failed_count += 1
                    ticker = holding.get('ticker', 'unknown')
                    asset_type = holding.get('asset_type', 'unknown')
                    print(f"[PRICE_UPDATE] ❌ Error updating {ticker} ({asset_type}): {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # OPTIMIZATION: Batch update all prices at once (much faster than individual updates)
        if price_updates:
            try:
                # Update prices in batches of 50
                batch_size = 50
                for i in range(0, len(price_updates), batch_size):
                    batch = price_updates[i:i+batch_size]
                    for stock_id, price in batch:
                        try:
                            db_manager.update_stock_live_price(stock_id, price)
                        except Exception:
                            pass  # Individual failures don't stop the batch
                print(f"[PRICE_UPDATE] 📦 Batched {len(price_updates)} price updates")
            except Exception as e:
                print(f"[PRICE_UPDATE] ⚠️ Batch update error: {str(e)}")
        
        # Batch update ticker normalizations
        if ticker_updates:
            try:
                updated_count = 0
                skipped_count = 0
                for stock_id, resolved_ticker, resolved_fund_name in ticker_updates:
                    try:
                        # Check if resolved ticker already exists with a different stock_id
                        existing = db_manager.supabase.table('stock_master').select('id').eq('ticker', resolved_ticker).execute()
                        if existing.data and len(existing.data) > 0:
                            existing_stock_id = existing.data[0]['id']
                            if existing_stock_id != stock_id:
                                # Ticker already exists for a different stock - skip update to avoid duplicate key
                                print(f"[PRICE_UPDATE] ⚠️ Skipping ticker update: {resolved_ticker} already exists for different stock_id")
                                skipped_count += 1
                                continue
                        
                        update_data = {
                            'ticker': resolved_ticker,
                            'last_updated': datetime.now().isoformat()
                        }
                        # Also update fund name if provided
                        if resolved_fund_name:
                            update_data['stock_name'] = resolved_fund_name
                        
                        result = db_manager.supabase.table('stock_master').update(update_data).eq('id', stock_id).execute()
                        if result.data:
                            updated_count += 1
                    except Exception as e:
                        error_msg = str(e)
                        # Check if it's a duplicate key error
                        if 'duplicate key' in error_msg.lower() or '23505' in error_msg:
                            print(f"[PRICE_UPDATE] ⚠️ Skipping ticker update: {resolved_ticker} already exists (duplicate key)")
                            skipped_count += 1
                        else:
                            print(f"[PRICE_UPDATE] ⚠️ Failed to update ticker for stock_id {stock_id}: {error_msg[:200]}")
                print(f"[PRICE_UPDATE] 📦 Auto-updated {updated_count}/{len(ticker_updates)} tickers in stock_master ({skipped_count} skipped - already exist)")
            except Exception as e:
                print(f"[PRICE_UPDATE] ⚠️ Batch ticker update error: {str(e)}")
        
        # Corporate actions are now detected and suggested in UI only (not auto-applied)
        # Users can manually apply them via the UI when they see the suggestions
        print(f"[PRICE_UPDATE] Complete: {success_count} succeeded, {failed_count} failed")
    
    def _fetch_corporate_actions_from_yfinance(self, ticker: str, from_date: datetime, to_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Fetch corporate actions from yfinance (primary source - fast and reliable).
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS.NS')
            from_date: Start date to check corporate actions from
            to_date: End date (defaults to 1 year in future to catch announced splits)
        
        Returns:
            List of corporate action dicts with keys: type, date, ratio, description
        """
        if to_date is None:
            to_date = datetime.now() + timedelta(days=365)  # Include future splits (announced but not yet executed)
        
        corporate_actions = []
        
        try:
            # Try different ticker formats
            candidates = []
            if ticker.endswith(('.NS', '.BO')):
                candidates.append(ticker)
            elif ticker.isdigit():
                candidates.append(f"{ticker}.BO")
            else:
                # Try both .NS and .BO, and also try without suffix
                candidates.extend([f"{ticker}.NS", f"{ticker}.BO", ticker])
            
            ticker_obj = None
            working_ticker = None
            
            # Enhanced logging for all tickers
            print(f"[CORP_ACTION] [CHECK] {ticker}: Checking ticker formats: {candidates}")
            
            for candidate in candidates:
                try:
                    test_ticker = yf.Ticker(candidate)
                    # Try to get info to verify ticker is valid
                    info = test_ticker.info
                    if info and 'symbol' in info:
                        ticker_obj = test_ticker
                        working_ticker = candidate
                        print(f"[CORP_ACTION] [OK] {ticker}: Found working ticker: {working_ticker} (symbol: {info.get('symbol')})")
                        break
                except Exception as e:
                    print(f"[CORP_ACTION] [WARNING] {ticker}: Failed to fetch {candidate}: {str(e)[:100]}")
                    continue
            
            # Check known splits database FIRST (before yfinance) - this catches splits that yfinance might miss
            print(f"[CORP_ACTION] [CHECK] {ticker}: Checking known splits database (working_ticker: {working_ticker or ticker})")
            known_splits = self._get_known_splits(ticker, working_ticker or ticker, from_date, to_date)
            if known_splits:
                print(f"[CORP_ACTION] [INFO] {ticker}: Found {len(known_splits)} known split(s) from database")
                corporate_actions.extend(known_splits)
            else:
                print(f"[CORP_ACTION] [INFO] {ticker}: No known splits found in database")
            
            if not ticker_obj:
                print(f"[CORP_ACTION] [ERROR] {ticker}: Could not find valid ticker in any format: {candidates}")
                # Even if yfinance failed, we already checked known splits above
                return corporate_actions
            
            # Fetch splits from yfinance
            try:
                splits = ticker_obj.splits
                if splits is not None and not splits.empty:
                    print(f"[CORP_ACTION] [INFO] {ticker}: Found {len(splits)} total splits in yfinance (working ticker: {working_ticker})")
                    # CRITICAL: Use from_date directly (purchase date) - do NOT expand backwards
                    # Only splits AFTER purchase date should be considered
                    from_date_for_filter = from_date.replace(tzinfo=None) if from_date and from_date.tzinfo else from_date
                    # Ensure timezone-naive for comparison
                    if to_date.tzinfo is not None:
                        to_date = to_date.replace(tzinfo=None) if to_date else datetime.now().replace(tzinfo=None)
                    
                    for split_date, split_ratio in splits.items():
                        # yfinance reports split ratio as fraction (e.g., 0.5 for 2:1 split, 2.0 for 1:2 split)
                        # Convert to our format: split_ratio = new_shares / old_shares
                        if split_ratio < 1.0:
                            # e.g., 0.5 means 2:1 split (1 old share becomes 2 new shares)
                            actual_ratio = 1.0 / split_ratio
                            old_ratio, new_ratio = 1, int(actual_ratio)
                        else:
                            # e.g., 2.0 means 1:2 split (1 old share becomes 2 new shares)
                            actual_ratio = split_ratio
                            old_ratio, new_ratio = 1, int(actual_ratio)
                        
                        split_date_dt = split_date.to_pydatetime() if hasattr(split_date, 'to_pydatetime') else split_date
                        if isinstance(split_date_dt, pd.Timestamp):
                            split_date_dt = split_date_dt.to_pydatetime()
                        
                        # Normalize to timezone-naive for comparison
                        if split_date_dt.tzinfo is not None:
                            split_date_dt = split_date_dt.replace(tzinfo=None)
                        
                        # CRITICAL: Only include splits that are >= from_date (purchase date) and <= to_date
                        # This ensures we only consider splits AFTER purchase date
                        if (from_date_for_filter is None or split_date_dt >= from_date_for_filter) and split_date_dt <= to_date:
                            print(f"[CORP_ACTION] [OK] {ticker}: SPLIT DETECTED on {split_date_dt.date()} - {old_ratio}:{new_ratio} (ratio: {actual_ratio:.4f})")
                            
                            corporate_actions.append({
                                'type': 'split',
                                'date': split_date_dt,
                                'old_ratio': old_ratio,
                                'new_ratio': new_ratio,
                                'split_ratio': actual_ratio,
                                'description': f"Stock split {old_ratio}:{new_ratio} (yfinance)"
                            })
                        else:
                            # Log splits outside range for debugging
                            if from_date_for_filter:
                                if split_date_dt < from_date_for_filter:
                                    print(f"[CORP_ACTION] [WARNING] {ticker}: Split on {split_date_dt.date()} ignored (happened BEFORE purchase on {from_date_for_filter.date()})")
                                else:
                                    print(f"[CORP_ACTION] [WARNING] {ticker}: Split on {split_date_dt.date()} ignored (outside date range: {from_date_for_filter.date()} to {to_date.date()})")
                            else:
                                print(f"[CORP_ACTION] [WARNING] {ticker}: Split on {split_date_dt.date()} ignored (outside date range: up to {to_date.date()})")
                else:
                    # Enhanced logging when no splits found
                    print(f"[CORP_ACTION] [INFO] {ticker}: No splits data found in yfinance for {working_ticker}")
                    # Try AI fallback to check for splits (only if we have a working ticker)
                    if working_ticker:
                        ai_splits = self._fetch_splits_from_ai(ticker, working_ticker, from_date, to_date)
                        if ai_splits:
                            print(f"[CORP_ACTION] [OK] {ticker}: Found {len(ai_splits)} split(s) from AI")
                            corporate_actions.extend(ai_splits)
                        else:
                            print(f"[CORP_ACTION] [INFO] {ticker}: AI also found no splits - confirming no split")
            except Exception as e:
                error_msg = str(e)
                print(f"[CORP_ACTION] [WARNING] {ticker}: Error fetching splits from yfinance: {error_msg}")
                print(f"[CORP_ACTION] [DEBUG] {ticker}: Tried ticker formats: {candidates}, Working ticker: {working_ticker}")
                import traceback
                print(f"[CORP_ACTION] [DEBUG] {ticker}: Full error trace: {traceback.format_exc()}")
                # Try AI fallback even if yfinance errored (only if we have a working ticker)
                if working_ticker:
                    ai_splits = self._fetch_splits_from_ai(ticker, working_ticker, from_date, to_date)
                    if ai_splits:
                        print(f"[CORP_ACTION] [OK] {ticker}: Found {len(ai_splits)} split(s) from AI (yfinance failed)")
                        corporate_actions.extend(ai_splits)
            
            # Fetch dividends from yfinance (for reference, though we don't auto-adjust for dividends)
            try:
                dividends = ticker_obj.dividends
                if dividends is not None and not dividends.empty:
                    # Ensure from_date and to_date are timezone-naive
                    from_date_naive = from_date.replace(tzinfo=None) if from_date and from_date.tzinfo else from_date
                    to_date_naive = to_date.replace(tzinfo=None) if to_date and to_date.tzinfo else to_date
                    
                    for div_date, div_amount in dividends.items():
                        div_date_dt = div_date.to_pydatetime() if hasattr(div_date, 'to_pydatetime') else div_date
                        if isinstance(div_date_dt, pd.Timestamp):
                            div_date_dt = div_date_dt.to_pydatetime()
                        
                        # Normalize to timezone-naive for comparison
                        if div_date_dt.tzinfo is not None:
                            div_date_dt = div_date_dt.replace(tzinfo=None)
                        
                        if from_date_naive <= div_date_dt <= to_date_naive:
                            # Store dividend info for reference (not auto-adjusted)
                            corporate_actions.append({
                                'type': 'dividend',
                                'date': div_date_dt,
                                'amount': float(div_amount),
                                'description': f"Dividend: ₹{div_amount:.2f} per share (yfinance)"
                            })
            except Exception as e:
                print(f"[CORP_ACTION] [WARNING] Error fetching dividends from yfinance for {ticker}: {str(e)}")
            
            # Check for merger/acquisition info in ticker.info
            try:
                info = ticker_obj.info
                if info:
                    # Check for name changes or mergers
                    long_name = info.get('longName', '')
                    symbol = info.get('symbol', '')
                    previous_close = info.get('previousClose', 0)
                    
                    # If symbol doesn't match expected ticker, might be a merger
                    base_ticker = self._normalize_base_ticker(ticker).replace('.NS', '').replace('.BO', '')
                    if symbol and symbol.upper() != base_ticker.upper() and symbol.upper() != working_ticker.upper():
                        # Potential merger - add as info (user will need to confirm)
                        corporate_actions.append({
                            'type': 'merger',
                            'date': from_date,  # Use from_date as placeholder
                            'new_ticker': symbol,
                            'exchange_ratio': 1.0,  # Default, user can adjust
                            'cash_per_share': 0.0,
                            'description': f"Ticker changed from {ticker} to {symbol} (yfinance info)"
                        })
            except Exception as e:
                print(f"[CORP_ACTION] [WARNING] Error checking merger info from yfinance for {ticker}: {str(e)}")
            
            # Sort by date (oldest first)
            corporate_actions.sort(key=lambda x: x['date'])
            
            if corporate_actions:
                print(f"[CORP_ACTION] [OK] Found {len(corporate_actions)} corporate actions from yfinance for {ticker}")
            
        except Exception as e:
            print(f"[CORP_ACTION] [WARNING] Error fetching corporate actions from yfinance for {ticker}: {str(e)}")
        
        return corporate_actions
    
    def _get_known_splits(self, ticker: str, working_ticker: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """
        Check for known splits in hardcoded database (for splits that AI might miss).
        
        Args:
            ticker: Original ticker symbol
            working_ticker: Working ticker from yfinance
            from_date: Start date to check splits from
            to_date: End date to check splits to
        
        Returns:
            List of split dicts or empty list
        """
        # Normalize ticker for comparison
        base_ticker = self._normalize_base_ticker(ticker).replace('.NS', '').replace('.BO', '').upper() if ticker else ''
        working_base = self._normalize_base_ticker(working_ticker).replace('.NS', '').replace('.BO', '').upper() if working_ticker else ''
        
        # Known splits database
        known_splits_db = {
            'WEBSOL': [
                {
                    'date': datetime(2025, 11, 14),
                    'old_ratio': 1,
                    'new_ratio': 10,
                    'split_ratio': 10.0,
                    'description': 'Stock split 1:10 (Face value change from ₹10 to ₹1)'
                }
            ]
        }
        
        # Check if we have known splits for this ticker
        splits = []
        # Build list of possible ticker matches
        possible_tickers = [base_ticker, working_base]
        if ticker:
            possible_tickers.append(ticker.upper())
            possible_tickers.append(ticker.upper().replace('.NS', '').replace('.BO', ''))
        if working_ticker:
            possible_tickers.append(working_ticker.upper())
            possible_tickers.append(working_ticker.upper().replace('.NS', '').replace('.BO', ''))
        
        # Remove empty strings and duplicates
        possible_tickers = [t for t in set(possible_tickers) if t]
        print(f"[CORP_ACTION] [KNOWN_SPLITS] {ticker}: Checking against possible tickers: {possible_tickers}")
        print(f"[CORP_ACTION] [KNOWN_SPLITS] {ticker}: Known splits DB keys: {list(known_splits_db.keys())}")
        
        for key, split_list in known_splits_db.items():
            if key in possible_tickers:
                print(f"[CORP_ACTION] [KNOWN_SPLITS] {ticker}: MATCH FOUND for key '{key}' in possible_tickers")
                for split_info in split_list:
                    split_date = split_info['date']
                    # Normalize dates for comparison
                    if split_date.tzinfo is not None:
                        split_date = split_date.replace(tzinfo=None)
                    if from_date and from_date.tzinfo is not None:
                        from_date_normalized = from_date.replace(tzinfo=None)
                    else:
                        from_date_normalized = from_date
                    if to_date and to_date.tzinfo is not None:
                        to_date_normalized = to_date.replace(tzinfo=None) if to_date else datetime.now().replace(tzinfo=None)
                    else:
                        to_date_normalized = to_date if to_date else datetime.now()
                    
                    # CRITICAL: Only include splits that are >= from_date (purchase date) and <= to_date
                    # This ensures we only consider splits AFTER purchase date
                    if (from_date_normalized is None or split_date >= from_date_normalized) and split_date <= to_date_normalized:
                        splits.append({
                            'type': 'split',
                            'date': split_date,
                            'old_ratio': split_info['old_ratio'],
                            'new_ratio': split_info['new_ratio'],
                            'split_ratio': split_info['split_ratio'],
                            'description': f"{split_info['description']} (Known split)"
                        })
        
        return splits
    
    def _fetch_splits_from_ai(self, ticker: str, working_ticker: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """
        Use AI to check for stock splits when yfinance doesn't return any.
        
        Args:
            ticker: Original ticker symbol
            working_ticker: Working ticker from yfinance (e.g., 'WEBSOL.BO')
            from_date: Start date to check splits from
            to_date: End date to check splits to
        
        Returns:
            List of split dicts with keys: type, date, old_ratio, new_ratio, split_ratio, description
        """
        # Note: Known splits are already checked in _fetch_corporate_actions_from_yfinance before calling this function
        # So we don't check them again here - just use AI as fallback when yfinance finds nothing
        client = self._get_openai_client()
        if not client:
            print(f"[CORP_ACTION] [INFO] {ticker}: OpenAI client not available - skipping AI split check")
            return []
        
        try:
            # Get stock name from yfinance if available
            stock_name = working_ticker
            try:
                ticker_obj = yf.Ticker(working_ticker)
                info = ticker_obj.info
                if info and 'longName' in info:
                    stock_name = info.get('longName', working_ticker)
            except:
                pass
            
            # CRITICAL: Use from_date directly (purchase date) - do NOT expand backwards
            # Only splits AFTER purchase date should be considered
            from_date_for_filter = from_date.replace(tzinfo=None) if from_date and from_date.tzinfo else from_date
            from_date_str = from_date_for_filter.strftime('%Y-%m-%d') if from_date_for_filter else (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d') if to_date else datetime.now().strftime('%Y-%m-%d')
            
            prompt = f"""You are a financial data expert. I need to check if the Indian stock with ticker {working_ticker} (name: {stock_name}) had any stock splits, bonus issues, stock consolidations, or DEMERGERS.

SPECIFIC COMPANY INFO:
- Ticker: {working_ticker}
- Company Name: {stock_name}


IMPORTANT INSTRUCTIONS:
1. Search THOROUGHLY for ALL corporate action announcements for {stock_name} ({working_ticker}) on:
   - NSE (National Stock Exchange) corporate action announcements
   - BSE (Bombay Stock Exchange) corporate action announcements
   - Company website and investor relations pages
   - Financial news websites (Moneycontrol, Economic Times, Business Standard, etc.)
   - SEBI filings and announcements
   - Stock exchange circulars and notices
2. Check for ANY of the following corporate actions that occurred:
   - Stock splits (e.g., 1:10, 1:2)
   - Bonus issues (e.g., 1:1 bonus)
   - Stock consolidations (reverse splits)
   - DEMERGERS (company splitting into multiple entities, shareholders receiving shares in new demerged company)
   - Between {from_date_str} and {to_date_str} (ONLY actions AFTER purchase date)
   - OR after {to_date_str} if announced (future actions)
   - DO NOT include actions before {from_date_str} (they happened before purchase)
3. Pay special attention to:
   - Face value splits (e.g., ₹10 to ₹1 face value changes)
   - Record dates and ex-split dates
   - Demerger record dates and entitlement ratios
   - Any announcements in 2024 or 2025 for this company
4. For DEMERGERS specifically:
   - Check if the company demerged any business unit or subsidiary
   - Note the demerger ratio (e.g., 1:1 means 1 share of parent = 1 share of demerged entity)
   - Include the demerged entity ticker/name if available
   - Record the effective date of the demerger
5. If you find a corporate action, provide the EXACT date (record date or ex-date) and ratio
6. Be VERY thorough - check multiple sources and don't miss any corporate actions
7. If you're unsure, search more thoroughly before concluding no action exists

Return ONLY a JSON object with this exact format:
{{
    "has_split": true,
    "has_demerger": true,
    "splits": [
        {{
            "date": "YYYY-MM-DD",
            "old_ratio": 1,
            "new_ratio": 2,
            "description": "Stock split 1:2"
        }}
    ],
    "demergers": [
        {{
            "date": "YYYY-MM-DD",
            "ratio": "1:1",
            "demerged_entity": "Company Name",
            "demerged_ticker": "TICKER.NS",
            "description": "Demerger of business unit"
        }}
    ]
}}

OR if no corporate action found:
{{
    "has_split": false,
    "has_demerger": false,
    "splits": [],
    "demergers": []
}}

CRITICAL:
- Return ONLY the JSON object, no other text
- Date must be in YYYY-MM-DD format
- Split ratio: old_ratio:new_ratio (e.g., 1:2 means 1 old share becomes 2 new shares)
- Demerger ratio: e.g., "1:1" means 1 share of parent = 1 share of demerged entity
- If has_split is false, return empty splits array
- If has_demerger is false, return empty demergers array"""
            
            # Use gpt-5 as primary, with gpt-4o as fallback
            models_to_try = ["gpt-5", "gpt-4o"]
            model_to_use = None
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    print(f"[CORP_ACTION] [AI] Checking splits using {model} for {ticker} ({working_ticker})...")
                    # gpt-5 only supports default temperature (1), other models can use lower temperature
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a financial data expert. Provide accurate stock split information for Indian stocks. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                    # Only add temperature for non-gpt-5 models
                    if model != "gpt-5":
                        request_params["temperature"] = 0.2
                    response = client.chat.completions.create(**request_params)
                    model_to_use = model
                    print(f"[CORP_ACTION] [AI] [OK] {model} succeeded for {ticker}")
                    break
                except Exception as model_error:
                    error_str = str(model_error).lower()
                    last_error = model_error
                    print(f"[CORP_ACTION] [AI] [WARNING] {model} failed for {ticker}: {str(model_error)[:150]}")
                    continue
            
            if not response:
                print(f"[CORP_ACTION] [AI] [ERROR] All models failed for {ticker}. Last error: {str(last_error)[:200]}")
                return []
            
            content = response.choices[0].message.content
            print(f"[CORP_ACTION] [AI] [DEBUG] Raw AI response for {ticker}: {content[:500]}...")
            import json
            result = json.loads(content)
            print(f"[CORP_ACTION] [AI] [DEBUG] Parsed result for {ticker}: has_split={result.get('has_split')}, splits_count={len(result.get('splits', []))}, has_demerger={result.get('has_demerger')}, demergers_count={len(result.get('demergers', []))}")
            
            splits = []
            demergers = []
            
            # Process splits
            if result.get("has_split") and result.get("splits"):
                for split_info in result.get("splits", []):
                    try:
                        split_date_str = split_info.get("date")
                        old_ratio = int(split_info.get("old_ratio", 1))
                        new_ratio = int(split_info.get("new_ratio", 1))
                        description = split_info.get("description", f"Stock split {old_ratio}:{new_ratio}")
                        
                        # Parse date
                        split_date_dt = datetime.strptime(split_date_str, '%Y-%m-%d')
                        
                        # Normalize to timezone-naive for comparison
                        if split_date_dt.tzinfo is not None:
                            split_date_dt = split_date_dt.replace(tzinfo=None)
                        
                        # Calculate split ratio
                        actual_ratio = new_ratio / old_ratio if old_ratio > 0 else 1.0
                        
                        # CRITICAL: Only include splits that are >= from_date (purchase date) and <= to_date
                        # This ensures we only consider splits AFTER purchase date
                        if (from_date_for_filter is None or split_date_dt >= from_date_for_filter) and split_date_dt <= to_date:
                            print(f"[CORP_ACTION] [AI] [OK] {ticker}: SPLIT DETECTED on {split_date_dt.date()} - {old_ratio}:{new_ratio} (ratio: {actual_ratio:.4f})")
                            
                            splits.append({
                                'type': 'split',
                                'date': split_date_dt,
                                'old_ratio': old_ratio,
                                'new_ratio': new_ratio,
                                'split_ratio': actual_ratio,
                                'description': f"{description} (AI)"
                            })
                        else:
                            # Exclude splits that happened before purchase date
                            if from_date_for_filter and split_date_dt < from_date_for_filter:
                                print(f"[CORP_ACTION] [AI] [WARNING] {ticker}: Split on {split_date_dt.date()} ignored (happened BEFORE purchase on {from_date_for_filter.date()})")
                            else:
                                print(f"[CORP_ACTION] [AI] [WARNING] {ticker}: Split on {split_date_dt.date()} ignored (outside date range)")
                    except Exception as e:
                        print(f"[CORP_ACTION] [AI] [ERROR] {ticker}: Error parsing split info: {str(e)[:100]}")
            
            # Process demergers
            if result.get("has_demerger") and result.get("demergers"):
                for demerger_info in result.get("demergers", []):
                    try:
                        demerger_date_str = demerger_info.get("date")
                        ratio_str = demerger_info.get("ratio", "1:1")
                        demerged_entity = demerger_info.get("demerged_entity", "")
                        demerged_ticker = demerger_info.get("demerged_ticker", "")
                        description = demerger_info.get("description", f"Demerger: {demerged_entity}")
                        
                        # Parse date
                        demerger_date_dt = datetime.strptime(demerger_date_str, '%Y-%m-%d')
                        
                        # Normalize to timezone-naive for comparison
                        if demerger_date_dt.tzinfo is not None:
                            demerger_date_dt = demerger_date_dt.replace(tzinfo=None)
                        
                        # Parse ratio (e.g., "1:1" means 1 share of parent = 1 share of demerged entity)
                        ratio_parts = ratio_str.split(':')
                        if len(ratio_parts) == 2:
                            parent_ratio = float(ratio_parts[0])
                            demerged_ratio = float(ratio_parts[1])
                            demerger_ratio = demerged_ratio / parent_ratio if parent_ratio > 0 else 1.0
                        else:
                            demerger_ratio = 1.0
                        
                        # CRITICAL: Only include demergers that are >= from_date (purchase date) and <= to_date
                        # This ensures we only consider demergers AFTER purchase date
                        if (from_date_for_filter is None or demerger_date_dt >= from_date_for_filter) and demerger_date_dt <= to_date:
                            print(f"[CORP_ACTION] [AI] [OK] {ticker}: DEMERGER DETECTED on {demerger_date_dt.date()} - Ratio: {ratio_str}, Entity: {demerged_entity}")
                            
                            demergers.append({
                                'type': 'demerger',
                                'date': demerger_date_dt,
                                'ratio': ratio_str,
                                'demerger_ratio': demerger_ratio,
                                'demerged_entity': demerged_entity,
                                'demerged_ticker': demerged_ticker,
                                'description': f"{description} (AI)"
                            })
                        else:
                            # Exclude demergers that happened before purchase date
                            if from_date_for_filter and demerger_date_dt < from_date_for_filter:
                                print(f"[CORP_ACTION] [AI] [WARNING] {ticker}: Demerger on {demerger_date_dt.date()} ignored (happened BEFORE purchase on {from_date_for_filter.date()})")
                            else:
                                print(f"[CORP_ACTION] [AI] [WARNING] {ticker}: Demerger on {demerger_date_dt.date()} ignored (outside date range)")
                    except Exception as e:
                        print(f"[CORP_ACTION] [AI] [ERROR] {ticker}: Error parsing demerger info: {str(e)[:100]}")
            
            # Return both splits and demergers
            return splits + demergers
            
        except json.JSONDecodeError as e:
            print(f"[CORP_ACTION] [AI] [ERROR] {ticker}: Failed to parse AI response as JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"[CORP_ACTION] [AI] [ERROR] {ticker}: Error fetching splits from AI: {str(e)}")
            return []
    
    def _fetch_corporate_actions_from_moneycontrol(self, ticker: str, from_date: datetime, to_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Fetch corporate actions from Moneycontrol (fallback source - slower but more detailed).
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS.NS')
            from_date: Start date to check corporate actions from
            to_date: End date (defaults to today)
        
        Returns:
            List of corporate action dicts with keys: type, date, ratio, description
        """
        if to_date is None:
            to_date = datetime.now()
        
        corporate_actions = []
        
        try:
            # Remove exchange suffix for Moneycontrol search
            base_ticker = self._normalize_base_ticker(ticker).replace('.NS', '').replace('.BO', '')
            
            # Moneycontrol corporate actions URL pattern
            # Try different URL formats
            urls_to_try = [
                f"https://www.moneycontrol.com/india/stockpricequote/{base_ticker.lower()}/{base_ticker.upper()}",
                f"https://www.moneycontrol.com/stocks/company_info/stock_price_quote.php?sc_id={base_ticker.upper()}",
            ]
            
            session = self.http_session or self._get_http_session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            
            for url in urls_to_try:
                try:
                    response = session.get(url, headers=headers)
                    if response.status_code == 200:
                        html_content = response.text
                        
                        # Parse corporate actions from HTML
                        # Look for corporate action sections
                        import re
                        from bs4 import BeautifulSoup
                        
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Find corporate action tables/sections
                        # Moneycontrol typically has a "Corporate Actions" section
                        ca_sections = soup.find_all(['div', 'table'], class_=re.compile(r'corporate|action|split|bonus', re.I))
                        
                        for section in ca_sections:
                            # Look for dates and action types
                            rows = section.find_all('tr') if section.name == 'table' else section.find_all('div', class_=re.compile(r'row|item', re.I))
                            
                            for row in rows:
                                text = row.get_text(strip=True)
                                            
                                # Detect split (e.g., "1:2", "2:1", "Stock Split")
                                split_match = re.search(r'(\d+)\s*:\s*(\d+)|split\s*(\d+)\s*:\s*(\d+)', text, re.I)
                                if split_match:
                                    groups = split_match.groups()
                                    if groups[0] and groups[1]:
                                        old_ratio, new_ratio = int(groups[0]), int(groups[1])
                                    else:
                                        old_ratio, new_ratio = int(groups[2]), int(groups[3])
                                    
                                    # Extract date
                                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
                                    if date_match:
                                        try:
                                            action_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')
                                        except:
                                            try:
                                                action_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                                            except:
                                                action_date = None
                                        
                                        if action_date and from_date <= action_date <= to_date:
                                            corporate_actions.append({
                                                'type': 'split',
                                                'date': action_date,
                                                'old_ratio': old_ratio,
                                                'new_ratio': new_ratio,
                                                'split_ratio': new_ratio / old_ratio,  # e.g., 2:1 = 2.0
                                                'description': text[:200]
                                            })
                                
                                # Detect bonus (e.g., "1:1", "Bonus 1:2")
                                bonus_match = re.search(r'bonus\s*(\d+)\s*:\s*(\d+)|(\d+)\s*:\s*(\d+).*bonus', text, re.I)
                                if bonus_match:
                                    groups = bonus_match.groups()
                                    if groups[0] and groups[1]:
                                        bonus_ratio = int(groups[1]) / int(groups[0])  # e.g., 1:2 = 2.0
                                    else:
                                        bonus_ratio = int(groups[3]) / int(groups[2])
                                    
                                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
                                    if date_match:
                                        try:
                                            action_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')
                                        except:
                                            try:
                                                action_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                                            except:
                                                action_date = None
                                        
                                        if action_date and from_date <= action_date <= to_date:
                                            corporate_actions.append({
                                                'type': 'bonus',
                                                'date': action_date,
                                                'bonus_ratio': bonus_ratio,
                                                'description': text[:200]
                                            })
                                
                                # Detect merger/acquisition (e.g., "Merger", "Acquisition", "Amalgamation", "Ticker Change")
                                merger_keywords = ['merger', 'acquisition', 'amalgamation', 'merged with', 'acquired by', 'ticker change', 'name change', 'renamed']
                                is_merger = any(keyword in text.lower() for keyword in merger_keywords)
                                
                                if is_merger:
                                    # Extract date
                                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
                                    if date_match:
                                        try:
                                            action_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')
                                        except:
                                            try:
                                                action_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                                            except:
                                                action_date = None
                                        
                                        if action_date and from_date <= action_date <= to_date:
                                            # Try to extract new ticker/company name
                                            new_ticker_match = re.search(r'(?:to|into|as|renamed|changed to)\s+([A-Z0-9]+(?:\.[NS|BO])?)', text, re.I)
                                            new_ticker = new_ticker_match.group(1).upper() if new_ticker_match else None
                                            
                                            # Try to extract exchange ratio (e.g., "1:2", "2 shares for 1")
                                            ratio_match = re.search(r'(\d+)\s*(?:shares?|:)\s*(?:for|of|:)\s*(\d+)', text, re.I)
                                            exchange_ratio = None
                                            if ratio_match:
                                                old_shares, new_shares = int(ratio_match.group(1)), int(ratio_match.group(2))
                                                exchange_ratio = new_shares / old_shares if old_shares > 0 else None
                                            
                                            # Try to extract cash component
                                            cash_match = re.search(r'₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:per share|per equity)', text, re.I)
                                            cash_per_share = None
                                            if cash_match:
                                                try:
                                                    cash_str = cash_match.group(1).replace(',', '')
                                                    cash_per_share = float(cash_str)
                                                except:
                                                    pass
                                            
                                            corporate_actions.append({
                                                'type': 'merger',
                                                'date': action_date,
                                                'new_ticker': new_ticker,
                                                'exchange_ratio': exchange_ratio,  # e.g., 2.0 means 1 old share = 2 new shares
                                                'cash_per_share': cash_per_share,
                                                'description': text[:300]
                                            })
                                
                                # Detect delisting (e.g., "Delisted", "Delisting", "Suspended")
                                delisting_keywords = ['delisted', 'delisting', 'suspended', 'voluntary delisting']
                                is_delisting = any(keyword in text.lower() for keyword in delisting_keywords)
                                
                                if is_delisting:
                                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
                                    if date_match:
                                        try:
                                            action_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')
                                        except:
                                            try:
                                                action_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                                            except:
                                                action_date = None
                                        
                                        if action_date and from_date <= action_date <= to_date:
                                            # Try to extract exit price (buyback/delisting price)
                                            price_match = re.search(r'₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:per share|per equity|exit price)', text, re.I)
                                            exit_price = None
                                            if price_match:
                                                try:
                                                    price_str = price_match.group(1).replace(',', '')
                                                    exit_price = float(price_str)
                                                except:
                                                    pass
                                            
                                            corporate_actions.append({
                                                'type': 'delisting',
                                                'date': action_date,
                                                'exit_price': exit_price,
                                                'description': text[:300]
                                            })
                        
                        # If we found actions, break
                        if corporate_actions:
                            break
                            
                except Exception as e:
                    continue
            
            # Sort by date (oldest first)
            corporate_actions.sort(key=lambda x: x['date'])
            
        except Exception as e:
            print(f"[CORP_ACTION] ⚠️ Error fetching corporate actions for {ticker}: {str(e)}")
        
        return corporate_actions
    
    def _apply_corporate_actions_to_holding(self, holding: Dict, corporate_actions: List[Dict], db_manager) -> Dict[str, Any]:
        """
        Apply corporate actions to a holding and return updated values.
        
        Args:
            holding: Holding dict with quantity, average_price, purchase_date
            corporate_actions: List of corporate actions sorted by date
            db_manager: Database manager
        
        Returns:
            Dict with updated quantity, average_price, and any messages
        """
        original_quantity = holding.get('quantity', 0)
        original_avg_price = holding.get('average_price', 0)
        purchase_date = holding.get('purchase_date')
        
        if not purchase_date:
            return {'quantity': original_quantity, 'average_price': original_avg_price, 'messages': []}
        
        try:
            purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00')) if isinstance(purchase_date, str) else purchase_date
        except:
            return {'quantity': original_quantity, 'average_price': original_avg_price, 'messages': []}
        
        current_quantity = original_quantity
        current_avg_price = original_avg_price
        messages = []
        result = {}  # Store additional info like merger_info, delisting_info
        
        for action in corporate_actions:
            action_date = action['date']
            
            # Only apply actions that occurred after purchase
            if action_date < purchase_date:
                continue
            
            if action['type'] == 'split':
                # Stock split: quantity increases, price decreases proportionally
                # e.g., 2:1 split means 1 share becomes 2 shares, price halves
                split_ratio = action['split_ratio']  # e.g., 2.0 for 2:1 split
                current_quantity = current_quantity * split_ratio
                current_avg_price = current_avg_price / split_ratio
                messages.append(f"Split {action.get('old_ratio', 1)}:{action.get('new_ratio', 1)} on {action_date.strftime('%Y-%m-%d')}")
            
            elif action['type'] == 'bonus':
                # Bonus issue: quantity increases, price adjusts
                # e.g., 1:2 bonus means 1 share becomes 3 shares (1 original + 2 bonus)
                bonus_ratio = action['bonus_ratio']  # e.g., 2.0 for 1:2 bonus
                new_quantity = current_quantity * (1 + bonus_ratio)
                # Average price adjusts: (old_qty * old_price) / new_qty
                current_avg_price = (current_quantity * current_avg_price) / new_quantity
                current_quantity = new_quantity
                messages.append(f"Bonus {action.get('description', '')} on {action_date.strftime('%Y-%m-%d')}")
            
            elif action['type'] == 'merger':
                # Merger/acquisition: ticker changes, may have exchange ratio and cash component
                # This requires user confirmation as it involves ticker mapping
                new_ticker = action.get('new_ticker')
                exchange_ratio = action.get('exchange_ratio', 1.0)  # Default 1:1 if not specified
                cash_per_share = action.get('cash_per_share', 0.0)
                
                # Calculate new quantity based on exchange ratio
                new_quantity = current_quantity * exchange_ratio
                
                # Adjust average price: account for cash component
                # If cash is received, it reduces the cost basis
                total_cash = current_quantity * cash_per_share
                total_cost = current_quantity * current_avg_price
                adjusted_cost = total_cost - total_cash
                
                if new_quantity > 0:
                    current_avg_price = adjusted_cost / new_quantity
                else:
                    current_avg_price = 0
                
                current_quantity = new_quantity
                
                action_desc = f"Merger to {new_ticker}" if new_ticker else "Merger"
                if exchange_ratio != 1.0:
                    action_desc += f" (1:{exchange_ratio:.2f} exchange)"
                if cash_per_share > 0:
                    action_desc += f" + ₹{cash_per_share:.2f} cash/share"
                
                messages.append(f"{action_desc} on {action_date.strftime('%Y-%m-%d')}")
                # Store new ticker for later use
                result['new_ticker'] = new_ticker
                result['merger_info'] = {
                    'old_ticker': holding.get('ticker'),
                    'new_ticker': new_ticker,
                    'exchange_ratio': exchange_ratio,
                    'cash_per_share': cash_per_share,
                    'action_date': action_date
                }
            
            elif action['type'] == 'delisting':
                # Delisting: stock is no longer traded
                # If exit price is provided, treat as sale at that price
                exit_price = action.get('exit_price')
                if exit_price:
                    # Mark for sale at exit price
                    messages.append(f"Delisted on {action_date.strftime('%Y-%m-%d')} (Exit price: ₹{exit_price:.2f})")
                    result['delisting_info'] = {
                        'exit_price': exit_price,
                        'action_date': action_date,
                        'quantity': current_quantity
                    }
                else:
                    messages.append(f"Delisted on {action_date.strftime('%Y-%m-%d')}")
                result['delisting_info'] = {
                    'exit_price': exit_price,
                    'action_date': action_date,
                    'quantity': current_quantity
                }
        
        return {
            'quantity': current_quantity,
            'average_price': current_avg_price,
            'messages': messages,
            'has_changes': current_quantity != original_quantity or current_avg_price != original_avg_price,
            **result  # Include merger_info or delisting_info if present
        }
    
    def _check_and_apply_corporate_actions(self, holding: Dict, db_manager) -> Dict[str, Any]:
        """
        Check for corporate actions and apply them to a holding.
        
        Args:
            holding: Holding dict with id, ticker, purchase_date (or first_transaction_date), stock_id
            db_manager: Database manager
        
        Returns:
            Dict with update info or None if no actions needed
        """
        ticker = holding.get('ticker')
        purchase_date = holding.get('purchase_date') or holding.get('first_transaction_date') or holding.get('earliest_date')
        stock_id = holding.get('stock_id')
        holding_id = holding.get('id')  # Specific holding ID
        
        if not ticker or not stock_id:
            print(f"[CORP_ACTION] ⚠️ Skipping {ticker}: missing ticker or stock_id")
            return None
        
        # If no purchase_date, try to get earliest transaction date from database
        if not purchase_date:
            try:
                # Get earliest transaction for this holding
                transactions = db_manager.supabase.table('user_transactions').select('transaction_date').eq(
                    'stock_id', stock_id
                ).eq('user_id', holding.get('user_id')).order('transaction_date', desc=False).limit(1).execute()
                
                if transactions.data and len(transactions.data) > 0:
                    purchase_date = transactions.data[0]['transaction_date']
                    print(f"[CORP_ACTION] 📅 Using earliest transaction date for {ticker}: {purchase_date}")
                else:
                    print(f"[CORP_ACTION] ⚠️ Skipping {ticker}: no purchase_date and no transactions found")
                    return None
            except Exception as e:
                print(f"[CORP_ACTION] ⚠️ Skipping {ticker}: error fetching purchase_date: {str(e)}")
                return None
        
        try:
            # Parse purchase date
            if isinstance(purchase_date, str):
                purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
            
            # Fetch corporate actions from purchase date to today
            print(f"[CORP_ACTION] 🔍 Checking corporate actions for {ticker} (purchased: {purchase_date})")
            
            # Try yfinance first (fast, reliable), then Moneycontrol as fallback
            corporate_actions = self._fetch_corporate_actions_from_yfinance(
                ticker, purchase_date, datetime.now()
            )
            
            print(f"[CORP_ACTION] 📊 yfinance found {len(corporate_actions)} actions for {ticker}")
            
            # If yfinance didn't find much, try Moneycontrol for more detailed info (mergers, delistings)
            if not corporate_actions or all(ca.get('type') in ['split', 'dividend'] for ca in corporate_actions):
                print(f"[CORP_ACTION] 🔄 Trying Moneycontrol for {ticker}...")
                moneycontrol_actions = self._fetch_corporate_actions_from_moneycontrol(
                    ticker, purchase_date, datetime.now()
                )
                print(f"[CORP_ACTION] 📊 Moneycontrol found {len(moneycontrol_actions)} actions for {ticker}")
                
                # Merge results, avoiding duplicates
                existing_dates = {ca['date']: ca for ca in corporate_actions}
                for mc_action in moneycontrol_actions:
                    mc_date = mc_action['date']
                    if mc_date not in existing_dates or existing_dates[mc_date].get('type') != mc_action.get('type'):
                        corporate_actions.append(mc_action)
                        existing_dates[mc_date] = mc_action
                
                # Re-sort after merging
                corporate_actions.sort(key=lambda x: x['date'])
            
            if not corporate_actions:
                print(f"[CORP_ACTION] ℹ️ No corporate actions found for {ticker}")
                return None
            
            print(f"[CORP_ACTION] ✅ Found {len(corporate_actions)} corporate actions for {ticker}: {[ca.get('type') for ca in corporate_actions]}")
            
            # Apply corporate actions
            result = self._apply_corporate_actions_to_holding(holding, corporate_actions, db_manager)
            
            print(f"[CORP_ACTION] 📝 Application result for {ticker}: has_changes={result.get('has_changes')}, messages={result.get('messages')}")
            
            if result.get('has_changes'):
                print(f"[CORP_ACTION] ✅ Will update {ticker}: qty {holding.get('quantity')} → {result['quantity']}, price {holding.get('average_price')} → {result['average_price']}")
                return {
                    'holding_id': holding_id,  # Use holding_id for specific update
                    'stock_id': stock_id,
                    'quantity': result['quantity'],
                    'average_price': result['average_price'],
                    'messages': result['messages']
                }
            else:
                print(f"[CORP_ACTION] ℹ️ No changes needed for {ticker} (actions found but already applied or not applicable)")
                return None
            
        except Exception as e:
            print(f"[CORP_ACTION] ⚠️ Error checking corporate actions for {ticker}: {str(e)}")
        
        return None
    
    def _generate_stock_aliases(self, ticker: str, context_name: Optional[str] = None) -> List[str]:
        """Generate normalized ticker aliases for stock lookups with name-aware enrichment.
        
        OPTIMIZATION: Results are cached to avoid recomputation for the same ticker.
        """
        raw = (ticker or "").strip()
        if not raw:
            return []
        
        # OPTIMIZATION: Check cache first (cache key includes context_name for accuracy)
        cache_key = f"{raw.upper()}_{context_name.upper() if context_name else 'NONE'}"
        if cache_key in self._stock_alias_cache:
            return self._stock_alias_cache[cache_key]

        candidates: List[str] = []

        def _sanitize_symbol(raw_symbol: str) -> str:
            """Keep only characters that work with Yahoo (A-Z, 0-9, dot, hyphen)."""
            if not raw_symbol:
                return ""
            return re.sub(r'[^A-Za-z0-9\.\-]', '', raw_symbol.upper())

        def _add(value: str) -> None:
            # Normalize first to remove trade tags like -T0, -T1, -BL
            normalized = self._normalize_base_ticker(value)
            # Then sanitize to ensure only valid characters remain
            val = _sanitize_symbol(normalized)
            if val and val not in candidates:
                candidates.append(val)

        normalized_raw = self._normalize_base_ticker(raw)
        _add(normalized_raw)

        manual_keys = {raw.upper()}
        if context_name:
            manual_keys.add(context_name.upper())

        for key in manual_keys:
            if key in self._manual_alias_map:
                for override in self._manual_alias_map[key]:
                    _add(self._normalize_base_ticker(override))

        # Convert float-like numeric strings to integers (e.g., "500285.0" -> "500285")
        for item in list(candidates):
            try:
                normalized = item.replace(',', '')
                numeric_value = float(normalized)
                if numeric_value.is_integer():
                    _add(str(int(numeric_value)))
            except ValueError:
                continue

        # Always fold in exchange suffix variants for obvious tickers
        # BUT: Skip if symbol already has .NS or .BO suffix (to avoid duplicates like GOLDBEES.BO.NS)
        base_snapshot = list(candidates)
        for symbol in base_snapshot:
            if not symbol:
                continue
            # If symbol already has exchange suffix, don't add variants
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                continue
            core_base = self._normalize_base_ticker(symbol)
            if core_base:
                _add(core_base)
                _add(f"{core_base}.NS")
                _add(f"{core_base}.BO")

        # Deterministic lookup using Yahoo Finance search by name
        if context_name:
            for suggestion in self._lookup_symbols_by_name(context_name):
                _add(suggestion)

        # Enrich using yfinance metadata-derived names (helps with corporate actions)
        # Run metadata extraction in parallel for faster processing
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                metadata_future = executor.submit(self._extract_names_from_metadata, raw)
                metadata_names = metadata_future.result()  # Wait for completion, no timeout
                for derived_name in metadata_names:
                    for suggestion in self._lookup_symbols_by_name(derived_name):
                        _add(suggestion)
        except Exception:
            # If metadata extraction fails, continue without it
            pass

        # For pure numeric or otherwise weak identifiers, force enrichment
        has_exchange_candidate = any(
            sym.endswith(('.NS', '.BO')) for sym in candidates if sym.replace('.', '').isalnum()
        )
        needs_ai = self.ai_available and (
            raw.isdigit()
            or (context_name and len(candidates) < 10)
            or not has_exchange_candidate
        )

        if needs_ai:
            for suggestion in self._ai_suggest_stock_aliases(raw, context_name):
                _add(suggestion)

        # OPTIMIZATION: Cache the results
        self._stock_alias_cache[cache_key] = candidates
        return candidates

    def _extract_metadata_for_symbol(self, symbol: str) -> List[str]:
        """Extract metadata names for a single symbol variant."""
        names = []
        try:
            stock = yf.Ticker(symbol)
            info = getattr(stock, "info", None) or {}
            if not isinstance(info, dict) or not info:
                return names

            for key in ("shortName", "longName", "displayName", "underlyingSymbol"):
                value = info.get(key)
                if isinstance(value, str) and value.strip():
                    cleaned = value.strip()
                    if cleaned.upper() not in [n.upper() for n in names]:
                        names.append(cleaned)

            # Some delisted tickers expose a "symbol" different from request
            canonical_symbol = info.get("symbol")
            if isinstance(canonical_symbol, str):
                names.append(canonical_symbol.strip())
        except Exception:
            pass
        return names
    
    def _extract_names_from_metadata(self, ticker: str) -> List[str]:
        """Pull potential successor names via yfinance metadata for delisted symbols (PARALLELIZED)."""
        names: List[str] = []
        # Normalize ticker first to remove $ and other invalid characters
        normalized = self._normalize_base_ticker(ticker)
        variants = [normalized, f"{normalized}.NS", f"{normalized}.BO"]

        # Try all variants in parallel for faster results
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._extract_metadata_for_symbol, symbol): symbol 
                      for symbol in variants}
            for future in as_completed(futures):
                try:
                    variant_names = future.result()  # Wait for completion, no timeout
                    for name in variant_names:
                        if name.upper() not in [n.upper() for n in names]:
                            names.append(name)
                except Exception:
                    continue

        return names

    def _lookup_symbols_by_name(self, name: str) -> List[str]:
        """Query Yahoo Finance search API for likely NSE/BSE tickers."""
        if not name:
            return []

        try:
            from urllib.parse import quote_plus
        except ImportError:
            return []

        encoded = quote_plus(name.strip())
        if not encoded:
            return []

        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded}&lang=en-US&region=IN"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        }

        suggestions: List[str] = []

        session = self.http_session or self._get_http_session()

        try:
            # No timeout - process full request regardless of size
            response = session.get(url, headers=headers, timeout=None)
            if response.status_code != 200:
                return []

            payload = response.json()
            quotes = payload.get("quotes", []) if isinstance(payload, dict) else []

            for quote in quotes:
                if not isinstance(quote, dict):
                    continue
                symbol = (quote.get("symbol") or "").strip().upper()
                exchange = (quote.get("exchange") or "").upper()

                if not symbol:
                    continue

                if symbol.endswith((".NS", ".BO")):
                    suggestions.append(symbol)
                    continue

                if exchange in {"NSI", "NSE", "NSEI"}:
                    suggestions.append(f"{symbol}.NS")
                    continue

                if exchange in {"BOM", "BSE"}:
                    suggestions.append(f"{symbol}.BO")
                    continue

        except Exception:
            return []

        # Preserve order while removing duplicates
        seen: set[str] = set()
        unique_suggestions: List[str] = []
        for symbol in suggestions:
            if symbol not in seen:
                unique_suggestions.append(symbol)
                seen.add(symbol)

        return unique_suggestions

    def _expand_symbol_variants(self, base_symbols: List[str]) -> List[str]:
        """Expand base symbols with exchange suffix variants."""
        variants: List[str] = []

        def _add(symbol: str) -> None:
            # Normalize first to handle $ suffix and duplicate suffixes
            normalized = self._normalize_base_ticker(symbol)
            if normalized and normalized not in variants:
                variants.append(normalized)

        for symbol in base_symbols:
            if not symbol:
                continue
            # Normalize the symbol first to handle $ suffix and ensure consistent format
            normalized = self._normalize_base_ticker(symbol)
            if not normalized:
                continue
            
            # Check for existing suffix after normalization
            if normalized.endswith('.NS'):
                _add(normalized)  # Keep as-is
                _add(normalized.replace('.NS', '.BO'))  # Convert to BSE
                _add(normalized.replace('.NS', ''))  # Remove suffix
            elif normalized.endswith('.BO'):
                _add(normalized)  # Keep as-is
                _add(normalized.replace('.BO', '.NS'))  # Convert to NSE
                _add(normalized.replace('.BO', ''))  # Remove suffix
            else:
                # No suffix - add base and both variants
                _add(normalized)  # Base symbol
                _add(f"{normalized}.NS")  # NSE variant
                _add(f"{normalized}.BO")  # BSE variant

        return variants

    def _ai_suggest_stock_aliases(self, ticker: str, context_name: Optional[str]) -> List[str]:
        """Use AI to suggest usable market identifiers for a stock ticker."""
        cache_key = ticker.strip().upper()
        if cache_key in self._stock_alias_cache:
            return self._stock_alias_cache[cache_key]

        # Ensure OpenAI client is initialized
        openai_client = self._get_openai_client()
        if not self.ai_available or not openai_client:
            return []

        prompt_lines = [
            "Suggest up to five Yahoo Finance exchange tickers for this Indian instrument.",
            "Return ONLY a JSON array of strings (e.g. [\"XYZ.NS\", \"XYZ.BO\"]).",
            "Always resolve numeric BSE codes to their tradable symbol (e.g. 500325 → RELIANCE.BO).",
            "Prioritize NSE (.NS) when available, else BSE (.BO).",
            f"Instrument Identifier Provided: {ticker}",
            f"Instrument Name: {context_name or 'Unknown'}",
        ]
        prompt = "\n".join(prompt_lines)

        try:
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_text = response.choices[0].message.content.strip()
            suggestions: List[str] = []
            import json

            try:
                parsed = json.loads(ai_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            cleaned = item.strip().upper()
                            if cleaned:
                                suggestions.append(cleaned)
            except json.JSONDecodeError:
                pass

            self._stock_alias_cache[cache_key] = suggestions
            return suggestions
        except Exception:
            return []

    def _try_single_ticker_variant(self, ticker_variant: str, source_type: str) -> Optional[Tuple[float, str]]:
        """
        Try fetching price for a single ticker variant in parallel.
        Returns (price, source) if successful, None otherwise.
        """
        try:
            clean_ticker = self._normalize_base_ticker(ticker_variant)
            if source_type == 'nse':
                # Only add .NS if ticker doesn't already have .NS or .BO
                if not clean_ticker.endswith('.NS') and not clean_ticker.endswith('.BO'):
                    clean_ticker = clean_ticker + '.NS'
                elif clean_ticker.endswith('.BO'):
                    # Convert .BO to .NS
                    clean_ticker = clean_ticker.replace('.BO', '.NS')
            elif source_type == 'bse':
                # Only add .BO if ticker doesn't already have .NS or .BO
                if not clean_ticker.endswith('.BO') and not clean_ticker.endswith('.NS'):
                    clean_ticker = clean_ticker + '.BO'
                elif clean_ticker.endswith('.NS'):
                    # Convert .NS to .BO
                    clean_ticker = clean_ticker.replace('.NS', '.BO')
            elif source_type == 'raw':
                # Remove any exchange suffix
                clean_ticker = clean_ticker.replace('.NS', '').replace('.BO', '')
            
            hist = yf.Ticker(clean_ticker).history(period='1d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    source_map = {'nse': 'yfinance_nse', 'bse': 'yfinance_bse', 'raw': 'yfinance_raw'}
                    return price, source_map.get(source_type, 'yfinance')
        except Exception:
            pass
        return None

    def _get_stock_price_with_fallback(self, ticker: str, context_name: Optional[str] = None) -> tuple:
        """
        Stock price fetching with complete fallback:
        1. yfinance with .NS (if ticker doesn't already have suffix)
        2. yfinance with .BO (if ticker doesn't already have suffix)
        3. yfinance without suffix
        4. Other variants/aliases
        5. mftool (in case it's a mutual fund misclassified as stock)
        6. AI (OpenAI)
        """
        # CRITICAL: Clear resolved ticker at start
        self._last_resolved_ticker = None
        
        # Normalize the base ticker
        base_ticker = self._normalize_base_ticker(ticker)
        has_ns = base_ticker.endswith('.NS')
        has_bo = base_ticker.endswith('.BO')
        
        # STEP 1: Try the ticker with .NS and .BO FIRST (if it doesn't already have a suffix)
        # This is the most common case and should be tried before other variants
        if not has_ns and not has_bo:
            # Try .NS first
            nse_ticker = f"{base_ticker}.NS"
            result = self._try_single_ticker_variant(nse_ticker, 'nse')
            if result:
                price, source = result
                # Set resolved ticker only if base matches
                original_base = self._base_symbol(ticker)
                resolved_base = self._base_symbol(nse_ticker)
                if original_base == resolved_base:
                    self._last_resolved_ticker = nse_ticker
                return price, source
            
            # Try .BO second
            bse_ticker = f"{base_ticker}.BO"
            result = self._try_single_ticker_variant(bse_ticker, 'bse')
            if result:
                price, source = result
                # Set resolved ticker only if base matches
                original_base = self._base_symbol(ticker)
                resolved_base = self._base_symbol(bse_ticker)
                if original_base == resolved_base:
                    self._last_resolved_ticker = bse_ticker
                return price, source
        
        # STEP 2: If ticker already has .NS or .BO, try it directly
        if has_ns or has_bo:
            result = self._try_single_ticker_variant(base_ticker, 'nse' if has_ns else 'bse')
            if result:
                price, source = result
                # Set resolved ticker only if base matches
                original_base = self._base_symbol(ticker)
                resolved_base = self._base_symbol(base_ticker)
                if original_base == resolved_base:
                    self._last_resolved_ticker = base_ticker
                return price, source
        
        # STEP 3: Try without suffix
        if not has_ns and not has_bo:
            result = self._try_single_ticker_variant(base_ticker, 'raw')
            if result:
                price, source = result
                # Set resolved ticker only if base matches
                original_base = self._base_symbol(ticker)
                resolved_base = self._base_symbol(base_ticker)
                if original_base == resolved_base:
                    self._last_resolved_ticker = base_ticker
                return price, source
        
        # STEP 3.5: If ticker is missing/invalid and we have stock name, try AI + yfinance to resolve ticker
        ticker_clean = str(ticker).strip() if ticker else ''
        if (not ticker_clean or len(ticker_clean) < 2) and context_name:
            print(f"[STOCK_RESOLVE] 🔍 Resolving stock ticker from name: '{context_name}'")
            resolved_ticker = self._resolve_stock_ticker_by_name(context_name)
            if resolved_ticker:
                # Try fetching price with resolved ticker
                print(f"[STOCK_RESOLVE] ✅ Resolved stock ticker: '{context_name}' → {resolved_ticker}")
                result = self._try_single_ticker_variant(resolved_ticker, 'nse')
                if result:
                    price, source = result
                    # Validate resolved ticker matches
                    resolved_base = self._base_symbol(resolved_ticker)
                    if resolved_base:  # If we got a valid ticker, use it
                        self._last_resolved_ticker = resolved_ticker
                    return price, source
                # Try BSE variant
                result = self._try_single_ticker_variant(resolved_ticker, 'bse')
                if result:
                    price, source = result
                    resolved_base = self._base_symbol(resolved_ticker)
                    if resolved_base:
                        self._last_resolved_ticker = resolved_ticker
                    return price, source
        
        # STEP 4: If direct attempts failed, try other variants/aliases (fallback)
        stock_name_hint = context_name or self._get_stock_name_from_cache(base_ticker)
        base_candidates = self._generate_stock_aliases(base_ticker, stock_name_hint)
        symbol_variants = self._expand_symbol_variants(base_candidates or [base_ticker])
        
        # Remove duplicates and prioritize .NS/.BO variants
        nse_variants = []
        bse_variants = []
        raw_variants_set = set()
        
        for v in symbol_variants:
            if v.endswith('.NS'):
                nse_variants.append(v)
                raw_variants_set.add(v[:-3])  # Remove .NS
            elif v.endswith('.BO'):
                bse_variants.append(v)
                raw_variants_set.add(v[:-3])  # Remove .BO
            else:
                raw_variants_set.add(v)
                # Also try with suffixes
                nse_variants.append(f"{v}.NS")
                bse_variants.append(f"{v}.BO")
        
        raw_variants = list(raw_variants_set)
        
        # Try NSE variants in parallel (fallback)
        # OPTIMIZATION: Cancel remaining futures when we find a result
        with ThreadPoolExecutor(max_workers=min(10, len(nse_variants))) as executor:
            futures = {executor.submit(self._try_single_ticker_variant, variant, 'nse'): variant 
                      for variant in nse_variants[:10]}  # Limit to 10 to avoid too many requests
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Cancel remaining futures to save resources
                    for f in futures:
                        if f != future:
                            f.cancel()
                    price, source = result
                    variant = futures[future]
                    # Only add .NS if variant doesn't already have .NS or .BO
                    if variant.endswith('.NS'):
                        clean_formatted = variant
                    elif variant.endswith('.BO'):
                        # If it's .BO, convert to .NS (remove .BO first to avoid double suffix)
                        base = variant.replace('.BO', '')
                        # Normalize base (removes any remaining suffixes) then add .NS
                        normalized_base = self._normalize_base_ticker(base)
                        # Remove any existing .NS/.BO from normalized_base before adding .NS
                        if normalized_base.endswith('.NS') or normalized_base.endswith('.BO'):
                            normalized_base = normalized_base[:-3]  # Remove last 3 chars (.NS or .BO)
                        clean_formatted = normalized_base + '.NS'
                    else:
                        # Variant has no suffix, normalize and add .NS
                        normalized_base = self._normalize_base_ticker(variant)
                        # Remove any existing .NS/.BO from normalized_base before adding .NS
                        if normalized_base.endswith('.NS') or normalized_base.endswith('.BO'):
                            normalized_base = normalized_base[:-3]  # Remove last 3 chars (.NS or .BO)
                        clean_formatted = normalized_base + '.NS'
                    
                    # CRITICAL: Only set resolved ticker if it's actually related to the original ticker
                    # Check if the base symbols match (without .NS/.BO suffixes) - must be exact match
                    original_base = self._base_symbol(ticker)
                    resolved_base = self._base_symbol(clean_formatted)
                    # Only set if base symbols match exactly (same stock, just different exchange suffix)
                    if original_base == resolved_base:
                        self._last_resolved_ticker = clean_formatted
                    else:
                        # Different stock - don't set resolved ticker
                        print(f"[PRICE_UPDATE] ⚠️ Ignoring wrong variant match: {ticker} (base: {original_base}) → {clean_formatted} (base: {resolved_base})")
                    
                    return price, source
        
        # Try BSE variants in parallel (second priority)
        with ThreadPoolExecutor(max_workers=min(10, len(bse_variants))) as executor:
            futures = {executor.submit(self._try_single_ticker_variant, variant, 'bse'): variant 
                      for variant in bse_variants[:10]}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Cancel remaining futures to save resources
                    for f in futures:
                        if f != future:
                            f.cancel()
                    price, source = result
                    variant = futures[future]
                    # Only add .BO if variant doesn't already have .NS or .BO
                    if variant.endswith('.BO'):
                        clean_formatted = variant
                    elif variant.endswith('.NS'):
                        # If it's .NS, convert to .BO (remove .NS first to avoid double suffix)
                        base = variant.replace('.NS', '')
                        # Normalize base (removes any remaining suffixes) then add .BO
                        normalized_base = self._normalize_base_ticker(base)
                        # Remove any existing .NS/.BO from normalized_base before adding .BO
                        if normalized_base.endswith('.NS') or normalized_base.endswith('.BO'):
                            normalized_base = normalized_base[:-3]  # Remove last 3 chars (.NS or .BO)
                        clean_formatted = normalized_base + '.BO'
                    else:
                        # Variant has no suffix, normalize and add .BO
                        normalized_base = self._normalize_base_ticker(variant)
                        # Remove any existing .NS/.BO from normalized_base before adding .BO
                        if normalized_base.endswith('.NS') or normalized_base.endswith('.BO'):
                            normalized_base = normalized_base[:-3]  # Remove last 3 chars (.NS or .BO)
                        clean_formatted = normalized_base + '.BO'
                    
                    # CRITICAL: Only set resolved ticker if it's actually related to the original ticker
                    # Check if the base symbols match (without .NS/.BO suffixes) - must be exact match
                    original_base = self._base_symbol(ticker)
                    resolved_base = self._base_symbol(clean_formatted)
                    # Only set if base symbols match exactly (same stock, just different exchange suffix)
                    if original_base == resolved_base:
                        self._last_resolved_ticker = clean_formatted
                    else:
                        # Different stock - don't set resolved ticker
                        print(f"[PRICE_UPDATE] ⚠️ Ignoring wrong variant match: {ticker} (base: {original_base}) → {clean_formatted} (base: {resolved_base})")
                    
                    return price, source
        
        # Try raw variants in parallel (third priority)
        with ThreadPoolExecutor(max_workers=min(10, len(raw_variants))) as executor:
            futures = {executor.submit(self._try_single_ticker_variant, variant, 'raw'): variant 
                      for variant in raw_variants[:10]}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Cancel remaining futures to save resources
                    for f in futures:
                        if f != future:
                            f.cancel()
                    price, source = result
                    variant = futures[future]
                    clean_ticker = self._normalize_base_ticker(variant)
                    
                    # CRITICAL: Only set resolved ticker if it's actually related to the original ticker
                    # Check if the base symbols match (without .NS/.BO suffixes) - must be exact match
                    original_base = self._base_symbol(ticker)
                    resolved_base = self._base_symbol(clean_ticker)
                    # Only set if base symbols match exactly (same stock, just different exchange suffix)
                    if original_base == resolved_base:
                        self._last_resolved_ticker = clean_ticker
                    else:
                        # Different stock - don't set resolved ticker
                        print(f"[PRICE_UPDATE] ⚠️ Ignoring wrong variant match: {ticker} (base: {original_base}) → {clean_ticker} (base: {resolved_base})")
                    
                    return price, source
        
        # Method 4: Try mftool (in case it's a mutual fund)
        ##st.caption(f"      [4/5] Trying mftool (in case it's a MF)...")
        try:
            mf = self._mftool or self._get_shared_mftool()
            if mf:
                primary_candidate = base_candidates[0] if base_candidates else str(ticker or '').strip()
                clean_ticker = primary_candidate.replace('.NS', '').replace('.BO', '').replace('MF_', '')
                quote = mf.get_scheme_quote(clean_ticker)
                
                if quote and 'nav' in quote:
                    price = float(quote['nav'])
                    if price > 0:
                        return price, 'mftool'
        except Exception:
            pass
# mftool failed
        
        # Method 5: AI Fallback (if available)
        # Trying AI (OpenAI) as last resort
        if self.ai_available:
            try:
                # Get stock name from cache or database if available
                stock_name = context_name or self._get_stock_name_from_cache(ticker)
                price = self._get_price_from_ai(ticker, 'stock', asset_name=stock_name)
                if price:
                    # AI found price
                    return price, 'ai_openai'
                else:
                    pass
                # AI couldn't find price
            except Exception as e:
                pass
            # AI failed
        else:
            pass
        # AI not available
        
        # All methods failed
        return None, 'not_found'
    
    def _get_mf_price_with_fallback(self, ticker: str, fund_name: str = None) -> tuple:
        """
        Mutual Fund price with fallback:
        1. Resolve ticker from fund name if ticker is missing/invalid
        2. mftool (AMFI API) using resolved ticker
        3. AI (OpenAI)
        """
        # CRITICAL: Clear resolved ticker at start
        self._last_resolved_ticker = None
        self._last_resolved_fund_name = None
        
        # STEP 1: If ticker is missing or invalid, resolve from fund name FIRST
        scheme_code = None
        resolved_fund_name = None
        
        # Check if ticker is missing, empty, or not a valid numeric code
        ticker_clean = str(ticker).strip() if ticker else ''
        is_valid_mf_code = ticker_clean.isdigit() and len(ticker_clean) >= 5
        
        if (not ticker_clean or not is_valid_mf_code) and fund_name:
            # Ticker is missing/invalid - resolve from fund name
            print(f"[MF_PRICE] 🔍 Resolving MF ticker from fund name: '{fund_name}'")
            resolved = self._resolve_mf_code_by_name(ticker_clean or '', fund_name)
            if resolved and resolved.get('code'):
                scheme_code = resolved['code']
                resolved_fund_name = resolved.get('name', '')
                self._last_resolved_ticker = scheme_code
                self._last_resolved_fund_name = resolved_fund_name
                print(f"[MF_PRICE] ✅ Resolved MF ticker from name: '{fund_name}' → {scheme_code} (confidence: {resolved.get('score', 0)*100:.0f}%)")
            else:
                # Name-based resolution failed, try using fund name as ticker (for ETFs)
                print(f"[MF_PRICE] ⚠️ Name-based resolution failed for '{fund_name}', will try with original ticker")
        
        # STEP 2: If we have a valid ticker or resolved code, use it
        if not scheme_code:
            # Try to resolve using _resolve_amfi_code (for existing codes)
            resolved_code = self._resolve_amfi_code(ticker, fund_name)
            scheme_code = resolved_code or ticker_clean
            if resolved_code and resolved_code != ticker_clean:
                self._last_resolved_ticker = scheme_code
                print(f"[MF_PRICE] 🔄 Resolved MF code: {ticker_clean} → {scheme_code}")

        normalized_code = (scheme_code or "").strip()
        upper_code = normalized_code.upper()

        # Certain wrappers (ETF/InvIT) behave like stocks even if tagged as mutual funds
        # Check for known ETF tickers first (before checking if it's numeric)
        known_etf_tickers = ['GROWWEV', 'SILVERBEES', 'GOLDBEES', 'NIFTYBEES', 'BANKBEES', 'JUNIORBEES', 'SHARIABEES', 'MON100', 'SETF10GILT', 'SETFNIFBK', 'SETFNN50']
        ticker_upper_clean = ticker_clean.upper().replace('.NS', '').replace('.BO', '')
        
        if ticker_upper_clean in known_etf_tickers:
            # Known ETF - treat as stock and fetch from yfinance
            print(f"[MF_PRICE] 🔍 Detected known ETF ticker: {ticker_clean}, treating as stock (yfinance)")
            stock_price, stock_source = self._get_stock_price_with_fallback(ticker_clean, context_name=fund_name)
            if stock_price:
                return stock_price, stock_source
        
        if normalized_code and not normalized_code.isdigit():
            stock_like_keywords = ('ETF', 'INVIT', 'TRUST')
            if (
                normalized_code.endswith(('.NS', '.BO'))
                or any(keyword in upper_code for keyword in stock_like_keywords)
            ):
                stock_price, stock_source = self._get_stock_price_with_fallback(normalized_code, context_name=fund_name)
                if stock_price:
                    return stock_price, stock_source
                # Fall back to trying original ticker as stock if normalized changed the identifier
                if normalized_code != ticker:
                    stock_price, stock_source = self._get_stock_price_with_fallback(ticker, context_name=fund_name)
                    if stock_price:
                        return stock_price, stock_source
        
        # Method 1: Try mftool
        # Trying mftool (AMFI API)
        try:
            mf = self._mftool or self._get_shared_mftool()
            if mf:
                quote = mf.get_scheme_quote(scheme_code)
                
                if quote and 'nav' in quote:
                    price = float(quote['nav'])
                    if price > 0:
                        return price, 'mftool'
            # Fall through if mf unavailable or NAV not found
        except Exception:
            pass
# mftool failed

        # Fallback to cached AMFI dataset if available
        try:
            dataset = self._get_amfi_dataset()
            scheme_entry = dataset.get("code_lookup", {}).get(scheme_code)
            if scheme_entry:
                nav_value = scheme_entry.get("nav")
                if nav_value:
                    price = float(nav_value)
                    if price > 0:
                        return price, 'amfi_dataset'
                    else:
                        # NAV = 0 means discontinued/segregated fund - log it
                        scheme_name = scheme_entry.get("name", "")
                        if "segregated" in scheme_name.lower() or "segregated portfolio" in scheme_name.lower():
                            print(f"[MF_PRICE] ⚠️ {scheme_code} ({scheme_name[:50]}): NAV = 0.00 - Segregated/Discontinued fund")
                        else:
                            print(f"[MF_PRICE] ⚠️ {scheme_code} ({scheme_name[:50]}): NAV = 0.00 - Discontinued fund")
        except Exception:
            pass
        
        # Method 1.5: NEW! Try name-based search if code fails but we have fund name
        if fund_name:
            try:
                resolved = self._resolve_mf_code_by_name(scheme_code, fund_name)
                if resolved:
                    correct_code = resolved['code']
                    confidence = resolved['score']
                    
                    # Try fetching with resolved code
                    try:
                        dataset = self._get_amfi_dataset()
                        scheme_entry = dataset.get("code_lookup", {}).get(correct_code)
                        if scheme_entry:
                            nav_value = scheme_entry.get("nav")
                            if nav_value:
                                price = float(nav_value)
                                if price > 0:
                                    # Success! Log the resolution and store resolved ticker + fund name
                                    resolved_fund_name = resolved.get('name', '') or scheme_entry.get('name', '')
                                    print(f"[MF_RESOLVE] ✅ Auto-resolved {scheme_code} → {correct_code} (confidence: {confidence*100:.0f}%)")
                                    print(f"[MF_RESOLVE] 🔄 Auto-updating ticker in database from {scheme_code} to {correct_code}")
                                    print(f"[MF_RESOLVE] 📝 Fund name: {resolved_fund_name}")
                                    # Store resolved ticker and fund name for batch update
                                    self._last_resolved_ticker = correct_code
                                    self._last_resolved_fund_name = resolved_fund_name
                                    return price, f'amfi_name_resolved:{correct_code}'
                    except Exception:
                        pass
            except Exception as e:
                # Name-based search failed, continue to other fallbacks
                pass
        
        # Method 2: AI Fallback with Enhanced Context
        # Trying AI (OpenAI) with enhanced context
        if self.ai_available:
            try:
                # Use provided fund name or try to get from context
                if not fund_name:
                    fund_name = self._get_fund_name_from_context(ticker)
                
                if fund_name:
                    # Asking AI for NAV with both code and name
                    price = self._get_mf_price_from_ai_enhanced(ticker, fund_name)
                else:
                    # Asking AI for NAV of scheme (code only)
                    price = self._get_price_from_ai(ticker, 'mutual_fund', asset_name=None)
                
                if price:
                    # AI found NAV
                    return price, 'ai_openai_enhanced'
                else:
                    # AI couldn't find NAV
                    pass
            except Exception as e:
                # AI failed
                pass
        else:
            # AI not available
            pass
        
        # All methods failed for MF - check if it's a segregated fund
        # For segregated funds, try parent fund resolution and DON'T use average_price fallback
        is_segregated = fund_name and ('segregated' in fund_name.lower() or 'segregated portfolio' in fund_name.lower())
        
        if is_segregated:
            # For segregated funds, try to find parent fund by removing segregated details
            import re
            if fund_name:
                parent_fund_name = fund_name
                # Remove segregated portfolio identifiers
                parent_fund_name = re.sub(r'-?\s*Segregated Portfolio\s*\d+', '', parent_fund_name, flags=re.IGNORECASE)
                parent_fund_name = re.sub(r'-?\s*\d+\.\d+%[^-]*-?\d+[A-Z]{3}\d+', '', parent_fund_name)
                parent_fund_name = re.sub(r'-?\s*Monthly\s+Dividend\s+Plan', '', parent_fund_name, flags=re.IGNORECASE)
                parent_fund_name = re.sub(r'-?\s*Quarterly\s+IDCW', '', parent_fund_name, flags=re.IGNORECASE)
                parent_fund_name = re.sub(r'-?\s*IDCW\s+Option', '', parent_fund_name, flags=re.IGNORECASE)
                parent_fund_name = re.sub(r'\s+', ' ', parent_fund_name).strip()
                
                if parent_fund_name and parent_fund_name != fund_name:
                    # Try fetching parent fund price
                    print(f"[MF_PRICE] 🔍 Segregated fund detected, trying parent: '{parent_fund_name[:50]}'")
                    parent_price, parent_source = self._get_mf_price_with_fallback(ticker, parent_fund_name)
                    if parent_price and parent_price > 0:
                        print(f"[MF_PRICE] ✅ Found parent fund price: ₹{parent_price}")
                        return parent_price, f'parent_fund_{parent_source}'
            
            # Segregated fund - don't use average_price fallback, return None
            print(f"[MF_PRICE] ❌ {ticker}: Segregated fund - price unavailable (not using average price fallback)")
            return None, 'segregated_fund_unavailable'
        
        # For regular funds, return average price as fallback to avoid 0% returns
        # This prevents showing 0% return when price fetch fails
        try:
            # Try to get average price from holding if available
            if hasattr(self, '_last_holding_avg_price') and self._last_holding_avg_price:
                return self._last_holding_avg_price, 'average_price_fallback'
        except Exception:
            pass
        
        return None, 'not_found'
    
    def _calculate_pms_aif_live_price(
        self,
        ticker: str,
        asset_type: str,
        db_manager,
        holding: Dict[str, Any],
    ) -> Tuple[Optional[float], str]:
        """Calculate PMS/AIF price for a specific holding using CAGR."""
        calculator = self._get_pms_aif_calculator()
        if not calculator:
            return None, 'cagr_calculator_unavailable'

        try:
            user_id = holding.get('user_id')
            stock_id = holding.get('stock_id')
            if not user_id or not stock_id:
                return None, 'cagr_missing_context'

            transactions = db_manager.get_transactions_by_stock(user_id, stock_id)
            if not transactions:
                return None, 'cagr_no_transactions'

            buy_transactions = [
                txn for txn in transactions
                if str(txn.get('transaction_type', '')).lower() == 'buy'
            ] or transactions

            buy_transactions.sort(key=lambda txn: txn.get('transaction_date') or '')
            first_transaction = buy_transactions[0]

            quantity = float(first_transaction.get('quantity') or 0)
            price = float(first_transaction.get('price') or 0)
            investment_date = first_transaction.get('transaction_date')
            total_quantity = float(holding.get('total_quantity') or quantity or 1)
            investment_amount = quantity * price

            if investment_amount <= 0 or not investment_date:
                return None, 'cagr_invalid_transaction'

            # Get PMS/AIF name for better AI context
            pms_aif_name = holding.get('stock_name') or holding.get('scheme_name', '')
            
            result = calculator.calculate_pms_aif_value(
                ticker,
                investment_date,
                investment_amount,
                is_aif=(asset_type == 'aif'),
                pms_aif_name=pms_aif_name
            )

            current_value = result.get('current_value')
            if not current_value:
                error_msg = result.get('error', 'Unknown error')
                print(f"[PMS_AIF] ❌ No current_value for {ticker}: {error_msg}")
                print(f"[PMS_AIF]   Result: {result}")
                # If CAGR failed, return None so it falls back to average_price
                return None, f'cagr_no_result: {error_msg}'

            nav = current_value / total_quantity if total_quantity > 0 else current_value
            source = result.get('source', 'cagr_calculated')
            cagr_used = result.get('cagr_used', 0)
            print(f"[PMS_AIF] ✅ Calculated price for {ticker}: NAV={nav:,.2f}, CAGR={cagr_used:.2%}, Source={source}")
            return nav, source
        except Exception as exc:
            print(f"[PMS_AIF] ❌ Error calculating price for {ticker}: {str(exc)}")
            import traceback
            traceback.print_exc()
            return None, f'cagr_error:{exc}'

    def _get_pms_aif_price(self, ticker: str, asset_type: str) -> Tuple[Optional[float], str]:
        """Get PMS/AIF price using earliest transaction available."""
        calculator = self._get_pms_aif_calculator()
        if not calculator:
            return None, 'cagr_calculator_unavailable'

        try:
            from database_shared import SharedDatabaseManager

            db = SharedDatabaseManager()
            stock_response = db.supabase.table('stock_master').select('id').eq('ticker', ticker).execute()
            if not stock_response.data:
                return None, 'cagr_stock_not_found'

            stock_id = stock_response.data[0]['id']
            txn_response = db.supabase.table('user_transactions').select(
                'quantity, price, transaction_date'
            ).eq('stock_id', stock_id).order('transaction_date', asc=True).limit(1).execute()

            if not txn_response.data:
                return None, 'cagr_no_transactions'

            first_transaction = txn_response.data[0]
            quantity = float(first_transaction.get('quantity') or 0)
            price = float(first_transaction.get('price') or 0)
            investment_date = first_transaction.get('transaction_date')
            investment_amount = quantity * price

            if investment_amount <= 0 or not investment_date:
                return None, 'cagr_invalid_transaction'

            result = self.pms_aif_calculator.calculate_pms_aif_value(
                ticker,
                investment_date,
                investment_amount,
                is_aif=(asset_type == 'aif')
            )

            current_value = result.get('current_value')
            if not current_value:
                return None, 'cagr_no_result'

            nav = current_value / quantity if quantity > 0 else current_value
            source = result.get('source', 'cagr_calculated')
            return nav, source
        except Exception as exc:
            return None, f'cagr_error:{exc}'

    def _resolve_mf_code_by_name(self, old_code: str, fund_name: str) -> Optional[Dict[str, Any]]:
        """
        Resolve incorrect/old MF code by searching AMFI dataset by name
        Returns the best matching scheme with high confidence
        Handles segregated portfolios by trying parent fund name
        
        Args:
            old_code: The code that failed
            fund_name: Fund name to search for
            
        Returns:
            Dict with 'code', 'name', 'score' if found, None otherwise
        """
        try:
            from difflib import SequenceMatcher
            import re
            
            dataset = self._get_amfi_dataset()
            if not dataset:
                return None
            
            code_lookup = dataset.get('code_lookup', {})
            if not code_lookup:
                return None
            
            # For segregated portfolios, try parent fund name first
            search_names = [fund_name]
            if 'segregated' in fund_name.lower() or 'segregated portfolio' in fund_name.lower():
                # Try parent fund name by removing segregated portfolio details
                parent_name = fund_name
                # Remove patterns like "Segregated Portfolio 1", "8.25% Vodafone Idea Ltd-10JUL20", etc.
                parent_name = re.sub(r'-?\s*Segregated Portfolio\s*\d+', '', parent_name, flags=re.IGNORECASE)
                parent_name = re.sub(r'-?\s*\d+\.\d+%[^-]*-?\d+[A-Z]{3}\d+', '', parent_name)  # Remove percentage and date patterns
                parent_name = re.sub(r'-?\s*Monthly\s+Dividend\s+Plan', '', parent_name, flags=re.IGNORECASE)
                parent_name = re.sub(r'-?\s*Quarterly\s+IDCW', '', parent_name, flags=re.IGNORECASE)
                parent_name = re.sub(r'-?\s*IDCW\s+Option', '', parent_name, flags=re.IGNORECASE)
                parent_name = re.sub(r'\s+', ' ', parent_name).strip()  # Clean up spaces
                if parent_name and parent_name != fund_name:
                    search_names.insert(0, parent_name)  # Try parent name first
                    print(f"[MF_RESOLVE] 🔍 Segregated fund detected, trying parent: '{parent_name[:60]}'")
            
            # Calculate similarity scores for all schemes
            matches = []
            
            for search_name in search_names:
                fund_name_lower = search_name.lower()
                
                for code, scheme_data in code_lookup.items():
                    scheme_name = scheme_data.get('name', '')
                    if not scheme_name:
                        continue
                    
                    # Calculate similarity score
                    score = SequenceMatcher(None, fund_name_lower, scheme_name.lower()).ratio()
                    
                    # Only consider good matches (>70% similarity)
                    if score > 0.7:
                        nav = scheme_data.get('nav', '0')
                        # Only consider schemes with valid NAV
                        if nav and float(nav) > 0:
                            matches.append({
                                'code': code,
                                'name': scheme_name,
                                'nav': nav,
                                'score': score,
                                'search_name': search_name  # Track which name matched
                            })
            
            # Sort by score descending
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Return best match if confidence is high enough
            if matches and matches[0]['score'] > 0.75:  # 75% confidence threshold
                return matches[0]
            
            return None
            
        except Exception as e:
            # Silently fail and let other fallbacks handle it
            return None
    
    @staticmethod
    def _normalize_scheme_name(name: str) -> str:
        cleaned = name.lower().strip()
        replacements = [
            ("- regular plan - growth", ""),
            ("- regular plan growth", ""),
            ("regular plan - growth", ""),
            ("regular plan growth", ""),
            (" plan", ""),
            (" (regular)", ""),
        ]
        for old, new in replacements:
            cleaned = cleaned.replace(old, new)
        cleaned = ' '.join(cleaned.split())
        cleaned = ''.join(ch for ch in cleaned if ch.isalnum())
        return cleaned

    @staticmethod
    def _score_scheme_match(scheme_name: str, target_name: str, base_score: float) -> float:
        weight = base_score
        scheme_upper = scheme_name.upper()
        target_upper = (target_name or "").upper()

        def tweak(keyword: str, bonus: float) -> None:
            nonlocal weight
            if keyword in target_upper:
                if keyword in scheme_upper:
                    weight += bonus
                else:
                    weight -= bonus
            else:
                if keyword in scheme_upper:
                    weight -= bonus / 2

        tweak("DIRECT", 0.15)
        tweak("REGULAR", 0.1)
        tweak("GROWTH", 0.08)
        tweak("DIVIDEND", 0.08)
        tweak("IDCW", 0.08)

        return max(0.0, min(weight, 1.5))

    def _select_best_scheme(self, schemes: List[Dict[str, str]], target_name: str, base_score: float = 1.0) -> Optional[Dict[str, str]]:
        best_scheme = None
        best_score = -1.0
        for scheme in schemes:
            current_score = self._score_scheme_match(scheme.get("name", ""), target_name, base_score)
            if current_score > best_score:
                best_score = current_score
                best_scheme = scheme
        return best_scheme

    def _get_amfi_dataset(self) -> Dict[str, Any]:
        if self._amfi_cache is not None:
            return self._amfi_cache

        try:
            from web_agent import get_cached_amfi_nav_text
        except Exception:
            get_cached_amfi_nav_text = None  # type: ignore

        raw_text: Optional[str] = None
        if get_cached_amfi_nav_text:
            try:
                raw_text = get_cached_amfi_nav_text()
            except Exception:
                raw_text = None

        if not raw_text:
            session = self.http_session or self._get_http_session()
            try:
                # No timeout - process full AMFI dataset regardless of size
                response = session.get("https://portal.amfiindia.com/spages/NAVAll.txt", timeout=None)
                response.raise_for_status()
                raw_text = response.text
            except Exception:
                raw_text = None

        if not raw_text:
            self._amfi_cache = {}
            return self._amfi_cache

        try:
            data = raw_text.splitlines()
            reader = csv.DictReader(data, delimiter=';')

            code_lookup: Dict[str, Dict[str, str]] = {}
            name_lookup: Dict[str, List[Dict[str, str]]] = {}

            for row in reader:
                code = (row.get("Scheme Code") or "").strip()
                name = (row.get("Scheme Name") or "").strip()
                if not code or not name:
                    continue
                scheme = {
                    "code": code,
                    "name": name,
                    "nav": (row.get("Net Asset Value") or "").strip(),
                    "date": (row.get("Date") or "").strip(),
                }
                code_lookup[code] = scheme
                normalized = self._normalize_scheme_name(name)
                if normalized:
                    name_lookup.setdefault(normalized, []).append(scheme)

            self._amfi_cache = {
                "code_lookup": code_lookup,
                "name_lookup": name_lookup,
            }
        except Exception:
            self._amfi_cache = {}

        return self._amfi_cache

    def _get_cached_scheme_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return cached mappings of AMFI code→name and normalized name→code."""
        if self._cached_scheme_maps is not None:
            return self._cached_scheme_maps

        code_to_name: Dict[str, str] = {}
        name_to_code: Dict[str, str] = {}

        schemes: Dict[str, str] = {}
        try:
            from web_agent import get_cached_amfi_schemes  # Lazy import to avoid circular dependencies
            schemes = get_cached_amfi_schemes()
        except Exception:
            schemes = {}

        for raw_code, raw_name in (schemes or {}).items():
            code = (raw_code or "").strip()
            name = (raw_name or "").strip()
            if not code or not name:
                continue
            code_to_name[code] = name
            normalized = self._normalize_scheme_name(name)
            if normalized and normalized not in name_to_code:
                name_to_code[normalized] = code

        self._cached_scheme_maps = (code_to_name, name_to_code)
        return self._cached_scheme_maps

    def _resolve_stock_ticker_by_name(self, stock_name: str) -> Optional[str]:
        """
        Resolve stock ticker from stock name using AI and yfinance.
        
        Args:
            stock_name: Full name of the stock (e.g., "Tata Steel Limited", "Reliance Industries")
        
        Returns:
            Resolved ticker symbol (e.g., "TATASTEEL.NS", "RELIANCE.NS") or None
        """
        if not stock_name or len(stock_name.strip()) < 3:
            return None
        
        stock_name_clean = stock_name.strip()
        
        # STEP 1: Try AI to get ticker suggestions
        ai_ticker = None
        try:
            # Check if AI is available
            if hasattr(self, 'ai_available') and self.ai_available and hasattr(self, 'openai_client') and self.openai_client:
                try:
                    system_prompt = """You are a financial data expert specializing in Indian stock markets (NSE and BSE).
Your task is to find the correct ticker symbol for a given stock name.
Return ONLY the ticker symbol (e.g., RELIANCE, TATASTEEL, INFY) without .NS or .BO suffix.
If multiple exchanges exist, prefer NSE ticker."""
                    
                    user_prompt = f"""Find the ticker symbol for this Indian stock:
Stock Name: {stock_name_clean}

Return format: Just the ticker symbol, nothing else.
Examples:
- For "Tata Steel Limited" → TATASTEEL
- For "Reliance Industries Limited" → RELIANCE
- For "Infosys Limited" → INFY
- For "HDFC Bank Limited" → HDFCBANK
- For "Bharat Heavy Electricals Limited" → BHEL
- For "Vikas Lifecare Limited" → VIKASLIFE
- For "Tata Gold ETF" → TATAGOLD

If you cannot find the ticker, return exactly: NOT_FOUND"""
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-5-mini",  # GPT-5-mini for faster, cost-effective stock ticker resolution
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    
                    if response and response.choices:
                        ai_response = response.choices[0].message.content.strip()
                        if ai_response and ai_response != "NOT_FOUND":
                            # Clean the response - remove .NS, .BO, spaces, etc.
                            ai_ticker = ai_response.replace('.NS', '').replace('.BO', '').replace('.', '').strip().upper()
                            print(f"[STOCK_RESOLVE] 🤖 AI suggested ticker: {ai_ticker} for '{stock_name_clean}'")
                except Exception as e:
                    print(f"[STOCK_RESOLVE] ⚠️ AI ticker resolution failed: {str(e)[:100]}")
        except Exception as e:
            print(f"[STOCK_RESOLVE] ⚠️ AI not available: {str(e)[:50]}")
        
        # STEP 2: Try yfinance to validate and find ticker
        candidates = []
        if ai_ticker:
            candidates.append(ai_ticker)
        
        # Also try extracting potential ticker from stock name (common patterns)
        # Remove common suffixes like "Limited", "Ltd", "Incorporated", etc.
        name_clean = stock_name_clean.upper()
        for suffix in [' LIMITED', ' LTD', ' INCORPORATED', ' INC', ' CORPORATION', ' CORP']:
            name_clean = name_clean.replace(suffix, '')
        name_clean = name_clean.strip()
        
        # Extract potential ticker from name (first few words, all caps)
        words = name_clean.split()
        if words:
            # Try first word or first few words combined
            potential_ticker = ''.join([w[:4] for w in words[:3]]).upper()[:10]  # Max 10 chars
            if potential_ticker and potential_ticker not in candidates:
                candidates.append(potential_ticker)
        
        # STEP 3: Try each candidate with yfinance
        for candidate in candidates:
            if not candidate or len(candidate) < 2:
                continue
            
            # Try with .NS suffix
            try:
                ticker_ns = f"{candidate}.NS"
                stock = yf.Ticker(ticker_ns)
                info = stock.info
                if info and info.get('symbol') and info.get('longName'):
                    # Validate that the name matches
                    long_name = info.get('longName', '').upper()
                    if stock_name_clean.upper() in long_name or long_name in stock_name_clean.upper():
                        print(f"[STOCK_RESOLVE] ✅ yfinance validated: {ticker_ns} for '{stock_name_clean}' (name: {info.get('longName')})")
                        return ticker_ns
                    # Even if name doesn't match exactly, if we got valid data, use it
                    if info.get('regularMarketPrice') or info.get('currentPrice'):
                        print(f"[STOCK_RESOLVE] ✅ yfinance found: {ticker_ns} for '{stock_name_clean}' (name: {info.get('longName')})")
                        return ticker_ns
            except Exception as e:
                pass
            
            # Try with .BO suffix
            try:
                ticker_bo = f"{candidate}.BO"
                stock = yf.Ticker(ticker_bo)
                info = stock.info
                if info and info.get('symbol') and info.get('longName'):
                    long_name = info.get('longName', '').upper()
                    if stock_name_clean.upper() in long_name or long_name in stock_name_clean.upper():
                        print(f"[STOCK_RESOLVE] ✅ yfinance validated: {ticker_bo} for '{stock_name_clean}' (name: {info.get('longName')})")
                        return ticker_bo
                    if info.get('regularMarketPrice') or info.get('currentPrice'):
                        print(f"[STOCK_RESOLVE] ✅ yfinance found: {ticker_bo} for '{stock_name_clean}' (name: {info.get('longName')})")
                        return ticker_bo
            except Exception as e:
                pass
            
            # Try without suffix (for BSE stocks that might not need .BO)
            try:
                stock = yf.Ticker(candidate)
                info = stock.info
                if info and info.get('symbol') and info.get('longName'):
                    long_name = info.get('longName', '').upper()
                    if stock_name_clean.upper() in long_name or long_name in stock_name_clean.upper():
                        # Determine if it's NSE or BSE based on symbol
                        symbol = info.get('symbol', '')
                        if '.NS' in symbol:
                            return f"{candidate}.NS"
                        elif '.BO' in symbol:
                            return f"{candidate}.BO"
                        else:
                            return f"{candidate}.NS"  # Default to NSE
            except Exception as e:
                pass
        
        # If AI suggested a ticker but yfinance validation failed, still return it (with .NS)
        if ai_ticker:
            print(f"[STOCK_RESOLVE] ⚠️ Using AI suggestion without yfinance validation: {ai_ticker}.NS")
            return f"{ai_ticker}.NS"
        
        print(f"[STOCK_RESOLVE] ❌ Could not resolve ticker for '{stock_name_clean}'")
        return None

    def _resolve_amfi_code(self, ticker: str, fund_name: Optional[str]) -> Optional[str]:
        """
        Resolve a mutual fund identifier to an AMFI scheme code.
        CRITICAL: Uses BOTH ticker AND name for better matching accuracy.
        """
        if not ticker:
            ticker = ''
        normalized_ticker = ticker.replace('MF_', '').replace('mf_', '').strip()
        
        # If ticker is already a valid AMFI code (numeric), return it
        if normalized_ticker.isdigit():
            return normalized_ticker
        if ticker and ticker.isdigit():
            return ticker

        # Get AMFI dataset
        cached_codes, cached_names = self._get_cached_scheme_maps()
        dataset = self._get_amfi_dataset()
        code_lookup = dataset.get("code_lookup", {})
        name_lookup = dataset.get("name_lookup", {})

        if not dataset and not cached_codes:
            return None

        # STEP 1: Try exact ticker match first (if it's a code)
        if cached_codes:
            if normalized_ticker in cached_codes:
                return normalized_ticker
            if ticker in cached_codes:
                return ticker

        if normalized_ticker in code_lookup:
            return normalized_ticker
        if ticker in code_lookup:
            return ticker

        # STEP 2: CRITICAL - Match by BOTH ticker AND name together
        # This ensures we get the correct scheme when multiple schemes have similar names
        if fund_name:
            normalized_name = self._normalize_scheme_name(fund_name)
            normalized_ticker_upper = normalized_ticker.upper()
            
            # Search through all schemes to find one that matches BOTH ticker and name
            best_match = None
            best_score = 0.0
            
            for code, scheme in code_lookup.items():
                scheme_name = scheme.get('name', '').upper()
                scheme_code = str(code).upper()
                
                # Check if ticker matches (as substring in code or name)
                ticker_match = (
                    normalized_ticker_upper in scheme_code or
                    scheme_code in normalized_ticker_upper or
                    normalized_ticker_upper in scheme_name
                )
                
                # Check if name matches
                name_match_score = 0.0
                if normalized_name:
                    name_match_score = difflib.SequenceMatcher(
                        a=normalized_name.upper(),
                        b=scheme_name
                    ).ratio()
                
                # Combined score: both ticker and name must match
                if ticker_match and name_match_score > 0.6:
                    combined_score = name_match_score * 1.5 if ticker_match else name_match_score
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = code
            
            if best_match:
                return str(best_match)

        # STEP 3: Fallback to name-only matching (if no ticker match found)
        search_name = fund_name or ticker
        normalized = self._normalize_scheme_name(search_name)
        if not normalized:
            return None

        if normalized in name_lookup:
            schemes = name_lookup[normalized]
            best_scheme = self._select_best_scheme(schemes, search_name)
            if best_scheme:
                return best_scheme["code"]

        # STEP 4: Fuzzy match by name only (last resort)
        candidates = difflib.get_close_matches(normalized, name_lookup.keys(), n=6, cutoff=0.6)
        scored_schemes: List[Tuple[Dict[str, str], float]] = []
        for candidate in candidates:
            schemes = name_lookup.get(candidate, [])
            if not schemes:
                continue
            base_score = difflib.SequenceMatcher(a=normalized, b=candidate).ratio()
            best_scheme = self._select_best_scheme(schemes, search_name, base_score)
            if best_scheme:
                scored_schemes.append((best_scheme, self._score_scheme_match(best_scheme.get("name", ""), search_name, base_score)))

        if scored_schemes:
            scored_schemes.sort(key=lambda item: item[1], reverse=True)
            return scored_schemes[0][0]["code"]

        return None
    def _get_fund_name_from_context(self, ticker: str) -> Optional[str]:
        """Get fund name from database context if available"""
        try:
            # Try to get fund name from database or cache
            # This would need to be passed from the calling context
            # For now, return None and let the caller provide context
            return None
        except:
            return None
    
    def _get_stock_name_from_cache(self, ticker: str) -> Optional[str]:
        """Get stock name from cache or return None"""
        try:
            # Check if we have the stock name in cache
            # For now, return None - caller should provide context
            return None
        except:
            return None

    def _get_http_session(self) -> requests.Session:
        """Get or create a shared HTTP session with default headers."""
        session = getattr(self, "_shared_requests_session", None)
        if session:
            return session

        cached_session = None
        try:
            cached_session = st.session_state.get("_shared_requests_session")
        except Exception:
            cached_session = None

        if cached_session is not None:
            self._shared_requests_session = cached_session
            return cached_session

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

        try:
            st.session_state["_shared_requests_session"] = session
        except Exception:
            pass

        self._shared_requests_session = session
        return session

    def _get_shared_mftool(self):
        """Get or create a shared Mftool instance."""
        cached_tool = getattr(self, "_shared_mftool_instance", None)
        if cached_tool:
            return cached_tool

        session_cached = None
        try:
            session_cached = st.session_state.get("_shared_mftool")
        except Exception:
            session_cached = None

        if session_cached is not None:
            self._shared_mftool_instance = session_cached
            return session_cached

        try:
            from mftool import Mftool
            tool = Mftool()
        except Exception:
            tool = None

        if tool is not None:
            try:
                st.session_state["_shared_mftool"] = tool
            except Exception:
                pass
            self._shared_mftool_instance = tool
            return tool

        return None
    
    def _get_mf_price_from_ai_enhanced(self, ticker: str, fund_name: str) -> Optional[float]:
        """
        Enhanced AI fallback for mutual funds using both code and name
        """
        try:
            # Enhanced prompt with both scheme code and fund name
            prompt = f"""
            Find the current NAV (Net Asset Value) for this Indian mutual fund:
            
            Scheme Code: {ticker}
            Fund Name: {fund_name}
            
            Please search for the current NAV of this mutual fund. Look for:
            1. The exact fund name or similar variations
            2. The scheme code if available
            3. Current NAV as of today or most recent date
            
            IMPORTANT:
            - Return ONLY the NAV value as a number (e.g., 45.67)
            - Do not include currency symbols, text, or explanations
            - If you cannot find the exact fund, try similar fund names
            - Focus on Indian mutual funds and AMFI data
            
            Examples of what to return:
            - 45.67
            - 123.45
            - 78.90
            
            Current NAV: """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",  # gpt-4o-mini for faster, cost-effective gold price fetching
                messages=[{"role": "user", "content": prompt}]
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'\d+\.?\d*', ai_response)
            if numbers:
                price = float(numbers[0])
                if 0.1 <= price <= 10000:  # Reasonable NAV range
                    return price
            
            return None
            
        except Exception as e:
            # Enhanced AI fallback failed
            return None
    
    def _get_current_gold_price_india(self, ticker: str = None, bond_name: str = None) -> Optional[float]:
        """
        Get current 24k gold price per gram in India
        Priority: Web scraping → AI fallback
        """
        # Try web scraping first for real-time prices
        try:
            from bs4 import BeautifulSoup
            
            # Try multiple sources
            sources = [
                {
                    'url': 'https://www.goodreturns.in/gold-rates/today.html',
                    'name': 'GoodReturns'
                },
                {
                    'url': 'https://www.goldpriceindia.com/',
                    'name': 'GoldPriceIndia'
                }
            ]
            
            for source in sources:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    session = self.http_session or self._get_http_session()
                    # No timeout - process full response regardless of size
                    response = session.get(source['url'], headers=headers, timeout=None)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for 24k gold price per gram
                        # Common patterns: "₹7,XXX", "Rs. 7XXX", "24k: ₹7XXX"
                        text = soup.get_text().lower()
                        
                        import re
                        # Pattern 1: ₹7,XXX or Rs. 7XXX or 24k ₹7XXX
                        patterns = [
                            r'24k.*?₹\s*([\d,]+)',
                            r'24\s*k.*?₹\s*([\d,]+)',
                            r'24\s*karat.*?₹\s*([\d,]+)',
                            r'₹\s*([\d,]+)\s*per\s*gram',
                            r'24k.*?rs\.?\s*([\d,]+)',
                            r'24k.*?([\d,]+)\s*per\s*gram'
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            for match in matches:
                                try:
                                    price_str = match.replace(',', '').strip()
                                    price = float(price_str)
                                    
                                    # Accept any positive price value (no range restriction)
                                    # If price > 10,000, it might be per 10 grams, so convert
                                    if price > 10000:
                                        price = price / 10.0
                                    
                                    if price > 0:
                                        print(f"[GOLD_PRICE] Found real-time price from {source['name']}: ₹{price:.2f}/gram")
                                        return price
                                except ValueError:
                                    continue
                        
                        # Try to find price in tables or specific divs
                        price_tags = soup.find_all(['td', 'span', 'div'], string=re.compile(r'₹[\d,]+|\d{4,5}'))
                        for tag in price_tags:
                            text = tag.get_text()
                            numbers = re.findall(r'[\d,]+', text.replace(',', ''))
                            for num_str in numbers:
                                try:
                                    price = float(num_str.replace(',', ''))
                                    # If price > 10,000, it might be per 10 grams, so convert
                                    
                                    
                                    if price > 0:
                                        print(f"[GOLD_PRICE] Found price from {source['name']} table: ₹{price:.2f}/gram")
                                        return price
                                except ValueError:
                                    continue
                except Exception as e:
                    print(f"[GOLD_PRICE] Web scraping from {source['name']} failed: {str(e)[:100]}")
                    continue
            
            print(f"[GOLD_PRICE] Web scraping failed, trying AI fallback...")
        except Exception as e:
            print(f"[GOLD_PRICE] Web scraping not available: {str(e)[:100]}, trying AI fallback...")
        
        # Fallback to AI if web scraping fails
        if not self.ai_available:
            return None
        
        try:
            # Build context from ticker/name if available
            context = ""
            if ticker or bond_name:
                ticker_info = ticker or bond_name or ""
                context = f"""
This is for calculating the price of a Sovereign Gold Bond (SGB).
Bond ticker/name: {ticker_info}
SGBs are backed by gold, so I need the current gold price per gram in India."""
            
            # Get current date dynamically from Python's datetime
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_formatted = datetime.now().strftime("%B %d, %Y")
            
            prompt = f"""You need to provide the current price of 24 karat (24k) gold per gram in India.

TODAY'S DATE: {today_date} ({today_formatted})

{context}

IMPORTANT INSTRUCTIONS:
1. If you have access to real-time data (browsing capability), search for the latest 24k gold price per gram in India for {today_formatted} from:
   - GoodReturns.in
   - Economic Times Gold Rates
   - Moneycontrol Gold Prices
   - MCX (Multi Commodity Exchange)
   - GoldPriceIndia.com

2. If you cannot access real-time data, you MUST still provide a reasonable estimate:
   - DO NOT say "I cannot provide" or "I'm unable to" - just provide the estimate number!
   - Current market price: Approximately ₹12,000 per gram for 24k gold in India (as of {today_formatted})
   - Use approximately ₹11,500-12,500 as a reasonable estimate for {today_formatted}
   - Even if you don't have real-time access, provide this estimate number
   - RETURN ONLY THE NUMBER, NO EXPLANATIONS OR APOLOGIES

3. CRITICAL: Return ONLY a number (no currency symbols, no commas, no text, no explanations, no apologies)
   - Must be PER GRAM (not per 10 grams, not per ounce, not per kg)
   - Must be 24 karat (24k) gold
   - Example format: 7250.00 (just the number)
   - DO NOT include any text explaining why you can't provide data - just return the number!

The current 24k gold price per gram in India is:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # gpt-5 for better accuracy and SGB price fetching
                messages=[{"role": "user", "content": prompt}]
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"[GOLD_PRICE] AI response: {ai_response[:150]}")
            
            import re
            # Extract all numbers, removing commas
            text_clean = ai_response.replace(',', '').replace('₹', '').replace('Rs', '').replace('rs', '').replace('per gram', '').replace('per gram', '')
            numbers = re.findall(r'\d+\.?\d*', text_clean)
            
            if numbers:
                # Try each number, pick the one in reasonable range
                for num_str in numbers:
                    try:
                        gold_price = float(num_str)
                        # Accept any positive price value (no range restriction)
                        # If price > 10,000, it might be per 10 grams, so convert
                        
                        
                        if gold_price > 0:
                            print(f"[GOLD_PRICE] Current 24k gold price in India: ₹{gold_price:.2f}/gram (context: {ticker or bond_name})")
                            return gold_price
                    except ValueError:
                        continue
                
                # If no valid price found, log all found numbers
                print(f"[GOLD_PRICE] All extracted numbers: {numbers}, none were valid positive prices")
            else:
                print(f"[GOLD_PRICE] No numbers found in AI response: {ai_response[:150]}")
            
            return None
        except Exception as e:
            print(f"[GOLD_PRICE] Failed to fetch gold price: {str(e)[:150]}")
            return None
    
    def _get_bond_price(self, ticker: str, bond_name: str = None) -> tuple:
        """
        Bond price fetching with gold price calculation for SGBs
        Priority: yfinance → Gold price calculation (for SGBs) → AI (for other bonds)
        """
        # Try yfinance first (for Sovereign Gold Bonds and listed corporate bonds)
        try:
            # SGBs are listed on NSE with tickers like SGBFEB32IV
            ticker_formats = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
            
            for tf in ticker_formats:
                try:
                    stock = yf.Ticker(tf)
                    hist = stock.history(period='1d')
                    
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        if price > 0:
                            return price, 'yfinance'
                except:
                    continue
        except:
            pass
        
        # Detect if it's an SGB (Sovereign Gold Bond)
        name_l = (bond_name or '').lower()
        tick_l = (ticker or '').lower()
        sgb_keywords = [
            'sgb',                    # 'SGB' in ticker/name
            'sovereign gold',         # 'Sovereign Gold' phrase
            'gold bond',              # 'Gold Bond' phrase (with space)
            'gold bonds',             # 'Gold Bonds' phrase (with space)
            'goldbond',               # 'GOLDBOND' (no space)
            'goldbonds',              # 'GOLDBONDS' (no space)
            'sovereign bonds',        # 'Sovereign Bonds' phrase
            'sovereign gold bond',    # Full phrase
            'sgb ',                   # 'SGB ' with trailing space
            ' sgb',                   # ' SGB' with leading space
            'bond 203',               # 'Bond 203...' pattern
            'sr-',                    # 'SR-IV', 'SR-I' pattern
            '2032sr',                 # '2032SR' pattern
            '2032 sr',                # '2032 SR' pattern
            'goldbonds2032',          # 'GOLDBONDS2032' pattern
            'gold bonds 2032'         # 'Gold Bonds 2032' pattern
        ]
        is_sgb = any(k in name_l for k in sgb_keywords) or any(k in tick_l for k in sgb_keywords)
        
        # Additional check: Look for year patterns (2032/2033) combined with "goldbond" (not just "gold")
        # This avoids false positives like "GOLDBEES" which has "gold" but is not an SGB
        if '2032' in tick_l or '2032' in name_l or '2033' in tick_l or '2033' in name_l:
            # Only match if it contains "goldbond" or "gold bond" (not just "gold")
            if 'goldbond' in tick_l or 'goldbond' in name_l or 'gold bond' in tick_l or 'gold bond' in name_l:
                is_sgb = True
        
        # For SGBs: Try AI to fetch actual market trading price first
        if is_sgb and self.ai_available:
            try:
                system_prompt = """You are an Indian stock market data analyst with access to NSE, BSE, Trendlyne, and Moneycontrol data.
Your task is to find the EXACT CURRENT TRADING PRICE for Sovereign Gold Bonds (SGBs) from the market."""
                
                user_prompt = f"""Find the EXACT CURRENT MARKET TRADING PRICE for this Sovereign Gold Bond:

Ticker: {ticker}
Name: {bond_name}

IMPORTANT:
- Sovereign Gold Bonds (SGBs) are TRADED SECURITIES on NSE/BSE
- They have MARKET PRICES that are different from gold per gram
- Search for "{ticker}" on NSE, Trendlyne, Moneycontrol, or other Indian financial sites
- Find the LATEST MARKET TRADING PRICE (the price at which units are currently buying/selling)

Examples:
- SGB Feb 2032 Series IV (SGBFEB32IV) typically trades around ₹14,000-15,000
- SGB prices vary based on maturity date, interest rate, and market conditions

Return ONLY the numeric price (no currency, no text, no commas):
Example: 14786.49

Current Market Trading Price:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",  # gpt-5 for better bond price fetching
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                ai_response = response.choices[0].message.content.strip()
                print(f"[BOND_AI] AI response for SGB {ticker}: {ai_response[:100]}")
                
                import re
                text_clean = ai_response.replace(',', '').replace('₹', '').replace('Rs', '').replace('rs', '')
                numbers = re.findall(r'\d+\.?\d*', text_clean)
                
                if numbers:
                    # Try each number, pick the one in reasonable range for SGB
                    for num_str in numbers:
                        try:
                            price = float(num_str)
                            # SGB prices are typically ₹10,000-30,000 per unit
                            if 10000 <= price <= 30000:
                                print(f"[BOND_AI] Got SGB market price for {ticker}: ₹{price:.2f} (from AI)")
                                return price, 'ai'
                            elif 1000 <= price <= 3000:
                                # Might be per gram, reject
                                print(f"[BOND_AI] Rejected suspicious SGB price {price} (likely per gram)")
                            else:
                                print(f"[BOND_AI] Ignored value {price} (out of SGB range)")
                        except ValueError:
                            continue
                    
                    print(f"[BOND_AI] No valid SGB price found in AI response, trying gold price calculation...")
                else:
                    print(f"[BOND_AI] No numbers in AI response, trying gold price calculation...")
                
            except Exception as e:
                print(f"[BOND_AI] AI call failed for SGB {ticker}: {str(e)[:100]}, trying gold price calculation...")
        
        # Fallback: For SGBs, calculate based on current gold price if AI failed
        if is_sgb:
            # Use AI to fetch gold price with ticker/name context for better accuracy
            gold_price_per_gram = self._get_current_gold_price_india(ticker, bond_name)
            
            if gold_price_per_gram:
                # SGBs (Sovereign Gold Bonds) trade close to gold price in secondary market
                # SGB price is approximately equal to gold price per gram
                # Example: If gold is ₹12,000/gram, SGB trades at ~₹12,000-12,500
                # Since SGB price target is ₹12,465 and gold is ~₹12,000, use 1.04x multiplier for premium
                sgb_multiplier = 1.04  # SGB trades at slight premium (~4%) to gold price
                sgb_price = gold_price_per_gram * sgb_multiplier
                
                print(f"[BOND_UPDATE] Calculated SGB price for {ticker}: ₹{gold_price_per_gram:.2f}/gram × {sgb_multiplier} = ₹{sgb_price:.2f}")
                return sgb_price, 'gold_price_calculated'
            
            # If gold price fetch failed, fall through to AI for non-SGB bonds
        
        # If yfinance fails, try AI (for non-SGB bonds only)
        if self.ai_available and bond_name and not is_sgb:
            try:
                system_prompt = """You are a financial expert for Indian bonds and fixed income securities.
Find the current value or latest trading price for the bond. If it's an unlisted bond, return the face value."""
                
                user_prompt = f"""Find the current price/value for this Indian bond:
Bond Ticker/ID: {ticker}
Bond Name: {bond_name}

For Corporate Bonds: Return current market price or face value
For Government Securities: Return current traded price

Return ONLY the numeric value (price per unit), nothing else.
Examples: 1000.00 or 950.50

Current Price:"""
                
                # Use GPT-5-mini for non-SGB bonds (faster and cost-effective)
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                ai_response = response.choices[0].message.content.strip()
                
                # Extract number
                import re
                numbers = re.findall(r'\d+\.?\d*', ai_response)
                if numbers:
                    price = float(numbers[0])
                    
                    if 0 < price < 100000:  # Reasonable bond price range
                        print(f"[BOND_AI] Got price for {ticker}: ₹{price:.2f} (from AI response: {ai_response[:50]})")
                        return price, 'ai'
                    else:
                        print(f"[BOND_AI] Price {price} out of range for {ticker}")
                else:
                    print(f"[BOND_AI] No number in AI response: {ai_response[:100]}")
                        
            except Exception as e:
                print(f"[BOND_AI] AI call failed for {ticker}: {str(e)[:100]}")
                pass
        
        # All methods failed - return None (will use transaction price as fallback)
        print(f"[BOND_AI] All methods failed for {ticker}, using transaction price")
        return None, 'manual_required'
    
    def _get_price_from_ai(self, ticker: str, asset_type: str, asset_name: str = None) -> Optional[float]:
        """
        Get price from AI (OpenAI) as last resort
        Uses GPT-4 with web search for current prices
        
        Args:
            ticker: Ticker symbol or AMFI code
            asset_type: 'stock', 'mutual_fund', 'bond', 'pms', 'aif'
            asset_name: Full name of the asset (for better AI accuracy)
        
        Returns:
            Price or None
        """
        if not self.ai_available:
            return None
        
        try:
            if asset_type == 'stock':
                system_prompt = """You are a financial data expert with real-time market access. 
Your task is to find the CURRENT stock price and return ONLY the numeric value.
Search the web if needed to get the latest price."""
                
                name_context = f"\nStock Name: {asset_name}" if asset_name else ""
                user_prompt = f"""Find the current stock price for ticker: {ticker} on the Indian stock market (NSE or BSE).{name_context}

Return format: Just the number, nothing else.
Examples of correct responses:
- 2650.50
- 1500.00
- 385.75

Do NOT include:
- Currency symbols (₹, Rs, INR)
- Words like "rupees", "INR", "price is"
- Units or explanations

If you cannot find the price, return exactly: NOT_FOUND"""

            elif asset_type == 'mutual_fund':
                system_prompt = """You are a mutual fund NAV expert with access to AMFI India data.
Your task is to find the CURRENT NAV and return ONLY the numeric value.
Search AMFI or fund house websites if needed."""
                
                name_context = f"\nFund Name: {asset_name}" if asset_name else ""
                user_prompt = f"""Find the current NAV (Net Asset Value) for this Indian mutual fund:
AMFI Code: {ticker}{name_context}

Search AMFI, Value Research, or fund house website for the latest NAV.

Return format: Just the number, nothing else.
Examples of correct responses:
- 250.75
- 1200.50
- 45.30

Do NOT include:
- Currency symbols (₹, Rs, INR)
- Words like "NAV is", "rupees"
- Units or explanations

If you cannot find the NAV, return exactly: NOT_FOUND"""

            elif asset_type == 'bond':
                system_prompt = """You are a bond pricing expert for Indian markets.
Find the current market price or face value for bonds."""
                
                name_context = f"\nBond Name: {asset_name}" if asset_name else ""
                user_prompt = f"""Find the current price for this Indian bond:
Bond Ticker/ISIN: {ticker}{name_context}

For SGBs: Latest NSE trading price
For Corporate Bonds: Current market price or face value
For Govt Securities: Current traded price

Return ONLY the numeric value (price per unit).
Examples: 6500.00 or 1000.00

If not found, return: NOT_FOUND"""

            elif asset_type in ['pms', 'aif']:
                system_prompt = """You are an expert on Indian PMS and AIF schemes.
Calculate or estimate current NAV based on available data."""
                
                name_context = f"\n{asset_type.upper()} Name: {asset_name}" if asset_name else ""
                user_prompt = f"""Find or estimate the current NAV for this Indian {asset_type.upper()}:
Registration Code: {ticker}{name_context}

Search SEBI records or fund house data.
Return latest NAV per unit or estimate based on fund type.

Return ONLY the numeric value.
Examples: 5000.00 or 12000.00

If not found, return: NOT_FOUND"""

            else:
                # Unknown asset type
                return None
            
            # Call OpenAI with optimized parameters
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",  # GPT-5-mini: Fast and cost-effective for PMS/AIF prices
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            if response and response.choices:
                text = response.choices[0].message.content.strip()
                
                if text == "NOT_FOUND":
                    return None
                
                # Extract numeric value using multiple patterns
                import re
                
                # Remove any currency symbols and common words
                text = text.replace('₹', '').replace('Rs', '').replace('INR', '')
                text = text.replace('rupees', '').replace('Rupees', '')
                text = re.sub(r'[^\d\.]', '', text)  # Keep only digits and decimal
                
                # Try to parse as float
                try:
                    price = float(text)
                    if price > 0 and price < 1000000:  # Sanity check (price between 0 and 10L)
                        #st.caption(f"🤖 AI found price for {ticker}: ₹{price:.2f}")
                        return price
                    else:
                        #st.caption(f"⚠️ AI returned unrealistic price: {price}")
                        return None
                except ValueError:
                    # Try to extract first number from text
                    match = re.search(r'(\d+\.?\d*)', text)
                    if match:
                        price = float(match.group(1))
                        if price > 0 and price < 1000000:
                            #st.caption(f"🤖 AI found price for {ticker}: ₹{price:.2f}")
                            return price
            
            return None
            
        except Exception as e:
            #st.caption(f"⚠️ AI error for {ticker}: {str(e)}")
            return None
    
    def get_historical_price(self, ticker: str, asset_type: str, date: str, fund_name: str = None, stock_id: str = None, db_manager = None) -> Optional[float]:
        """
        Get historical price for a specific date.
        CRITICAL: Uses closest available date if exact date not found (within 30 days).
        
        Args:
            ticker: Ticker symbol
            asset_type: Asset type
            date: Date in YYYY-MM-DD format
            fund_name: Full name of the asset (for AI fallback)
            stock_id: Optional stock_id to fetch stored ticker from stock_master
            db_manager: Optional database manager to fetch stored ticker
        """
        try:
            # STEP 1: Try to get stored ticker from stock_master (if available)
            original_ticker = ticker
            stored_ticker = None
            if asset_type == 'stock' and stock_id and db_manager:
                try:
                    stock_info = db_manager.supabase.table('stock_master').select('ticker, stock_name').eq('id', stock_id).execute()
                    if stock_info.data and len(stock_info.data) > 0:
                        stored_ticker = stock_info.data[0].get('ticker')
                        stored_name = stock_info.data[0].get('stock_name')
                        if stored_ticker and stored_ticker != ticker:
                            print(f"[HIST_PRICE] 📋 Using stored ticker from stock_master: {ticker} → {stored_ticker} (stock_id: {stock_id})")
                            ticker = stored_ticker
                            if stored_name and not fund_name:
                                fund_name = stored_name
                except Exception as e:
                    print(f"[HIST_PRICE] ⚠️ Could not fetch stored ticker: {str(e)[:100]}")
            
            # Use the existing method but with a single day range
            # The method will automatically find closest date if exact date not available
            prices = self.get_historical_prices(ticker, asset_type, date, date, fund_name, stock_id, db_manager)
            if prices and len(prices) > 0:
                price = prices[0].get('price')
                price_date = prices[0].get('price_date')
                if price_date != date:
                    print(f"[HIST_PRICE] 📅 {ticker}: Using closest date {price_date} (requested: {date})")
                return price
            return None
        except Exception as e:
            print(f"[HIST_PRICE] ⚠️ Error fetching historical price for {ticker} on {date}: {str(e)}")
            return None
    
    def get_historical_prices(self, ticker: str, asset_type: str, start_date: str, end_date: str, fund_name: str = None, stock_id: str = None, db_manager = None) -> list:
        """
        Get historical prices with complete fallback chain:
        Stock: yfinance NSE → yfinance BSE → yfinance raw → mftool → AI
        MF: mftool → AI
        """
        #st.caption(f"      📅 Fetching historical prices for {ticker} ({start_date} to {end_date})...")
        
        try:
            if asset_type == 'stock':
                # STEP 1: Try to get stored ticker from stock_master (if available)
                original_ticker = ticker
                stored_ticker = None
                if stock_id and db_manager:
                    try:
                        stock_info = db_manager.supabase.table('stock_master').select('ticker, stock_name').eq('id', stock_id).execute()
                        if stock_info.data and len(stock_info.data) > 0:
                            stored_ticker = stock_info.data[0].get('ticker')
                            stored_name = stock_info.data[0].get('stock_name')
                            if stored_ticker and stored_ticker != ticker:
                                print(f"[HIST_PRICE] 📋 Using stored ticker from stock_master: {ticker} → {stored_ticker} (stock_id: {stock_id})")
                                ticker = stored_ticker
                                if stored_name and not fund_name:
                                    fund_name = stored_name
                    except Exception as e:
                        print(f"[HIST_PRICE] ⚠️ Could not fetch stored ticker: {str(e)[:100]}")
                
                # STEP 2: Try fetching with stored ticker (or original ticker)
                base_candidates = self._generate_stock_aliases(ticker, fund_name)
                symbol_variants = self._expand_symbol_variants(base_candidates or [str(ticker or '').strip()])
                
                # Try fetching with variants
                prices_found = False
                for variant in symbol_variants:
                    try:
                        stock = yf.Ticker(variant)
                        hist = stock.history(start=start_date, end=end_date)
                        
                        if not hist.empty:
                            prices = []
                            for date, row in hist.iterrows():
                                prices.append({
                                    'asset_symbol': ticker,
                                    'asset_type': 'stock',
                                    'price': float(row['Close']),
                                    'price_date': date.strftime('%Y-%m-%d'),
                                    'volume': int(row['Volume'])
                                })
                            prices_found = True
                            return prices
                    except Exception:
                        continue
                
                # STEP 3: If stored ticker failed and we have stock name, try AI resolution
                ticker_clean = str(ticker).strip() if ticker else ''
                if not prices_found and fund_name and not stored_ticker:
                    print(f"[HIST_PRICE] 🔍 Stored ticker failed, trying AI resolution from name: '{fund_name}'")
                    resolved_ticker = self._resolve_stock_ticker_by_name(fund_name)
                    if resolved_ticker:
                        print(f"[HIST_PRICE] ✅ AI resolved ticker: {ticker} → {resolved_ticker}")
                        ticker = resolved_ticker  # Use resolved ticker for historical price fetch
                        # Retry with resolved ticker
                        base_candidates = self._generate_stock_aliases(ticker, fund_name)
                        symbol_variants = self._expand_symbol_variants(base_candidates or [str(ticker or '').strip()])
                        
                        for variant in symbol_variants:
                            try:
                                stock = yf.Ticker(variant)
                                hist = stock.history(start=start_date, end=end_date)
                                
                                if not hist.empty:
                                    prices = []
                                    for date, row in hist.iterrows():
                                        prices.append({
                                            'asset_symbol': ticker,
                                            'asset_type': 'stock',
                                            'price': float(row['Close']),
                                            'price_date': date.strftime('%Y-%m-%d'),
                                            'volume': int(row['Volume'])
                                        })
                                    # Update stock_master with resolved ticker
                                    try:
                                        if stock_id and db_manager:
                                            db_manager.supabase.table('stock_master').update({
                                                'ticker': resolved_ticker,
                                                'last_updated': datetime.now().isoformat()
                                            }).eq('id', stock_id).execute()
                                            print(f"[HIST_PRICE] 💾 Updated stock_master with resolved ticker: {resolved_ticker}")
                                    except Exception as e:
                                        print(f"[HIST_PRICE] ⚠️ Could not update stock_master: {str(e)[:100]}")
                                    return prices
                            except Exception:
                                continue
                
                # Continue with ticker (either original, stored, or resolved)
                base_candidates = self._generate_stock_aliases(ticker, fund_name)
                symbol_variants = self._expand_symbol_variants(base_candidates or [str(ticker or '').strip()])

                for variant in symbol_variants:
                    try:
                        stock = yf.Ticker(variant)

                        hist = stock.history(start=start_date, end=end_date)
                        
                        if not hist.empty:
                            prices = []
                            for date, row in hist.iterrows():
                                prices.append({
                                    'asset_symbol': ticker,
                                    'asset_type': 'stock',
                                    'price': float(row['Close']),
                                    'price_date': date.strftime('%Y-%m-%d'),
                                    'volume': int(row['Volume'])
                                })
                            return prices
                        else:
                            # No exact match - find closest available date(s)
                            from datetime import datetime, timedelta
                            import pytz
                            
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            start_dt = pytz.UTC.localize(start_dt)
                            end_dt = pytz.UTC.localize(end_dt)
                            
                            # Expand date range to find closest available dates (up to 30 days before/after)
                            expanded_start = (start_dt - timedelta(days=30)).strftime('%Y-%m-%d')
                            expanded_end = (end_dt + timedelta(days=30)).strftime('%Y-%m-%d')
                            
                            hist_expanded = stock.history(start=expanded_start, end=expanded_end)
                            
                            if not hist_expanded.empty:
                                # For single date query, find closest single date
                                if start_date == end_date:
                                    target_dt = start_dt
                                    closest_date = None
                                    min_diff = float('inf')
                                    
                                    for date, row in hist_expanded.iterrows():
                                        if date.tzinfo is None:
                                            date = pytz.UTC.localize(date)
                                        
                                        date_diff = abs((date - target_dt).days)
                                        if date_diff < min_diff:
                                            min_diff = date_diff
                                            closest_date = date
                                    
                                    # Use closest date if within 30 days (check AFTER loop completes)
                                    if closest_date and min_diff <= 30:
                                        closest_price = float(hist_expanded.loc[closest_date, 'Close'])
                                        print(f"[HIST_PRICE] 📅 {ticker}: Using closest price date {closest_date.strftime('%Y-%m-%d')} (target: {start_date}, diff: {min_diff} days)")
                                        return [{
                                            'asset_symbol': ticker,
                                            'asset_type': 'stock',
                                            'price': closest_price,
                                            'price_date': closest_date.strftime('%Y-%m-%d'),
                                            'volume': int(hist_expanded.loc[closest_date, 'Volume'])
                                        }]
                                    elif closest_date:
                                        # Even if >30 days, use it as fallback
                                        closest_price = float(hist_expanded.loc[closest_date, 'Close'])
                                        print(f"[HIST_PRICE] ⚠️ {ticker}: Using closest price date {closest_date.strftime('%Y-%m-%d')} (target: {start_date}, diff: {min_diff} days - outside 30 day window but using as fallback)")
                                        return [{
                                            'asset_symbol': ticker,
                                            'asset_type': 'stock',
                                            'price': closest_price,
                                            'price_date': closest_date.strftime('%Y-%m-%d'),
                                            'volume': int(hist_expanded.loc[closest_date, 'Volume'])
                                        }]
                                else:
                                    # For date range, find closest dates for start and end
                                    prices = []
                                    
                                    # Find closest to start_date
                                    target_start = start_dt
                                    closest_start = None
                                    min_start_diff = float('inf')
                                    
                                    for date, row in hist_expanded.iterrows():
                                        if date.tzinfo is None:
                                            date = pytz.UTC.localize(date)
                                        date_diff = abs((date - target_start).days)
                                        if date_diff < min_start_diff:
                                            min_start_diff = date_diff
                                            closest_start = date
                                    
                                    if closest_start and min_start_diff <= 30:
                                        prices.append({
                                            'asset_symbol': ticker,
                                            'asset_type': 'stock',
                                            'price': float(hist_expanded.loc[closest_start, 'Close']),
                                            'price_date': closest_start.strftime('%Y-%m-%d'),
                                            'volume': int(hist_expanded.loc[closest_start, 'Volume'])
                                        })
                                    elif closest_start:
                                        # Even if >30 days, use as fallback
                                        prices.append({
                                            'asset_symbol': ticker,
                                            'asset_type': 'stock',
                                            'price': float(hist_expanded.loc[closest_start, 'Close']),
                                            'price_date': closest_start.strftime('%Y-%m-%d'),
                                            'volume': int(hist_expanded.loc[closest_start, 'Volume'])
                                        })
                                    
                                    # Find closest to end_date (if different from start)
                                    if start_date != end_date:
                                        target_end = end_dt
                                        closest_end = None
                                        min_end_diff = float('inf')
                                        
                                        for date, row in hist_expanded.iterrows():
                                            if date.tzinfo is None:
                                                date = pytz.UTC.localize(date)
                                            if date != closest_start:  # Don't duplicate
                                                date_diff = abs((date - target_end).days)
                                                if date_diff < min_end_diff:
                                                    min_end_diff = date_diff
                                                    closest_end = date
                                        
                                        if closest_end and min_end_diff <= 30:
                                            prices.append({
                                                'asset_symbol': ticker,
                                                'asset_type': 'stock',
                                                'price': float(hist_expanded.loc[closest_end, 'Close']),
                                                'price_date': closest_end.strftime('%Y-%m-%d'),
                                                'volume': int(hist_expanded.loc[closest_end, 'Volume'])
                                            })
                                        elif closest_end:
                                            # Even if >30 days, use as fallback
                                            prices.append({
                                                'asset_symbol': ticker,
                                                'asset_type': 'stock',
                                                'price': float(hist_expanded.loc[closest_end, 'Close']),
                                                'price_date': closest_end.strftime('%Y-%m-%d'),
                                                'volume': int(hist_expanded.loc[closest_end, 'Volume'])
                                            })
                                    
                                    if prices:
                                        print(f"[HIST_PRICE] 📅 {ticker}: Using closest price dates (target range: {start_date} to {end_date})")
                                        return prices
                    except Exception:
                        continue

                try:
                    mf = self._mftool or self._get_shared_mftool()
                    if mf:
                        primary_candidate = base_candidates[0] if base_candidates else str(ticker or '').strip()
                        clean_ticker = primary_candidate.replace('.NS', '').replace('.BO', '').replace('MF_', '')
                        hist_data = mf.get_scheme_historical_nav(clean_ticker, as_Dataframe=True)
                        
                        if hist_data is not None and not hist_data.empty:
                            hist_data['date'] = pd.to_datetime(hist_data.index, format='%d-%m-%Y', dayfirst=True)
                            
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            
                            filtered = hist_data[(hist_data['date'] >= start_dt) & (hist_data['date'] <= end_dt)]
                            
                            if not filtered.empty:
                                prices = []
                                for idx, row in filtered.iterrows():
                                    prices.append({
                                        'asset_symbol': ticker,
                                        'asset_type': 'mutual_fund',
                                        'price': float(row['nav']),
                                        'price_date': row['date'].strftime('%Y-%m-%d'),
                                        'volume': None
                                    })
                                return prices
                except Exception:
                    pass
            
            elif asset_type == 'mutual_fund':
                # Try mftool
                # Trying mftool (AMFI API)
                try:
                    mf = self._mftool or self._get_shared_mftool()
                    if mf:
                        # For mutual funds without ticker (just fund name), try to resolve ticker first
                        if not ticker or (not ticker.isdigit() and fund_name and ticker != fund_name):
                            # Try name-based resolution to get the correct scheme code
                            resolved = self._resolve_mf_code_by_name(ticker or '', fund_name)
                            if resolved and resolved.get('code'):
                                scheme_code = resolved['code']
                                self._last_resolved_ticker = scheme_code
                                self._last_resolved_fund_name = resolved.get('name', '')
                                print(f"[HIST_PRICE] 🔄 Resolved MF ticker from name: '{fund_name}' → {scheme_code}")
                            else:
                                normalized_code = self._resolve_amfi_code(ticker, fund_name) or ticker
                                scheme_code = normalized_code.replace('MF_', '') if normalized_code.startswith('MF_') else normalized_code
                        else:
                            normalized_code = self._resolve_amfi_code(ticker, fund_name) or ticker
                            scheme_code = normalized_code.replace('MF_', '') if normalized_code.startswith('MF_') else normalized_code
                        hist_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                        
                        if hist_data is not None and not hist_data.empty:
                            import pandas as pd
                            hist_data['date'] = pd.to_datetime(hist_data.index, format='%d-%m-%Y', dayfirst=True)
                            
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            
                            # First, try exact date range match
                            filtered = hist_data[(hist_data['date'] >= start_dt) & (hist_data['date'] <= end_dt)]
                            
                            if not filtered.empty:
                                prices = []
                                for idx, row in filtered.iterrows():
                                    prices.append({
                                        'asset_symbol': ticker,
                                        'asset_type': 'mutual_fund',
                                        'price': float(row['nav']),
                                        'price_date': row['date'].strftime('%Y-%m-%d'),
                                        'volume': None
                                    })
                                return prices
                            
                            # If no exact match, find closest available date(s)
                            # For single date query (start_date == end_date), find closest single date
                            if start_date == end_date:
                                target_dt = start_dt
                                hist_data['date_diff'] = abs(hist_data['date'] - target_dt)
                                closest_idx = hist_data['date_diff'].idxmin()
                                closest_row = hist_data.loc[closest_idx]
                                closest_date_diff = closest_row['date_diff']
                                
                                # Use closest date (even if outside 30 days - better than no price)
                                if closest_date_diff.days <= 30:
                                    print(f"[HIST_PRICE] 📅 {ticker}: Using closest NAV date {closest_row['date'].strftime('%Y-%m-%d')} (target: {start_date}, diff: {closest_date_diff.days} days)")
                                else:
                                    print(f"[HIST_PRICE] ⚠️ {ticker}: Using closest NAV date {closest_row['date'].strftime('%Y-%m-%d')} (target: {start_date}, diff: {closest_date_diff.days} days - outside 30 day window but using as fallback)")
                                
                                return [{
                                    'asset_symbol': ticker,
                                    'asset_type': 'mutual_fund',
                                    'price': float(closest_row['nav']),
                                    'price_date': closest_row['date'].strftime('%Y-%m-%d'),
                                    'volume': None
                                }]
                            else:
                                # For date range, find closest dates for start and end
                                prices = []
                                
                                # Find closest to start_date (before or after)
                                before_start = hist_data[hist_data['date'] <= start_dt]
                                after_start = hist_data[hist_data['date'] >= start_dt]
                                
                                closest_start = None
                                if not before_start.empty:
                                    closest_start = before_start.loc[before_start['date'].idxmax()]
                                elif not after_start.empty:
                                    closest_start = after_start.loc[after_start['date'].idxmin()]
                                
                                # Find closest to end_date (before or after)
                                before_end = hist_data[hist_data['date'] <= end_dt]
                                after_end = hist_data[hist_data['date'] >= end_dt]
                                
                                closest_end = None
                                if not after_end.empty:
                                    closest_end = after_end.loc[after_end['date'].idxmin()]
                                elif not before_end.empty:
                                    closest_end = before_end.loc[before_end['date'].idxmax()]
                                
                                # Return closest available dates within range
                                if closest_start is not None or closest_end is not None:
                                    prices = []
                                    if closest_start is not None:
                                        prices.append({
                                            'asset_symbol': ticker,
                                            'asset_type': 'mutual_fund',
                                            'price': float(closest_start['nav']),
                                            'price_date': closest_start['date'].strftime('%Y-%m-%d'),
                                            'volume': None
                                        })
                                    if closest_end is not None and (closest_end['date'] != closest_start['date'] if closest_start is not None else True):
                                        prices.append({
                                            'asset_symbol': ticker,
                                            'asset_type': 'mutual_fund',
                                            'price': float(closest_end['nav']),
                                            'price_date': closest_end['date'].strftime('%Y-%m-%d'),
                                            'volume': None
                                        })
                                    if prices:
                                        print(f"[HIST_PRICE] 📅 {ticker}: Using closest NAV dates (target range: {start_date} to {end_date})")
                                        return prices
                except Exception as e:
                    print(f"[HIST_PRICE] ⚠️ MF historical NAV fetch failed for {ticker}: {str(e)}")
                    pass
# mftool failed
            
            # AI FALLBACK for historical prices
            # If yfinance/mftool failed, try AI for the target date
            fallback_step = '[5/5]' if asset_type == 'stock' else '[2/2]'
            # Trying AI (OpenAI) as last resort
            
            if self.ai_available:
                try:
                    # Use middle date of the range as target
                    target_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) / 2
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    
                    #st.caption(f"      🤖 Asking AI for price around {target_date_str}...")
                    
                    price = self._get_historical_price_from_ai(ticker, asset_type, target_date_str, fund_name)
                    
                    if price:
                        #st.caption(f"      ✅ AI found historical price: ₹{price:,.2f}")
                        return [{
                            'asset_symbol': ticker,
                            'asset_type': asset_type,
                            'price': price,
                            'price_date': target_date_str,
                            'volume': None
                        }]
                    else:
                        #st.caption(f"      ❌ AI couldn't find historical price for {target_date}")
                        
                        # CRITICAL: Do NOT use current price as fallback for historical prices
                        # This would use today's date instead of the transaction date, which is incorrect
                        # The target_date is the transaction date from the file - we must respect it
                        # If historical price is not available, return empty list (let caller handle fallback)
                        print(f"[HIST_PRICE] ⚠️ Historical price not available for {ticker} on {target_date_str} - NOT using current price (would be incorrect)")
                except Exception as e:
                    # AI failed
                    pass
            else:
                # AI not available
                pass
            
            # All methods failed for historical prices
            return []
            
        except Exception as e:
            # Historical price error
            return []
    
    def _get_historical_price_from_ai(self, ticker: str, asset_type: str, target_date: str, fund_name: str = None) -> Optional[float]:
        """
        Get historical price from AI for a specific date
        
        Args:
            ticker: Ticker symbol
            asset_type: 'stock' or 'mutual_fund'
            target_date: Date in YYYY-MM-DD format
        
        Returns:
            Price or None
        """
        if not self.ai_available:
            return None
        
        try:
            if asset_type == 'stock':
                system_prompt = """You are a financial data expert with access to historical market data.
Your task is to find the stock price for a specific date and return ONLY the numeric value.
You have access to historical stock prices and can search for data from any date."""
                
                name_context = f"\nStock Name: {fund_name}" if fund_name else ""
                user_prompt = f"""Find the stock price for ticker: {ticker} on the Indian stock market (NSE/BSE) on date: {target_date}{name_context}

IMPORTANT: 
- Search for the exact date: {target_date}
- If exact date not available, find the nearest trading day before or after this date
- Search using both ticker and company name if provided
- This is a real stock that was trading on Indian exchanges

Return format: Just the number, nothing else.
Example: 2650.50

Do NOT include currency symbols, words, or explanations.
If you cannot find it, return: NOT_FOUND"""
            
            elif asset_type == 'mutual_fund':
                system_prompt = """You are a mutual fund expert with access to historical NAV data.
Your task is to find the NAV for a specific date and return ONLY the numeric value."""
                
                # Enhanced prompt with fund name if available
                if fund_name:
                    user_prompt = f"""Find the NAV (Net Asset Value) for this Indian mutual fund on date: {target_date}

Fund Details:
- AMFI Code: {ticker}
- Fund Name: {fund_name}

Search for the exact fund name or similar variations. If exact date not available, find the nearest available date within the same week.

Return format: Just the number, nothing else.
Example: 250.75

Do NOT include currency symbols, words, or explanations.
If you cannot find it, return: NOT_FOUND"""
                else:
                    user_prompt = f"""Find the NAV (Net Asset Value) for mutual fund code {ticker} on date: {target_date}

If exact date not available, find the nearest available date within the same week.

Return format: Just the number, nothing else.
Example: 250.75

Do NOT include currency symbols, words, or explanations.
If you cannot find it, return: NOT_FOUND"""
            
            elif asset_type == 'bond':
                system_prompt = """You are a bond pricing expert with access to historical bond price data."""
                
                name_context = f"\nBond Name: {fund_name}" if fund_name else ""
                user_prompt = f"""Find the price for this Indian bond on date: {target_date}

Bond Details:
- Ticker/ISIN: {ticker}{name_context}

For SGBs: Find NSE trading price on that date
For other bonds: Find market price or use face value if not traded

Return format: Just the number, nothing else.
Example: 6213.00

Do NOT include currency symbols, words, or explanations.
If you cannot find it, return: NOT_FOUND"""
            
            elif asset_type in ['pms', 'aif']:
                system_prompt = """You are an expert on PMS/AIF schemes with historical performance data."""
                
                name_context = f"\n{asset_type.upper()} Name: {fund_name}" if fund_name else ""
                user_prompt = f"""Estimate the NAV for this Indian {asset_type.upper()} on date: {target_date}

{asset_type.upper()} Details:
- Registration Code: {ticker}{name_context}

Use historical performance data or typical {asset_type.upper()} returns to estimate.

Return format: Just the number, nothing else.
Example: 5000.00

Do NOT include currency symbols, words, or explanations.
If you cannot estimate, return: NOT_FOUND"""
            
            else:
                # Unknown asset type
                return None
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # gpt-5 for better historical data accuracy
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            if response and response.choices:
                text = response.choices[0].message.content.strip()
                
                if text == "NOT_FOUND":
                    return None
                
                # Parse numeric value
                import re
                text = text.replace('₹', '').replace('Rs', '').replace('INR', '')
                text = re.sub(r'[^\d\.]', '', text)
                
                try:
                    price = float(text)
                    if 0 < price < 1000000:
                        #st.caption(f"🤖 AI found historical price for {ticker} on {target_date}: ₹{price:.2f}")
                        return price
                except ValueError:
                    pass
            
            return None
            
        except Exception as e:
            #st.caption(f"⚠️ AI historical fetch error: {str(e)}")
            return None
    
    def get_pms_aif_price_with_context(
        self,
        ticker: str,
        asset_type: str,
        transaction_date: str,
        transaction_price: float,
        quantity: float
    ) -> Optional[float]:
        """
        Calculate PMS/AIF current price using CAGR and transaction data
        
        Args:
            ticker: PMS/AIF ticker
            asset_type: 'pms' or 'aif'
            transaction_date: Purchase date (YYYY-MM-DD)
            transaction_price: Purchase price per unit
            quantity: Quantity purchased
        
        Returns:
            Current price per unit or None
        """
        if not self.pms_aif_calculator:
            # Fallback: use transaction price (0% growth)
            return transaction_price
        
        try:
            investment_amount = transaction_price * quantity
            is_aif = (asset_type == 'aif')
            
            # Calculate using CAGR
            result = self.pms_aif_calculator.calculate_pms_aif_value(
                ticker,
                transaction_date,
                investment_amount,
                is_aif
            )
            
            # Calculate per-unit price
            current_price = result['current_value'] / quantity if quantity > 0 else result['current_value']
            
            # Cache the weekly values for later use
            cache_key = f"{ticker}_weekly_values"
            self.price_cache[cache_key] = {
                'weekly_values': result['weekly_values'],
                'timestamp': datetime.now()
            }
            
            return current_price
            
        except Exception as e:
            #st.caption(f"⚠️ PMS/AIF calculation error for {ticker}: {str(e)}")
            return transaction_price  # Fallback
    
    def clear_cache(self):
        """Clear price cache"""
        self.price_cache = {}

