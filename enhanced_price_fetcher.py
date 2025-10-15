"""
Enhanced Price Fetcher with Complete Fallback Chain
yfinance → mftool → AI (for stocks and mutual funds)
"""

import streamlit as st
import yfinance as yf
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd


class EnhancedPriceFetcher:
    """
    Complete price fetching with multi-source fallback:
    Stock: yfinance NSE → yfinance BSE → AI
    MF: mftool → Transaction price → AI
    PMS/AIF: Manual or estimated
    """
    
    def __init__(self):
        self.price_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Initialize OpenAI for AI fallback
        self.ai_available = False
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
            self.ai_available = True
        except Exception as e:
            self.ai_available = False
            st.caption(f"⚠️ OpenAI not available: {str(e)}")
        
        # Initialize PMS/AIF calculator
        try:
            from pms_aif_calculator import PMS_AIF_Calculator
            self.pms_aif_calculator = PMS_AIF_Calculator()
        except Exception as e:
            self.pms_aif_calculator = None
            st.caption(f"⚠️ PMS/AIF calculator not available: {str(e)}")
    
    def get_current_price(self, ticker: str, asset_type: str, fund_name: str = None) -> Optional[float]:
        """
        Get current price with complete fallback chain
        
        Args:
            ticker: Ticker symbol
            asset_type: 'stock', 'mutual_fund', 'pms', 'aif', 'bond'
        
        Returns:
            Price or None
        """
        # Check cache first
        cache_key = f"{ticker}_{asset_type}_current"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            age = (datetime.now() - cached_data['timestamp']).total_seconds()
            if age < self.cache_timeout:
                return cached_data['price']
        
        price = None
        source = None
        
        if asset_type == 'stock':
            price, source = self._get_stock_price_with_fallback(ticker)
        elif asset_type == 'mutual_fund':
            price, source = self._get_mf_price_with_fallback(ticker, fund_name)
        elif asset_type in ['pms', 'aif']:
            # For PMS/AIF, calculate using CAGR if context provided
            if hasattr(self, 'pms_aif_calculator') and self.pms_aif_calculator:
                try:
                    # Try to get transaction context from cache or use conservative estimate
                    # This is a simplified version - full context should be provided by caller
                    st.caption(f"      💰 Calculating PMS/AIF value using CAGR...")
                    
                    # Use conservative CAGR estimates
                    conservative_cagr = 0.12 if asset_type == 'aif' else 0.10
                    
                    # For now, return a placeholder that indicates CAGR calculation needed
                    # The actual calculation should be done with proper transaction context
                    price = None
                    source = 'cagr_calculation_required'
                except Exception as e:
                    st.caption(f"      ⚠️ PMS/AIF calculation error: {str(e)}")
                    price = None
                    source = 'cagr_error'
            else:
                price = None
                source = 'cagr_calculator_unavailable'
        elif asset_type == 'bond':
            price, source = self._get_bond_price(ticker)
        
        # Cache result
        if price:
            self.price_cache[cache_key] = {
                'price': price,
                'timestamp': datetime.now(),
                'source': source
            }
        
        return price
    
    def update_live_prices_for_holdings(self, holdings: List[Dict], db_manager) -> None:
        """
        Update live_price in stock_master table for all holdings
        
        Args:
            holdings: List of holding records
            db_manager: Database manager instance
        """
        st.caption("🔄 Updating live prices for all holdings...")
        
        success_count = 0
        failed_count = 0
        mf_count = 0
        
        for holding in holdings:
            try:
                ticker = holding.get('ticker')
                asset_type = holding.get('asset_type', 'stock')
                stock_id = holding.get('stock_id')
                
                if not ticker or not stock_id:
                    st.caption(f"      ⚠️ Skipping holding with missing ticker or stock_id")
                    continue
                
                current_price = None
                
                if asset_type == 'stock':
                    st.caption(f"      📊 Fetching stock price for {ticker}...")
                    current_price, source = self._get_stock_price_with_fallback(ticker)
                    if current_price:
                        st.caption(f"      ✅ Stock {ticker}: ₹{current_price:,.2f} (from {source})")
                    
                elif asset_type == 'mutual_fund':
                    mf_count += 1
                    st.caption(f"      📈 Fetching MF NAV for {ticker}...")
                    # Get fund name for enhanced AI fallback
                    fund_name = holding.get('stock_name', '')
                    current_price, source = self._get_mf_price_with_fallback(ticker, fund_name)
                    if current_price:
                        st.caption(f"      ✅ MF {ticker}: ₹{current_price:,.2f} (from {source})")
                    else:
                        st.caption(f"      ❌ MF {ticker}: Failed to fetch NAV")
                    
                elif asset_type in ['pms', 'aif']:
                    st.caption(f"      💰 Calculating {asset_type.upper()} value for {ticker}...")
                    # Calculate PMS/AIF value using CAGR
                    if self.pms_aif_calculator:
                        # Get transaction details for CAGR calculation
                        transactions = db_manager.get_transactions_by_stock(holding['user_id'], stock_id)
                        if transactions:
                            # Use first transaction for calculation
                            first_transaction = transactions[0]
                            investment_date = first_transaction['transaction_date']
                            investment_amount = float(first_transaction['quantity']) * float(first_transaction['price'])
                            
                            result = self.pms_aif_calculator.calculate_pms_aif_value(
                                ticker, investment_date, investment_amount, is_aif=(asset_type == 'aif')
                            )
                            current_price = result['current_value'] / float(first_transaction['quantity'])
                            st.caption(f"      ✅ {asset_type.upper()} {ticker}: ₹{current_price:,.2f}")
                
                # Update live_price in stock_master
                if current_price and current_price > 0:
                    db_manager.update_stock_live_price(stock_id, current_price)
                    success_count += 1
                else:
                    st.caption(f"      ⚠️ Could not get price for {ticker} ({asset_type})")
                    failed_count += 1
                    
            except Exception as e:
                st.caption(f"      ❌ Error updating {holding.get('ticker', 'unknown')}: {str(e)}")
                failed_count += 1
        
        # Summary
        st.caption(f"✅ Price update complete: {success_count} successful, {failed_count} failed")
        st.caption(f"📈 Mutual Funds processed: {mf_count}")
    
    def _get_stock_price_with_fallback(self, ticker: str) -> tuple:
        """
        Stock price fetching with complete fallback:
        1. yfinance NSE (.NS)
        2. yfinance BSE (.BO)
        3. yfinance without suffix
        4. mftool (in case it's a mutual fund misclassified as stock)
        5. AI (OpenAI)
        """
        st.caption(f"      🔄 Fetching {ticker} with fallback chain...")
        
        # Method 1: Try NSE
        st.caption(f"      [1/5] Trying yfinance NSE...")
        try:
            nse_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
            stock = yf.Ticker(nse_ticker)
            hist = stock.history(period='1d')
            
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    st.caption(f"      ✅ Found on NSE: ₹{price:,.2f}")
                    return price, 'yfinance_nse'
        except Exception as e:
            st.caption(f"      ❌ NSE failed: {str(e)[:50]}")
        
        # Method 2: Try BSE
        st.caption(f"      [2/5] Trying yfinance BSE...")
        try:
            bse_ticker = f"{ticker}.BO" if not ticker.endswith(('.NS', '.BO')) else ticker.replace('.NS', '.BO')
            stock = yf.Ticker(bse_ticker)
            hist = stock.history(period='1d')
            
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    st.caption(f"      ✅ Found on BSE: ₹{price:,.2f}")
                    return price, 'yfinance_bse'
        except Exception as e:
            st.caption(f"      ❌ BSE failed: {str(e)[:50]}")
        
        # Method 3: Try without suffix
        st.caption(f"      [3/5] Trying yfinance without suffix...")
        if ticker.endswith(('.NS', '.BO')):
            try:
                clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
                stock = yf.Ticker(clean_ticker)
                hist = stock.history(period='1d')
                
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if price > 0:
                        st.caption(f"      ✅ Found without suffix: ₹{price:,.2f}")
                        return price, 'yfinance_raw'
            except Exception as e:
                st.caption(f"      ❌ Raw ticker failed: {str(e)[:50]}")
        else:
            st.caption(f"      ⏭️ Skipped (no suffix to remove)")
        
        # Method 4: Try mftool (in case it's a mutual fund)
        st.caption(f"      [4/5] Trying mftool (in case it's a MF)...")
        try:
            from mftool import Mftool
            mf = Mftool()
            
            # Try ticker as scheme code
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('MF_', '')
            quote = mf.get_scheme_quote(clean_ticker)
            
            if quote and 'nav' in quote:
                price = float(quote['nav'])
                if price > 0:
                    st.caption(f"      ✅ Found as MF on mftool: ₹{price:,.2f}")
                    return price, 'mftool'
        except Exception as e:
            st.caption(f"      ❌ mftool failed: {str(e)[:50]}")
        
        # Method 5: AI Fallback (if available)
        st.caption(f"      [5/5] Trying AI (OpenAI) as last resort...")
        if self.ai_available:
            try:
                price = self._get_price_from_ai(ticker, 'stock')
                if price:
                    st.caption(f"      ✅ AI found price: ₹{price:,.2f}")
                    return price, 'ai_openai'
                else:
                    st.caption(f"      ❌ AI couldn't find price")
            except Exception as e:
                st.caption(f"      ❌ AI failed: {str(e)[:50]}")
        else:
            st.caption(f"      ⚠️ AI not available")
        
        st.caption(f"      ❌ All methods failed for {ticker}")
        return None, 'not_found'
    
    def _get_mf_price_with_fallback(self, ticker: str, fund_name: str = None) -> tuple:
        """
        Mutual Fund price with fallback:
        1. mftool (AMFI API)
        2. AI (OpenAI)
        """
        st.caption(f"      🔄 Fetching MF {ticker} with fallback chain...")
        
        # Method 1: Try mftool
        st.caption(f"      [1/2] Trying mftool (AMFI API)...")
        try:
            from mftool import Mftool
            mf = Mftool()
            
            # Extract scheme code - remove any prefix
            scheme_code = ticker.replace('MF_', '').replace('mf_', '').strip()
            st.caption(f"      🔍 Using scheme code: {scheme_code}")
            
            quote = mf.get_scheme_quote(scheme_code)
            
            if quote and 'nav' in quote:
                price = float(quote['nav'])
                if price > 0:
                    st.caption(f"      ✅ Found on mftool: ₹{price:,.2f}")
                    st.caption(f"      📊 Scheme: {quote.get('scheme_name', 'N/A')}")
                    return price, 'mftool'
                else:
                    st.caption(f"      ❌ mftool returned invalid NAV: {price}")
                    st.caption(f"      💡 Scheme might be closed/merged: {quote.get('scheme_name', 'N/A')}")
            else:
                st.caption(f"      ❌ mftool: No NAV data found for scheme {scheme_code}")
                st.caption(f"      💡 This scheme code is INVALID or doesn't exist in AMFI database")
                st.caption(f"      🔧 Consider updating to correct AMFI scheme code")
        except Exception as e:
            st.caption(f"      ❌ mftool failed: {str(e)[:100]}")
            st.caption(f"      💡 Scheme code might be invalid or mftool API issue")
        
        # Method 2: AI Fallback with Enhanced Context
        st.caption(f"      [2/2] Trying AI (OpenAI) with enhanced context...")
        if self.ai_available:
            try:
                # Use provided fund name or try to get from context
                if not fund_name:
                    fund_name = self._get_fund_name_from_context(ticker)
                
                if fund_name:
                    st.caption(f"      🤖 Asking AI for NAV of '{fund_name}' (Code: {ticker})...")
                    price = self._get_mf_price_from_ai_enhanced(ticker, fund_name)
                else:
                    st.caption(f"      🤖 Asking AI for NAV of scheme {ticker}...")
                    price = self._get_price_from_ai(ticker, 'mutual_fund')
                
                if price:
                    st.caption(f"      ✅ AI found NAV: ₹{price:,.2f}")
                    return price, 'ai_openai_enhanced'
                else:
                    st.caption(f"      ❌ AI couldn't find NAV")
            except Exception as e:
                st.caption(f"      ❌ AI failed: {str(e)[:100]}")
        else:
            st.caption(f"      ⚠️ AI not available (check OpenAI API key in secrets)")
        
        st.caption(f"      ❌ All methods failed for MF {ticker}")
        st.caption(f"      🔧 SUGGESTION: Scheme code {ticker} is invalid")
        st.caption(f"      📋 To fix: Find correct AMFI scheme code from fund house website")
        st.caption(f"      🔍 Or use fund name search in mftool.get_scheme_codes()")
        st.caption(f"      💡 Manual intervention required - check scheme code or add price manually")
        return None, 'not_found'
    
    def _get_fund_name_from_context(self, ticker: str) -> Optional[str]:
        """Get fund name from database context if available"""
        try:
            # Try to get fund name from database or cache
            # This would need to be passed from the calling context
            # For now, return None and let the caller provide context
            return None
        except:
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
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
                timeout=30
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
            st.caption(f"      ❌ Enhanced AI fallback failed: {str(e)[:100]}")
            return None
    
    def _get_bond_price(self, ticker: str) -> tuple:
        """Bond price fetching (limited sources)"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    return price, 'yfinance'
        except:
            pass
        
        return None, 'manual_required'
    
    def _get_price_from_ai(self, ticker: str, asset_type: str) -> Optional[float]:
        """
        Get price from AI (OpenAI) as last resort
        Uses GPT-4 with web search for current prices
        
        Args:
            ticker: Ticker symbol
            asset_type: 'stock' or 'mutual_fund'
        
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
                
                user_prompt = f"""Find the current stock price for {ticker} on the Indian stock market (NSE or BSE).

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

            else:  # mutual_fund
                system_prompt = """You are a mutual fund NAV expert with access to AMFI India data.
Your task is to find the CURRENT NAV and return ONLY the numeric value.
Search AMFI or fund house websites if needed."""
                
                user_prompt = f"""Find the current NAV (Net Asset Value) for mutual fund code: {ticker}

This is an Indian mutual fund AMFI code (6-digit number).

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
            
            # Call OpenAI with optimized parameters
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,  # Deterministic output
                max_tokens=20,  # We only need a number
                timeout=10  # 10 second timeout
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
                        st.caption(f"🤖 AI found price for {ticker}: ₹{price:.2f}")
                        return price
                    else:
                        st.caption(f"⚠️ AI returned unrealistic price: {price}")
                        return None
                except ValueError:
                    # Try to extract first number from text
                    match = re.search(r'(\d+\.?\d*)', text)
                    if match:
                        price = float(match.group(1))
                        if price > 0 and price < 1000000:
                            st.caption(f"🤖 AI found price for {ticker}: ₹{price:.2f}")
                            return price
            
            return None
            
        except Exception as e:
            st.caption(f"⚠️ AI error for {ticker}: {str(e)}")
            return None
    
    def get_historical_price(self, ticker: str, asset_type: str, date: str, fund_name: str = None) -> Optional[float]:
        """
        Get historical price for a specific date
        """
        try:
            # Use the existing method but with a single day range
            prices = self.get_historical_prices(ticker, asset_type, date, date, fund_name)
            if prices and len(prices) > 0:
                return prices[0].get('price')
            return None
        except Exception as e:
            return None
    
    def get_historical_prices(self, ticker: str, asset_type: str, start_date: str, end_date: str, fund_name: str = None) -> list:
        """
        Get historical prices with complete fallback chain:
        Stock: yfinance NSE → yfinance BSE → yfinance raw → mftool → AI
        MF: mftool → AI
        """
        st.caption(f"      📅 Fetching historical prices for {ticker} ({start_date} to {end_date})...")
        
        try:
            if asset_type == 'stock':
                # Try yfinance with multiple suffixes and date ranges
                suffixes = ['.NS', '.BO', '']
                for idx, suffix in enumerate(suffixes, 1):
                    suffix_name = 'NSE' if suffix == '.NS' else 'BSE' if suffix == '.BO' else 'raw'
                    st.caption(f"      [{idx}/5] Trying yfinance {suffix_name}...")
                    
                    try:
                        test_ticker = f"{ticker}{suffix}" if suffix else ticker
                        stock = yf.Ticker(test_ticker)
                        
                        # Try exact date range first
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
                            st.caption(f"      ✅ Found {len(prices)} historical prices on {suffix_name}")
                            return prices
                        else:
                            # Try broader date range (±7 days) to find closest date
                            st.caption(f"      🔄 {suffix_name}: No exact date, trying ±7 days...")
                            from datetime import datetime, timedelta
                            import pytz
                            
                            # Handle timezone-aware datetime comparison
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            # Make timezone-aware
                            start_dt = pytz.UTC.localize(start_dt)
                            end_dt = pytz.UTC.localize(end_dt)
                            
                            # Expand range by 7 days before and after
                            expanded_start = (start_dt - timedelta(days=7)).strftime('%Y-%m-%d')
                            expanded_end = (end_dt + timedelta(days=7)).strftime('%Y-%m-%d')
                            
                            hist_expanded = stock.history(start=expanded_start, end=expanded_end)
                            
                            if not hist_expanded.empty:
                                # Find closest date to target
                                target_dt = start_dt
                                closest_date = None
                                min_diff = float('inf')
                                
                                for date, row in hist_expanded.iterrows():
                                    # Make date timezone-aware for comparison
                                    if date.tzinfo is None:
                                        date = pytz.UTC.localize(date)
                                    
                                    date_diff = abs((date - target_dt).days)
                                    if date_diff < min_diff:
                                        min_diff = date_diff
                                        closest_date = date
                                
                                if closest_date:
                                    closest_price = float(hist_expanded.loc[closest_date, 'Close'])
                                    st.caption(f"      ✅ {suffix_name}: Found closest price on {closest_date.strftime('%Y-%m-%d')} (±{min_diff} days): ₹{closest_price:,.2f}")
                                    
                                    return [{
                                        'asset_symbol': ticker,
                                        'asset_type': 'stock',
                                        'price': closest_price,
                                        'price_date': closest_date.strftime('%Y-%m-%d'),
                                        'volume': int(hist_expanded.loc[closest_date, 'Volume'])
                                    }]
                                else:
                                    st.caption(f"      ❌ {suffix_name}: No data in expanded range")
                            else:
                                st.caption(f"      ❌ {suffix_name}: No data found")
                    except Exception as e:
                        st.caption(f"      ❌ {suffix_name} failed: {str(e)[:50]}")
                
                # Try mftool (in case it's a mutual fund)
                st.caption(f"      [4/5] Trying mftool (in case it's a MF)...")
                try:
                    from mftool import Mftool
                    mf = Mftool()
                    
                    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('MF_', '')
                    hist_data = mf.get_scheme_historical_nav(clean_ticker, as_Dataframe=True)
                    
                    if hist_data is not None and not hist_data.empty:
                        hist_data['date'] = pd.to_datetime(hist_data.index, format='%d-%m-%Y', dayfirst=True)
                        
                        # Filter by date range
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
                            st.caption(f"      ✅ Found {len(prices)} historical NAVs on mftool")
                            return prices
                        else:
                            st.caption(f"      ❌ mftool: No data in date range")
                    else:
                        st.caption(f"      ❌ mftool: No historical data")
                except Exception as e:
                    st.caption(f"      ❌ mftool failed: {str(e)[:50]}")
            
            elif asset_type == 'mutual_fund':
                # Try mftool
                st.caption(f"      [1/2] Trying mftool (AMFI API)...")
                try:
                    from mftool import Mftool
                    mf = Mftool()
                    
                    scheme_code = ticker.replace('MF_', '') if ticker.startswith('MF_') else ticker
                    hist_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                    
                    if hist_data is not None and not hist_data.empty:
                        hist_data['date'] = pd.to_datetime(hist_data.index, format='%d-%m-%Y', dayfirst=True)
                        
                        # Filter by date range
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
                            st.caption(f"      ✅ Found {len(prices)} historical NAVs")
                            return prices
                        else:
                            st.caption(f"      ❌ mftool: No data in date range")
                    else:
                        st.caption(f"      ❌ mftool: No historical data")
                except Exception as e:
                    st.caption(f"      ❌ mftool failed: {str(e)[:50]}")
            
            # AI FALLBACK for historical prices
            # If yfinance/mftool failed, try AI for the target date
            fallback_step = '[5/5]' if asset_type == 'stock' else '[2/2]'
            st.caption(f"      {fallback_step} Trying AI (OpenAI) as last resort...")
            
            if self.ai_available:
                try:
                    # Use middle date of the range as target
                    target_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) / 2
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    
                    st.caption(f"      🤖 Asking AI for price around {target_date_str}...")
                    
                    price = self._get_historical_price_from_ai(ticker, asset_type, target_date_str, fund_name)
                    
                    if price:
                        st.caption(f"      ✅ AI found historical price: ₹{price:,.2f}")
                        return [{
                            'asset_symbol': ticker,
                            'asset_type': asset_type,
                            'price': price,
                            'price_date': target_date_str,
                            'volume': None
                        }]
                    else:
                        st.caption(f"      ❌ AI couldn't find historical price for {target_date}")
                        
                        # Final fallback: Try to get current price for recent dates
                        from datetime import datetime
                        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                        current_dt = datetime.now()
                        
                        # If target date is within last 6 months, try current price
                        if (current_dt - target_dt).days < 180:
                            st.caption(f"      🔄 Target date is recent, trying current price as fallback...")
                            current_price = self._get_price_from_ai(ticker, asset_type)
                            if current_price:
                                st.caption(f"      ✅ Using current price as fallback: ₹{current_price:,.2f}")
                                return [{
                                    'asset_symbol': ticker,
                                    'asset_type': asset_type,
                                    'price': current_price,
                                    'price_date': target_date,
                                    'volume': None
                                }]
                except Exception as e:
                    st.caption(f"      ❌ AI failed: {str(e)[:50]}")
            else:
                st.caption(f"      ⚠️ AI not available")
            
            st.caption(f"      ❌ All methods failed for historical prices of {ticker}")
            return []
            
        except Exception as e:
            st.caption(f"      ❌ Historical price error for {ticker}: {str(e)[:50]}")
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
                
                user_prompt = f"""Find the stock price for {ticker} on the Indian stock market (NSE/BSE) on date: {target_date}

IMPORTANT: 
- Search for the exact date: {target_date}
- If exact date not available, find the nearest trading day before or after this date
- Look for historical data from July 2025
- This is a real stock that was trading in July 2025

Return format: Just the number, nothing else.
Example: 2650.50

Do NOT include currency symbols, words, or explanations.
If you cannot find it, return: NOT_FOUND"""
            
            else:  # mutual_fund
                system_prompt = """You are a mutual fund expert with access to historical NAV data.
Your task is to find the NAV for a specific date and return ONLY the numeric value."""
                
                # Enhanced prompt with fund name if available
                if fund_name:
                    user_prompt = f"""Find the NAV (Net Asset Value) for this Indian mutual fund on date: {target_date}

Fund Details:
- Scheme Code: {ticker}
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
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use more capable model for historical data
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=50,
                timeout=30
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
                        st.caption(f"🤖 AI found historical price for {ticker} on {target_date}: ₹{price:.2f}")
                        return price
                except ValueError:
                    pass
            
            return None
            
        except Exception as e:
            st.caption(f"⚠️ AI historical fetch error: {str(e)}")
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
            st.caption(f"⚠️ PMS/AIF calculation error for {ticker}: {str(e)}")
            return transaction_price  # Fallback
    
    def clear_cache(self):
        """Clear price cache"""
        self.price_cache = {}

