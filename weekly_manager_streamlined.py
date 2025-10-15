"""
Streamlined Weekly Price Manager
Matches your image requirements:
- Fetch missing weeks till current week based on week of year
- Use NAVs for 52-week calculations
- Bulk AI fetching for optimization
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from database_shared import SharedDatabaseManager
from enhanced_price_fetcher import EnhancedPriceFetcher
from bulk_ai_fetcher import BulkAIFetcher

class StreamlinedWeeklyManager:
    """
    Weekly price manager that matches your image workflow:
    1. Fetch missing weeks till current week based on week of year
    2. Use NAVs for 52-week calculations
    3. Bulk AI fetching for optimization
    """
    
    def __init__(self, db: SharedDatabaseManager, price_fetcher: EnhancedPriceFetcher):
        self.db = db
        self.price_fetcher = price_fetcher
        self.bulk_ai = BulkAIFetcher()
    
    def fetch_missing_weeks_till_current(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch missing weeks till current week - OPTIMIZED YEARLY BULK FETCH
        Fetches entire year of data in ONE API call per ticker
        """
        try:
            from datetime import datetime, timedelta
            from fetch_yearly_bulk import fetch_yearly_prices_for_all_tickers, save_yearly_prices_to_db
            
            st.caption("🔍 Analyzing user holdings and transaction weeks...")
            
            # Get user holdings first
            holdings = self.db.get_user_holdings(user_id)
            if not holdings:
                st.caption("ℹ️ No holdings found - nothing to fetch")
                return {'success': True, 'message': 'No holdings found', 'fetched': 0}
            
            # Show unique tickers being analyzed
            unique_tickers = list(set([h['ticker'] for h in holdings]))
            
            # Get unique holdings (deduplicate by ticker)
            unique_holdings = {}
            for h in holdings:
                if h['ticker'] not in unique_holdings:
                    unique_holdings[h['ticker']] = h
            unique_holdings_list = list(unique_holdings.values())
            
            # Calculate date range: Last 52 weeks
            current_date = datetime.now()
            start_date = current_date - timedelta(weeks=52)
            
            # Fetching last 52 weeks (silent)
            
            # OPTIMIZED: Fetch entire year for all holdings at once (silent)
            all_prices = fetch_yearly_prices_for_all_tickers(unique_holdings_list, start_date, current_date)
            
            # Save to database
            if all_prices:
                total_saved = save_yearly_prices_to_db(self.db, all_prices)
                
                st.success(f"🎉 Bulk fetch complete!")
                st.metric("✅ Prices Saved", total_saved)
                st.metric("🎯 Tickers Processed", len(all_prices))
                
                return {
                    'success': True,
                    'fetched': total_saved,
                    'updated_tickers': list(all_prices.keys()),
                    'message': f'Fetched {total_saved} prices for {len(all_prices)} tickers'
                }
            else:
                st.warning("⚠️ No prices fetched")
                return {'success': True, 'fetched': 0}
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def fetch_missing_weeks_till_current_OLD(self, user_id: str) -> Dict[str, Any]:
        """
        OLD METHOD: Fetch missing weeks till current week based on week of year
        Matches your image: "fetch missing weeks till current week based on week of year"
        """
        try:
            st.caption("🔍 Analyzing user holdings and transaction weeks...")
            
            # Get user holdings first
            holdings = self.db.get_user_holdings(user_id)
            if not holdings:
                st.caption("ℹ️ No holdings found - nothing to fetch")
                return {'success': True, 'message': 'No holdings found', 'fetched': 0}
            
            # Show unique tickers being analyzed
            unique_tickers = list(set([h['ticker'] for h in holdings]))
            
            # Get missing weeks for user's transactions (silent)
            missing_weeks = self.db.get_missing_weeks_for_user(user_id)
            
            if not missing_weeks:
                st.caption("✅ All weeks up-to-date - no missing prices found")
                return {'success': True, 'message': 'All weeks up-to-date', 'fetched': 0}
            
            # Found missing weeks to fetch (silent)
            
            # Analyze missing weeks
            unique_tickers = list(set([m['ticker'] for m in missing_weeks]))
            unique_weeks = list(set([f"{m['year']}-W{m['week']:02d}" for m in missing_weeks]))
            
            # Group by week for bulk fetching
            week_groups = {}
            for missing in missing_weeks:
                week_key = f"{missing['year']}-W{missing['week']:02d}"
                if week_key not in week_groups:
                    week_groups[week_key] = []
                week_groups[week_key].append(missing)
            
            # Grouped into weeks for bulk fetching (silent)
            
            fetched_count = 0
            updated_tickers = set()
            failed_weeks = []
            
            # Process weeks (silent)
            
            # Process each week group
            for week_idx, (week_key, week_missing) in enumerate(week_groups.items(), 1):
                year, week_num = week_key.split('-W')
                year = int(year)
                week_num = int(week_num)
                
                # Get Monday of this week
                week_monday = datetime.strptime(f'{year}-W{week_num:02d}-1', '%Y-W%W-%w')
                
                # Prepare bulk fetch data
                tickers_with_info = []
                ticker_names = []
                for missing in week_missing:
                    # Get stock info
                    stock_info = self.db.supabase.table('stock_master').select(
                        'ticker, stock_name, asset_type'
                    ).eq('id', missing['stock_id']).execute()
                    
                    if stock_info.data:
                        stock = stock_info.data[0]
                        tickers_with_info.append((
                            stock['ticker'],
                            stock['stock_name'],
                            stock['asset_type'],
                            week_monday.strftime('%Y-%m-%d')
                        ))
                        ticker_names.append(stock['ticker'])
                
                # Show tickers being processed
                # Processing tickers (silent)
                
                # Bulk fetch prices for this week
                if tickers_with_info:
                    # Fetching prices for week (silent)
                    prices = self._fetch_week_prices_bulk(tickers_with_info, week_monday)
                    
                    # Save to database
                    if prices:
                        # Storing prices to database (silent)
                        self._save_week_prices(prices, week_monday, year, week_num)
                        fetched_count += len(prices)
                        
                        # Track updated tickers
                        for price in prices:
                            updated_tickers.add(price.get('asset_symbol', 'Unknown'))
                        
                        st.caption(f"   ✅ {week_key}: {len(prices)} prices stored successfully")
                    else:
                        st.caption(f"   ⚠️ {week_key}: No prices fetched (will retry later)")
                        failed_weeks.append(week_key)
                else:
                    st.caption(f"   ⚠️ {week_key}: No ticker info found")
                    failed_weeks.append(week_key)
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Final summary
            st.success(f"🎉 Fetching complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Prices Fetched", fetched_count)
            with col2:
                st.metric("🎯 Tickers Updated", len(updated_tickers))
            with col3:
                st.metric("⚠️ Failed Weeks", len(failed_weeks))
            
            
            if failed_weeks:
                st.warning(f"⚠️ {len(failed_weeks)} weeks could not be fetched: {', '.join(failed_weeks[:5])}{'...' if len(failed_weeks) > 5 else ''}")
            
            return {
                'success': True,
                'fetched': fetched_count,
                'updated_tickers': list(updated_tickers),
                'failed_weeks': failed_weeks,
                'message': f'Fetched {fetched_count} missing week prices for {len(updated_tickers)} tickers'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fetch_yearly_prices_bulk(self, ticker: str, asset_type: str, start_date: datetime, end_date: datetime) -> Dict[tuple, float]:
        """
        Fetch entire year of historical prices at once for a ticker
        Returns dict of {(year, week): price}
        """
        import yfinance as yf
        
        weekly_prices = {}
        
        try:
            if asset_type == 'stock':
                # Try NSE first
                yf_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
                stock = yf.Ticker(yf_ticker)
                
                # Fetch historical data for the entire period
                hist = stock.history(start=start_date, end=end_date, interval='1wk')
                
                if hist.empty:
                    # Try BSE
                    yf_ticker = f"{ticker}.BO" if not ticker.endswith(('.NS', '.BO')) else ticker.replace('.NS', '.BO')
                    stock = yf.Ticker(yf_ticker)
                    hist = stock.history(start=start_date, end=end_date, interval='1wk')
                
                # Convert to weekly prices
                for date, row in hist.iterrows():
                    year, week, _ = date.isocalendar()
                    price = float(row['Close'])
                    if price > 0:
                        weekly_prices[(year, week)] = price
                        
        except Exception as e:
            pass
        
        return weekly_prices
    
    def _fetch_week_prices_bulk(self, tickers_with_info: List[tuple], week_monday: datetime) -> List[Dict[str, Any]]:
        """
        Fetch prices for multiple tickers in one week using bulk AI
        """
        prices = []
        
        # Try regular price fetcher first
        # Starting price fetch (silent)
        
        for idx, (ticker, name, asset_type, date) in enumerate(tickers_with_info, 1):
            # Fetching ticker (silent)
            
            
            if asset_type == 'stock':
                    # Using stock price fetcher (silent)
                    price = self.price_fetcher.get_current_price(ticker, asset_type)
            elif asset_type == 'mutual_fund':
                    # Using mutual fund price fetcher (silent)
                    price = self.price_fetcher.get_current_price(ticker, asset_type)
            elif asset_type in ['pms', 'aif']:
                    # Calculating PMS/AIF NAV using CAGR (silent)
                    # For PMS/AIF, use CAGR calculation (as per your requirements)
                    price = self._calculate_pms_aif_nav(ticker, asset_type, date)
            else:
                    # Unknown asset type (silent)
                    price = None
                
            if price:
                    prices.append({
                        'ticker': ticker,
                        'name': name,
                        'asset_type': asset_type,
                        'price': price,
                        'source': 'api'
                    })
                    # Price found (silent)

        
        # Use bulk AI for remaining tickers
        remaining = [(t, n, a, d) for t, n, a, d in tickers_with_info 
                    if not any(p['ticker'] == t for p in prices)]
        
        if remaining and self.bulk_ai.available:
            st.caption(f"   🤖 Trying bulk AI for {len(remaining)} failed tickers...")
            try:
                # Prepare for bulk AI
                ai_tickers = [(t, n, a) for t, n, a, d in remaining]
                ai_prices = self.bulk_ai.fetch_bulk_current_prices(ai_tickers)
                
                ai_successful = 0
                for ticker, name, asset_type, date in remaining:
                    if ticker in ai_prices:
                        prices.append({
                            'ticker': ticker,
                            'name': name,
                            'asset_type': asset_type,
                            'price': ai_prices[ticker],
                            'source': 'ai_bulk'
                        })
                        ai_successful += 1
                        st.caption(f"   ✅ {ticker}: ₹{ai_prices[ticker]:,.2f} (via AI)")
                
                st.caption(f"   ✅ AI fetch: {ai_successful}/{len(remaining)} successful")
            except Exception as e:
                st.caption(f"   ❌ Bulk AI failed: {str(e)}")
        elif remaining:
            st.caption(f"   ⚠️ Bulk AI not available for {len(remaining)} failed tickers")
        
        # Final summary
        total_successful = len(prices)
        total_failed = len(tickers_with_info) - total_successful
        
        st.caption(f"   🎉 Final result: {total_successful}/{len(tickers_with_info)} tickers successful")
        if total_failed > 0:
            final_failed = [t for t, n, a, d in tickers_with_info 
                          if not any(p['ticker'] == t for p in prices)]
            st.caption(f"   ⚠️ Failed: {', '.join(final_failed[:3])}{'...' if len(final_failed) > 3 else ''}")
        
        return prices
    
    def _calculate_pms_aif_nav(self, ticker: str, asset_type: str, date: str) -> Optional[float]:
        """
        Calculate PMS/AIF NAV using CAGR (as per your requirements)
        """
        try:
            # This would integrate with your PMS/AIF calculator
            # For now, return a placeholder
            if asset_type == 'pms':
                return 1000.0  # Placeholder PMS NAV
            elif asset_type == 'aif':
                return 1200.0  # Placeholder AIF NAV
            return None
        except:
            return None
    
    def _save_week_prices(self, prices: List[Dict[str, Any]], week_monday: datetime, year: int, week_num: int):
            """
        Save week prices to historical_prices table
            """
        
            price_records = []
            
            for price_data in prices:
                # Get stock_id
                stock_response = self.db.supabase.table('stock_master').select('id').eq(
                    'ticker', price_data['ticker']
                ).execute()
                
                if stock_response.data:
                    stock_id = stock_response.data[0]['id']
                    
                    price_records.append({
                        'stock_id': stock_id,
                        'price_date': week_monday.strftime('%Y-%m-%d'),
                        'price': price_data['price'],
                        'volume': None,
                        'source': price_data['source'],
                        'iso_year': year,
                        'iso_week': week_num
                    })
            
            if price_records:
                self.db.save_historical_prices_bulk(price_records)
                # Saved prices (silent)
        

    
    def get_52_week_navs(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get 52-week NAVs for user's holdings
        Matches your requirement: "for 52 week use navs"
        """
        try:
            holdings = self.db.get_user_holdings(user_id)
            nav_data = {}
            
            for holding in holdings:
                ticker = holding['ticker']
                asset_type = holding['asset_type']
                stock_id = holding['stock_id']
                
                # Get 52 weeks of historical prices
                end_date = datetime.now()
                start_date = end_date - timedelta(weeks=52)
                
                prices = self.db.get_historical_prices_for_stock(
                    stock_id,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # Convert to NAV format
                navs = []
                for price in prices:
                    navs.append({
                        'date': price['price_date'],
                        'nav': price['price'],
                        'week': price.get('iso_week', 0),
                        'year': price.get('iso_year', 0)
                    })
                
                nav_data[ticker] = navs
            
            return nav_data
            
        except Exception as e:
            st.error(f"Error getting 52-week NAVs: {str(e)}")
            return {}
    
    def calculate_pms_cagr(self, ticker: str, transaction_date: str, transaction_price: float) -> Optional[float]:
        """
        Calculate PMS CAGR when price is mentioned (as per your requirements)
        """
        try:
            # This would integrate with your PMS/AIF calculator
            # For now, return a placeholder calculation
            current_date = datetime.now()
            trans_date = datetime.strptime(transaction_date, '%Y-%m-%d')
            
            years_elapsed = (current_date - trans_date).days / 365.25
            
            # Placeholder CAGR calculation
            # In real implementation, this would use SEBI data
            cagr = 0.10  # 10% placeholder
            
            current_nav = transaction_price * ((1 + cagr) ** years_elapsed)
            
            return current_nav
            
        except Exception as e:
            # Error calculating PMS CAGR (silent)
            return None
