"""
Database Manager with Shared Architecture
- stock_master: Shared across all users
- historical_prices: Shared across all users
- user_transactions: User-specific, links to shared tables
"""

import streamlit as st
from supabase import create_client, Client
from typing import Optional, Dict, List, Any, Tuple
import hashlib
from datetime import datetime
import pandas as pd

class SharedDatabaseManager:
    """Manages database with shared historical data architecture"""
    
    def __init__(self):
        try:
            # Get and validate Supabase credentials
            supabase_url = st.secrets["supabase"]["url"]
            supabase_key = st.secrets["supabase"]["key"]
            
            # Validate URL format
            if not supabase_url or not supabase_url.startswith("https://"):
                raise ValueError(f"Invalid Supabase URL format. Must start with 'https://'")
            
            if not supabase_key or len(supabase_key) < 20:
                raise ValueError("Invalid Supabase key. Key appears to be too short or empty.")
            
            # Create client
            self.supabase: Client = create_client(
                supabase_url.strip(),
                supabase_key.strip()
            )
            
        except KeyError as e:
            st.error(f"❌ Missing Supabase configuration in secrets: {e}")
            raise
        except Exception as e:
            st.error(f"❌ Database connection error: {str(e)}")
            raise
    
    # ========================================================================
    # USER MANAGEMENT (Unchanged)
    # ========================================================================
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, password: str, full_name: str, email: str = None) -> Dict[str, Any]:
        """Register new user with username"""
        try:
            password_hash = self.hash_password(password)
            
            response = self.supabase.table('users').insert({
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'full_name': full_name
            }).execute()
            
            return {'success': True, 'user': response.data[0]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def login_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login user with username"""
        try:
            password_hash = self.hash_password(password)
            
            response = self.supabase.table('users').select('*').eq(
                'username', username
            ).eq('password_hash', password_hash).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            st.error(f"Login error: {str(e)}")
            return None
    
    # ========================================================================
    # PDF STORAGE MANAGEMENT
    # ========================================================================
    
    def save_pdf(self, user_id: str, filename: str, pdf_text: str, ai_summary: str = None) -> Dict[str, Any]:
        """Save PDF content to database"""
        try:
            # Clean text to remove null bytes and other problematic characters
            cleaned_text = self._clean_text_for_db(pdf_text) if pdf_text else ""
            cleaned_summary = self._clean_text_for_db(ai_summary) if ai_summary else None
            
            response = self.supabase.table('user_pdfs').insert({
                'user_id': user_id,
                'filename': filename,
                'pdf_text': cleaned_text,
                'ai_summary': cleaned_summary
            }).execute()
            
            return {'success': True, 'pdf': response.data[0]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _clean_text_for_db(self, text: str) -> str:
        """Clean text to remove characters that PostgreSQL can't handle"""
        if not text:
            return ""
        
        import re
        
        # Convert to string if not already
        text = str(text)
        
        # Remove ALL null bytes and variants
        text = text.replace('\x00', ' ')
        text = text.replace('\u0000', ' ')
        text = text.replace('\\x00', ' ')
        text = text.replace('\\u0000', ' ')
        
        # Remove all control characters except newline and tab
        # This removes characters from 0x00-0x1F except \n (0x0A) and \t (0x09)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
        
        # Remove any remaining non-printable characters
        text = ''.join(char if char.isprintable() or char in '\n\t\r ' else ' ' for char in text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        # Limit length to avoid database issues (max 1MB of text)
        max_length = 1000000
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        
        return text
    
    def get_user_pdfs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all PDFs for a user"""
        try:
            response = self.supabase.table('user_pdfs').select('*').eq(
                'user_id', user_id
            ).order('uploaded_at', desc=True).execute()
            
            return response.data
        except Exception as e:
            st.error(f"Error fetching PDFs: {str(e)}")
            return []
    
    def get_pdf_by_id(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific PDF by ID"""
        try:
            response = self.supabase.table('user_pdfs').select('*').eq(
                'id', pdf_id
            ).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            st.error(f"Error fetching PDF: {str(e)}")
            return None
    
    def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF"""
        try:
            self.supabase.table('user_pdfs').delete().eq('id', pdf_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting PDF: {str(e)}")
            return False
    
    def get_all_pdfs_text(self, user_id: str) -> str:
        """Get combined text from all user PDFs for AI context"""
        try:
            pdfs = self.get_user_pdfs(user_id)
            if not pdfs:
                return ""
            
            combined_text = "\n\n--- PDF DOCUMENTS ---\n\n"
            for pdf in pdfs:
                combined_text += f"\n📄 {pdf['filename']}:\n"
                combined_text += pdf['pdf_text'][:2000] + "...\n"  # Limit per PDF
            
            return combined_text
        except Exception as e:
            return ""
    
    def create_portfolio(self, user_id: str, portfolio_name: str) -> Dict[str, Any]:
        """Create portfolio"""
        try:
            response = self.supabase.table('portfolios').insert({
                'user_id': user_id,
                'portfolio_name': portfolio_name
            }).execute()
            
            return {'success': True, 'portfolio': response.data[0]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_portfolios(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user portfolios"""
        try:
            response = self.supabase.table('portfolios').select('*').eq(
                'user_id', user_id
            ).execute()
            return response.data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    # ========================================================================
    # SHARED STOCK MASTER (NEW!)
    # ========================================================================
    
    def get_or_create_stock(
        self,
        ticker: str,
        stock_name: str,
        asset_type: str,
        sector: str = None
    ) -> Optional[str]:
        """
        Get existing stock or create new one in stock_master
        Returns stock_id (UUID)
        """
        try:
            # Try to find existing
            response = self.supabase.table('stock_master').select('*').eq(
                'ticker', ticker
            ).eq('stock_name', stock_name).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]['id']
            
            # Create new
            insert_data = {
                'ticker': ticker,
                'stock_name': stock_name,
                'asset_type': asset_type
            }
            
            if sector:
                insert_data['sector'] = sector
            
            response = self.supabase.table('stock_master').insert(insert_data).execute()
            
            if response.data:
                return response.data[0]['id']
            
            return None
            
        except Exception as e:
            st.error(f"Error creating stock: {str(e)}")
            return None
    
    def update_stock_live_price(self, stock_id: str, live_price: float):
        """Update live price in stock_master"""
        try:
            self.supabase.table('stock_master').update({
                'live_price': live_price,
                'last_updated': datetime.now().isoformat()
            }).eq('id', stock_id).execute()
        except Exception as e:
            # Update stock price error
            pass
    
    def get_transactions_by_stock(self, user_id: str, stock_id: str) -> List[Dict[str, Any]]:
        """Get all transactions for a specific stock"""
        try:
            response = self.supabase.table('user_transactions').select('*').eq('user_id', user_id).eq('stock_id', stock_id).order('transaction_date').execute()
            return response.data
        except Exception as e:
            #st.caption(f"⚠️ Get transactions by stock error: {str(e)}")
            return []
    
    def get_all_unique_stocks(self) -> List[Dict[str, Any]]:
        """Get all unique stocks from stock_master"""
        try:
            response = self.supabase.table('stock_master').select('*').execute()
            return response.data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    # ========================================================================
    # SHARED HISTORICAL PRICES (NEW!)
    # ========================================================================
    
    def save_historical_prices_bulk(self, prices: List[Dict[str, Any]]) -> bool:
        """
        Save multiple historical prices to shared table
        
        Args:
            prices: List of {stock_id, price_date, price, source, iso_year, iso_week}
        
        Returns:
            Success boolean
        """
        try:
            # Upsert to avoid duplicates
            self.supabase.table('historical_prices').upsert(
                prices,
                on_conflict='stock_id,price_date'
            ).execute()
            return True
        except Exception as e:
            st.error(f"Error saving historical prices: {str(e)}")
            return False
    
    def get_historical_prices_for_stock(
        self,
        stock_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get historical prices for a stock from shared table"""
        try:
            response = self.supabase.table('historical_prices').select('*').eq(
                'stock_id', stock_id
            ).gte('price_date', start_date).lte('price_date', end_date).order(
                'price_date', desc=False
            ).execute()
            
            return response.data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    def get_historical_prices_for_stock_silent(self, stock_id: str) -> List[Dict[str, Any]]:
        """Get all historical prices for a stock without logging (for charts)"""
        try:
            response = self.supabase.table('historical_prices').select('*').eq(
                'stock_id', stock_id
            ).order('price_date', desc=False).execute()
            
            return response.data
        except Exception as e:
            return []
    
    def get_missing_weeks_for_stock(
        self,
        stock_id: str,
        year: int,
        week_numbers: List[int]
    ) -> List[int]:
        """
        Check which ISO week numbers are missing for a stock
        
        Args:
            stock_id: Stock UUID
            year: Year (e.g., 2024)
            week_numbers: List of ISO week numbers to check (e.g., [40, 41, 42])
        
        Returns:
            List of missing week numbers
        """
        try:
            # Get all cached weeks for this stock in this year
            response = self.supabase.table('historical_prices').select(
                'iso_week'
            ).eq('stock_id', stock_id).eq('iso_year', year).execute()
            
            cached_weeks = [row['iso_week'] for row in response.data]
            missing = [w for w in week_numbers if w not in cached_weeks]
            
            return missing
        except Exception as e:
            return week_numbers  # Assume all missing if error
    
    # ========================================================================
    # USER TRANSACTIONS (UPDATED!)
    # ========================================================================
    
    def add_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add transaction with week tracking (as per your image requirements)
        
        Args:
            transaction_data: Must include ticker, stock_name, asset_type, sector
        """
        try:
            # Get or create stock in shared table
            stock_id = self.get_or_create_stock(
                transaction_data['ticker'],
                transaction_data['stock_name'],
                transaction_data['asset_type'],
                transaction_data.get('sector')
            )
            
            if not stock_id:
                return {'success': False, 'error': 'Could not create stock'}
            
            # Calculate week info from transaction date
            try:
                trans_date = pd.to_datetime(transaction_data['transaction_date'])
                iso_year = trans_date.isocalendar()[0]
                iso_week = trans_date.isocalendar()[1]
                week_label = f"Wk{iso_week} {iso_year}"
            except Exception as e:
                # Fallback if isocalendar fails
                from datetime import datetime
                trans_date = datetime.strptime(transaction_data['transaction_date'], '%Y-%m-%d')
                iso_year = trans_date.year
                iso_week = trans_date.isocalendar()[1]
                week_label = f"Wk{iso_week} {iso_year}"
            
            # Create transaction with week tracking
            trans_insert = {
                'user_id': transaction_data['user_id'],
                'portfolio_id': transaction_data['portfolio_id'],
                'stock_id': stock_id,
                'quantity': transaction_data['quantity'],
                'price': transaction_data['price'],
                'transaction_date': transaction_data['transaction_date'],
                'transaction_type': transaction_data['transaction_type'],
                'channel': transaction_data.get('channel', 'Direct'),
                'notes': transaction_data.get('notes', ''),
                # Week tracking (as per your image)
                'iso_year': iso_year,
                'iso_week': iso_week,
                'week_label': week_label
            }
            
            response = self.supabase.table('user_transactions').insert(trans_insert).execute()
            
            # Update holdings
            self._update_holdings(transaction_data['user_id'], transaction_data['portfolio_id'], stock_id)
            
            return {'success': True, 'transaction': response.data[0]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _update_holdings(self, user_id: str, portfolio_id: str, stock_id: str):
        """Update holdings based on transactions"""
        try:
            # Get all transactions for this stock
            response = self.supabase.table('user_transactions').select('*').eq(
                'user_id', user_id
            ).eq('portfolio_id', portfolio_id).eq('stock_id', stock_id).execute()
            
            transactions = response.data
            
            if not transactions:
                return
            
            # Calculate total quantity and average price
            total_qty = 0
            total_cost = 0
            
            for trans in transactions:
                qty = float(trans['quantity'])
                price = float(trans['price'])
                
                if trans['transaction_type'] == 'buy':
                    total_cost += qty * price
                    total_qty += qty
                else:  # sell
                    total_qty -= qty
            
            if total_qty > 0:
                avg_price = total_cost / total_qty
                
                # Upsert holding
                self.supabase.table('holdings').upsert({
                    'user_id': user_id,
                    'portfolio_id': portfolio_id,
                    'stock_id': stock_id,
                    'total_quantity': total_qty,
                    'average_price': avg_price
                }, on_conflict='user_id,portfolio_id,stock_id').execute()
            else:
                # Delete holding if quantity = 0
                self.supabase.table('holdings').delete().eq(
                    'user_id', user_id
                ).eq('portfolio_id', portfolio_id).eq('stock_id', stock_id).execute()
        
        except Exception as e:
            pass
#st.caption(f"⚠️ Update holdings error: {str(e)}")
    
    def get_user_holdings(self, user_id: str, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user holdings with stock details (uses view)"""
        try:
            query = self.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
            
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            response = query.execute()
            holdings = response.data
            
            # Get channel information from transactions for each holding
            for holding in holdings:
                stock_id = holding['stock_id']
                # Get the most recent channel for this stock
                channel_response = self.supabase.table('user_transactions').select('channel').eq(
                    'user_id', user_id
                ).eq('stock_id', stock_id).order('transaction_date', desc=True).limit(1).execute()
                
                if channel_response.data:
                    holding['channel'] = channel_response.data[0]['channel']
                else:
                    holding['channel'] = 'Direct'
            
            return holdings
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    def get_user_holdings_silent(self, user_id: str, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user holdings without logging (for charts page)"""
        try:
            # Get holdings from the view
            query = self.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
            
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            response = query.execute()
            holdings = response.data
            
            # Get channel information from transactions for each holding
            for holding in holdings:
                stock_id = holding['stock_id']
                # Get the most recent channel for this stock
                channel_response = self.supabase.table('user_transactions').select('channel').eq(
                    'user_id', user_id
                ).eq('stock_id', stock_id).order('transaction_date', desc=True).limit(1).execute()
                
                if channel_response.data:
                    holding['channel'] = channel_response.data[0]['channel']
                else:
                    holding['channel'] = 'Direct'
            
            return holdings
        except Exception as e:
            return []
    
    def get_user_transactions(self, user_id: str, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user transactions with stock details (uses view)"""
        try:
            query = self.supabase.table('user_transactions_detailed').select('*').eq(
                'user_id', user_id
            ).order('transaction_date', desc=True)
            
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            response = query.execute()
            return response.data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    def get_user_transaction_weeks(self, user_id: str) -> List[Tuple[int, int]]:
        """
        Get unique (year, week) combinations from user's transactions
        Used to determine which weeks need historical prices
        """
        try:
            #st.caption(f"      🔍 Querying user_transactions for user {user_id[:8]}...")
            response = self.supabase.table('user_transactions').select(
                'iso_year, iso_week'
            ).eq('user_id', user_id).execute()
            
            #st.caption(f"      📊 Found {len(response.data)} transaction records")
            
            # Get unique combinations
            weeks = set()
            for row in response.data:
                if row['iso_year'] and row['iso_week']:
                    weeks.add((row['iso_year'], row['iso_week']))
            
            #st.caption(f"      ✅ Extracted {len(weeks)} unique (year, week) combinations")
            return list(weeks)
        except Exception as e:
            st.error(f"❌ Error getting transaction weeks: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return []
    
    def get_missing_weeks_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get missing weeks for user's transactions
        Returns list of {stock_id, ticker, year, week} that need historical prices
        
        OPTIMIZED: Fetch transaction weeks + last 52 weeks for all holdings
        """
        try:
            from datetime import datetime, timedelta
            
            #st.caption("   🔍 Step 1: Getting user transactions...")
            
            # Get all transactions with their weeks
            response = self.supabase.table('user_transactions').select(
                'stock_id, iso_year, iso_week'
            ).eq('user_id', user_id).execute()
            
            if not response.data:
                #st.caption("   ⚠️ No transactions found")
                return []
            
            # Get unique stock IDs and transaction weeks
            stock_ids = set()
            transaction_weeks = set()
            
            for row in response.data:
                if row['stock_id']:
                    stock_ids.add(row['stock_id'])
                if row['iso_year'] and row['iso_week']:
                    transaction_weeks.add((row['iso_year'], row['iso_week']))
            
            stock_ids = list(stock_ids)
            #st.caption(f"   ✅ Found {len(stock_ids)} unique stocks")
            #st.caption(f"   ✅ Found {len(transaction_weeks)} transaction weeks")
            
            # Calculate last 52 weeks
            current_date = datetime.now()
            start_date = current_date - timedelta(weeks=52)
            
            current_year, current_week, _ = current_date.isocalendar()
            start_year, start_week, _ = start_date.isocalendar()
            
            #st.caption(f"   📅 Last 52 weeks: {start_year}-W{start_week:02d} to {current_year}-W{current_week:02d}")
            
            # Generate all weeks in the last 52 weeks
            last_52_weeks = []
            temp_date = start_date
            while temp_date <= current_date:
                year, week, _ = temp_date.isocalendar()
                if (year, week) not in last_52_weeks:
                    last_52_weeks.append((year, week))
                temp_date += timedelta(weeks=1)
            
            # Combine transaction weeks + last 52 weeks (unique)
            all_weeks = list(set(transaction_weeks) | set(last_52_weeks))
            #st.caption(f"   ✅ Total weeks to check: {len(all_weeks)} (transaction weeks + last 52 weeks)")
            
            # Get stock details
            #st.caption("   🔍 Step 2: Getting stock details from stock_master...")
            stock_details = {}
            for stock_id in stock_ids:
                stock_response = self.supabase.table('stock_master').select(
                    'id, ticker, stock_name'
                ).eq('id', stock_id).execute()
                
                if stock_response.data:
                    stock = stock_response.data[0]
                    stock_details[stock_id] = stock['ticker']
            
            #st.caption(f"   ✅ Retrieved details for {len(stock_details)} stocks")
            
            # OPTIMIZATION: Get ALL existing prices for this user's stocks in ONE query
            #st.caption("   🔍 Step 3: Checking existing prices (bulk query)...")
            
            existing_prices = {}
            if stock_ids:
                # Get all existing prices for user's stocks in one query
                existing_response = self.supabase.table('historical_prices').select(
                    'stock_id, iso_year, iso_week'
                ).in_('stock_id', stock_ids).execute()
                
                # Build a set of existing (stock_id, year, week) combinations
                for row in existing_response.data:
                    key = (row['stock_id'], row['iso_year'], row['iso_week'])
                    existing_prices[key] = True
                
                #st.caption(f"   ✅ Found {len(existing_prices)} existing price records")
            
            # Check which combinations are missing
            missing_weeks = []
            total_checks = len(stock_ids) * len(all_weeks)
            
            #st.caption(f"   📊 Checking {total_checks} combinations ({len(stock_ids)} stocks × {len(all_weeks)} weeks)")
            
            for stock_id in stock_ids:
                ticker = stock_details.get(stock_id, 'Unknown')
                
                for year, week in all_weeks:
                    key = (stock_id, year, week)
                    if key not in existing_prices:
                        missing_weeks.append({
                            'stock_id': stock_id,
                            'ticker': ticker,
                            'year': year,
                            'week': week
                        })
            
            #st.caption(f"   ✅ Found {len(missing_weeks)} missing week prices to fetch")
            #st.caption(f"   📈 Includes: Transaction weeks + Last 52 weeks")
            
            return missing_weeks
            
        except Exception as e:
            st.error(f"❌ Error getting missing weeks: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return []

