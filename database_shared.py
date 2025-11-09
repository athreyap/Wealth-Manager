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
from datetime import datetime, timedelta
import pandas as pd
from dateutil import parser as dateutil_parser
import re
from collections import defaultdict
from pms_aif_calculator import PMS_AIF_Calculator

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
    
    def register_user(self, username: str, password: str, full_name: str, email: str = None, risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """Register new user with username (case-insensitive)"""
        try:
            password_hash = self.hash_password(password)
            
            # Store username in lowercase for case-insensitive login
            username_lower = username.lower()
            
            # If email is not provided or empty, use None (NULL in database)
            # Multiple users can have NULL email without violating unique constraint
            if not email or str(email).strip() == '':
                email_value = None
            else:
                email_value = email.strip()
            
            response = self.supabase.table('users').insert({
                'username': username_lower,
                'email': email_value,  # None if empty, allows multiple users without email
                'password_hash': password_hash,
                'full_name': full_name,
                'risk_tolerance': risk_tolerance,
                'investment_goals': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).execute()
            
            return {'success': True, 'user': response.data[0]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def login_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login user with username (case-insensitive)"""
        try:
            password_hash = self.hash_password(password)
            
            # Make username case-insensitive by converting to lowercase
            username_lower = username.lower()
            
            response = self.supabase.table('users').select('*').eq(
                'username', username_lower
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
    
    def get_user_pdfs(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all PDFs (shared across all users)"""
        try:
            # Fetch ALL PDFs regardless of user_id (shared library)
            response = self.supabase.table('user_pdfs').select('*').order('uploaded_at', desc=True).execute()
            
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
    
    def get_all_pdfs_text(self, user_id: str = None) -> str:
        """Get combined text from all PDFs (shared library) for AI context"""
        try:
            # Get ALL PDFs (shared across all users)
            pdfs = self.get_user_pdfs(user_id)
            if not pdfs:
                return ""
            
            combined_text = "\n\n--- SHARED PDF DOCUMENTS (ALL USERS) ---\n\n"
            for pdf in pdfs:
                uploader = f" (Uploaded by user)"  # Can add uploader name if needed
                combined_text += f"\n📄 {pdf['filename']}{uploader}:\n"
                combined_text += pdf['pdf_text'][:2000] + "...\n"  # Limit per PDF
            
            return combined_text
        except Exception as e:
            return ""
    
    # ========================================================================
    # CHAT HISTORY MANAGEMENT (User-Specific)
    # ========================================================================
    
    def save_chat_history(self, user_id: str, question: str, answer: str) -> bool:
        """Save a chat question/answer to database (user-specific)"""
        try:
            self.supabase.table('user_chat_history').insert({
                'user_id': user_id,
                'question': question,
                'answer': answer
            }).execute()
            return True
        except Exception as e:
            st.error(f"Error saving chat history: {str(e)}")
            return False
    
    def get_user_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a specific user (user-specific, NOT shared)"""
        try:
            response = self.supabase.table('user_chat_history').select('*').eq(
                'user_id', user_id
            ).order('created_at', desc=True).limit(limit).execute()
            
            return response.data
        except Exception as e:
            st.error(f"Error fetching chat history: {str(e)}")
            return []
    
    def delete_chat_history(self, user_id: str, chat_id: str = None) -> bool:
        """Delete chat history for a user (all or specific entry)"""
        try:
            if chat_id:
                # Delete specific chat entry
                self.supabase.table('user_chat_history').delete().eq('id', chat_id).eq('user_id', user_id).execute()
            else:
                # Delete all chat history for user
                self.supabase.table('user_chat_history').delete().eq('user_id', user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting chat history: {str(e)}")
            return False
    
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
            # Try to find existing by ticker only (more flexible)
            response = self.supabase.table('stock_master').select('*').eq(
                'ticker', ticker
            ).execute()
            
            if response.data and len(response.data) > 0:
                existing_stock = response.data[0]
                
                # Update stock info if we have better data
                update_data = {}
                if stock_name and stock_name != 'Unknown' and (not existing_stock.get('stock_name') or existing_stock.get('stock_name') == 'Unknown'):
                    update_data['stock_name'] = stock_name
                
                # For bonds, always set sector to "bond"
                if asset_type == 'bond':
                    if existing_stock.get('sector') != 'bond':
                        update_data['sector'] = 'bond'
                elif sector and sector != 'Unknown' and (not existing_stock.get('sector') or existing_stock.get('sector') == 'Unknown'):
                    update_data['sector'] = sector
                
                if update_data:
                    self.supabase.table('stock_master').update(update_data).eq('id', existing_stock['id']).execute()
                
                return existing_stock['id']
            
            # If no existing stock, try to fetch better info from external sources
            enhanced_info = self._fetch_stock_info(ticker, asset_type)
            
            # Use fetched info if available, otherwise use provided data
            final_stock_name = enhanced_info.get('stock_name') or stock_name or 'Unknown'
            
            # For bonds, always set sector to "bond"
            if asset_type == 'bond':
                final_sector = 'bond'
            else:
                final_sector = enhanced_info.get('sector') or sector or 'Unknown'
            
            # Create new
            insert_data = {
                'ticker': ticker,
                'stock_name': final_stock_name,
                'asset_type': asset_type,
                'sector': final_sector
            }
            
            response = self.supabase.table('stock_master').insert(insert_data).execute()
            
            if response.data:
                return response.data[0]['id']
            
            return None
            
        except Exception as e:
            st.error(f"Error creating stock: {str(e)}")
            return None
    
    def _fetch_stock_info(self, ticker: str, asset_type: str) -> Dict[str, Any]:
        """
        Fetch stock name and sector from external sources
        """
        try:
            if asset_type == 'stock':
                return self._fetch_stock_info_yfinance(ticker)
            elif asset_type == 'mutual_fund':
                return self._fetch_mf_info_mftool(ticker)
            else:
                return {}
        except Exception as e:
            # Silently fail - we'll use provided data
            return {}
    
    def _fetch_stock_info_yfinance(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info from yfinance"""
        try:
            import yfinance as yf
            
            # Try different ticker formats
            ticker_formats = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
            
            for ticker_format in ticker_formats:
                try:
                    stock = yf.Ticker(ticker_format)
                    info = stock.info
                    
                    if info and info.get('longName'):
                        return {
                            'stock_name': info.get('longName', ''),
                            'sector': info.get('sector', 'Unknown')
                        }
                except:
                    continue
            
            return {}
        except:
            return {}
    
    def _fetch_mf_info_mftool(self, ticker: str) -> Dict[str, Any]:
        """Fetch mutual fund info from mftool"""
        try:
            from mftool import Mftool
            mf = Mftool()
            
            # Get scheme info
            scheme_info = mf.get_scheme_info(ticker)
            
            if scheme_info:
                return {
                    'stock_name': scheme_info.get('schemeName', ''),
                    'sector': scheme_info.get('category', 'Unknown')
                }
            
            return {}
        except:
            return {}
    
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
    
    def update_stock_ticker(self, stock_id: str, new_ticker: str):
        """Normalize ticker symbol for a stock."""
        try:
            self.supabase.table('stock_master').update({
                'ticker': new_ticker,
                'last_updated': datetime.now().isoformat()
            }).eq('id', stock_id).execute()
        except Exception:
            pass
    
    def get_stock_last_updated(self, stock_id: str) -> Optional[str]:
        """Get last updated timestamp for a stock"""
        try:
            response = self.supabase.table('stock_master').select('last_updated').eq('id', stock_id).execute()
            if response.data:
                return response.data[0].get('last_updated')
            return None
        except Exception as e:
            return None
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile settings"""
        try:
            # Add updated timestamp
            profile_data['updated_at'] = datetime.now().isoformat()
            
            response = self.supabase.table('users').update(profile_data).eq('id', user_id).execute()
            
            if response.data:
                return {'success': True, 'user': response.data[0]}
            return {'success': False, 'error': 'No data updated'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with settings"""
        try:
            response = self.supabase.table('users').select('*').eq('id', user_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            return None
    
    def add_investment_goal(self, user_id: str, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Add investment goal to user profile"""
        try:
            # Get current user profile
            user_profile = self.get_user_profile(user_id)
            if not user_profile:
                return {'success': False, 'error': 'User not found'}
            
            # Get current goals
            current_goals = user_profile.get('investment_goals', [])
            
            # Add new goal with ID
            goal['id'] = f"goal_{int(datetime.now().timestamp())}"
            goal['created_at'] = datetime.now().isoformat()
            current_goals.append(goal)
            
            # Update user profile
            return self.update_user_profile(user_id, {'investment_goals': current_goals})
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_investment_goal(self, user_id: str, goal_id: str, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific investment goal"""
        try:
            # Get current user profile
            user_profile = self.get_user_profile(user_id)
            if not user_profile:
                return {'success': False, 'error': 'User not found'}
            
            # Get current goals
            current_goals = user_profile.get('investment_goals', [])
            
            # Find and update goal
            for i, goal in enumerate(current_goals):
                if goal.get('id') == goal_id:
                    goal_data['updated_at'] = datetime.now().isoformat()
                    current_goals[i].update(goal_data)
                    break
            else:
                return {'success': False, 'error': 'Goal not found'}
            
            # Update user profile
            return self.update_user_profile(user_id, {'investment_goals': current_goals})
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_investment_goal(self, user_id: str, goal_id: str) -> Dict[str, Any]:
        """Delete investment goal"""
        try:
            # Get current user profile
            user_profile = self.get_user_profile(user_id)
            if not user_profile:
                return {'success': False, 'error': 'User not found'}
            
            # Get current goals
            current_goals = user_profile.get('investment_goals', [])
            
            # Remove goal
            current_goals = [goal for goal in current_goals if goal.get('id') != goal_id]
            
            # Update user profile
            return self.update_user_profile(user_id, {'investment_goals': current_goals})
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
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
    
    def _normalize_transaction_date(self, value: Any) -> Optional[str]:
        """Normalize transaction date strings to YYYY-MM-DD."""
        if value is None or value == '':
            return None
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')

        text = str(value).strip()
        # Try parsing with dateutil, preferring day-first for Indian data
        for dayfirst in (True, False):
            try:
                parsed = dateutil_parser.parse(text, dayfirst=dayfirst, fuzzy=True)
                return parsed.strftime('%Y-%m-%d')
            except Exception:
                continue

        # As a fallback, strip out noisy characters and retry
        cleaned = re.sub(r'[^0-9A-Za-z:/\\ -]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        for dayfirst in (True, False):
            try:
                parsed = dateutil_parser.parse(cleaned, dayfirst=dayfirst, fuzzy=True)
                return parsed.strftime('%Y-%m-%d')
            except Exception:
                continue

        return None

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
            
            # Normalize and calculate week info from transaction date
            normalized_date = self._normalize_transaction_date(transaction_data['transaction_date'])
            if not normalized_date:
                return {'success': False, 'error': f"Invalid transaction date: {transaction_data['transaction_date']}"}
            transaction_data['transaction_date'] = normalized_date

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
    
    def recalculate_holdings(self, user_id: str, portfolio_id: Optional[str] = None) -> int:
        """
        Recalculate holdings from user_transactions
        Groups transactions by stock_id and calculates total quantity and average price
        """
        try:
            # Get all transactions for this user
            query = self.supabase.table('user_transactions').select('*').eq('user_id', user_id)
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            transactions = query.execute().data
            
            if not transactions:
                return 0
            
            # Group by stock_id and calculate holdings
            from collections import defaultdict
            holdings_calc = defaultdict(lambda: {'buy_qty': 0, 'sell_qty': 0, 'total_cost': 0, 'portfolio_id': None})
            
            for txn in transactions:
                stock_id = txn['stock_id']
                quantity = float(txn['quantity'])
                price = float(txn['price'])
                txn_type = txn['transaction_type'].lower()
                
                if txn_type == 'buy':
                    holdings_calc[stock_id]['buy_qty'] += quantity
                    holdings_calc[stock_id]['total_cost'] += quantity * price
                elif txn_type == 'sell':
                    holdings_calc[stock_id]['sell_qty'] += quantity
                
                holdings_calc[stock_id]['portfolio_id'] = txn['portfolio_id']
            
            # Upsert holdings
            updated_count = 0
            for stock_id, calc in holdings_calc.items():
                total_quantity = calc['buy_qty'] - calc['sell_qty']
                
                if total_quantity > 0:  # Only store holdings with positive quantity
                    average_price = calc['total_cost'] / calc['buy_qty'] if calc['buy_qty'] > 0 else 0
                    
                    # Upsert (update if exists, insert if not)
                    self.supabase.table('holdings').upsert({
                        'user_id': user_id,
                        'portfolio_id': calc['portfolio_id'],
                        'stock_id': stock_id,
                        'total_quantity': total_quantity,
                        'average_price': average_price
                    }, on_conflict='user_id,portfolio_id,stock_id').execute()
                    
                    updated_count += 1
                else:
                    # Quantity is 0 or negative, delete holding
                    self.supabase.table('holdings').delete().eq(
                        'user_id', user_id
                    ).eq('stock_id', stock_id).execute()
            
            print(f"[HOLDINGS] Recalculated {updated_count} holdings for user {user_id}")
            return updated_count
            
        except Exception as e:
            print(f"[HOLDINGS] Error recalculating: {e}")
            return 0
    
    def get_user_holdings(self, user_id: str, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user holdings with stock details (uses view)"""
        try:
            query = self.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
            
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            response = query.execute()
            holdings = response.data
            
            # Normalize live price information before channel lookup
            for holding in holdings:
                stock_id = holding['stock_id']

                try:
                    live_price = float(holding.get('live_price') or 0)
                except (TypeError, ValueError):
                    live_price = 0.0
                try:
                    current_price_val = float(holding.get('current_price') or 0)
                except (TypeError, ValueError):
                    current_price_val = 0.0
                if live_price > 0 and current_price_val == 0:
                    holding['current_price'] = live_price

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
            
            # Normalize live price information before channel lookup
            for holding in holdings:
                stock_id = holding['stock_id']

                try:
                    live_price = float(holding.get('live_price') or 0)
                except (TypeError, ValueError):
                    live_price = 0.0
                try:
                    current_price_val = float(holding.get('current_price') or 0)
                except (TypeError, ValueError):
                    current_price_val = 0.0
                if live_price > 0 and current_price_val == 0:
                    holding['current_price'] = live_price

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
    
    def fetch_and_store_missing_weekly_prices(
        self,
        user_id: str,
        missing_weeks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fetch and store weekly historical prices for missing combinations.
        Returns list of entries that could not be fetched.
        """
        if not missing_weeks:
            return []

        try:
            stock_ids = {entry.get('stock_id') for entry in missing_weeks if entry.get('stock_id')}
            if not stock_ids:
                return missing_weeks

            stock_details: Dict[str, Dict[str, Any]] = {}
            response = self.supabase.table('stock_master').select(
                'id, ticker, stock_name, asset_type'
            ).in_('id', list(stock_ids)).execute()
            for row in response.data or []:
                stock_details[row['id']] = row

            remaining: List[Dict[str, Any]] = []

            for stock_id in stock_ids:
                stock_info = stock_details.get(stock_id)
                if not stock_info:
                    remaining.extend([entry for entry in missing_weeks if entry.get('stock_id') == stock_id])
                    continue

                ticker = stock_info.get('ticker')
                asset_type = stock_info.get('asset_type') or 'stock'
                stock_name = stock_info.get('stock_name')

                if not ticker:
                    remaining.extend([entry for entry in missing_weeks if entry.get('stock_id') == stock_id])
                    continue

                stock_missing = [
                    entry for entry in missing_weeks if entry.get('stock_id') == stock_id
                ]
                if not stock_missing:
                    continue

                try:
                    leftovers = self._store_weekly_prices_bulk(
                        ticker,
                        {},
                        asset_type,
                        stock_name,
                        target_weeks=stock_missing,
                        user_id=user_id,
                    )
                    if leftovers:
                        leftover_set = {(int(year), int(week)) for year, week in leftovers}
                        for entry in stock_missing:
                            pair = (int(entry.get('year')), int(entry.get('week')))
                            if pair in leftover_set:
                                remaining.append(entry)
                except Exception as exc:
                    print(f"[WEEKLY_FETCH] Failed for {ticker}: {exc}")
                    remaining.extend([entry for entry in missing_weeks if entry.get('stock_id') == stock_id])

            return remaining
        except Exception as exc:
            print(f"[WEEKLY_FETCH] Error while fetching weekly prices: {exc}")
            return missing_weeks

    def get_channel_weekly_performance(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Calculate P&L % for each channel using the most recent 52 weeks of historical prices.
        Returns a DataFrame with columns: channel, investment_52w, current_value_52w, pnl_pct_52w.
        """
        try:
            holdings = self.get_user_holdings(user_id)
            if not holdings:
                return None

            channel_data: Dict[str, Dict[str, float]] = {}
            today = datetime.now().date()

            for holding in holdings:
                channel = holding.get('channel') or 'Unknown'
                channel = str(channel).strip() or 'Unknown'
                total_qty = float(holding.get('total_quantity') or 0)
                if total_qty <= 0:
                    continue

                stock_id = holding.get('stock_id')
                ticker = holding.get('ticker')
                if not stock_id or not ticker:
                    continue

                hist_response = (
                    self.supabase.table('historical_prices')
                    .select('price_date, price')
                    .eq('stock_id', stock_id)
                    .order('price_date', desc=True)
                    .limit(60)
                    .execute()
                )
                history = hist_response.data or []
                if not history:
                    continue

                history_sorted = sorted(history, key=lambda row: row['price_date'])
                latest_entry = history_sorted[-1]
                latest_price = float(latest_entry.get('price') or 0)
                if latest_price <= 0:
                    continue

                baseline_price = None
                for row in history_sorted:
                    price_date_str = row.get('price_date')
                    if not price_date_str:
                        continue
                    try:
                        price_date = datetime.strptime(price_date_str, '%Y-%m-%d').date()
                    except ValueError:
                        continue

                    if (today - price_date).days >= 365:
                        baseline_price = float(row.get('price') or 0)
                    else:
                        break

                if not baseline_price or baseline_price <= 0:
                    baseline_price = float(history_sorted[0].get('price') or 0)

                if baseline_price <= 0:
                    continue

                baseline_value = baseline_price * total_qty
                latest_value = latest_price * total_qty

                channel_entry = channel_data.setdefault(
                    channel,
                    {'investment': 0.0, 'current': 0.0}
                )
                channel_entry['investment'] += baseline_value
                channel_entry['current'] += latest_value

            rows = []
            for channel, values in channel_data.items():
                investment = values['investment']
                current_val = values['current']
                if investment > 0:
                    pnl_pct = (current_val - investment) / investment * 100
                elif current_val > 0:
                    pnl_pct = 100.0
                else:
                    pnl_pct = 0.0
                rows.append({
                    'channel': channel,
                    'investment_52w': investment,
                    'current_value_52w': current_val,
                    'pnl_pct_52w': pnl_pct,
                })

            if not rows:
                return None

            result = pd.DataFrame(rows)
            return result.sort_values('channel').reset_index(drop=True)
        except Exception as exc:
            print(f"[CHANNEL_52W] Error computing weekly performance: {exc}")
            return None

    def get_channel_weekly_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Build weekly P&L history per channel for the last ~52 weeks.
        Returns list of {channel, date, pnl_pct, investment, current_value}
        """
        try:
            holdings = self.get_user_holdings(user_id)
            if not holdings:
                return []

            channel_investments: Dict[str, float] = defaultdict(float)
            channel_values: Dict[str, Dict[datetime.date, float]] = defaultdict(lambda: defaultdict(float))

            for holding in holdings:
                channel = str(holding.get('channel') or 'Unknown').strip() or 'Unknown'
                quantity = float(holding.get('total_quantity') or 0)
                avg_price = float(holding.get('average_price') or 0)
                stock_id = holding.get('stock_id')

                if quantity <= 0 or not stock_id:
                    continue

                channel_investments[channel] += max(quantity * avg_price, 0.0)

                hist_resp = (
                    self.supabase.table('historical_prices')
                    .select('price_date, price')
                    .eq('stock_id', stock_id)
                    .order('price_date', desc=True)
                    .limit(65)
                    .execute()
                )
                rows = hist_resp.data or []
                for row in rows:
                    price_date = row.get('price_date')
                    price = row.get('price')
                    if not price_date or price is None:
                        continue
                    try:
                        dt = datetime.strptime(price_date, '%Y-%m-%d').date()
                    except ValueError:
                        continue
                    channel_values[channel][dt] += float(price) * quantity

            history_entries: List[Dict[str, Any]] = []
            for channel, date_map in channel_values.items():
                sorted_dates = sorted(date_map.keys())
                if len(sorted_dates) > 52:
                    sorted_dates = sorted_dates[-52:]

                for dt in sorted_dates:
                    current_value = date_map[dt]
                    investment = channel_investments.get(channel, 0.0)
                    pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0.0
                    history_entries.append({
                        'channel': channel,
                        'date': dt.strftime('%Y-%m-%d'),
                        'pnl_pct': pnl_pct,
                        'investment': investment,
                        'current_value': current_value,
                    })

            history_entries.sort(key=lambda entry: (entry['channel'], entry['date']))
            return history_entries
        except Exception as exc:
            print(f"[CHANNEL_52W] Error building weekly history: {exc}")
            return []
    
    def bulk_process_new_stocks_with_comprehensive_data(
        self,
        tickers: List[str],
        asset_types: Dict[str, str] = None,
        transactions: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Bulk process new stocks with comprehensive data fetching (info + prices + weekly data)
        Uses AI to fetch all data including prices for specific transaction dates
        
        Args:
            tickers: List of ticker symbols to process
            asset_types: Dict mapping ticker to asset_type (optional)
            transactions: List of original transactions with dates (optional)
        
        Returns:
            Dict mapping ticker to stock_id
        """
        print(f"\n[BULK_PROCESS] Starting for {len(tickers)} tickers: {tickers}")
        
        try:
            # Use bulk AI fetcher to get stock info and basic data
            from bulk_ai_fetcher import BulkAIFetcher
            bulk_fetcher = BulkAIFetcher()
            
            print(f"[BULK_PROCESS] BulkAIFetcher available: {bulk_fetcher.available}")
            
            if not bulk_fetcher.available:
                print(f"[BULK_PROCESS] Falling back to individual processing")
                return self._fallback_individual_stock_processing(tickers, asset_types)
            
            # Fetch stock info from AI (names, sectors)
            comprehensive_data = bulk_fetcher.fetch_bulk_comprehensive_data(tickers)
            
            stock_ids = {}
            
            for ticker in tickers:
                try:
                    # Get data for this ticker
                    ticker_data = comprehensive_data.get(ticker, {})
                    
                    # Determine asset type
                    asset_type = asset_types.get(ticker, 'stock') if asset_types else 'stock'
                    
                    # Extract stock info (AI provides name and sector)
                    stock_name = ticker_data.get('stock_name', 'Unknown')
                    sector = ticker_data.get('sector', 'Unknown')
                    ai_current_price = ticker_data.get('current_price', 0.0)
                    weekly_prices = ticker_data.get('weekly_prices', {})
                    transaction_prices = ticker_data.get('transaction_prices', {})
                    
                    # For current price, try multi-source fetching (yfinance → mftool → enhanced_fetcher)

                    live_price, resolved_ticker, price_source = self._get_live_price_multi_source(ticker, asset_type, stock_name)
                    if live_price <= 0 and ai_current_price:
                        current_price = ai_current_price
                        resolved_ticker = ticker
                        price_source = 'ai_bulk'
                    else:
                        current_price = live_price
                    
                    # Get or create stock in database
                    stock_id = self.get_or_create_stock(
                        ticker=ticker,
                        stock_name=stock_name,
                        asset_type=asset_type,
                        sector=sector
                    )
                    
                    if stock_id:
                        stock_ids[ticker] = stock_id
                        print(f"[BULK] Processing {ticker}: stock_id={stock_id}, current_price={current_price}, resolved_ticker={resolved_ticker}")
                        
                        if resolved_ticker != ticker and current_price > 0:
                            try:
                                self.update_stock_ticker(stock_id, resolved_ticker)
                                ticker = resolved_ticker
                            except Exception:
                                pass
                        
                        # Store current price if available (prioritizes API over AI)
                        if current_price > 0:
                            print(f"[BULK] Calling _store_current_price for {ticker}")
                            self._store_current_price(ticker, current_price, asset_type)
                        else:
                            print(f"[BULK] Skipping current price for {ticker}: price={current_price}")
                        
                        # Store weekly prices if available (fetches real prices from APIs)
                        print(f"[BULK] Calling _store_weekly_prices_bulk for {ticker}")
                        self._store_weekly_prices_bulk(ticker, weekly_prices, asset_type, stock_name)
                    else:
                        print(f"[BULK] No stock_id for {ticker}, skipping")
                    
                except Exception as e:
                    print(f"[BULK] ERROR processing {ticker}: {e}")
                    # Continue with other tickers if one fails
                    continue
            
            return stock_ids
            
        except Exception as e:
            # Fallback to individual processing
            return self._fallback_individual_stock_processing(tickers, asset_types)
    
    def _fallback_individual_stock_processing(
        self,
        tickers: List[str],
        asset_types: Dict[str, str] = None
    ) -> Dict[str, str]:
        """Fallback method for individual stock processing - also fetches prices!"""
        print(f"[FALLBACK] Processing {len(tickers)} tickers individually...")
        stock_ids = {}
        
        for ticker in tickers:
            try:
                asset_type = asset_types.get(ticker, 'stock') if asset_types else 'stock'
                print(f"[FALLBACK] Processing {ticker} ({asset_type})")
                
                # Get or create stock
                stock_id = self.get_or_create_stock(
                    ticker=ticker,
                    stock_name='Unknown',
                    asset_type=asset_type,
                    sector='Unknown'
                )
                
                if stock_id:
                    stock_ids[ticker] = stock_id
                    print(f"[FALLBACK] Stock created/found: {stock_id}")
                    
                    # Fetch and store current price
                    print(f"[FALLBACK] Fetching current price for {ticker}...")
                    current_price, resolved_ticker, price_source = self._get_live_price_multi_source(ticker, asset_type, None)
                    if current_price and current_price > 0:
                        if resolved_ticker != ticker:
                            try:
                                self.update_stock_ticker(stock_id, resolved_ticker)
                                ticker = resolved_ticker
                            except Exception:
                                pass
                        self._store_current_price(ticker, current_price, asset_type)
                    else:
                        print(f"[FALLBACK] No current price found for {ticker}")
                    
                    # Fetch and store 52-week historical data
                    print(f"[FALLBACK] Fetching historical data for {ticker}...")
                    self._store_weekly_prices_bulk(ticker, {}, asset_type, None)
                    
            except Exception as e:
                print(f"[FALLBACK] ERROR with {ticker}: {e}")
                continue
        
        print(f"[FALLBACK] Completed: {len(stock_ids)} stocks processed")
        return stock_ids
    
    def _get_live_price_multi_source(self, ticker: str, asset_type: str = 'stock', stock_name: str = None) -> Tuple[float, str, str]:
        """
        Get live current price from multiple sources
        Priority: yfinance → smart ticker mapping → mftool → enhanced_price_fetcher
        """
        try:
            import yfinance as yf
            
            # Smart ticker mapping for known issues
            ticker_corrections = {
                'OBEROI.NS': 'OBEROIRLTY.NS',
                'NIPPONAMC.NS': 'SILVERBEES.NS',
                'NIPPONAMC - NETFSILVER': 'SILVERBEES.NS',
            }
            
            # Apply correction if available
            corrected_ticker = ticker_corrections.get(ticker, ticker)
            
            # Try yfinance first (for stocks and ETFs)
            if asset_type in ['stock', 'etf']:
                ticker_formats = [corrected_ticker, ticker, f"{ticker}.NS", f"{ticker}.BO"]
                
                for tf in ticker_formats:
                    try:
                        stock = yf.Ticker(tf)
                        
                        hist = stock.history(period='1d')
                        price = float(hist['Close'].iloc[-1]) if not hist.empty else None
                        if (price is None or price <= 0):
                            fast_info = getattr(stock, "fast_info", {}) or {}
                            price = (
                                fast_info.get("lastPrice")
                                or fast_info.get("regularMarketPrice")
                                or fast_info.get("previousClose")
                            )
                        if (price is None or price <= 0):
                            info = getattr(stock, "info", {}) or {}
                            price = (
                                info.get('currentPrice') or 
                                info.get('regularMarketPrice') or 
                                info.get('navPrice') or
                                info.get('previousClose')
                            )
                        
                        if price and price > 0:
                            resolved = tf
                            source = 'yfinance_raw'
                            if tf.endswith('.BO'):
                                source = 'yfinance_bse'
                            elif tf.endswith('.NS'):
                                source = 'yfinance_nse'
                            return float(price), resolved, source
                    except:
                        continue
            
            # Try MFTool for mutual funds
            if asset_type in ['mutual_fund', 'etf']:
                try:
                    from mftool import Mftool
                    mf = Mftool()

                    # Attempt to resolve AMFI code using enhanced fetcher logic
                    resolved_code = None
                    try:
                        from enhanced_price_fetcher import EnhancedPriceFetcher
                        resolver = EnhancedPriceFetcher()
                        resolved_code = resolver._resolve_amfi_code(ticker, stock_name)
                    except Exception:
                        resolved_code = None
                    
                    possible_codes = []
                    if resolved_code:
                        possible_codes.append(resolved_code)
                    possible_codes.append(ticker)
                    
                    for code_candidate in possible_codes:
                        if not code_candidate:
                            continue
                        quote = mf.get_scheme_quote(code_candidate)
                        if quote and quote.get('nav'):
                            return float(quote.get('nav')), code_candidate, 'mftool'

                    # Try searching by name if ticker fails
                    if stock_name:
                        search = mf.search_by_scheme_name(stock_name)
                        if search:
                            for scheme_code in search.keys():
                                quote = mf.get_scheme_quote(scheme_code)
                                if quote and quote.get('nav'):
                                    return float(quote.get('nav')), scheme_code, 'mftool'
                except:
                    pass
            
            # Try enhanced_price_fetcher as fallback
            try:
                from enhanced_price_fetcher import EnhancedPriceFetcher
                fetcher = EnhancedPriceFetcher()
                price, source = fetcher.get_current_price(ticker, asset_type, stock_name)
                resolved_ticker = getattr(fetcher, "_last_resolved_ticker", ticker)
                if price and price > 0:
                    return float(price), resolved_ticker, source or 'enhanced_fetcher'
            except:
                pass
        
        except Exception:
            pass
        
        return 0.0, ticker, 'not_found'
    
    def _store_current_price(self, ticker: str, price: float, asset_type: str = 'stock'):
        """
        Store current price for a ticker by updating stock_master
        Uses AI-fetched prices which are already accurate and date-specific
        """
        try:
            from datetime import datetime
            
            print(f"[PRICE] Storing current price for {ticker}: Rs {price}")
            
            # AI has already fetched accurate current price
            # No need for additional API calls
            if price > 0:
                # Update live_price and last_updated in stock_master
                result = self.supabase.table('stock_master').update({
                    'live_price': price,
                    'last_updated': datetime.now().isoformat()
                }).eq('ticker', ticker).execute()
                
                if result.data:
                    print(f"[PRICE] SUCCESS: Updated live_price for {ticker}")
                else:
                    print(f"[PRICE] ERROR: No stock found with ticker {ticker}")
            else:
                print(f"[PRICE] ERROR: Invalid price: {price}")
                
        except Exception as e:
            print(f"[PRICE] ERROR: {e}")
            pass
    
    def _store_weekly_prices_bulk(
        self,
        ticker: str,
        weekly_prices: Dict[str, Any],
        asset_type: str = 'stock',
        stock_name: str = None,
        target_weeks: Optional[List[Dict[str, int]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        """
        Store weekly historical prices for a ticker
        Multi-source: yfinance → mftool → enhanced_price_fetcher
        """
        try:
            from datetime import datetime, timedelta
            import yfinance as yf
            from enhanced_price_fetcher import EnhancedPriceFetcher
            
            print(f"\n[HIST] Fetching weekly data for {ticker} ({stock_name})")
            
            # Get stock_id and stock info
            stock_response = self.supabase.table('stock_master').select('id, stock_name').eq('ticker', ticker).execute()
            if not stock_response.data:
                print(f"[HIST] ERROR Stock not found in stock_master: {ticker}")
                return []
            
            stock_id = stock_response.data[0]['id']
            db_stock_name = stock_response.data[0].get('stock_name') or stock_name
            print(f"[HIST] Stock ID: {stock_id}")
            
            # Check if we already have historical data (shared across users)
            existing_hist = self.supabase.table('historical_prices').select('id, iso_year, iso_week').eq('stock_id', stock_id).execute()
            existing_pairs = {(row['iso_year'], row['iso_week']) for row in existing_hist.data or []}
            existing_count = len(existing_hist.data) if existing_hist.data else 0
            print(f"[HIST] Existing historical records: {existing_count}")
            
            required_pairs: Optional[set] = None
            if target_weeks:
                required_pairs = {
                    (int(entry.get('year')), int(entry.get('week')))
                    for entry in target_weeks
                    if entry.get('year') is not None and entry.get('week') is not None
                }
                required_pairs -= existing_pairs
                if not required_pairs:
                    print("[HIST] All requested weeks already cached")
                    return []
            elif existing_count >= 40:
                # Already has substantial data, skip to avoid duplicates
                print(f"[HIST] SKIP: Already has {existing_count} records (>=40)")
                return []
            
            prices_to_insert = []
            
            # Smart ticker mapping for known issues
            ticker_corrections = {
                'OBEROI.NS': 'OBEROIRLTY.NS',
                'NIPPONAMC.NS': 'SILVERBEES.NS',
                'NIPPONAMC - NETFSILVER': 'SILVERBEES.NS',
            }
            
            corrected_ticker = ticker_corrections.get(ticker, ticker)
            print(f"[HIST] Using ticker: {corrected_ticker} (original: {ticker})")

            fetcher = EnhancedPriceFetcher()
            alias_candidates = fetcher._generate_stock_aliases(corrected_ticker)
            if str(ticker).strip() not in alias_candidates:
                alias_candidates.append(str(ticker).strip())
            ticker_formats = fetcher._expand_symbol_variants(alias_candidates or [corrected_ticker])

            # SOURCE 0: Mutual fund NAV history via AMFI/mftool
            if asset_type in ['mutual_fund', 'etf']:
                try:
                    from mftool import Mftool
                    mf = Mftool()
                    scheme_code = fetcher._resolve_amfi_code(ticker, stock_name) or ticker
                    nav_history = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

                    if nav_history is not None and not nav_history.empty:
                        df_nav = nav_history.copy()
                        df_nav.index.name = 'date'
                        df_nav.reset_index(inplace=True)
                        df_nav['date'] = pd.to_datetime(df_nav['date'], format='%d-%m-%Y', errors='coerce')
                        df_nav = df_nav.dropna(subset=['date'])
                        df_nav['iso_year'] = df_nav['date'].dt.isocalendar().year.astype(int)
                        df_nav['iso_week'] = df_nav['date'].dt.isocalendar().week.astype(int)
                        df_nav['nav'] = pd.to_numeric(df_nav['nav'], errors='coerce')
                        df_nav = df_nav.dropna(subset=['nav'])
                        df_nav = df_nav.sort_values('date')

                        if required_pairs:
                            required_df = df_nav[df_nav.apply(
                                lambda row: (int(row['iso_year']), int(row['iso_week'])) in required_pairs, axis=1)]
                        else:
                            cutoff_date = datetime.now() - timedelta(weeks=60)
                            required_df = df_nav[df_nav['date'] >= cutoff_date]

                        if not required_df.empty:
                            df_weekly = required_df.groupby(['iso_year', 'iso_week']).tail(1)
                            for _, row in df_weekly.iterrows():
                                iso_pair = (int(row['iso_year']), int(row['iso_week']))
                                if required_pairs is not None and iso_pair not in required_pairs:
                                    continue

                                prices_to_insert.append({
                                    'stock_id': stock_id,
                                    'price_date': row['date'].strftime('%Y-%m-%d'),
                                    'price': float(row['nav']),
                                    'source': 'amfi_nav',
                                    'iso_year': int(row['iso_year']),
                                    'iso_week': int(row['iso_week']),
                                })
                                if required_pairs is not None and iso_pair in required_pairs:
                                    required_pairs.discard(iso_pair)

                            if prices_to_insert:
                                print(f"[HIST] AMFI: Storing {len(prices_to_insert)} NAV rows for {ticker}")
                                self.save_historical_prices_bulk(prices_to_insert)
                                if required_pairs is None or not required_pairs:
                                    return []
                                prices_to_insert = []
                    else:
                        print(f"[HIST] AMFI: No NAV history for {ticker}")
                except Exception as e:
                    print(f"[HIST] AMFI NAV fetch failed for {ticker}: {e}")

            # SOURCE 1: Try yfinance first
            print(f"[HIST] SOURCE 1: Trying yfinance...")
            try:
                hist_data = None
                range_start = None
                range_end = None
                if required_pairs:
                    earliest_year, earliest_week = min(required_pairs)
                    latest_year, latest_week = max(required_pairs)
                    range_start = datetime.fromisocalendar(earliest_year, earliest_week, 1) - timedelta(days=3)
                    range_end = datetime.fromisocalendar(latest_year, latest_week, 7) + timedelta(days=3)
                    print(f"[HIST] Targeted ISO span: {earliest_year}-W{earliest_week:02d} → {latest_year}-W{latest_week:02d}")
                
                for tf in ticker_formats:
                    try:
                        print(f"[HIST]   Trying format: {tf}")
                        stock = yf.Ticker(tf)
                        if range_start and range_end:
                            hist_data = stock.history(
                                start=range_start.strftime('%Y-%m-%d'),
                                end=range_end.strftime('%Y-%m-%d'),
                                interval="1wk",
                            )
                        else:
                            hist_data = stock.history(period="1y", interval="1wk")
                        if not hist_data.empty:
                            print(f"[HIST]   SUCCESS with {tf}: {len(hist_data)} weeks")
                            break
                    except Exception as e:
                        print(f"[HIST]   FAILED {tf}: {str(e)[:50]}")
                        continue
                
                if hist_data is not None and not hist_data.empty:
                    # Got real data from yfinance
                    print(f"[HIST] Processing {len(hist_data)} weeks of data...")
                    for date_index, row in hist_data.iterrows():
                        try:
                            price_date = date_index.strftime('%Y-%m-%d')
                            price = float(row['Close'])
                            
                            date_obj = date_index.to_pydatetime()
                            iso_calendar = date_obj.isocalendar()
                            iso_pair = (iso_calendar[0], iso_calendar[1])
                            
                            if required_pairs is not None and iso_pair not in required_pairs:
                                continue
                            
                            prices_to_insert.append({
                                'stock_id': stock_id,
                                'price_date': price_date,
                                'price': price,
                                'source': 'yfinance_weekly',
                                'iso_year': iso_calendar[0],
                                'iso_week': iso_calendar[1]
                            })
                            
                            if required_pairs is not None and iso_pair in required_pairs:
                                required_pairs.discard(iso_pair)
                        except:
                            continue
                    
                    if prices_to_insert:
                        print(f"[HIST] Storing {len(prices_to_insert)} records to database...")
                        self.save_historical_prices_bulk(prices_to_insert)
                        print(f"[HIST] SUCCESS: Stored {len(prices_to_insert)} historical prices for {ticker}")
                        if required_pairs is None or not required_pairs:
                            return []
                        prices_to_insert = []
                    else:
                        print(f"[HIST] ERROR: No valid prices to insert")
                else:
                    print(f"[HIST] ERROR: No data from yfinance")
            except Exception as e:
                print(f"[HIST] ERROR yfinance: {e}")
                pass

            # PMS/AIF weekly values using CAGR-based calculator
            if asset_type in ['pms', 'aif'] and user_id:
                try:
                    txn_resp = (
                        self.supabase.table('user_transactions')
                        .select('transaction_date', 'transaction_type', 'quantity', 'price')
                        .eq('user_id', user_id)
                        .eq('stock_id', stock_id)
                        .order('transaction_date', desc=False)
                        .execute()
                    )

                    transactions = txn_resp.data or []
                    buy_transactions = [
                        txn for txn in transactions
                        if str(txn.get('transaction_type', '')).lower() == 'buy'
                    ] or transactions

                    if buy_transactions:
                        first_txn = buy_transactions[0]
                        quantity = float(first_txn.get('quantity') or 0)
                        unit_price = float(first_txn.get('price') or 0)
                        investment_date = first_txn.get('transaction_date')
                        if quantity > 0 and unit_price > 0 and investment_date:
                            investment_amount = quantity * unit_price
                            calc = PMS_AIF_Calculator()
                            calc_result = calc.calculate_pms_aif_value(
                                ticker,
                                investment_date,
                                investment_amount,
                                is_aif=(asset_type == 'aif')
                            )
                            weekly_values = calc_result.get('weekly_values') or []
                            for entry in weekly_values:
                                price_date = entry.get('price_date')
                                total_value = entry.get('price')
                                if not price_date or total_value is None:
                                    continue
                                try:
                                    price_dt = datetime.strptime(price_date, '%Y-%m-%d')
                                except ValueError:
                                    continue
                                iso_calendar = price_dt.isocalendar()
                                iso_pair = (iso_calendar[0], iso_calendar[1])
                                if required_pairs is not None and iso_pair not in required_pairs:
                                    continue
                                nav_per_unit = float(total_value) / quantity if quantity > 0 else float(total_value)
                                prices_to_insert.append({
                                    'stock_id': stock_id,
                                    'price_date': price_date,
                                    'price': nav_per_unit,
                                    'source': 'pms_cagr',
                                    'iso_year': iso_calendar[0],
                                    'iso_week': iso_calendar[1],
                                })
                                if required_pairs is not None and iso_pair in required_pairs:
                                    required_pairs.discard(iso_pair)

                            if prices_to_insert:
                                print(f"[HIST] PMS/AIF: Stored {len(prices_to_insert)} CAGR rows for {ticker}")
                                self.save_historical_prices_bulk(prices_to_insert)
                                if required_pairs is None or not required_pairs:
                                    return []
                                prices_to_insert = []
                except Exception as exc:
                    print(f"[HIST] PMS/AIF weekly generation failed for {ticker}: {exc}")
                    pass
            
            # SOURCE 2: Try MFTool for mutual funds/ETFs
            if asset_type in ['mutual_fund', 'etf'] and db_stock_name:
                try:
                    from mftool import Mftool
                    mf = Mftool()
                    
                    # Search by stock name
                    search_results = mf.search_by_scheme_name(db_stock_name)
                    if search_results:
                        scheme_code = list(search_results.keys())[0]
                        # MFTool doesn't provide historical, skip
                except:
                    pass
            
            # SOURCE 3: Try enhanced_price_fetcher
            if asset_type == 'bond':
                uppercase_ticker = str(ticker or '').upper()
                if uppercase_ticker.startswith('SGB'):
                    try:
                        if required_pairs and required_pairs:
                            earliest_year, earliest_week = min(required_pairs)
                            latest_year, latest_week = max(required_pairs)
                            gold_start = datetime.fromisocalendar(earliest_year, earliest_week, 1)
                            gold_end = datetime.fromisocalendar(latest_year, latest_week, 7)
                        else:
                            gold_end = datetime.now()
                            gold_start = gold_end - timedelta(weeks=60)

                        gold_hist = yf.Ticker("XAUUSD=X").history(
                            start=(gold_start - timedelta(days=3)).strftime('%Y-%m-%d'),
                            end=(gold_end + timedelta(days=3)).strftime('%Y-%m-%d'),
                            interval="1wk",
                        )
                        fx_hist = yf.Ticker("USDINR=X").history(
                            start=(gold_start - timedelta(days=3)).strftime('%Y-%m-%d'),
                            end=(gold_end + timedelta(days=3)).strftime('%Y-%m-%d'),
                            interval="1wk",
                        )
                        if not gold_hist.empty and not fx_hist.empty:
                            merged = pd.DataFrame({
                                'gold': gold_hist['Close'],
                                'usdinr': fx_hist['Close'],
                            }).dropna()
                            for price_date, row in merged.iterrows():
                                price_dt = price_date.to_pydatetime()
                                iso_calendar = price_dt.isocalendar()
                                iso_pair = (iso_calendar[0], iso_calendar[1])
                                if required_pairs is not None and iso_pair not in required_pairs:
                                    continue
                                usd_price = float(row['gold'])
                                usd_inr = float(row['usdinr'])
                                if usd_price <= 0 or usd_inr <= 0:
                                    continue
                                price_inr_per_gram = (usd_price * usd_inr) / 31.1035
                                price_with_premium = price_inr_per_gram * 1.04
                                prices_to_insert.append({
                                    'stock_id': stock_id,
                                    'price_date': price_dt.strftime('%Y-%m-%d'),
                                    'price': price_with_premium,
                                    'source': 'gold_reference',
                                    'iso_year': iso_calendar[0],
                                    'iso_week': iso_calendar[1],
                                })
                                if required_pairs is not None and iso_pair in required_pairs:
                                    required_pairs.discard(iso_pair)

                            if prices_to_insert:
                                print(f"[HIST] GOLD: Stored {len(prices_to_insert)} inferred SGB prices for {ticker}")
                                self.save_historical_prices_bulk(prices_to_insert)
                                if required_pairs is None or not required_pairs:
                                    return []
                                prices_to_insert = []
                    except Exception as exc:
                        print(f"[HIST] Gold history fallback failed for {ticker}: {exc}")

            # SOURCE 3: Try enhanced_price_fetcher
            try:
                from datetime import timedelta
                
                # Calculate date range (52 weeks back from today)
                if required_pairs:
                    earliest_year, earliest_week = min(required_pairs)
                    latest_year, latest_week = max(required_pairs)
                    start_date = datetime.fromisocalendar(earliest_year, earliest_week, 1)
                    end_date = datetime.fromisocalendar(latest_year, latest_week, 7)
                else:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(weeks=52)
                
                # Fetch historical prices
                historical_data = fetcher.get_historical_prices(
                    ticker=ticker,
                    asset_type=asset_type,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    fund_name=db_stock_name
                )
                
                if historical_data and len(historical_data) > 0:
                    # Process returned data
                    for item in historical_data:
                        try:
                            date_str = item.get('date')
                            price = item.get('price')
                            
                            if date_str and price:
                                price_date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                iso_calendar = price_date_obj.isocalendar()
                                iso_pair = (iso_calendar[0], iso_calendar[1])
                                if required_pairs is not None and iso_pair not in required_pairs:
                                    continue
                                
                                prices_to_insert.append({
                                    'stock_id': stock_id,
                                    'price_date': date_str,
                                    'price': float(price),
                                    'source': 'enhanced_fetcher',
                                    'iso_year': iso_calendar[0],
                                    'iso_week': iso_calendar[1]
                                })
                                
                                if required_pairs is not None and iso_pair in required_pairs:
                                    required_pairs.discard(iso_pair)
                        except:
                            continue
                    
                    if prices_to_insert:
                        self.save_historical_prices_bulk(prices_to_insert)
                        if required_pairs is None or not required_pairs:
                            return []
                        prices_to_insert = []
            except:
                pass
            
            if required_pairs:
                return list(required_pairs)
            return []

        except Exception as e:
            # Silent failure - non-critical
            return list(required_pairs) if target_weeks and required_pairs else []

