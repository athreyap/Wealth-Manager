"""
Database Manager with Shared Architecture
- stock_master: Shared across all users
- historical_prices: Shared across all users
- user_transactions: User-specific, links to shared tables
"""

import streamlit as st
from supabase import create_client, Client
from typing import Optional, Dict, List, Any, Tuple, Callable
import hashlib
from datetime import datetime, timedelta
import pandas as pd
from dateutil import parser as dateutil_parser
import re
from collections import defaultdict
from pms_aif_calculator import PMS_AIF_Calculator
import time
import httpx


def _normalize_stockmaster_ticker(value: str) -> str:
    """Normalize ticker strings before storing / searching in stock_master."""
    if not value:
        return value
    text = str(value).strip()
    if ':' in text:
        parts = [segment for segment in re.split(r'[:/]', text) if segment]
        if parts:
            text = parts[-1].strip()
    text = text.replace('\u00a0', ' ')
    text = ''.join(text.split())  # remove whitespace
    text = text.lstrip('$')
    text = re.sub(r'(?:[-_.]?)(T0|T1|BL)(?=\.|$)', '', text, flags=re.IGNORECASE)
    text = text.replace('-', '')
    if text.endswith('.0') and text.replace('.0', '').isdigit():
        text = text[:-2]
    return text.upper()


def _tx_normalize_ticker(raw: Any) -> str:
    """Normalize ticker values (shared with transaction pipeline).
    CRITICAL: Preserves .NS and .BO suffixes for stocks to distinguish exchanges.
    """
    if raw is None:
        return ''
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        text = f"{raw}"
    else:
        text = str(raw)

    if not text:
        return ''

    text = text.strip()
    if ':' in text:
        parts = [segment for segment in re.split(r'[:/]', text) if segment]
        if parts:
            text = parts[-1].strip()
    text = text.replace('\u00a0', ' ')
    
    # CRITICAL: Preserve .NS and .BO suffixes for stocks (they distinguish exchanges)
    has_ns_suffix = text.upper().endswith('.NS')
    has_bo_suffix = text.upper().endswith('.BO')
    exchange_suffix = '.NS' if has_ns_suffix else ('.BO' if has_bo_suffix else '')
    
    # Remove suffix temporarily for normalization
    if exchange_suffix:
        text = text[:-len(exchange_suffix)]
    
    text = ''.join(text.split())  # remove whitespace
    text = text.lstrip('$')
    text = re.sub(r'^(NSE|BSE|NSI|BOM)(EQ|BE|BZ|XT|XD|P|W|Z|T0|T1)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:[-_.]?)(EQ|BE|BZ|XT|XD|P|W|Z|T0|T1)$', '', text, flags=re.IGNORECASE)
    if text.endswith('.0') and text.replace('.0', '').isdigit():
        text = text[:-2]
    text = text.replace('-', '')
    
    # Restore exchange suffix if it was present (for stocks)
    if exchange_suffix and not text.isdigit():
        text = text.upper() + exchange_suffix
        return text
    
    if text.isdigit():
        return text
    return text.upper()

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
    
    def _retry_db_operation(self, operation: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Any:
        """
        Retry a database operation with exponential backoff.
        
        Args:
            operation: A callable that performs the database operation
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        
        Returns:
            The result of the operation
        
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return operation()
            except (httpx.ReadError, httpx.ConnectError, OSError) as e:
                # These are transient network errors that should be retried
                last_exception = e
                error_code = getattr(e, 'errno', None)
                error_msg = str(e)
                
                # Check if it's a retryable error
                is_retryable = (
                    isinstance(e, (httpx.ReadError, httpx.ConnectError)) or
                    (isinstance(e, OSError) and error_code == 11) or  # EAGAIN: Resource temporarily unavailable
                    'temporarily unavailable' in error_msg.lower() or
                    'connection' in error_msg.lower()
                )
                
                if not is_retryable or attempt == max_retries - 1:
                    # Not retryable or last attempt, raise immediately
                    raise
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                print(f"[DB_RETRY] ⚠️ Database operation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                print(f"[DB_RETRY]    Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            except Exception as e:
                # Check if it's a Supabase API error (502, 500, etc.) or Cloudflare error
                error_msg = str(e)
                
                # Try to extract error code from various exception formats
                error_code = ''
                if hasattr(e, 'message'):
                    error_dict = e.message if isinstance(e.message, dict) else {}
                    error_code = str(error_dict.get('code', '')) if isinstance(error_dict, dict) else ''
                elif hasattr(e, 'code'):
                    error_code = str(e.code)
                elif hasattr(e, 'args') and e.args:
                    # Check if error code is in args
                    for arg in e.args:
                        if isinstance(arg, dict):
                            error_code = str(arg.get('code', ''))
                            break
                
                # Check for retryable Supabase/Cloudflare errors
                is_retryable_api_error = (
                    error_code in ['502', '500', '503', '504', '502', '500'] or  # HTTP error codes
                    'JSON could not be generated' in error_msg or  # Cloudflare HTML response
                    'Internal server error' in error_msg or
                    'cloudflare' in error_msg.lower() or
                    '500: Internal server error' in error_msg or
                    '<!DOCTYPE html>' in error_msg or  # HTML error page
                    'Error code 500' in error_msg
                )
                
                if is_retryable_api_error and attempt < max_retries - 1:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    print(f"[DB_RETRY] ⚠️ Supabase API error (attempt {attempt + 1}/{max_retries}): {error_code or 'Unknown'}")
                    print(f"[DB_RETRY]    Error: {error_msg[:200]}...")
                    print(f"[DB_RETRY]    Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                
                # Non-retryable errors, raise immediately
                raise
        
        # If we exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception
    
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
        sector: str = None,
        resolved_ticker: str = None  # If ticker was resolved from name, use this instead
    ) -> Optional[str]:
        """
        Get existing stock or create new one in stock_master
        Returns stock_id (UUID)
        CRITICAL: Always uses authoritative name from yfinance/AMFI/mftool to ensure ticker uniqueness
        """
        try:
            # Use resolved_ticker if provided (from name-based resolution), otherwise use original ticker
            actual_ticker = resolved_ticker if resolved_ticker else ticker
            normalised_ticker = _tx_normalize_ticker(actual_ticker)
            
            # CRITICAL: Fetch authoritative name from external sources FIRST
            # This ensures we always use the correct name from yfinance/AMFI/mftool
            # Pass stock_name for better AMFI mapping (uses both ticker AND name)
            # Use actual_ticker (resolved if provided) for fetching info
            print(f"[STOCK_CREATE] 🔍 Fetching stock info for ticker: {actual_ticker} (asset_type: {asset_type}, provided_name: {stock_name})")
            enhanced_info = self._fetch_stock_info(actual_ticker, asset_type, stock_name)
            authoritative_name = enhanced_info.get('stock_name', '').strip()
            if authoritative_name:
                print(f"[STOCK_CREATE] ✅ Got authoritative name: '{authoritative_name}' (source: {enhanced_info.get('source', 'unknown')})")
            
            # Use authoritative name if available, otherwise fall back to provided name
            # CRITICAL: Don't use provided stock_name if it looks like a channel/filename
            # BUT: Allow valid ticker codes (SGB codes, all-caps alphanumeric codes, etc.)
            if not authoritative_name and stock_name:
                stock_name_lower = stock_name.lower().strip()
                stock_name_upper = stock_name.upper().strip()
                
                # Check if it's a valid ticker code pattern (SGB codes, all-caps alphanumeric, bond codes, etc.)
                # Bond codes may contain special characters like %, -, ., etc.
                # Stock tickers can be 3-4 characters (BHEL, TCS, HDFC, etc.)
                has_letters = any(c.isalpha() for c in stock_name_upper)
                has_digits = any(c.isdigit() for c in stock_name_upper)
                is_valid_ticker_code = (
                    stock_name_upper.startswith('SGB') or  # Sovereign Gold Bond codes (SGBJUN31I, SGBFEB32IV)
                    (stock_name_upper.isupper() and stock_name_upper.isalnum() and len(stock_name_upper) >= 3) or  # All-caps alphanumeric codes (BHEL, TATAGOLD, etc.) - allow 3+ chars
                    (stock_name_upper.isupper() and has_digits) or  # Contains digits (likely a code)
                    (stock_name_upper.isupper() and has_letters and (has_digits or '%' in stock_name_upper or 'BOND' in stock_name_upper))  # Bond codes with special chars (2.50%GOLDBONDS2031SR-I, etc.)
                )
                
                # Only reject if it's clearly a channel/filename AND not a valid ticker code
                is_invalid_name = (
                    not is_valid_ticker_code and (
                        len(stock_name_lower) < 10 and ' ' not in stock_name_lower or
                        stock_name_lower in ['pornima', 'zerodha', 'groww', 'paytm', 'upstox', 'angel', 'icici', 'hdfc', 'sbi'] or
                        (stock_name_lower.islower() and len(stock_name_lower.split()) == 1)
                    )
                )
                if is_invalid_name:
                    print(f"[STOCK_CREATE] ⚠️ Ignoring invalid stock_name '{stock_name}' (looks like channel/filename), using 'Unknown'")
                    stock_name = None
            
            final_stock_name = authoritative_name if authoritative_name else (stock_name or 'Unknown').strip()
            
            # If no existing stock, try to find by ticker only - wrap in retry logic
            def _find_by_ticker():
                return self.supabase.table('stock_master').select('*').eq(
                    'ticker', normalised_ticker
                ).execute()
            
            response = self._retry_db_operation(_find_by_ticker, max_retries=3, base_delay=1.0)
            
            if response.data and len(response.data) > 0:
                # Stock with this ticker exists - ensure it has the correct authoritative name
                existing_stock = response.data[0]
                existing_name = (existing_stock.get('stock_name') or '').strip()
                
                # ALWAYS update with authoritative name if we fetched it and it's different
                # This ensures all stocks with the same ticker have the same name
                # CRITICAL: Also update if existing name looks like a channel/filename (e.g., "pornima")
                update_data = {}
                
                # Check if existing name looks invalid (channel/filename pattern)
                existing_name_lower = existing_name.lower().strip()
                is_invalid_existing_name = (
                    len(existing_name_lower) < 10 and ' ' not in existing_name_lower or
                    existing_name_lower in ['pornima', 'zerodha', 'groww', 'paytm', 'upstox', 'angel', 'icici', 'hdfc', 'sbi'] or
                    (existing_name_lower.islower() and len(existing_name_lower.split()) == 1)
                )
                
                # Always update if we have authoritative name AND (it differs OR existing name is invalid)
                if authoritative_name:
                    if authoritative_name != existing_name or is_invalid_existing_name:
                        update_data['stock_name'] = authoritative_name
                        reason = "invalid existing name (channel/filename)" if is_invalid_existing_name else "name differs"
                        print(f"[STOCK_UPDATE] Updating {normalised_ticker}: '{existing_name}' → '{authoritative_name}' ({reason}, from {enhanced_info.get('source', 'external')})")
                elif is_invalid_existing_name:
                    # Existing name is invalid but we couldn't fetch authoritative name - try harder
                    print(f"[STOCK_UPDATE] ⚠️ Existing name '{existing_name}' is invalid but couldn't fetch authoritative name for {normalised_ticker}")
                
                # Update ticker if needed (normalization)
                if existing_stock.get('ticker') != normalised_ticker:
                    update_data['ticker'] = normalised_ticker
                
                # For bonds, always set sector to "bond"
                if asset_type == 'bond':
                    if existing_stock.get('sector') != 'bond':
                        update_data['sector'] = 'bond'
                elif enhanced_info.get('sector') and enhanced_info.get('sector') != 'Unknown':
                    # Use authoritative sector if available
                    if existing_stock.get('sector') != enhanced_info.get('sector'):
                        update_data['sector'] = enhanced_info.get('sector')
                elif sector and sector != 'Unknown' and (not existing_stock.get('sector') or existing_stock.get('sector') == 'Unknown'):
                    update_data['sector'] = sector
                
                # Update asset_type if it's different (shouldn't happen, but fix if it does)
                if existing_stock.get('asset_type') != asset_type:
                    update_data['asset_type'] = asset_type
                
                if update_data:
                    def _update_stock():
                        return self.supabase.table('stock_master').update(update_data).eq('id', existing_stock['id']).execute()
                    self._retry_db_operation(_update_stock, max_retries=3, base_delay=1.0)
                
                return existing_stock['id']
            
            # If no existing stock, create new one with authoritative name
            # For bonds, always set sector to "bond"
            if asset_type == 'bond':
                final_sector = 'bond'
            else:
                final_sector = enhanced_info.get('sector') or sector or 'Unknown'
            
            # Create new - wrap in retry logic for transient errors
            insert_data = {
                'ticker': normalised_ticker,
                'stock_name': final_stock_name,
                'asset_type': asset_type,
                'sector': final_sector
            }
            
            def _insert_stock():
                return self.supabase.table('stock_master').insert(insert_data).execute()
            
            response = self._retry_db_operation(_insert_stock, max_retries=3, base_delay=1.0)
            
            if response.data:
                return response.data[0]['id']
            
            return None
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a retryable error that we should handle gracefully
            if '502' in error_msg or '500' in error_msg or 'JSON could not be generated' in error_msg:
                print(f"[DB_RETRY] ⚠️ Failed to create stock after retries: {error_msg[:200]}")
                st.warning(f"⚠️ Temporary database error. Please try again in a moment.")
            else:
                st.error(f"Error creating stock: {error_msg[:200]}")
            return None
    
    def _fetch_stock_info(self, ticker: str, asset_type: str, stock_name: str = None) -> Dict[str, Any]:
        """
        Fetch stock name and sector from external sources
        """
        try:
            if asset_type == 'stock':
                return self._fetch_stock_info_yfinance(ticker)
            elif asset_type == 'mutual_fund':
                # Pass stock_name for better AMFI mapping (uses both ticker AND name)
                return self._fetch_mf_info_mftool(ticker, stock_name)
            else:
                return {}
        except Exception as e:
            # Silently fail - we'll use provided data
            return {}
    
    def _extract_base_symbol(self, ticker: Optional[str]) -> str:
        """Return canonical base symbol (strip exchange suffix, trade flags, case)."""
        if not ticker:
            return ""
        normalized = _normalize_stockmaster_ticker(ticker)
        if normalized.endswith('.NS') or normalized.endswith('.BO'):
            return normalized[:-3]
        return normalized

    def _sector_from_asset_type(self, asset_type: Optional[str]) -> str:
        """Fallback sector label derived from asset type."""
        if not asset_type:
            return 'Unknown'
        mapping = {
            'stock': 'Equity',
            'equity': 'Equity',
            'mutual_fund': 'Mutual Fund',
            'mf': 'Mutual Fund',
            'pms': 'PMS / AIF',
            'aif': 'PMS / AIF',
            'bond': 'Debt',
            'debt': 'Debt',
            'etf': 'ETF',
            'index_fund': 'ETF',
            'cash': 'Cash',
        }
        return mapping.get(asset_type.lower(), 'Unknown')

    def _resolve_sector_for_stock(self, stock_id: str, portfolio_id: Optional[str] = None) -> str:
        """
        Resolve sector for a stock, collapsing tickers that only differ by exchange suffix.
        Falls back to asset type if no definitive sector is found.
        """
        try:
            response = (
                self.supabase.table('stock_master')
                .select('ticker, sector, asset_type')
                .eq('id', stock_id)
                .limit(1)
                .execute()
            )
            stock = (response.data or [{}])[0]
            sector_value = (stock.get('sector') or '').strip()
            if sector_value and sector_value.lower() != 'unknown':
                return sector_value

            base_symbol = self._extract_base_symbol(stock.get('ticker'))
            if base_symbol:
                try:
                    alt_response = (
                        self.supabase.table('stock_master')
                        .select('sector')
                        .ilike('ticker', f"%{base_symbol}%")
                        .execute()
                    )
                    for candidate in alt_response.data or []:
                        candidate_sector = (candidate.get('sector') or '').strip()
                        if candidate_sector and candidate_sector.lower() != 'unknown':
                            return candidate_sector
                except Exception:
                    pass

            return self._sector_from_asset_type(stock.get('asset_type'))
        except Exception:
            return 'Unknown'
    
    def _fetch_stock_info_yfinance(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info from yfinance - returns authoritative name"""
        try:
            import yfinance as yf
            
            # Normalize ticker - remove any existing .NS or .BO suffix to get base symbol
            base_ticker = ticker.upper().strip()
            has_ns = base_ticker.endswith('.NS')
            has_bo = base_ticker.endswith('.BO')
            
            if has_ns:
                base_ticker = base_ticker[:-3]
            elif has_bo:
                base_ticker = base_ticker[:-3]
            
            # Try different ticker formats (avoid duplicates)
            ticker_formats = []
            # If original ticker already had a suffix, try that first
            if has_ns:
                ticker_formats.append(f"{base_ticker}.NS")
            elif has_bo:
                ticker_formats.append(f"{base_ticker}.BO")
            else:
                # No suffix in original - try both exchanges
                ticker_formats.append(f"{base_ticker}.NS")
                ticker_formats.append(f"{base_ticker}.BO")
            
            # Also try base ticker without suffix (for BSE stocks that might not need .BO)
            # BUT: Only if original ticker didn't have a suffix (to avoid wrong matches)
            # If original had .NS or .BO, we should stick with that exchange
            if not has_ns and not has_bo:
                if base_ticker not in ticker_formats:
                    ticker_formats.append(base_ticker)
            
            for ticker_format in ticker_formats:
                try:
                    stock = yf.Ticker(ticker_format)
                    info = stock.info
                    
                    if info and info.get('longName'):
                        stock_name = info.get('longName', '').strip()
                        if stock_name:
                            print(f"[STOCK_INFO] ✅ Fetched stock name for {ticker} → '{stock_name}' using ticker format: {ticker_format}")
                            return {
                                'stock_name': stock_name,
                                'sector': info.get('sector', 'Unknown'),
                                'source': 'yfinance',
                                'ticker_used': ticker_format
                            }
                except Exception as e:
                    # Log error for debugging but continue to next format
                    print(f"[STOCK_INFO] ⚠️ Failed to fetch info for {ticker_format}: {str(e)[:100]}")
                    continue
            
            return {}
        except Exception as e:
            return {}
    
    def _fetch_mf_info_mftool(self, ticker: str, fund_name: str = None) -> Dict[str, Any]:
        """
        Fetch mutual fund info from mftool, with AMFI fallback.
        CRITICAL: Resolves ticker using BOTH ticker AND name for accurate AMFI code mapping.
        IGNORES fund_name if it looks like a channel/filename (e.g., "pornima", short names, etc.)
        """
        # Normalize ticker: remove any prefixes/suffixes, ensure it's just the AMFI code
        normalized_ticker = str(ticker).strip()
        # Remove common prefixes
        normalized_ticker = normalized_ticker.replace('MF_', '').replace('mf_', '').strip()
        
        # CRITICAL: Validate fund_name - ignore if it looks like a channel/filename
        # BUT: Allow valid ticker codes (SGB codes, all-caps alphanumeric codes, etc.)
        if fund_name:
            fund_name_lower = fund_name.lower().strip()
            fund_name_upper = fund_name.upper().strip()
            
            # Check if it's a valid ticker code pattern (SGB codes, all-caps alphanumeric, bond codes, etc.)
            # Bond codes may contain special characters like %, -, ., etc.
            # Stock tickers can be 3-4 characters (BHEL, TCS, HDFC, etc.)
            has_letters = any(c.isalpha() for c in fund_name_upper)
            has_digits = any(c.isdigit() for c in fund_name_upper)
            is_valid_ticker_code = (
                fund_name_upper.startswith('SGB') or  # Sovereign Gold Bond codes (SGBJUN31I, SGBFEB32IV)
                (fund_name_upper.isupper() and fund_name_upper.isalnum() and len(fund_name_upper) >= 3) or  # All-caps alphanumeric codes (BHEL, TATAGOLD, etc.) - allow 3+ chars
                (fund_name_upper.isupper() and has_digits) or  # Contains digits (likely a code)
                (fund_name_upper.isupper() and has_letters and (has_digits or '%' in fund_name_upper or 'BOND' in fund_name_upper))  # Bond codes with special chars (2.50%GOLDBONDS2031SR-I, etc.)
            )
            
            # Only reject if it's clearly a channel/filename AND not a valid ticker code
            invalid_patterns = [
                not is_valid_ticker_code and len(fund_name_lower) < 10 and ' ' not in fund_name_lower,  # Short single word
                not is_valid_ticker_code and fund_name_lower in ['pornima', 'zerodha', 'groww', 'paytm', 'upstox', 'angel', 'icici', 'hdfc', 'sbi'],  # Common channels
                not is_valid_ticker_code and fund_name_lower.islower() and len(fund_name_lower.split()) == 1  # Single lowercase word
            ]
            if any(invalid_patterns):
                print(f"[MF_INFO] ⚠️ Ignoring invalid fund_name '{fund_name}' (looks like channel/filename), fetching from AMFI/mftool")
                fund_name = None
        
        # If ticker is already a valid AMFI code (numeric), use it directly
        if normalized_ticker.isdigit():
            scheme_code = normalized_ticker
        else:
            # CRITICAL: Resolve AMFI code using BOTH ticker AND name
            scheme_code = None
            try:
                import importlib
                import difflib
                web_agent_module = importlib.import_module('web_agent')
                if hasattr(web_agent_module, 'get_amfi_dataset'):
                    amfi_data = web_agent_module.get_amfi_dataset()
                    if amfi_data and 'code_lookup' in amfi_data:
                        code_lookup = amfi_data['code_lookup']
                        
                        # STEP 1: Try exact ticker match first
                        if normalized_ticker in code_lookup:
                            scheme_code = normalized_ticker
                        else:
                            # STEP 2: Match by BOTH ticker AND name together
                            if fund_name:
                                normalized_name = (fund_name or '').strip().upper()
                                normalized_ticker_upper = normalized_ticker.upper()
                                
                                best_match = None
                                best_score = 0.0
                                
                                for code, scheme in code_lookup.items():
                                    scheme_name = (scheme.get('name', '') or '').upper()
                                    scheme_code_str = str(code).upper()
                                    
                                    # Check ticker match (in code or scheme name)
                                    ticker_match = (
                                        normalized_ticker_upper in scheme_code_str or
                                        scheme_code_str in normalized_ticker_upper or
                                        normalized_ticker_upper in scheme_name
                                    )
                                    
                                    # Check name match
                                    name_score = 0.0
                                    if normalized_name:
                                        name_score = difflib.SequenceMatcher(
                                            a=normalized_name,
                                            b=scheme_name
                                        ).ratio()
                                    
                                    # Combined score: both must match
                                    if ticker_match and name_score > 0.6:
                                        combined_score = name_score * 1.5
                                        if combined_score > best_score:
                                            best_score = combined_score
                                            best_match = code
                                
                                if best_match:
                                    scheme_code = str(best_match)
                            
                            # STEP 3: Fallback to name-only matching
                            if not scheme_code and fund_name:
                                normalized_name = (fund_name or '').strip().upper()
                                for code, scheme in code_lookup.items():
                                    scheme_name = (scheme.get('name', '') or '').upper()
                                    if normalized_name in scheme_name or scheme_name in normalized_name:
                                        scheme_code = str(code)
                                        break
                            
                            # STEP 4: Last resort - use original ticker
                            if not scheme_code:
                                scheme_code = normalized_ticker
            except Exception as e:
                print(f"[AMFI_RESOLVE] ⚠️ Error resolving AMFI code: {str(e)}")
                scheme_code = normalized_ticker
        
        # Try mftool with resolved code
        try:
            from mftool import Mftool
            mf = Mftool()
            
            # Ensure scheme_code is a string of digits for mftool
            if scheme_code.isdigit():
                # Get scheme info using the AMFI code
                scheme_info = mf.get_scheme_info(scheme_code)
                
                if scheme_info and scheme_info.get('schemeName'):
                    return {
                        'stock_name': scheme_info.get('schemeName', ''),
                        'sector': scheme_info.get('category', 'Unknown'),
                        'source': 'mftool'
                    }
        except Exception as e:
            # mftool failed, try AMFI dataset fallback
            pass
        
        # Fallback: Try AMFI dataset directly (avoid circular import by using lazy import)
        try:
            import importlib
            web_agent_module = importlib.import_module('web_agent')
            if hasattr(web_agent_module, 'get_amfi_dataset'):
                amfi_data = web_agent_module.get_amfi_dataset()
                if amfi_data and 'code_lookup' in amfi_data:
                    # Try with resolved code first
                    scheme = amfi_data['code_lookup'].get(scheme_code)
                    if not scheme and normalized_ticker != scheme_code:
                        # Try with original normalized ticker
                        scheme = amfi_data['code_lookup'].get(normalized_ticker)
                    if scheme and scheme.get('name'):
                        return {
                            'stock_name': scheme.get('name', ''),
                            'sector': 'Mutual Fund',
                            'source': 'amfi_dataset'
                        }
        except:
            pass
        
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
            if not prices:
                return True

            dedup_map: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
            for entry in prices:
                if not entry:
                    continue
                key = (entry.get('stock_id'), entry.get('price_date'))
                if not key[0] or not key[1]:
                    continue
                dedup_map[key] = entry

            if not dedup_map:
                return True

            chunk_size = 500
            items = list(dedup_map.values())

            # Upsert to avoid duplicates
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                self.supabase.table('historical_prices').upsert(
                    chunk,
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
            # Use resolved_ticker if provided (from name-based resolution during validation)
            resolved_ticker = transaction_data.get('_resolved_ticker') or transaction_data.get('resolved_ticker')
            stock_id = self.get_or_create_stock(
                transaction_data['ticker'],
                transaction_data['stock_name'],
                transaction_data['asset_type'],
                transaction_data.get('sector'),
                resolved_ticker=resolved_ticker
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
            
            # CRITICAL: Check for duplicate transaction before inserting
            # A duplicate is defined as: same user, portfolio, stock, date, quantity, price, and type
            # This prevents storing the same transaction from different files
            # Use approximate matching for quantity/price to handle floating point precision
            new_quantity = float(transaction_data['quantity'])
            new_price = float(transaction_data['price'])
            
            # Get existing transactions for this user/portfolio/stock/date/type
            existing_transactions = self._retry_db_operation(
                lambda: self.supabase.table('user_transactions').select('id, quantity, price').eq(
                    'user_id', transaction_data['user_id']
                ).eq('portfolio_id', transaction_data['portfolio_id']).eq(
                    'stock_id', stock_id
                ).eq('transaction_date', transaction_data['transaction_date']).eq(
                    'transaction_type', transaction_data['transaction_type']
                ).execute(),
                max_retries=2,
                base_delay=0.5
            )
            
            # Check if any existing transaction matches (with tolerance for floating point)
            if existing_transactions.data:
                for existing in existing_transactions.data:
                    existing_qty = float(existing.get('quantity', 0) or 0)
                    existing_price = float(existing.get('price', 0) or 0)
                    
                    # Match if quantity and price are within 0.01 (handles floating point precision)
                    qty_match = abs(existing_qty - new_quantity) < 0.01
                    price_match = abs(existing_price - new_price) < 0.01
                    
                    if qty_match and price_match:
                        # Duplicate transaction found - skip insertion
                        existing_id = existing['id']
                        return {
                            'success': False,
                            'error': 'Duplicate transaction',
                            'duplicate': True,
                            'existing_id': existing_id
                        }
            
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
            
            # Upsert holdings (update all holdings, including zero-quantity ones)
            # We keep zero-quantity holdings in database for historical purposes
            # But filter them out in get_user_holdings() so they don't appear in UI
            updated_count = 0
            for stock_id, calc in holdings_calc.items():
                total_quantity = calc['buy_qty'] - calc['sell_qty']
                average_price = calc['total_cost'] / calc['buy_qty'] if calc['buy_qty'] > 0 else 0
                
                # Update ALL holdings (including zero-quantity) - we'll filter them out in get_user_holdings()
                # Note: sector is not stored in holdings table, it's resolved from stock_master when needed
                self.supabase.table('holdings').upsert({
                    'user_id': user_id,
                    'portfolio_id': calc['portfolio_id'],
                    'stock_id': stock_id,
                    'total_quantity': total_quantity,
                    'average_price': average_price
                }, on_conflict='user_id,portfolio_id,stock_id').execute()
                
                updated_count += 1
                if total_quantity <= 0.0001:
                    print(f"[HOLDINGS] Updated zero-quantity holding (will be filtered in UI): stock_id={stock_id}, quantity={total_quantity}")
            
            print(f"[HOLDINGS] Recalculated {updated_count} holdings for user {user_id}")
            return updated_count
            
        except Exception as e:
            print(f"[HOLDINGS] Error recalculating: {e}")
            return 0
    
    def _consolidate_duplicate_holdings(self, holdings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate duplicate holdings that have the same normalized ticker but different stock_ids.
        This happens when ticker resolution creates new stock_ids (e.g., RTNINDIA vs RTNINDIA.NS).
        
        Strategy:
        1. First, drop exact duplicates (all columns match with normalized ticker)
        2. Drop duplicates where normalized ticker + quantity + date match
        3. Then consolidate remaining duplicates by merging quantities and recalculating average price
        4. Keep the most authoritative stock_id (prefer one with .NS/.BO suffix)
        """
        if not holdings:
            return holdings
        
        def normalize_ticker(ticker: str) -> str:
            """Normalize ticker by removing exchange suffixes"""
            if not ticker:
                return ''
            ticker = str(ticker).strip().upper()
            # Remove .NS or .BO suffix
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                return ticker[:-3]
            return ticker
        
        def get_holding_key(holding: Dict[str, Any], include_date: bool = False) -> tuple:
            """Create a key for comparing holdings"""
            ticker = normalize_ticker(holding.get('ticker', ''))
            asset_type = holding.get('asset_type', 'stock')
            stock_name = str(holding.get('stock_name', '')).strip().upper()
            quantity = float(holding.get('total_quantity', 0) or 0)
            
            key_parts = [ticker, asset_type]
            
            if stock_name and stock_name not in ['', 'UNKNOWN', 'NONE', 'NULL']:
                key_parts.append(stock_name)
            
            if include_date:
                # Include last_updated date for exact matching
                last_updated = holding.get('last_updated', '')
                if last_updated:
                    # Normalize date to just the date part (ignore time)
                    if isinstance(last_updated, str):
                        date_part = last_updated.split('T')[0] if 'T' in last_updated else last_updated.split(' ')[0]
                    else:
                        date_part = str(last_updated).split('T')[0] if 'T' in str(last_updated) else str(last_updated).split(' ')[0]
                    key_parts.append(date_part)
                key_parts.append(quantity)  # Include quantity when checking date
            
            return tuple(key_parts)
        
        # STEP 1: Drop exact duplicates (all columns match with normalized ticker)
        seen_exact = {}
        deduplicated = []
        exact_duplicates_dropped = 0
        
        for holding in holdings:
            # Create a comprehensive key that includes all relevant fields
            ticker = normalize_ticker(holding.get('ticker', ''))
            asset_type = holding.get('asset_type', 'stock')
            stock_name = str(holding.get('stock_name', '')).strip().upper()
            quantity = float(holding.get('total_quantity', 0) or 0)
            avg_price = float(holding.get('average_price', 0) or 0)
            current_price = float(holding.get('current_price') or holding.get('live_price') or 0)
            sector = str(holding.get('sector', '')).strip().upper()
            last_updated = holding.get('last_updated', '')
            
            # Normalize date
            if last_updated:
                if isinstance(last_updated, str):
                    date_part = last_updated.split('T')[0] if 'T' in last_updated else last_updated.split(' ')[0]
                else:
                    date_part = str(last_updated).split('T')[0] if 'T' in str(last_updated) else str(last_updated).split(' ')[0]
            else:
                date_part = ''
            
            # Create comprehensive key for exact matching
            exact_key = (
                ticker,
                asset_type,
                stock_name,
                round(quantity, 6),  # Round to handle floating point precision
                round(avg_price, 6),
                round(current_price, 6),
                sector,
                date_part
            )
            
            if exact_key not in seen_exact:
                seen_exact[exact_key] = holding
                deduplicated.append(holding)
            else:
                exact_duplicates_dropped += 1
                print(f"[HOLDINGS] 🗑️ Dropped exact duplicate: {holding.get('ticker')} - {holding.get('stock_name')} (all columns match)")
        
        if exact_duplicates_dropped > 0:
            print(f"[HOLDINGS] ✅ Dropped {exact_duplicates_dropped} exact duplicate holdings")
        
        holdings = deduplicated
        
        # STEP 2: Drop duplicates where normalized ticker + quantity + date match
        seen_quantity_date = {}
        deduplicated = []
        quantity_date_duplicates_dropped = 0
        
        for holding in holdings:
            key = get_holding_key(holding, include_date=True)
            
            if key not in seen_quantity_date:
                seen_quantity_date[key] = holding
                deduplicated.append(holding)
            else:
                quantity_date_duplicates_dropped += 1
                print(f"[HOLDINGS] 🗑️ Dropped duplicate (ticker+quantity+date match): {holding.get('ticker')} - {holding.get('stock_name')} (qty: {holding.get('total_quantity')}, date: {holding.get('last_updated')})")
        
        if quantity_date_duplicates_dropped > 0:
            print(f"[HOLDINGS] ✅ Dropped {quantity_date_duplicates_dropped} duplicates (ticker+quantity+date match)")
        
        holdings = deduplicated
        
        # STEP 3: Group remaining holdings by normalized ticker + asset_type + stock_name
        # This ensures we consolidate holdings that are truly the same asset
        grouped = defaultdict(list)
        for holding in holdings:
            ticker = holding.get('ticker', '')
            asset_type = holding.get('asset_type', 'stock')
            stock_name = str(holding.get('stock_name', '')).strip().upper()
            normalized = normalize_ticker(ticker)
            
            # Create a key that includes ticker, asset_type, and stock_name
            # This ensures we only consolidate holdings that are truly the same
            # If stock_name is missing or 'UNKNOWN', just use ticker + asset_type
            if stock_name and stock_name not in ['', 'UNKNOWN', 'NONE', 'NULL']:
                key = (normalized, asset_type, stock_name)
            else:
                # Fallback: group by ticker + asset_type only if stock_name is missing
                key = (normalized, asset_type, None)
            grouped[key].append(holding)
        
        consolidated = []
        duplicates_found = 0
        
        for key, group in grouped.items():
            normalized_ticker, asset_type = key[0], key[1]
            if len(group) == 1:
                # No duplicates, keep as-is
                consolidated.append(group[0])
            else:
                # Multiple holdings with same normalized ticker - consolidate
                duplicates_found += len(group) - 1
                
                # Prefer stock_id with .NS/.BO suffix (more authoritative)
                group_sorted = sorted(group, key=lambda h: (
                    1 if not str(h.get('ticker', '')).endswith(('.NS', '.BO')) else 0,
                    h.get('last_updated', '') or ''
                ), reverse=True)
                
                # Use the most authoritative holding as base
                base_holding = group_sorted[0].copy()
                
                # Sum quantities and calculate weighted average price
                total_quantity = 0.0
                total_cost = 0.0
                
                for h in group:
                    qty = float(h.get('total_quantity', 0) or 0)
                    avg_price = float(h.get('average_price', 0) or 0)
                    
                    total_quantity += qty
                    total_cost += qty * avg_price
                
                # Recalculate average price
                if total_quantity > 0:
                    new_avg_price = total_cost / total_quantity
                else:
                    new_avg_price = base_holding.get('average_price', 0)
                
                # Update consolidated holding
                base_holding['total_quantity'] = total_quantity
                base_holding['average_price'] = new_avg_price
                
                # Use the best current_price (prefer non-zero)
                best_price = 0.0
                for h in group:
                    price = float(h.get('current_price') or h.get('live_price') or 0)
                    if price > best_price:
                        best_price = price
                if best_price > 0:
                    base_holding['current_price'] = best_price
                    base_holding['live_price'] = best_price
                
                # Use the most complete stock_name
                for h in group:
                    name = h.get('stock_name', '')
                    if name and name != 'Unknown' and len(name) > len(base_holding.get('stock_name', '')):
                        base_holding['stock_name'] = name
                
                consolidated.append(base_holding)
                
                # Log consolidation
                tickers = [h.get('ticker') for h in group]
                print(f"[HOLDINGS] 🔄 Consolidated {len(group)} duplicate holdings: {tickers} → {base_holding.get('ticker')} (qty: {total_quantity:.2f})")
        
        if duplicates_found > 0:
            print(f"[HOLDINGS] ✅ Consolidated {duplicates_found} duplicate holdings into {len(consolidated)} unique holdings")
        
        return consolidated
    
    def get_user_holdings(self, user_id: str, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user holdings with stock details (uses view)"""
        try:
            query = self.supabase.table('user_holdings_detailed').select('*').eq('user_id', user_id)
            
            if portfolio_id:
                query = query.eq('portfolio_id', portfolio_id)
            
            response = query.execute()
            holdings = response.data

            # CRITICAL: Filter out holdings where total quantity (calculated from buys - sells) is zero or negative
            # The total_quantity field is calculated from transactions: buy_qty - sell_qty in recalculate_holdings()
            # If the net quantity from all transactions is zero (all sold), exclude from display and calculations
            # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
            filtered_holdings = []
            for h in holdings:
                quantity_raw = h.get('total_quantity', 0)
                
                # Enhanced parsing to handle all edge cases
                quantity = 0.0
                if quantity_raw is None:
                    quantity = 0.0
                elif isinstance(quantity_raw, str):
                    # Handle string "0", "0.0", "", etc.
                    quantity_raw = quantity_raw.strip()
                    if not quantity_raw or quantity_raw in ['0', '0.0', '0.00', '']:
                        quantity = 0.0
                    else:
                        try:
                            quantity = float(quantity_raw)
                        except (ValueError, TypeError):
                            quantity = 0.0
                elif isinstance(quantity_raw, (int, float)):
                    quantity = float(quantity_raw)
                else:
                    # For any other type, try to convert
                    try:
                        quantity = float(quantity_raw)
                    except (ValueError, TypeError):
                        quantity = 0.0
                
                # Treat anything <= 0.0001 as effectively zero (fully sold position)
                # This quantity represents: total_buys - total_sells from all transactions
                if quantity > 0.0001:
                    filtered_holdings.append(h)
                else:
                    # Debug: Log filtered holdings
                    ticker = h.get('ticker', 'Unknown')
                    stock_name = h.get('stock_name', 'Unknown')
                    print(f"[GET_HOLDINGS] Filtering out zero-quantity holding: {ticker} - {stock_name} (calculated_qty={quantity}, raw={quantity_raw}, type={type(quantity_raw).__name__})")
            holdings = filtered_holdings

            latest_channels = self._prefetch_latest_channels(user_id)
            
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

                holding['channel'] = latest_channels.get(stock_id, 'Direct')
            
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

            # CRITICAL: Filter out holdings where total quantity (calculated from buys - sells) is zero or negative
            # The total_quantity field is calculated from transactions: buy_qty - sell_qty in recalculate_holdings()
            # If the net quantity from all transactions is zero (all sold), exclude from display and calculations
            # Use a small epsilon to handle floating point precision issues (e.g., 0.0001)
            filtered_holdings = []
            for h in holdings:
                quantity_raw = h.get('total_quantity', 0)
                
                # Enhanced parsing to handle all edge cases
                quantity = 0.0
                if quantity_raw is None:
                    quantity = 0.0
                elif isinstance(quantity_raw, str):
                    # Handle string "0", "0.0", "", etc.
                    quantity_raw = quantity_raw.strip()
                    if not quantity_raw or quantity_raw in ['0', '0.0', '0.00', '']:
                        quantity = 0.0
                    else:
                        try:
                            quantity = float(quantity_raw)
                        except (ValueError, TypeError):
                            quantity = 0.0
                elif isinstance(quantity_raw, (int, float)):
                    quantity = float(quantity_raw)
                else:
                    # For any other type, try to convert
                    try:
                        quantity = float(quantity_raw)
                    except (ValueError, TypeError):
                        quantity = 0.0
                
                # Treat anything <= 0.0001 as effectively zero (fully sold position)
                # This quantity represents: total_buys - total_sells from all transactions
                if quantity > 0.0001:
                    filtered_holdings.append(h)
                else:
                    # Debug: Log filtered holdings (silent mode - no print)
                    pass
            holdings = filtered_holdings

            # CRITICAL: Consolidate duplicate holdings (same ticker, different stock_ids)
            holdings = self._consolidate_duplicate_holdings(holdings)

            latest_channels = self._prefetch_latest_channels(user_id)
            
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

                holding['channel'] = latest_channels.get(stock_id, 'Direct')
            
            return holdings
        except Exception as e:
            return []

    def _prefetch_latest_channels(self, user_id: str) -> Dict[str, str]:
        """Fetch latest channel per stock in a single query."""
        try:
            response = (
                self.supabase.table('user_transactions')
                .select('stock_id, channel, transaction_date')
                .eq('user_id', user_id)
                .order('transaction_date', desc=True)
                .limit(5000)
                .execute()
            )
        except Exception:
            return {}

        channel_map: Dict[str, str] = {}
        for row in response.data or []:
            stock_id = row.get('stock_id')
            if not stock_id or stock_id in channel_map:
                continue
            channel_value = row.get('channel') or 'Direct'
            channel_map[stock_id] = channel_value

            if len(channel_map) > 5000:
                break

        return channel_map
    
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
            response = self._retry_db_operation(
                lambda: self.supabase.table('user_transactions').select(
                    'stock_id, iso_year, iso_week'
                ).eq('user_id', user_id).execute(),
                max_retries=3,
                base_delay=1.0
            )
            
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
                try:
                    stock_response = self._retry_db_operation(
                        lambda: self.supabase.table('stock_master').select(
                            'id, ticker, stock_name'
                        ).eq('id', stock_id).execute(),
                        max_retries=3,
                        base_delay=1.0
                    )
                    
                    if stock_response.data:
                        stock = stock_response.data[0]
                        stock_details[stock_id] = stock['ticker']
                except Exception as e:
                    print(f"[DB_ERROR] ⚠️ Error getting stock details for stock_id {stock_id}: {e}")
                    # Continue with other stocks even if one fails
                    continue
            
            #st.caption(f"   ✅ Retrieved details for {len(stock_details)} stocks")
            
            # OPTIMIZATION: Get ALL existing prices for this user's stocks in ONE query
            #st.caption("   🔍 Step 3: Checking existing prices (bulk query)...")
            
            existing_prices = {}
            if stock_ids:
                # Get all existing prices for user's stocks in one query
                existing_response = self._retry_db_operation(
                    lambda: self.supabase.table('historical_prices').select(
                        'stock_id, iso_year, iso_week'
                    ).in_('stock_id', stock_ids).execute(),
                    max_retries=3,
                    base_delay=1.0
                )
                
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
            
        except (httpx.ReadError, httpx.ConnectError, OSError) as e:
            error_msg = str(e)
            if 'temporarily unavailable' in error_msg.lower() or (isinstance(e, OSError) and getattr(e, 'errno', None) == 11):
                st.warning(f"⚠️ Network error: Database connection temporarily unavailable. Please try again in a moment.")
                print(f"[DB_ERROR] Network error getting missing weeks: {error_msg}")
            else:
                st.error(f"❌ Network error getting missing weeks: {error_msg}")
            return []
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error getting missing weeks: {error_msg}")
            print(f"[DB_ERROR] Error getting missing weeks: {error_msg}")
            import traceback
            print(f"[DB_ERROR] Traceback:\n{traceback.format_exc()}")
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

    def get_sector_weekly_performance(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Calculate P&L % for each sector using the most recent 52 weeks of historical prices.
        Returns a DataFrame with columns: sector, investment_52w, current_value_52w, pnl_pct_52w.
        """
        try:
            holdings = self.get_user_holdings(user_id)
            if not holdings:
                return None

            sector_data: Dict[str, Dict[str, float]] = {}
            today = datetime.now().date()

            for holding in holdings:
                raw_sector = str(holding.get('sector') or '').strip()
                asset_type_value = str(holding.get('asset_type') or '').strip().lower()
                if raw_sector and raw_sector.lower() != 'unknown':
                    sector = raw_sector
                else:
                    if asset_type_value in ['mutual_fund']:
                        sector = 'Mutual Fund'
                    elif asset_type_value in ['etf']:
                        sector = 'ETF'    
                    elif asset_type_value in ['pms', 'pms equity']:
                        sector = 'PMS'
                    elif asset_type_value in ['aif', 'aif cat iii', 'aif equity']:
                        sector = 'AIF'
                    elif asset_type_value in ['bond', 'gold bond', 'sgb']:
                        sector = 'Bond'
                    elif asset_type_value == 'stock':
                        sector = 'Stock'
                    else:
                        sector = 'Unknown'
                sector = sector or 'Unknown'
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

                sector_entry = sector_data.setdefault(
                    sector,
                    {'investment': 0.0, 'current': 0.0}
                )
                sector_entry['investment'] += baseline_value
                sector_entry['current'] += latest_value

            rows = []
            for sector, values in sector_data.items():
                investment = values['investment']
                current_val = values['current']
                if investment > 0:
                    pnl_pct = (current_val - investment) / investment * 100
                elif current_val > 0:
                    pnl_pct = 100.0
                else:
                    pnl_pct = 0.0
                rows.append({
                    'sector': sector,
                    'investment_52w': investment,
                    'current_value_52w': current_val,
                    'pnl_pct_52w': pnl_pct,
                })

            if not rows:
                return None

            result = pd.DataFrame(rows)
            return result.sort_values('sector').reset_index(drop=True)
        except Exception as exc:
            print(f"[SECTOR_52W] Error computing weekly performance: {exc}")
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

    def get_sector_weekly_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Build weekly P&L history per sector for the last ~52 weeks.
        Returns list of {sector, date, pnl_pct, investment, current_value}
        """
        try:
            holdings = self.get_user_holdings(user_id)
            if not holdings:
                return []

            sector_investments: Dict[str, float] = defaultdict(float)
            sector_values: Dict[str, Dict[datetime.date, float]] = defaultdict(lambda: defaultdict(float))

            for holding in holdings:
                raw_sector = str(holding.get('sector') or '').strip()
                asset_type_value = str(holding.get('asset_type') or '').strip().lower()
                if raw_sector:
                    sector = raw_sector
                else:
                    if asset_type_value in ['mutual_fund', 'etf']:
                        sector = 'Mutual Fund'
                    elif asset_type_value in ['pms', 'pms equity']:
                        sector = 'PMS'
                    elif asset_type_value in ['aif', 'aif cat iii', 'aif equity']:
                        sector = 'AIF'
                    elif asset_type_value in ['bond', 'gold bond', 'sgb']:
                        sector = 'Bond'
                    elif asset_type_value == 'stock':
                        sector = 'Stock'
                    else:
                        sector = 'Unknown'

                quantity = float(holding.get('total_quantity') or 0)
                avg_price = float(holding.get('average_price') or 0)
                stock_id = holding.get('stock_id')

                if quantity <= 0 or not stock_id:
                    continue

                sector_investments[sector] += max(quantity * avg_price, 0.0)

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
                    sector_values[sector][dt] += float(price) * quantity

            history_entries: List[Dict[str, Any]] = []
            for sector, date_map in sector_values.items():
                sorted_dates = sorted(date_map.keys())
                if len(sorted_dates) > 52:
                    sorted_dates = sorted_dates[-52:]

                for dt in sorted_dates:
                    current_value = date_map[dt]
                    investment = sector_investments.get(sector, 0.0)
                    pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0.0
                    history_entries.append({
                        'sector': sector,
                        'date': dt.strftime('%Y-%m-%d'),
                        'pnl_pct': pnl_pct,
                        'investment': investment,
                        'current_value': current_value,
                    })

            history_entries.sort(key=lambda entry: (entry['sector'], entry['date']))
            return history_entries
        except Exception as exc:
            print(f"[SECTOR_52W] Error building weekly history: {exc}")
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
            existing_rows = existing_hist.data or []
            existing_pairs = {(row.get('iso_year'), row.get('iso_week')) for row in existing_rows if row.get('iso_year') is not None and row.get('iso_week') is not None}
            existing_count = len(existing_rows)
            print(f"[HIST] Existing historical records: {existing_count}")
            
            required_pairs: Optional[set] = None
            if target_weeks:
                required_pairs = {
                    (int(entry.get('year')), int(entry.get('week')))
                    for entry in target_weeks
                    if entry.get('year') is not None and entry.get('week') is not None
                }
                if required_pairs:
                    required_pairs -= existing_pairs
                if not required_pairs:
                    print(f"[HIST] ✅ All {len(target_weeks)} requested weeks already cached for {ticker}")
                    return []
                else:
                    print(f"[HIST] 📊 Need to fetch {len(required_pairs)} missing weeks (out of {len(target_weeks)} requested) for {ticker}")
            else:
                # OPTIMIZATION: Even without target_weeks, check what's missing in last 52 weeks
                # Don't fetch if we already have substantial recent data
                current_date = datetime.now()
                start_date = current_date - timedelta(weeks=52)
                current_year, current_week, _ = current_date.isocalendar()
                start_year, start_week, _ = start_date.isocalendar()
                
                # Generate last 52 weeks
                last_52_weeks = set()
                temp_date = start_date
                while temp_date <= current_date:
                    year, week, _ = temp_date.isocalendar()
                    last_52_weeks.add((year, week))
                    temp_date += timedelta(weeks=1)
                
                # Check how many of the last 52 weeks we already have
                existing_recent = existing_pairs & last_52_weeks
                if len(existing_recent) >= 40:  # Already have 40+ of last 52 weeks
                    print(f"[HIST] ✅ SKIP: Already have {len(existing_recent)}/52 recent weeks for {ticker} (>=40, sufficient)")
                    return []
                elif existing_count >= 100:  # Have 100+ total records, likely complete
                    print(f"[HIST] ✅ SKIP: Already have {existing_count} total records for {ticker} (>=100, likely complete)")
                    return []
                else:
                    # Only fetch missing recent weeks
                    required_pairs = last_52_weeks - existing_pairs
                    if not required_pairs:
                        print(f"[HIST] ✅ All recent weeks already cached for {ticker}")
                        return []
                    print(f"[HIST] 📊 Need to fetch {len(required_pairs)} missing recent weeks for {ticker} (have {len(existing_recent)}/52)")
            
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
                    mf = fetcher._mftool or fetcher._get_shared_mftool()
                    if mf is None:
                        raise RuntimeError("mftool unavailable")

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
                            # Only fetch weeks that are actually missing
                            required_df = df_nav[df_nav.apply(
                                lambda row: (int(row['iso_year']), int(row['iso_week'])) in required_pairs, axis=1)]
                        else:
                            # OPTIMIZATION: Only fetch missing recent weeks, not all historical data
                            # Check what's missing in last 52 weeks
                            cutoff_date = datetime.now() - timedelta(weeks=52)
                            recent_df = df_nav[df_nav['date'] >= cutoff_date]
                            # Filter out weeks we already have
                            if existing_pairs:
                                recent_df = recent_df[recent_df.apply(
                                    lambda row: (int(row['iso_year']), int(row['iso_week'])) not in existing_pairs, axis=1)]
                            required_df = recent_df
                            if required_df.empty:
                                print(f"[HIST] ✅ All recent weeks already cached for MF {ticker}")
                                return []

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
                    # Process returned data - only insert missing weeks
                    for item in historical_data:
                        try:
                            date_str = item.get('date')
                            price = item.get('price')
                            
                            # Skip if we already have this week
                            if date_str:
                                try:
                                    price_dt = datetime.strptime(date_str, '%Y-%m-%d')
                                    iso_calendar = price_dt.isocalendar()
                                    iso_pair = (iso_calendar[0], iso_calendar[1])
                                    if iso_pair in existing_pairs:
                                        continue  # Skip - already have this week
                                    if required_pairs is not None and iso_pair not in required_pairs:
                                        continue  # Skip - not in required weeks
                                except:
                                    pass  # Continue if date parsing fails
                            
                            if date_str and price:
                                price_date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                iso_calendar = price_date_obj.isocalendar()
                                iso_pair = (iso_calendar[0], iso_calendar[1])
                                
                                # OPTIMIZATION: Skip if we already have this week
                                if iso_pair in existing_pairs:
                                    continue  # Already have this week, skip
                                
                                # Skip if not in required weeks (when required_pairs is specified)
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

