"""
Unified AI-Powered File Processor
Handles CSV, PDF, Excel, and any other file format intelligently
"""

import openai
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .base_agent import BaseAgent

try:
    from enhanced_price_fetcher import EnhancedPriceFetcher
except ImportError:  # pragma: no cover - runtime dependency
    EnhancedPriceFetcher = None  # type: ignore[misc]


class AIFileProcessor(BaseAgent):
    """
    Universal AI agent that processes ANY file type and extracts transactions
    Supports: CSV, PDF, Excel, TXT, and more
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ai_file_processor",
            agent_name="AI Universal File Processor"
        )
        
        self.capabilities = [
            "csv_processing",
            "pdf_processing",
            "excel_processing",
            "transaction_extraction",
            "data_validation",
            "missing_data_inference",
            "sector_classification",
            "asset_type_detection",
            "price_handling"
        ]
        
        # Initialize OpenAI client
        try:
            import streamlit as st
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        # Initialize price fetcher for automatic price backfill
        if EnhancedPriceFetcher is not None:
            try:
                self.price_fetcher = EnhancedPriceFetcher()
            except Exception as exc:  # pragma: no cover - runtime dependency
                self.logger.error(f"Failed to initialize EnhancedPriceFetcher: {exc}")
                self.price_fetcher = None
        else:
            self.price_fetcher = None

        self._price_cache: Dict[Tuple[str, str, str], Optional[float]] = {}
        self.processed_transactions = []
    
    def process_file(self, file_data: Any, filename: str) -> List[Dict[str, Any]]:
        """
        Universal file processor - handles ANY file type
        
        Args:
            file_data: File object or file content
            filename: Name of the file
            
        Returns:
            List of extracted transactions
        """
        
        self.update_status("analyzing")
        
        try:
            if not self.openai_client:
                self.logger.error("OpenAI client not available")
                return []
            
            # Detect file type and extract content
            file_content, file_type = self._extract_file_content(file_data, filename)
            
            if not file_content:
                self.logger.error("Could not extract content from file")
                return []
            
            # Use AI to extract transactions
            transactions = self._ai_extract_transactions(file_content, filename, file_type)
            
            # Validate and enhance transactions
            validated_transactions = self._validate_and_enhance(transactions, filename)
            
            # Cache transactions
            self.processed_transactions = validated_transactions
            
            self.update_status("active")
            return validated_transactions
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            self.update_status("error")
            return []
    
    def _extract_file_content(self, file_data: Any, filename: str) -> tuple:
        """Extract content from any file type"""
        
        try:
            file_ext = filename.lower().split('.')[-1]
            
            # CSV Files
            if file_ext == 'csv':
                content = file_data.getvalue().decode('utf-8')
                return content, 'csv'
            
            # Excel Files
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_data)
                content = df.to_csv(index=False)
                return content, 'excel'
            
            # PDF Files
            elif file_ext == 'pdf':
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(file_data)
                pdf_text = ""
                
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
                
                return pdf_text, 'pdf'
            
            # Text Files
            elif file_ext in ['txt', 'text']:
                content = file_data.getvalue().decode('utf-8')
                return content, 'txt'
            
            # Unknown format - try to read as text
            else:
                try:
                    content = file_data.getvalue().decode('utf-8')
                    return content, 'unknown'
                except:
                    return None, 'unsupported'
                    
        except Exception as e:
            self.logger.error(f"Error extracting file content: {e}")
            return None, 'error'
    
    def _ai_extract_transactions(self, file_content: str, filename: str, file_type: str) -> List[Dict[str, Any]]:
        """Use AI to extract transactions from any file format"""
        
        try:
            # Create intelligent prompt based on file type
            prompt = f"""
You are parsing an IndMoney transaction report. The first row of each sheet is the header and matches one of these patterns:
- "Scrip Symbol", "Scrip Name", "Txn Date", "Quantity", "Price", "Amount", "Transaction Type", etc.

Map the file into JSON using the canonical schema below. Every row that represents a buy or sell transaction must appear in the output.

Schema (JSON array of objects):
[
  {{
    "date": "YYYY-MM-DD",
    "ticker": "exchange-ready ticker",
    "stock_name": "full security name",
    "scheme_name": null,
    "quantity": 0.0,
    "price": 0.0,
    "amount": 0.0,
    "transaction_type": "buy" | "sell",
    "asset_type": "stock" | "mutual_fund" | "bond" | "pms" | "aif",
    "sector": "Sector name or Unknown",
    "channel": "Broker/platform or filename"
  }}
]

Column rules:
- `Scrip Symbol`, `Trading Symbol`, or similar → `"ticker"`. Normalize to a trading code (e.g., RELIANCE → RELIANCE.NS if needed). Never return ISINs or descriptive names in `"ticker"`.
- `Scrip Name`, `Security Name`, `Scheme Name` → `"stock_name"`.
- `Txn Date`, `Transaction Date`, `Trade Date` → `"date"` (normalize to YYYY-MM-DD).
- `Quantity`, `Qty`, `Units` → `"quantity"`.
- `Price`, `Rate`, `NAV` → `"price"` (use 0 if missing).
- `Amount`, `Value`, `Consideration` → `"amount"` (use 0 if missing).
- Transaction verb columns (`Transaction Type`, `Action`, `Side`) → `"transaction_type"` (map all buy-like terms to `"buy"`; sell/redemption/switch-out to `"sell"`).
- Identify `"asset_type"` heuristically: numeric ticker → mutual_fund; `.NS`/`.BO` suffix or a typical stock symbol → stock; contains "bond"/"debenture"/"SGB" → bond; contains "PMS" → pms; contains "AIF" → aif.
- `"channel"`: prefer columns named `Channel`, `Broker`, `Platform`. If none exist, use the filename `{filename}`.
- `"sector"`: use any explicit sector column; otherwise default to `"Unknown"` unless you can infer from the company name.
- `"scheme_name"`: only set for mutual funds; otherwise null.

Validation:
- Skip rows whose quantity and amount are both blank or zero.
- Ensure quantity is positive. If a row shows negative quantity/amount for a sell, output a positive quantity and mark `"transaction_type": "sell"`.
- If price is absent, set it to 0 (downstream logic will backfill).

Output ONLY the JSON array—no commentary.

File excerpt (comma-separated rows):
{file_content[:6000]}
"""

            # Call OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better file processing
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial data extraction AI specializing in parsing investment transaction files.

CAPABILITIES:
- Extract transactions from ANY file format (CSV, PDF, Excel, Text)
- Intelligently map columns and fields
- Infer missing data using domain knowledge
- Detect asset types automatically
- Normalize dates and formats
- Handle messy, unstructured data

INTELLIGENCE FEATURES:
- Recognize stock tickers vs mutual fund codes
- Infer sectors from company names
- Detect transaction types from context
- Handle missing prices gracefully
- Smart column mapping even with different naming

CRITICAL RULES:
1. Extract ALL transactions from the file
2. Return ONLY valid JSON array
3. Use 0 for missing prices (system handles fetching)
4. Infer missing data intelligently
5. Validate all data types"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=4000,
                # Note: GPT-5 only supports default temperature (1)
                timeout=90
            )
            
            # Extract JSON from response
            ai_response = response.choices[0].message.content
            transactions = self._extract_json(ai_response)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"AI extraction failed: {e}")
            # Try fallback parsing
            return self._fallback_extraction(file_content, file_type)
    
    def _extract_json(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract JSON array from AI response"""
        
        try:
            # Find JSON array in response
            start_idx = ai_response.find('[')
            end_idx = ai_response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                transactions = json.loads(json_str)
                
                if isinstance(transactions, list):
                    return transactions
            
            return []
            
        except Exception as e:
            self.logger.error(f"JSON extraction error: {e}")
            return []
    
    def _validate_and_enhance(self, transactions: List[Dict[str, Any]], source_filename: str) -> List[Dict[str, Any]]:
        """Validate and enhance transaction data"""
        
        validated = []
        for trans in transactions:
            try:
                # Skip if missing critical fields
                if not trans.get('ticker') or not trans.get('date'):
                    continue
                
                # Normalize and validate fields
                transaction_type_raw = (
                    trans.get('transaction_type')
                    or trans.get('type')
                    or trans.get('action')
                    or trans.get('side')
                )
                amount_value = trans.get('amount', trans.get('value', 0))

                validated_trans = {
                    'date': self._normalize_date(trans.get('date', '')),
                    'ticker': str(trans.get('ticker', '')).strip(),
                    'stock_name': trans.get('stock_name', trans.get('ticker', 'Unknown')),
                    'scheme_name': trans.get('scheme_name'),
                    'quantity': self._safe_float(trans.get('quantity', 0)),
                    'price': self._safe_float(trans.get('price', 0)),  # 0 if missing - will be fetched
                    'transaction_type': str(transaction_type_raw or 'buy').lower(),
                    'asset_type': self._detect_asset_type(trans),
                    'sector': trans.get('sector') or 'Unknown',
                    'channel': self._infer_channel_from_filename(
                        source_filename,
                        trans.get('channel')
                    )
                }

                if validated_trans['price'] <= 0 and amount_value:
                    amount_float = self._safe_float(amount_value)
                    if amount_float > 0 and validated_trans['quantity'] > 0:
                        validated_trans['price'] = amount_float / validated_trans['quantity']
                
                # Validate quantity is positive
                if validated_trans['quantity'] <= 0:
                    continue
                
                # Ensure price is non-negative
                if validated_trans['price'] < 0:
                    validated_trans['price'] = 0

                # Backfill missing price using historical data
                fetched_price = self._fetch_price_for_transaction(validated_trans)
                if fetched_price and fetched_price > 0:
                    validated_trans['price'] = round(float(fetched_price), 4)
                
                validated.append(validated_trans)
                
            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                continue
        
        return validated
    
    def _detect_asset_type(self, trans: Dict[str, Any]) -> str:
        """Intelligently detect asset type"""
        
        ticker = str(trans.get('ticker', '')).strip()
        name = str(trans.get('stock_name', '')).lower()
        
        # Check explicit asset_type first
        if trans.get('asset_type'):
            return str(trans['asset_type']).lower()
        
        # Numeric ticker = likely mutual fund
        if ticker.isdigit() and len(ticker) >= 5:
            return 'mutual_fund'
        
        # Contains .NS or .BO = stock
        if '.NS' in ticker.upper() or '.BO' in ticker.upper():
            return 'stock'
        
        # Check name for clues
        if 'fund' in name or 'scheme' in name:
            return 'mutual_fund'
        
        if 'pms' in name or 'portfolio management' in name:
            return 'pms'
        
        if 'aif' in name or 'alternative investment' in name:
            return 'aif'
        
        if 'bond' in name or 'debenture' in name:
            return 'bond'
        
        # Default to stock
        return 'stock'
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD"""
        
        try:
            formats = [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d.%m.%Y',
                '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(date_str).strip(), fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
            
            return str(date_str)
            
        except:
            return str(date_str)
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert to float"""
        
        try:
            if value is None or value == '':
                return 0.0
            
            # Remove common currency symbols and commas
            if isinstance(value, str):
                cleaned = value.replace('₹', '').replace('Rs', '').replace(',', '').strip()
                cleaned = re.sub(r'[^0-9\.\-]', '', cleaned)
                if cleaned in {'', '.', '-', '-.'}:
                    return 0.0
                value = cleaned
            
            return float(value)
        except:
            return 0.0
    
    def _fallback_extraction(self, content: str, file_type: str) -> List[Dict[str, Any]]:
        """Fallback extraction using pandas"""
        
        try:
            if file_type == 'csv':
                import io
                df = pd.read_csv(io.StringIO(content))
                
                transactions = []
                for _, row in df.iterrows():
                    trans = {
                        'date': row.get('date', row.get('Date', '')),
                        'ticker': row.get('ticker', row.get('Ticker', row.get('Symbol', ''))),
                        'stock_name': row.get('stock_name', row.get('Stock Name', '')),
                        'quantity': self._safe_float(row.get('quantity', row.get('Quantity', 0))),
                        'price': self._safe_float(row.get('price', row.get('Price', 0))),
                        'transaction_type': str(row.get('transaction_type', row.get('Type', 'buy'))).lower(),
                        'asset_type': row.get('asset_type', row.get('Asset Type', 'stock')),
                        'sector': row.get('sector', row.get('Sector', 'Unknown')),
                        'channel': row.get('channel', row.get('Channel', 'Direct'))
                    }
                    transactions.append(trans)
                
                return transactions
        except:
            pass
        
        return []

    def _infer_channel_from_filename(self, filename: str, explicit_channel: Optional[str] = None) -> str:
        """Infer channel/platform from explicit value or fallback to filename stem."""
        if explicit_channel:
            candidate = str(explicit_channel).strip()
            if candidate:
                return candidate

        if not filename:
            return "Direct"

        stem = Path(filename).stem
        clean = re.sub(r'[_\-\s]+', ' ', stem).strip()
        return clean.title() if clean else "Direct"

    def _fetch_price_for_transaction(self, trans: Dict[str, Any]) -> Optional[float]:
        """Fetch historical price for transaction date when price missing/zero."""
        if not self.price_fetcher:
            return None

        price = trans.get('price') or 0
        if price and price > 0:
            return price

        ticker = trans.get('ticker')
        date = trans.get('date')
        asset_type = (trans.get('asset_type') or 'stock').lower()

        if not ticker or not date:
            return None

        cache_key = (ticker, date, asset_type)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        fund_name = trans.get('stock_name')
        fetched_price = None

        try:
            fetched_price = self.price_fetcher.get_historical_price(
                ticker,
                asset_type,
                date,
                fund_name=fund_name
            )

            if not fetched_price:
                current_price, _ = self.price_fetcher.get_current_price(
                    ticker,
                    asset_type,
                    fund_name=fund_name
                )
                fetched_price = current_price

            if fetched_price and fetched_price > 0:
                self._price_cache[cache_key] = fetched_price
                return fetched_price
        except Exception as exc:
            self.logger.warning(f"Price backfill failed for {ticker} on {date}: {exc}")

        self._price_cache[cache_key] = None
        return None
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze method required by BaseAgent"""
        file_data = data.get('file_data')
        filename = data.get('filename', 'unknown')
        
        transactions = self.process_file(file_data, filename)
        
        return self.format_response([{"transactions": transactions}], "high")
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get insights (required by BaseAgent)"""
        return self.processed_transactions
    
    def get_processed_transactions(self) -> List[Dict[str, Any]]:
        """Get processed transactions"""
        return self.processed_transactions

