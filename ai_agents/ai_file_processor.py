"""
Unified AI-Powered File Processor
Handles CSV, PDF, Excel, and any other file format intelligently
"""

import openai
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent


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
            validated_transactions = self._validate_and_enhance(transactions)
            
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
            prompt = f"""Extract ALL investment transactions from this {file_type.upper()} file intelligently:

FILENAME: {filename}
FILE TYPE: {file_type}

FILE CONTENT:
{file_content[:8000]}  

YOUR TASK:
1. **IDENTIFY TRANSACTIONS**: Find all buy/sell transactions for stocks, mutual funds, PMS, AIF, or bonds
2. **EXTRACT DATA**: For each transaction, extract:
   - Date (normalize to YYYY-MM-DD)
   - Ticker/Symbol/Scheme Code
   - Stock/Fund Name
   - Quantity
   - Price (if available, otherwise use 0)
   - Transaction Type (buy/sell)
   - Asset Type (stock/mutual_fund/pms/aif/bond)
   - Sector (if available or infer from name)
   - Channel/Platform (if available or use filename)

3. **INTELLIGENT INFERENCE**:
   - **Stock Name**: If missing, infer from ticker
   - **Sector**: Determine from company name (e.g., "Infosys" → "Technology")
   - **Asset Type**: 
     * Numeric codes (120760, 122639) → mutual_fund
     * .NS or .BO suffix → stock
     * Company names → stock
     * "Fund" in name → mutual_fund
   - **Channel**: Extract from file or use filename
   - **Price**: If missing or 0, leave as 0 (will be fetched later)

4. **DATA VALIDATION**:
   - Ensure dates are valid
   - Ensure quantities are positive numbers
   - Ensure prices are non-negative (0 if missing)
   - Normalize transaction types to "buy" or "sell"

5. **HANDLE ALL FORMATS**:
   - CSV: Parse columns intelligently
   - PDF: Extract from tables or text
   - Excel: Handle multiple sheets
   - Text: Parse structured or unstructured data

CRITICAL RULES:
- Extract EVERY transaction found in the file
- If data is missing and cannot be inferred, use reasonable defaults
- If price is missing/zero, set price to 0 (system will fetch it later)
- Return ONLY valid JSON array, no other text

RETURN THIS EXACT JSON FORMAT:
[
  {{
    "date": "YYYY-MM-DD",
    "ticker": "stock ticker or scheme code",
    "stock_name": "full name of security",
    "scheme_name": "MF scheme name if applicable, else null",
    "quantity": 100.0,
    "price": 1500.50,
    "transaction_type": "buy",
    "asset_type": "stock",
    "sector": "Technology",
    "channel": "Zerodha"
  }}
]"""

            # Call OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
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
                max_tokens=4000,
                temperature=0.1,
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
    
    def _validate_and_enhance(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance transaction data"""
        
        validated = []
        
        for trans in transactions:
            try:
                # Skip if missing critical fields
                if not trans.get('ticker') or not trans.get('date'):
                    continue
                
                # Normalize and validate fields
                validated_trans = {
                    'date': self._normalize_date(trans.get('date', '')),
                    'ticker': str(trans.get('ticker', '')).strip(),
                    'stock_name': trans.get('stock_name', trans.get('ticker', 'Unknown')),
                    'scheme_name': trans.get('scheme_name'),
                    'quantity': self._safe_float(trans.get('quantity', 0)),
                    'price': self._safe_float(trans.get('price', 0)),  # 0 if missing - will be fetched
                    'transaction_type': str(trans.get('transaction_type', 'buy')).lower(),
                    'asset_type': self._detect_asset_type(trans),
                    'sector': trans.get('sector', 'Unknown'),
                    'channel': trans.get('channel', 'Direct')
                }
                
                # Validate quantity is positive
                if validated_trans['quantity'] <= 0:
                    continue
                
                # Ensure price is non-negative
                if validated_trans['price'] < 0:
                    validated_trans['price'] = 0
                
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
                value = value.replace('₹', '').replace('Rs', '').replace(',', '').strip()
            
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

