"""
AI-Powered CSV Transaction Parser
Intelligently reads and processes CSV files with transactions
"""

import openai
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent


class AICSVTransactionParser(BaseAgent):
    """
    AI agent that intelligently parses CSV transaction files
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ai_csv_parser",
            agent_name="AI CSV Transaction Parser"
        )
        
        self.capabilities = [
            "csv_parsing",
            "transaction_extraction",
            "data_validation",
            "missing_data_inference",
            "sector_classification",
            "asset_type_detection"
        ]
        
        # Initialize OpenAI client
        try:
            import streamlit as st
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        self.parsed_transactions = []
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CSV file and extract transactions using AI"""
        
        self.update_status("analyzing")
        
        try:
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            csv_content = data.get("csv_content")
            filename = data.get("filename", "unknown.csv")
            
            if not csv_content:
                return self.format_response([], "low", error="No CSV content provided")
            
            # Use AI to parse and extract transactions
            transactions = self._ai_parse_csv(csv_content, filename)
            
            # Cache transactions
            self.parsed_transactions = transactions
            
            self.update_status("active")
            return self.format_response(transactions, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI CSV parsing: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_parse_csv(self, csv_content: str, filename: str) -> List[Dict[str, Any]]:
        """Use AI to intelligently parse CSV and extract transaction data"""
        
        try:
            # First, try to read CSV with pandas to get structure
            import io
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Get CSV preview (first 10 rows)
            csv_preview = df.head(10).to_string()
            column_names = list(df.columns)
            
            # Create AI prompt
            prompt = f"""Analyze this CSV file and extract ALL transaction data intelligently:

FILENAME: {filename}

CSV COLUMNS: {column_names}

CSV PREVIEW (first 10 rows):
{csv_preview}

FULL CSV DATA:
{csv_content}

YOUR TASK:
1. Extract ALL transactions from the CSV
2. Identify the correct columns for each field (date, ticker, quantity, price, etc.)
3. Infer missing data intelligently:
   - If stock_name is missing, infer from ticker
   - If sector is missing, determine from stock/company name
   - If asset_type is missing, determine from ticker format:
     * Numbers (e.g., 120760, 122639) = mutual_fund
     * .NS/.BO suffix = stock
     * No suffix but company name = stock
   - If channel is missing, use filename or "Direct"
   - If price is 0 or missing, leave as 0 (will be fetched later)
4. Validate and clean all data
5. Return in the exact JSON format below

IMPORTANT RULES:
- Extract EVERY transaction row from the CSV
- Handle different date formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
- Detect transaction_type (buy/sell) from any column
- Be intelligent about mapping columns even if names don't match exactly
- If a field is truly missing and can't be inferred, use reasonable defaults

Return JSON array with this EXACT structure:
[
  {{
    "date": "YYYY-MM-DD",
    "ticker": "stock ticker or MF scheme code",
    "stock_name": "full name of stock/MF",
    "scheme_name": "MF scheme name (if mutual fund, else null)",
    "quantity": 100,
    "price": 1500.50,
    "transaction_type": "buy or sell",
    "asset_type": "stock, mutual_fund, pms, aif, or bond",
    "sector": "sector classification",
    "channel": "trading platform or channel name"
  }}
]

CRITICAL: Return ONLY the JSON array, no other text."""

            # Call OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better CSV parsing
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial data analyst specializing in transaction data extraction and parsing.

Your task is to intelligently parse CSV files containing investment transactions and extract structured data.

CAPABILITIES:
- Smart column mapping (handle variations in column names)
- Data type detection (stocks vs mutual funds vs PMS/AIF)
- Missing data inference (sectors, asset types, company names)
- Date format normalization
- Data validation and cleaning

RULES:
1. Extract ALL rows from the CSV
2. Map columns intelligently even if names don't match exactly
3. Infer missing data using context and domain knowledge
4. Normalize all data to standard formats
5. Return valid JSON only"""
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
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            transactions = self._extract_json_from_response(ai_response)
            
            # Validate and enhance transactions
            validated_transactions = self._validate_transactions(transactions)
            
            return validated_transactions
            
        except Exception as e:
            self.logger.error(f"AI CSV parsing failed: {e}")
            # Fallback to basic pandas parsing
            return self._fallback_parse(csv_content, filename)
    
    def _extract_json_from_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract JSON array from AI response"""
        
        try:
            # Try to find JSON in the response
            start_idx = ai_response.find('[')
            end_idx = ai_response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                transactions = json.loads(json_str)
                
                if isinstance(transactions, list):
                    return transactions
            
            # If no JSON found, return empty list
            self.logger.error("No valid JSON found in AI response")
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON from response: {e}")
            return []
    
    def _validate_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance transaction data"""
        
        validated = []
        
        for trans in transactions:
            try:
                # Ensure required fields
                if not trans.get('date') or not trans.get('ticker'):
                    continue
                
                # Normalize date format
                trans['date'] = self._normalize_date(trans.get('date'))
                
                # Ensure numeric fields
                trans['quantity'] = float(trans.get('quantity', 0))
                trans['price'] = float(trans.get('price', 0))
                
                # Default values
                trans['transaction_type'] = trans.get('transaction_type', 'buy').lower()
                trans['asset_type'] = trans.get('asset_type', 'stock').lower()
                trans['channel'] = trans.get('channel', 'Direct')
                
                # Infer asset type if not set correctly
                if trans['asset_type'] == 'stock':
                    ticker = trans['ticker']
                    # Check if it's actually a mutual fund (numeric scheme code)
                    if ticker.isdigit():
                        trans['asset_type'] = 'mutual_fund'
                        trans['scheme_name'] = trans.get('stock_name', '')
                
                validated.append(trans)
                
            except Exception as e:
                self.logger.error(f"Error validating transaction: {e}")
                continue
        
        return validated
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format"""
        
        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(date_str), fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
            
            # If no format matches, return as is
            return str(date_str)
            
        except:
            return str(date_str)
    
    def _fallback_parse(self, csv_content: str, filename: str) -> List[Dict[str, Any]]:
        """Fallback to basic pandas parsing if AI fails"""
        
        try:
            import io
            df = pd.read_csv(io.StringIO(csv_content))
            
            transactions = []
            
            for _, row in df.iterrows():
                trans = {
                    'date': row.get('date', row.get('Date', '')),
                    'ticker': row.get('ticker', row.get('Ticker', row.get('Symbol', ''))),
                    'stock_name': row.get('stock_name', row.get('Stock Name', row.get('Name', ''))),
                    'scheme_name': row.get('scheme_name', None),
                    'quantity': float(row.get('quantity', row.get('Quantity', 0))),
                    'price': float(row.get('price', row.get('Price', 0))),
                    'transaction_type': str(row.get('transaction_type', row.get('Type', 'buy'))).lower(),
                    'asset_type': str(row.get('asset_type', row.get('Asset Type', 'stock'))).lower(),
                    'sector': row.get('sector', row.get('Sector', 'Unknown')),
                    'channel': row.get('channel', row.get('Channel', filename.replace('.csv', '')))
                }
                
                # Normalize date
                trans['date'] = self._normalize_date(trans['date'])
                
                transactions.append(trans)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Fallback parsing failed: {e}")
            return []
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent (required by BaseAgent)"""
        return self.parsed_transactions
    
    def get_parsed_transactions(self) -> List[Dict[str, Any]]:
        """Get parsed transactions"""
        return self.parsed_transactions

