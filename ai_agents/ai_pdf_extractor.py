"""
AI-Powered PDF Transaction Extractor
Uses OpenAI to intelligently extract transaction data from PDF files
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, MessagePriority

class AIPDFTransactionExtractor(BaseAgent):
    """
    AI-powered agent for extracting transaction data from PDF files
    """
    
    def __init__(self, agent_id: str = "ai_pdf_extractor"):
        super().__init__(agent_id, "AI PDF Transaction Extractor")
        self.capabilities = [
            "ai_pdf_analysis",
            "intelligent_transaction_extraction",
            "data_validation",
            "format_understanding",
            "context_aware_parsing"
        ]
        
        # Initialize OpenAI client
        try:
            import streamlit as st
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        self.extracted_transactions = []
        self.extraction_metadata = {}
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method for AI-powered PDF transaction extraction
        """
        try:
            self.update_status("analyzing")
            
            pdf_text = data.get("pdf_text", "")
            pdf_filename = data.get("pdf_filename", "unknown.pdf")
            user_id = data.get("user_id")
            
            if not pdf_text:
                return self.format_response([], "low", error="No PDF text provided")
            
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to extract transactions
            transactions = self._ai_extract_transactions(pdf_text, pdf_filename)
            
            # Validate and clean transactions
            validated_transactions = self._validate_ai_transactions(transactions)
            
            # Store extracted transactions
            self.extracted_transactions = validated_transactions
            
            # Generate insights about extraction
            insights = self._generate_ai_extraction_insights(validated_transactions, pdf_filename)
            
            self.update_status("active")
            return self.format_response(insights, "medium")
            
        except Exception as e:
            self.logger.error(f"Error in AI PDF extraction: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_extract_transactions(self, pdf_text: str, filename: str) -> List[Dict[str, Any]]:
        """Use AI to extract transaction data from PDF text"""
        
        # Truncate text if too long (OpenAI has token limits)
        max_chars = 50000  # Leave room for prompt and response
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[Text truncated for processing...]"
        
        # Create AI prompt for transaction extraction
        prompt = self._create_extraction_prompt(pdf_text, filename)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better PDF extraction
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial data extraction specialist. Your task is to extract transaction data from financial documents with high accuracy.

IMPORTANT RULES:
1. Extract ONLY actual transactions (buy/sell/dividend/bonus)
2. Ignore headers, footers, summaries, and non-transaction data
3. Be precise with dates, amounts, and quantities
4. Identify asset types correctly (stocks, mutual funds, PMS, AIF)
5. Extract channel/broker information from context
6. Return data in the exact JSON format requested

For each transaction, provide:
- date: Transaction date (YYYY-MM-DD format)
- ticker: Stock symbol (if applicable)
- stock_name: Full stock name (if applicable) - NOT filename or channel name
- scheme_name: Mutual fund/PMS scheme name (if applicable) - NOT filename or channel name
- quantity: Number of units/shares
- price: Price per unit/share
- amount: Total transaction amount
- transaction_type: buy/sell/dividend/bonus
- asset_type: stock/mutual_fund/pms/aif
- channel: Broker/fund house name
- sector: Industry sector (if identifiable)

CRITICAL: Do NOT use filename or channel/broker name as stock_name or scheme_name. Extract the actual fund/security name from the document content.

Be conservative - only extract data you're confident about."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=4000,
                # Note: GPT-5 only supports default temperature (1)
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            transactions = self._parse_ai_response(ai_response)
            
            # Store metadata
            self.extraction_metadata = {
                "ai_model": "gpt-4o",
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "extraction_timestamp": datetime.now().isoformat(),
                "filename": filename
            }
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"AI extraction failed: {e}")
            return []
    
    def _create_extraction_prompt(self, pdf_text: str, filename: str) -> str:
        """Create a detailed prompt for AI transaction extraction"""
        
        return f"""Extract all financial transactions from the following document:

FILENAME: {filename}

DOCUMENT TEXT:
{pdf_text}

Please analyze this document and extract ALL financial transactions. Return the data as a JSON array with the following structure:

[
  {{
    "date": "2024-01-15",
    "ticker": "INFY.NS",
    "stock_name": "Infosys Limited",
    "scheme_name": null,
    "quantity": 100,
    "price": 1500.50,
    "amount": 150050,
    "transaction_type": "buy",
    "asset_type": "stock",
    "channel": "Zerodha",
    "sector": "Technology"
  }},
  {{
    "date": "2024-01-20",
    "ticker": null,
    "stock_name": null,
    "scheme_name": "HDFC Equity Fund",
    "quantity": 500,
    "price": 45.20,
    "amount": 22600,
    "transaction_type": "buy",
    "asset_type": "mutual_fund",
    "channel": "HDFC Mutual Fund",
    "sector": null
  }}
]

GUIDELINES:
1. Extract ONLY actual transactions (buy/sell/dividend/bonus)
2. Use YYYY-MM-DD format for dates
3. Include ticker symbols for stocks (e.g., INFY.NS, TCS.NS)
4. For mutual funds, use scheme_name instead of ticker
5. For PMS/AIF, use scheme_name and set asset_type accordingly
6. Extract channel from context (broker name, fund house, etc.)
7. If sector is identifiable, include it
8. Be precise with numbers - no rounding unless original data is rounded
9. If a field is not available, use null
10. CRITICAL: Do NOT use filename "{filename}" or channel/broker name as stock_name or scheme_name. Extract the actual fund/security name from the document. If stock_name/scheme_name is missing or looks like a channel name, set it to null.
11. Return ONLY the JSON array, no additional text

Focus on accuracy over quantity. It's better to extract fewer transactions correctly than many with errors."""
    
    def _parse_ai_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract transaction data"""
        try:
            # Clean the response
            ai_response = ai_response.strip()
            
            # Remove any markdown formatting
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            # Try to find JSON array in the response
            json_start = ai_response.find('[')
            json_end = ai_response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                transactions = json.loads(json_str)
                
                # Validate that it's a list
                if isinstance(transactions, list):
                    return transactions
                else:
                    self.logger.error("AI response is not a list")
                    return []
            else:
                self.logger.error("No JSON array found in AI response")
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"AI response: {ai_response[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {e}")
            return []
    
    def _validate_ai_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean AI-extracted transactions"""
        validated = []
        
        for i, transaction in enumerate(transactions):
            try:
                # Basic validation
                if not transaction.get('date'):
                    continue
                
                if not transaction.get('amount') and not transaction.get('quantity'):
                    continue
                
                # Clean and format data
                cleaned_transaction = {
                    'id': f"ai_extracted_{i + 1}",
                    'date': self._clean_date(transaction.get('date')),
                    'ticker': transaction.get('ticker'),
                    'stock_name': transaction.get('stock_name'),
                    'scheme_name': transaction.get('scheme_name'),
                    'quantity': self._clean_number(transaction.get('quantity')),
                    'price': self._clean_number(transaction.get('price')),
                    'amount': self._clean_number(transaction.get('amount')),
                    'transaction_type': self._clean_transaction_type(transaction.get('transaction_type')),
                    'asset_type': self._clean_asset_type(transaction.get('asset_type')),
                    'channel': self._clean_channel(transaction.get('channel')),
                    'sector': transaction.get('sector'),
                    'extraction_method': 'ai_powered',
                    'confidence': 'high'  # AI extraction is generally high confidence
                }
                
                # Calculate missing fields
                if not cleaned_transaction['amount'] and cleaned_transaction['quantity'] and cleaned_transaction['price']:
                    cleaned_transaction['amount'] = cleaned_transaction['quantity'] * cleaned_transaction['price']
                
                if not cleaned_transaction['price'] and cleaned_transaction['amount'] and cleaned_transaction['quantity']:
                    cleaned_transaction['price'] = cleaned_transaction['amount'] / cleaned_transaction['quantity']
                
                validated.append(cleaned_transaction)
                
            except Exception as e:
                self.logger.error(f"Error validating transaction {i}: {e}")
                continue
        
        return validated
    
    def _clean_date(self, date_str: str) -> Optional[str]:
        """Clean and validate date string"""
        if not date_str:
            return None
        
        try:
            # Try to parse various date formats
            date_formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%d-%m-%Y',
                '%m/%d/%Y',
                '%d %b %Y',
                '%d %B %Y'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If no format matches, try to extract date components
            date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_str)
            if date_match:
                day, month, year = date_match.groups()
                if len(year) == 2:
                    year = '20' + year
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return None
            
        except Exception:
            return None
    
    def _clean_number(self, value: Any) -> Optional[float]:
        """Clean and convert number values"""
        if value is None:
            return None
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[₹,INR\s]', '', value)
                return float(cleaned)
            
            return None
        except (ValueError, TypeError):
            return None
    
    def _clean_transaction_type(self, transaction_type: str) -> str:
        """Clean transaction type"""
        if not transaction_type:
            return 'buy'
        
        transaction_type = transaction_type.lower().strip()
        
        if transaction_type in ['buy', 'purchase', 'bought', 'acquired']:
            return 'buy'
        elif transaction_type in ['sell', 'sold', 'disposed', 'exit']:
            return 'sell'
        elif transaction_type in ['dividend', 'bonus', 'split']:
            return 'dividend'
        else:
            return 'buy'  # Default to buy
    
    def _clean_asset_type(self, asset_type: str) -> str:
        """Clean asset type"""
        if not asset_type:
            return 'stock'
        
        asset_type = asset_type.lower().strip()
        
        if asset_type in ['stock', 'equity', 'shares']:
            return 'stock'
        elif asset_type in ['mutual_fund', 'mf', 'mutual fund']:
            return 'mutual_fund'
        elif asset_type in ['pms', 'portfolio management']:
            return 'pms'
        elif asset_type in ['aif', 'alternative investment']:
            return 'aif'
        else:
            return 'stock'  # Default to stock
    
    def _clean_channel(self, channel: str) -> str:
        """Clean channel name"""
        if not channel:
            return 'Direct'
        
        channel = channel.strip()
        
        # Map common variations
        channel_mappings = {
            'hdfc bank': 'HDFC',
            'hdfc mutual fund': 'HDFC Mutual Fund',
            'icici bank': 'ICICI',
            'icici prudential': 'ICICI Prudential',
            'sbi': 'SBI',
            'state bank': 'SBI',
            'zerodha': 'Zerodha',
            'upstox': 'Upstox',
            'angel broking': 'Angel Broking',
            'kotak': 'Kotak',
            'axis': 'Axis'
        }
        
        channel_lower = channel.lower()
        for key, value in channel_mappings.items():
            if key in channel_lower:
                return value
        
        return channel
    
    def _generate_ai_extraction_insights(self, transactions: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Generate insights about the AI extraction process"""
        insights = []
        
        if not transactions:
            insights.append({
                "type": "ai_extraction_summary",
                "severity": "medium",
                "title": "No Transactions Found by AI",
                "description": f"AI analysis of {filename} found no extractable transactions",
                "recommendation": "The document may not contain transaction data or may be in an unrecognized format",
                "data": {
                    "filename": filename,
                    "transactions_found": 0,
                    "extraction_method": "ai_powered",
                    "ai_model": self.extraction_metadata.get("ai_model", "unknown")
                }
            })
        else:
            # Summary insights
            total_amount = sum(t.get('amount', 0) for t in transactions)
            asset_types = list(set(t.get('asset_type', 'unknown') for t in transactions))
            channels = list(set(t.get('channel', 'unknown') for t in transactions))
            
            insights.append({
                "type": "ai_extraction_summary",
                "severity": "low",
                "title": f"AI Successfully Extracted {len(transactions)} Transactions",
                "description": f"AI analysis found {len(transactions)} transactions from {filename} with total value ₹{total_amount:,.0f}",
                "recommendation": "Review the AI-extracted transactions and upload to your portfolio",
                "data": {
                    "filename": filename,
                    "transactions_found": len(transactions),
                    "total_amount": total_amount,
                    "asset_types": asset_types,
                    "channels": channels,
                    "extraction_method": "ai_powered",
                    "ai_model": self.extraction_metadata.get("ai_model", "unknown"),
                    "tokens_used": self.extraction_metadata.get("tokens_used", 0)
                }
            })
            
            # Asset type breakdown
            for asset_type in asset_types:
                type_transactions = [t for t in transactions if t.get('asset_type') == asset_type]
                type_amount = sum(t.get('amount', 0) for t in type_transactions)
                
                insights.append({
                    "type": "ai_asset_analysis",
                    "severity": "low",
                    "title": f"AI Found {len(type_transactions)} {asset_type.replace('_', ' ').title()} Transactions",
                    "description": f"AI identified {len(type_transactions)} {asset_type} transactions worth ₹{type_amount:,.0f}",
                    "recommendation": f"These {asset_type} transactions can be added to your portfolio",
                    "data": {
                        "asset_type": asset_type,
                        "count": len(type_transactions),
                        "total_amount": type_amount,
                        "extraction_method": "ai_powered"
                    }
                })
            
            # Channel analysis
            for channel in channels:
                channel_transactions = [t for t in transactions if t.get('channel') == channel]
                channel_amount = sum(t.get('amount', 0) for t in channel_transactions)
                
                insights.append({
                    "type": "ai_channel_analysis",
                    "severity": "low",
                    "title": f"AI Found {len(channel_transactions)} Transactions via {channel}",
                    "description": f"AI identified {len(channel_transactions)} transactions through {channel} worth ₹{channel_amount:,.0f}",
                    "recommendation": f"Transactions from {channel} can be grouped together in your portfolio",
                    "data": {
                        "channel": channel,
                        "count": len(channel_transactions),
                        "total_amount": channel_amount,
                        "extraction_method": "ai_powered"
                    }
                })
        
        return insights
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent"""
        return self.extracted_transactions
    
    def get_extracted_transactions(self) -> List[Dict[str, Any]]:
        """Get the AI-extracted transactions"""
        return self.extracted_transactions
    
    def get_extraction_metadata(self) -> Dict[str, Any]:
        """Get metadata about the extraction process"""
        return self.extraction_metadata
    
    def handle_message(self, message):
        """Handle incoming messages from other agents"""
        self.logger.info(f"AI PDF Extractor received message: {message.message_type.value}")
        
        if message.message_type == MessageType.REQUEST:
            if message.content.get("request_type") == "ai_extract_transactions":
                response = self.analyze({
                    "pdf_text": message.content.get("pdf_text", ""),
                    "pdf_filename": message.content.get("pdf_filename", ""),
                    "user_id": message.content.get("user_id")
                })
                return response
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI extraction process"""
        return {
            "transactions_extracted": len(self.extracted_transactions),
            "ai_model": self.extraction_metadata.get("ai_model", "unknown"),
            "tokens_used": self.extraction_metadata.get("tokens_used", 0),
            "last_extraction": self.last_update.isoformat() if self.last_update else None,
            "capabilities": self.capabilities,
            "extraction_method": "ai_powered"
        }
