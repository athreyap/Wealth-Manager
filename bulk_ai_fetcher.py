"""
Bulk AI Price Fetcher
Fetches prices for multiple tickers in ONE AI call (10x faster!)
"""

import streamlit as st
from openai import OpenAI
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

class BulkAIFetcher:
    """
    Fetch prices for multiple tickers in bulk using AI
    10 tickers in one API call = 10x faster!
    """
    
    def __init__(self):
        try:
            self.client = OpenAI(api_key=st.secrets["api_keys"]["openai"])
            self.available = True
        except Exception as e:
            self.available = False
            #st.caption(f"⚠️ Bulk AI fetcher not available: {str(e)}")
    
    def fetch_bulk_current_prices(
        self,
        tickers_with_info: List[Tuple[str, str, str]],  # [(ticker, name, asset_type), ...]
        max_batch: int = 10
    ) -> Dict[str, float]:
        """
        Fetch current prices for multiple tickers in one AI call
        
        Args:
            tickers_with_info: List of (ticker, stock_name, asset_type) tuples
            max_batch: Max tickers per AI call (default 10)
        
        Returns:
            Dict mapping ticker to price
        """
        if not self.available:
            return {}
        
        all_prices = {}
        
        # Process in batches of max_batch
        for i in range(0, len(tickers_with_info), max_batch):
            batch = tickers_with_info[i:i+max_batch]
            
            try:
                batch_prices = self._fetch_batch_current(batch)
                all_prices.update(batch_prices)
            except Exception as e:
                pass
#st.caption(f"⚠️ Batch {i//max_batch + 1} failed: {str(e)}")
        
        return all_prices
    
    def _fetch_batch_current(self, batch: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Fetch prices for one batch"""
        
        # Build ticker list for prompt
        ticker_list = []
        for idx, (ticker, name, asset_type) in enumerate(batch, 1):
            if asset_type == 'mutual_fund':
                ticker_list.append(f"{idx}. MF Code: {ticker} ({name})")
            else:
                ticker_list.append(f"{idx}. {ticker} - {name}")
        
        ticker_str = '\n'.join(ticker_list)
        
        system_prompt = """You are a financial data API that returns current prices for Indian securities.
You must return ONLY valid JSON with ticker->price mapping.
For mutual funds, return NAV. For stocks, return market price."""
        
        user_prompt = f"""Find CURRENT prices for these Indian securities:

{ticker_str}

Return ONLY valid JSON mapping ticker to price:
{{
  "RELIANCE": 2650.50,
  "TCS": 3800.00,
  "120760": 85.43
}}

Rules:
- Return ONLY the JSON object, no other text
- Use ticker as key (not full name)
- Price as number (no currency symbols)
- Use null if ticker not found
- No explanations or comments"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # gpt-4o for better accuracy and structured JSON
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}  # Force JSON
            )
            
            content = response.choices[0].message.content
            prices = json.loads(content)
            
            # Validate and clean
            clean_prices = {}
            for ticker, price in prices.items():
                if price and price != 'null':
                    try:
                        price_val = float(price)
                        if 0 < price_val < 1000000:  # Sanity check
                            clean_prices[ticker] = price_val
                    except (ValueError, TypeError):
                        pass
            
            #st.caption(f"🤖 Bulk AI: Fetched {len(clean_prices)}/{len(batch)} prices")
            return clean_prices
            
        except Exception as e:
            #st.caption(f"⚠️ Bulk AI error: {str(e)}")
            return {}
    
    def fetch_bulk_historical_prices(
        self,
        tickers_with_dates: List[Tuple[str, str, str, str]],  # [(ticker, name, asset_type, date), ...]
        max_batch: int = 10
    ) -> Dict[Tuple[str, str], float]:
        """
        Fetch historical prices for multiple tickers+dates in bulk
        
        Args:
            tickers_with_dates: List of (ticker, name, asset_type, date) tuples
            max_batch: Max per AI call
        
        Returns:
            Dict mapping (ticker, date) to price
        """
        if not self.available:
            return {}
        
        all_prices = {}
        
        # Process in batches
        for i in range(0, len(tickers_with_dates), max_batch):
            batch = tickers_with_dates[i:i+max_batch]
            
            try:
                batch_prices = self._fetch_batch_historical(batch)
                all_prices.update(batch_prices)
            except Exception as e:
                pass
#st.caption(f"⚠️ Historical batch {i//max_batch + 1} failed: {str(e)}")
        
        return all_prices
    
    def _fetch_batch_historical(self, batch: List[Tuple[str, str, str, str]]) -> Dict[Tuple[str, str], float]:
        """Fetch historical prices for one batch"""
        
        # Build request list
        request_list = []
        for idx, (ticker, name, asset_type, date) in enumerate(batch, 1):
            if asset_type == 'mutual_fund':
                request_list.append(f"{idx}. MF {ticker} ({name}) on {date}")
            else:
                request_list.append(f"{idx}. {ticker} ({name}) on {date}")
        
        request_str = '\n'.join(request_list)
        
        system_prompt = """You are a financial data API returning historical prices.
Return ONLY valid JSON with combined key->price mapping."""
        
        user_prompt = f"""Find historical prices for these securities:

{request_str}

Return JSON with "TICKER|DATE" as key:
{{
  "RELIANCE|2024-10-07": 2650.50,
  "TCS|2024-10-07": 3800.00,
  "120760|2024-10-07": 85.43
}}

Rules:
- Format: "TICKER|YYYY-MM-DD": price
- ONLY the JSON, no other text
- Price as number (no symbols)
- Use null if not found"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # gpt-4o
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            prices = json.loads(content)
            
            # Parse and clean
            clean_prices = {}
            for key, price in prices.items():
                if '|' in key and price and price != 'null':
                    try:
                        ticker, date = key.split('|')
                        price_val = float(price)
                        if 0 < price_val < 1000000:
                            clean_prices[(ticker, date)] = price_val
                    except (ValueError, TypeError):
                        pass
            
            #st.caption(f"🤖 Bulk AI Historical: Fetched {len(clean_prices)}/{len(batch)} prices")
            return clean_prices
            
        except Exception as e:
            #st.caption(f"⚠️ Bulk historical AI error: {str(e)}")
            return {}
    
    def fetch_missing_weeks_bulk(
        self,
        ticker_week_list: List[Dict[str, Any]]  # [{ticker, name, asset_type, week_dates[]}, ...]
    ) -> Dict[Tuple[str, str], float]:
        """
        Fetch multiple weeks for multiple tickers efficiently
        Groups by date for bulk fetching
        
        Args:
            ticker_week_list: List of dicts with ticker info and missing week dates
        
        Returns:
            Dict mapping (ticker, date) to price
        """
        # Group by date for efficient fetching
        date_batches = {}
        
        for item in ticker_week_list:
            ticker = item['ticker']
            name = item['name']
            asset_type = item['asset_type']
            
            for week_date in item['week_dates']:
                if week_date not in date_batches:
                    date_batches[week_date] = []
                
                date_batches[week_date].append((ticker, name, asset_type, week_date))
        
        # Fetch each date batch
        all_prices = {}
        
        for date, batch in date_batches.items():
            if len(batch) <= 10:
                # Fetch entire batch
                prices = self._fetch_batch_historical(batch)
                all_prices.update(prices)
            else:
                # Split into multiple batches
                for i in range(0, len(batch), 10):
                    sub_batch = batch[i:i+10]
                    prices = self._fetch_batch_historical(sub_batch)
                    all_prices.update(prices)
        
        return all_prices

