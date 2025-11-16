"""
PMS and AIF Value Calculator
Calculates current value and 52-week historical values using CAGR from SEBI
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import requests
from bs4 import BeautifulSoup
import re
import json


class PMS_AIF_Calculator:
    """
    Calculate PMS/AIF values using CAGR from SEBI
    Formula: Current Value = Initial Investment × (1 + CAGR)^years_elapsed
    """
    
    def __init__(self):
        self.sebi_cache = {}
        self._openai_client = None
        self._openai_initialized = False
    
    def calculate_pms_aif_value(
        self,
        ticker: str,
        investment_date: str,
        investment_amount: float,
        is_aif: bool = False,
        pms_aif_name: str = None
    ) -> Dict[str, Any]:
        """
        Calculate current value and generate 52-week history for PMS/AIF using AI
        
        Args:
            ticker: PMS/AIF registration code (e.g., INP000005000)
            investment_date: Purchase date (YYYY-MM-DD)
            investment_amount: Initial investment (₹)
            is_aif: True if AIF, False if PMS
            pms_aif_name: Optional name of PMS/AIF for better AI context
        
        Returns:
            Dict with current_value, weekly_values, cagr_used, source
        """
        try:
            # Calculate time elapsed
            invest_dt = pd.to_datetime(investment_date)
            current_dt = datetime.now()
            years_elapsed = (current_dt - invest_dt).days / 365.25
            months_elapsed = years_elapsed * 12
            
            # CRITICAL: Use AI to get CAGR - no random/conservative values
            cagr_data = self._get_cagr_from_ai(ticker, is_aif, pms_aif_name, investment_date)
            
            if cagr_data and cagr_data.get('cagr') and cagr_data.get('cagr') > 0:
                # Use AI-provided CAGR
                cagr = cagr_data['cagr']
                cagr_period = cagr_data.get('period', 'AI Estimated')
                source = f"AI ({cagr_period})"
            else:
                # If AI fails, return error - don't use random values
                return {
                    'ticker': ticker,
                    'initial_investment': investment_amount,
                    'current_value': investment_amount,  # Return original investment if AI fails
                    'absolute_gain': 0,
                    'percentage_gain': 0,
                    'years_elapsed': years_elapsed,
                    'cagr_used': 0,
                    'cagr_period': 'AI Unavailable',
                    'source': 'AI fetch failed - using investment value',
                    'weekly_values': [],
                    'error': 'AI CAGR calculation failed'
                }
            
            # Calculate current value
            current_value = investment_amount * ((1 + cagr) ** years_elapsed)
            
            # Calculate absolute gain
            absolute_gain = current_value - investment_amount
            percentage_gain = (absolute_gain / investment_amount * 100) if investment_amount > 0 else 0
            
            # Generate 52-week historical values
            weekly_values = self._generate_weekly_values(
                investment_amount,
                invest_dt,
                cagr,
                ticker,
                'aif' if is_aif else 'pms',
                weeks=52
            )
            
            result = {
                'ticker': ticker,
                'initial_investment': investment_amount,
                'current_value': current_value,
                'absolute_gain': absolute_gain,
                'percentage_gain': percentage_gain,
                'years_elapsed': years_elapsed,
                'cagr_used': cagr,
                'cagr_period': cagr_period,
                'source': source,
                'weekly_values': weekly_values,
                'calculation_method': 'CAGR-based calculation'
            }
            
            ##st.caption(f"💰 {ticker}: ₹{investment_amount:,.0f} → ₹{current_value:,.0f} ({percentage_gain:.2f}%) using {cagr_period}")
            
            return result
            
        except Exception as e:
            # Log error (don't use st.error as it may not be available)
            print(f"[PMS_AIF] Error calculating value for {ticker}: {str(e)}")
            
            # Return zero-growth fallback
            return {
                'ticker': ticker,
                'initial_investment': investment_amount,
                'current_value': investment_amount,
                'absolute_gain': 0,
                'percentage_gain': 0,
                'cagr_used': 0,
                'source': 'Error - using investment value',
                'weekly_values': [],
                'error': str(e)
            }
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client"""
        if self._openai_initialized:
            return self._openai_client
        
        self._openai_initialized = True
        try:
            from openai import OpenAI
            if "api_keys" not in st.secrets:
                raise KeyError("'api_keys' not found in st.secrets")
            if "open_ai" not in st.secrets.get("api_keys", {}):
                raise KeyError("'open_ai' not found in st.secrets['api_keys']")
            self._openai_client = OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
            return self._openai_client
        except Exception as e:
            self._openai_client = None
            return None
    
    def _get_cagr_from_ai(
        self,
        ticker: str,
        is_aif: bool,
        pms_aif_name: str = None,
        investment_date: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get CAGR for PMS/AIF using AI
        
        Args:
            ticker: Registration code (e.g., INP000005000)
            is_aif: True for AIF, False for PMS
            pms_aif_name: Optional name for better context
            investment_date: Investment date for context
        
        Returns:
            Dict with cagr, period, source or None
        """
        client = self._get_openai_client()
        if not client:
            return None
        
        try:
            asset_type = "AIF" if is_aif else "PMS"
            name_context = f" named '{pms_aif_name}'" if pms_aif_name else ""
            date_context = f" invested on {investment_date}" if investment_date else ""
            
            prompt = f"""You are a financial data expert. I need the CAGR (Compound Annual Growth Rate) for a {asset_type} (Portfolio Management Service/Alternative Investment Fund) with registration code {ticker}{name_context}{date_context}.

IMPORTANT INSTRUCTIONS:
1. First, search for the specific {asset_type} product with registration code {ticker}
2. If found, use the actual CAGR data from SEBI, fund house website, or financial databases
3. If NOT found, provide a reasonable estimate based on:
   - Similar {asset_type} products from the same fund house (if name is provided)
   - Industry average CAGR for similar strategy types
   - Typical {asset_type} performance ranges (usually 8-25% CAGR for equity-focused, 6-12% for balanced)
4. DO NOT return null/not_found unless you have absolutely no way to estimate

For Indian PMS/AIF products:
- Equity-focused PMS typically have 12-20% CAGR
- Balanced PMS typically have 10-15% CAGR
- Large-cap focused typically have 10-18% CAGR
- Mid-cap focused typically have 15-25% CAGR
- AIF Category III typically have 12-22% CAGR

Please provide:
1. The CAGR percentage (as a decimal, e.g., 0.15 for 15%)
2. The period this CAGR is based on (e.g., "3Y", "5Y", "Since Inception", etc.)
3. The source of this data (e.g., "SEBI", "Fund House", "Industry Average", "Similar Product", "Estimated")
4. Whether this is exact data or estimated (use "exact" or "closest_available")

Return ONLY a JSON object with this exact format:
{{
    "cagr": 0.15,
    "period": "3Y",
    "source": "SEBI/Public Data",
    "data_type": "exact"
}}

OR if using closest available/estimated:
{{
    "cagr": 0.14,
    "period": "3Y",
    "source": "Industry Average/Estimated",
    "data_type": "closest_available"
}}

CRITICAL: Always provide a CAGR value (between 0.08 and 0.25 for typical Indian PMS/AIF). Only return null if you cannot provide any reasonable estimate."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial data expert. Provide accurate CAGR data for Indian PMS/AIF products. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            print(f"[PMS_AIF_AI] Raw AI response for {ticker}: {content[:200]}")
            result = json.loads(content)
            print(f"[PMS_AIF_AI] Parsed result: {result}")
            
            cagr_value = result.get('cagr')
            if cagr_value is not None and cagr_value != 'null':
                try:
                    cagr = float(cagr_value)
                    print(f"[PMS_AIF_AI] CAGR as float: {cagr}")
                    
                    # Allow CAGR up to 2.0 (200%) for exceptional cases
                    if 0 < cagr <= 2.0:
                        data_type = result.get('data_type', 'exact')
                        period = result.get('period', 'AI Estimated')
                        source = result.get('source', 'AI')
                        
                        # Log if using closest available data
                        if data_type == 'closest_available':
                            print(f"[PMS_AIF_AI] Using closest available CAGR {cagr*100:.1f}% ({period}) from {source}")
                        
                        return {
                            'cagr': cagr,
                            'period': period,
                            'source': source,
                            'data_type': data_type
                        }
                    else:
                        print(f"[PMS_AIF_AI] CAGR {cagr} is outside valid range (0-2.0)")
                except (ValueError, TypeError) as e:
                    print(f"[PMS_AIF_AI] Failed to convert CAGR to float: {cagr_value}, error: {e}")
            else:
                print(f"[PMS_AIF_AI] CAGR is None or null: {cagr_value}")
            
            return None
            
        except Exception as e:
            print(f"[PMS_AIF_AI] ⚠️ AI CAGR fetch failed for {ticker}: {str(e)}")
            return None
    
    def _get_sebi_cagr(self, ticker: str, is_aif: bool) -> Optional[Dict[str, Any]]:
        """
        Fetch CAGR from SEBI website
        
        Args:
            ticker: Registration code
            is_aif: True for AIF, False for PMS
        
        Returns:
            Dict with cagr and period or None
        """
        # Check cache first
        cache_key = f"{ticker}_{is_aif}"
        if cache_key in self.sebi_cache:
            return self.sebi_cache[cache_key]
        
        try:
            if is_aif:
                data = self._fetch_aif_from_sebi(ticker)
            else:
                data = self._fetch_pms_from_sebi(ticker)
            
            if data:
                # Extract best available CAGR
                cagr_result = self._extract_best_cagr(data)
                
                if cagr_result:
                    self.sebi_cache[cache_key] = cagr_result
                    return cagr_result
            
            return None
            
        except Exception as e:
            # SEBI fetch error
            return None
    
    def _extract_best_cagr(self, sebi_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Extract the best available CAGR from SEBI data
        Priority: 5Y > 3Y > 1Y > 6M > 3M > 1M
        
        Args:
            sebi_data: DataFrame with SEBI performance data
        
        Returns:
            Dict with cagr and period
        """
        if sebi_data.empty:
            return None
        
        first_record = sebi_data.iloc[0]
        
        # Priority order
        cagr_columns = [
            ('5Y CAGR', '5y_cagr'),
            ('3Y CAGR', '3y_cagr'),
            ('1Y Return', '1y_return'),
            ('6M Return', '6m_return'),
            ('3M Return', '3m_return'),
            ('1M Return', '1m_return')
        ]
        
        for col_name, result_key in cagr_columns:
            if col_name in first_record:
                value_str = str(first_record[col_name]).strip()
                
                if value_str and value_str != 'N/A' and value_str != '':
                    try:
                        # Parse percentage
                        cagr = float(value_str.replace('%', '').strip()) / 100
                        
                        # Annualize if needed
                        if '1M' in col_name:
                            cagr = ((1 + cagr) ** 12) - 1
                        elif '3M' in col_name:
                            cagr = ((1 + cagr) ** 4) - 1
                        elif '6M' in col_name:
                            cagr = ((1 + cagr) ** 2) - 1
                        
                        return {
                            'cagr': cagr,
                            'period': col_name,
                            'original_value': value_str
                        }
                    except ValueError:
                        continue
        
        return None
    
    def _fetch_pms_from_sebi(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch PMS data from SEBI website"""
        try:
            url = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doPmr=yes"
            
            # Try to read tables from SEBI page
            tables = pd.read_html(url)
            
            if tables:
                for table in tables:
                    # Look for the table with Registration No. column
                    if 'Registration No.' in table.columns:
                        # Search by registration code
                        matches = table[table['Registration No.'].astype(str).str.contains(ticker, na=False, case=False)]
                        
                        if not matches.empty:
                            return matches
            
            return None
            
        except Exception as e:
            # SEBI PMS fetch error
            return None
    
    def _fetch_aif_from_sebi(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch AIF data from SEBI website"""
        try:
            # AIF data URL (update if SEBI changes it)
            url = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=10"
            
            tables = pd.read_html(url)
            
            if tables:
                for table in tables:
                    # Search for ticker in all columns
                    for col in table.columns:
                        if table[col].astype(str).str.contains(ticker, na=False, case=False).any():
                            matches = table[table[col].astype(str).str.contains(ticker, na=False, case=False)]
                            if not matches.empty:
                                return matches
            
            return None
            
        except Exception as e:
            # SEBI AIF fetch error
            return None
    
    def _generate_weekly_values(
        self,
        initial_investment: float,
        investment_date: datetime,
        cagr: float,
        ticker: str,
        asset_type: str,
        weeks: int = 52
    ) -> List[Dict[str, Any]]:
        """
        Generate weekly NAVs for the past 52 weeks using CAGR
        
        Args:
            initial_investment: Initial amount invested
            investment_date: Date of investment
            cagr: Annual CAGR rate (as decimal, e.g., 0.15 for 15%)
            ticker: Ticker symbol
            asset_type: 'pms' or 'aif'
            weeks: Number of weeks to generate
        
        Returns:
            List of dicts with price_date, price (total NAV), asset_symbol, asset_type
        """
        weekly_values = []
        
        try:
            current_dt = datetime.now()
            
            # Generate weekly dates (Mondays)
            weekly_dates = pd.date_range(
                end=current_dt,
                periods=weeks,
                freq='W-MON'
            )
            
            for week_date in weekly_dates:
                # Calculate years from investment date to this week
                years_to_week = (week_date - investment_date).days / 365.25
                
                if years_to_week < 0:
                    # Before investment date - no value
                    continue
                
                # Calculate total portfolio value at this week using CAGR
                # This is the NAV (Net Asset Value) for the entire PMS/AIF investment
                nav_at_week = initial_investment * ((1 + cagr) ** years_to_week)
                
                weekly_values.append({
                    'asset_symbol': ticker,
                    'asset_type': asset_type,
                    'price': round(nav_at_week, 2),  # Total NAV
                    'price_date': week_date.strftime('%Y-%m-%d'),
                    'volume': None
                })
            
            return weekly_values
            
        except Exception as e:
            #st.caption(f"⚠️ Error generating weekly values: {str(e)}")
            return []
    
    def calculate_pms_aif_for_holding(self, holding: Dict[str, Any], transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate PMS/AIF values for a holding based on transaction data
        
        Args:
            holding: Holding record
            transaction: First transaction record (buy)
        
        Returns:
            Dict with calculated values
        """
        try:
            ticker = holding['asset_symbol']
            is_aif = 'aif' in holding['asset_type'].lower()
            
            # Get investment details from transaction
            quantity = float(transaction['quantity'])
            price = float(transaction['price'])
            investment_date = transaction['transaction_date']
            investment_amount = quantity * price
            
            # Calculate values
            result = self.calculate_pms_aif_value(
                ticker,
                investment_date,
                investment_amount,
                is_aif
            )
            
            # Calculate "price per unit" for display
            # Since PMS/AIF are usually 1 unit purchases, the current_value IS the price
            current_price = result['current_value'] / quantity if quantity > 0 else result['current_value']
            
            return {
                'current_price': current_price,
                'current_value': result['current_value'],
                'absolute_gain': result['absolute_gain'],
                'percentage_gain': result['percentage_gain'],
                'cagr_used': result['cagr_used'],
                'source': result['source'],
                'weekly_values': result['weekly_values']
            }
            
        except Exception as e:
            #st.caption(f"⚠️ Error calculating PMS/AIF for {holding['asset_symbol']}: {str(e)}")
            
            # Return zero-growth fallback
            return {
                'current_price': float(transaction['price']),
                'current_value': float(transaction['quantity']) * float(transaction['price']),
                'absolute_gain': 0,
                'percentage_gain': 0,
                'cagr_used': 0,
                'source': 'Error - using transaction price',
                'weekly_values': []
            }


def fetch_pms_cagr_from_sebi(registration_code: str) -> Optional[Dict[str, Any]]:
    """
    Simplified SEBI PMS CAGR fetcher
    
    Args:
        registration_code: PMS code (e.g., INP000005000)
    
    Returns:
        Dict with CAGR data or None
    """
    try:
        url = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doPmr=yes"
        
        # Try to read the PMS table from SEBI
        tables = pd.read_html(url)
        
        if not tables:
            return None
        
        # Find the main PMS table (usually the largest one)
        pms_table = max(tables, key=lambda x: len(x))
        
        # Search by registration code
        if 'Registration No.' in pms_table.columns:
            matches = pms_table[
                pms_table['Registration No.'].astype(str).str.contains(
                    registration_code, 
                    na=False, 
                    case=False
                )
            ]
            
            if not matches.empty:
                record = matches.iloc[0]
                
                # Extract CAGR values
                result = {
                    'ticker': registration_code,
                    'name': record.get('Portfolio Manager', 'Unknown'),
                    'strategy': record.get('Strategy', 'N/A')
                }
                
                # Extract performance metrics
                performance_cols = {
                    '5Y CAGR': '5y_cagr',
                    '3Y CAGR': '3y_cagr',
                    '1Y Return': '1y_return'
                }
                
                for col, key in performance_cols.items():
                    if col in record:
                        value_str = str(record[col]).strip()
                        if value_str and value_str != 'N/A':
                            try:
                                result[key] = float(value_str.replace('%', '').strip()) / 100
                            except:
                                pass
                
                return result
        
        return None
        
    except Exception as e:
        return None


def calculate_current_value_from_cagr(
    initial_investment: float,
    investment_date: str,
    cagr: float
) -> Dict[str, Any]:
    """
    Calculate current value using CAGR
    
    Args:
        initial_investment: Initial amount
        investment_date: Purchase date (YYYY-MM-DD)
        cagr: Annual CAGR (as decimal, e.g., 0.15 for 15%)
    
    Returns:
        Dict with calculation details
    """
    invest_dt = pd.to_datetime(investment_date)
    current_dt = datetime.now()
    years_elapsed = (current_dt - invest_dt).days / 365.25
    
    # Formula: FV = PV × (1 + r)^t
    current_value = initial_investment * ((1 + cagr) ** years_elapsed)
    
    absolute_gain = current_value - initial_investment
    percentage_gain = (absolute_gain / initial_investment * 100) if initial_investment > 0 else 0
    
    return {
        'initial_investment': initial_investment,
        'current_value': current_value,
        'absolute_gain': absolute_gain,
        'percentage_gain': percentage_gain,
        'years_elapsed': years_elapsed,
        'cagr_used': cagr
    }


def generate_52_week_values(
    initial_investment: float,
    investment_date: str,
    cagr: float,
    ticker: str,
    asset_type: str
) -> List[Dict[str, Any]]:
    """
    Generate 52 weeks of NAVs using CAGR
    
    Args:
        initial_investment: Initial amount
        investment_date: Purchase date
        cagr: Annual CAGR (as decimal, e.g., 0.15 for 15%)
        ticker: Ticker symbol
        asset_type: 'pms' or 'aif'
    
    Returns:
        List of weekly NAV records
    """
    weekly_values = []
    
    invest_dt = pd.to_datetime(investment_date)
    current_dt = datetime.now()
    
    # Generate 52 weekly dates (Mondays)
    weekly_dates = pd.date_range(
        end=current_dt,
        periods=52,
        freq='W-MON'
    )
    
    for week_date in weekly_dates:
        # Calculate years from investment to this week
        years_to_week = (week_date - invest_dt).days / 365.25
        
        if years_to_week < 0:
            # Before investment - no value
            continue
        
        # Calculate total NAV at this week using CAGR
        nav_at_week = initial_investment * ((1 + cagr) ** years_to_week)
        
        weekly_values.append({
            'asset_symbol': ticker,
            'asset_type': asset_type,
            'price': round(nav_at_week, 2),  # Total NAV
            'price_date': week_date.strftime('%Y-%m-%d'),
            'volume': None
        })
    
    return weekly_values


# ============================================================================
# SIMPLIFIED API FOR USE IN APP
# ============================================================================

def get_pms_current_value(ticker: str, investment_date: str, investment_amount: float, quantity: float = 1.0) -> Dict[str, Any]:
    """
    Simple API to get PMS current value
    
    Returns:
        Dict with current_price (per unit) and calculation details
    """
    calculator = PMS_AIF_Calculator()
    result = calculator.calculate_pms_aif_value(ticker, investment_date, investment_amount, is_aif=False)
    
    # Calculate per-unit price
    current_price = result['current_value'] / quantity if quantity > 0 else result['current_value']
    
    return {
        'current_price': current_price,
        'total_value': result['current_value'],
        'gain': result['absolute_gain'],
        'gain_pct': result['percentage_gain'],
        'cagr': result['cagr_used'],
        'source': result['source'],
        'weekly_values': result['weekly_values']
    }


def get_aif_current_value(ticker: str, investment_date: str, investment_amount: float, quantity: float = 1.0) -> Dict[str, Any]:
    """
    Simple API to get AIF current value
    
    Returns:
        Dict with current_price (per unit) and calculation details
    """
    calculator = PMS_AIF_Calculator()
    result = calculator.calculate_pms_aif_value(ticker, investment_date, investment_amount, is_aif=True)
    
    # Calculate per-unit price
    current_price = result['current_value'] / quantity if quantity > 0 else result['current_value']
    
    return {
        'current_price': current_price,
        'total_value': result['current_value'],
        'gain': result['absolute_gain'],
        'gain_pct': result['percentage_gain'],
        'cagr': result['cagr_used'],
        'source': result['source'],
        'weekly_values': result['weekly_values']
    }

