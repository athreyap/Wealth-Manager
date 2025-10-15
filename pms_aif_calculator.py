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


class PMS_AIF_Calculator:
    """
    Calculate PMS/AIF values using CAGR from SEBI
    Formula: Current Value = Initial Investment × (1 + CAGR)^years_elapsed
    """
    
    def __init__(self):
        self.sebi_cache = {}
        self.conservative_pms_cagr = 0.10  # 10% conservative estimate
        self.conservative_aif_cagr = 0.12  # 12% conservative estimate
    
    def calculate_pms_aif_value(
        self,
        ticker: str,
        investment_date: str,
        investment_amount: float,
        is_aif: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate current value and generate 52-week history for PMS/AIF
        
        Args:
            ticker: PMS/AIF registration code (e.g., INP000005000)
            investment_date: Purchase date (YYYY-MM-DD)
            investment_amount: Initial investment (₹)
            is_aif: True if AIF, False if PMS
        
        Returns:
            Dict with current_value, weekly_values, cagr_used, source
        """
        try:
            # Calculate time elapsed
            invest_dt = pd.to_datetime(investment_date)
            current_dt = datetime.now()
            years_elapsed = (current_dt - invest_dt).days / 365.25
            months_elapsed = years_elapsed * 12
            
            # Try to get CAGR from SEBI
            cagr_data = self._get_sebi_cagr(ticker, is_aif)
            
            if cagr_data and cagr_data.get('cagr'):
                # Use SEBI CAGR
                cagr = cagr_data['cagr']
                cagr_period = cagr_data['period']
                source = f"SEBI ({cagr_period})"
            else:
                # Use conservative estimate
                cagr = self.conservative_aif_cagr if is_aif else self.conservative_pms_cagr
                cagr_period = "Conservative Estimate"
                source = "Estimated (SEBI data unavailable)"
                st.caption(f"⚠️ Using conservative {cagr*100:.0f}% CAGR for {ticker}")
            
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
            
            st.caption(f"💰 {ticker}: ₹{investment_amount:,.0f} → ₹{current_value:,.0f} ({percentage_gain:.2f}%) using {cagr_period}")
            
            return result
            
        except Exception as e:
            st.error(f"Error calculating PMS/AIF value for {ticker}: {str(e)}")
            
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
            st.caption(f"⚠️ SEBI fetch error for {ticker}: {str(e)}")
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
            st.caption(f"⚠️ SEBI PMS fetch error: {str(e)}")
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
            st.caption(f"⚠️ SEBI AIF fetch error: {str(e)}")
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
            st.caption(f"⚠️ Error generating weekly values: {str(e)}")
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
            st.caption(f"⚠️ Error calculating PMS/AIF for {holding['asset_symbol']}: {str(e)}")
            
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

