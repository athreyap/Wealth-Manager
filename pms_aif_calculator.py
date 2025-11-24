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
        Calculate current value and generate 52-week history for PMS/AIF using AI to fetch actual NAVs
        
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
            
            # CRITICAL: Use AI to fetch actual NAVs instead of calculating from CAGR
            print(f"[PMS_AIF] 🔍 Fetching actual NAVs for {ticker} ({'AIF' if is_aif else 'PMS'}) - Investment: Rs. {investment_amount:,.2f} on {investment_date}")
            nav_data = self._get_navs_from_ai(ticker, is_aif, pms_aif_name, investment_date, investment_amount)
            
            if nav_data and nav_data.get('navs') and len(nav_data.get('navs', [])) > 0:
                # Use AI-provided NAVs
                navs = nav_data['navs']
                current_nav = nav_data.get('current_nav', navs[-1].get('nav', 0) if navs else 0)
                source = nav_data.get('source', 'AI')
                print(f"[PMS_AIF] Got {len(navs)} NAV values from AI for {ticker}, current NAV: Rs. {current_nav:,.2f}")
                
                # Calculate current value based on NAV
                initial_nav = navs[0].get('nav', 0) if navs else investment_amount
                if initial_nav > 0:
                    units = investment_amount / initial_nav
                    current_value = units * current_nav
                else:
                    current_value = investment_amount
                
                # Calculate CAGR from NAVs if we have enough data
                cagr_used = 0
                if len(navs) >= 2:
                    first_nav = navs[0].get('nav', 0)
                    last_nav = navs[-1].get('nav', 0)
                    if first_nav > 0 and last_nav > 0:
                        # Calculate CAGR from first to last NAV
                        weeks_elapsed = len(navs) - 1
                        years_elapsed_nav = weeks_elapsed / 52.0
                        if years_elapsed_nav > 0:
                            cagr_used = ((last_nav / first_nav) ** (1 / years_elapsed_nav)) - 1
                
                # Format weekly_values to match expected structure (price_date, price, asset_symbol, asset_type)
                formatted_weekly_values = []
                for nav_entry in navs:
                    formatted_weekly_values.append({
                        'price_date': nav_entry.get('date'),
                        'price': nav_entry.get('nav'),
                        'asset_symbol': ticker,
                        'asset_type': 'aif' if is_aif else 'pms'
                    })
                
                return {
                    'ticker': ticker,
                    'initial_investment': investment_amount,
                    'current_value': current_value,
                    'absolute_gain': current_value - investment_amount,
                    'percentage_gain': ((current_value - investment_amount) / investment_amount * 100) if investment_amount > 0 else 0,
                    'years_elapsed': years_elapsed,
                    'cagr_used': cagr_used,
                    'cagr_period': f"{len(navs)} weeks",
                    'source': source,
                    'weekly_values': formatted_weekly_values,
                    'current_nav': current_nav,
                    'initial_nav': initial_nav
                }
            
            # Fallback: Try CAGR-based calculation if NAV fetch fails
            if nav_data is None:
                print(f"[PMS_AIF] ⚠️ WARNING: NAV fetch returned None for {ticker} (OpenAI client may not be available), falling back to CAGR calculation...")
            else:
                print(f"[PMS_AIF] ⚠️ WARNING: AI NAV fetch failed for {ticker} (no NAVs returned), falling back to CAGR calculation...")
            print(f"[PMS_AIF] 🔍 Fetching CAGR for {ticker} ({'AIF' if is_aif else 'PMS'}) - Investment: Rs. {investment_amount:,.2f} on {investment_date}")
            cagr_data = self._get_cagr_from_ai(ticker, is_aif, pms_aif_name, investment_date)
            
            if cagr_data and cagr_data.get('cagr') is not None and cagr_data.get('cagr') > 0:
                # Use AI-provided CAGR
                cagr = cagr_data['cagr']
                cagr_period = cagr_data.get('period', 'AI Estimated')
                source = f"AI ({cagr_period})"
                print(f"[PMS_AIF] Got CAGR {cagr:.2%} ({cagr_period}) for {ticker}")
            else:
                # If AI fails, try SEBI data as fallback
                print(f"[PMS_AIF] WARNING: AI CAGR failed for {ticker}, trying SEBI data...")
                sebi_data = None
                if not is_aif:
                    # Try SEBI PMS data
                    sebi_data = self._fetch_pms_from_sebi(ticker)
                else:
                    # Try SEBI AIF data
                    sebi_data = self._fetch_aif_from_sebi(ticker)
                
                if sebi_data and not sebi_data.empty:
                    best_cagr = self._extract_best_cagr(sebi_data)
                    if best_cagr and best_cagr.get('cagr') and best_cagr.get('cagr') > 0:
                        cagr = best_cagr['cagr']
                        cagr_period = best_cagr.get('period', 'SEBI')
                        source = f"SEBI ({cagr_period})"
                        print(f"[PMS_AIF] Got CAGR {cagr:.2%} from SEBI ({cagr_period}) for {ticker}")
                    else:
                        print(f"[PMS_AIF] ERROR: SEBI data available but no valid CAGR found for {ticker}")
                        return {
                            'ticker': ticker,
                            'initial_investment': investment_amount,
                            'current_value': investment_amount,  # Return original investment if all fails
                            'absolute_gain': 0,
                            'percentage_gain': 0,
                            'years_elapsed': years_elapsed,
                            'cagr_used': 0,
                            'cagr_period': 'Unavailable',
                            'source': 'CAGR fetch failed - using investment value',
                            'weekly_values': [],
                            'error': f'AI and SEBI CAGR calculation failed. AI result: {cagr_data}, SEBI result: {best_cagr}'
                        }
                else:
                    print(f"[PMS_AIF] ERROR: No SEBI data available for {ticker}")
                    return {
                        'ticker': ticker,
                        'initial_investment': investment_amount,
                        'current_value': investment_amount,  # Return original investment if all fails
                        'absolute_gain': 0,
                        'percentage_gain': 0,
                        'years_elapsed': years_elapsed,
                        'cagr_used': 0,
                        'cagr_period': 'AI Unavailable',
                        'source': 'AI fetch failed - using investment value',
                        'weekly_values': [],
                        'error': f'AI CAGR calculation failed. AI result: {cagr_data}'
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
    
    def _get_navs_from_ai(
        self,
        ticker: str,
        is_aif: bool,
        pms_aif_name: str = None,
        investment_date: str = None,
        investment_amount: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get actual NAVs (Net Asset Values) for PMS/AIF using AI - fetches 52 weeks of NAV data
        
        Args:
            ticker: Registration code (e.g., INP000005000)
            is_aif: True for AIF, False for PMS
            pms_aif_name: Optional name for better context
            investment_date: Investment date for context
            investment_amount: Optional investment amount (used for estimated NAVs if AI returns null)
        
        Returns:
            Dict with navs (list of {date, nav}), current_nav, source or None
        """
        client = self._get_openai_client()
        if not client:
            print(f"[PMS_AIF_AI] [ERROR] OpenAI client not available - cannot fetch NAVs for {ticker}. Check API key in secrets.toml")
            return None
        
        try:
            asset_type = "AIF" if is_aif else "PMS"
            name_context = f" named '{pms_aif_name}'" if pms_aif_name else ""
            date_context = f" invested on {investment_date}" if investment_date else ""
            
            # Generate list of dates for last 52 weeks (Mondays)
            from datetime import timedelta
            nav_dates = []
            current_date = datetime.now()
            for i in range(52):
                week_date = current_date - timedelta(weeks=i)
                # Get Monday of that week
                monday = week_date - timedelta(days=week_date.weekday())
                nav_dates.append(monday.strftime('%Y-%m-%d'))
            nav_dates.reverse()  # Oldest first
            
            prompt = f"""You are a financial data expert. I need the ACTUAL NAV (Net Asset Value) data for a {asset_type} (Portfolio Management Service/Alternative Investment Fund) with registration code {ticker}{name_context}{date_context}.

CRITICAL: You MUST search for EXACT, OFFICIAL NAV data first. Only use "estimated" as an absolute last resort if you cannot find ANY official data after extensive searching.

IMPORTANT INSTRUCTIONS:
1. Search THOROUGHLY for the specific {asset_type} product with registration code {ticker} on:
   - SEBI website (sebi.gov.in) - search for registered PMS/AIF products
   - Fund house/manager's official website - look for investor relations or NAV pages
   - Financial databases (Moneycontrol, Value Research, Morningstar India, etc.)
   - Stock exchange websites (NSE/BSE) if the product is listed
   - Company annual reports and investor presentations
2. Fetch ACTUAL, OFFICIAL NAV values from these sources - do NOT estimate unless you absolutely cannot find any data
3. Provide NAV values for the last 52 weeks (one per week, typically Monday dates)
4. If you find official NAV data (even if incomplete), use "exact" or "closest_available" as data_type
5. ONLY use "estimated" if you have searched extensively and found NO official NAV data anywhere
6. If you find partial official data, use "closest_available" and note what percentage is official vs estimated

For Indian PMS/AIF products, NAVs are typically published:
- Weekly or monthly on fund house websites
- On SEBI website for registered products (search by registration code)
- On financial data platforms like Moneycontrol, Value Research, etc.

Please provide:
1. Current NAV (most recent available - prefer official sources)
2. Historical NAVs for the last 52 weeks (one per week)
3. The source of this data (e.g., "SEBI", "Fund House Website", "Financial Database", "Estimated")
4. Whether this is exact data, closest available, or estimated (prefer "exact" or "closest_available" over "estimated")

Return ONLY a JSON object with this exact format:
{{
    "current_nav": 150.50,
    "source": "SEBI/Fund House Website",
    "data_type": "exact",
    "navs": [
        {{"date": "2024-01-01", "nav": 100.00}},
        {{"date": "2024-01-08", "nav": 101.50}},
        {{"date": "2024-01-15", "nav": 102.25}},
        ...
        {{"date": "{nav_dates[-1]}", "nav": 150.50}}
    ]
}}

OR if using closest available (partial official data):
{{
    "current_nav": 150.50,
    "source": "Fund House Website (partial data)",
    "data_type": "closest_available",
    "navs": [
        {{"date": "2024-01-01", "nav": 100.00}},
        ...
    ]
}}

OR if absolutely no official data found (YOU MUST STILL PROVIDE ESTIMATED NAVs):
{{
    "current_nav": 150.50,
    "source": "Estimated from similar products/industry average",
    "data_type": "estimated",
    "navs": [
        {{"date": "2024-01-01", "nav": 100.00}},
        {{"date": "2024-01-08", "nav": 101.50}},
        ...
        {{"date": "{nav_dates[-1]}", "nav": 150.50}}
    ]
}}

CRITICAL RULES:
- NEVER return null for current_nav
- NEVER return an empty navs array
- If you cannot find official NAVs, you MUST estimate NAVs based on typical {asset_type} performance
- Use a reasonable CAGR assumption (12-18% for equity PMS, 10-15% for balanced PMS)
- Calculate estimated NAVs for ALL 52 weeks using the CAGR assumption
- Start with an initial NAV (e.g., 100) and grow it week by week using the CAGR

CRITICAL: 
- Provide at least 12 weeks of NAV data (preferably 52 weeks)
- Ensure NAVs are in chronological order (oldest first)
- Use actual dates from the list: {', '.join(nav_dates[:5])} ... {', '.join(nav_dates[-3:])}
- NAV values should be positive numbers
- PREFER "exact" or "closest_available" over "estimated"
- Only use "estimated" if you have searched extensively and found NO official NAV data
- If you find any official NAV data (even partial), use "closest_available" instead of "estimated"
- NEVER return null for current_nav or empty navs array - ALWAYS provide estimated NAVs based on typical PMS/AIF performance if official data is unavailable
- If you cannot find official NAVs, estimate based on:
  * Typical {asset_type} performance (usually 12-20% CAGR for equity-focused PMS)
  * Similar products from the same fund house (if name is provided)
  * Industry averages for the strategy type
  * Calculate estimated NAVs using a reasonable CAGR assumption (e.g., 15% for equity PMS)
"""

            # Use gpt-5 as primary, with gpt-4o as fallback
            models_to_try = ["gpt-5", "gpt-4o"]
            model_to_use = None
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    print(f"[PMS_AIF_AI] Fetching NAVs using {model} for {ticker}...")
                    # gpt-5 only supports default temperature (1), other models can use lower temperature
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a financial data expert. Provide accurate NAV data for Indian PMS/AIF products. Return only valid JSON with actual NAV values."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                    # Only add temperature for non-gpt-5 models
                    if model != "gpt-5":
                        request_params["temperature"] = 0.2  # Lower temperature for more accurate data
                    response = client.chat.completions.create(**request_params)
                    model_to_use = model
                    print(f"[PMS_AIF_AI] [OK] {model} succeeded for {ticker}")
                    break
                except Exception as model_error:
                    error_str = str(model_error).lower()
                    last_error = model_error
                    print(f"[PMS_AIF_AI] [WARNING] {model} failed for {ticker}: {str(model_error)[:150]}")
                    continue
            
            if not response:
                print(f"[PMS_AIF_AI] [ERROR] All models failed for {ticker}. Last error: {str(last_error)[:200]}")
                return None
            
            content = response.choices[0].message.content
            print(f"[PMS_AIF_AI] Got NAV response from {model_to_use} for {ticker}: {content[:300]}")
            
            # Try to parse JSON
            try:
                result = json.loads(content)
                print(f"[PMS_AIF_AI] Parsed JSON result: {len(result.get('navs', []))} NAVs found")
            except json.JSONDecodeError as e:
                print(f"[PMS_AIF_AI] [ERROR] JSON parse error for {ticker}: {str(e)}")
                print(f"[PMS_AIF_AI]   Raw content: {content[:500]}")
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        print(f"[PMS_AIF_AI] Extracted JSON from markdown: {len(result.get('navs', []))} NAVs")
                    except:
                        print(f"[PMS_AIF_AI] [ERROR] Failed to parse extracted JSON")
                        return None
                else:
                    print(f"[PMS_AIF_AI] [ERROR] No JSON found in response")
                    return None
            
            navs = result.get('navs', [])
            current_nav = result.get('current_nav')
            
            # If AI returned null or empty, generate estimated NAVs as fallback
            if not navs or len(navs) == 0 or current_nav is None:
                print(f"[PMS_AIF_AI] [WARNING] {ticker}: AI returned null/empty NAVs, generating estimated NAVs based on typical {asset_type} performance")
                # Generate estimated NAVs using a reasonable CAGR assumption
                # For equity-focused PMS, use 15% CAGR; for balanced, use 12%
                estimated_cagr = 0.15 if 'equity' in (pms_aif_name or '').lower() else 0.12
                
                # Calculate estimated NAVs for the last 52 weeks
                # Use investment amount as base if available, otherwise use 100
                if investment_amount and investment_amount > 0:
                    # Estimate initial NAV based on investment (assume 1 unit = initial investment)
                    initial_nav = investment_amount
                else:
                    initial_nav = 100.0
                
                navs = []
                for i, date_str in enumerate(nav_dates):
                    weeks_ago = len(nav_dates) - 1 - i
                    years_ago = weeks_ago / 52.0
                    # Calculate NAV backwards from current (most recent) to oldest
                    estimated_nav = initial_nav * ((1 + estimated_cagr) ** years_ago)
                    navs.append({
                        'date': date_str,
                        'nav': round(estimated_nav, 2)
                    })
                current_nav = navs[-1].get('nav', initial_nav) if navs else initial_nav
                print(f"[PMS_AIF_AI] [INFO] {ticker}: Generated {len(navs)} estimated NAVs using {estimated_cagr*100:.1f}% CAGR assumption (initial NAV: {initial_nav:,.2f})")
            
            if navs and len(navs) > 0:
                # Validate NAVs
                valid_navs = []
                for nav_entry in navs:
                    if isinstance(nav_entry, dict) and 'nav' in nav_entry and 'date' in nav_entry:
                        nav_value = nav_entry.get('nav')
                        try:
                            nav_float = float(nav_value)
                            if nav_float > 0:
                                valid_navs.append({
                                    'date': nav_entry.get('date'),
                                    'nav': nav_float
                                })
                        except (ValueError, TypeError):
                            continue
                
                if valid_navs:
                    current_nav = result.get('current_nav', valid_navs[-1].get('nav', 0))
                    source = result.get('source', 'AI')
                    data_type = result.get('data_type', 'estimated')
                    
                    print(f"[PMS_AIF_AI] Validated {len(valid_navs)} NAVs for {ticker}, current NAV: Rs. {current_nav:,.2f}")
                    
                    return {
                        'navs': valid_navs,
                        'current_nav': float(current_nav) if current_nav else valid_navs[-1].get('nav', 0),
                        'source': f"{source} ({data_type})",
                        'data_type': data_type
                    }
                else:
                    print(f"[PMS_AIF_AI] [ERROR] No valid NAVs found in response for {ticker}")
                    return None
            else:
                print(f"[PMS_AIF_AI] [ERROR] No NAVs in response for {ticker}")
                return None
                
        except Exception as e:
            print(f"[PMS_AIF_AI] [ERROR] Error fetching NAVs for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
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
            print(f"[PMS_AIF_AI] [ERROR] OpenAI client not available - cannot fetch CAGR for {ticker}. Check API key in secrets.toml")
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

            # Use gpt-5 as primary, with gpt-4o as fallback
            models_to_try = ["gpt-5", "gpt-4o"]
            model_to_use = None
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    print(f"[PMS_AIF_AI] Trying model: {model} for {ticker}...")
                    # gpt-5 only supports default temperature (1), other models can use lower temperature
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a financial data expert. Provide accurate CAGR data for Indian PMS/AIF products. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                    # Only add temperature for non-gpt-5 models
                    if model != "gpt-5":
                        request_params["temperature"] = 0.3
                    response = client.chat.completions.create(**request_params)
                    model_to_use = model
                    print(f"[PMS_AIF_AI] [OK] {model} succeeded for {ticker}")
                    break
                except Exception as model_error:
                    error_str = str(model_error).lower()
                    last_error = model_error
                    print(f"[PMS_AIF_AI] [WARNING] {model} failed for {ticker}: {str(model_error)[:150]}")
                    # Continue to next model
                    continue
            
            if not response:
                print(f"[PMS_AIF_AI] [ERROR] All models failed for {ticker}. Last error: {str(last_error)[:200]}")
                return None
            
            content = response.choices[0].message.content
            print(f"[PMS_AIF_AI] Got response from {model_to_use} for {ticker}: {content[:200]}")
            
            # Try to parse JSON
            try:
                result = json.loads(content)
                print(f"[PMS_AIF_AI] Parsed JSON result: {result}")
            except json.JSONDecodeError as e:
                print(f"[PMS_AIF_AI] [ERROR] JSON parse error for {ticker}: {str(e)}")
                print(f"[PMS_AIF_AI]   Raw content: {content}")
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        print(f"[PMS_AIF_AI] Extracted JSON from markdown: {result}")
                    except:
                        print(f"[PMS_AIF_AI] [ERROR] Failed to parse extracted JSON")
                        return None
                else:
                    # Try to find JSON object in content
                    json_match = re.search(r'\{[^{}]*"cagr"[^{}]*\}', content)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                            print(f"[PMS_AIF_AI] Extracted JSON object: {result}")
                        except:
                            print(f"[PMS_AIF_AI] ERROR: Failed to parse extracted JSON object")
                            return None
                    else:
                        print(f"[PMS_AIF_AI] ERROR: No JSON found in response")
                        return None
            
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
            print(f"[PMS_AIF_AI] WARNING: AI CAGR fetch failed for {ticker}: {str(e)}")
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

