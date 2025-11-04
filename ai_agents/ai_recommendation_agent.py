"""
AI-Powered Investment Recommendation Agent
Suggests new holdings (stocks, MFs, PMS, bonds, etc.) based on portfolio analysis
"""

import openai
import json
from typing import List, Dict, Any
from datetime import datetime
from .base_agent import BaseAgent


class AIInvestmentRecommendationAgent(BaseAgent):
    """
    AI agent that recommends new investment opportunities
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ai_investment_recommendation_agent",
            agent_name="AI Investment Recommendation Agent"
        )
        
        self.capabilities = [
            "stock_recommendations",
            "mutual_fund_recommendations",
            "pms_recommendations",
            "bond_recommendations",
            "aif_recommendations",
            "diversification_opportunities",
            "emerging_sectors",
            "sell_recommendations",
            "rebalancing_suggestions"
        ]
        
        # Initialize OpenAI client
        try:
            import streamlit as st
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        self.recommendation_cache = []
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment recommendations"""
        
        self.update_status("analyzing")
        
        try:
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to generate recommendations
            recommendations = self._ai_generate_recommendations(data)
            
            # Cache recommendations
            self.recommendation_cache = recommendations
            
            self.update_status("active")
            return self.format_response(recommendations, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI recommendation generation: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_generate_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently generate investment recommendations"""
        
        portfolio_data = data.get("portfolio_data", {})
        user_profile = data.get("user_profile", {})
        
        # Debug logging
        self.logger.info(f"Portfolio data keys: {list(portfolio_data.keys()) if portfolio_data else 'None'}")
        self.logger.info(f"Holdings count: {len(portfolio_data.get('holdings', [])) if portfolio_data else 0}")
        
        if not portfolio_data or not portfolio_data.get('holdings'):
            self.logger.warning("No portfolio data or holdings found")
            return [{
                "type": "investment_recommendation",
                "severity": "low",
                "title": "No Portfolio Data",
                "description": "No portfolio data found for generating recommendations",
                "recommendation": "Upload transaction files to enable personalized recommendations",
                "data": {"portfolio_data_available": False}
            }]
        
        # Prepare comprehensive context
        recommendation_context = self._prepare_recommendation_context(portfolio_data, user_profile)
        
        # Get PDF context for market insights
        pdf_context = data.get("pdf_context", "")
        
        # Create AI prompt with PDF context
        prompt = self._create_recommendation_prompt(recommendation_context, user_profile, pdf_context)
        
        # Safety check - ensure prompt is not None
        if prompt is None:
            self.logger.error("_create_recommendation_prompt returned None")
            prompt = "Analyze the following portfolio and recommend investment opportunities:\n"
        
        # Add detailed portfolio holdings
        prompt += f"\n\n📊 CURRENT PORTFOLIO HOLDINGS:\n"
        holdings = portfolio_data.get('holdings', [])
        if holdings:
            prompt += f"Total Holdings: {len(holdings)}\n\n"
            
            # Group by asset type
            asset_groups = {}
            for holding in holdings:
                asset_type = holding.get('asset_type', 'Unknown')
                if asset_type not in asset_groups:
                    asset_groups[asset_type] = []
                asset_groups[asset_type].append(holding)
            
            for asset_type, holdings_list in asset_groups.items():
                prompt += f"\n{asset_type.upper()} Holdings:\n"
                sorted_holdings = sorted(holdings_list, 
                                       key=lambda x: x.get('current_value', 0) or 0, 
                                       reverse=True)[:5]
                for holding in sorted_holdings:
                    try:
                        ticker = str(holding.get('ticker', 'N/A')) if holding.get('ticker') else 'N/A'
                        name = str(holding.get('stock_name', 'N/A')) if holding.get('stock_name') else 'N/A'
                        value = float(holding.get('current_value', 0)) if holding.get('current_value') is not None else 0.0
                        pnl_pct = float(holding.get('pnl_percentage', 0)) if holding.get('pnl_percentage') is not None else 0.0
                        sector = str(holding.get('sector', 'Unknown')) if holding.get('sector') else 'Unknown'
                        prompt += f"  • {ticker} ({name}) - ₹{value:,.0f} ({pnl_pct:+.1f}%) - {sector}\n"
                    except Exception as e:
                        self.logger.error(f"Error processing holding data: {e}")
                        prompt += f"  • Error processing holding data\n"
        
        try:
            # Get current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better investment recommendations
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert investment advisor and portfolio strategist specializing in the Indian stock market. Your task is to recommend NEW investment opportunities that complement the existing portfolio.

📅 CURRENT DATE: {current_date} (Year: {current_year})
⚠️ CRITICAL: Today's date is {current_date}. Always use this date when:
- Making investment recommendations based on current market conditions
- Referencing recent market performance and trends
- Calculating time horizons for recommendations
- Analyzing current valuations and entry points
Do NOT use 2024 or any other year - use {current_year}.

IMPORTANT RULES:
1. Analyze the current portfolio to identify gaps and opportunities
2. Recommend SPECIFIC stocks, mutual funds, PMS, bonds, or AIFs with actual names and ticker symbols
3. Provide CONCRETE reasons why each recommendation fits the portfolio
4. Consider diversification, sector exposure, and risk profile
5. Recommend only Indian market instruments (NSE/BSE stocks, Indian MFs, etc.)
6. Ensure recommendations are actionable and based on current market conditions
7. Return recommendations in the exact JSON format requested

For each recommendation, provide:

BUY RECOMMENDATIONS:
- type: "stock_recommendation", "mutual_fund_recommendation", "pms_recommendation", "bond_recommendation", or "diversification_opportunity"
- severity: high/medium/low based on importance and opportunity
- title: Clear name of the recommended instrument
- description: Why this is a good investment NOW
- recommendation: Specific action steps (ticker, amount, allocation)
- data: ticker, asset_type, sector, suggested_allocation, expected_return, risk_level, investment_thesis, why_now

SELL RECOMMENDATIONS:
- type: "sell_recommendation"
- severity: high/medium/low based on urgency
- title: "SELL: [Stock Name]" or "EXIT: [Stock Name]"
- description: Why this holding should be reduced or exited (concentration risk, underperformance, sector headwinds)
- recommendation: Specific sell quantity and redeployment strategy
- data: 
  * action: "SELL"
  * ticker: The stock/MF to sell
  * current_holding_quantity: Exact shares/units currently held
  * suggested_sell_quantity: Exact shares/units to sell
  * percentage_to_sell: % of holding to sell (e.g., 50% for partial exit, 100% for complete exit)
  * current_value: Current value of the holding
  * value_after_sale: Value remaining after sale
  * funds_freed: Amount available for redeployment
  * reason: Why sell this specific holding
  * rebalancing_strategy: Where to redeploy the freed capital
  * tax_consideration: Tax implications (STCG/LTCG)
  * why_now: Why sell at this time

Focus on:
BUY: Filling portfolio gaps, diversification, emerging sectors, quality instruments
SELL: Excessive concentration (>15% in single stock), sector over-allocation (>30%), underperformers (<-20% loss), fundamentally weak stocks

CRITICAL: 
- Use ACTUAL holdings from the portfolio for sell recommendations
- Provide EXACT quantities currently held
- Calculate EXACT sell quantities based on rebalancing needs
- Recommend REAL instruments with proper ticker symbols for buys"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                # Note: GPT-5 only supports default temperature (1)
                # No max_completion_tokens - let OpenAI use default to allow reasoning + response (like AI assistant)
                # No timeout - let OpenAI use default (like AI assistant)
                # Removed response_format - GPT-5 may not support it, relying on prompt instructions instead
            )
            
            # Parse AI response with error handling
            if not response or not response.choices or len(response.choices) == 0:
                self.logger.error("Empty response from OpenAI API")
                return []
            
            ai_response = response.choices[0].message.content
            
            if not ai_response or not ai_response.strip():
                self.logger.error("AI response content is empty")
                return []
            
            # Log AI response length and first 500 chars for debugging
            self.logger.info(f"AI response received: {len(ai_response)} characters")
            self.logger.info(f"AI response preview: {ai_response[:500]}...")
            
            # Debug: Log the raw response to help with JSON parsing issues
            if len(ai_response) > 2000:
                self.logger.info(f"AI response is long ({len(ai_response)} chars), checking for JSON structure...")
                if '[' in ai_response and ']' in ai_response:
                    start_idx = ai_response.find('[')
                    end_idx = ai_response.rfind(']') + 1
                    json_part = ai_response[start_idx:end_idx]
                    self.logger.info(f"JSON part length: {len(json_part)} chars")
                    self.logger.info(f"JSON part preview: {json_part[:200]}...")
            
            # Extract recommendations from response
            recommendations = self._parse_ai_recommendation_response(ai_response)
            
            self.logger.info(f"Parsed {len(recommendations)} recommendations from AI response")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"AI recommendation generation failed: {e}")
            return [{
                "type": "investment_recommendation",
                "severity": "medium",
                "title": "Recommendation Generation Error",
                "description": f"AI recommendation generation failed: {str(e)}",
                "recommendation": "Please try again or contact support",
                "data": {"error": str(e)}
            }]
    
    def _prepare_recommendation_context(self, portfolio_data: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for recommendations"""
        
        holdings = portfolio_data.get("holdings", [])
        
        if not holdings:
            return {
                "holdings_count": 0,
                "portfolio_data_available": False
            }
        
        import pandas as pd
        df = pd.DataFrame(holdings)
        
        # Portfolio metrics
        total_value = sum(holding.get('current_value', 0) or 0 for holding in holdings)
        total_investment = sum(holding.get('investment', 0) or 0 for holding in holdings)
        total_pnl_pct = ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        # Asset allocation
        asset_allocation = {}
        sector_allocation = {}
        channel_allocation = {}
        
        for holding in holdings:
            asset_type = holding.get('asset_type', 'Unknown')
            sector = holding.get('sector', 'Unknown')
            channel = holding.get('channel', 'Unknown')
            value = holding.get('current_value', 0) or 0
            
            asset_allocation[asset_type] = asset_allocation.get(asset_type, 0) + value
            sector_allocation[sector] = sector_allocation.get(sector, 0) + value
            channel_allocation[channel] = channel_allocation.get(channel, 0) + value
        
        # Calculate percentages
        asset_allocation_pct = {k: (v/total_value*100) for k, v in asset_allocation.items()} if total_value > 0 else {}
        sector_allocation_pct = {k: (v/total_value*100) for k, v in sector_allocation.items()} if total_value > 0 else {}
        
        # Identify gaps
        missing_asset_types = []
        if asset_allocation_pct.get('mutual_fund', 0) < 10:
            missing_asset_types.append('mutual_fund')
        if asset_allocation_pct.get('bond', 0) < 5:
            missing_asset_types.append('bond')
        if asset_allocation_pct.get('pms', 0) == 0 and total_value > 5000000:  # 50L+
            missing_asset_types.append('pms')
        
        # Identify underrepresented sectors
        underrepresented_sectors = [sector for sector, pct in sector_allocation_pct.items() 
                                    if pct < 5 and sector != 'Unknown']
        
        return {
            "holdings_count": len(holdings),
            "total_portfolio_value": total_value,
            "total_investment": total_investment,
            "total_pnl_percentage": total_pnl_pct,
            "asset_allocation": asset_allocation,
            "asset_allocation_percentage": asset_allocation_pct,
            "sector_allocation": sector_allocation,
            "sector_allocation_percentage": sector_allocation_pct,
            "channel_allocation": channel_allocation,
            "missing_asset_types": missing_asset_types,
            "underrepresented_sectors": underrepresented_sectors,
            "user_risk_tolerance": user_profile.get('risk_tolerance', 'moderate'),
            "user_goals": user_profile.get('goals', []),
            "portfolio_data_available": True
        }
    
    def _create_recommendation_prompt(self, context: Dict[str, Any], user_profile: Dict[str, Any], pdf_context: str = "") -> str:
        """Create AI prompt for investment recommendations with PDF insights"""
        
        # Safe JSON serialization
        try:
            context_json = json.dumps(context, indent=2, default=str) if context else "{}"
        except Exception as e:
            self.logger.error(f"Error serializing context: {e}")
            context_json = "{}"
        
        try:
            user_profile_json = json.dumps(user_profile, indent=2, default=str) if user_profile else "{}"
        except Exception as e:
            self.logger.error(f"Error serializing user_profile: {e}")
            user_profile_json = "{}"
        
        prompt = f"""Analyze the following portfolio and recommend NEW investment opportunities:

PORTFOLIO CONTEXT:
{context_json}

USER PROFILE:
{user_profile_json}"""

        # Add PDF context if available
        if pdf_context and pdf_context.strip():
            try:
                # Ensure pdf_context is a string and not None
                pdf_text = str(pdf_context)[:3000] if pdf_context else ""
                if pdf_text.strip():
                    prompt += f"""

📚 RESEARCH DOCUMENTS & MARKET INSIGHTS:
{pdf_text}

Use these research insights to enhance your recommendations with:
- Market trends and sector outlooks mentioned in the documents
- Specific stocks or funds recommended in the research
- Economic forecasts and their impact on asset allocation
- Risk factors highlighted in the research"""
            except Exception as e:
                self.logger.error(f"Error processing PDF context: {e}")
                # Continue without PDF context

        prompt += """

RECOMMENDATION REQUIREMENTS:
1. BUY RECOMMENDATIONS: Recommend NEW instruments to fill portfolio gaps (missing sectors, asset types)
2. SELL RECOMMENDATIONS: Identify holdings to reduce or exit (underperformers, over-concentrated positions)
3. GROWTH OPPORTUNITIES: Identify high-potential stocks or sectors based on current market trends
4. STABILITY: Suggest defensive assets or bonds for portfolio stability
5. TAX EFFICIENCY: Recommend tax-efficient instruments where applicable
6. RISK-ADJUSTED RETURNS: Balance risk and return based on user risk tolerance
7. REBALANCING: Suggest specific quantities to buy/sell for optimal portfolio balance

Return your analysis as a JSON object with a "recommendations" array. Use this EXACT format:

{{
  "recommendations": [
    {{
      "type": "stock_recommendation",
      "severity": "high",
      "title": "HDFC Bank Ltd (HDFCBANK.NS)",
      "description": "HDFC Bank is India's largest private sector bank with strong fundamentals. Current market conditions favor banking stocks due to improving credit growth and stable NIM. Your portfolio lacks banking sector exposure (only 5%), making this an ideal diversification opportunity. The stock has shown resilience with consistent RoE above 17% and is trading at attractive valuations.",
      "recommendation": "Invest ₹2,00,000 (10% of portfolio) in HDFCBANK.NS. Buy gradually over 2-3 months to average out entry price. Target allocation: 10-15% of portfolio in banking sector.",
      "data": {{
        "ticker": "HDFCBANK.NS",
        "asset_type": "stock",
        "sector": "Banking & Financial Services",
        "suggested_allocation_percentage": 10,
        "suggested_amount": 200000,
        "expected_return": "12-15% annually",
        "risk_level": "medium",
        "investment_thesis": "Strong fundamentals, improving credit growth, attractive valuations, fills banking sector gap",
        "current_price_approx": "1600",
        "why_now": "Banking sector showing strong credit growth, NIM stabilization, and favorable regulatory environment"
      }}
    }},
    {{
      "type": "mutual_fund_recommendation",
      "severity": "medium",
    "title": "Parag Parikh Flexi Cap Fund - Direct Plan",
    "description": "This fund offers international diversification with 35% allocation to US stocks while maintaining strong domestic equity exposure. Your portfolio currently has 0% international exposure, and your risk profile allows for 20% international allocation. The fund has consistently outperformed its benchmark with 18% 3-year CAGR.",
    "recommendation": "Invest ₹1,50,000 via SIP (₹25,000/month for 6 months) in Parag Parikh Flexi Cap Fund. This provides instant international diversification and professional management.",
    "data": {{
      "ticker": "122639",
      "asset_type": "mutual_fund",
      "sector": "Multi Cap / International",
      "suggested_allocation_percentage": 7,
      "suggested_amount": 150000,
      "expected_return": "15-18% annually",
      "risk_level": "medium-high",
      "investment_thesis": "International diversification, proven track record, fills global exposure gap",
      "current_nav_approx": "72",
      "why_now": "US market correction offers good entry point, rupee depreciation benefits international allocation"
    }}
  }},
  {{
    "type": "sell_recommendation",
    "severity": "high",
    "title": "SELL: Reduce INFY.NS (Infosys Ltd)",
    "description": "Infosys currently represents 25% of your portfolio, creating excessive concentration risk in a single stock. The Technology sector alone accounts for 45% of your holdings, far exceeding the recommended 20-25% sector allocation. While Infosys is a quality stock, this concentration exposes you to significant sector-specific risks. Recent IT sector headwinds and client budget cuts suggest taking partial profits.",
    "recommendation": "SELL 50% of your INFY.NS holding (reduce from 41,710 shares to 20,855 shares). This will free up approximately ₹30.7L which can be redeployed into Banking, Healthcare, and defensive sectors for better diversification.",
    "data": {{
      "action": "SELL",
      "ticker": "INFY.NS",
      "current_holding_quantity": 41710,
      "suggested_sell_quantity": 20855,
      "percentage_to_sell": 50,
      "current_value": 61497225,
      "value_after_sale": 30748612,
      "funds_freed": 30748613,
      "reason": "Excessive concentration risk (25% of portfolio in single stock)",
      "sector_concentration": "Technology: 45% (Target: 20-25%)",
      "rebalancing_strategy": "Redeploy proceeds into Banking (10%), Healthcare (10%), and Bonds (10%)",
      "tax_consideration": "Long-term capital gains apply - LTCG tax on ₹16.15L profit",
      "why_now": "IT sector facing headwinds, profit booking at good levels, reduce concentration risk"
    }}
  }},
  {{
    "type": "sell_recommendation",
    "severity": "medium",
    "title": "EXIT: BAJFINANCE.NS (Bajaj Finance Ltd)",
    "description": "Bajaj Finance is showing a significant loss of -68.9% with current value at ₹12.7L against investment of ₹40.9L. The NBFC sector has faced regulatory challenges and rising NPAs. Technical analysis shows weakness with price at ₹1,059 vs your average of ₹3,409. Cut losses and redeploy capital into better opportunities.",
    "recommendation": "SELL COMPLETE HOLDING of BAJFINANCE.NS (12,010 shares). Exit this position entirely to prevent further losses. Redeploy the ₹12.7L into quality banking stocks or debt funds for stability.",
    "data": {{
      "action": "SELL",
      "ticker": "BAJFINANCE.NS",
      "current_holding_quantity": 12010,
      "suggested_sell_quantity": 12010,
      "percentage_to_sell": 100,
      "current_value": 12729399,
      "value_after_sale": 0,
      "funds_freed": 12729399,
      "current_loss": -28219056,
      "loss_percentage": -68.9,
      "reason": "Stop-loss trigger: Loss exceeds 60%, poor fundamentals, sector headwinds",
      "rebalancing_strategy": "Redeploy into HDFCBANK.NS (₹6L), Bonds (₹4L), Debt MF (₹2.7L)",
      "tax_consideration": "Loss can be set off against capital gains - tax benefit on ₹28.2L loss",
      "why_now": "Cut losses early, NBFC sector facing regulatory pressure, better opportunities available"
    }}
  }}
  ]
}}

CRITICAL: Return ONLY valid JSON. No markdown, no explanations, no text outside the JSON object.

IMPORTANT: 
- Recommend REAL instruments with actual ticker symbols
- Provide CURRENT market context and specific reasons
- Match recommendations to user risk tolerance and goals
- Ensure recommendations are actionable TODAY
- Include at least one recommendation for each missing asset type
- Focus on HIGH-QUALITY instruments with proven track records"""
        
        return prompt
        
    def _parse_ai_recommendation_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract recommendations"""
        
        try:
            # Clean the response
            ai_response = ai_response.strip()
            
            # Remove any markdown formatting
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:].strip()
            elif ai_response.startswith('```'):
                ai_response = ai_response[3:].strip()
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3].strip()
            
            # First try parsing as JSON object with "insights" or "recommendations" key
            try:
                parsed = json.loads(ai_response)
                if isinstance(parsed, dict):
                    # Try "insights" key first
                    if "insights" in parsed and isinstance(parsed["insights"], list):
                        return self._validate_recommendations(parsed["insights"])
                    # Try "recommendations" key
                    elif "recommendations" in parsed and isinstance(parsed["recommendations"], list):
                        return self._validate_recommendations(parsed["recommendations"])
                    # If it's a list directly (fallback for old format)
                    elif isinstance(parsed, list):
                        return self._validate_recommendations(parsed)
            except json.JSONDecodeError:
                pass
            
            # Try direct JSON parsing as array (old format)
            try:
                recommendations = json.loads(ai_response)
                if isinstance(recommendations, list):
                    return self._validate_recommendations(recommendations)
            except json.JSONDecodeError:
                pass
            
            # Try to find JSON array in the response
            start_idx = ai_response.find('[')
            end_idx = ai_response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                try:
                    recommendations = json.loads(json_str)
                    if isinstance(recommendations, list):
                        return self._validate_recommendations(recommendations)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed: {e}")
                    # Try to clean the JSON string
                    cleaned_json = self._clean_json_string(json_str)
                    if cleaned_json:
                        try:
                            recommendations = json.loads(cleaned_json)
                            if isinstance(recommendations, list):
                                return self._validate_recommendations(recommendations)
                        except json.JSONDecodeError:
                            pass
            
            # If all JSON parsing failed, try to extract recommendations using regex
            return self._extract_recommendations_with_regex(ai_response)
            
        except Exception as e:
            self.logger.error(f"Error parsing AI recommendation response: {e}")
            return [{
                "type": "investment_recommendation",
                "severity": "low",
                "title": "Recommendation Parsing Error",
                "description": f"Error parsing recommendations: {str(e)}",
                "recommendation": "Please try again",
                "data": {"error": str(e)}
            }]
    
    def _validate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean recommendation structure"""
        validated_recommendations = []
        for rec in recommendations:
            if isinstance(rec, dict) and 'title' in rec and 'description' in rec:
                # Ensure all required fields
                validated_rec = {
                    "type": rec.get("type", "investment_recommendation"),
                    "severity": rec.get("severity", "medium"),
                    "title": rec.get("title", "Investment Opportunity"),
                    "description": rec.get("description", ""),
                    "recommendation": rec.get("recommendation", ""),
                    "data": rec.get("data", {})
                }
                validated_recommendations.append(validated_rec)
        
        return validated_recommendations if validated_recommendations else []
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to fix common issues"""
        import re
        
        # Remove extra text before/after JSON
        json_str = json_str.strip()
        
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix missing commas between objects
        
        return json_str
    
    def _extract_recommendations_with_regex(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract recommendations using regex as fallback"""
        import re
        
        # Simple regex to extract recommendation objects
        pattern = r'\{[^{}]*"title"[^{}]*\}'
        matches = re.findall(pattern, ai_response, re.DOTALL)
        
        recommendations = []
        for match in matches:
            try:
                # Try to parse each match as JSON
                rec = json.loads(match)
                if isinstance(rec, dict) and 'title' in rec:
                    recommendations.append({
                        "type": rec.get("type", "investment_recommendation"),
                        "severity": rec.get("severity", "medium"),
                        "title": rec.get("title", "Investment Opportunity"),
                        "description": rec.get("description", ""),
                        "recommendation": rec.get("recommendation", ""),
                        "data": rec.get("data", {})
                    })
            except json.JSONDecodeError:
                continue
        
        return recommendations if recommendations else [{
            "type": "investment_recommendation",
            "severity": "low",
            "title": "Recommendation Parsing Error",
            "description": "Could not parse AI response. The AI may have returned malformed JSON.",
            "recommendation": "Please try again or contact support",
            "data": {"error": "JSON parsing failed"}
        }]
    
    def get_cached_recommendations(self) -> List[Dict[str, Any]]:
        """Get cached recommendations"""
        return self.recommendation_cache
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent (required by BaseAgent)"""
        return self.recommendation_cache

