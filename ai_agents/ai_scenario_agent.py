"""
AI-Powered Scenario Analysis Agent
Uses OpenAI to intelligently analyze various market scenarios and their impact
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, MessagePriority

class AIScenarioAnalysisAgent(BaseAgent):
    """
    AI-powered agent for intelligent scenario analysis and stress testing
    """
    
    def __init__(self, agent_id: str = "ai_scenario_agent"):
        super().__init__(agent_id, "AI Scenario Analysis Agent")
        self.capabilities = [
            "intelligent_scenario_analysis",
            "stress_testing",
            "what_if_analysis",
            "monte_carlo_simulation",
            "risk_scenario_modeling",
            "goal_achievement_analysis"
        ]
        
        # Initialize OpenAI client
        try:
            import streamlit as st
            self.openai_client = openai.OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        self.analysis_cache = []
        self.last_analysis_data = {}
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method using AI for intelligent scenario analysis
        """
        try:
            self.update_status("analyzing")
            self.last_analysis_data = data
            
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to analyze scenarios
            insights = self._ai_analyze_scenarios(data)
            
            # Cache insights
            self.analysis_cache = insights
            
            self.update_status("active")
            return self.format_response(insights, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI scenario analysis: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_analyze_scenarios(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently analyze various scenarios"""
        
        portfolio_data = data.get("portfolio_data", {})
        user_profile = data.get("user_profile", {})
        
        if not portfolio_data:
            return [{
                "type": "scenario_analysis",
                "severity": "low",
                "title": "No Portfolio Data",
                "description": "No portfolio data found for scenario analysis",
                "recommendation": "Upload transaction files to enable scenario analysis",
                "data": {"portfolio_data_available": False}
            }]
        
        # Prepare scenario data for AI analysis
        scenario_summary = self._prepare_scenario_summary(portfolio_data, user_profile)
        
        # Create AI prompt for scenario analysis
        prompt = self._create_scenario_analysis_prompt(scenario_summary, user_profile)
        
        # Add comprehensive portfolio context
        prompt += f"\n\n📊 DETAILED PORTFOLIO DATA:\n"
        prompt += f"Total Holdings: {len(portfolio_data.get('holdings', []))}\n"
        
        if portfolio_data.get('holdings'):
            prompt += f"\nTop Holdings by Value:\n"
            sorted_holdings = sorted(portfolio_data['holdings'], 
                                   key=lambda x: x.get('current_value', 0) or 0, reverse=True)[:10]
            for i, holding in enumerate(sorted_holdings, 1):
                ticker = holding.get('ticker', 'N/A')
                name = holding.get('stock_name', 'N/A')
                value = holding.get('current_value', 0) or 0
                pnl_pct = holding.get('pnl_percentage', 0) or 0
                asset_type = holding.get('asset_type', 'Unknown')
                sector = holding.get('sector', 'Unknown')
                prompt += f"{i}. {ticker} ({name}) - ₹{value:,.0f} ({pnl_pct:+.1f}%) - {asset_type} - {sector}\n"
        
        try:
            # Get current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better scenario analysis
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert financial risk analyst and scenario modeling specialist. Your task is to analyze investment portfolios under various market scenarios and provide intelligent insights about potential outcomes.

📅 CURRENT DATE: {current_date} (Year: {current_year})
⚠️ CRITICAL: Today's date is {current_date}. Always use this date when:
- Modeling future scenarios and time horizons
- Referencing current market conditions
- Calculating scenario outcomes over time
- Analyzing risk projections
Do NOT use 2024 or any other year - use {current_year}.

IMPORTANT RULES:
1. Analyze portfolio data intelligently - understand the structure and risk profile
2. Model various market scenarios and their potential impact
3. Provide specific, actionable scenario insights
4. Consider different market conditions and economic factors
5. Be thorough but concise in your analysis
6. Return insights in the exact JSON format requested

For each insight, provide:
- type: Category of scenario (stress_test, what_if, goal_analysis, etc.)
- severity: high/medium/low based on importance and urgency
- title: Clear, descriptive title
- description: Detailed explanation of the scenario and its impact
- recommendation: Specific actionable advice for the scenario
- data: Relevant metrics and supporting data

Focus on:
- Stress testing under adverse market conditions
- What-if analysis for different scenarios
- Goal achievement probability analysis
- Risk scenario modeling
- Monte Carlo simulation insights
- Market crash scenarios
- Economic downturn impacts"""
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
            
            # Extract insights from response
            insights = self._parse_ai_scenario_response(ai_response)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"AI scenario analysis failed: {e}")
            return [{
                "type": "scenario_analysis",
                "severity": "medium",
                "title": "Scenario Analysis Error",
                "description": f"AI scenario analysis failed: {str(e)}",
                "recommendation": "Please try again or contact support",
                "data": {"error": str(e)}
            }]
    
    def _prepare_scenario_summary(self, portfolio_data: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare scenario data for AI analysis"""
        
        holdings = portfolio_data.get("holdings", [])
        
        if not holdings:
            return {
                "holdings_count": 0,
                "user_profile": user_profile,
                "portfolio_data_available": False
            }
        
        # Convert to DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame(holdings)
        
        # Intelligent column detection
        value_columns = [col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower()]
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        sector_columns = [col for col in df.columns if 'sector' in col.lower()]
        asset_type_columns = [col for col in df.columns if 'asset' in col.lower() or 'type' in col.lower()]
        pnl_columns = [col for col in df.columns if 'pnl' in col.lower() or 'profit' in col.lower() or 'loss' in col.lower()]
        
        # Calculate scenario metrics
        scenario_summary = {
            "holdings_count": len(holdings),
            "total_holdings": len(holdings),
            "user_profile": user_profile,
            "portfolio_data_available": True,
            "available_columns": list(df.columns),
            "data_structure": {
                "value_columns": value_columns,
                "price_columns": price_columns,
                "sector_columns": sector_columns,
                "asset_type_columns": asset_type_columns,
                "pnl_columns": pnl_columns
            }
        }
        
        # Portfolio value analysis
        if value_columns:
            value_col = value_columns[0]
            current_portfolio_value = df[value_col].sum()
            scenario_summary["current_portfolio_value"] = current_portfolio_value
            scenario_summary["primary_value_column"] = value_col
            
            # Individual holding analysis
            scenario_summary["holding_values"] = df[value_col].tolist()
            scenario_summary["max_holding_value"] = df[value_col].max()
            scenario_summary["min_holding_value"] = df[value_col].min()
            scenario_summary["avg_holding_value"] = df[value_col].mean()
        
        # Asset allocation analysis
        if asset_type_columns and value_columns:
            asset_col = asset_type_columns[0]
            value_col = value_columns[0]
            
            asset_allocation = df.groupby(asset_col)[value_col].sum() / df[value_col].sum()
            scenario_summary["current_asset_allocation"] = asset_allocation.to_dict()
            scenario_summary["asset_types"] = df[asset_col].unique().tolist()
            scenario_summary["asset_type_count"] = len(df[asset_col].unique())
        
        # Sector allocation analysis
        if sector_columns and value_columns:
            sector_col = sector_columns[0]
            value_col = value_columns[0]
            
            sector_allocation = df.groupby(sector_col)[value_col].sum() / df[value_col].sum()
            scenario_summary["current_sector_allocation"] = sector_allocation.to_dict()
            scenario_summary["sectors"] = df[sector_col].unique().tolist()
            scenario_summary["sector_count"] = len(df[sector_col].unique())
        
        # Performance analysis
        if pnl_columns:
            pnl_col = pnl_columns[0]
            scenario_summary["total_pnl"] = df[pnl_col].sum()
            scenario_summary["positive_holdings"] = len(df[df[pnl_col] > 0])
            scenario_summary["negative_holdings"] = len(df[df[pnl_col] < 0])
            scenario_summary["neutral_holdings"] = len(df[df[pnl_col] == 0])
            scenario_summary["avg_pnl"] = df[pnl_col].mean()
            scenario_summary["max_gain"] = df[pnl_col].max()
            scenario_summary["max_loss"] = df[pnl_col].min()
            
            # Performance by asset type
            if asset_type_columns:
                asset_col = asset_type_columns[0]
                asset_performance = {}
                for asset_type in df[asset_col].unique():
                    asset_df = df[df[asset_col] == asset_type]
                    asset_pnl = asset_df[pnl_col].sum()
                    asset_value = asset_df[value_col].sum() if value_columns else 0
                    asset_performance[asset_type] = {
                        "total_pnl": asset_pnl,
                        "total_value": asset_value,
                        "pnl_percentage": (asset_pnl / asset_value * 100) if asset_value > 0 else 0,
                        "holdings_count": len(asset_df)
                    }
                scenario_summary["asset_type_performance"] = asset_performance
        
        # Risk analysis
        if pnl_columns and len(df) > 1:
            pnl_col = pnl_columns[0]
            pnl_values = df[pnl_col].dropna()
            if len(pnl_values) > 1:
                scenario_summary["portfolio_volatility"] = {
                    "std_deviation": pnl_values.std(),
                    "variance": pnl_values.var(),
                    "range": pnl_values.max() - pnl_values.min(),
                    "coefficient_of_variation": pnl_values.std() / abs(pnl_values.mean()) if pnl_values.mean() != 0 else 0
                }
        
        # Concentration analysis
        if value_columns:
            value_col = value_columns[0]
            total_value = df[value_col].sum()
            
            # Top 5 holdings concentration
            top_5_holdings = df.nlargest(5, value_col)
            top_5_concentration = top_5_holdings[value_col].sum() / total_value
            
            scenario_summary["concentration_analysis"] = {
                "top_5_concentration": top_5_concentration,
                "top_5_holdings": top_5_holdings[['ticker', value_col]].to_dict('records') if 'ticker' in df.columns else [],
                "max_single_holding": df[value_col].max() / total_value,
                "diversification_score": 1 - top_5_concentration  # Higher is more diversified
            }
        
        # Add sample holdings for context
        scenario_summary["sample_holdings"] = holdings[:5]  # First 5 holdings for context
        
        return scenario_summary
    
    def _create_scenario_analysis_prompt(self, scenario_summary: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Create AI prompt for scenario analysis"""
        
        return f"""Analyze the following investment portfolio under various market scenarios and provide intelligent scenario insights:

SCENARIO SUMMARY:
{json.dumps(scenario_summary, indent=2, default=str)}

USER PROFILE:
{json.dumps(user_profile, indent=2, default=str)}

SCENARIO ANALYSIS REQUIREMENTS:
1. STRESS TESTING: Analyze portfolio performance under adverse market conditions (market crashes, recessions, sector downturns)
2. WHAT-IF ANALYSIS: Model different investment scenarios (additional investments, rebalancing, sector rotation)
3. GOAL ACHIEVEMENT: Assess probability of meeting financial goals under different market conditions
4. RISK SCENARIOS: Identify potential risks and their impact on portfolio value
5. MONTE CARLO SIMULATION: Provide probabilistic outcomes based on historical market data
6. ECONOMIC SCENARIOS: Analyze impact of inflation, interest rate changes, currency fluctuations

Return your analysis as a JSON object with an "insights" array. Use this EXACT format:

{{
  "insights": [
    {{
      "type": "stress_test",
      "severity": "high",
      "title": "Market Crash Scenario Impact",
      "description": "In a 30% market crash, portfolio would lose approximately ₹15,00,000 (30% of current value)",
      "recommendation": "Consider adding defensive assets and reducing equity exposure",
      "data": {{
        "scenario": "market_crash_30_percent",
        "current_value": 5000000,
        "projected_loss": 1500000,
        "remaining_value": 3500000
      }}
    }},
    {{
      "type": "what_if",
      "severity": "medium",
      "title": "Additional Investment Impact",
      "description": "Investing additional ₹10,00,000 would increase portfolio value by 20%",
      "recommendation": "Consider systematic investment plan for additional funds",
      "data": {{
        "scenario": "additional_investment",
        "additional_amount": 1000000,
        "current_value": 5000000,
        "new_total_value": 6000000,
        "increase_percentage": 20
      }}
    }}
  ]
}}

CRITICAL: Return ONLY valid JSON. No markdown, no explanations, no text outside the JSON object.

ANALYSIS GUIDELINES:
1. Model stress test scenarios (market crashes, economic downturns)
2. Analyze what-if scenarios (additional investments, withdrawals)
3. Assess goal achievement probability under different market conditions
4. Evaluate portfolio resilience to various risk scenarios
5. Consider sector-specific stress scenarios
6. Model interest rate change impacts
7. Analyze inflation impact scenarios
8. Provide specific recommendations for each scenario
9. Consider user's risk tolerance and investment goals
10. Prioritize insights by severity (high = urgent, medium = important, low = informational)

Focus on providing valuable, actionable scenario insights that will help understand portfolio risks and opportunities under different market conditions."""
    
    def _parse_ai_scenario_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract scenario insights"""
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
            
            # Try to parse as JSON object first
            try:
                parsed = json.loads(ai_response)
                if isinstance(parsed, dict) and "insights" in parsed:
                    insights = parsed["insights"]
                    if isinstance(insights, list):
                        return insights
            except:
                pass
            
            # Fallback: Try to find JSON array in the response
            json_start = ai_response.find('[')
            json_end = ai_response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                insights = json.loads(json_str)
                
                # Validate that it's a list
                if isinstance(insights, list):
                    return insights
                else:
                    self.logger.error("AI response is not a list")
                    return []
            else:
                self.logger.error("No JSON found in AI response")
                self.logger.error(f"Response preview: {ai_response[:500]}")
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"AI response: {ai_response[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {e}")
            return []
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent"""
        return self.analysis_cache
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI scenario analysis"""
        return {
            "insights_count": len(self.analysis_cache),
            "high_priority_insights": len([i for i in self.analysis_cache if i.get('severity') == 'high']),
            "medium_priority_insights": len([i for i in self.analysis_cache if i.get('severity') == 'medium']),
            "low_priority_insights": len([i for i in self.analysis_cache if i.get('severity') == 'low']),
            "last_analysis": self.last_update.isoformat() if self.last_update else None,
            "capabilities": self.capabilities,
            "analysis_method": "ai_powered"
        }
    
    def handle_message(self, message):
        """Handle incoming messages from other agents"""
        self.logger.info(f"AI Scenario Agent received message: {message.message_type.value}")
        
        if message.message_type == MessageType.REQUEST:
            if message.content.get("request_type") == "scenario_analysis":
                response = self.analyze({
                    "portfolio_data": message.content.get("portfolio_data", {}),
                    "user_profile": message.content.get("user_profile", {})
                })
                return response
