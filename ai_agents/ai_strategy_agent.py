"""
AI-Powered Investment Strategy Agent
Uses OpenAI to intelligently provide investment strategy and recommendations
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, MessagePriority

class AIInvestmentStrategyAgent(BaseAgent):
    """
    AI-powered agent for intelligent investment strategy and recommendations
    """
    
    def __init__(self, agent_id: str = "ai_strategy_agent"):
        super().__init__(agent_id, "AI Investment Strategy Agent")
        self.capabilities = [
            "intelligent_strategy_analysis",
            "goal_based_investing",
            "risk_management_strategies",
            "asset_allocation_optimization",
            "rebalancing_recommendations",
            "tax_optimization_strategies"
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
        Main analysis method using AI for intelligent strategy analysis
        """
        try:
            self.update_status("analyzing")
            self.last_analysis_data = data
            
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to analyze strategy data
            insights = self._ai_analyze_strategy(data)
            
            # Cache insights
            self.analysis_cache = insights
            
            self.update_status("active")
            return self.format_response(insights, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI strategy analysis: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_analyze_strategy(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently analyze investment strategy"""
        
        portfolio_data = data.get("portfolio_data", {})
        user_profile = data.get("user_profile", {})
        
        if not portfolio_data:
            return [{
                "type": "strategy_analysis",
                "severity": "low",
                "title": "No Portfolio Data",
                "description": "No portfolio data found for strategy analysis",
                "recommendation": "Upload transaction files to enable strategy analysis",
                "data": {"portfolio_data_available": False}
            }]
        
        # Prepare strategy data for AI analysis
        strategy_summary = self._prepare_strategy_summary(portfolio_data, user_profile)
        
        # Create AI prompt for strategy analysis
        prompt = self._create_strategy_analysis_prompt(strategy_summary, user_profile)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert investment strategist and financial advisor. Your task is to analyze investment portfolios and provide intelligent strategy recommendations.

IMPORTANT RULES:
1. Analyze portfolio data intelligently - understand the structure and performance
2. Consider user's risk tolerance, goals, and investment preferences
3. Provide specific, actionable strategy recommendations
4. Focus on long-term wealth building and risk management
5. Be thorough but concise in your analysis
6. Return insights in the exact JSON format requested

For each insight, provide:
- type: Category of strategy (asset_allocation, rebalancing, risk_management, etc.)
- severity: high/medium/low based on importance and urgency
- title: Clear, descriptive title
- description: Detailed explanation of the strategy recommendation
- recommendation: Specific actionable strategy advice
- data: Relevant metrics and supporting data

Focus on:
- Asset allocation optimization
- Risk management strategies
- Rebalancing recommendations
- Goal-based investing strategies
- Tax optimization opportunities
- Diversification strategies
- Investment timing and market conditions"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.1,  # Low temperature for consistent analysis
                timeout=60
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract insights from response
            insights = self._parse_ai_strategy_response(ai_response)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"AI strategy analysis failed: {e}")
            return [{
                "type": "strategy_analysis",
                "severity": "medium",
                "title": "Strategy Analysis Error",
                "description": f"AI strategy analysis failed: {str(e)}",
                "recommendation": "Please try again or contact support",
                "data": {"error": str(e)}
            }]
    
    def _prepare_strategy_summary(self, portfolio_data: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare strategy data for AI analysis"""
        
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
        
        # Calculate strategy metrics
        strategy_summary = {
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
            strategy_summary["total_portfolio_value"] = df[value_col].sum()
            strategy_summary["primary_value_column"] = value_col
            
            # Individual holding analysis
            strategy_summary["holding_values"] = df[value_col].tolist()
            strategy_summary["max_holding_value"] = df[value_col].max()
            strategy_summary["min_holding_value"] = df[value_col].min()
            strategy_summary["avg_holding_value"] = df[value_col].mean()
        
        # Asset allocation analysis
        if asset_type_columns and value_columns:
            asset_col = asset_type_columns[0]
            value_col = value_columns[0]
            
            asset_allocation = df.groupby(asset_col)[value_col].sum() / df[value_col].sum()
            strategy_summary["current_asset_allocation"] = asset_allocation.to_dict()
            strategy_summary["asset_types"] = df[asset_col].unique().tolist()
            strategy_summary["asset_type_count"] = len(df[asset_col].unique())
        
        # Sector allocation analysis
        if sector_columns and value_columns:
            sector_col = sector_columns[0]
            value_col = value_columns[0]
            
            sector_allocation = df.groupby(sector_col)[value_col].sum() / df[value_col].sum()
            strategy_summary["current_sector_allocation"] = sector_allocation.to_dict()
            strategy_summary["sectors"] = df[sector_col].unique().tolist()
            strategy_summary["sector_count"] = len(df[sector_col].unique())
        
        # Performance analysis
        if pnl_columns:
            pnl_col = pnl_columns[0]
            strategy_summary["total_pnl"] = df[pnl_col].sum()
            strategy_summary["positive_holdings"] = len(df[df[pnl_col] > 0])
            strategy_summary["negative_holdings"] = len(df[df[pnl_col] < 0])
            strategy_summary["neutral_holdings"] = len(df[df[pnl_col] == 0])
            strategy_summary["avg_pnl"] = df[pnl_col].mean()
            strategy_summary["max_gain"] = df[pnl_col].max()
            strategy_summary["max_loss"] = df[pnl_col].min()
            
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
                strategy_summary["asset_type_performance"] = asset_performance
        
        # Risk analysis
        if pnl_columns and len(df) > 1:
            pnl_col = pnl_columns[0]
            pnl_values = df[pnl_col].dropna()
            if len(pnl_values) > 1:
                strategy_summary["portfolio_volatility"] = {
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
            
            strategy_summary["concentration_analysis"] = {
                "top_5_concentration": top_5_concentration,
                "top_5_holdings": top_5_holdings[['ticker', value_col]].to_dict('records') if 'ticker' in df.columns else [],
                "max_single_holding": df[value_col].max() / total_value,
                "diversification_score": 1 - top_5_concentration  # Higher is more diversified
            }
        
        # Add sample holdings for context
        strategy_summary["sample_holdings"] = holdings[:5]  # First 5 holdings for context
        
        return strategy_summary
    
    def _create_strategy_analysis_prompt(self, strategy_summary: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Create AI prompt for strategy analysis"""
        
        return f"""Analyze the following investment portfolio and provide intelligent strategy recommendations:

STRATEGY SUMMARY:
{json.dumps(strategy_summary, indent=2, default=str)}

USER PROFILE:
{json.dumps(user_profile, indent=2, default=str)}

Please analyze this portfolio and provide strategy insights in the following JSON format:

[
  {{
    "type": "asset_allocation",
    "severity": "high",
    "title": "Asset Allocation Imbalance",
    "description": "Portfolio is 90% stocks with no bond allocation, creating high risk",
    "recommendation": "Consider adding 20-30% bond allocation for risk management",
    "data": {{
      "current_stock_allocation": 0.90,
      "current_bond_allocation": 0.00,
      "recommended_bond_allocation": 0.25
    }}
  }},
  {{
    "type": "rebalancing",
    "severity": "medium",
    "title": "Portfolio Rebalancing Needed",
    "description": "Technology sector has grown to 60% of portfolio, exceeding target allocation",
    "recommendation": "Rebalance by reducing technology allocation to 30% and increasing other sectors",
    "data": {{
      "current_tech_allocation": 0.60,
      "target_tech_allocation": 0.30,
      "rebalancing_amount": 0.30
    }}
  }}
]

ANALYSIS GUIDELINES:
1. Analyze current asset allocation vs. recommended allocation based on risk tolerance
2. Identify rebalancing opportunities and needs
3. Assess portfolio concentration and diversification
4. Evaluate risk management strategies
5. Consider goal-based investing recommendations
6. Identify tax optimization opportunities
7. Suggest sector rotation strategies
8. Provide specific allocation recommendations
9. Consider user's investment goals and timeline
10. Prioritize insights by severity (high = urgent, medium = important, low = informational)

Focus on providing valuable, actionable strategy recommendations that will help optimize the portfolio for the user's specific goals and risk tolerance."""
    
    def _parse_ai_strategy_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract strategy insights"""
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
                insights = json.loads(json_str)
                
                # Validate that it's a list
                if isinstance(insights, list):
                    return insights
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
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent"""
        return self.analysis_cache
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI strategy analysis"""
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
        self.logger.info(f"AI Strategy Agent received message: {message.message_type.value}")
        
        if message.message_type == MessageType.REQUEST:
            if message.content.get("request_type") == "strategy_analysis":
                response = self.analyze({
                    "portfolio_data": message.content.get("portfolio_data", {}),
                    "user_profile": message.content.get("user_profile", {})
                })
                return response
