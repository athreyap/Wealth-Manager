"""
AI-Powered Market Analysis Agent
Uses OpenAI to intelligently analyze market conditions and trends
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, MessagePriority

class AIMarketAnalysisAgent(BaseAgent):
    """
    AI-powered agent for intelligent market analysis
    """
    
    def __init__(self, agent_id: str = "ai_market_agent"):
        super().__init__(agent_id, "AI Market Analysis Agent")
        self.capabilities = [
            "intelligent_market_analysis",
            "sector_analysis",
            "market_sentiment_analysis",
            "trend_identification",
            "opportunity_detection",
            "risk_assessment"
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
        Main analysis method using AI for intelligent market analysis
        """
        try:
            self.update_status("analyzing")
            self.last_analysis_data = data
            
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to analyze market data
            insights = self._ai_analyze_market(data)
            
            # Cache insights
            self.analysis_cache = insights
            
            self.update_status("active")
            return self.format_response(insights, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI market analysis: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_analyze_market(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently analyze market data"""
        
        holdings = data.get("holdings", [])
        user_profile = data.get("user_profile", {})
        
        if not holdings:
            return [{
                "type": "market_analysis",
                "severity": "low",
                "title": "No Market Data",
                "description": "No holdings found to analyze market conditions",
                "recommendation": "Upload transaction files to enable market analysis",
                "data": {"holdings_count": 0}
            }]
        
        # Prepare market data for AI analysis
        market_summary = self._prepare_market_summary(holdings)
        
        # Create AI prompt for market analysis
        prompt = self._create_market_analysis_prompt(market_summary, user_profile)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # GPT-5 for better market analysis
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert market analyst and financial strategist. Your task is to analyze market conditions and provide intelligent insights about market trends, opportunities, and risks.

IMPORTANT RULES:
1. Analyze market data intelligently - understand patterns and trends
2. Identify market opportunities and risks
3. Provide specific, actionable market insights
4. Consider current market conditions and economic factors
5. Be thorough but concise in your analysis
6. Return insights in the exact JSON format requested

For each insight, provide:
- type: Category of analysis (market_sentiment, sector_analysis, market_conditions, etc.)
- severity: high/medium/low based on importance and urgency
- title: Clear, descriptive title
- description: Detailed explanation of the market finding
- recommendation: Specific actionable market advice
- data: Relevant metrics and supporting data

Focus on:
- Market sentiment analysis
- Sector performance and rotation opportunities
- Market volatility and risk assessment
- Economic indicators and trends
- Investment opportunities
- Market timing considerations"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=3000,
                # Note: GPT-5 only supports default temperature (1)
                timeout=60
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract insights from response
            insights = self._parse_ai_market_response(ai_response)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"AI market analysis failed: {e}")
            return [{
                "type": "market_analysis",
                "severity": "medium",
                "title": "Market Analysis Error",
                "description": f"AI market analysis failed: {str(e)}",
                "recommendation": "Please try again or contact support",
                "data": {"error": str(e)}
            }]
    
    def _prepare_market_summary(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare market data for AI analysis"""
        
        if not holdings:
            return {"holdings_count": 0}
        
        # Convert to DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame(holdings)
        
        # Intelligent column detection
        value_columns = [col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower()]
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        sector_columns = [col for col in df.columns if 'sector' in col.lower()]
        asset_type_columns = [col for col in df.columns if 'asset' in col.lower() or 'type' in col.lower()]
        pnl_columns = [col for col in df.columns if 'pnl' in col.lower() or 'profit' in col.lower() or 'loss' in col.lower()]
        
        # Calculate market metrics
        market_summary = {
            "holdings_count": len(holdings),
            "total_holdings": len(holdings),
            "available_columns": list(df.columns),
            "data_structure": {
                "value_columns": value_columns,
                "price_columns": price_columns,
                "sector_columns": sector_columns,
                "asset_type_columns": asset_type_columns,
                "pnl_columns": pnl_columns
            }
        }
        
        # Market sentiment analysis
        if pnl_columns:
            pnl_col = pnl_columns[0]
            positive_holdings = len(df[df[pnl_col] > 0])
            negative_holdings = len(df[df[pnl_col] < 0])
            neutral_holdings = len(df[df[pnl_col] == 0])
            total_holdings = len(df)
            
            market_summary["market_sentiment"] = {
                "positive_ratio": positive_holdings / total_holdings if total_holdings > 0 else 0,
                "negative_ratio": negative_holdings / total_holdings if total_holdings > 0 else 0,
                "neutral_ratio": neutral_holdings / total_holdings if total_holdings > 0 else 0,
                "positive_count": positive_holdings,
                "negative_count": negative_holdings,
                "neutral_count": neutral_holdings,
                "total_count": total_holdings
            }
            
            # Determine overall sentiment
            if positive_holdings > negative_holdings * 1.5:
                market_summary["overall_sentiment"] = "bullish"
            elif negative_holdings > positive_holdings * 1.5:
                market_summary["overall_sentiment"] = "bearish"
            else:
                market_summary["overall_sentiment"] = "neutral"
        
        # Sector analysis
        if sector_columns and value_columns:
            sector_col = sector_columns[0]
            value_col = value_columns[0]
            
            sector_performance = {}
            for sector in df[sector_col].unique():
                sector_df = df[df[sector_col] == sector]
                if pnl_columns:
                    sector_pnl = sector_df[pnl_col].sum()
                    sector_value = sector_df[value_col].sum()
                    sector_performance[sector] = {
                        "total_pnl": sector_pnl,
                        "total_value": sector_value,
                        "pnl_percentage": (sector_pnl / sector_value * 100) if sector_value > 0 else 0,
                        "holdings_count": len(sector_df)
                    }
            
            market_summary["sector_performance"] = sector_performance
            
            # Find best and worst performing sectors
            if sector_performance:
                best_sector = max(sector_performance.items(), key=lambda x: x[1]["pnl_percentage"])
                worst_sector = min(sector_performance.items(), key=lambda x: x[1]["pnl_percentage"])
                
                market_summary["best_performing_sector"] = {
                    "name": best_sector[0],
                    "performance": best_sector[1]["pnl_percentage"]
                }
                market_summary["worst_performing_sector"] = {
                    "name": worst_sector[0],
                    "performance": worst_sector[1]["pnl_percentage"]
                }
        
        # Asset type analysis
        if asset_type_columns and value_columns:
            asset_col = asset_type_columns[0]
            value_col = value_columns[0]
            
            asset_performance = {}
            for asset_type in df[asset_col].unique():
                asset_df = df[df[asset_col] == asset_type]
                if pnl_columns:
                    asset_pnl = asset_df[pnl_col].sum()
                    asset_value = asset_df[value_col].sum()
                    asset_performance[asset_type] = {
                        "total_pnl": asset_pnl,
                        "total_value": asset_value,
                        "pnl_percentage": (asset_pnl / asset_value * 100) if asset_value > 0 else 0,
                        "holdings_count": len(asset_df)
                    }
            
            market_summary["asset_type_performance"] = asset_performance
        
        # Market volatility analysis
        if pnl_columns:
            pnl_col = pnl_columns[0]
            pnl_values = df[pnl_col].dropna()
            if len(pnl_values) > 1:
                market_summary["market_volatility"] = {
                    "std_deviation": pnl_values.std(),
                    "variance": pnl_values.var(),
                    "range": pnl_values.max() - pnl_values.min(),
                    "coefficient_of_variation": pnl_values.std() / abs(pnl_values.mean()) if pnl_values.mean() != 0 else 0
                }
        
        # Add sample holdings for context
        market_summary["sample_holdings"] = holdings[:5]  # First 5 holdings for context
        
        return market_summary
    
    def _create_market_analysis_prompt(self, market_summary: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Create AI prompt for market analysis"""
        
        return f"""Analyze the following market data and provide intelligent market insights and recommendations:

MARKET SUMMARY:
{json.dumps(market_summary, indent=2, default=str)}

USER PROFILE:
{json.dumps(user_profile, indent=2, default=str)}

Please analyze this market data and provide insights in the following JSON format:

[
  {{
    "type": "market_sentiment",
    "severity": "high",
    "title": "Market Sentiment: Bearish",
    "description": "Only 20% of holdings are positive, indicating bearish market sentiment",
    "recommendation": "Consider defensive positioning and risk management strategies",
    "data": {{
      "sentiment": "bearish",
      "positive_ratio": 0.20,
      "positive_count": 20,
      "total_count": 100
    }}
  }},
  {{
    "type": "sector_analysis",
    "severity": "medium",
    "title": "Technology Sector Outperformance",
    "description": "Technology sector showing 25% returns while overall market is flat",
    "recommendation": "Consider increasing technology allocation or sector rotation",
    "data": {{
      "sector": "Technology",
      "performance": 0.25,
      "market_performance": 0.02
    }}
  }}
]

ANALYSIS GUIDELINES:
1. Analyze market sentiment based on positive/negative holdings ratio
2. Identify sector rotation opportunities and trends
3. Assess market volatility and risk levels
4. Identify asset type performance patterns
5. Provide market timing insights and recommendations
6. Consider economic indicators and market conditions
7. Suggest diversification opportunities
8. Evaluate market risk and provide risk management advice
9. Be specific with numbers and percentages when available
10. Prioritize insights by severity (high = urgent, medium = important, low = informational)

Focus on providing valuable, actionable market insights that will help optimize investment decisions and market timing."""
    
    def _parse_ai_market_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract market insights"""
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
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI market analysis"""
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
        self.logger.info(f"AI Market Agent received message: {message.message_type.value}")
        
        if message.message_type == MessageType.REQUEST:
            if message.content.get("request_type") == "market_analysis":
                response = self.analyze({
                    "holdings": message.content.get("holdings", []),
                    "user_profile": message.content.get("user_profile", {})
                })
                return response
