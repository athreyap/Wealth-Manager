"""
AI-Powered Portfolio Analysis Agent
Uses OpenAI to intelligently analyze portfolio data without hardcoded logic
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, MessagePriority

class AIPortfolioAnalysisAgent(BaseAgent):
    """
    AI-powered agent for intelligent portfolio analysis
    """
    
    def __init__(self, agent_id: str = "ai_portfolio_agent"):
        super().__init__(agent_id, "AI Portfolio Analysis Agent")
        self.capabilities = [
            "intelligent_portfolio_analysis",
            "adaptive_data_understanding",
            "risk_assessment",
            "performance_analysis",
            "concentration_analysis",
            "rebalancing_recommendations"
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
        Main analysis method using AI for intelligent portfolio analysis
        """
        try:
            self.update_status("analyzing")
            self.last_analysis_data = data
            
            if not self.openai_client:
                return self.format_response([], "low", error="OpenAI client not available")
            
            # Use AI to analyze portfolio data
            insights = self._ai_analyze_portfolio(data)
            
            # Cache insights
            self.analysis_cache = insights
            
            self.update_status("active")
            return self.format_response(insights, "high")
            
        except Exception as e:
            self.logger.error(f"Error in AI portfolio analysis: {str(e)}")
            self.update_status("error")
            return self.format_response([], "low", error=str(e))
    
    def _ai_analyze_portfolio(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently analyze portfolio data"""
        
        holdings = data.get("holdings", [])
        user_profile = data.get("user_profile", {})
        
        if not holdings:
            return [{
                "type": "portfolio_analysis",
                "severity": "low",
                "title": "No Portfolio Data",
                "description": "No holdings found in the portfolio data",
                "recommendation": "Upload transaction files to build your portfolio",
                "data": {"holdings_count": 0}
            }]
        
        # Prepare data for AI analysis
        portfolio_summary = self._prepare_portfolio_summary(holdings)
        
        # Create AI prompt for portfolio analysis
        prompt = self._create_portfolio_analysis_prompt(portfolio_summary, user_profile)
        
        try:
            # Get current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # gpt-5 for enhanced portfolio analysis
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert portfolio analyst and financial advisor. Your task is to analyze portfolio data and provide intelligent insights and recommendations.

📅 CURRENT DATE: {current_date} (Year: {current_year})
⚠️ CRITICAL: Today's date is {current_date}. Always use this date when:
- Calculating time periods and holding periods
- Referencing current market conditions
- Making time-based recommendations
- Analyzing transaction dates
Do NOT use 2024 or any other year - use {current_year}.

IMPORTANT RULES:
1. Analyze the data intelligently - understand the structure and meaning of any column names
2. Identify risks, opportunities, and areas for improvement
3. Provide specific, actionable recommendations
4. Consider the user's risk tolerance and investment goals
5. Be thorough but concise in your analysis
6. Return insights in the exact JSON format requested

For each insight, provide:
- type: Category of analysis (concentration_risk, performance_analysis, diversification_opportunity, etc.)
- severity: high/medium/low based on importance and urgency
- title: Clear, descriptive title
- description: Detailed explanation of the finding
- recommendation: Specific actionable advice
- data: Relevant metrics and supporting data

Focus on:
- Portfolio concentration and diversification
- Risk assessment and management
- Performance analysis
- Rebalancing opportunities
- Asset allocation optimization
- Tax efficiency opportunities"""
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
            insights = self._parse_ai_portfolio_response(ai_response)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"AI portfolio analysis failed: {e}")
            return [{
                "type": "portfolio_analysis",
                "severity": "medium",
                "title": "Analysis Error",
                "description": f"AI analysis failed: {str(e)}",
                "recommendation": "Please try again or contact support",
                "data": {"error": str(e)}
            }]
    
    def _prepare_portfolio_summary(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare portfolio data for AI analysis"""
        
        if not holdings:
            return {"holdings_count": 0}
        
        # Convert to DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame(holdings)
        
        # Intelligent column detection
        value_columns = [col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower()]
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        quantity_columns = [col for col in df.columns if 'quantity' in col.lower() or 'units' in col.lower()]
        sector_columns = [col for col in df.columns if 'sector' in col.lower()]
        asset_type_columns = [col for col in df.columns if 'asset' in col.lower() or 'type' in col.lower()]
        pnl_columns = [col for col in df.columns if 'pnl' in col.lower() or 'profit' in col.lower() or 'loss' in col.lower()]
        
        # Calculate portfolio metrics
        portfolio_summary = {
            "holdings_count": len(holdings),
            "total_holdings": len(holdings),
            "available_columns": list(df.columns),
            "data_structure": {
                "value_columns": value_columns,
                "price_columns": price_columns,
                "quantity_columns": quantity_columns,
                "sector_columns": sector_columns,
                "asset_type_columns": asset_type_columns,
                "pnl_columns": pnl_columns
            }
        }
        
        # Calculate total portfolio value
        if value_columns:
            primary_value_col = value_columns[0]  # Use first value column found
            portfolio_summary["total_portfolio_value"] = df[primary_value_col].sum()
            portfolio_summary["primary_value_column"] = primary_value_col
            
            # Individual holding values
            portfolio_summary["holding_values"] = df[primary_value_col].tolist()
            portfolio_summary["max_holding_value"] = df[primary_value_col].max()
            portfolio_summary["min_holding_value"] = df[primary_value_col].min()
            portfolio_summary["avg_holding_value"] = df[primary_value_col].mean()
        
        # Sector analysis
        if sector_columns:
            sector_col = sector_columns[0]
            if value_columns:
                sector_allocation = df.groupby(sector_col)[primary_value_col].sum() / df[primary_value_col].sum()
                portfolio_summary["sector_allocation"] = sector_allocation.to_dict()
                portfolio_summary["sectors"] = df[sector_col].unique().tolist()
                portfolio_summary["sector_count"] = len(df[sector_col].unique())
        
        # Asset type analysis
        if asset_type_columns:
            asset_col = asset_type_columns[0]
            if value_columns:
                asset_allocation = df.groupby(asset_col)[primary_value_col].sum() / df[primary_value_col].sum()
                portfolio_summary["asset_allocation"] = asset_allocation.to_dict()
                portfolio_summary["asset_types"] = df[asset_col].unique().tolist()
                portfolio_summary["asset_type_count"] = len(df[asset_col].unique())
        
        # Performance analysis
        if pnl_columns:
            pnl_col = pnl_columns[0]
            portfolio_summary["total_pnl"] = df[pnl_col].sum()
            portfolio_summary["positive_holdings"] = len(df[df[pnl_col] > 0])
            portfolio_summary["negative_holdings"] = len(df[df[pnl_col] < 0])
            portfolio_summary["neutral_holdings"] = len(df[df[pnl_col] == 0])
            portfolio_summary["avg_pnl"] = df[pnl_col].mean()
            portfolio_summary["max_gain"] = df[pnl_col].max()
            portfolio_summary["max_loss"] = df[pnl_col].min()
        
        # Add sample holdings for context
        portfolio_summary["sample_holdings"] = holdings[:5]  # First 5 holdings for context
        
        return portfolio_summary
    
    def _create_portfolio_analysis_prompt(self, portfolio_summary: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Create AI prompt for portfolio analysis"""
        
        return f"""Analyze the following portfolio data and provide intelligent insights and recommendations:

PORTFOLIO SUMMARY:
{json.dumps(portfolio_summary, indent=2, default=str)}

USER PROFILE:
{json.dumps(user_profile, indent=2, default=str)}

Return your analysis as a JSON object with an "insights" array. Use this EXACT format:

{{
  "insights": [
    {{
      "type": "concentration_risk",
      "severity": "high",
      "title": "High Sector Concentration",
      "description": "Technology sector represents 80% of portfolio, creating concentration risk",
      "recommendation": "Consider diversifying into other sectors like healthcare, finance, or consumer goods",
      "data": {{
        "sector": "Technology",
        "allocation": 0.80,
        "recommended_max": 0.30
      }}
    }},
    {{
      "type": "performance_analysis",
      "severity": "medium",
      "title": "Mixed Performance Results",
      "description": "Portfolio shows 60% positive holdings with average return of 15%",
      "recommendation": "Review underperforming holdings and consider rebalancing",
      "data": {{
        "positive_holdings": 60,
        "total_holdings": 100,
        "avg_return": 0.15
      }}
    }}
  ]
}}

CRITICAL: Return ONLY valid JSON. No markdown, no explanations, no text outside the JSON object.

ANALYSIS GUIDELINES:
1. Identify concentration risks (sector, asset type, individual holdings)
2. Assess portfolio diversification and suggest improvements
3. Analyze performance patterns and identify opportunities
4. Consider rebalancing needs based on current allocation
5. Evaluate risk levels and suggest risk management strategies
6. Provide tax optimization opportunities if applicable
7. Consider user's risk tolerance and investment goals
8. Be specific with numbers and percentages when available
9. Prioritize insights by severity (high = urgent, medium = important, low = informational)
10. Provide actionable recommendations for each insight

Focus on providing valuable, actionable insights that will help improve the portfolio's performance and risk profile."""
    
    def _parse_ai_portfolio_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract portfolio insights"""
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
                self.logger.error("No JSON array found in AI response")
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
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI portfolio analysis"""
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
        self.logger.info(f"AI Portfolio Agent received message: {message.message_type.value}")
        
        if message.message_type == MessageType.REQUEST:
            if message.content.get("request_type") == "portfolio_analysis":
                response = self.analyze({
                    "holdings": message.content.get("holdings", []),
                    "user_profile": message.content.get("user_profile", {})
                })
                return response
