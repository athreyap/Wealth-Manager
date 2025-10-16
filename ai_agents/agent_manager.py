"""
AI Agent Manager
Manages and coordinates all AI agents for the wealth management system
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st

from .ai_portfolio_agent import AIPortfolioAnalysisAgent
from .ai_market_agent import AIMarketAnalysisAgent
from .ai_strategy_agent import AIInvestmentStrategyAgent
from .ai_scenario_agent import AIScenarioAnalysisAgent
from .ai_recommendation_agent import AIInvestmentRecommendationAgent
from .communication import AgentCommunication
from .performance_optimizer import performance_optimizer

class AgentManager:
    """
    Manages all AI agents and coordinates their activities
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AgentManager")
        self.communication = AgentCommunication()
        
        # Initialize AI-powered agents
        self.portfolio_agent = AIPortfolioAnalysisAgent()
        self.market_agent = AIMarketAnalysisAgent()
        self.strategy_agent = AIInvestmentStrategyAgent()
        self.scenario_agent = AIScenarioAnalysisAgent()
        self.recommendation_agent = AIInvestmentRecommendationAgent()
        
        # Register agents for communication
        self.communication.register_agent("portfolio_agent", self.portfolio_agent)
        self.communication.register_agent("market_agent", self.market_agent)
        self.communication.register_agent("strategy_agent", self.strategy_agent)
        self.communication.register_agent("scenario_agent", self.scenario_agent)
        self.communication.register_agent("recommendation_agent", self.recommendation_agent)
        
        self.agents = {
            "portfolio": self.portfolio_agent,
            "market": self.market_agent,
            "strategy": self.strategy_agent,
            "scenario": self.scenario_agent,
            "recommendation": self.recommendation_agent
        }
        
        self.last_analysis = None
        self.analysis_cache = {}
    
    def analyze_portfolio(self, holdings: List[Dict[str, Any]], user_profile: Dict[str, Any] = None, pdf_context: str = None) -> Dict[str, Any]:
        """
        Run comprehensive portfolio analysis using all agents
        
        Args:
            holdings: List of portfolio holdings
            user_profile: User preferences and goals
            pdf_context: Text from uploaded PDFs for additional context
        """
        try:
            analysis_data = {
                "holdings": holdings,
                "portfolio_data": {
                    "holdings": holdings,
                    "total_holdings": len(holdings),
                    "portfolio_summary": self._calculate_portfolio_summary(holdings)
                },
                "user_id": user_profile.get("user_id") if user_profile else None,
                "user_profile": user_profile or {},
                "pdf_context": pdf_context or ""  # Add PDF context
            }
            
            # Run analysis with each agent using performance optimization
            portfolio_insights = performance_optimizer.optimize_agent_analysis(
                "portfolio_agent", 
                self.portfolio_agent.analyze, 
                analysis_data, 
                "portfolio_analysis"
            )
            
            market_insights = performance_optimizer.optimize_agent_analysis(
                "market_agent", 
                self.market_agent.analyze, 
                analysis_data, 
                "market_insights"
            )
            
            strategy_insights = performance_optimizer.optimize_agent_analysis(
                "strategy_agent", 
                self.strategy_agent.analyze, 
                {
                    "portfolio_data": analysis_data,
                    "user_profile": user_profile or {}
                }, 
                "recommendations"
            )
            
            scenario_insights = performance_optimizer.optimize_agent_analysis(
                "scenario_agent", 
                self.scenario_agent.analyze, 
                {
                    "portfolio_data": analysis_data.get("portfolio_data"),
                    "user_profile": user_profile or {},
                    "pdf_context": pdf_context or ""
                }, 
                "scenario_analysis"
            )
            
            recommendation_insights = performance_optimizer.optimize_agent_analysis(
                "recommendation_agent",
                self.recommendation_agent.analyze,
                {
                    "portfolio_data": analysis_data.get("portfolio_data"),
                    "user_profile": user_profile or {},
                    "pdf_context": pdf_context or ""
                },
                "investment_recommendations"
            )
            
            # Combine insights
            combined_analysis = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_insights": portfolio_insights.get("insights", []),
                "market_insights": market_insights.get("insights", []),
                "strategy_insights": strategy_insights.get("insights", []),
                "scenario_insights": scenario_insights.get("insights", []),
                "investment_recommendations": recommendation_insights.get("insights", []),
                "summary": self._generate_analysis_summary(portfolio_insights, market_insights, strategy_insights, scenario_insights, recommendation_insights)
            }
            
            # Cache the analysis
            self.analysis_cache = combined_analysis
            self.last_analysis = datetime.now()
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error in portfolio analysis: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _generate_analysis_summary(self, portfolio_insights: Dict, market_insights: Dict, strategy_insights: Dict, scenario_insights: Dict, recommendation_insights: Dict = None) -> Dict[str, Any]:
        """Generate a summary of all agent insights"""
        
        # Count insights by severity
        all_insights = []
        all_insights.extend(portfolio_insights.get("insights", []))
        all_insights.extend(market_insights.get("insights", []))
        all_insights.extend(strategy_insights.get("insights", []))
        all_insights.extend(scenario_insights.get("insights", []))
        if recommendation_insights:
            all_insights.extend(recommendation_insights.get("insights", []))
        
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        insight_types = set()
        
        for insight in all_insights:
            severity = insight.get("severity", "low")
            severity_counts[severity] += 1
            insight_types.add(insight.get("type", "unknown"))
        
        return {
            "total_insights": len(all_insights),
            "severity_breakdown": severity_counts,
            "insight_types": list(insight_types),
            "high_priority_count": severity_counts["high"],
            "requires_attention": severity_counts["high"] > 0 or severity_counts["medium"] > 2
        }
    
    def get_proactive_alerts(self) -> List[Dict[str, Any]]:
        """Get proactive alerts from all agents"""
        alerts = []
        
        # Get insights from each agent
        portfolio_insights = self.portfolio_agent.get_insights()
        market_insights = self.market_agent.get_insights()
        strategy_insights = self.strategy_agent.get_insights()
        
        # Filter for high-priority insights that should be alerts
        for insight in portfolio_insights + market_insights + strategy_insights:
            if insight.get("severity") in ["high", "critical"]:
                alerts.append({
                    "type": "proactive_alert",
                    "source": "ai_agent",
                    "title": insight.get("title", "Portfolio Alert"),
                    "description": insight.get("description", ""),
                    "recommendation": insight.get("recommendation", ""),
                    "severity": insight.get("severity", "medium"),
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return self.communication.get_agent_status()
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get a summary of all recommendations"""
        if not self.analysis_cache:
            return {"message": "No analysis available"}
        
        return {
            "last_analysis": self.analysis_cache.get("timestamp"),
            "summary": self.analysis_cache.get("summary", {}),
            "portfolio_insights_count": len(self.analysis_cache.get("portfolio_insights", [])),
            "market_insights_count": len(self.analysis_cache.get("market_insights", [])),
            "strategy_insights_count": len(self.analysis_cache.get("strategy_insights", []))
        }
    
    def _calculate_portfolio_summary(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio summary for scenario analysis"""
        if not holdings:
            return {}
        
        total_value = sum(holding.get('current_value', 0) or 0 for holding in holdings)
        total_investment = sum(holding.get('investment', 0) or 0 for holding in holdings)
        total_pnl = total_value - total_investment
        total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Asset type breakdown
        asset_types = {}
        sectors = {}
        channels = {}
        
        for holding in holdings:
            asset_type = holding.get('asset_type', 'Unknown')
            sector = holding.get('sector', 'Unknown')
            channel = holding.get('channel', 'Unknown')
            
            asset_types[asset_type] = asset_types.get(asset_type, 0) + (holding.get('current_value', 0) or 0)
            sectors[sector] = sectors.get(sector, 0) + (holding.get('current_value', 0) or 0)
            channels[channel] = channels.get(channel, 0) + (holding.get('current_value', 0) or 0)
        
        return {
            "total_value": total_value,
            "total_investment": total_investment,
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_pct,
            "holdings_count": len(holdings),
            "asset_type_allocation": asset_types,
            "sector_allocation": sectors,
            "channel_allocation": channels,
            "top_performers": sorted(holdings, key=lambda x: x.get('pnl_percentage', 0), reverse=True)[:5],
            "worst_performers": sorted(holdings, key=lambda x: x.get('pnl_percentage', 0))[:5]
        }
    
    def get_top_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top priority recommendations"""
        if not self.analysis_cache:
            return []
        
        all_insights = []
        all_insights.extend(self.analysis_cache.get("portfolio_insights", []))
        all_insights.extend(self.analysis_cache.get("market_insights", []))
        all_insights.extend(self.analysis_cache.get("strategy_insights", []))
        
        # Sort by severity (high > medium > low)
        severity_order = {"high": 3, "medium": 2, "low": 1}
        sorted_insights = sorted(
            all_insights, 
            key=lambda x: severity_order.get(x.get("severity", "low"), 1),
            reverse=True
        )
        
        return sorted_insights[:limit]

# Streamlit integration functions
def get_agent_manager():
    """Get or create agent manager in session state"""
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = AgentManager()
    return st.session_state.agent_manager

def run_ai_analysis(holdings: List[Dict[str, Any]], user_profile: Dict[str, Any] = None, pdf_context: str = None) -> Dict[str, Any]:
    """Run AI analysis on portfolio data with comprehensive context including PDFs"""
    agent_manager = get_agent_manager()
    
    # Enhance analysis data with comprehensive context like AI Assistant gets
    enhanced_holdings = holdings.copy()
    
    # Add comprehensive context to each holding
    for holding in enhanced_holdings:
        # Add calculated metrics
        current_price = holding.get('current_price') or holding.get('average_price', 0)
        current_value = (current_price or 0) * (holding.get('total_quantity', 0) or 0)
        investment = (holding.get('average_price', 0) or 0) * (holding.get('total_quantity', 0) or 0)
        pnl = current_value - investment
        pnl_pct = ((current_value - investment) / investment * 100) if investment > 0 else 0
        
        holding['current_value'] = current_value
        holding['investment'] = investment
        holding['pnl'] = pnl
        holding['pnl_percentage'] = pnl_pct
        holding['performance_rating'] = "Excellent" if pnl_pct > 50 else "Good" if pnl_pct > 20 else "Average" if pnl_pct > 0 else "Poor"
    
    return agent_manager.analyze_portfolio(enhanced_holdings, user_profile, pdf_context)

def get_ai_recommendations(limit: int = 5) -> List[Dict[str, Any]]:
    """Get top AI recommendations"""
    agent_manager = get_agent_manager()
    return agent_manager.get_top_recommendations(limit)

def get_ai_alerts() -> List[Dict[str, Any]]:
    """Get proactive AI alerts"""
    agent_manager = get_agent_manager()
    return agent_manager.get_proactive_alerts()
