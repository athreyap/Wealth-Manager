"""
AI Agents Module for Wealth Management System
Modular AI agent system that works alongside existing functionality
"""

try:
    from .base_agent import BaseAgent
    from .ai_portfolio_agent import AIPortfolioAnalysisAgent
    from .ai_market_agent import AIMarketAnalysisAgent
    from .ai_strategy_agent import AIInvestmentStrategyAgent
    from .ai_scenario_agent import AIScenarioAnalysisAgent
    from .ai_recommendation_agent import AIInvestmentRecommendationAgent
    from .ai_pdf_extractor import AIPDFTransactionExtractor
    from .ai_csv_parser import AICSVTransactionParser
    from .communication import AgentCommunication, AgentMessage, MessageType, MessagePriority
    from .agent_manager import AgentManager, get_agent_manager, run_ai_analysis, get_ai_recommendations, get_ai_alerts
    from .performance_optimizer import PerformanceOptimizer, performance_optimizer
    
    __all__ = [
        'BaseAgent',
        'AIPortfolioAnalysisAgent', 
        'AIMarketAnalysisAgent',
        'AIInvestmentStrategyAgent',
        'AIScenarioAnalysisAgent',
        'AIInvestmentRecommendationAgent',
        'AIPDFTransactionExtractor',
        'AICSVTransactionParser',
        'AgentCommunication',
        'AgentMessage',
        'MessageType',
        'MessagePriority',
        'AgentManager',
        'get_agent_manager',
        'run_ai_analysis',
        'get_ai_recommendations',
        'get_ai_alerts',
        'PerformanceOptimizer',
        'performance_optimizer'
    ]
    
    # Module is successfully imported
    _IMPORT_SUCCESS = True
    
except ImportError as e:
    # Gracefully handle import errors
    import logging
    logging.warning(f"AI Agents import error: {e}")
    _IMPORT_SUCCESS = False
    __all__ = []
