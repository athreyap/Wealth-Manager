"""
Base Agent Class for AI Agent System
Provides common functionality for all specialized agents
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

class BaseAgent(ABC):
    """
    Base class for all AI agents in the wealth management system
    """
    
    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"Agent.{agent_name}")
        self.last_update = datetime.now()
        self.status = "initialized"
        self.capabilities = []
        
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method - must be implemented by each agent
        """
        pass
    
    @abstractmethod
    def get_insights(self) -> List[Dict[str, Any]]:
        """
        Get current insights from the agent
        """
        pass
    
    def update_status(self, status: str):
        """Update agent status"""
        self.status = status
        self.last_update = datetime.now()
        self.logger.info(f"Agent {self.agent_name} status updated to: {status}")
    
    def log_activity(self, activity: str, data: Dict[str, Any] = None):
        """Log agent activity"""
        log_data = {
            "agent": self.agent_name,
            "activity": activity,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.logger.info(f"Activity: {activity}", extra=log_data)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "last_update": self.last_update.isoformat(),
            "capabilities": self.capabilities
        }
    
    def validate_data(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data has required fields"""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
    
    def format_response(self, insights: List[Dict[str, Any]], priority: str = "medium", error: str = None) -> Dict[str, Any]:
        """Format agent response in standard format"""
        response = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "insights": insights,
            "status": self.status
        }
        
        if error:
            response["error"] = error
            
        return response
