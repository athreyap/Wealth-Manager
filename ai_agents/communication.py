"""
Agent Communication Protocol
Handles communication between different AI agents
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from enum import Enum

class MessageType(Enum):
    """Types of messages between agents"""
    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    UPDATE = "update"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentMessage:
    """Standard message format for agent communication"""
    
    def __init__(self, 
                 from_agent: str,
                 to_agent: str,
                 message_type: MessageType,
                 content: Dict[str, Any],
                 priority: MessagePriority = MessagePriority.MEDIUM):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now()
        self.message_id = f"{from_agent}_{to_agent}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class AgentCommunication:
    """
    Manages communication between AI agents
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AgentCommunication")
        self.message_queue = []
        self.registered_agents = {}
        self.message_history = []
    
    def register_agent(self, agent_id: str, agent_instance):
        """Register an agent for communication"""
        self.registered_agents[agent_id] = agent_instance
        self.logger.info(f"Agent {agent_id} registered for communication")
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send a message between agents"""
        try:
            # Add to message queue
            self.message_queue.append(message)
            self.message_history.append(message)
            
            # Log the message
            self.logger.info(f"Message sent: {message.from_agent} -> {message.to_agent} ({message.message_type.value})")
            
            # Process message if target agent is registered
            if message.to_agent in self.registered_agents:
                self._process_message(message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False
    
    def broadcast_alert(self, from_agent: str, alert_type: str, severity: str, data: Dict[str, Any]):
        """Broadcast an alert to all registered agents"""
        message = AgentMessage(
            from_agent=from_agent,
            to_agent="ALL",
            message_type=MessageType.BROADCAST,
            content={
                "alert_type": alert_type,
                "severity": severity,
                "data": data
            },
            priority=MessagePriority.HIGH if severity == "high" else MessagePriority.MEDIUM
        )
        
        # Send to all registered agents
        for agent_id in self.registered_agents.keys():
            if agent_id != from_agent:  # Don't send to self
                message.to_agent = agent_id
                self.send_message(message)
    
    def _process_message(self, message: AgentMessage):
        """Process a received message"""
        try:
            target_agent = self.registered_agents[message.to_agent]
            
            # Call the agent's message handler if it exists
            if hasattr(target_agent, 'handle_message'):
                target_agent.handle_message(message)
            else:
                self.logger.warning(f"Agent {message.to_agent} has no message handler")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history for an agent or all agents"""
        if agent_id:
            filtered_messages = [msg for msg in self.message_history if msg.from_agent == agent_id or msg.to_agent == agent_id]
        else:
            filtered_messages = self.message_history
        
        # Return most recent messages
        return [msg.to_dict() for msg in filtered_messages[-limit:]]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {}
        for agent_id, agent in self.registered_agents.items():
            if hasattr(agent, 'get_agent_info'):
                status[agent_id] = agent.get_agent_info()
            else:
                status[agent_id] = {
                    "agent_id": agent_id,
                    "status": "unknown",
                    "last_update": datetime.now().isoformat()
                }
        return status
