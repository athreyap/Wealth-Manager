"""
Performance Optimizer for AI Agents
Handles caching, optimization, and performance monitoring
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import functools
import hashlib
import json

class PerformanceOptimizer:
    """
    Optimizes AI agent performance through caching and monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceOptimizer")
        self.cache = {}
        self.cache_ttl = {
            "portfolio_analysis": 300,  # 5 minutes
            "market_insights": 600,     # 10 minutes
            "scenario_analysis": 900,   # 15 minutes
            "recommendations": 300,     # 5 minutes
            "alerts": 60                # 1 minute
        }
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": 0,
            "error_count": 0
        }
    
    def cache_key(self, agent_id: str, data: Dict[str, Any]) -> str:
        """Generate cache key for data"""
        # Create a hash of the relevant data
        cache_data = {
            "agent_id": agent_id,
            "data": data
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        ttl = self.cache_ttl.get(cache_type, 300)
        
        # Check if cache has expired
        if datetime.now() - cache_entry["timestamp"] > timedelta(seconds=ttl):
            del self.cache[cache_key]
            return False
        
        return True
    
    def get_cached_result(self, cache_key: str, cache_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid"""
        if self.is_cache_valid(cache_key, cache_type):
            self.performance_metrics["cache_hits"] += 1
            self.logger.debug(f"Cache hit for {cache_type}")
            return self.cache[cache_key]["result"]
        
        self.performance_metrics["cache_misses"] += 1
        return None
    
    def cache_result(self, cache_key: str, cache_type: str, result: Dict[str, Any]):
        """Cache analysis result"""
        self.cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now(),
            "type": cache_type
        }
        self.logger.debug(f"Cached result for {cache_type}")
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache entries"""
        if cache_type:
            # Clear specific cache type
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry.get("type") == cache_type
            ]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            # Clear all cache
            self.cache.clear()
        
        self.logger.info(f"Cleared cache for {cache_type or 'all'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.performance_metrics["total_requests"]
        cache_hits = self.performance_metrics["cache_hits"]
        cache_misses = self.performance_metrics["cache_misses"]
        
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "avg_response_time": self.performance_metrics["avg_response_time"],
            "error_count": self.performance_metrics["error_count"]
        }
    
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator to monitor function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.performance_metrics["total_requests"] += 1
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update average response time
                current_avg = self.performance_metrics["avg_response_time"]
                total_requests = self.performance_metrics["total_requests"]
                self.performance_metrics["avg_response_time"] = (
                    (current_avg * (total_requests - 1) + execution_time) / total_requests
                )
                
                self.logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                self.performance_metrics["error_count"] += 1
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    def optimize_agent_analysis(self, agent_id: str, analysis_func: Callable, 
                               data: Dict[str, Any], cache_type: str) -> Dict[str, Any]:
        """Optimize agent analysis with caching"""
        cache_key = self.cache_key(agent_id, data)
        
        # Try to get cached result
        cached_result = self.get_cached_result(cache_key, cache_type)
        if cached_result:
            return cached_result
        
        # Run analysis and cache result
        start_time = time.time()
        try:
            result = analysis_func(data)
            execution_time = time.time() - start_time
            
            # Cache the result
            self.cache_result(cache_key, cache_type, result)
            
            self.logger.info(f"{agent_id} analysis completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            self.logger.error(f"Error in {agent_id} analysis: {str(e)}")
            return {"error": str(e), "insights": []}
    
    def batch_optimize_analyses(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize multiple analyses in batch"""
        results = []
        
        for analysis in analyses:
            agent_id = analysis.get("agent_id")
            analysis_func = analysis.get("analysis_func")
            data = analysis.get("data")
            cache_type = analysis.get("cache_type", "default")
            
            if all([agent_id, analysis_func, data]):
                result = self.optimize_agent_analysis(agent_id, analysis_func, data, cache_type)
                results.append(result)
            else:
                results.append({"error": "Invalid analysis configuration"})
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        cache_stats = self.get_cache_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_statistics": cache_stats,
            "cache_entries": [
                {
                    "key": key[:8] + "...",  # Truncate key for privacy
                    "type": entry.get("type"),
                    "age_seconds": (datetime.now() - entry["timestamp"]).total_seconds(),
                    "size_bytes": len(json.dumps(entry["result"]))
                }
                for key, entry in self.cache.items()
            ],
            "recommendations": self._get_optimization_recommendations(cache_stats)
        }
    
    def _get_optimization_recommendations(self, cache_stats: Dict[str, Any]) -> List[str]:
        """Get optimization recommendations based on performance metrics"""
        recommendations = []
        
        hit_rate = cache_stats.get("hit_rate", 0)
        avg_response_time = cache_stats.get("avg_response_time", 0)
        error_count = cache_stats.get("error_count", 0)
        
        if hit_rate < 50:
            recommendations.append("Consider increasing cache TTL for better hit rates")
        
        if avg_response_time > 5:
            recommendations.append("High response times detected - consider optimizing analysis algorithms")
        
        if error_count > cache_stats.get("total_requests", 1) * 0.1:
            recommendations.append("High error rate detected - review error handling")
        
        if cache_stats.get("cache_size", 0) > 100:
            recommendations.append("Large cache size - consider implementing cache eviction policy")
        
        return recommendations

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
