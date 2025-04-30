from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, limits: Dict[str, int]):
        """
        Initialize rate limiter with limits per hour for different operations.
        
        Args:
            limits: Dictionary mapping operation names to their hourly limits
        """
        self.limits = limits
        self.counters: Dict[str, list] = {op: [] for op in limits}
        
    def _cleanup_old_requests(self, operation: str):
        """Remove requests older than 1 hour."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        self.counters[operation] = [ts for ts in self.counters[operation] if ts > hour_ago]
        
    def can_proceed(self, operation: str) -> bool:
        """
        Check if an operation can proceed under rate limits.
        
        Args:
            operation: Name of the operation to check
            
        Returns:
            bool: True if operation can proceed, False if rate limited
        """
        if operation not in self.limits:
            logger.warning(f"Unknown operation: {operation}")
            return False
            
        self._cleanup_old_requests(operation)
        
        if len(self.counters[operation]) >= self.limits[operation]:
            return False
            
        self.counters[operation].append(datetime.utcnow())
        return True
        
    def get_remaining(self, operation: str) -> Optional[int]:
        """
        Get remaining requests for an operation.
        
        Args:
            operation: Name of the operation to check
            
        Returns:
            int: Number of remaining requests in the current hour
        """
        if operation not in self.limits:
            return None
            
        self._cleanup_old_requests(operation)
        return self.limits[operation] - len(self.counters[operation])
        
    def reset(self, operation: Optional[str] = None):
        """
        Reset counters for specified operation or all operations.
        
        Args:
            operation: Optional operation name to reset. If None, resets all.
        """
        if operation:
            if operation in self.counters:
                self.counters[operation] = []
        else:
            self.counters = {op: [] for op in self.limits} 